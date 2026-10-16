"""One experimental cell: mask, impute, classify, evaluate.

A cell is a (mechanism, level, imputer, seed) combination. All classifiers in
the run share it, because they share the imputed matrices - imputing once and
fitting four classifiers on the result cuts the expensive stage (MissForest at
high missingness) by a factor of four and, more importantly, guarantees the
four classifiers are compared on *identical* data.

Stage order, which is the leakage contract of the whole study:

    split  ->  mask  ->  fit imputer on train  ->  transform all splits
           ->  fit classifier on train (early stopping on val)
           ->  threshold from val  ->  score test

Masking is applied to each split separately, after the split. The mask is part
of the simulated data-generating process rather than a fitted transform, so
this keeps every per-split statistic inside its own split.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config import NO_MECHANISM, Config
from .data.cohort import Cohort
from .imputation import impute_splits
from .metrics import (
    classification_metrics,
    imputation_metrics,
    imputation_metrics_per_column,
    pick_threshold,
    train_statistics,
)
from .missingness import apply_missingness, describe_mask
from .models import fit_classifier
from .splits import Split

EVAL_SPLITS = ("val", "test")


@dataclass
class CellResult:
    """Everything one cell contributes to the result set."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    per_column: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def run_cell(
    cohort: Cohort,
    split: Split,
    mechanism: str,
    level: float,
    imputer: str,
    classifiers: Sequence[str],
    seed: int,
    cfg: Config,
    collect_per_column: bool = False,
    collect_diagnostics: bool = False,
) -> CellResult:
    """Run one cell and return a result row per classifier per eval split.

    `mechanism=NO_MECHANISM` with `level=0` is the reference condition: no
    artificial masking, only the dataset's own missingness. The imputer still
    runs, because that natural missingness must be filled before any classifier
    can be fitted, so every imputation curve has an anchor at level 0.
    """
    try:
        return _run_cell(
            cohort,
            split,
            mechanism,
            level,
            imputer,
            classifiers,
            seed,
            cfg,
            collect_per_column,
            collect_diagnostics,
        )
    except Exception as exc:  # noqa: BLE001 - one bad cell must not kill a 1000-cell run
        detail = f"{type(exc).__name__}: {exc}"
        return CellResult(
            rows=[
                _base_row(mechanism, level, imputer, classifier, seed, cfg)
                | {"split": "test", "status": "failed", "error": detail}
                for classifier in classifiers
            ],
            diagnostics={"traceback": traceback.format_exc()},
            error=detail,
        )


def _run_cell(
    cohort: Cohort,
    split: Split,
    mechanism: str,
    level: float,
    imputer: str,
    classifiers: Sequence[str],
    seed: int,
    cfg: Config,
    collect_per_column: bool,
    collect_diagnostics: bool,
) -> CellResult:
    frames = split.frames(cohort.X)
    targets = split.targets(cohort.y)

    # ---- 1. impose the missingness condition --------------------------
    baseline = mechanism == NO_MECHANISM or level <= 0.0
    masked: dict[str, pd.DataFrame] = {}
    truths: dict[str, pd.DataFrame] = {}
    masks: dict[str, pd.DataFrame] = {}
    realised: dict[str, float] = {}
    diagnostics: dict[str, Any] = {}

    if baseline:
        masked = {name: frame.copy() for name, frame in frames.items()}
    else:
        mcfg = cfg.missingness
        for i, (name, frame) in enumerate(frames.items()):
            result = apply_missingness(
                frame,
                mechanism=mechanism,
                level=level,
                # Distinct stream per split: train/val/test masks are
                # independent draws from the same mechanism.
                seed=seed * 1000 + i,
                maskable=cohort.maskable,
                protected=cohort.protected,
                mar_strength=mcfg.mar_strength,
                mnar_strength=mcfg.mnar_strength,
                mnar_direction=mcfg.mnar_direction,
                min_observed_per_column=mcfg.min_observed_per_column,
            )
            masked[name] = result.X_masked
            truths[name] = result.truth
            masks[name] = result.mask
            realised[name] = result.realised_level
        if collect_diagnostics:
            diagnostics["mask_check_test"] = {
                "mechanism": mechanism,
                "requested_level": level,
                **describe_mask(
                    _rebuild(
                        masks["test"], truths["test"], frames["test"], realised["test"]
                    ),
                    frames["test"],
                    cohort.protected,
                ),
            }

    # ---- 2. impute: fitted on train, applied everywhere ---------------
    # Units for the z-scored reconstruction metrics, taken from the training
    # split *before* masking. This is a reporting scale, not a fitted
    # transform: nothing here reaches the imputer or the classifier, and it
    # stays inside the training rows, so it leaks nothing.
    #
    # It has to come from the unmasked data. MNAR removes the upper tail of
    # each column, which shrinks the training spread to about 0.7 of its true
    # value (0.4 for the worst columns) while MCAR leaves it at 0.99. Scaling
    # by the masked spread would therefore inflate MNAR's reported error by
    # roughly 45% for a reason that has nothing to do with imputation quality,
    # and the mechanism panels of every reconstruction figure would not be
    # comparable with each other.
    stats = train_statistics(frames["train"])
    imputed, fitted = impute_splits(
        imputer,
        masked,
        seed=seed,
        params=cfg.imputer_params.get(imputer),
        n_jobs=cfg.compute.inner_threads,
    )

    # p in adjusted R-squared: the other features the imputer draws on.
    n_features_used = max(1, cohort.n_features - 1)

    imp_metrics: dict[str, dict[str, float]] = {}
    for name in EVAL_SPLITS:
        if baseline:
            imp_metrics[name] = {"n_masked": 0}
        else:
            imp_metrics[name] = imputation_metrics(
                truths[name], imputed[name], masks[name], n_features_used, stats
            )

    per_column = pd.DataFrame()
    if collect_per_column and not baseline:
        per_column = imputation_metrics_per_column(
            truths["test"], imputed["test"], masks["test"], n_features_used
        )
        per_column.insert(0, "seed", seed)
        per_column.insert(0, "imputer", imputer)
        per_column.insert(0, "level", level)
        per_column.insert(0, "mechanism", mechanism)

    # ---- 3. classify and evaluate -------------------------------------
    rows: list[dict[str, Any]] = []
    for classifier in classifiers:
        started = time.perf_counter()
        model = fit_classifier(
            classifier,
            imputed["train"],
            targets["train"],
            imputed["val"],
            targets["val"],
            seed=seed,
            params=cfg.model_params.get(classifier),
            n_threads=cfg.compute.inner_threads,
        )

        proba = {name: model.predict_proba(imputed[name]) for name in EVAL_SPLITS}
        # Threshold is selected on validation and reused on test, so the test
        # F1/precision are not optimised against the data they score.
        threshold = pick_threshold(targets["val"].to_numpy(), proba["val"])

        for name in EVAL_SPLITS:
            row = _base_row(mechanism, level, imputer, classifier, seed, cfg)
            row.update(
                {
                    "split": name,
                    "status": "ok",
                    "error": None,
                    "realised_level": realised.get(name, 0.0),
                    "n_train": int(len(imputed["train"])),
                    "n_eval": int(len(imputed[name])),
                    "train_prevalence": float(targets["train"].mean()),
                    "imputer_fit_seconds": round(fitted.fit_seconds, 4),
                    "model_fit_seconds": round(model.fit_seconds, 4),
                    "cell_seconds": round(time.perf_counter() - started, 4),
                    "model_iterations": model.n_iterations,
                }
            )
            row.update({f"imp_{k}": v for k, v in imp_metrics[name].items()})
            row.update(classification_metrics(targets[name].to_numpy(), proba[name], threshold))
            rows.append(row)

    return CellResult(rows=rows, per_column=per_column, diagnostics=diagnostics)


def _base_row(
    mechanism: str, level: float, imputer: str, classifier: str, seed: int, cfg: Config
) -> dict[str, Any]:
    return {
        "run_name": cfg.run_name,
        "dataset": cfg.dataset.name,
        "mechanism": mechanism,
        "level": float(level),
        "imputer": imputer,
        "classifier": classifier,
        "seed": int(seed),
    }


def _rebuild(
    mask: pd.DataFrame, truth: pd.DataFrame, original: pd.DataFrame, realised_level: float
):
    """Minimal MaskResult-shaped object for the diagnostic helper.

    `realised_level` is passed in rather than recomputed: the masker measures it
    against the *maskable* columns, and recounting here over every column would
    quietly divide by the protected ones too and report a level that is too low.
    """
    from .missingness import MaskResult

    return MaskResult(
        X_masked=original.mask(mask),
        mask=mask,
        truth=truth,
        realised_level=float(realised_level),
    )


def baseline_cells(cfg: Config) -> list[tuple[str, float, str, int]]:
    """Reference cells with no artificial masking, one per imputer and seed."""
    if not cfg.experiment.include_baseline:
        return []
    return [
        (NO_MECHANISM, 0.0, imputer, seed)
        for imputer in cfg.experiment.imputers
        for seed in cfg.experiment.seeds
    ]


def grid_cells(cfg: Config) -> list[tuple[str, float, str, int]]:
    """Every (mechanism, level, imputer, seed) cell the run will execute."""
    e = cfg.experiment
    cells = [
        (mechanism, level, imputer, seed)
        for mechanism in e.mechanisms
        for level in e.levels
        for imputer in e.imputers
        for seed in e.seeds
    ]
    return baseline_cells(cfg) + cells


__all__ = ["CellResult", "run_cell", "grid_cells", "baseline_cells"]
