"""Grid orchestration: build the cohort once, then run every cell.

Two properties make a thousand-cell run survivable in practice:

* **Resumable.** Each finished cell is appended to `cells.jsonl` as it lands.
  Re-running the same config skips cells already recorded there, so an
  interrupted run continues instead of restarting.
* **Fail-soft.** A cell that raises is recorded with `status="failed"` and its
  traceback; the rest of the grid still completes. A run that silently dropped
  failures would be worse than one that reports them.

The cohort is built once and held fixed across seeds. The seeds vary the split,
the masking draw, the imputers and the classifiers - everything downstream of
the data itself - which is what the proposal's reproducibility section asks for.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from .config import Config, build_manifest
from .data import load_cohort
from .data.cohort import Cohort
from .experiment import grid_cells, run_cell
from .missingness import describe_mask
from .splits import Split, make_split

CELL_KEYS = ("mechanism", "level", "imputer", "seed")


def run(
    cfg: Config,
    output_root: str | Path | None = None,
    resume: bool = True,
    progress: bool = True,
) -> dict[str, Any]:
    """Execute the full grid defined by `cfg`.

    Returns a summary dict with the output directory and the result frame.
    """
    started = time.perf_counter()
    out_dir = Path(output_root or cfg.output_dir) / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort = load_cohort(cfg.dataset.name, cfg.dataset.params, seed=0)
    splits = {seed: make_split(cohort, seed=seed, **asdict(cfg.split)) for seed in cfg.experiment.seeds}

    manifest = build_manifest(
        cfg,
        extra={
            "cohort": _cohort_summary(cohort),
            "splits": {str(seed): split.sizes for seed, split in splits.items()},
        },
    )
    _write_json(out_dir / "manifest.json", manifest)
    _write_json(out_dir / "config.resolved.json", cfg.to_dict())
    cohort.missingness_profile().to_csv(out_dir / "natural_missingness_profile.csv", index=False)

    cells = grid_cells(cfg)
    cells_log = out_dir / "cells.jsonl"
    done = _completed_cells(cells_log) if resume else set()
    if not resume and cells_log.exists():
        cells_log.unlink()

    todo = [c for c in cells if _cell_key(c) not in done]
    if progress:
        print(
            f"[{cfg.run_name}] {len(cells)} cells "
            f"({len(cfg.experiment.classifiers)} classifiers x 2 splits each "
            f"= {cfg.n_result_rows()} rows); {len(done)} already done, "
            f"{len(todo)} to run on {cfg.compute.n_jobs} workers",
            file=sys.stderr,
            flush=True,
        )

    if todo:
        _execute(cfg, cohort, splits, todo, cells_log, progress)

    frame = _load_results(cells_log)
    if frame.empty:
        raise RuntimeError(f"no results produced; inspect {cells_log}")

    frame.to_parquet(out_dir / "results.parquet", index=False)
    frame.to_csv(out_dir / "results.csv", index=False)

    per_col = _load_per_column(out_dir)
    if per_col is not None and not per_col.empty:
        per_col.to_csv(out_dir / "imputation_per_column.csv", index=False)

    failures = frame[frame["status"] == "failed"]
    elapsed = time.perf_counter() - started
    manifest.update(
        {
            "finished_seconds": round(elapsed, 1),
            "result_rows": int(len(frame)),
            "failed_rows": int(len(failures)),
        }
    )
    _write_json(out_dir / "manifest.json", manifest)

    if progress:
        print(
            f"[{cfg.run_name}] done in {elapsed / 60:.1f} min -> {out_dir} "
            f"({len(frame)} rows, {len(failures)} failed)",
            file=sys.stderr,
            flush=True,
        )
    return {"output_dir": out_dir, "results": frame, "manifest": manifest, "cohort": cohort}


def _execute(
    cfg: Config,
    cohort: Cohort,
    splits: dict[int, Split],
    todo: list[tuple[str, float, str, int]],
    cells_log: Path,
    progress: bool,
) -> None:
    classifiers = list(cfg.experiment.classifiers)
    # Diagnostics and per-column tables are collected on the first seed only:
    # they are for verifying and interpreting the design, and three identical
    # copies of them add nothing but file size.
    first_seed = cfg.experiment.seeds[0]

    jobs = (
        delayed(run_cell)(
            cohort,
            splits[seed],
            mechanism,
            level,
            imputer,
            classifiers,
            seed,
            cfg,
            collect_per_column=(seed == first_seed),
            collect_diagnostics=(seed == first_seed),
        )
        for mechanism, level, imputer, seed in todo
    )

    parallel = Parallel(
        n_jobs=cfg.compute.n_jobs,
        backend="loky",
        return_as="generator_unordered",
        pre_dispatch="2*n_jobs",
    )

    per_column_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    started = time.perf_counter()

    # Cap the thread pool *inside* each worker. Setting n_jobs=1 on the
    # estimators is not enough: XGBoost, LightGBM, CatBoost and the BLAS behind
    # NumPy each start an OpenMP pool sized to the machine on first use, so the
    # workers end up contending for cores instead of fitting. This exports the
    # OMP/BLAS limits into the worker environment before those libraries are
    # imported, which is the only moment they are read. It has to be the
    # context manager: passing inner_max_num_threads to Parallel directly is
    # silently ignored, and loky falls back to cores // n_jobs.
    with parallel_config(backend="loky", inner_max_num_threads=max(1, cfg.compute.inner_threads)), \
            open(cells_log, "a", encoding="utf-8") as log:
        for i, result in enumerate(parallel(jobs), start=1):
            for row in result.rows:
                log.write(json.dumps(row, default=_json_default) + "\n")
            log.flush()
            if not result.per_column.empty:
                per_column_frames.append(result.per_column)
            if result.diagnostics:
                diagnostics.append(result.diagnostics)
            if result.error:
                print(f"  cell failed: {result.error}", file=sys.stderr, flush=True)
            if progress and (i % 10 == 0 or i == len(todo)):
                rate = i / max(1e-9, time.perf_counter() - started)
                remaining = (len(todo) - i) / rate / 60 if rate > 0 else float("nan")
                print(
                    f"  {i}/{len(todo)} cells  ({rate * 60:.1f}/min, "
                    f"~{remaining:.1f} min left)",
                    file=sys.stderr,
                    flush=True,
                )

    out_dir = cells_log.parent
    if per_column_frames:
        combined = pd.concat(per_column_frames, ignore_index=True)
        path = out_dir / "imputation_per_column.partial.csv"
        combined.to_csv(path, mode="a", header=not path.exists(), index=False)
    if diagnostics:
        _write_json(out_dir / "mask_diagnostics.json", diagnostics)


def verify_mechanisms(
    cfg: Config, level: float = 0.3, seed: int | None = None
) -> pd.DataFrame:
    """Check that MCAR, MAR and MNAR are actually distinct, before a full run.

    Reports, for each mechanism, how strongly the masking indicator correlates
    with the hidden value and with the observed conditioning index. Expected
    signature:

    * MCAR - both near zero
    * MAR  - conditioner correlation large, hidden-value correlation small
    * MNAR - hidden-value correlation large

    If a run's results look surprising, check this table first: it separates
    "the methods behave unexpectedly" from "the scenarios were not what I
    thought".
    """
    from .missingness import apply_missingness

    seed = cfg.experiment.seeds[0] if seed is None else seed
    cohort = load_cohort(cfg.dataset.name, cfg.dataset.params, seed=0)
    split = make_split(cohort, seed=seed, **asdict(cfg.split))
    X = cohort.X.iloc[split.train]

    rows = []
    for mechanism in cfg.experiment.mechanisms:
        result = apply_missingness(
            X,
            mechanism=mechanism,
            level=level,
            seed=seed,
            maskable=cohort.maskable,
            protected=cohort.protected,
            mar_strength=cfg.missingness.mar_strength,
            mnar_strength=cfg.missingness.mnar_strength,
            mnar_direction=cfg.missingness.mnar_direction,
            min_observed_per_column=cfg.missingness.min_observed_per_column,
        )
        rows.append({"mechanism": mechanism, **describe_mask(result, X, cohort.protected)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _cell_key(cell: tuple[str, float, str, int]) -> tuple:
    mechanism, level, imputer, seed = cell
    return (str(mechanism), round(float(level), 6), str(imputer), int(seed))


def _completed_cells(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    done: set[tuple] = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated final line from an interrupted run
            done.add(
                (
                    str(row["mechanism"]),
                    round(float(row["level"]), 6),
                    str(row["imputer"]),
                    int(row["seed"]),
                )
            )
    return done


def _load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    # A resumed run can append a cell twice if it was interrupted mid-write.
    return frame.drop_duplicates(
        subset=["mechanism", "level", "imputer", "classifier", "seed", "split"], keep="last"
    ).reset_index(drop=True)


def _load_per_column(out_dir: Path) -> pd.DataFrame | None:
    path = out_dir / "imputation_per_column.partial.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    return frame.drop_duplicates(
        subset=["mechanism", "level", "imputer", "seed", "column"], keep="last"
    )


def _cohort_summary(cohort: Cohort) -> dict[str, Any]:
    profile = cohort.missingness_profile()
    return {
        "source": cohort.meta.get("source", "unknown"),
        "n_samples": cohort.n_samples,
        "n_features": cohort.n_features,
        "prevalence": round(cohort.prevalence, 5),
        "n_maskable": len(cohort.maskable),
        "n_protected": len(cohort.protected),
        "overall_natural_missingness": round(float(cohort.X.isna().to_numpy().mean()), 5),
        "most_missing": profile.head(5)[["column", "frac_missing"]].to_dict("records"),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")

