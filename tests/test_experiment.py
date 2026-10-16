"""Cell execution and grid orchestration.

Covers the contract the runner depends on: a cell produces the rows it
promises, a broken cell is reported rather than silently dropped, the grid is
the size the config says it is, and the same seed gives the same answer.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from icumiss.config import Config
from icumiss.data import load_cohort
from icumiss.experiment import baseline_cells, grid_cells, run_cell
from icumiss.runner import _cell_key, _completed_cells, _load_results
from icumiss.splits import make_split

CLASSIFIERS = ["logreg", "xgboost"]


@pytest.fixture(scope="module")
def cfg():
    return Config.from_dict(
        {
            "run_name": "unit",
            "dataset": {"name": "synthetic", "params": {"n_patients": 900, "seed": 0}},
            "experiment": {
                "mechanisms": ["mcar", "mar", "mnar"],
                "levels": [0.2, 0.5],
                "imputers": ["median", "mice"],
                "classifiers": CLASSIFIERS,
                "seeds": [0, 1],
            },
            "imputer_params": {"mice": {"max_iter": 3}},
            "model_params": {"xgboost": {"n_estimators": 40, "early_stopping_rounds": 10}},
        }
    )


@pytest.fixture(scope="module")
def setup(cfg):
    cohort = load_cohort(cfg.dataset.name, cfg.dataset.params)
    split = make_split(cohort, seed=0, test_size=0.2, val_size=0.15)
    return cohort, split


def test_cell_returns_a_row_per_classifier_and_split(cfg, setup):
    cohort, split = setup
    result = run_cell(cohort, split, "mcar", 0.3, "median", CLASSIFIERS, 0, cfg)

    assert result.error is None
    assert len(result.rows) == len(CLASSIFIERS) * 2
    frame = pd.DataFrame(result.rows)
    assert set(frame["split"]) == {"val", "test"}
    assert set(frame["classifier"]) == set(CLASSIFIERS)
    assert (frame["status"] == "ok").all()


def test_cell_records_the_metrics_the_study_reports(cfg, setup):
    cohort, split = setup
    result = run_cell(cohort, split, "mnar", 0.4, "median", ["logreg"], 0, cfg)
    row = pd.DataFrame(result.rows).query("split == 'test'").iloc[0]

    for metric in ("auroc", "auprc", "f1", "precision", "log_loss", "brier"):
        assert np.isfinite(row[metric]), f"{metric} missing"
    for metric in ("imp_mae_raw", "imp_rmse_raw", "imp_r2_raw", "imp_adj_r2_raw",
                   "imp_mae_z", "imp_rmse_z", "imp_r2_z"):
        assert np.isfinite(row[metric]), f"{metric} missing"

    assert row["imp_n_masked"] > 0
    assert row["realised_level"] == pytest.approx(0.4, abs=0.02)
    assert 0.0 <= row["auroc"] <= 1.0


def test_baseline_cell_has_no_imputation_ground_truth(cfg, setup):
    """At level 0 nothing was masked, so there is nothing to score against."""
    cohort, split = setup
    result = run_cell(cohort, split, "none", 0.0, "median", ["logreg"], 0, cfg)
    row = pd.DataFrame(result.rows).query("split == 'test'").iloc[0]

    assert row["imp_n_masked"] == 0
    assert row["realised_level"] == 0.0
    # The classifier still ran: natural missingness had to be filled first.
    assert np.isfinite(row["auroc"])


def test_a_broken_cell_is_reported_not_swallowed(cfg, setup):
    """A thousand-cell run must not die on one bad cell, nor hide it."""
    cohort, split = setup
    result = run_cell(cohort, split, "mcar", 0.3, "not_an_imputer", CLASSIFIERS, 0, cfg)

    assert result.error is not None
    assert len(result.rows) == len(CLASSIFIERS)
    assert all(row["status"] == "failed" for row in result.rows)
    assert "traceback" in result.diagnostics


def test_same_seed_reproduces_the_same_result(cfg, setup):
    cohort, split = setup
    a = run_cell(cohort, split, "mar", 0.3, "median", ["logreg"], 0, cfg)
    b = run_cell(cohort, split, "mar", 0.3, "median", ["logreg"], 0, cfg)

    row_a = pd.DataFrame(a.rows).query("split == 'test'").iloc[0]
    row_b = pd.DataFrame(b.rows).query("split == 'test'").iloc[0]
    assert row_a["auroc"] == pytest.approx(row_b["auroc"], abs=1e-12)
    assert row_a["imp_rmse_z"] == pytest.approx(row_b["imp_rmse_z"], abs=1e-12)


def test_different_seeds_give_different_results(cfg, setup):
    cohort, split = setup
    a = run_cell(cohort, split, "mcar", 0.3, "median", ["logreg"], 0, cfg)
    b = run_cell(cohort, split, "mcar", 0.3, "median", ["logreg"], 1, cfg)
    row_a = pd.DataFrame(a.rows).query("split == 'test'").iloc[0]
    row_b = pd.DataFrame(b.rows).query("split == 'test'").iloc[0]
    assert row_a["imp_rmse_z"] != pytest.approx(row_b["imp_rmse_z"], abs=1e-9)


def test_mnar_is_harder_to_reconstruct_than_mcar(cfg, setup):
    """A substantive check, not a plumbing one.

    Under MNAR the hidden values are systematically the large ones, so no
    imputer that only sees the small ones can recover them. Reconstruction
    error must be clearly worse than under MCAR at the same level. If this ever
    fails, the mechanisms are not doing what the study says they do.
    """
    cohort, split = setup
    mcar = run_cell(cohort, split, "mcar", 0.4, "median", ["logreg"], 0, cfg)
    mnar = run_cell(cohort, split, "mnar", 0.4, "median", ["logreg"], 0, cfg)

    rmse_mcar = pd.DataFrame(mcar.rows).query("split == 'test'").iloc[0]["imp_rmse_z"]
    rmse_mnar = pd.DataFrame(mnar.rows).query("split == 'test'").iloc[0]["imp_rmse_z"]
    assert rmse_mnar > rmse_mcar


def test_grid_size_matches_the_configured_design(cfg):
    cells = grid_cells(cfg)
    assert len(cells) == cfg.n_cells()
    assert len(cells) == len(baseline_cells(cfg)) + 3 * 2 * 2 * 2
    assert len(set(cells)) == len(cells), "the grid contains duplicate cells"
    assert cfg.n_result_rows() == cfg.n_cells() * len(CLASSIFIERS) * 2


def test_resume_skips_cells_already_recorded(cfg, tmp_path):
    """Resumability: a re-run must not repeat finished work."""
    log = tmp_path / "cells.jsonl"
    done_cells = grid_cells(cfg)[:5]
    with open(log, "w", encoding="utf-8") as fh:
        for mechanism, level, imputer, seed in done_cells:
            fh.write(json.dumps({
                "mechanism": mechanism, "level": level, "imputer": imputer,
                "seed": seed, "classifier": "logreg", "split": "test", "status": "ok",
            }) + "\n")

    done = _completed_cells(log)
    assert done == {_cell_key(c) for c in done_cells}
    remaining = [c for c in grid_cells(cfg) if _cell_key(c) not in done]
    assert len(remaining) == cfg.n_cells() - 5


def test_a_truncated_log_line_does_not_break_resume(tmp_path):
    """An interrupted run leaves a half-written final line."""
    log = tmp_path / "cells.jsonl"
    log.write_text(
        json.dumps({"mechanism": "mcar", "level": 0.1, "imputer": "median",
                    "seed": 0, "classifier": "logreg", "split": "test"}) + "\n"
        + '{"mechanism": "mcar", "level": 0.2, "imp'
    )
    assert len(_completed_cells(log)) == 1
    assert len(_load_results(log)) == 1


def test_duplicate_rows_from_a_resumed_run_are_collapsed(tmp_path):
    log = tmp_path / "cells.jsonl"
    row = {"mechanism": "mcar", "level": 0.1, "imputer": "median", "seed": 0,
           "classifier": "logreg", "split": "test", "status": "ok", "auroc": 0.8}
    with open(log, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.write(json.dumps({**row, "auroc": 0.9}) + "\n")

    frame = _load_results(log)
    assert len(frame) == 1
    assert frame.iloc[0]["auroc"] == 0.9  # the later write wins
