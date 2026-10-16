"""Metric definitions, checked against values worked out by hand.

The imputation metrics are the ones worth pinning down: they are computed on a
subset of a matrix, in two different scalings, with an adjusted-R-squared term
that depends on a feature count. All three are easy to get subtly wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icumiss.metrics import (
    classification_metrics,
    imputation_metrics,
    imputation_metrics_per_column,
    pick_threshold,
    train_statistics,
)


@pytest.fixture
def tiny():
    """Two columns, three masked cells, hand-checkable numbers."""
    truth = pd.DataFrame({"a": [1.0, 2.0, np.nan], "b": [np.nan, 10.0, np.nan]})
    imputed = pd.DataFrame({"a": [1.5, 2.5, 9.0], "b": [9.0, 12.0, 9.0]})
    mask = pd.DataFrame({"a": [True, True, False], "b": [False, True, False]})
    return truth, imputed, mask


def test_metrics_use_only_masked_cells(tiny):
    truth, imputed, mask = tiny
    out = imputation_metrics(truth, imputed, mask, n_features_used=1)

    # Masked residuals are -0.5, -0.5 and -2.0; the unmasked cells are ignored.
    assert out["n_masked"] == 3
    assert out["mae_raw"] == pytest.approx((0.5 + 0.5 + 2.0) / 3)
    assert out["rmse_raw"] == pytest.approx(np.sqrt((0.25 + 0.25 + 4.0) / 3))


def test_r2_matches_the_proposals_formula(tiny):
    truth, imputed, mask = tiny
    out = imputation_metrics(truth, imputed, mask, n_features_used=1)

    y = np.array([1.0, 2.0, 10.0])
    yhat = np.array([1.5, 2.5, 12.0])
    sse = np.sum((y - yhat) ** 2)
    sst = np.sum((y - y.mean()) ** 2)
    expected_r2 = 1 - sse / sst
    assert out["r2_raw"] == pytest.approx(expected_r2)

    n, p = 3, 1
    expected_adj = 1 - (1 - expected_r2) * (n - 1) / (n - p - 1)
    assert out["adj_r2_raw"] == pytest.approx(expected_adj)


def test_adjusted_r2_is_undefined_when_degrees_of_freedom_run_out(tiny):
    truth, imputed, mask = tiny
    out = imputation_metrics(truth, imputed, mask, n_features_used=5)
    assert np.isnan(out["adj_r2_raw"])
    assert np.isfinite(out["r2_raw"])


def test_perfect_imputation_scores_perfectly():
    truth = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    mask = pd.DataFrame({"a": [True, True, True]})
    out = imputation_metrics(truth, truth.copy(), mask, n_features_used=1)
    assert out["mae_raw"] == pytest.approx(0.0)
    assert out["rmse_raw"] == pytest.approx(0.0)
    assert out["r2_raw"] == pytest.approx(1.0)


def test_mean_imputation_scores_zero_r2():
    """Predicting the mean of the masked values is the R-squared = 0 reference."""
    truth = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    imputed = pd.DataFrame({"a": [2.5, 2.5, 2.5, 2.5]})
    mask = pd.DataFrame({"a": [True] * 4})
    out = imputation_metrics(truth, imputed, mask, n_features_used=1)
    assert out["r2_raw"] == pytest.approx(0.0)


def test_standardised_metrics_are_scale_free():
    """The z-scored metric must not change when a column is re-expressed in other units.

    This is why the z variants exist: the raw pooled numbers are dominated by
    whichever variable happens to have the widest range.
    """
    truth = pd.DataFrame({"small": [1.0, 2.0, 3.0], "large": [100.0, 200.0, 300.0]})
    imputed = pd.DataFrame({"small": [1.1, 2.1, 3.1], "large": [110.0, 210.0, 310.0]})
    mask = pd.DataFrame({"small": [True] * 3, "large": [True] * 3})
    stats = {"small": (2.0, 1.0), "large": (200.0, 100.0)}

    out = imputation_metrics(truth, imputed, mask, n_features_used=1, train_stats=stats)

    # Both columns are off by 0.1 standard deviations, so the pooled z error is 0.1.
    assert out["mae_z"] == pytest.approx(0.1)
    # The raw version is dominated by the large column: (0.1 + 10)/2.
    assert out["mae_raw"] == pytest.approx(5.05)


def test_per_column_metrics_split_the_pooled_number(tiny):
    truth, imputed, mask = tiny
    per_col = imputation_metrics_per_column(truth, imputed, mask, n_features_used=1)
    assert set(per_col["column"]) == {"a", "b"}
    assert per_col.set_index("column").loc["a", "n_masked"] == 2
    assert per_col.set_index("column").loc["a", "mae_raw"] == pytest.approx(0.5)


def test_train_statistics_ignore_missing_values():
    frame = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    stats = train_statistics(frame)
    assert stats["a"][0] == pytest.approx(2.0)


def test_empty_mask_returns_nan_not_an_error():
    truth = pd.DataFrame({"a": [np.nan, np.nan]})
    imputed = pd.DataFrame({"a": [1.0, 2.0]})
    mask = pd.DataFrame({"a": [False, False]})
    out = imputation_metrics(truth, imputed, mask, n_features_used=1)
    assert out["n_masked"] == 0
    assert np.isnan(out["rmse_raw"])


# ----------------------------------------------------------------------
# classification
# ----------------------------------------------------------------------
def test_perfect_and_random_classifiers_bracket_auroc():
    y = np.array([0, 0, 1, 1])
    assert classification_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]), 0.5)["auroc"] == 1.0
    assert classification_metrics(y, np.array([0.9, 0.8, 0.2, 0.1]), 0.5)["auroc"] == 0.0


def test_brier_and_log_loss_reward_honest_probabilities():
    y = np.array([0, 0, 0, 1])
    confident_right = classification_metrics(y, np.array([0.02, 0.02, 0.02, 0.98]), 0.5)
    hedged = classification_metrics(y, np.array([0.25, 0.25, 0.25, 0.25]), 0.5)
    assert confident_right["brier"] < hedged["brier"]
    assert confident_right["log_loss"] < hedged["log_loss"]


def test_threshold_is_applied_as_given():
    y = np.array([0, 1, 1])
    probs = np.array([0.4, 0.45, 0.9])
    at_half = classification_metrics(y, probs, 0.5)
    lower = classification_metrics(y, probs, 0.42)
    assert lower["recall"] > at_half["recall"]
    assert lower["threshold"] == pytest.approx(0.42)


def test_threshold_selection_beats_the_default_on_imbalanced_data():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=600) < 0.08).astype(int)
    # A usefully ranked but badly calibrated score: no case ever reaches 0.5.
    probs = np.clip(0.02 + 0.25 * y + rng.normal(0, 0.05, 600), 0.001, 0.999)

    chosen = pick_threshold(y, probs)
    assert chosen < 0.5
    tuned = classification_metrics(y, probs, chosen)
    default = classification_metrics(y, probs, 0.5)
    assert tuned["f1"] > default["f1"]
    assert default["f1"] == pytest.approx(0.0)


def test_single_class_eval_returns_nan_for_ranking_metrics():
    y = np.zeros(10, dtype=int)
    out = classification_metrics(y, np.linspace(0.1, 0.9, 10), 0.5)
    assert np.isnan(out["auroc"])
    assert np.isnan(out["auprc"])
    assert np.isfinite(out["brier"])
