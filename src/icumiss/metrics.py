"""Evaluation metrics for both stages of the experiment.

Stage 1 - reconstruction quality, scored **only on artificially masked cells**,
where the original value is known. Naturally missing cells have no ground truth
and are excluded everywhere.

Stage 2 - downstream mortality prediction: discrimination (AUROC), positive
class performance under imbalance (AUPRC, F1, precision) and probability
quality (log loss, Brier).

Two measurement decisions deserve to be stated plainly, because they change how
the headline numbers read:

**Scale.** Pooling raw MAE/RMSE across columns measured in different units lets
the widest-ranged variable dominate: platelets span hundreds, pH spans
fractions of a unit. The proposal's formulas are reported as written
(`*_raw`), and the same metrics are also reported on z-scores taken from the
*training* distribution (`*_z`). The z-scored versions are the ones to compare
methods on; the raw ones are reported because the proposal specifies them, and
per-column values are available for interpretation.

**Threshold.** F1 and precision need a cut-point, and at ICU mortality
prevalence (roughly 10%) the default 0.5 makes every model look falsely
precise. The threshold is therefore chosen on the validation split to maximise
F1 and applied unchanged to the test split. Metrics at the fixed 0.5 cut are
also reported for reference.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ----------------------------------------------------------------------
# stage 1: imputation quality
# ----------------------------------------------------------------------
def imputation_metrics(
    truth: pd.DataFrame,
    imputed: pd.DataFrame,
    mask: pd.DataFrame,
    n_features_used: int,
    train_stats: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Reconstruction error over the masked cells.

    Parameters
    ----------
    truth
        Original values, NaN everywhere except the masked positions.
    imputed
        Imputer output for the same rows and columns.
    mask
        True at artificially masked positions.
    n_features_used
        `p` in the adjusted-R-squared formula: the number of other features the
        imputer draws on when filling a column.
    train_stats
        Per-column (mean, sd) from the training split, used for the z-scored
        variants. Training statistics are used - not statistics of the
        evaluation split - so the scaling itself introduces no leakage.

    Returns
    -------
    dict
        `n_masked`, then MAE / RMSE / R2 / adjusted R2 in both raw (`_raw`) and
        z-scored (`_z`) form.
    """
    common = [c for c in truth.columns if c in imputed.columns and c in mask.columns]
    mask_arr = mask[common].to_numpy()
    y_true = truth[common].to_numpy(dtype=float)[mask_arr]
    y_hat = imputed[common].to_numpy(dtype=float)[mask_arr]

    valid = np.isfinite(y_true) & np.isfinite(y_hat)
    y_true, y_hat = y_true[valid], y_hat[valid]
    out: dict[str, float] = {"n_masked": int(y_true.size)}

    if y_true.size == 0:
        return {**out, **{f"{m}_{s}": np.nan for m in ("mae", "rmse", "r2", "adj_r2") for s in ("raw", "z")}}

    out.update(_error_block(y_true, y_hat, n_features_used, suffix="raw"))

    if train_stats is not None:
        z_true, z_hat = _zscore_masked(truth, imputed, mask, common, train_stats)
        out.update(_error_block(z_true, z_hat, n_features_used, suffix="z"))
    else:
        out.update({f"{m}_z": np.nan for m in ("mae", "rmse", "r2", "adj_r2")})

    return out


def _error_block(
    y_true: np.ndarray, y_hat: np.ndarray, n_features_used: int, suffix: str
) -> dict[str, float]:
    resid = y_true - y_hat
    n = y_true.size
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))

    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(sse / n))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan

    # adj R2 = 1 - (1 - R2)(n - 1)/(n - p - 1); undefined once p + 1 >= n.
    p = int(n_features_used)
    denom = n - p - 1
    adj = 1.0 - (1.0 - r2) * (n - 1) / denom if (np.isfinite(r2) and denom > 0) else np.nan

    return {
        f"mae_{suffix}": mae,
        f"rmse_{suffix}": rmse,
        f"r2_{suffix}": float(r2) if np.isfinite(r2) else np.nan,
        f"adj_r2_{suffix}": float(adj) if np.isfinite(adj) else np.nan,
    }


def _zscore_masked(
    truth: pd.DataFrame,
    imputed: pd.DataFrame,
    mask: pd.DataFrame,
    columns: list[str],
    train_stats: Mapping[str, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Pool masked cells after putting every column on a common scale."""
    t_parts, h_parts = [], []
    for col in columns:
        sel = mask[col].to_numpy()
        if not sel.any():
            continue
        mu, sd = train_stats.get(col, (0.0, 1.0))
        sd = sd if (np.isfinite(sd) and sd > 1e-12) else 1.0
        t = (truth[col].to_numpy(dtype=float)[sel] - mu) / sd
        h = (imputed[col].to_numpy(dtype=float)[sel] - mu) / sd
        keep = np.isfinite(t) & np.isfinite(h)
        t_parts.append(t[keep])
        h_parts.append(h[keep])
    if not t_parts:
        return np.array([]), np.array([])
    return np.concatenate(t_parts), np.concatenate(h_parts)


def imputation_metrics_per_column(
    truth: pd.DataFrame,
    imputed: pd.DataFrame,
    mask: pd.DataFrame,
    n_features_used: int,
) -> pd.DataFrame:
    """The same reconstruction metrics, one row per column.

    Needed to interpret the pooled numbers: a method can win overall while
    losing badly on the one variable that carries the clinical signal.
    """
    rows = []
    for col in truth.columns:
        if col not in imputed.columns or col not in mask.columns:
            continue
        sel = mask[col].to_numpy()
        if not sel.any():
            continue
        t = truth[col].to_numpy(dtype=float)[sel]
        h = imputed[col].to_numpy(dtype=float)[sel]
        keep = np.isfinite(t) & np.isfinite(h)
        if keep.sum() == 0:
            continue
        block = _error_block(t[keep], h[keep], n_features_used, suffix="raw")
        rows.append({"column": col, "n_masked": int(keep.sum()), **block})
    return pd.DataFrame(rows)


def train_statistics(X_train: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Per-column (mean, sd) over observed training values."""
    stats = {}
    for col in X_train.columns:
        values = X_train[col].to_numpy(dtype=float)
        if np.isfinite(values).any():
            stats[col] = (float(np.nanmean(values)), float(np.nanstd(values)))
        else:
            stats[col] = (0.0, 1.0)
    return stats


# ----------------------------------------------------------------------
# stage 2: classification
# ----------------------------------------------------------------------
def pick_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1") -> float:
    """Choose a decision threshold on held-out validation probabilities.

    Candidates are the unique predicted probabilities (thinned for speed), so
    the search covers every achievable operating point.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.sum() == 0 or y_true.sum() == y_true.size:
        return 0.5

    candidates = np.unique(y_prob)
    if candidates.size > 512:
        candidates = np.quantile(y_prob, np.linspace(0.01, 0.99, 512))
    candidates = np.unique(np.clip(candidates, 1e-6, 1 - 1e-6))

    scorer = {"f1": f1_score, "precision": precision_score, "recall": recall_score}[metric]
    best_score, best_t = -np.inf, 0.5
    for t in candidates:
        score = scorer(y_true, (y_prob >= t).astype(int), zero_division=0)
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t


def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, float]:
    """Discrimination, threshold and probability-quality metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-15, 1 - 1e-15)

    single_class = y_true.sum() == 0 or y_true.sum() == y_true.size
    y_pred = (y_prob >= threshold).astype(int)
    y_pred_half = (y_prob >= 0.5).astype(int)

    return {
        "auroc": np.nan if single_class else float(roc_auc_score(y_true, y_prob)),
        "auprc": np.nan if single_class else float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_at_0.5": float(f1_score(y_true, y_pred_half, zero_division=0)),
        "precision_at_0.5": float(precision_score(y_true, y_pred_half, zero_division=0)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "threshold": float(threshold),
        "mean_predicted_risk": float(y_prob.mean()),
        "observed_prevalence": float(y_true.mean()),
        "n_samples": int(y_true.size),
    }
