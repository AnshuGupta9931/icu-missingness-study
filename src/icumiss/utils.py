"""Small numeric helpers shared across the pipeline."""

from __future__ import annotations

import numpy as np


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    """Numerically stable logistic function."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def solve_intercept(scores: np.ndarray, target_rate: float, tol: float = 1e-9) -> float:
    """Find b with mean(sigmoid(scores + b)) == target_rate.

    The mean is strictly increasing in b, so plain bisection is both safe and
    exact to machine precision. Used to hit a requested event rate without
    disturbing the relative ordering the scores encode.
    """
    if not 0.0 < target_rate < 1.0:
        raise ValueError(f"target_rate must be in (0, 1); got {target_rate}")
    lo, hi = -60.0, 60.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if sigmoid(scores + mid).mean() < target_rate:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def standardize(values: np.ndarray) -> np.ndarray:
    """Z-score, ignoring NaN; constant or all-NaN input maps to zeros."""
    values = np.asarray(values, dtype=float)
    mu = np.nanmean(values) if np.isfinite(values).any() else 0.0
    sd = np.nanstd(values) if np.isfinite(values).any() else 0.0
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(values)
    return (values - mu) / sd


def logistic_noise(rng: np.random.Generator, size: int | tuple[int, ...]) -> np.ndarray:
    """Draws from the standard logistic distribution.

    Adding logistic noise to a score and then taking the top-k is equivalent to
    sampling k items without replacement under a logit model on that score. It
    gives an exact count while preserving the intended dependence.
    """
    u = rng.uniform(low=np.finfo(float).tiny, high=1.0, size=size)
    return np.log(u) - np.log1p(-u)


def top_k_mask(scores: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask selecting the k largest scores (ties broken by position)."""
    n = scores.shape[0]
    k = int(np.clip(k, 0, n))
    mask = np.zeros(n, dtype=bool)
    if k == 0:
        return mask
    if k == n:
        mask[:] = True
        return mask
    idx = np.argpartition(-scores, k - 1)[:k]
    mask[idx] = True
    return mask
