"""Controlled MCAR / MAR / MNAR masking.

The study does not claim to identify the true mechanism behind the missingness
already present in observational data - that is not identifiable from the data
alone. Instead each mechanism is *imposed* by masking values that were
originally recorded, so the experimental condition is known by construction and
the masked values provide a ground truth to score imputation against.

Only currently-observed cells in maskable columns are eligible. Protected
columns (demographics) are never masked: they stay complete so that MAR can
depend on genuinely observed data.

How a level is hit exactly
--------------------------
Each eligible column gets a per-cell score, standard logistic noise is added,
and the top ``k = round(level * n_observed)`` cells are masked. Taking a noisy
top-k is equivalent to sampling k cells without replacement under a logit model
on the score, so the realised missingness level is exact (which keeps the x-axis
of every results plot meaningful) while the dependence the mechanism calls for
is preserved. ``strength`` is the logit coefficient on the standardised score:
0 degenerates to MCAR, and larger values sharpen the dependence.

The three mechanisms differ only in what the score is built from:

=========  ======================================================  ===========================
mechanism  score depends on                                        identifiability
=========  ======================================================  ===========================
MCAR       nothing (noise only)                                    missingness ignorable
MAR        other *observed* variables (the protected columns)      ignorable given observed data
MNAR       the value being masked itself                           non-ignorable
=========  ======================================================  ===========================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .utils import logistic_noise, standardize, top_k_mask


@dataclass
class MaskResult:
    """A masked copy of a feature matrix plus everything needed to score it.

    Attributes
    ----------
    X_masked
        Feature matrix with the artificially masked cells set to NaN. This is
        the only version an imputer is ever allowed to see.
    mask
        True where a cell was artificially masked. Ground truth exists at
        exactly these positions, and imputation metrics are computed only here.
    truth
        The original values at the masked positions, aligned to `X_masked`.
    realised_level
        Masked cells as a fraction of eligible observed cells. Equals the
        requested level up to rounding.
    """

    X_masked: pd.DataFrame
    mask: pd.DataFrame
    truth: pd.DataFrame
    realised_level: float
    per_column: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_masked(self) -> int:
        return int(self.mask.to_numpy().sum())


def apply_missingness(
    X: pd.DataFrame,
    mechanism: str,
    level: float,
    seed: int,
    maskable: Sequence[str],
    protected: Sequence[str],
    mar_strength: float = 2.0,
    mnar_strength: float = 2.0,
    mnar_direction: str = "high",
    min_observed_per_column: int = 10,
) -> MaskResult:
    """Impose `mechanism` at `level` on the maskable columns of `X`.

    Parameters
    ----------
    X
        Feature matrix, which may already contain natural missing values. Those
        are left exactly as they are: they are part of the dataset, not part of
        the simulated condition, and they carry no ground truth.
    mechanism
        "mcar", "mar" or "mnar".
    level
        Fraction of eligible observed cells to mask, in (0, 1).
    seed
        Controls the masking draw. Independent streams are derived per column so
        that a column's masking does not shift when the column order changes.
    maskable, protected
        Column roles, from the cohort.
    mar_strength, mnar_strength
        Logit coefficient on the standardised driver score.
    mnar_direction
        "high" makes large values more likely to go missing (the documented
        sensitivity-analysis rule); "low" reverses it.
    min_observed_per_column
        Never leave a column with fewer than this many observed values, so an
        imputer is never handed an all-NaN training column. With realistic
        cohort sizes this cap does not bind below 70% missingness; whenever it
        does bind, the realised level is reported and will differ from the
        request.
    """
    mechanism = mechanism.lower()
    if mechanism not in ("mcar", "mar", "mnar"):
        raise ValueError(f"unknown mechanism {mechanism!r}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1); got {level}")
    if mnar_direction not in ("high", "low"):
        raise ValueError("mnar_direction must be 'high' or 'low'")

    maskable = [c for c in maskable if c in X.columns]
    protected = [c for c in protected if c in X.columns]
    if not maskable:
        raise ValueError("no maskable columns present in X")

    conditioners = _mar_conditioners(X, protected, maskable) if mechanism == "mar" else None

    mask = pd.DataFrame(False, index=X.index, columns=X.columns)
    rows = []

    for col in maskable:
        values = X[col].to_numpy(dtype=float)
        observed = np.flatnonzero(~np.isnan(values))
        n_obs = observed.size
        if n_obs == 0:
            rows.append(
                {"column": col, "n_observed": 0, "n_masked": 0, "realised_level": np.nan}
            )
            continue

        k = int(round(level * n_obs))
        capped = min(k, max(0, n_obs - min_observed_per_column))
        # A column with very few observations keeps at least one value.
        if capped == 0 and k > 0 and n_obs > 1:
            capped = min(k, n_obs - 1)

        # Per-column stream: reproducible and order-independent.
        rng = np.random.default_rng([seed, _column_stream(col), _mechanism_stream(mechanism)])

        if capped > 0:
            score = _score(
                mechanism=mechanism,
                values=values,
                observed=observed,
                conditioners=conditioners,
                mar_strength=mar_strength,
                mnar_strength=mnar_strength,
                mnar_direction=mnar_direction,
                rng=rng,
            )
            chosen = observed[top_k_mask(score, capped)]
            mask.loc[X.index[chosen], col] = True

        rows.append(
            {
                "column": col,
                "n_observed": int(n_obs),
                "n_masked": int(capped),
                "requested_level": float(level),
                "realised_level": float(capped / n_obs),
                "level_capped": bool(capped < k),
            }
        )

    mask_arr = mask.to_numpy()
    X_masked = X.mask(mask)
    truth = X.where(mask)

    eligible_observed = int((~X[maskable].isna()).to_numpy().sum())
    realised = float(mask_arr.sum() / eligible_observed) if eligible_observed else 0.0

    return MaskResult(
        X_masked=X_masked,
        mask=mask,
        truth=truth,
        realised_level=realised,
        per_column=pd.DataFrame(rows),
    )


def _score(
    mechanism: str,
    values: np.ndarray,
    observed: np.ndarray,
    conditioners: np.ndarray | None,
    mar_strength: float,
    mnar_strength: float,
    mnar_direction: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Per-observed-cell masking score for one column."""
    n_obs = observed.size
    noise = logistic_noise(rng, n_obs)

    if mechanism == "mcar":
        # Nothing but noise: missingness is independent of everything, observed
        # or not, which is exactly the MCAR definition.
        return noise

    if mechanism == "mar":
        assert conditioners is not None
        # Depends only on fully observed variables, never on this column.
        driver = standardize(conditioners[observed])
        return mar_strength * driver + noise

    # MNAR: depends on the value that is about to be hidden, so the missingness
    # cannot be explained by anything that remains observable.
    driver = standardize(values[observed])
    sign = 1.0 if mnar_direction == "high" else -1.0
    return mnar_strength * sign * driver + noise


def _mar_conditioners(
    X: pd.DataFrame, protected: Sequence[str], maskable: Sequence[str]
) -> np.ndarray:
    """A single standardised severity-like index built from observed columns.

    Protected columns are used because they are guaranteed complete, which is
    what makes the scenario MAR rather than "missing at random given data that
    is itself missing". Weights are fixed, not random, so the MAR condition is
    identical across seeds and only the draw changes.
    """
    usable = [c for c in protected if X[c].notna().all() and X[c].nunique() > 1]
    if not usable:
        raise ValueError(
            "MAR needs at least one complete, non-constant protected column to "
            "condition on; none available"
        )

    # Age dominates where present - it is the standard clinical driver of both
    # test ordering and outcome - with the remaining protected columns adding a
    # smaller, deterministic contribution.
    weights = {c: 1.0 for c in usable}
    if "age" in weights:
        weights["age"] = 2.0
    if "elective_admission" in weights:
        weights["elective_admission"] = -1.0

    total = np.zeros(len(X), dtype=float)
    for col, weight in weights.items():
        total += weight * standardize(X[col].to_numpy(dtype=float))
    return standardize(total)


def _column_stream(col: str) -> int:
    """Stable 32-bit stream id from a column name."""
    import zlib

    return int(zlib.crc32(col.encode("utf-8")) & 0x7FFFFFFF)


def _mechanism_stream(mechanism: str) -> int:
    return {"mcar": 11, "mar": 22, "mnar": 33}[mechanism]


# ----------------------------------------------------------------------
# diagnostics
# ----------------------------------------------------------------------
def describe_mask(result: MaskResult, X: pd.DataFrame, protected: Sequence[str]) -> dict:
    """Evidence that the imposed mechanism behaves as intended.

    Returns the correlation between a cell's masking indicator and (a) the
    value that was hidden, and (b) the observed conditioning index. The
    expected signature is:

    * MCAR - both near zero
    * MAR  - (a) near zero once the conditioner is accounted for, (b) large
    * MNAR - (a) large

    Run this on a pilot before trusting a full result set; it is the check that
    the mechanisms are actually distinct.
    """
    masked_cols = [c for c in result.mask.columns if result.mask[c].any()]
    if not masked_cols:
        return {"corr_with_hidden_value": np.nan, "corr_with_conditioner": np.nan}

    conditioner = _mar_conditioners(X, protected, masked_cols)

    value_corrs, cond_corrs = [], []
    for col in masked_cols:
        values = X[col].to_numpy(dtype=float)
        observed = ~np.isnan(values)
        if observed.sum() < 10:
            continue
        indicator = result.mask[col].to_numpy()[observed].astype(float)
        if indicator.std() < 1e-12:
            continue
        z_val = standardize(values[observed])
        value_corrs.append(float(np.corrcoef(indicator, z_val)[0, 1]))
        z_cond = conditioner[observed]
        if z_cond.std() > 1e-12:
            cond_corrs.append(float(np.corrcoef(indicator, z_cond)[0, 1]))

    return {
        "corr_with_hidden_value": float(np.mean(value_corrs)) if value_corrs else np.nan,
        "corr_with_conditioner": float(np.mean(cond_corrs)) if cond_corrs else np.nan,
        "n_columns_masked": len(masked_cols),
        "realised_level": result.realised_level,
    }
