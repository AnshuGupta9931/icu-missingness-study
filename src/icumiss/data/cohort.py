"""The cohort container passed between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Cohort:
    """A modelling-ready ICU cohort.

    Attributes
    ----------
    X
        Numeric feature matrix. May already contain the dataset's own
        (naturally occurring) missing values.
    y
        Binary mortality outcome, 1 = died.
    maskable
        Columns eligible for artificial masking. Clinical measurements only:
        the columns a real EHR could plausibly fail to record.
    protected
        Columns that are never masked and are guaranteed complete. These are
        the conditioning variables for MAR, which is what makes the MAR
        scenario genuinely "missing at random given observed data".
    groups
        Named column groups (vitals, labs, ...) used for reporting.
    """

    X: pd.DataFrame
    y: pd.Series
    maskable: list[str]
    protected: list[str]
    groups: dict[str, list[str]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.X) != len(self.y):
            raise ValueError(f"X has {len(self.X)} rows but y has {len(self.y)}")
        unknown = set(self.maskable) | set(self.protected)
        unknown -= set(self.X.columns)
        if unknown:
            raise ValueError(f"columns declared but absent from X: {sorted(unknown)}")
        overlap = set(self.maskable) & set(self.protected)
        if overlap:
            raise ValueError(f"columns cannot be both maskable and protected: {sorted(overlap)}")
        if self.protected:
            incomplete = [c for c in self.protected if self.X[c].isna().any()]
            if incomplete:
                raise ValueError(
                    "protected columns must be complete (they are the MAR "
                    f"conditioners); these contain NaN: {incomplete}"
                )
        non_numeric = [c for c in self.X.columns if not pd.api.types.is_numeric_dtype(self.X[c])]
        if non_numeric:
            raise ValueError(f"X must be fully numeric; offending columns: {non_numeric}")
        if not set(pd.unique(self.y.dropna())) <= {0, 1}:
            raise ValueError("y must be binary 0/1")

    # ---- descriptive helpers -----------------------------------------
    @property
    def n_samples(self) -> int:
        return len(self.X)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def prevalence(self) -> float:
        return float(self.y.mean())

    def missingness_profile(self) -> pd.DataFrame:
        """Per-column natural missingness, for objective 1 of the proposal."""
        na = self.X.isna()
        prof = pd.DataFrame(
            {
                "column": self.X.columns,
                "n_missing": na.sum().to_numpy(),
                "frac_missing": na.mean().to_numpy(),
                "maskable": [c in set(self.maskable) for c in self.X.columns],
                "group": [self._group_of(c) for c in self.X.columns],
            }
        )
        # Does missingness in this column track the outcome? A first, purely
        # descriptive signal that the natural pattern is not MCAR.
        rates = []
        for col in self.X.columns:
            m = na[col]
            if m.any() and not m.all():
                rates.append(float(self.y[m].mean() - self.y[~m].mean()))
            else:
                rates.append(np.nan)
        prof["mortality_gap_when_missing"] = rates
        return prof.sort_values("frac_missing", ascending=False).reset_index(drop=True)

    def _group_of(self, col: str) -> str:
        for name, cols in self.groups.items():
            if col in cols:
                return name
        return "other"
