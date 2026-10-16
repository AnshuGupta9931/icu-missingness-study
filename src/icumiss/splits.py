"""Leakage-safe cohort splitting.

The split happens *before* masking and *before* any imputer is fitted. That
ordering is what keeps the experiment honest: every statistic an imputer or a
scaler uses comes from the training rows only, so no information crosses from
the evaluation data into the fitted transform.

The validation set has two jobs: early stopping for the gradient-boosting
classifiers, and selecting the decision threshold used for F1 and precision.
Neither may touch the test set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data.cohort import Cohort


@dataclass
class Split:
    """Row indices for the three partitions, plus convenience accessors."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    seed: int

    def __post_init__(self) -> None:
        overlap = (
            set(self.train) & set(self.val)
            or set(self.train) & set(self.test)
            or set(self.val) & set(self.test)
        )
        if overlap:
            raise ValueError(f"partitions overlap on {len(overlap)} rows")

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.train), "val": len(self.val), "test": len(self.test)}

    def frames(self, X: pd.DataFrame) -> dict[str, pd.DataFrame]:
        return {
            "train": X.iloc[self.train],
            "val": X.iloc[self.val],
            "test": X.iloc[self.test],
        }

    def targets(self, y: pd.Series) -> dict[str, pd.Series]:
        return {
            "train": y.iloc[self.train],
            "val": y.iloc[self.val],
            "test": y.iloc[self.test],
        }


def make_split(
    cohort: Cohort,
    seed: int,
    test_size: float = 0.20,
    val_size: float = 0.15,
    stratify: bool = True,
) -> Split:
    """Stratified three-way split.

    `val_size` is a fraction of the whole cohort, not of the remainder, so the
    partition proportions read the way they are written in the config.
    """
    n = cohort.n_samples
    idx = np.arange(n)
    y = cohort.y.to_numpy()

    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must leave training rows")

    strat = y if stratify else None
    train_val_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=strat, shuffle=True
    )

    # Rescale so val_size stays a fraction of the full cohort.
    val_fraction_of_remainder = val_size / (1.0 - test_size)
    strat_tv = y[train_val_idx] if stratify else None
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction_of_remainder,
        random_state=seed,
        stratify=strat_tv,
        shuffle=True,
    )

    split = Split(
        train=np.sort(train_idx), val=np.sort(val_idx), test=np.sort(test_idx), seed=seed
    )

    for name, part in (("train", split.train), ("val", split.val), ("test", split.test)):
        if y[part].sum() < 2:
            raise ValueError(
                f"{name} split contains fewer than 2 positive cases "
                f"({int(y[part].sum())}); the cohort is too small or too "
                "imbalanced for this split configuration"
            )
    return split
