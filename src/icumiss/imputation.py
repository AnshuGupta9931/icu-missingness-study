"""The four imputation strategies, wrapped behind one leakage-safe interface.

Every imputer is **fitted on the training rows only** and then applied
unchanged to validation and test. This is the single most important property in
this module: fitting an imputer on the pooled data would let test-set
distributions leak into the transform and would quietly inflate every
downstream metric.

The outcome is never passed to an imputer. At prediction time mortality is
unknown, so using it to fill features would not be reproducible in deployment.

=============  ====================================================================
method         implementation
=============  ====================================================================
``median``     Per-column training median. Transparent baseline; ignores all
               between-feature structure, so it is the reference every other
               method has to beat.
``knn``        `KNNImputer` on standardised features. Standardisation matters:
               on raw units the neighbour distance would be dominated by
               platelets (~200) and glucose (~140) and would effectively ignore
               pH (~7.4). Values are returned in original units.
``mice``       `IterativeImputer` with Bayesian ridge regressions - chained
               equations, cycling through columns. Single imputation rather
               than pooled multiple imputation, because the downstream stage
               needs one dataset per condition; see README for the implication.
``missforest`` `IterativeImputer` with random-forest regressions, i.e. the
               MissForest algorithm. Captures nonlinearity and interactions the
               ridge version cannot, at a much higher compute cost.
=============  ====================================================================
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (required import)
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler

def _writable(frame: pd.DataFrame) -> np.ndarray:
    """A float array scikit-learn is free to write into.

    pandas hands out read-only views under copy-on-write, and scikit-learn's
    `IterativeImputer` assigns into the array it is given when
    `keep_empty_features` is set. Without an explicit copy that raises
    "assignment destination is read-only".
    """
    return np.array(frame.to_numpy(dtype=float), copy=True, order="C")


DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "median": {},
    "knn": {"n_neighbors": 5, "weights": "distance"},
    "mice": {"max_iter": 10, "tol": 1e-3},
    "missforest": {"max_iter": 5, "n_estimators": 50, "max_depth": None, "min_samples_leaf": 2},
}


@dataclass
class FittedImputer:
    """An imputer fitted on training data, ready to transform any split."""

    name: str
    columns: list[str]
    _transform_fn: Any
    fit_seconds: float = 0.0
    train_min: np.ndarray | None = None
    train_max: np.ndarray | None = None
    clip_to_train_range: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Impute `X`, returning a complete frame in the original units."""
        if list(X.columns) != self.columns:
            raise ValueError(
                "column mismatch between fit and transform: "
                f"fitted on {len(self.columns)} columns, got {X.shape[1]}"
            )
        original = _writable(X[self.columns])
        missing = np.isnan(original)
        values = self._transform_fn(original.copy())

        if self.clip_to_train_range and self.train_min is not None:
            # Keeps model-based imputers from producing impossible values
            # (negative platelet counts, a pH of 4). Applied identically to
            # every method so the comparison between them stays fair.
            values = np.clip(values, self.train_min, self.train_max)

        # Fill the gaps and nothing else. Recorded measurements pass through
        # untouched: clipping must not revise a real out-of-training-range
        # observation, and the scale/inverse-scale round trip inside the KNN and
        # MICE wrappers must not leave float drift on values that were never
        # missing in the first place.
        values = np.where(missing, values, original)

        out = pd.DataFrame(values, index=X.index, columns=self.columns)
        if out.isna().to_numpy().any():
            # Only reachable if a column was entirely missing in training, in
            # which case there is nothing to learn from and the training median
            # is unavailable too.
            out = out.fillna(pd.Series(self.train_min, index=self.columns))
        return out


def fit_imputer(
    name: str,
    X_train: pd.DataFrame,
    seed: int = 0,
    params: dict[str, Any] | None = None,
    clip_to_train_range: bool = True,
    n_jobs: int = 1,
) -> FittedImputer:
    """Fit one imputation strategy on the training matrix.

    Parameters
    ----------
    name
        "median", "knn", "mice" or "missforest".
    X_train
        Training features, already masked. The only data the imputer may see.
    seed
        Passed to every stochastic component.
    params
        Overrides merged over `DEFAULT_PARAMS[name]`.
    clip_to_train_range
        Constrain imputed values to the per-column training range.
    n_jobs
        Threads for the random forests inside MissForest.
    """
    name = name.lower()
    if name not in DEFAULT_PARAMS:
        raise ValueError(f"unknown imputer {name!r}; expected one of {sorted(DEFAULT_PARAMS)}")

    columns = list(X_train.columns)
    merged = {**DEFAULT_PARAMS[name], **(params or {})}
    values = _writable(X_train)

    with warnings.catch_warnings():
        # IterativeImputer warns when it stops on max_iter rather than on tol.
        # That is an expected outcome at high missingness, not a failure, and it
        # is recorded in the run log instead of being printed 252 times.
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        start = time.perf_counter()
        transform_fn = _BUILDERS[name](values, merged, seed, n_jobs)
        fit_seconds = time.perf_counter() - start

    train_min = np.nanmin(values, axis=0)
    train_max = np.nanmax(values, axis=0)
    # An all-NaN training column yields NaN bounds; fall back to a neutral 0.
    train_min = np.where(np.isfinite(train_min), train_min, 0.0)
    train_max = np.where(np.isfinite(train_max), train_max, 0.0)

    return FittedImputer(
        name=name,
        columns=columns,
        _transform_fn=transform_fn,
        fit_seconds=fit_seconds,
        train_min=train_min,
        train_max=train_max,
        clip_to_train_range=clip_to_train_range,
        params=merged,
    )


# ----------------------------------------------------------------------
# builders: each returns a callable ndarray -> ndarray
# ----------------------------------------------------------------------
def _build_median(values: np.ndarray, params: dict, seed: int, n_jobs: int):
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    imp.fit(values)
    return lambda arr: imp.transform(arr)


def _build_knn(values: np.ndarray, params: dict, seed: int, n_jobs: int):
    scaler = StandardScaler().fit(values)  # NaN-aware: ignores missing entries
    imp = KNNImputer(keep_empty_features=True, **params)
    imp.fit(scaler.transform(values))

    def transform(arr: np.ndarray) -> np.ndarray:
        return scaler.inverse_transform(imp.transform(scaler.transform(arr)))

    return transform


def _build_mice(values: np.ndarray, params: dict, seed: int, n_jobs: int):
    # Standardised so the Bayesian ridge prior is comparable across features
    # that differ by orders of magnitude in scale.
    scaler = StandardScaler().fit(values)
    imp = IterativeImputer(
        estimator=BayesianRidge(),
        random_state=seed,
        initial_strategy="median",
        imputation_order="ascending",
        sample_posterior=False,
        keep_empty_features=True,
        **params,
    )
    imp.fit(scaler.transform(values))

    def transform(arr: np.ndarray) -> np.ndarray:
        return scaler.inverse_transform(imp.transform(scaler.transform(arr)))

    return transform


def _build_missforest(values: np.ndarray, params: dict, seed: int, n_jobs: int):
    # Config keys are split by where they belong: the chained-equation loop
    # versus the forest fitted at each step. Without this, setting `tol` or
    # `max_iter` in a config would be forwarded to RandomForestRegressor and
    # fail with an unhelpful error about an unexpected keyword.
    params = dict(params)
    loop_keys = ("max_iter", "tol", "imputation_order", "initial_strategy")
    loop = {k: params.pop(k) for k in loop_keys if k in params}

    forest = RandomForestRegressor(
        random_state=seed,
        n_jobs=n_jobs,
        bootstrap=True,
        **params,
    )
    # Trees are scale-invariant, so no standardisation is needed here.
    imp = IterativeImputer(
        estimator=forest,
        random_state=seed,
        sample_posterior=False,
        keep_empty_features=True,
        **{
            "max_iter": 5,
            "tol": 1e-3,
            "initial_strategy": "median",
            "imputation_order": "ascending",
            **loop,
        },
    )
    imp.fit(values)
    return lambda arr: imp.transform(arr)


_BUILDERS = {
    "median": _build_median,
    "knn": _build_knn,
    "mice": _build_mice,
    "missforest": _build_missforest,
}


def impute_splits(
    name: str,
    frames: dict[str, pd.DataFrame],
    seed: int = 0,
    params: dict[str, Any] | None = None,
    clip_to_train_range: bool = True,
    n_jobs: int = 1,
) -> tuple[dict[str, pd.DataFrame], FittedImputer]:
    """Fit on `frames["train"]`, then transform every split.

    Returns the imputed frames and the fitted imputer, so the caller can record
    its parameters and fit time.
    """
    if "train" not in frames:
        raise ValueError("frames must contain a 'train' entry")
    fitted = fit_imputer(
        name,
        frames["train"],
        seed=seed,
        params=params,
        clip_to_train_range=clip_to_train_range,
        n_jobs=n_jobs,
    )
    return {split: fitted.transform(frame) for split, frame in frames.items()}, fitted
