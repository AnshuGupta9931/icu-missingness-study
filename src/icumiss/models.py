"""The four classifiers, behind one fit/predict interface.

All four are trained on the *imputed* matrices, so every model sees complete
data and the comparison isolates the effect of the imputation rather than each
library's own missing-value handling. (XGBoost and LightGBM can route NaN
internally; letting them do so here would confound the very thing the
experiment measures.)

Gradient boosters stop early on the validation split, which keeps model
capacity from being a hidden free parameter that varies with missingness level.

No class weighting or resampling is applied. Log loss and the Brier score are
headline metrics in this study, and both reward honest probabilities;
reweighting improves F1 at the cost of calibration and would muddy that
reading. `model_params` in the config can switch it on for a sensitivity check.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "logreg": {"C": 1.0, "max_iter": 3000, "class_weight": None},
    "xgboost": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 1.0,
        "early_stopping_rounds": 40,
    },
    "lightgbm": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "early_stopping_rounds": 40,
    },
    "catboost": {
        "iterations": 500,
        "learning_rate": 0.05,
        "depth": 4,
        "l2_leaf_reg": 3.0,
        "early_stopping_rounds": 40,
    },
}


@dataclass
class FittedModel:
    """A trained classifier plus the provenance of its training run."""

    name: str
    model: Any
    columns: list[str]
    fit_seconds: float
    n_iterations: int | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predicted probability of mortality."""
        values = X[self.columns]
        proba = self.model.predict_proba(values)
        return np.asarray(proba)[:, 1]


def fit_classifier(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int = 0,
    params: dict[str, Any] | None = None,
    n_threads: int = 1,
) -> FittedModel:
    """Train one classifier, using the validation split for early stopping."""
    name = name.lower()
    if name not in DEFAULT_PARAMS:
        raise ValueError(f"unknown classifier {name!r}; expected {sorted(DEFAULT_PARAMS)}")

    columns = list(X_train.columns)
    merged = {**DEFAULT_PARAMS[name], **(params or {})}

    if X_train.isna().to_numpy().any():
        raise ValueError(
            "classifier received missing values; imputation must run first "
            "(this is the guard that keeps native NaN handling from "
            "confounding the imputation comparison)"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = time.perf_counter()
        model, n_iter = _FITTERS[name](
            X_train[columns], y_train, X_val[columns], y_val, seed, merged, n_threads
        )
        fit_seconds = time.perf_counter() - start

    return FittedModel(
        name=name,
        model=model,
        columns=columns,
        fit_seconds=fit_seconds,
        n_iterations=n_iter,
        params=merged,
    )


# ----------------------------------------------------------------------
# fitters
# ----------------------------------------------------------------------
def _fit_logreg(X_train, y_train, X_val, y_val, seed, params, n_threads):
    # Scaling is inside the pipeline, so it is fitted on training rows only.
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    random_state=seed, solver="lbfgs", n_jobs=n_threads, **params
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe, int(pipe.named_steps["clf"].n_iter_[0])


def _fit_xgboost(X_train, y_train, X_val, y_val, seed, params, n_threads):
    from xgboost import XGBClassifier

    params = dict(params)
    model = XGBClassifier(
        random_state=seed,
        n_jobs=n_threads,
        eval_metric="logloss",
        objective="binary:logistic",
        tree_method="hist",
        **params,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    best = getattr(model, "best_iteration", None)
    return model, None if best is None else int(best) + 1


def _fit_lightgbm(X_train, y_train, X_val, y_val, seed, params, n_threads):
    import lightgbm as lgb

    params = dict(params)
    rounds = params.pop("early_stopping_rounds", 40)
    model = lgb.LGBMClassifier(
        random_state=seed,
        n_jobs=n_threads,
        verbosity=-1,
        # Reproducibility is a stated requirement of this project, and
        # LightGBM's default histogram construction is not bit-stable across
        # thread counts; these two flags make it so.
        deterministic=True,
        force_row_wise=True,
        **params,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(rounds, verbose=False), lgb.log_evaluation(0)],
    )
    return model, int(model.best_iteration_ or params.get("n_estimators", 0))


def _fit_catboost(X_train, y_train, X_val, y_val, seed, params, n_threads):
    from catboost import CatBoostClassifier

    params = dict(params)
    model = CatBoostClassifier(
        random_seed=seed,
        thread_count=n_threads,
        verbose=0,
        allow_writing_files=False,
        loss_function="Logloss",
        **params,
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=0)
    return model, int(model.get_best_iteration() or 0) + 1


_FITTERS = {
    "logreg": _fit_logreg,
    "xgboost": _fit_xgboost,
    "lightgbm": _fit_lightgbm,
    "catboost": _fit_catboost,
}
