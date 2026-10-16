"""Leakage guards.

If any of these fail, every number the project reports is optimistic and the
conclusions are unsafe. They are the tests to run first after touching the
pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icumiss.config import Config
from icumiss.data import synthetic
from icumiss.imputation import fit_imputer, impute_splits
from icumiss.splits import make_split


@pytest.fixture(scope="module")
def cohort():
    return synthetic.generate(n_patients=1200, seed=0)


@pytest.fixture(scope="module")
def split(cohort):
    return make_split(cohort, seed=0, test_size=0.2, val_size=0.15)


def test_partitions_are_disjoint_and_complete(cohort, split):
    allocated = np.concatenate([split.train, split.val, split.test])
    assert len(np.unique(allocated)) == len(allocated)
    assert set(allocated) == set(range(cohort.n_samples))


def test_split_proportions_are_as_configured(cohort, split):
    n = cohort.n_samples
    assert len(split.test) / n == pytest.approx(0.20, abs=0.02)
    assert len(split.val) / n == pytest.approx(0.15, abs=0.02)


def test_split_is_stratified(cohort, split):
    overall = cohort.prevalence
    for part in (split.train, split.val, split.test):
        assert cohort.y.iloc[part].mean() == pytest.approx(overall, abs=0.03)


def test_split_is_reproducible(cohort):
    a = make_split(cohort, seed=3, test_size=0.2, val_size=0.15)
    b = make_split(cohort, seed=3, test_size=0.2, val_size=0.15)
    assert np.array_equal(a.test, b.test)

    c = make_split(cohort, seed=4, test_size=0.2, val_size=0.15)
    assert not np.array_equal(a.test, c.test)


@pytest.mark.parametrize("imputer", ["median", "knn", "mice", "missforest"])
def test_imputer_output_does_not_depend_on_unseen_data(cohort, split, imputer):
    """The decisive leakage test.

    An imputer fitted on train must produce identical values for a test row
    regardless of what else is in the test set. If the fitted transform depended
    on the evaluation data, perturbing the rest of the test set would move this
    row's imputed values - and test-set statistics would be leaking into it.
    """
    params = {"missforest": {"n_estimators": 10, "max_iter": 2}, "mice": {"max_iter": 3}}.get(imputer)
    frames = split.frames(cohort.X)
    fitted = fit_imputer(imputer, frames["train"], seed=0, params=params)

    test = frames["test"]
    first_three = test.iloc[:3]

    alone = fitted.transform(first_three)
    with_everyone = fitted.transform(test).iloc[:3]

    pd.testing.assert_frame_equal(alone, with_everyone, atol=1e-10)

    # And again with the rest of the test set corrupted beyond recognition.
    corrupted = test.copy()
    corrupted.iloc[3:] = corrupted.iloc[3:] * 1000.0 + 500.0
    with_corruption = fitted.transform(corrupted).iloc[:3]
    pd.testing.assert_frame_equal(alone, with_corruption, atol=1e-10)


@pytest.mark.parametrize("imputer", ["median", "knn", "mice", "missforest"])
def test_imputation_fills_everything(cohort, split, imputer):
    params = {"missforest": {"n_estimators": 10, "max_iter": 2}, "mice": {"max_iter": 3}}.get(imputer)
    frames = split.frames(cohort.X)
    imputed, _ = impute_splits(imputer, frames, seed=0, params=params)
    for name, frame in imputed.items():
        assert not frame.isna().to_numpy().any(), f"{imputer} left NaN in {name}"
        assert list(frame.columns) == list(frames[name].columns)
        assert len(frame) == len(frames[name])


@pytest.mark.parametrize("imputer", ["median", "knn", "mice", "missforest"])
def test_observed_values_are_preserved(cohort, split, imputer):
    """Imputation must fill gaps, not revise recorded measurements."""
    params = {"missforest": {"n_estimators": 10, "max_iter": 2}, "mice": {"max_iter": 3}}.get(imputer)
    frames = split.frames(cohort.X)
    imputed, _ = impute_splits(imputer, frames, seed=0, params=params)

    for name, original in frames.items():
        observed = original.notna()
        before = original.to_numpy(dtype=float)[observed.to_numpy()]
        after = imputed[name].to_numpy(dtype=float)[observed.to_numpy()]
        assert np.allclose(before, after, rtol=1e-6, atol=1e-6), f"{imputer} altered observed values in {name}"


def test_imputer_never_sees_the_outcome(cohort, split):
    """The feature matrix handed to an imputer must not contain the target."""
    frames = split.frames(cohort.X)
    assert cohort.y.name not in frames["train"].columns
    assert "mortality" not in frames["train"].columns

    # A column perfectly correlated with the outcome would be caught by this:
    for col in frames["train"].columns:
        values = frames["train"][col]
        target = cohort.y.iloc[split.train]
        both = pd.concat([values, target], axis=1).dropna()
        if len(both) > 50 and both.iloc[:, 0].nunique() > 1:
            corr = abs(both.corr().iloc[0, 1])
            assert corr < 0.95, f"{col} is suspiciously predictive of the outcome (r={corr:.3f})"


def test_imputed_values_stay_in_the_training_range(cohort, split):
    """The clip guard keeps model-based imputers physiologically plausible.

    The guard applies to filled cells only. A *recorded* test value outside the
    training range is a real measurement and must survive untouched, so the
    check is restricted to the positions that were actually missing.
    """
    frames = split.frames(cohort.X)
    fitted = fit_imputer("mice", frames["train"], seed=0, params={"max_iter": 3})
    test = frames["test"]
    imputed = fitted.transform(test)
    was_missing = test.isna()

    assert was_missing.to_numpy().any(), "nothing was missing, so this proves nothing"

    for i, col in enumerate(fitted.columns):
        filled = imputed.loc[was_missing[col], col]
        if filled.empty:
            continue
        assert filled.min() >= fitted.train_min[i] - 1e-9
        assert filled.max() <= fitted.train_max[i] + 1e-9

    # Nothing impossible gets through: counts stay non-negative.
    for col in ("platelets", "wbc", "lactate", "urine_output_24h"):
        if col in imputed.columns:
            assert imputed[col].min() >= 0.0


def test_classifier_rejects_missing_values(cohort, split):
    """A classifier must never silently route NaN - that would confound the study."""
    from icumiss.models import fit_classifier

    frames = split.frames(cohort.X)
    targets = split.targets(cohort.y)
    with pytest.raises(ValueError, match="missing values"):
        fit_classifier(
            "logreg", frames["train"], targets["train"], frames["val"], targets["val"]
        )


def test_tiny_split_is_rejected_rather_than_silently_degenerate():
    cohort = synthetic.generate(n_patients=40, mortality_rate=0.05, seed=0)
    with pytest.raises(ValueError, match="fewer than 2 positive"):
        make_split(cohort, seed=0, test_size=0.2, val_size=0.15)


def test_config_rejects_impossible_split():
    with pytest.raises(ValueError):
        Config.from_dict({"split": {"test_size": 0.6, "val_size": 0.35}})
