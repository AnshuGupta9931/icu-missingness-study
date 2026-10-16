"""The masking contract.

These tests guard the properties the whole study rests on: the mechanisms are
what they claim to be, the levels are exact, ground truth is only ever claimed
where it exists, and the same seed gives the same mask.
"""

from __future__ import annotations

import numpy as np
import pytest

from icumiss.data import synthetic
from icumiss.missingness import apply_missingness, describe_mask


@pytest.fixture(scope="module")
def cohort():
    return synthetic.generate(n_patients=1500, seed=0)


def mask_it(cohort, mechanism, level, seed=0, **kwargs):
    return apply_missingness(
        cohort.X,
        mechanism=mechanism,
        level=level,
        seed=seed,
        maskable=cohort.maskable,
        protected=cohort.protected,
        **kwargs,
    )


@pytest.mark.parametrize("mechanism", ["mcar", "mar", "mnar"])
@pytest.mark.parametrize("level", [0.1, 0.3, 0.5, 0.7])
def test_realised_level_matches_request(cohort, mechanism, level):
    result = mask_it(cohort, mechanism, level)
    assert result.realised_level == pytest.approx(level, abs=0.01)


@pytest.mark.parametrize("mechanism", ["mcar", "mar", "mnar"])
def test_only_observed_cells_are_masked(cohort, mechanism):
    """Masking an already-missing cell would invent ground truth that never existed."""
    result = mask_it(cohort, mechanism, 0.5)
    originally_missing = cohort.X.isna().to_numpy()
    assert not (result.mask.to_numpy() & originally_missing).any()


@pytest.mark.parametrize("mechanism", ["mcar", "mar", "mnar"])
def test_protected_columns_are_never_masked(cohort, mechanism):
    result = mask_it(cohort, mechanism, 0.7)
    for col in cohort.protected:
        assert not result.mask[col].any()
        assert result.X_masked[col].notna().all()


@pytest.mark.parametrize("mechanism", ["mcar", "mar", "mnar"])
def test_truth_is_recoverable_exactly_at_masked_positions(cohort, mechanism):
    result = mask_it(cohort, mechanism, 0.4)
    mask = result.mask.to_numpy()
    original = cohort.X.to_numpy(dtype=float)
    truth = result.truth.to_numpy(dtype=float)

    assert np.allclose(truth[mask], original[mask], equal_nan=False)
    # And nowhere else: a non-masked position must carry no ground truth.
    assert np.isnan(truth[~mask]).all()
    # The masked cells really are absent from what the imputer will see.
    assert np.isnan(result.X_masked.to_numpy(dtype=float)[mask]).all()


@pytest.mark.parametrize("mechanism", ["mcar", "mar", "mnar"])
def test_same_seed_gives_same_mask(cohort, mechanism):
    a = mask_it(cohort, mechanism, 0.3, seed=7)
    b = mask_it(cohort, mechanism, 0.3, seed=7)
    assert a.mask.equals(b.mask)


@pytest.mark.parametrize("mechanism", ["mcar", "mar", "mnar"])
def test_different_seed_gives_different_mask(cohort, mechanism):
    a = mask_it(cohort, mechanism, 0.3, seed=1)
    b = mask_it(cohort, mechanism, 0.3, seed=2)
    assert not a.mask.equals(b.mask)
    # ...but the same amount masked, so the condition is unchanged.
    assert a.n_masked == b.n_masked


def test_mcar_is_independent_of_everything(cohort):
    diag = describe_mask(mask_it(cohort, "mcar", 0.3), cohort.X, cohort.protected)
    assert abs(diag["corr_with_hidden_value"]) < 0.08
    assert abs(diag["corr_with_conditioner"]) < 0.08


def test_mar_depends_on_observed_data_not_the_hidden_value(cohort):
    diag = describe_mask(mask_it(cohort, "mar", 0.3), cohort.X, cohort.protected)
    assert abs(diag["corr_with_conditioner"]) > 0.25
    assert abs(diag["corr_with_hidden_value"]) < 0.20


def test_mnar_depends_on_the_hidden_value(cohort):
    diag = describe_mask(mask_it(cohort, "mnar", 0.3), cohort.X, cohort.protected)
    assert abs(diag["corr_with_hidden_value"]) > 0.25


def test_mnar_direction_is_respected(cohort):
    """'high' must hide large values; 'low' must hide small ones."""
    high = mask_it(cohort, "mnar", 0.3, mnar_direction="high")
    low = mask_it(cohort, "mnar", 0.3, mnar_direction="low")

    col = "creatinine"
    values = cohort.X[col].to_numpy(dtype=float)
    observed = ~np.isnan(values)
    overall = values[observed].mean()

    assert high.truth[col].mean() > overall
    assert low.truth[col].mean() < overall


def test_zero_strength_collapses_mar_to_independence(cohort):
    """The strength parameter is the knob it claims to be."""
    diag = describe_mask(
        mask_it(cohort, "mar", 0.3, mar_strength=0.0), cohort.X, cohort.protected
    )
    assert abs(diag["corr_with_conditioner"]) < 0.08


def test_stronger_mnar_increases_dependence(cohort):
    weak = describe_mask(
        mask_it(cohort, "mnar", 0.3, mnar_strength=0.5), cohort.X, cohort.protected
    )
    strong = describe_mask(
        mask_it(cohort, "mnar", 0.3, mnar_strength=4.0), cohort.X, cohort.protected
    )
    assert strong["corr_with_hidden_value"] > weak["corr_with_hidden_value"]


def test_column_masking_is_independent_of_column_order(cohort):
    """Per-column RNG streams, so reordering columns must not change a mask."""
    shuffled = cohort.X[list(reversed(cohort.X.columns))]
    normal = apply_missingness(
        cohort.X, "mcar", 0.3, 0, cohort.maskable, cohort.protected
    )
    reordered = apply_missingness(
        shuffled, "mcar", 0.3, 0, cohort.maskable, cohort.protected
    )
    for col in cohort.maskable:
        assert normal.mask[col].equals(reordered.mask[col])


def test_min_observed_guard_leaves_a_column_usable(cohort):
    result = mask_it(cohort, "mcar", 0.95, min_observed_per_column=25)
    for col in cohort.maskable:
        assert result.X_masked[col].notna().sum() >= 25


@pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5])
def test_invalid_level_is_rejected(cohort, level):
    with pytest.raises(ValueError):
        mask_it(cohort, "mcar", level)


def test_unknown_mechanism_is_rejected(cohort):
    with pytest.raises(ValueError, match="unknown mechanism"):
        mask_it(cohort, "mnar_v2", 0.3)
