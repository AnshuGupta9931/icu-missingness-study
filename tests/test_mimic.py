"""MIMIC-IV extraction helpers.

The full extraction cannot be tested here - MIMIC-IV needs credentialed access
and must never enter the repository. What *can* be tested is every
transformation applied to the events once they are read, and those are exactly
the steps where a silent error would produce a plausible-looking but wrong
cohort: unit conversion, implausible-value filtering, aggregation, GCS
assembly, and the assembly of the final cohort object.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from icumiss.data import mimic
from icumiss.data.mimic import (
    CHART_SPECS,
    LAB_SPECS,
    PROTECTED,
    _aggregate,
    _apply_ranges,
    _convert_units,
    _derive,
    _frame_to_cohort,
)


def test_missing_distribution_fails_with_instructions(tmp_path):
    """A missing dataset must explain how to get it, not just raise."""
    with pytest.raises(FileNotFoundError) as excinfo:
        mimic.verify_layout(tmp_path)
    message = str(excinfo.value)
    assert "physionet.org" in message
    assert "credentialed" in message
    assert "synthetic" in message  # points at the fallback that does work


def test_implausible_values_are_dropped():
    """Charting and unit errors must not reach the model."""
    specs = {s.name: s for s in LAB_SPECS}
    events = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            "variable": ["ph", "ph", "sodium", "sodium"],
            # pH valid range is 6.6-7.8; sodium 100-180.
            "valuenum": [7.35, 73.5, 140.0, 1400.0],
            "charttime": pd.to_datetime(["2150-01-01"] * 4),
        }
    )
    kept = _apply_ranges(events, specs)
    assert list(kept["valuenum"]) == [7.35, 140.0]


def test_range_filter_leaves_other_variables_alone():
    """Filtering one variable must not take another's rows with it."""
    specs = {s.name: s for s in LAB_SPECS}
    events = pd.DataFrame(
        {
            "stay_id": [1, 2],
            "variable": ["lactate", "not_configured"],
            "valuenum": [2.0, 99999.0],
            "charttime": pd.to_datetime(["2150-01-01"] * 2),
        }
    )
    kept = _apply_ranges(events, specs)
    assert len(kept) == 2


def test_fahrenheit_is_converted_and_folded_into_the_celsius_column():
    """Both temperature itemids must end up as one variable in Celsius."""
    specs = {s.name: s for s in CHART_SPECS}
    events = pd.DataFrame(
        {
            "stay_id": [1, 2],
            "variable": ["temperature_f", "temperature"],
            "valuenum": [98.6, 37.0],
            "charttime": pd.to_datetime(["2150-01-01"] * 2),
        }
    )
    out = _convert_units(events, specs)
    assert set(out["variable"]) == {"temperature"}
    assert out["valuenum"].iloc[0] == pytest.approx(37.0, abs=1e-6)


@pytest.mark.parametrize(
    "how,expected", [("mean", 20.0), ("last", 30.0), ("first", 10.0), ("min", 10.0), ("max", 30.0)]
)
def test_aggregation_collapses_the_window_as_configured(how, expected):
    events = pd.DataFrame(
        {
            "stay_id": [1, 1, 1],
            "variable": ["heart_rate"] * 3,
            "valuenum": [10.0, 20.0, 30.0],
            "charttime": pd.to_datetime(["2150-01-01 01:00", "2150-01-01 05:00", "2150-01-01 09:00"]),
        }
    )
    wide = _aggregate(events, how)
    assert wide.loc[0, "heart_rate"] == pytest.approx(expected)


def test_last_uses_chart_time_not_row_order():
    """Rows arrive in chunk order, so 'last' must sort by time first."""
    events = pd.DataFrame(
        {
            "stay_id": [1, 1],
            "variable": ["heart_rate"] * 2,
            "valuenum": [99.0, 60.0],
            "charttime": pd.to_datetime(["2150-01-01 02:00", "2150-01-01 20:00"]),
        }
    )
    assert _aggregate(events, "last").loc[0, "heart_rate"] == pytest.approx(60.0)


def test_gcs_requires_all_three_components():
    """A partially charted GCS is not a valid total and must stay missing."""
    frame = pd.DataFrame(
        {
            "stay_id": [1, 2],
            "gcs_eye": [4.0, 4.0],
            "gcs_verbal": [5.0, np.nan],
            "gcs_motor": [6.0, 6.0],
        }
    )
    out = _derive(frame)
    assert out.loc[0, "gcs"] == pytest.approx(15.0)
    assert np.isnan(out.loc[1, "gcs"])
    assert not {"gcs_eye", "gcs_verbal", "gcs_motor"} & set(out.columns)


def _wide_frame(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "stay_id": np.arange(n),
            "subject_id": np.arange(n),
            "hadm_id": np.arange(n),
            "age": rng.uniform(40, 90, n),
            "sex_male": rng.integers(0, 2, n).astype(float),
            "elective_admission": rng.integers(0, 2, n).astype(float),
            "prior_icu_admission": np.zeros(n),
            "lactate": rng.uniform(0.5, 6, n),
            "creatinine": rng.uniform(0.4, 4, n),
            "target": rng.integers(0, 2, n),
        }
    )


def test_cohort_assembly_drops_identifiers_and_the_target():
    """A stay_id left in the matrix would be a free, meaningless feature."""
    cohort = _frame_to_cohort(_wide_frame(), meta={}, min_feature_completeness=0.3)
    for leaked in ("stay_id", "subject_id", "hadm_id", "target"):
        assert leaked not in cohort.X.columns
    assert set(cohort.protected) == set(PROTECTED)
    assert set(cohort.maskable) == {"lactate", "creatinine"}
    assert len(cohort.y) == len(cohort.X)


def test_almost_never_recorded_features_are_dropped():
    frame = _wide_frame()
    frame["exotic_assay"] = np.nan
    frame.loc[:4, "exotic_assay"] = 1.0  # recorded for 5 of 60 stays

    cohort = _frame_to_cohort(frame, meta={}, min_feature_completeness=0.30)
    assert "exotic_assay" not in cohort.X.columns
    assert "exotic_assay" in cohort.meta["dropped_sparse_features"]


def test_protected_columns_are_completed_and_the_fill_is_recorded():
    """MAR conditioning needs complete columns; any filling must be visible."""
    frame = _wide_frame()
    frame.loc[0, "age"] = np.nan

    cohort = _frame_to_cohort(frame, meta={}, min_feature_completeness=0.3)
    assert cohort.X["age"].notna().all()
    assert cohort.meta["protected_filled"]["age"] == 1


def test_protected_columns_survive_the_sparsity_filter():
    """Demographics are never dropped, however sparse the rest of the matrix."""
    frame = _wide_frame()
    frame["lactate"] = np.nan
    cohort = _frame_to_cohort(frame, meta={}, min_feature_completeness=0.3)
    for col in PROTECTED:
        assert col in cohort.X.columns


def test_itemid_specs_are_internally_consistent():
    """Guards against a typo in the extraction table itself."""
    for spec in (*CHART_SPECS, *LAB_SPECS):
        assert spec.itemids, f"{spec.name} has no itemids"
        assert len(set(spec.itemids)) == len(spec.itemids), f"{spec.name} repeats an itemid"
        low, high = spec.valid_range
        assert low < high, f"{spec.name} has an inverted valid range"

    seen: dict[int, str] = {}
    for spec in (*CHART_SPECS, *LAB_SPECS):
        for itemid in spec.itemids:
            assert itemid not in seen, (
                f"itemid {itemid} is claimed by both {seen[itemid]} and {spec.name}"
            )
            seen[itemid] = spec.name
