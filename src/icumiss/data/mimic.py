"""MIMIC-IV cohort extraction for 24-hour ICU mortality prediction.

MIMIC-IV is the project's primary dataset. It requires credentialed PhysioNet
access, and patient-level data never enters this repository - point `root` at
your own local copy of the uncompressed-or-gzipped CSV distribution:

    <root>/hosp/patients.csv.gz
    <root>/hosp/admissions.csv.gz
    <root>/hosp/labevents.csv.gz
    <root>/icu/icustays.csv.gz
    <root>/icu/chartevents.csv.gz
    <root>/icu/outputevents.csv.gz

Cohort definition
-----------------
* Adults (age >= 18) on their **first** ICU stay only, so rows are independent.
* ICU stay longer than the observation window, so every patient has a full
  window of data.
* Features summarise the first 24 hours after ICU admission.
* Target: death during the ICU stay.

Two leakage guards matter here and are enforced rather than assumed:

1. Patients who die *inside* the observation window are excluded. Predicting an
   event that has already happened inflates every metric.
2. Nothing recorded after `intime + window` is read, so no feature can carry
   information from the future. Discharge-time and length-of-stay fields are
   never used as features for the same reason.

`chartevents` and `labevents` are hundreds of millions of rows, so both are
streamed in chunks and filtered to the itemids and stays of interest. The
extracted matrix is cached as Parquet (gitignored); extraction runs once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .cohort import Cohort

OBSERVATION_HOURS = 24.0

PROTECTED = ["age", "sex_male", "elective_admission", "prior_icu_admission"]


@dataclass(frozen=True)
class ItemSpec:
    """How one clinical variable is pulled out of MIMIC-IV.

    itemids are for MIMIC-IV v2.x/v3.x metavision-era items. Verify them
    against your own `icu/d_items.csv.gz` and `hosp/d_labitems.csv.gz` before
    trusting a result set - `verify_itemids()` does this for you.
    """

    name: str
    source: str  # "chartevents" | "labevents" | "outputevents"
    itemids: tuple[int, ...]
    valid_range: tuple[float, float]
    group: str
    convert: str | None = None  # "f_to_c" for Fahrenheit temperature items


CHART_SPECS: tuple[ItemSpec, ...] = (
    ItemSpec("heart_rate", "chartevents", (220045,), (20, 250), "vitals"),
    ItemSpec("sbp", "chartevents", (220179, 220050), (30, 250), "vitals"),
    ItemSpec("dbp", "chartevents", (220180, 220051), (10, 180), "vitals"),
    ItemSpec("map_mmhg", "chartevents", (220181, 220052, 225312), (20, 200), "vitals"),
    ItemSpec("resp_rate", "chartevents", (220210, 224690), (3, 70), "vitals"),
    ItemSpec("temperature", "chartevents", (223762,), (32.0, 42.5), "vitals"),
    ItemSpec("temperature_f", "chartevents", (223761,), (89.6, 108.5), "vitals", "f_to_c"),
    ItemSpec("spo2", "chartevents", (220277,), (50, 100), "vitals"),
    ItemSpec("gcs_eye", "chartevents", (220739,), (1, 4), "vitals"),
    ItemSpec("gcs_verbal", "chartevents", (223900,), (1, 5), "vitals"),
    ItemSpec("gcs_motor", "chartevents", (223901,), (1, 6), "vitals"),
)

LAB_SPECS: tuple[ItemSpec, ...] = (
    ItemSpec("lactate", "labevents", (50813,), (0.1, 30), "labs"),
    ItemSpec("creatinine", "labevents", (50912,), (0.1, 20), "labs"),
    ItemSpec("bun", "labevents", (51006,), (1, 200), "labs"),
    ItemSpec("sodium", "labevents", (50983, 50824), (100, 180), "labs"),
    ItemSpec("potassium", "labevents", (50971, 50822), (1.5, 9.5), "labs"),
    ItemSpec("bicarbonate", "labevents", (50882, 50803), (5, 50), "labs"),
    ItemSpec("wbc", "labevents", (51301, 51300), (0.1, 100), "labs"),
    ItemSpec("hemoglobin", "labevents", (51222, 50811), (3, 22), "labs"),
    ItemSpec("platelets", "labevents", (51265,), (5, 1200), "labs"),
    ItemSpec("glucose", "labevents", (50931, 50809), (20, 1000), "labs"),
    ItemSpec("albumin", "labevents", (50862,), (0.5, 6.5), "labs"),
    ItemSpec("bilirubin", "labevents", (50885,), (0.05, 60), "labs"),
    ItemSpec("inr", "labevents", (51237,), (0.5, 15), "labs"),
    ItemSpec("ph", "labevents", (50820,), (6.6, 7.8), "labs"),
)

# Urine output itemids (outputevents), summed over the window.
URINE_ITEMIDS: tuple[int, ...] = (
    226559,  # Foley
    226560,  # Void
    226561,  # Condom cath
    226584,  # Ileoconduit
    226563,  # Suprapubic
    226564,  # R nephrostomy
    226565,  # L nephrostomy
    226567,  # Straight cath
    226557,  # R ureteral stent
    226558,  # L ureteral stent
    227488,  # GU irrigant volume in (subtracted)
    227489,  # GU irrigant/urine volume out
)

# Derived columns assembled after extraction.
DERIVED = ["gcs", "urine_output_24h"]

CHUNK_ROWS = 2_000_000


# ----------------------------------------------------------------------
# public entry points
# ----------------------------------------------------------------------
def load(
    root: str | Path,
    cache_path: str | Path | None = None,
    observation_hours: float = OBSERVATION_HOURS,
    aggregation: str = "mean",
    target: str = "icu_mortality",
    min_feature_completeness: float = 0.30,
    rebuild: bool = False,
) -> Cohort:
    """Load the MIMIC-IV cohort, building and caching it on first use.

    Parameters
    ----------
    root
        Directory containing the `hosp/` and `icu/` MIMIC-IV modules.
    cache_path
        Where to cache the extracted matrix. Defaults to
        `data/processed/mimic_<window>h_<aggregation>_<target>.parquet`.
    aggregation
        How repeated measurements inside the window collapse to one value per
        variable: "mean", "last", "min", "max" or "first".
    target
        "icu_mortality" (death during the ICU stay) or "hospital_mortality".
    min_feature_completeness
        Drop variables observed in fewer than this fraction of stays. Columns
        that are almost never recorded add noise to every imputation metric
        without carrying usable signal.
    rebuild
        Ignore any cache and re-extract.
    """
    root = Path(root).expanduser()
    if cache_path is None:
        cache_path = (
            Path("data/processed")
            / f"mimic_{int(observation_hours)}h_{aggregation}_{target}.parquet"
        )
    cache_path = Path(cache_path)

    if cache_path.exists() and not rebuild:
        frame = pd.read_parquet(cache_path)
        meta_file = cache_path.with_suffix(".meta.json")
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        return _frame_to_cohort(frame, meta, min_feature_completeness)

    verify_layout(root)
    frame, meta = _build(root, observation_hours, aggregation, target)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False)
    cache_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return _frame_to_cohort(frame, meta, min_feature_completeness)


def verify_layout(root: str | Path) -> None:
    """Fail early, and with instructions, if the distribution is not present."""
    root = Path(root).expanduser()
    required = [
        "hosp/patients",
        "hosp/admissions",
        "hosp/labevents",
        "icu/icustays",
        "icu/chartevents",
        "icu/outputevents",
    ]
    missing = [stem for stem in required if _resolve(root, stem) is None]
    if missing:
        raise FileNotFoundError(
            f"MIMIC-IV files not found under {root}.\n"
            f"Missing: {', '.join(missing)} (.csv or .csv.gz)\n\n"
            "MIMIC-IV needs credentialed PhysioNet access:\n"
            "  1. complete the CITI 'Data or Specimens Only Research' training\n"
            "  2. sign the PhysioNet credentialed data use agreement\n"
            "  3. download MIMIC-IV from https://physionet.org/content/mimiciv/\n\n"
            "Until then run the pipeline on dataset.name='synthetic', which "
            "exercises every stage end to end."
        )


def verify_itemids(root: str | Path) -> pd.DataFrame:
    """Check every configured itemid against the MIMIC-IV dictionaries.

    itemids differ between MIMIC versions. Run this once per environment and
    keep the output with the result set.
    """
    root = Path(root).expanduser()
    rows = []

    d_items_path = _resolve(root, "icu/d_items")
    if d_items_path is not None:
        d_items = pd.read_csv(d_items_path, usecols=["itemid", "label", "unitname"])
        known = d_items.set_index("itemid")
        for spec in CHART_SPECS:
            for iid in spec.itemids:
                hit = known.loc[iid] if iid in known.index else None
                rows.append(
                    {
                        "variable": spec.name,
                        "table": "chartevents",
                        "itemid": iid,
                        "found": hit is not None,
                        "label": None if hit is None else hit["label"],
                        "unit": None if hit is None else hit["unitname"],
                    }
                )
        for iid in URINE_ITEMIDS:
            hit = known.loc[iid] if iid in known.index else None
            rows.append(
                {
                    "variable": "urine_output_24h",
                    "table": "outputevents",
                    "itemid": iid,
                    "found": hit is not None,
                    "label": None if hit is None else hit["label"],
                    "unit": None if hit is None else hit["unitname"],
                }
            )

    d_lab_path = _resolve(root, "hosp/d_labitems")
    if d_lab_path is not None:
        d_lab = pd.read_csv(d_lab_path, usecols=["itemid", "label", "fluid"])
        known_lab = d_lab.set_index("itemid")
        for spec in LAB_SPECS:
            for iid in spec.itemids:
                hit = known_lab.loc[iid] if iid in known_lab.index else None
                rows.append(
                    {
                        "variable": spec.name,
                        "table": "labevents",
                        "itemid": iid,
                        "found": hit is not None,
                        "label": None if hit is None else hit["label"],
                        "unit": None if hit is None else hit["fluid"],
                    }
                )

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# extraction
# ----------------------------------------------------------------------
def _build(
    root: Path, window_hours: float, aggregation: str, target: str
) -> tuple[pd.DataFrame, dict]:
    if aggregation not in ("mean", "last", "first", "min", "max"):
        raise ValueError(f"unsupported aggregation {aggregation!r}")
    if target not in ("icu_mortality", "hospital_mortality"):
        raise ValueError(f"unsupported target {target!r}")

    stays, meta = _base_cohort(root, window_hours, target)
    window = stays.set_index("stay_id")[["intime", "window_end"]]

    chart = _stream_chartevents(root, window, aggregation)
    labs = _stream_labevents(root, stays, aggregation)
    urine = _stream_urine(root, window)

    frame = stays[["stay_id", "subject_id", "hadm_id", *PROTECTED, "target"]].copy()
    frame = frame.merge(chart, on="stay_id", how="left")
    frame = frame.merge(labs, on="stay_id", how="left")
    frame = frame.merge(urine, on="stay_id", how="left")

    frame = _derive(frame)
    meta["n_rows"] = int(len(frame))
    meta["prevalence"] = float(frame["target"].mean())
    meta["aggregation"] = aggregation
    meta["observation_hours"] = window_hours
    meta["target"] = target
    return frame, meta


def _base_cohort(root: Path, window_hours: float, target: str) -> tuple[pd.DataFrame, dict]:
    """First adult ICU stays that survive the full observation window."""
    patients = pd.read_csv(
        _require(root, "hosp/patients"),
        usecols=["subject_id", "gender", "anchor_age", "dod"],
        parse_dates=["dod"],
    )
    admissions = pd.read_csv(
        _require(root, "hosp/admissions"),
        usecols=[
            "subject_id",
            "hadm_id",
            "admittime",
            "deathtime",
            "admission_type",
            "hospital_expire_flag",
        ],
        parse_dates=["admittime", "deathtime"],
    )
    icustays = pd.read_csv(
        _require(root, "icu/icustays"),
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los"],
        parse_dates=["intime", "outtime"],
    )

    counts = {"icu_stays": len(icustays)}

    stays = icustays.merge(patients, on="subject_id", how="inner")
    stays = stays.merge(admissions, on=["subject_id", "hadm_id"], how="inner")

    # Was there an earlier ICU stay for this patient? Used both to restrict to
    # first stays and as the `prior_icu_admission` feature's definition.
    stays = stays.sort_values(["subject_id", "intime"])
    stays["stay_number"] = stays.groupby("subject_id").cumcount() + 1

    stays = stays[stays["anchor_age"] >= 18]
    counts["after_adult_filter"] = len(stays)

    stays = stays[stays["stay_number"] == 1]
    counts["after_first_stay_filter"] = len(stays)

    stays["window_end"] = stays["intime"] + pd.to_timedelta(window_hours, unit="h")

    # Leakage guard 1: the stay must extend past the observation window.
    stays = stays[stays["outtime"] > stays["window_end"]]
    counts["after_los_filter"] = len(stays)

    # Leakage guard 2: drop deaths inside the observation window.
    died_in_window = stays["deathtime"].notna() & (stays["deathtime"] <= stays["window_end"])
    stays = stays[~died_in_window]
    counts["after_death_in_window_filter"] = len(stays)

    if target == "icu_mortality":
        # Death recorded at or before ICU discharge.
        outcome = stays["deathtime"].notna() & (
            stays["deathtime"] <= stays["outtime"] + pd.Timedelta(hours=6)
        )
    else:
        outcome = stays["hospital_expire_flag"].astype(bool)
    stays["target"] = outcome.astype(int)

    stays["age"] = stays["anchor_age"].astype(float)
    stays["sex_male"] = (stays["gender"].astype(str).str.upper() == "M").astype(float)
    stays["elective_admission"] = (
        stays["admission_type"].astype(str).str.upper().str.contains("ELECTIVE|SURGICAL SAME DAY")
    ).astype(float)
    # First stay by construction, so this is a prior *hospital* admission flag.
    stays["prior_icu_admission"] = 0.0

    stays = stays.reset_index(drop=True)
    meta = {"source": "mimic-iv", "root": str(root), "cohort_counts": counts}
    return stays, meta


def _stream_chartevents(root: Path, window: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    specs = {iid: spec for spec in CHART_SPECS for iid in spec.itemids}
    path = _require(root, "icu/chartevents")
    collected: list[pd.DataFrame] = []

    reader = pd.read_csv(
        path,
        usecols=["stay_id", "charttime", "itemid", "valuenum"],
        parse_dates=["charttime"],
        chunksize=CHUNK_ROWS,
    )
    for chunk in reader:
        chunk = chunk[chunk["itemid"].isin(specs.keys())]
        if chunk.empty:
            continue
        chunk = chunk[chunk["valuenum"].notna()]
        chunk = chunk.join(window, on="stay_id", how="inner")
        chunk = chunk[
            (chunk["charttime"] >= chunk["intime"]) & (chunk["charttime"] <= chunk["window_end"])
        ]
        if not chunk.empty:
            collected.append(chunk[["stay_id", "charttime", "itemid", "valuenum"]])

    if not collected:
        return pd.DataFrame({"stay_id": []})
    events = pd.concat(collected, ignore_index=True)
    events["variable"] = events["itemid"].map({k: v.name for k, v in specs.items()})
    events = _apply_ranges(events, {s.name: s for s in CHART_SPECS})
    events = _convert_units(events, {s.name: s for s in CHART_SPECS})
    return _aggregate(events, aggregation)


def _stream_labevents(root: Path, stays: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    """labevents is keyed on hadm_id, so the window is applied via the stay."""
    specs = {iid: spec for spec in LAB_SPECS for iid in spec.itemids}
    path = _require(root, "hosp/labevents")
    window = stays.set_index("hadm_id")[["stay_id", "intime", "window_end"]]
    collected: list[pd.DataFrame] = []

    reader = pd.read_csv(
        path,
        usecols=["hadm_id", "charttime", "itemid", "valuenum"],
        parse_dates=["charttime"],
        chunksize=CHUNK_ROWS,
    )
    for chunk in reader:
        chunk = chunk[chunk["itemid"].isin(specs.keys())]
        if chunk.empty:
            continue
        chunk = chunk[chunk["valuenum"].notna() & chunk["hadm_id"].notna()]
        chunk = chunk.join(window, on="hadm_id", how="inner")
        chunk = chunk[
            (chunk["charttime"] >= chunk["intime"]) & (chunk["charttime"] <= chunk["window_end"])
        ]
        if not chunk.empty:
            collected.append(chunk[["stay_id", "charttime", "itemid", "valuenum"]])

    if not collected:
        return pd.DataFrame({"stay_id": []})
    events = pd.concat(collected, ignore_index=True)
    events["variable"] = events["itemid"].map({k: v.name for k, v in specs.items()})
    events = _apply_ranges(events, {s.name: s for s in LAB_SPECS})
    return _aggregate(events, aggregation)


def _stream_urine(root: Path, window: pd.DataFrame) -> pd.DataFrame:
    """Total urine output over the window (a sum, not an average)."""
    path = _require(root, "icu/outputevents")
    collected: list[pd.DataFrame] = []
    reader = pd.read_csv(
        path,
        usecols=["stay_id", "charttime", "itemid", "value"],
        parse_dates=["charttime"],
        chunksize=CHUNK_ROWS,
    )
    for chunk in reader:
        chunk = chunk[chunk["itemid"].isin(URINE_ITEMIDS)]
        if chunk.empty:
            continue
        chunk = chunk[chunk["value"].notna() & (chunk["value"] >= 0)]
        chunk = chunk.join(window, on="stay_id", how="inner")
        chunk = chunk[
            (chunk["charttime"] >= chunk["intime"]) & (chunk["charttime"] <= chunk["window_end"])
        ]
        if not chunk.empty:
            # GU irrigant volume in is fluid instilled, not urine produced.
            chunk = chunk.assign(
                signed=np.where(chunk["itemid"] == 227488, -chunk["value"], chunk["value"])
            )
            collected.append(chunk[["stay_id", "signed"]])

    if not collected:
        return pd.DataFrame({"stay_id": [], "urine_output_24h": []})
    events = pd.concat(collected, ignore_index=True)
    total = events.groupby("stay_id")["signed"].sum().clip(lower=0, upper=20000)
    return total.rename("urine_output_24h").reset_index()


def _apply_ranges(events: pd.DataFrame, specs: dict[str, ItemSpec]) -> pd.DataFrame:
    """Drop physiologically impossible values (charting and unit errors)."""
    keep = np.ones(len(events), dtype=bool)
    for name, spec in specs.items():
        sel = (events["variable"] == name).to_numpy()
        if not sel.any():
            continue
        lo, hi = spec.valid_range
        vals = events["valuenum"].to_numpy()
        keep &= ~(sel & ((vals < lo) | (vals > hi)))
    return events[keep]


def _convert_units(events: pd.DataFrame, specs: dict[str, ItemSpec]) -> pd.DataFrame:
    events = events.copy()
    for name, spec in specs.items():
        if spec.convert == "f_to_c":
            sel = events["variable"] == name
            events.loc[sel, "valuenum"] = (events.loc[sel, "valuenum"] - 32.0) * 5.0 / 9.0
            # Fold the converted Fahrenheit readings into the Celsius column.
            events.loc[sel, "variable"] = name.replace("_f", "")
    return events


def _aggregate(events: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    if aggregation in ("last", "first"):
        events = events.sort_values("charttime")
        wide = (
            events.groupby(["stay_id", "variable"])["valuenum"]
            .agg("last" if aggregation == "last" else "first")
            .unstack("variable")
        )
    else:
        wide = (
            events.groupby(["stay_id", "variable"])["valuenum"]
            .agg(aggregation)
            .unstack("variable")
        )
    return wide.reset_index()


def _derive(frame: pd.DataFrame) -> pd.DataFrame:
    """Assemble GCS from its components and tidy up the feature set."""
    frame = frame.copy()
    parts = ["gcs_eye", "gcs_verbal", "gcs_motor"]
    if all(p in frame.columns for p in parts):
        # A partially charted GCS is not a valid total, so require all three.
        complete = frame[parts].notna().all(axis=1)
        frame["gcs"] = np.where(complete, frame[parts].sum(axis=1), np.nan)
        frame = frame.drop(columns=parts)
    if "urine_output_24h" not in frame.columns:
        frame["urine_output_24h"] = np.nan
    return frame


def _frame_to_cohort(
    frame: pd.DataFrame, meta: dict, min_feature_completeness: float
) -> Cohort:
    y = frame["target"].astype(int)
    drop = ["stay_id", "subject_id", "hadm_id", "target"]
    X = frame.drop(columns=[c for c in drop if c in frame.columns])

    # Protected columns must be complete; fill the rare gap and record it.
    filled = {}
    for col in PROTECTED:
        if col not in X.columns:
            X[col] = 0.0
        n_na = int(X[col].isna().sum())
        if n_na:
            X[col] = X[col].fillna(X[col].median())
            filled[col] = n_na

    completeness = X.notna().mean()
    sparse = [
        c
        for c in X.columns
        if c not in PROTECTED and completeness[c] < min_feature_completeness
    ]
    X = X.drop(columns=sparse)

    maskable = [c for c in X.columns if c not in PROTECTED]
    vitals = [s.name for s in CHART_SPECS if s.name in maskable] + [
        c for c in ("gcs", "urine_output_24h") if c in maskable
    ]
    labs = [s.name for s in LAB_SPECS if s.name in maskable]

    X = X[PROTECTED + maskable].astype(float)
    meta = {
        **meta,
        "dropped_sparse_features": sparse,
        "protected_filled": filled,
        "feature_completeness": completeness.round(4).to_dict(),
    }
    return Cohort(
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True).rename("mortality"),
        maskable=maskable,
        protected=PROTECTED,
        groups={"demographics": PROTECTED, "vitals": vitals, "labs": labs},
        meta=meta,
    )


def _resolve(root: Path, stem: str) -> Path | None:
    for suffix in (".csv.gz", ".csv", ".parquet"):
        candidate = root / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _require(root: Path, stem: str) -> Path:
    path = _resolve(root, stem)
    if path is None:
        raise FileNotFoundError(f"{stem}.csv[.gz] not found under {root}")
    return path
