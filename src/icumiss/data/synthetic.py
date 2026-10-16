"""A synthetic ICU cohort with realistic structure and natural missingness.

Purpose. MIMIC-IV requires credentialed PhysioNet access and patient-level data
must not enter the repository, so the pipeline needs a dataset it can be
developed, tested and demonstrated against. This generator provides one. It is
*not* a stand-in for the real analysis: the headline results of the project come
from MIMIC-IV (see mimic.py).

Three properties are deliberately built in, because each one is needed for the
experiment to be able to discriminate between methods at all:

1. Outcome risk depends on the *clinical values themselves*, nonlinearly and
   with interactions (shock = high lactate together with low MAP, U-shaped risk
   in sodium and white-cell count). An imputation method that distorts those
   values therefore loses real predictive signal, and tree ensembles have
   something to find that logistic regression cannot.
2. Natural missingness is informative, not uniform: selective labs such as
   lactate, albumin and INR are ordered mainly when a patient looks unwell, so
   their missingness tracks severity. The profiling step detects this.
3. Features are correlated through a shared latent severity, so neighbour- and
   model-based imputers have borrowable information while a median baseline
   does not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils import sigmoid, solve_intercept
from .cohort import Cohort

PROTECTED = ["age", "sex_male", "elective_admission", "prior_icu_admission"]

VITALS = [
    "heart_rate",
    "sbp",
    "dbp",
    "map_mmhg",
    "resp_rate",
    "temperature",
    "spo2",
    "gcs",
    "urine_output_24h",
]

LABS = [
    "lactate",
    "creatinine",
    "bun",
    "sodium",
    "potassium",
    "bicarbonate",
    "wbc",
    "hemoglobin",
    "platelets",
    "glucose",
    "albumin",
    "bilirubin",
    "inr",
    "ph",
]

# Base natural-missingness rates. Vitals are charted continuously, routine
# chemistry comes with the morning panel, and the last group is ordered on
# clinical suspicion - which is what makes the natural pattern non-MCAR.
_ROUTINE_MISSING = {
    "heart_rate": 0.005,
    "sbp": 0.010,
    "dbp": 0.012,
    "map_mmhg": 0.015,
    "resp_rate": 0.020,
    "temperature": 0.030,
    "spo2": 0.015,
    "gcs": 0.060,
    "urine_output_24h": 0.120,
    "creatinine": 0.030,
    "bun": 0.040,
    "sodium": 0.030,
    "potassium": 0.035,
    "bicarbonate": 0.055,
    "wbc": 0.040,
    "hemoglobin": 0.035,
    "platelets": 0.045,
    "glucose": 0.030,
}

# Ordered on suspicion: (overall missing rate, how strongly severity reduces it)
_SELECTIVE_MISSING = {
    "lactate": (0.45, 1.10),
    "albumin": (0.38, 0.55),
    "bilirubin": (0.35, 0.60),
    "inr": (0.32, 0.70),
    "ph": (0.40, 1.00),
}


def generate(
    n_patients: int = 6000,
    mortality_rate: float = 0.12,
    natural_missingness: bool = True,
    seed: int = 0,
) -> Cohort:
    """Build a synthetic ICU cohort.

    Parameters
    ----------
    n_patients
        Cohort size.
    mortality_rate
        Target in-ICU mortality prevalence; the risk intercept is solved to hit
        it exactly in expectation.
    natural_missingness
        Apply the dataset's own missingness pattern. Turning this off yields a
        complete matrix, which is useful when a fully known ground truth is
        wanted.
    seed
        Controls the whole generation process.
    """
    rng = np.random.default_rng(seed)
    n = int(n_patients)

    # ---- latent severity and admission characteristics ----------------
    severity = rng.normal(0.0, 1.0, n)
    age = np.clip(rng.normal(65.0, 16.0, n), 18.0, 95.0)
    sex_male = rng.binomial(1, 0.55, n).astype(float)
    elective = rng.binomial(1, 0.25, n).astype(float)
    prior_icu = rng.binomial(1, 0.18, n).astype(float)

    # Elective admissions are less sick; age and prior ICU stays add severity.
    severity = severity - 0.60 * elective + 0.012 * (age - 65.0) + 0.25 * prior_icu
    severity = (severity - severity.mean()) / severity.std()

    def noise(scale: float) -> np.ndarray:
        return rng.normal(0.0, scale, n)

    # ---- physiology ---------------------------------------------------
    cols: dict[str, np.ndarray] = {}
    cols["heart_rate"] = np.clip(86 + 14 * severity + noise(9), 35, 190)
    cols["sbp"] = np.clip(122 - 16 * severity + noise(13), 50, 220)
    cols["dbp"] = np.clip(68 - 8 * severity + noise(9), 25, 130)
    # MAP is a deterministic function of SBP/DBP plus charting slop, which
    # creates the kind of near-redundancy real EHR tables contain.
    cols["map_mmhg"] = np.clip(
        (cols["sbp"] + 2 * cols["dbp"]) / 3.0 + noise(2.5), 30, 150
    )
    cols["resp_rate"] = np.clip(18 + 4.5 * severity + noise(3), 5, 55)
    cols["temperature"] = np.clip(36.9 + 0.35 * severity + noise(0.55), 33.0, 41.5)
    cols["spo2"] = np.clip(97.5 - 2.6 * severity + noise(1.8), 60, 100)
    cols["gcs"] = np.clip(np.rint(14.4 - 3.4 * severity + noise(1.6)), 3, 15)
    cols["urine_output_24h"] = np.clip(
        np.exp(7.35 - 0.42 * severity + noise(0.42)), 0, 6000
    )

    cols["lactate"] = np.clip(np.exp(0.35 + 0.45 * severity + noise(0.35)), 0.3, 25)
    cols["creatinine"] = np.clip(np.exp(-0.05 + 0.40 * severity + noise(0.40)), 0.2, 15)
    cols["bun"] = np.clip(np.exp(2.85 + 0.35 * severity + noise(0.38)), 3, 180)
    cols["sodium"] = np.clip(139 - 1.2 * severity + noise(4.2), 115, 165)
    cols["potassium"] = np.clip(4.1 + 0.30 * severity + noise(0.55), 2.2, 7.8)
    cols["bicarbonate"] = np.clip(24.5 - 2.9 * severity + noise(3.4), 8, 42)
    cols["wbc"] = np.clip(np.exp(2.30 + 0.32 * severity + noise(0.42)), 0.3, 80)
    cols["hemoglobin"] = np.clip(11.2 - 1.25 * severity + noise(1.7), 4, 19)
    cols["platelets"] = np.clip(225 - 48 * severity + noise(72), 5, 800)
    cols["glucose"] = np.clip(np.exp(4.90 + 0.17 * severity + noise(0.30)), 35, 700)
    cols["albumin"] = np.clip(3.35 - 0.42 * severity + noise(0.52), 1.0, 5.5)
    cols["bilirubin"] = np.clip(np.exp(-0.45 + 0.42 * severity + noise(0.55)), 0.1, 40)
    cols["inr"] = np.clip(np.exp(0.14 + 0.22 * severity + noise(0.26)), 0.8, 12)
    cols["ph"] = np.clip(7.385 - 0.038 * severity + noise(0.055), 6.75, 7.70)

    cols["age"] = age
    cols["sex_male"] = sex_male
    cols["elective_admission"] = elective
    cols["prior_icu_admission"] = prior_icu

    order = PROTECTED + VITALS + LABS
    X_true = pd.DataFrame({c: cols[c] for c in order})

    # ---- mortality risk from the true clinical values -----------------
    y, risk = _draw_outcome(X_true, mortality_rate, rng)

    # ---- the dataset's own missingness --------------------------------
    X = X_true.copy()
    if natural_missingness:
        X = _apply_natural_missingness(X, severity, rng)

    cohort = Cohort(
        X=X,
        y=pd.Series(y, name="mortality"),
        maskable=VITALS + LABS,
        protected=PROTECTED,
        groups={"demographics": PROTECTED, "vitals": VITALS, "labs": LABS},
        meta={
            "source": "synthetic",
            "seed": seed,
            "n_patients": n,
            "target_mortality_rate": mortality_rate,
            "realised_mortality_rate": float(y.mean()),
            "natural_missingness": natural_missingness,
            "mean_true_risk": float(risk.mean()),
            # Kept for diagnostics only; never used as a model feature.
            "_latent_severity": severity,
            "_complete_matrix": X_true,
        },
    )
    return cohort


def _draw_outcome(
    X: pd.DataFrame, mortality_rate: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Risk as a nonlinear function of the clinical values, then a draw."""
    lactate = X["lactate"].to_numpy()
    map_ = X["map_mmhg"].to_numpy()
    sodium = X["sodium"].to_numpy()
    wbc = X["wbc"].to_numpy()

    score = (
        0.028 * (X["age"].to_numpy() - 65.0)
        + 0.70 * np.log(lactate)
        + 0.32 * np.log(X["creatinine"].to_numpy())
        + 0.16 * (15.0 - X["gcs"].to_numpy())
        + 0.020 * (80.0 - map_)
        + 0.045 * (24.5 - X["bicarbonate"].to_numpy())
        + 0.26 * np.log(X["bilirubin"].to_numpy())
        + 0.30 * np.log(X["inr"].to_numpy())
        - 0.34 * X["albumin"].to_numpy()
        - 0.065 * X["hemoglobin"].to_numpy()
        - 0.0016 * X["platelets"].to_numpy()
        + 0.55 * np.abs(X["temperature"].to_numpy() - 37.0)
        - 0.70 * X["elective_admission"].to_numpy()
        + 0.35 * X["prior_icu_admission"].to_numpy()
    )

    # Threshold and interaction effects: a linear model cannot represent these,
    # so the gradient-boosting classifiers have genuine headroom.
    score += 0.55 * (lactate > 4.0)
    score += 0.75 * ((lactate > 2.0) & (map_ < 65.0))  # shock
    score += 0.45 * (X["urine_output_24h"].to_numpy() < 500.0)  # oliguria

    # U-shaped risk: both tails of sodium and white-cell count are dangerous.
    score += np.minimum(0.012 * (sodium - 139.0) ** 2, 1.6)
    score += np.minimum(0.040 * (np.log(wbc) - np.log(9.0)) ** 2 * 10.0, 1.2)

    score = 0.85 * score  # keep the logit range sane before calibration
    intercept = solve_intercept(score, mortality_rate)
    risk = sigmoid(score + intercept)
    y = rng.binomial(1, risk).astype(int)
    return y, risk


def _apply_natural_missingness(
    X: pd.DataFrame, severity: np.ndarray, rng: np.random.Generator
) -> pd.DataFrame:
    """Routine columns go missing near-at-random; selective labs do not."""
    X = X.copy()
    n = len(X)

    for col, rate in _ROUTINE_MISSING.items():
        X.loc[rng.uniform(size=n) < rate, col] = np.nan

    for col, (rate, strength) in _SELECTIVE_MISSING.items():
        # Sicker patients are more likely to have the test ordered, so the
        # score that drives *missingness* points the other way.
        score = -strength * severity
        prob = sigmoid(score + solve_intercept(score, rate))
        X.loc[rng.uniform(size=n) < prob, col] = np.nan

    return X
