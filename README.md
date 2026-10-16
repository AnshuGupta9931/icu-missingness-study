# Impact of Missing Data Mechanisms and Imputation Strategies on ML-Based ICU Mortality Prediction

A controlled, reproducible implementation of the project proposal in
`Project_102_revised.pdf`. It connects three stages that the literature usually
studies separately:

```
missingness mechanism  ->  imputation  ->  ICU mortality prediction
```

and asks the question that only shows up when you join them: **does the method
that reconstructs missing values most accurately also give the best mortality
prediction?**

On the synthetic cohort the answer is *only about two-thirds of the time, and
barely better than chance under MNAR*. See **[RESULTS.md](RESULTS.md)** for the
full findings from the completed 264-cell run.

---

## 1. What the experiment does

A four-factor factorial design, every cell repeated with three random seeds.

| Factor | Levels | Count |
|---|---|---|
| Missingness mechanism | MCAR, MAR, MNAR | 3 |
| Missingness level | 10% … 70% in steps of 10 | 7 |
| Imputation method | median, KNN, MICE, MissForest | 4 |
| Classifier | logistic regression, XGBoost, CatBoost, LightGBM | 4 |
| Seeds | 0, 1, 2 | 3 |

That is 252 experimental cells, plus 12 reference cells with no artificial
missingness, evaluated on two splits for four classifiers: **2,112 result
rows**.

Each cell runs the pipeline end to end:

```
cohort -> split -> mask -> impute (fit on train only) -> classify -> evaluate
```

---

## 2. Quickstart

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt       # or requirements.lock.txt for exact versions

# macOS only: XGBoost and LightGBM need the OpenMP runtime
brew install libomp                             # if Homebrew works on your machine
bash scripts/setup_libomp_macos.sh              # otherwise: installs it into .venv

# 1. look at the cohort and its own missingness  (proposal objective 1)
.venv/bin/python scripts/profile_cohort.py --config configs/pilot.yaml

# 2. confirm MCAR / MAR / MNAR are actually distinct  -- run this first, always
.venv/bin/python scripts/verify_design.py --config configs/pilot.yaml

# 3. pilot study (~1 minute)
.venv/bin/python scripts/run_experiments.py --config configs/pilot.yaml

# 4. the full grid
.venv/bin/python scripts/run_experiments.py --config configs/full_synthetic.yaml

# re-render tables and figures without re-running any model
.venv/bin/python scripts/make_report.py --run results/full_synthetic

# the test suite, including the leakage guards
.venv/bin/python -m pytest
```

The scripts put `src/` on the path themselves, and `pyproject.toml` does the
same for pytest, so no `PYTHONPATH` export is needed.

---

## 3. Datasets

### MIMIC-IV — the primary dataset

MIMIC-IV requires credentialed PhysioNet access: CITI "Data or Specimens Only
Research" training, a signed data use agreement, then download from
<https://physionet.org/content/mimiciv/>. **Patient-level data must never be
committed**; `.gitignore` is set up to prevent it.

```bash
# 1. check the files and resolve every itemid against your MIMIC version
.venv/bin/python scripts/verify_mimic.py --root /path/to/mimic-iv

# 2. point the config at your copy
#    configs/full_mimic.yaml -> dataset.params.root

# 3. extract and inspect the cohort before running anything
.venv/bin/python scripts/profile_cohort.py --config configs/full_mimic.yaml
```

Cohort definition (`src/icumiss/data/mimic.py`):

* adults (age ≥ 18) on their **first** ICU stay only, so rows are independent;
* features summarise the **first 24 hours** after ICU admission;
* target: death during the ICU stay.

Two leakage guards are enforced rather than assumed:

1. **Patients who die inside the observation window are excluded.** Predicting
   an event that has already happened inflates every metric.
2. **Nothing after `intime + 24h` is read.** Discharge time and length of stay
   are never used as features.

> Run `scripts/verify_mimic.py` and keep its itemid report with your results.
> itemids differ between MIMIC versions, and a silently wrong itemid is the
> easiest way to produce a confident wrong answer.

### Synthetic cohort — for development and demonstration

Because MIMIC-IV cannot live in the repository, `src/icumiss/data/synthetic.py`
generates a cohort that exercises every stage. It is **not** a substitute for
the real analysis, but it is built so the experiment can actually discriminate
between methods:

* risk depends on the clinical values **nonlinearly and with interactions**
  (shock = high lactate with low MAP; U-shaped risk in sodium and white-cell
  count), so distorting those values costs real predictive signal and the tree
  ensembles have something logistic regression cannot reach;
* natural missingness is **informative**: lactate, albumin, INR and pH are
  ordered on clinical suspicion, so they are missing more often in healthier
  patients;
* features share a latent severity, so neighbour- and model-based imputers have
  information to borrow and the median baseline does not.

Default: 8,000 patients, 27 features, ~12% mortality, ~9% of cells missing
before any masking.

---

## 4. How each mechanism is imposed

The study does **not** claim to identify the mechanism behind missingness that
is already in observational data — that is not identifiable from the data
alone. Each mechanism is instead *imposed* by masking values that were
originally recorded, so the condition is known by construction and the masked
values give a ground truth to score against.

| Mechanism | Masking depends on | Implementation |
|---|---|---|
| **MCAR** | nothing | noise only |
| **MAR** | other **observed** variables | a fixed-weight index over the protected columns (age, sex, admission type) |
| **MNAR** | **the value being hidden** | the standardised value of the column itself; `mnar_direction: high` hides large values |

Only currently-observed cells in maskable columns are eligible, so ground truth
is never invented where none existed. Protected demographic columns are never
masked, which is what makes MAR genuinely "missing at random *given observed
data*".

**Levels are exact.** Each column gets a per-cell score, standard logistic noise
is added, and the top `k = round(level × n_observed)` cells are masked. A noisy
top-k is equivalent to sampling k cells without replacement under a logit model
on the score, so the realised level matches the request to the rounding — which
is what makes the x-axis of every figure mean what it says — while the intended
dependence is preserved.

Verified on the synthetic cohort by `scripts/verify_design.py`:

| requested | mechanism | corr(mask, hidden value) | corr(mask, observed index) | realised |
|---|---|---|---|---|
| 0.30 | MCAR | 0.001 | −0.001 | 0.300 |
| 0.30 | MAR | −0.010 | **0.574** | 0.300 |
| 0.30 | MNAR | **0.570** | −0.006 | 0.300 |

---

## 5. The leakage contract

This is the part that decides whether any of the numbers mean anything.

* **Split first.** Train/validation/test is stratified on mortality and carved
  out before masking and before any imputer is fitted.
* **Imputers are fitted on training rows only**, then applied unchanged to
  validation and test. Fitting on pooled data would let test distributions leak
  into the transform and would quietly inflate everything downstream.
* **The outcome is never given to an imputer.** Mortality is unknown at
  prediction time.
* **Observed values are never modified.** Imputation fills gaps and nothing
  else: the range clip and the internal scale/inverse-scale round trip apply
  only to cells that were actually missing.
* **Classifiers are given complete matrices and reject NaN.** XGBoost and
  LightGBM can route missing values natively; letting them would confound the
  very thing being measured, so it is blocked with an explicit error.
* **The decision threshold is chosen on validation and applied to test**, so
  F1 and precision are never tuned against the data they score.

`tests/test_leakage.py` enforces these. The decisive one
(`test_imputer_output_does_not_depend_on_unseen_data`) corrupts the rest of the
test set and asserts a given row's imputed values do not move.

---

## 6. Metrics

### Imputation quality — scored only on artificially masked cells

MAE, RMSE, R² and adjusted R², exactly as the proposal defines them:

```
R²     = 1 − Σ(y − ŷ)² / Σ(y − mean(y))²
adj R² = 1 − (1 − R²)(n − 1) / (n − p − 1)
```

with `n` = masked entries and `p` = the other features the imputer draws on.
Naturally missing cells have no ground truth and are excluded everywhere.

Each metric is reported twice:

* `*_raw` — in original units, as the proposal specifies;
* `*_z` — on z-scores taken from the training distribution **before masking**.

**Compare methods on the `_z` versions.** Pooling raw errors across columns
measured in different units lets the widest-ranged variable dominate: platelets
span hundreds, pH spans fractions of a unit, so a raw pooled RMSE is largely a
statement about platelets. Per-column values are in
`tables/imputation_per_column.csv` for interpretation.

The scale deliberately comes from the **unmasked** training split. MNAR removes
the upper tail of each column, shrinking the observed training spread to about
0.69 of its true value (0.41 for the worst columns) while MCAR leaves it at
0.99. Scaling by the masked spread would inflate MNAR's error by roughly 45%
for a reason that has nothing to do with imputation quality, and the mechanism
panels could not be compared with one another. This is a reporting scale
computed from training rows only — it never reaches the imputer or the
classifier, so it leaks nothing.

A useful sanity anchor: median imputation scores **R² ≈ 0** by construction,
because predicting a constant is what R² is defined against.

### Downstream prediction

AUROC, AUPRC, F1, precision, recall, log loss and the Brier score.

F1 and precision need a cut-point, and at ~12% prevalence the default 0.5 makes
every model look falsely precise (it often predicts no deaths at all, giving
F1 = 0). The threshold is therefore chosen on validation to maximise F1 and
applied unchanged to test; `f1_at_0.5` and `precision_at_0.5` are also recorded.

No class weighting or resampling is applied. Log loss and the Brier score are
headline metrics and both reward honest probabilities; reweighting improves F1
at the cost of calibration. `model_params` in the config can switch it on for a
sensitivity check.

---

## 7. Results layout

```
results/<run_name>/
  manifest.json                     provenance: git commit, library versions, platform, timings
  config.resolved.json              the exact resolved configuration
  cells.jsonl                       append-only cell log (makes runs resumable)
  results.parquet / results.csv     the tidy result frame
  natural_missingness_profile.csv   the cohort's own missingness
  mask_diagnostics.json             evidence the mechanisms behaved as intended
  imputation_per_column.csv         reconstruction metrics per variable
  tables/                           10 summary tables
  figures/                          13 figures
```

The tables that answer the proposal's research questions:

| Question | Table |
|---|---|
| How does increasing missingness affect performance? | `degradation_auroc.csv` |
| How does performance differ across MCAR/MAR/MNAR? | `mechanism_comparison.csv` |
| Which imputer is best, and where? | `best_imputer_auroc.csv`, `best_imputer_auprc.csv`, `best_imputer_brier.csv` |
| **Does better imputation mean better prediction?** | `reconstruction_vs_downstream.csv` |
| Does the preferred method depend on mechanism / level / classifier? | `preference_stability.csv` |

`reconstruction_vs_downstream.csv` is the central analysis. Within each
(mechanism, level, classifier) it ranks the four imputers twice — by
reconstruction error and by downstream performance — and compares the rankings
with Spearman's rho. Four points is few, so no single row is conclusive; the
distribution across conditions and the `best_agrees` column carry the finding.

---

## 8. Reproducibility

Everything the proposal's section 18 asks for:

* **Three seeds minimum** (0, 1, 2). The seed controls the split, the masking,
  the imputers and the classifiers. Per-column RNG streams mean a column's mask
  does not shift when the column order changes.
* **Every run writes its resolved config** and a manifest recording the git
  commit, platform and the version of every library that can change a result.
* **Pinned versions** in `requirements.lock.txt`; LightGBM runs with
  `deterministic=True, force_row_wise=True`, since its default histogram
  construction is not bit-stable across thread counts.
* **Resumable and fail-soft.** Finished cells are appended to `cells.jsonl`;
  re-running the same config continues where it stopped. A cell that raises is
  recorded with `status="failed"` and its traceback, and the rest of the grid
  completes — a run that silently dropped failures would be worse than one that
  reports them.
* **`CHANGELOG.md`** records every change with date, author, what and why.
* **Patient-level data is never committed.**

---

## 9. Decisions worth knowing about

Points where implementation required a judgement the proposal did not specify.
Each is defensible, and each is written down so a reader can disagree.

1. **MissForest is `IterativeImputer` with random-forest regressors.** That is
   the MissForest algorithm, in a form that supports a strict `fit`/`transform`
   split — which the leakage contract requires and the original R package does
   not offer.
2. **MICE is single imputation.** `IterativeImputer` with Bayesian ridge
   regressions, one completed dataset per condition. Proper multiple imputation
   would produce *m* datasets and pool the results; the downstream stage needs
   one dataset per cell, and pooling across *m* would change what is being
   compared. This narrows MICE's usual advantage (uncertainty propagation) and
   should be stated as a limitation rather than glossed over.
3. **Imputed values are clipped to the training range** of each column, so
   model-based imputers cannot produce negative platelet counts or a pH of 4.
   Applied identically to all four methods, and only to cells that were
   actually missing.
4. **KNN and MICE run on standardised features.** On raw units the KNN distance
   would be dominated by platelets and glucose and would effectively ignore pH.
5. **Masking is applied per split**, after the split, with independent draws.
6. **The level-0 reference still runs an imputer**, because the cohort's own
   natural missingness has to be filled before any classifier can be fitted.
   That gives every curve an anchor at level 0.
7. **Early stopping on the validation split** for all three boosters, so model
   capacity is not a hidden free parameter that drifts with missingness level.
8. **A guard keeps at least 10 observed values per column**, so an imputer is
   never handed an all-NaN training column. At realistic cohort sizes it never
   binds below 70% missingness, and whenever it does bind the realised level is
   recorded and will differ from the request.

### Scope, per the proposal

Excluded as the proposal specifies: fuzzy decision trees, fuzzy membership
functions, genetic-algorithm optimisation, deep-learning imputation. NGBoost
and eICU-CRD external validation are optional extensions; the dataset registry
(`src/icumiss/data/__init__.py`) is the one place to add eICU.

---

## 10. Project structure

```
src/icumiss/
  config.py           typed config, YAML loading, provenance capture
  utils.py            numeric helpers (stable sigmoid, intercept solver, noisy top-k)
  data/
    cohort.py         the Cohort container and its validation
    synthetic.py      synthetic ICU cohort generator
    mimic.py          MIMIC-IV extraction (streamed, cached)
    __init__.py       dataset registry
  missingness.py      MCAR / MAR / MNAR masking + diagnostics
  splits.py           leakage-safe stratified splitting
  imputation.py       median / KNN / MICE / MissForest, fit on train only
  models.py           logistic regression / XGBoost / CatBoost / LightGBM
  metrics.py          imputation and classification metrics
  experiment.py       one cell: mask -> impute -> classify -> evaluate
  runner.py           grid orchestration, resumable and parallel
  analysis/
    tables.py         the summary tables
    plots.py          the figure set

scripts/
  run_experiments.py      run a grid from a config
  make_report.py          rebuild tables and figures for a finished run
  profile_cohort.py       cohort and natural-missingness profile
  verify_design.py        check MCAR/MAR/MNAR are distinct  <- run this first
  verify_mimic.py         check MIMIC-IV files and itemids
  setup_libomp_macos.sh   install the OpenMP runtime into .venv

configs/
  pilot.yaml            small, fast, for validating the pipeline
  full_synthetic.yaml   the full factorial design on synthetic data
  full_mimic.yaml       the full factorial design on MIMIC-IV

tests/                  86 tests: masking contract, leakage guards, metric definitions
```

---

## 11. Troubleshooting

**`Library not loaded: @rpath/libomp.dylib`** — XGBoost and LightGBM need the
OpenMP runtime. `brew install libomp`, or `bash scripts/setup_libomp_macos.sh`
if Homebrew is unavailable (for example when its prefix is owned by another
user account). Re-run the script after upgrading either library, which restores
the unpatched wheel.

**A run is slow.** MissForest at high missingness dominates. Lower
`imputer_params.missforest.n_estimators`, or raise `compute.n_jobs`. The run is
resumable, so stopping it costs only the cells in flight.

**Memory pressure.** Lower `compute.n_jobs`; each worker holds its own copy of
the cohort and the fitted forests.

**A cell failed.** Its traceback is in `results/<run>/cells.jsonl` under the row
with `status="failed"`. Fix the cause and re-run the same command — completed
cells are skipped.
