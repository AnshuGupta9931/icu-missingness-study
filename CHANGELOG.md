# Change log

Every change to the implementation, with the reason for it. Required by
section 18 of the project proposal ("a change log recording every change: date,
author, what changed and why").

Format: `## YYYY-MM-DD — author`, then entries grouped as Added / Changed /
Fixed / Decided. "Decided" records a methodological choice that affects how
results should be read, not a code change.

---

## 2026-10-10 — Anshu Gupta

Initial implementation of the revised proposal.

### Added

* **Project skeleton**: `src/icumiss/` package, `configs/`, `scripts/`,
  `tests/`, pinned dependencies in `requirements.txt` and
  `requirements.lock.txt`, `.gitignore` that blocks patient-level data from
  ever being committed.
* **`config.py`** — typed configuration objects loaded from YAML, with
  validation, a config fingerprint, and a provenance manifest capturing the git
  commit, platform and the version of every library that can change a result.
* **`data/cohort.py`** — the `Cohort` container. Validates on construction that
  the matrix is numeric, the outcome is binary, and the protected columns are
  genuinely complete, since MAR depends on that last property being true.
* **`data/synthetic.py`** — synthetic ICU cohort generator. Needed because
  MIMIC-IV requires credentialed access and cannot enter the repository, so
  without it the pipeline could not be developed, tested or demonstrated.
* **`data/mimic.py`** — MIMIC-IV extraction for 24-hour ICU mortality:
  streams `chartevents` and `labevents` in chunks, filters to the configured
  itemids and observation window, applies physiological range filters, derives
  GCS and 24-hour urine output, and caches the result as Parquet.
* **`missingness.py`** — controlled MCAR / MAR / MNAR masking, with
  `describe_mask` diagnostics that measure whether each imposed mechanism
  behaves as intended.
* **`splits.py`** — stratified train/validation/test splitting, performed
  before masking and before any imputer is fitted.
* **`imputation.py`** — median, KNN, MICE and MissForest behind one
  fit-on-train / transform-everywhere interface.
* **`models.py`** — logistic regression, XGBoost, CatBoost and LightGBM behind
  one interface, with early stopping on the validation split.
* **`metrics.py`** — MAE, RMSE, R² and adjusted R² on masked cells only;
  AUROC, AUPRC, F1, precision, recall, log loss and Brier downstream.
* **`experiment.py` / `runner.py`** — one cell (mask → impute → classify →
  evaluate) and the grid around it: parallel, resumable, fail-soft.
* **`analysis/tables.py`, `analysis/plots.py`** — 10 summary tables and 13
  figures, including the central reconstruction-versus-downstream analysis.
* **Scripts** — `run_experiments.py`, `make_report.py`, `profile_cohort.py`,
  `verify_design.py`, `verify_mimic.py`, `setup_libomp_macos.sh`.
* **Tests** — 86 tests covering the masking contract, the leakage guards, the
  metric definitions against hand-computed values, and cell/grid orchestration
  (including resume after an interrupted run).
* **`pyproject.toml`** — package metadata plus pytest configuration, so the
  test suite and the scripts run without a `PYTHONPATH` export.

### Fixed

* **Imputers crashed under pandas 3.0.** `DataFrame.to_numpy()` returns a
  read-only array under copy-on-write, and scikit-learn's `IterativeImputer`
  writes into the array it is given when `keep_empty_features=True`, raising
  "assignment destination is read-only". Imputers are now handed an explicit
  writable copy.
* **Imputation was altering values that were never missing.** Two causes: the
  training-range clip applied to every cell rather than only the filled ones,
  and the scale → impute → inverse-scale round trip inside the KNN and MICE
  wrappers left float drift on observed values. `transform` now writes imputed
  values only where the input was NaN. This matters beyond tidiness: a recorded
  measurement outside the training range is real data, and silently clipping it
  would corrupt the features every classifier is then judged on.
* **The z-scored reconstruction metrics were not comparable across
  mechanisms.** The scaling statistics were taken from the *masked* training
  split. MNAR removes the upper tail of every column, which shrinks the
  training spread to roughly 0.69 of its true value (0.41 for the worst
  columns), while MCAR leaves it at 0.99 — so dividing by that spread inflated
  MNAR's reported error by about 45% for a reason unrelated to imputation
  quality, and the mechanism panels of the reconstruction figures could not be
  read against each other. The scale now comes from the training split before
  masking, which is fixed across every condition. It remains a reporting scale
  computed from training rows only: it reaches neither the imputer nor the
  classifier, so it leaks nothing. Comparisons *between imputers within* a
  condition were never affected; comparisons *between mechanisms* were.
* **Workers oversubscribed the machine, roughly halving throughput.** Setting
  `n_jobs=1` / `thread_count=1` on the estimators is not sufficient: XGBoost,
  LightGBM, CatBoost and the BLAS behind NumPy each start an OpenMP pool sized
  to the core count the first time they are used, so six worker processes were
  running about sixty threads on eight cores. The grid now runs inside
  `joblib.parallel_config(inner_max_num_threads=...)`, which exports the
  OMP/BLAS limits into each worker before those libraries are imported — the
  only moment they are read. Worker thread count dropped from 10 to 2 and
  throughput rose from roughly 5 to 12 cells per minute. Note that passing
  `inner_max_num_threads` to `Parallel` directly is silently ignored; it has to
  be the context manager.
* **The mask diagnostic reported a missingness level that was too low.** It
  recounted the realised level over all columns, including the protected ones
  that are never masked, instead of over the maskable columns the masker
  measures against. The recorded `realised_level` in the results was always
  correct; only the diagnostic was wrong.
* **Figures**: the baseline annotation and the sub-titles overlapped the data
  and each other; the error-band caption claimed "across seeds" on panels that
  also average over classifiers; `semibold` is not a DejaVu Sans weight, so
  every figure logged a font warning; the best-imputer heatmap used white text
  on the yellow cells and hatched every cell when only one seed made ties
  impossible to assess.

### Decided

Choices the proposal did not specify, recorded because they affect how the
results should be read. Each is argued in README section 9.

* **MissForest** is implemented as `IterativeImputer` with random-forest
  regressors — the MissForest algorithm in a form that supports the strict
  fit/transform separation the leakage contract requires.
* **MICE is single imputation**, not pooled multiple imputation, because the
  downstream stage needs exactly one dataset per condition. This narrows MICE's
  usual advantage and is a stated limitation.
* **Imputed values are clipped to the training range**, applied identically to
  all four methods and only to cells that were missing.
* **KNN and MICE operate on standardised features**; MissForest does not, since
  trees are scale-invariant.
* **Missingness levels are hit exactly** by adding logistic noise to a
  per-cell score and taking the top-k, which is equivalent to sampling k cells
  without replacement under a logit model on that score. Exact levels keep the
  x-axis of every figure meaningful.
* **The decision threshold is selected on validation**, not fixed at 0.5. At
  ~12% prevalence a 0.5 cut often predicts no deaths at all and reports F1 = 0,
  which measures the prevalence rather than the model.
* **No class weighting.** Log loss and Brier are headline metrics and both
  reward honest probabilities; reweighting trades calibration for F1.
* **The level-0 reference still runs an imputer**, because the cohort's own
  natural missingness must be filled before any classifier can be fitted.
* **Imputation metrics are reported in both raw and z-scored form**, with the
  z-scored versions recommended for comparing methods: pooling raw errors
  across columns in different units makes the pooled number a statement about
  whichever variable has the widest range.

### Environment note

XGBoost and LightGBM wheels on macOS do not ship `libomp.dylib`. Homebrew could
not install it on this machine because its prefix is owned by a different user
account, and taking ownership would have affected that account. Instead
`scripts/setup_libomp_macos.sh` installs the verified Homebrew bottle into
`.venv/` and patches the two wheels to look there, leaving the rest of the
system untouched. Re-run it after upgrading either library.

---

## Template for future entries

```markdown
## YYYY-MM-DD — author

### Added / Changed / Fixed / Decided
* what changed, and why it was necessary
```

---

## 2026-10-10 (later) — Anshu Gupta

### Added

* **Full factorial run on the synthetic cohort** (`results/full_synthetic/`):
  264 cells, 2,112 result rows, 0 failures, 56.5 minutes on six workers.
* **`RESULTS.md`** — findings against each of the proposal's five research
  questions, with the caveats that limit them.

### Findings worth flagging

* AUROC understates the damage: between 10% and 70% masking it falls 5.7%
  while AUPRC falls 21%.
* MAR was the *most* damaging mechanism and MNAR the least, which inverts the
  usual expectation. The MAR rule conditions on an age-dominated index, so the
  loss concentrates on the highest-risk patients; MNAR spreads it within each
  column instead. The driver's alignment with the outcome mattered more than
  the mechanism label.
* Reconstruction error picked the best downstream method in 59.5% of
  conditions overall and only 46.4% under MNAR — the central negative result
  the study was designed to detect.
* KNN imputation degrades catastrophically with missingness and falls behind a
  median baseline on reconstruction past roughly 50% masking.
