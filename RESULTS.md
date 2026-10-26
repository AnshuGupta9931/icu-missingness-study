# Results — full factorial runs on MIMIC-IV and the synthetic cohort

Two complete 264-cell runs: **MIMIC-IV**, the project's primary result set, and
the **synthetic** cohort, on which the pipeline and the analysis were developed.

**For every finding condensed into a single table, see
[FINDINGS.md](FINDINGS.md)** (machine-readable: `results/findings_summary.csv`).

| | MIMIC-IV | synthetic |
|---|---|---|
| cells → result rows | 264 → 2,112 | 264 → 2,112 |
| failures | 0 | 0 |
| cohort | 51,666 patients, 26 features | 8,000 patients, 27 features |
| mortality | 6.8% | 12.2% |
| missing before masking | 6.6% | 9.2% |
| runtime | 2.6 h | 56.5 min |

The MIMIC-IV run is reported in the next section. Sections 1–5 after it report
the synthetic run, where the analysis was built; those numbers remain true of
the synthetic cohort and are kept for comparison, not superseded.

Reproduce with `python scripts/run_experiments.py --config configs/full_mimic.yaml`
and `--config configs/full_synthetic.yaml` respectively.

---

## MIMIC-IV full grid — the primary result set

Run `full_mimic`, 264 cells → 2,112 result rows, 0 failures, 9,380 seconds
(2.6 hours). Cohort: 51,666 patients, 26 features (22 maskable, 4 protected),
6.8% ICU mortality, 6.6% of cells missing before any masking. Adults, first ICU
stay only, first 24 hours after admission; patients who died inside the
observation window are excluded.

| | no masking | 10% masked | 70% masked |
|---|---|---|---|
| AUROC | 0.901 | 0.890 | 0.791 |
| AUPRC | — | 0.449 | 0.283 |
| Brier | — | 0.0482 | 0.0559 |

**Does better imputation accuracy mean better mortality prediction?** On
MIMIC-IV the method that reconstructs best is the method that predicts best in
**25.0%** of conditions — 21 of 84. The median Spearman correlation between
reconstruction error and AUROC is **−0.40**. Restricting to the 81 conditions
where the imputers are separated by more than seed noise gives **24.7%**
(20 of 81).

The sharpest single illustration: **median imputation was never the best
reconstructor in any of the 84 conditions, yet it was the best predictor in 51
of them.** The best reconstructor was always MICE (44 conditions) or MissForest
(40). The best predictor was most often median (51), then MissForest (26), then
MICE (7).

Agreement by mechanism, out of 28 conditions each:

| mechanism | best-method agreement | median rho |
|---|---|---|
| MCAR | 35.7% (10 / 28) | −0.8 |
| MAR | 21.4% (6 / 28) | −0.8 |
| **MNAR** | **17.9% (5 / 28)** | **−0.2** |

An earlier reduced MIMIC pilot (`results/pilot_mimic/`: 2 mechanisms, 3 levels,
1 seed, 2 classifiers) gave 58.3% agreement and is superseded by this run. The
two figures are not directly comparable — the pilot sampled a much smaller and
different set of conditions — and no explanation for the size of the gap is
offered here.

### Compute and platform notes

The grid, imputer settings and seeds are identical to `configs/full_mimic.yaml`;
only compute settings differ. The run used `configs/full_mimic_windows.yaml`
with `n_jobs: 1` and `inner_threads: 4`, because a full-size MissForest cell on
this cohort peaks at about 7 GB and parallel workers exhausted 16 GB of RAM.
MissForest results do not depend on thread count; LightGBM may differ in the
last decimal places from a single-threaded run.

The run executed on Windows 11 under Python 3.12.1 with pandas 3.1.0 and numpy
2.5.4, and `results/full_mimic/manifest.json` records those versions. The first
27 of the 264 cells were computed earlier on macOS (Python 3.13, pandas 3.0.6)
and carried over in `cells.jsonl`, so the manifest's version block does not
describe every cell in the set.

---

## The five research questions — synthetic cohort

### 1. How does increasing missingness affect performance?

Discrimination degrades gradually; the positive class suffers far more.

| | no masking | 10% masked | 70% masked |
|---|---|---|---|
| AUROC | 0.858 | 0.857 | 0.808 |
| AUPRC | — | 0.517 | 0.407 |
| Brier | — | 0.0791 | 0.0885 |

AUROC falls 5.7% between 10% and 70% missingness; **AUPRC falls 21%**, nearly
four times as much. Reporting AUROC alone would make the degradation look
roughly a quarter as severe as it is for the minority class that actually
matters in a mortality model. Calibration worsens monotonically too.

### 2. How does performance differ under MCAR, MAR and MNAR?

The expected ordering does not hold. **MAR is the most damaging mechanism, and
MNAR the least** — the opposite of what reconstruction error predicts.

| mechanism | AUROC at 10% | AUROC at 70% | AUPRC at 70% |
|---|---|---|---|
| MCAR | 0.858 | 0.824 | 0.427 |
| MAR | 0.857 | **0.778** | **0.324** |
| MNAR | 0.855 | 0.823 | 0.470 |

The explanation is *where* the information is lost, not how much. MAR
missingness here depends on the protected index dominated by age, so it
concentrates on older, sicker patients — exactly the cases the model most needs
to resolve. MCAR and MNAR spread the loss across patients instead: MNAR removes
the upper tail *within each column*, leaving every patient with some intact
measurements.

This is worth stating carefully, because it is a property of the MAR rule
chosen, not a universal law. A MAR rule conditioning on something unrelated to
risk would not behave this way. What the result shows is that **the alignment
between the missingness driver and the outcome matters more than the mechanism
label.**

### 3. Which imputation method performs best?

Averaged over every condition:

| imputer | AUROC | AUPRC | Brier | reconstruction RMSE (z) |
|---|---|---|---|---|
| MICE | **0.8455** | **0.4826** | **0.0820** | 0.73 – 1.35 |
| MissForest | 0.8430 | 0.4792 | 0.0823 | 0.75 – 1.39 |
| Median | 0.8414 | 0.4729 | 0.0831 | ~1.0 (1.85 under MNAR) |
| KNN | 0.8240 | 0.4470 | 0.0853 | 0.83 – 1.52 |

Two findings stand out.

**KNN degrades catastrophically with missingness** — the single largest effect
in the study. Its AUROC falls 13.6% under MAR at 70% masking, against 6.8% for
MICE. Its reconstruction error actually becomes *worse than the median
baseline* beyond about 50% masking (1.11 vs 1.00 under MCAR at 70%). The reason
is structural: with most values gone, the neighbours that define each
imputation are themselves mostly imputed, so errors compound. KNN should not be
used on heavily incomplete ICU data.

**The median baseline is far more competitive than its reconstruction error
suggests.** It has the worst reconstruction of the four by a wide margin, yet
lands within 0.004 AUROC of MICE and beats KNN decisively everywhere. Under
MNAR it wins 10 of 28 conditions outright — more than MICE's 3.

### 4. Does better imputation accuracy mean better mortality prediction?

**Partly, and much less than the reconstruction numbers imply.**

Across the 84 conditions, the Spearman correlation between reconstruction error
and AUROC has median −0.8: the direction is right, lower error does tend to
mean better prediction. But the method that reconstructs best is the method
that predicts best in only **59.5%** of conditions, and restricting to the 46
conditions where the imputers are separated by more than seed noise only raises
that to 67.4%. A third of the time, picking the imputer by reconstruction error
picks the wrong one. On the MIMIC-IV cohort the same figure is **25.0%** (21 of
84) — see the MIMIC-IV section above.

The agreement depends strongly on the mechanism:

| mechanism | best-method agreement | median rho |
|---|---|---|
| MAR | 78.6% | −0.8 |
| MCAR | 53.6% | −0.8 |
| **MNAR** | **46.4%** | **−0.4** |

**Under MNAR, reconstruction accuracy is close to useless as a selection
criterion** — worse than a coin flip at picking the best downstream method.
This is the mechanism where ground truth is least recoverable, and it is
exactly where practitioners most need guidance.

The effect sizes make the same point from the other direction. Under MCAR at
10% masking, KNN reconstructs 17% better than median (RMSE 0.83 vs 1.00) and
converts that into 0.001 AUROC — about a tenth of the seed-to-seed variation.
Large reconstruction gains buy almost nothing downstream.

### 5. How do imputation methods interact with the classifier?

The preferred imputer is not stable. Wins by mechanism (out of 28 conditions):

| | KNN | Median | MICE | MissForest |
|---|---|---|---|---|
| MCAR | 2 | 0 | **15** | 11 |
| MAR | 0 | 4 | **22** | 2 |
| MNAR | 2 | **10** | 3 | **13** |

And by classifier (out of 21):

| | KNN | Median | MICE | MissForest |
|---|---|---|---|---|
| Logistic regression | 1 | 0 | **13** | 7 |
| XGBoost | 1 | 5 | **12** | 3 |
| CatBoost | 0 | 2 | **10** | 9 |
| LightGBM | 2 | **7** | 5 | 7 |

MICE dominates under MCAR and MAR and for the linear model, but under MNAR it
drops to third behind MissForest and the plain median. LightGBM prefers a
different imputer from the other three classifiers. **There is no single
recommendation that holds across the grid**, which is the interaction the
proposal set out to detect.

---

## What this means for practice

1. **Report AUPRC and a calibration metric, not AUROC alone.** AUROC understates
   the damage by roughly fourfold at high missingness.
2. **Do not select an imputation method by reconstruction error**, especially
   when missingness may be non-ignorable. It picks the wrong method a third of
   the time overall and more than half the time under MNAR.
3. **Avoid KNN imputation beyond moderate missingness.** It is the worst method
   in this study on every downstream metric and falls behind a median baseline
   even on reconstruction past ~50%.
4. **MICE is the best default**, but the margin over a median baseline is small
   enough that the baseline is a defensible choice when transparency matters.
5. **Which patients lose data matters more than how much is lost.** The MAR
   condition, where missingness tracked age, was the most damaging despite
   being the mechanism textbooks treat as benign.

---

## Caveats

* **Synthetic cohort.** The conclusions describe a dataset built to have
  informative natural missingness, nonlinear risk and correlated features. The
  MIMIC-IV run is what the project reports.
* **One MAR rule and one MNAR rule.** Finding 2 depends on the MAR conditioner
  being correlated with the outcome, and the MNAR rule hides the upper tail.
  Other rules would give other answers; `missingness.mar_strength`,
  `mnar_strength` and `mnar_direction` exist to probe exactly that.
* **Three seeds.** Enough to see that 38 of 84 conditions have imputer
  differences smaller than run-to-run noise, not enough for formal inference on
  individual cells. `spread_exceeds_seed_noise` in
  `tables/reconstruction_vs_downstream.csv` flags which rows carry weight.
* **MICE is single imputation here** (see README section 9), so it is not
  credited with the uncertainty propagation that is normally its main argument.
* **Default hyperparameters.** No per-condition tuning; the classifiers use
  early stopping only. A tuned KNN might degrade less gracefully than shown.

---

## Figures

In `results/full_synthetic/figures/`:

| file | what it shows |
|---|---|
| `auroc_vs_level.png` | the headline degradation curves, one panel per mechanism |
| `auprc_vs_level.png` | the same for AUPRC — note the much steeper fall |
| `brier_vs_level.png` | calibration decay |
| `reconstruction_rmse_vs_level.png` | reconstruction error, identical across classifiers |
| `reconstruction_vs_downstream.png` | **the central scatter**: error against AUROC, per imputer |
| `best_imputer_auroc.png` | which method wins where; faded cells are ties within seed noise |
| `auroc_heatmap.png` | imputer × level grid of AUROC |
| `natural_missingness.png` | the cohort's own missingness before any masking |
| `auroc_vs_level_<classifier>.png` | the degradation curves per classifier |
