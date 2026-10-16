# Results — full factorial run on the synthetic cohort

Run `full_synthetic`, 264 cells → 2,112 result rows, 0 failures, 56.5 minutes
on 6 workers. Cohort: 8,000 patients, 27 features, 12.2% mortality, 9.2% of
cells missing before any masking. Split 5,200 / 1,200 / 1,600.

**These are results on the synthetic cohort, not on MIMIC-IV.** They
demonstrate that the pipeline answers the proposal's five research questions
and establish what the analysis looks like. The headline findings of the
project come from re-running `configs/full_mimic.yaml` once MIMIC-IV access is
in place. Every number below is reproducible with
`python scripts/run_experiments.py --config configs/full_synthetic.yaml`.

---

## The five research questions

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
picks the wrong one.

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
