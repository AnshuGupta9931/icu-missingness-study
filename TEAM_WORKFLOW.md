# Team workflow — repository, ownership and commit plan

**Project:** Impact of Missing Data Mechanisms and Imputation Strategies on
Machine-Learning-Based ICU Mortality Prediction

**Repository:** https://github.com/AnshuGupta9931/icu-missingness-study

| Name | Role | Owns |
|---|---|---|
| **Anshu Gupta** | Team lead — integration, orchestration, documentation | Config, experiment runner, docs, repo admin |
| **Aditya Chaudhary** | Data engineering | Cohort building, MIMIC-IV extraction, synthetic data |
| **Ankit Kumar Anand** | Experimental design | MCAR/MAR/MNAR masking, leakage-safe splitting |
| **Annepu Maharshi** | Methods and analysis | Imputation, classifiers, metrics, tables and figures |

---

## 0. How this works — read this first

The code is complete and sitting in a **reference copy** on Anshu's machine at
`~/Desktop/icumiss-source/`. It is **not** a Git repository. It is the source
you copy your files from.

The GitHub repository starts almost empty and is filled in **rounds**. In each
round, the owner copies their own files out of the reference copy into their
clone, commits them, and opens a Pull Request.

```
~/Desktop/icumiss-source/          the complete code (not a git repo)
          |
          |  each person copies only their own files
          v
~/Desktop/icu-missingness-study/   the git repo, grows round by round
```

**Why not just push everything at once?** Because then the entire project is
one person's commit. Filling it in rounds gives every member real, reviewable
commits in their own area — which is the point of a team repository.

**What "ownership" means.** You are not pretending you typed your section from
scratch. You are taking responsibility for it: you can explain how it works,
defend its design choices, debug it, and answer a question about it in the
viva. **Read your files line by line before you commit them.**

Check your university's policy on declaring AI assistance. `CHANGELOG.md`
records what was built and why, which is the document to point to if asked.

---

## 1. Environment — and the one real gotcha

`.venv/` is gitignored. Each member builds their own; virtual environments
contain OS-specific binaries and are never shared through Git.

```bash
# macOS / Linux
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### The gotcha: macOS only

XGBoost and LightGBM on macOS need `libomp`, which does not ship with them.
Without it `import xgboost` fails outright. **Linux and Windows users skip
this.**

```bash
brew install libomp                    # try this first
bash scripts/setup_libomp_macos.sh     # fallback, once that script is in the repo
```

The fallback exists because on the lead's machine Homebrew is owned by a
different user account and refuses to run. It installs `libomp` inside `.venv`
only. Re-run it after upgrading xgboost or lightgbm.

### Environment facts

| Question | Answer |
|---|---|
| Python version? | **3.11 or newer.** Built on 3.13.3. On 3.10 or older the install fails. |
| `requirements.txt` or `requirements.lock.txt`? | `requirements.txt` day to day. `requirements.lock.txt` (exact pins) when generating results you will write up, so everyone's numbers match. |
| Will the lock file work on Windows? | Yes — it contains no platform-specific packages. If a pin fails to build, use `requirements.txt` and say so in the team chat. |
| Can I commit `.venv/`? | **Never.** Hundreds of MB of OS-specific binaries. Already ignored. |
| Do we all need MIMIC-IV? | No. Only whoever runs the final experiments. Everyone else develops on the synthetic cohort. |
| Tests fail because modules are missing | Expected until the relevant round has merged. Run only the tests for code that is already in. |

---

## 2. Rules that prevent painful conflicts

1. **Only Anshu commits anything inside `results/`.** Result files are large
   and regenerated wholesale; two people committing them produces conflicts in
   1 MB CSVs that cannot be resolved by hand.
2. **`results/**/cells.jsonl` stays gitignored.** It is a 2.2 MB append-only
   log that conflicts on every run. Its rows are in the tracked `results.csv`
   and `results.parquet`, so nothing is lost.
3. **Stay inside your own files.** Need a change in someone else's module?
   Message them or open an issue. Do not edit it on your branch.
4. **Never push to `main` directly.** Branch → push → Pull Request → one
   review → merge.
5. **Never run `git push --all`.** Push the branch you mean, by name.

---

## 3. Status

| Round | Who | What | Status |
|---|---|---|---|
| 1 | Anshu | Repository scaffold | **done** |
| 2 | Ankit | `utils.py` — the foundation others import | next |
| 3 | Aditya, Ankit, Maharshi *(parallel)* | Core modules | waiting |
| 4 | Anshu | Integration layer | waiting |
| 5 | Everyone *(parallel)* | Scripts and tests | waiting |
| 6 | Anshu | Full documentation and results | waiting |

Rounds follow the import dependencies, so the repository is never broken
mid-history.

---

## 4. Getting set up — everyone except Anshu

Do this once.

```bash
cd ~/Desktop
git clone https://github.com/AnshuGupta9931/icu-missingness-study.git
cd icu-missingness-study

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Ask Anshu for `icumiss-source.zip`, unzip it to `~/Desktop/icumiss-source/`,
and **do not put it inside the repository folder**.

---

## 5. The pattern every round follows

```bash
# 1. start from the latest main
cd ~/Desktop/icu-missingness-study
git checkout main
git pull

# 2. make your branch
git checkout -b <branch-name>

# 3. copy YOUR files from the reference copy
cp ~/Desktop/icumiss-source/<your files> <same path in repo>

# 4. read what you just copied, then commit
git add <your files>
git commit -m "<message>"

# 5. push and open a Pull Request on GitHub
git push -u origin <branch-name>
```

---

## 6. Who owns what, and the exact commands

### Round 2 — Ankit Kumar Anand

```bash
cd ~/Desktop/icu-missingness-study && git checkout main && git pull
git checkout -b feat/numeric-utilities

cp ~/Desktop/icumiss-source/src/icumiss/utils.py src/icumiss/utils.py

git add src/icumiss/utils.py
git commit -m "feat: add numeric helpers for controlled masking

Numerically stable sigmoid, a bisection solver for hitting an exact event
rate, logistic noise, and noisy top-k selection. Adding logistic noise to a
score and taking the top k is equivalent to sampling k cells without
replacement under a logit model on that score, which is how an exact
missingness level is achieved."

git push -u origin feat/numeric-utilities
```

### Round 3a — Aditya Chaudhary

| File | What it does |
|---|---|
| `src/icumiss/data/__init__.py` | Dataset registry — picks synthetic or MIMIC |
| `src/icumiss/data/cohort.py` | Cohort container and its validation rules |
| `src/icumiss/data/synthetic.py` | Generates 8,000 realistic fake ICU patients |
| `src/icumiss/data/mimic.py` | Reads MIMIC-IV: streams the huge tables, builds the 24-hour cohort |

```bash
cd ~/Desktop/icu-missingness-study && git checkout main && git pull
git checkout -b feat/data-layer

mkdir -p src/icumiss/data
cp ~/Desktop/icumiss-source/src/icumiss/data/*.py src/icumiss/data/

git add src/icumiss/data/
git commit -m "feat: add cohort container, synthetic generator and MIMIC-IV extraction

Cohort validates on construction that the matrix is numeric, the outcome
binary, and the protected columns complete, since MAR conditioning depends on
that last property holding.

The synthetic generator produces a cohort with informative natural missingness
and nonlinear risk, so the pipeline can be developed before credentialed
MIMIC-IV access is granted.

MIMIC-IV extraction streams chartevents and labevents in chunks, filters to
the 24-hour window, drops physiologically impossible values, and caches the
result. Two leakage guards are enforced: patients who die inside the
observation window are excluded, and nothing recorded after the window is
read."

git push -u origin feat/data-layer
```

**Why this is a real job:** `mimic.py` is the only part of the project that has
never run against real data, because nobody has access yet. When access
arrives, this is where the debugging happens. The itemid check matters — MIMIC
uses numeric codes for each measurement and they change between versions, so a
wrong code produces confidently wrong results.

### Round 3b — Ankit Kumar Anand

| File | What it does |
|---|---|
| `src/icumiss/missingness.py` | **The scientific core** — imposes MCAR, MAR and MNAR |
| `src/icumiss/splits.py` | Leakage-safe stratified train/validation/test splitting |

```bash
cd ~/Desktop/icu-missingness-study && git checkout main && git pull
git checkout -b feat/missingness-mechanisms

cp ~/Desktop/icumiss-source/src/icumiss/missingness.py src/icumiss/
cp ~/Desktop/icumiss-source/src/icumiss/splits.py src/icumiss/

git add src/icumiss/missingness.py src/icumiss/splits.py
git commit -m "feat: add controlled MCAR/MAR/MNAR masking and leakage-safe splits

Each mechanism is imposed by masking values that were originally recorded, so
the condition is known by construction and the masked values provide the
ground truth that reconstruction is scored against. Only observed cells are
eligible, so ground truth is never invented where none existed. Protected
demographic columns are never masked, which is what makes the MAR scenario
genuinely missing-at-random given observed data.

Splitting happens before masking and before any imputer is fitted."

git push -u origin feat/missingness-mechanisms
```

**Why this is a real job:** this is what an examiner will question hardest. Be
ready to explain why masking is done at all (you cannot grade a guess without
the answer key), how an exact missingness level is achieved, and why MAR
conditions only on complete columns.

### Round 3c — Annepu Maharshi

| File | What it does |
|---|---|
| `src/icumiss/imputation.py` | The four methods: median, KNN, MICE, MissForest |
| `src/icumiss/models.py` | The four classifiers |
| `src/icumiss/metrics.py` | MAE/RMSE/R²/adjusted R², plus AUROC/AUPRC/F1/log loss/Brier |

```bash
cd ~/Desktop/icu-missingness-study && git checkout main && git pull
git checkout -b feat/imputation-models-metrics

cp ~/Desktop/icumiss-source/src/icumiss/imputation.py src/icumiss/
cp ~/Desktop/icumiss-source/src/icumiss/models.py src/icumiss/
cp ~/Desktop/icumiss-source/src/icumiss/metrics.py src/icumiss/

git add src/icumiss/imputation.py src/icumiss/models.py src/icumiss/metrics.py
git commit -m "feat: add the four imputers, four classifiers and both metric stages

Every imputer is fitted on training rows only and applied unchanged to
validation and test. The outcome is never shown to an imputer, and imputation
fills gaps without modifying recorded measurements.

Classifiers are trained on imputed matrices and reject NaN, so native
missing-value handling cannot confound the imputation comparison.

Imputation metrics are scored only on artificially masked cells, in both
original and standardised units, and the decision threshold for F1 and
precision is selected on validation rather than fixed at 0.5."

git push -u origin feat/imputation-models-metrics
```

Then, after that merges:

```bash
git checkout main && git pull
git checkout -b feat/analysis-tables-figures

mkdir -p src/icumiss/analysis
cp ~/Desktop/icumiss-source/src/icumiss/analysis/*.py src/icumiss/analysis/

git add src/icumiss/analysis/
git commit -m "feat: add summary tables and the figure set

Ten tables including reconstruction_vs_downstream, which ranks the imputers
twice per condition and compares the two rankings - the project's central
analysis. Near-tied conditions are flagged against seed-level noise so they
are not over-read.

Figures use a colourblind-safe palette with distinct markers and line styles,
so series identity never rests on colour alone."

git push -u origin feat/analysis-tables-figures
```

**Why this is a real job:** `tables.py` computes the project's headline answer,
and your figures are the results chapter.

### Round 4 — Anshu Gupta

```bash
cd ~/Desktop/icu-missingness-study && git checkout main && git pull
git checkout -b feat/experiment-runner

cp ~/Desktop/icumiss-source/src/icumiss/experiment.py src/icumiss/
cp ~/Desktop/icumiss-source/src/icumiss/runner.py src/icumiss/
mkdir -p configs && cp ~/Desktop/icumiss-source/configs/*.yaml configs/

git add src/icumiss/experiment.py src/icumiss/runner.py configs/
git commit -m "feat: add cell execution and resumable grid orchestration

One cell runs mask, impute, classify, score. The four classifiers share the
imputed matrices, which cuts the expensive MissForest stage fourfold and
guarantees they are compared on identical data.

The grid is resumable and fail-soft: finished cells are logged as they land,
and a failing cell is recorded with its traceback rather than killing the run.
Worker threads are capped so the OpenMP pools inside the boosters do not
oversubscribe the machine."

git push -u origin feat/experiment-runner
```

### Round 5 — everyone, in parallel

Each person branches from an updated `main` and adds their scripts and tests.

```bash
# Anshu
git checkout -b feat/cli-and-orchestration-tests
mkdir -p scripts tests
cp ~/Desktop/icumiss-source/scripts/run_experiments.py scripts/
cp ~/Desktop/icumiss-source/scripts/setup_libomp_macos.sh scripts/
cp ~/Desktop/icumiss-source/tests/test_experiment.py tests/
git add scripts/run_experiments.py scripts/setup_libomp_macos.sh tests/test_experiment.py
git commit -m "feat: add the experiment CLI and orchestration tests"

# Aditya
git checkout -b feat/data-scripts-and-tests
mkdir -p scripts tests
cp ~/Desktop/icumiss-source/scripts/profile_cohort.py scripts/
cp ~/Desktop/icumiss-source/scripts/verify_mimic.py scripts/
cp ~/Desktop/icumiss-source/tests/test_mimic.py tests/
git add scripts/profile_cohort.py scripts/verify_mimic.py tests/test_mimic.py
git commit -m "feat: add cohort profiling, MIMIC verification and extraction tests"

# Ankit
git checkout -b feat/design-verification-and-leakage-tests
mkdir -p scripts tests
cp ~/Desktop/icumiss-source/scripts/verify_design.py scripts/
cp ~/Desktop/icumiss-source/tests/test_missingness.py tests/
cp ~/Desktop/icumiss-source/tests/test_leakage.py tests/
git add scripts/verify_design.py tests/test_missingness.py tests/test_leakage.py
git commit -m "feat: add mechanism verification and the leakage guard suite"

# Maharshi
git checkout -b feat/report-cli-and-metric-tests
mkdir -p scripts tests
cp ~/Desktop/icumiss-source/scripts/make_report.py scripts/
cp ~/Desktop/icumiss-source/tests/test_metrics.py tests/
git add scripts/make_report.py tests/test_metrics.py
git commit -m "feat: add the reporting CLI and metric tests"
```

Each then pushes and opens a PR:

```bash
git push -u origin <your-branch-name>
```

Test counts, once everything has merged: 40 masking, 21 leakage, 16 MIMIC,
14 metrics, 11 orchestration — **102 total**.

### Round 6 — Anshu Gupta

```bash
cd ~/Desktop/icu-missingness-study && git checkout main && git pull
git checkout -b docs/documentation-and-results

cp ~/Desktop/icumiss-source/README.md .
cp ~/Desktop/icumiss-source/START_HERE.md .
cp ~/Desktop/icumiss-source/RESULTS.md .
cp ~/Desktop/icumiss-source/CHANGELOG.md .
cp -R ~/Desktop/icumiss-source/results .

.venv/bin/python -m pytest      # expect 102 passed

git add README.md START_HERE.md RESULTS.md CHANGELOG.md results/
git commit -m "docs: add full documentation and the factorial result set

264 cells, 2,112 result rows, zero failures. The best-reconstructing imputer
was also the best predictor in only 59.5% of conditions, and 46.4% under MNAR,
which is the dissociation the study was designed to detect."

git push -u origin docs/documentation-and-results
```

---

## 7. Day-to-day workflow after setup

```bash
git checkout main
git pull
git checkout -b feat/short-description

# ... make changes ...
.venv/bin/python -m pytest        # must pass before pushing
git add <only your files>
git commit -m "feat: what changed and why"
git push -u origin feat/short-description
```

Open a Pull Request, get one review, merge.

**Prefixes:** `feat:` new capability · `fix:` bug fix · `docs:` documentation ·
`test:` tests · `chore:` tooling and config · `refactor:` no behaviour change.

**Update `CHANGELOG.md` whenever you change behaviour.** The proposal
(section 18) requires a change log with date, author, what changed and why.

---

## 8. If something goes wrong

**`ModuleNotFoundError: No module named 'icumiss'`**
Run commands as `.venv/bin/python scripts/...` from the repository root.

**Tests fail with import errors**
Expected until the round providing that module has merged. Run only the tests
for code already in `main`.

**`Library not loaded: @rpath/libomp.dylib`** (macOS)
Run `bash scripts/setup_libomp_macos.sh`.

**A dependency will not install**
Check `python3 --version` is 3.11+. Then try `requirements.txt` instead of the
lock file.

**`git pull` refuses: "untracked working tree files would be overwritten"**
You have a local copy of a file that a merged PR is now bringing in. Delete
your local copy and pull again.

**You committed to the wrong branch**
```bash
git reset --soft HEAD~1
git stash
git checkout -b correct-branch-name
git stash pop
```

**You accidentally committed `.venv/` or patient data**
Tell Anshu **before pushing**. Once it is in shared history it is painful to
remove.
