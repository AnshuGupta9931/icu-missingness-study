# ICU Missingness Study

**Impact of Missing Data Mechanisms and Imputation Strategies on
Machine-Learning-Based ICU Mortality Prediction**

A controlled experiment connecting three stages that are usually studied
separately:

```
missingness mechanism  ->  imputation  ->  ICU mortality prediction
```

The question it exists to answer: **does the method that reconstructs missing
values most accurately also give the best mortality prediction?**

## Design

| Factor | Levels |
|---|---|
| Missingness mechanism | MCAR, MAR, MNAR |
| Missingness level | 10% to 70%, in steps of 10 |
| Imputation method | median, KNN, MICE, MissForest |
| Classifier | logistic regression, XGBoost, CatBoost, LightGBM |
| Random seeds | 0, 1, 2 |

Primary dataset: **MIMIC-IV** (credentialed PhysioNet access required).
Patient-level data is never committed to this repository.

## Team

| Name | Area |
|---|---|
| Anshu Gupta | Team lead — integration, orchestration, documentation |
| Aditya Chaudhary | Data engineering — cohort building, MIMIC-IV extraction |
| Ankit Kumar Anand | Experimental design — missingness mechanisms, splitting |
| Annepu Maharshi | Methods and analysis — imputation, classifiers, metrics |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest
```

macOS users also need the OpenMP runtime for XGBoost and LightGBM:
`brew install libomp`.

## Status

Repository scaffold. Modules are being added by their owners — see
`TEAM_WORKFLOW.md` for the ownership map and commit plan.
