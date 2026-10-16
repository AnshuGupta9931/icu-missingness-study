"""icumiss - controlled study of missingness mechanisms, imputation and ICU mortality prediction.

Pipeline stage order is fixed and leakage-safe:

    cohort -> split -> mask -> impute (fit on train only) -> classify -> evaluate

See README.md for the experimental design and `configs/` for run definitions.
"""

__version__ = "0.1.0"
