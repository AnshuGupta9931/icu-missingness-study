#!/usr/bin/env python
"""Describe the cohort and its naturally occurring missingness.

    python scripts/profile_cohort.py --config configs/pilot.yaml

This covers the proposal's first objective - "study the characteristics and
patterns of missing data in the selected ICU dataset" - and is the sanity check
to run before any experiment. It reports cohort size, outcome prevalence,
per-column missingness, and whether a column's missingness tracks the outcome.

That last column is the informative one. A large `mortality_gap_when_missing`
says the dataset's own missingness carries outcome information, so it is not
MCAR. In ICU data the gap is usually negative for tests ordered on suspicion
(lactate, arterial pH): the patients who never had the test drawn were the ones
who did not look sick enough to warrant it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from icumiss.config import Config  # noqa: E402
from icumiss.data import load_cohort  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=None, help="write the profile to this CSV")
    parser.add_argument("--figure", default=None, help="write the profile figure to this PNG")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    cohort = load_cohort(cfg.dataset.name, cfg.dataset.params, seed=0)
    profile = cohort.missingness_profile()

    pd.set_option("display.width", 160)
    print(f"dataset              {cfg.dataset.name}")
    print(f"patients             {cohort.n_samples:,}")
    print(f"features             {cohort.n_features} "
          f"({len(cohort.maskable)} maskable, {len(cohort.protected)} protected)")
    print(f"mortality prevalence {cohort.prevalence:.3%}  "
          f"({int(cohort.y.sum()):,} deaths)")
    print(f"missing cells        {cohort.X.isna().to_numpy().mean():.3%} of the matrix")
    print(f"complete rows        {(~cohort.X.isna().any(axis=1)).mean():.3%}")

    counts = cohort.meta.get("cohort_counts")
    if counts:
        print("\ncohort construction")
        for stage, n in counts.items():
            print(f"  {stage:34s} {n:,}")

    print("\nper-column missingness")
    shown = profile.copy()
    shown["frac_missing"] = (shown["frac_missing"] * 100).round(2)
    shown["mortality_gap_when_missing"] = shown["mortality_gap_when_missing"].round(4)
    print(shown.to_string(index=False))

    informative = profile[profile["mortality_gap_when_missing"].abs() > 0.02]
    if not informative.empty:
        print(
            f"\n{len(informative)} column(s) show a mortality gap above 2 points "
            "between missing and observed, so the cohort's own missingness is "
            "not plausibly MCAR:"
        )
        for _, row in informative.iterrows():
            direction = "lower" if row["mortality_gap_when_missing"] < 0 else "higher"
            print(
                f"  {row['column']:22s} {row['frac_missing']:.1%} missing, "
                f"mortality {abs(row['mortality_gap_when_missing']):.1%} {direction} "
                "when the value is absent"
            )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        profile.to_csv(args.out, index=False)
        print(f"\nprofile written to {args.out}")

    if args.figure:
        from icumiss.analysis.plots import natural_missingness

        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        natural_missingness(profile, args.figure)
        print(f"figure written to {args.figure}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
