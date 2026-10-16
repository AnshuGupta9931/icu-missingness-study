#!/usr/bin/env python
"""Check a local MIMIC-IV copy before extracting anything from it.

    python scripts/verify_mimic.py --root /path/to/mimic-iv

Does three things:

1. confirms the expected files are present;
2. resolves every configured itemid against `d_items` / `d_labitems` and prints
   the label it maps to - itemids change between MIMIC versions, and a silently
   wrong itemid produces a confidently wrong result set;
3. optionally builds the cohort and reports its size and prevalence.

Keep the itemid report with any result set produced from this cohort.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from icumiss.data import mimic  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="directory containing hosp/ and icu/")
    parser.add_argument("--out", default="results/mimic_itemid_report.csv")
    parser.add_argument("--build", action="store_true", help="also extract the cohort (slow, streams the big tables)")
    parser.add_argument("--observation-hours", type=float, default=24.0)
    parser.add_argument("--aggregation", default="mean")
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    try:
        mimic.verify_layout(root)
    except FileNotFoundError as exc:
        print(exc)
        return 1
    print(f"file layout OK under {root}\n")

    report = mimic.verify_itemids(root)
    if report.empty:
        print("could not read d_items / d_labitems, so itemids were not checked")
    else:
        pd.set_option("display.width", 160)
        pd.set_option("display.max_rows", 200)
        print(report.to_string(index=False))

        missing = report[~report["found"]]
        print()
        if missing.empty:
            print(f"all {len(report)} itemids resolved against the MIMIC dictionaries")
        else:
            print(
                f"{len(missing)} itemid(s) did NOT resolve. Check these against your "
                "MIMIC version and update CHART_SPECS / LAB_SPECS in "
                "src/icumiss/data/mimic.py before running experiments:"
            )
            for _, row in missing.iterrows():
                print(f"  {row['variable']:20s} itemid {row['itemid']} ({row['table']})")

        print(
            "\nRead the labels above and confirm each one is the quantity you "
            "intended. A resolved itemid pointing at the wrong measurement is "
            "the failure mode this check exists to catch."
        )
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.out, index=False)
        print(f"\nreport written to {args.out}")

    if args.build:
        print("\nextracting cohort (streams chartevents and labevents; this takes a while)...")
        cohort = mimic.load(
            root=root,
            observation_hours=args.observation_hours,
            aggregation=args.aggregation,
        )
        print(f"\npatients             {cohort.n_samples:,}")
        print(f"features             {cohort.n_features}")
        print(f"mortality prevalence {cohort.prevalence:.3%}")
        print(f"missing cells        {cohort.X.isna().to_numpy().mean():.3%}")
        counts = cohort.meta.get("cohort_counts", {})
        for stage, n in counts.items():
            print(f"  {stage:34s} {n:,}")
        dropped = cohort.meta.get("dropped_sparse_features")
        if dropped:
            print(f"\ndropped as too sparse: {dropped}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
