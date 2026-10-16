#!/usr/bin/env python
"""Rebuild the tables and figures for a finished run.

    python scripts/make_report.py --run results/full_synthetic

Reads `results.parquet` (or `cells.jsonl` if the parquet is absent, so a run
that is still in flight can be inspected) and regenerates everything under
`tables/` and `figures/`. Separate from the runner so the analysis can be
changed and re-applied without re-running a single model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from icumiss.analysis import plots, tables  # noqa: E402


def load_results(run_dir: Path) -> pd.DataFrame:
    parquet = run_dir / "results.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)

    csv = run_dir / "results.csv"
    if csv.exists():
        return pd.read_csv(csv)

    jsonl = run_dir / "cells.jsonl"
    if jsonl.exists():
        from icumiss.runner import _load_results

        frame = _load_results(jsonl)
        print(f"reading partial results from {jsonl} ({len(frame)} rows)", file=sys.stderr)
        return frame

    raise FileNotFoundError(f"no results found in {run_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="a run directory under results/")
    parser.add_argument("--classifiers", nargs="*", default=None,
                        help="also render a per-classifier AUROC figure for each of these")
    args = parser.parse_args()

    run_dir = Path(args.run)
    frame = load_results(run_dir)
    ok = frame[frame["status"] == "ok"] if "status" in frame.columns else frame
    print(f"{len(frame)} rows ({len(ok)} ok) from {run_dir}", file=sys.stderr)

    written_tables = tables.write_all(frame, run_dir / "tables")

    profile_path = run_dir / "natural_missingness_profile.csv"
    profile = pd.read_csv(profile_path) if profile_path.exists() else None

    classifiers = args.classifiers
    if classifiers is None:
        classifiers = sorted(ok["classifier"].unique()) if "classifier" in ok else []

    written_figures = plots.write_all(frame, run_dir / "figures", profile=profile, classifiers=classifiers)

    for group, written in (("tables", written_tables), ("figures", written_figures)):
        done = [n for n, r in written.items() if not str(r).startswith("skipped")]
        skipped = {n: r for n, r in written.items() if str(r).startswith("skipped")}
        print(f"\n{len(done)} {group} -> {run_dir / group}", file=sys.stderr)
        for name, reason in skipped.items():
            print(f"  {name}: {reason}", file=sys.stderr)

    print("\n" + tables.headline(frame).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
