#!/usr/bin/env python
"""Run an experiment grid from a config file.

    python scripts/run_experiments.py --config configs/pilot.yaml
    python scripts/run_experiments.py --config configs/full_synthetic.yaml
    python scripts/run_experiments.py --config configs/full_mimic.yaml --n-jobs 8

Runs are resumable: re-running the same config continues from where it stopped.
Pass --fresh to discard previous cells and start over.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from icumiss.analysis import plots, tables  # noqa: E402
from icumiss.config import Config  # noqa: E402
from icumiss.runner import run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="path to a YAML run config")
    parser.add_argument("--run-name", default=None, help="override run_name (and so the output folder)")
    parser.add_argument("--n-jobs", type=int, default=None, help="override compute.n_jobs")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="override experiment.seeds")
    parser.add_argument("--levels", type=float, nargs="+", default=None, help="override experiment.levels")
    parser.add_argument("--fresh", action="store_true", help="ignore previously completed cells")
    parser.add_argument("--no-report", action="store_true", help="skip tables and figures")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.run_name:
        cfg.run_name = args.run_name
    if args.n_jobs:
        cfg.compute.n_jobs = args.n_jobs
    if args.seeds:
        cfg.experiment.seeds = args.seeds
    if args.levels:
        cfg.experiment.levels = args.levels
    cfg.validate()

    print(
        f"Run '{cfg.run_name}': {cfg.n_cells()} cells -> {cfg.n_result_rows()} result rows\n"
        f"  dataset     {cfg.dataset.name}\n"
        f"  mechanisms  {cfg.experiment.mechanisms}\n"
        f"  levels      {cfg.experiment.levels}\n"
        f"  imputers    {cfg.experiment.imputers}\n"
        f"  classifiers {cfg.experiment.classifiers}\n"
        f"  seeds       {cfg.experiment.seeds}",
        file=sys.stderr,
    )

    outcome = run(cfg, resume=not args.fresh)
    out_dir = Path(outcome["output_dir"])
    frame = outcome["results"]

    if not args.no_report:
        written_tables = tables.write_all(frame, out_dir / "tables")
        written_figures = plots.write_all(
            frame,
            out_dir / "figures",
            profile=outcome["cohort"].missingness_profile(),
            classifiers=cfg.experiment.classifiers,
        )
        skipped = {
            name: reason
            for name, reason in {**written_tables, **written_figures}.items()
            if str(reason).startswith("skipped")
        }
        print(
            f"\n{len(written_tables)} tables -> {out_dir / 'tables'}"
            f"\n{len(written_figures)} figures -> {out_dir / 'figures'}",
            file=sys.stderr,
        )
        for name, reason in skipped.items():
            print(f"  {name}: {reason}", file=sys.stderr)

        print("\n" + tables.headline(frame).to_string(index=False), file=sys.stderr)

    failed = int((frame["status"] == "failed").sum())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
