#!/usr/bin/env python
"""Check that the three missingness mechanisms are actually distinct.

    python scripts/verify_design.py --config configs/pilot.yaml

Run this before trusting any result set, and again after changing anything in
`missingness.py`. It measures, for each mechanism, how strongly the masking
indicator correlates with

* the value that was hidden (``corr_with_hidden_value``), and
* the observed conditioning index built from the protected columns
  (``corr_with_conditioner``).

The signature to expect:

    MCAR   both near zero               missingness depends on nothing
    MAR    conditioner correlation high  depends on observed data only
    MNAR   hidden-value correlation high depends on the hidden value itself

It also confirms the realised missingness level matches the request, which is
what makes the x-axis of every results figure mean what it says.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from icumiss.config import Config  # noqa: E402
from icumiss.runner import verify_mechanisms  # noqa: E402

TOLERANCE = 0.08  # "near zero" for a correlation that should not exist
EXPECTED_DEPENDENCE = 0.25  # a dependence the design does intend


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--levels", type=float, nargs="+", default=[0.1, 0.3, 0.7])
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.experiment.mechanisms = ["mcar", "mar", "mnar"]

    frames = []
    for level in args.levels:
        table = verify_mechanisms(cfg, level=level)
        table.insert(0, "requested_level", level)
        frames.append(table)
    report = pd.concat(frames, ignore_index=True)

    pd.set_option("display.width", 150)
    pd.set_option("display.float_format", lambda v: f"{v: .4f}")
    print(report.to_string(index=False))

    problems = []
    for _, row in report.iterrows():
        mechanism, level = row["mechanism"], row["requested_level"]
        value_corr = abs(row["corr_with_hidden_value"])
        cond_corr = abs(row["corr_with_conditioner"])

        if abs(row["realised_level"] - level) > 0.02:
            problems.append(
                f"{mechanism} at {level:.0%}: realised level {row['realised_level']:.3f} "
                "differs from the request by more than 2 points"
            )
        if mechanism == "mcar" and (value_corr > TOLERANCE or cond_corr > TOLERANCE):
            problems.append(
                f"MCAR at {level:.0%}: masking is not independent "
                f"(value {value_corr:.3f}, conditioner {cond_corr:.3f})"
            )
        if mechanism == "mar":
            if cond_corr < EXPECTED_DEPENDENCE:
                problems.append(
                    f"MAR at {level:.0%}: conditioner dependence only {cond_corr:.3f}; "
                    "raise missingness.mar_strength"
                )
            if value_corr > 0.20:
                problems.append(
                    f"MAR at {level:.0%}: masking tracks the hidden value "
                    f"({value_corr:.3f}); MAR should not depend on it directly"
                )
        if mechanism == "mnar" and value_corr < EXPECTED_DEPENDENCE:
            problems.append(
                f"MNAR at {level:.0%}: hidden-value dependence only {value_corr:.3f}; "
                "raise missingness.mnar_strength"
            )

    print()
    if problems:
        print("DESIGN CHECK FAILED")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("DESIGN CHECK PASSED - the three mechanisms are distinct and levels are exact.")
    print(
        "\nNote: under MAR the hidden-value correlation is small but not exactly zero, "
        "because the conditioning variables (age, admission type) are themselves "
        "correlated with the clinical values. That is a property of MAR on real "
        "clinical data, not a defect in the masking."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
