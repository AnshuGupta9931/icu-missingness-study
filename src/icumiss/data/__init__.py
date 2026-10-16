"""Dataset registry.

A dataset is anything that can return a `Cohort`. Keeping the registry behind
one function means the rest of the pipeline never knows or cares whether it is
running on MIMIC-IV, on the synthetic cohort, or on eICU later on.
"""

from __future__ import annotations

from typing import Any

from .cohort import Cohort


def load_cohort(name: str, params: dict[str, Any] | None = None, seed: int = 0) -> Cohort:
    """Build a cohort by registry name.

    Parameters
    ----------
    name
        "synthetic" or "mimic".
    params
        Dataset-specific keyword arguments, taken straight from the run config.
    seed
        Used only by generators that need randomness; real datasets ignore it.
    """
    params = dict(params or {})
    key = name.lower().replace("-", "").replace("_", "")

    if key == "synthetic":
        from . import synthetic

        params.setdefault("seed", seed)
        return synthetic.generate(**params)

    if key in ("mimic", "mimiciv", "mimic4"):
        from . import mimic

        if "root" not in params:
            raise ValueError(
                "dataset.params.root must point at your local MIMIC-IV "
                "distribution (the directory containing hosp/ and icu/)"
            )
        return mimic.load(**params)

    raise ValueError(f"unknown dataset {name!r}; expected 'synthetic' or 'mimic'")


__all__ = ["Cohort", "load_cohort"]
