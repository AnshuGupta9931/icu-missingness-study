"""Run configuration: typed objects, YAML loading and provenance capture.

Every run writes its resolved configuration back out next to the results, so a
result set can always be traced to the exact settings that produced it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

MECHANISMS = ("mcar", "mar", "mnar")
IMPUTERS = ("median", "knn", "mice", "missforest")
CLASSIFIERS = ("logreg", "xgboost", "catboost", "lightgbm")

# Sentinel mechanism for the reference cells that carry no artificial
# missingness. They still run an imputer, because the cohort's own natural
# missingness has to be filled before a classifier can be fitted.
NO_MECHANISM = "none"


@dataclass
class DatasetConfig:
    name: str = "synthetic"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitConfig:
    test_size: float = 0.20
    val_size: float = 0.15  # fraction of the whole cohort, carved out of the non-test part
    stratify: bool = True


@dataclass
class MissingnessConfig:
    # Strength of the dependence between the masking score and its driver.
    # 0 collapses MAR/MNAR to MCAR; larger values sharpen the dependence.
    mar_strength: float = 2.0
    mnar_strength: float = 2.0
    # "high": large values are more likely to be missing (the documented
    # sensitivity-analysis rule); "low" flips the direction.
    mnar_direction: str = "high"
    # Safety net so an imputer is never handed an all-missing training column.
    min_observed_per_column: int = 10


@dataclass
class ExperimentConfig:
    mechanisms: list[str] = field(default_factory=lambda: list(MECHANISMS))
    levels: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    imputers: list[str] = field(default_factory=lambda: list(IMPUTERS))
    classifiers: list[str] = field(default_factory=lambda: list(CLASSIFIERS))
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    # Adds a reference condition with no artificial missingness, so every curve
    # has an upper anchor to be read against.
    include_baseline: bool = True


@dataclass
class ComputeConfig:
    n_jobs: int = 4
    # Threads handed to each model inside a parallel worker. Kept at 1 by
    # default to avoid oversubscribing the machine.
    inner_threads: int = 1
    cache_imputations: bool = True


@dataclass
class Config:
    run_name: str = "run"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    missingness: MissingnessConfig = field(default_factory=MissingnessConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    imputer_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    output_dir: str = "results"

    # ---- construction -------------------------------------------------
    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        raw = dict(raw or {})
        cfg = cls(
            run_name=raw.get("run_name", "run"),
            dataset=DatasetConfig(**(raw.get("dataset") or {})),
            split=SplitConfig(**(raw.get("split") or {})),
            missingness=MissingnessConfig(**(raw.get("missingness") or {})),
            experiment=ExperimentConfig(**(raw.get("experiment") or {})),
            imputer_params=raw.get("imputer_params") or {},
            model_params=raw.get("model_params") or {},
            compute=ComputeConfig(**(raw.get("compute") or {})),
            output_dir=raw.get("output_dir", "results"),
        )
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh))

    def validate(self) -> None:
        bad = set(self.experiment.mechanisms) - set(MECHANISMS)
        if bad:
            raise ValueError(f"unknown mechanism(s) {sorted(bad)}; expected {MECHANISMS}")
        bad = set(self.experiment.imputers) - set(IMPUTERS)
        if bad:
            raise ValueError(f"unknown imputer(s) {sorted(bad)}; expected {IMPUTERS}")
        bad = set(self.experiment.classifiers) - set(CLASSIFIERS)
        if bad:
            raise ValueError(f"unknown classifier(s) {sorted(bad)}; expected {CLASSIFIERS}")
        for lv in self.experiment.levels:
            if not 0.0 < lv < 1.0:
                raise ValueError(f"missingness level must be in (0, 1); got {lv}")
        if self.missingness.mnar_direction not in ("high", "low"):
            raise ValueError("missingness.mnar_direction must be 'high' or 'low'")
        if not self.experiment.seeds:
            raise ValueError("at least one seed is required")
        if self.split.test_size + self.split.val_size >= 0.9:
            raise ValueError("test_size + val_size leaves too little training data")

    # ---- serialisation ------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        """Short stable hash of the resolved config, used in the run id."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:10]

    def n_cells(self) -> int:
        """Number of (mechanism, level, imputer, seed) cells, baseline included."""
        e = self.experiment
        n = len(e.mechanisms) * len(e.levels) * len(e.imputers) * len(e.seeds)
        if e.include_baseline:
            # One reference cell per imputer and seed, at level 0.
            n += len(e.imputers) * len(e.seeds)
        return n

    def n_result_rows(self) -> int:
        """Result rows written: one per cell, classifier and evaluation split."""
        return self.n_cells() * len(self.experiment.classifiers) * 2


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).resolve().parents[2],
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def library_versions() -> dict[str, str]:
    """Versions of every library whose behaviour can change a result."""
    import importlib

    names = [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "xgboost",
        "lightgbm",
        "catboost",
        "joblib",
    ]
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for name in names:
        try:
            versions[name] = getattr(importlib.import_module(name), "__version__", "unknown")
        except ImportError:
            versions[name] = "not installed"
    return versions


def build_manifest(cfg: Config, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Provenance record written beside every result set."""
    manifest = {
        "run_name": cfg.run_name,
        "config_fingerprint": cfg.fingerprint(),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "libraries": library_versions(),
        "planned_cells": cfg.n_cells(),
        "planned_result_rows": cfg.n_result_rows(),
        "config": cfg.to_dict(),
    }
    if extra:
        manifest.update(extra)
    return manifest
