"""Summary tables, including the one the project exists to produce.

Everything here reads the tidy result frame and aggregates **across seeds**,
reporting a mean and a standard deviation, because a single-seed difference
between two imputation methods is usually noise.

`reconstruction_vs_downstream` is the central analysis: it asks whether the
method that reconstructs masked values most accurately is also the method that
yields the best mortality prediction. A negative Spearman correlation between
reconstruction error and AUROC would mean "lower error, better prediction"; a
correlation near zero means reconstruction accuracy does not transfer, which is
the finding the proposal is set up to detect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Lower is better for these; higher is better for everything else reported.
LOWER_IS_BETTER = {
    "imp_mae_raw",
    "imp_rmse_raw",
    "imp_mae_z",
    "imp_rmse_z",
    "log_loss",
    "brier",
}

CLASSIFICATION_METRICS = ["auroc", "auprc", "f1", "precision", "recall", "log_loss", "brier"]
IMPUTATION_METRICS = ["imp_rmse_z", "imp_mae_z", "imp_r2_z", "imp_rmse_raw", "imp_mae_raw", "imp_r2_raw", "imp_adj_r2_raw"]

GRID = ["mechanism", "level", "imputer", "classifier"]


def test_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Successful test-split rows: the basis of every reported number."""
    out = frame[(frame["split"] == "test") & (frame["status"] == "ok")].copy()
    if out.empty:
        raise ValueError("no successful test-split rows in the result frame")
    return out


def _agg(frame: pd.DataFrame, by: list[str], metrics: list[str]) -> pd.DataFrame:
    metrics = [m for m in metrics if m in frame.columns]
    grouped = frame.groupby(by, dropna=False)[metrics]
    mean = grouped.mean().add_suffix("_mean")
    std = grouped.std(ddof=1).add_suffix("_std")
    n = grouped.size().rename("n_seeds")
    return pd.concat([mean, std, n], axis=1).reset_index()


def classification_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Mortality-prediction metrics per experimental cell, averaged over seeds."""
    return _agg(test_rows(frame), GRID, CLASSIFICATION_METRICS)


def imputation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Reconstruction metrics per condition.

    Collapsed over classifiers: the imputation happens before any classifier is
    fitted, so these numbers are identical across the four of them and
    averaging over that axis would only fake extra precision.
    """
    rows = test_rows(frame)
    rows = rows[rows["classifier"] == rows["classifier"].iloc[0]]
    return _agg(rows, ["mechanism", "level", "imputer"], IMPUTATION_METRICS)


def degradation_table(frame: pd.DataFrame, metric: str = "auroc") -> pd.DataFrame:
    """Change in `metric` relative to the level-0 reference, per condition.

    Answers "how does increasing missingness affect performance?" directly: the
    reference is the same imputer and classifier with no artificial masking, so
    the difference isolates the effect of the induced missingness.
    """
    rows = test_rows(frame)
    baseline = rows[rows["mechanism"] == "none"]
    if baseline.empty:
        raise ValueError(
            "no level-0 reference rows found; re-run with "
            "experiment.include_baseline = true to enable this table"
        )

    ref = (
        baseline.groupby(["imputer", "classifier"])[metric]
        .mean()
        .rename(f"{metric}_baseline")
        .reset_index()
    )
    grid = rows[rows["mechanism"] != "none"]
    summary = _agg(grid, GRID, [metric]).merge(ref, on=["imputer", "classifier"], how="left")

    summary[f"{metric}_delta"] = summary[f"{metric}_mean"] - summary[f"{metric}_baseline"]
    denom = summary[f"{metric}_baseline"].replace(0, np.nan)
    summary[f"{metric}_pct_change"] = 100.0 * summary[f"{metric}_delta"] / denom
    return summary.sort_values(["mechanism", "classifier", "level", "imputer"])


def best_imputer_table(frame: pd.DataFrame, metric: str = "auroc") -> pd.DataFrame:
    """Winning imputer for each (mechanism, level, classifier).

    Used to answer whether the preferred strategy is stable or shifts with the
    experimental conditions. The margin over the runner-up is reported next to
    the winner, because a win of 0.001 AUROC is not a recommendation.
    """
    summary = _agg(test_rows(frame), GRID, [metric])
    summary = summary[summary["mechanism"] != "none"]
    ascending = metric in LOWER_IS_BETTER
    col = f"{metric}_mean"

    out = []
    for (mechanism, level, classifier), group in summary.groupby(
        ["mechanism", "level", "classifier"]
    ):
        group = group.sort_values(col, ascending=ascending)
        best, runner = group.iloc[0], (group.iloc[1] if len(group) > 1 else None)
        margin = np.nan if runner is None else abs(best[col] - runner[col])
        out.append(
            {
                "mechanism": mechanism,
                "level": level,
                "classifier": classifier,
                "metric": metric,
                "best_imputer": best["imputer"],
                "best_value": best[col],
                "best_seed_sd": best.get(f"{metric}_std", np.nan),
                "runner_up": None if runner is None else runner["imputer"],
                "margin": margin,
                # A margin inside one seed-level standard deviation is not a
                # result; flagging it here keeps that caveat attached to the row.
                "margin_exceeds_seed_sd": (
                    bool(margin > best.get(f"{metric}_std", np.nan))
                    if np.isfinite(margin) and np.isfinite(best.get(f"{metric}_std", np.nan))
                    else False
                ),
            }
        )
    return pd.DataFrame(out).sort_values(["mechanism", "classifier", "level"])


def reconstruction_vs_downstream(
    frame: pd.DataFrame,
    imputation_metric: str = "imp_rmse_z",
    downstream_metric: str = "auroc",
) -> pd.DataFrame:
    """Does better reconstruction give better prediction? One row per condition.

    Within each (mechanism, level, classifier) the four imputers are ranked
    twice - by reconstruction error and by downstream performance - and the two
    rankings are compared with Spearman's rho. Four points is few, so rho takes
    only five possible values and no single row is conclusive; the distribution
    across conditions, and `best_agrees`, are what carry the finding.
    """
    from scipy.stats import spearmanr

    rows = test_rows(frame)
    rows = rows[rows["mechanism"] != "none"]
    summary = _agg(rows, GRID, [imputation_metric, downstream_metric])

    imp_col, down_col = f"{imputation_metric}_mean", f"{downstream_metric}_mean"
    imp_ascending = imputation_metric in LOWER_IS_BETTER
    down_ascending = downstream_metric in LOWER_IS_BETTER

    out = []
    for (mechanism, level, classifier), group in summary.groupby(
        ["mechanism", "level", "classifier"]
    ):
        group = group.dropna(subset=[imp_col, down_col])
        if len(group) < 3:
            continue
        rho, pval = spearmanr(group[imp_col], group[down_col])
        best_imp = group.sort_values(imp_col, ascending=imp_ascending).iloc[0]["imputer"]
        best_down = group.sort_values(down_col, ascending=down_ascending).iloc[0]["imputer"]

        # How far apart the imputers are, against how much a result moves when
        # only the seed changes. Where the spread is the smaller of the two,
        # the ranking is noise and rho describes nothing - which is the usual
        # situation at low missingness, and the easiest way to over-read these
        # results.
        spread = float(group[down_col].max() - group[down_col].min())
        seed_noise = float(group.get(f"{downstream_metric}_std", pd.Series(dtype=float)).mean())

        out.append(
            {
                "mechanism": mechanism,
                "level": level,
                "classifier": classifier,
                "n_imputers": len(group),
                "spearman_rho": float(rho),
                "p_value": float(pval),
                "best_by_reconstruction": best_imp,
                "best_by_prediction": best_down,
                "best_agrees": bool(best_imp == best_down),
                f"{downstream_metric}_spread": spread,
                f"{downstream_metric}_seed_noise": seed_noise,
                "spread_exceeds_seed_noise": (
                    bool(spread > seed_noise) if np.isfinite(seed_noise) else None
                ),
                f"{imputation_metric}_spread": float(
                    group[imp_col].max() - group[imp_col].min()
                ),
            }
        )
    return pd.DataFrame(out).sort_values(["mechanism", "classifier", "level"])


def preference_stability(frame: pd.DataFrame, metric: str = "auroc") -> pd.DataFrame:
    """How often each imputer wins, broken down by mechanism and by classifier.

    Answers "does the preferred imputation strategy change with the
    mechanism, the level or the classifier?" If one method wins everywhere the
    recommendation is simple; if the winner moves, the interaction is the result.
    """
    best = best_imputer_table(frame, metric)
    by_mechanism = (
        best.groupby(["mechanism", "best_imputer"]).size().rename("wins").reset_index()
    )
    by_mechanism["scope"] = "mechanism"
    by_mechanism = by_mechanism.rename(columns={"mechanism": "stratum"})

    by_classifier = (
        best.groupby(["classifier", "best_imputer"]).size().rename("wins").reset_index()
    )
    by_classifier["scope"] = "classifier"
    by_classifier = by_classifier.rename(columns={"classifier": "stratum"})

    by_level = best.groupby(["level", "best_imputer"]).size().rename("wins").reset_index()
    by_level["scope"] = "level"
    by_level["stratum"] = by_level["level"].astype(str)
    by_level = by_level.drop(columns=["level"])

    out = pd.concat([by_mechanism, by_classifier, by_level], ignore_index=True)
    total = out.groupby("scope")["wins"].transform("sum")
    out["share"] = out["wins"] / total
    return out[["scope", "stratum", "best_imputer", "wins", "share"]]


def mechanism_comparison(frame: pd.DataFrame, metric: str = "auroc") -> pd.DataFrame:
    """Performance under each mechanism at matched levels, averaged over cells."""
    rows = test_rows(frame)
    rows = rows[rows["mechanism"] != "none"]
    return _agg(rows, ["mechanism", "level"], CLASSIFICATION_METRICS)


def headline(frame: pd.DataFrame) -> pd.DataFrame:
    """A handful of numbers that describe the run at a glance."""
    rows = test_rows(frame)
    grid = rows[rows["mechanism"] != "none"]
    base = rows[rows["mechanism"] == "none"]

    items = [
        ("test-split results (cells x classifiers)", float(len(rows))),
        ("mean AUROC, no artificial missingness", float(base["auroc"].mean()) if len(base) else np.nan),
        ("mean AUROC, 10% missing", float(grid[grid["level"] == 0.1]["auroc"].mean())),
        ("mean AUROC, 70% missing", float(grid[grid["level"] == 0.7]["auroc"].mean())),
        ("mean AUPRC, 10% missing", float(grid[grid["level"] == 0.1]["auprc"].mean())),
        ("mean AUPRC, 70% missing", float(grid[grid["level"] == 0.7]["auprc"].mean())),
        ("mean Brier, 10% missing", float(grid[grid["level"] == 0.1]["brier"].mean())),
        ("mean Brier, 70% missing", float(grid[grid["level"] == 0.7]["brier"].mean())),
    ]

    agreement = reconstruction_vs_downstream(frame)
    if not agreement.empty:
        items += [
            ("median Spearman rho, reconstruction vs AUROC", float(agreement["spearman_rho"].median())),
            ("conditions where best imputer agrees across stages", float(agreement["best_agrees"].mean())),
        ]
        # Restricted to the conditions where the imputers are actually
        # separated by more than seed noise; elsewhere the ranking is noise.
        decisive = agreement[agreement["spread_exceeds_seed_noise"] == True]  # noqa: E712
        if not decisive.empty:
            items += [
                ("conditions where imputer choice beats seed noise", float(len(decisive))),
                ("...of those, best imputer agrees across stages", float(decisive["best_agrees"].mean())),
            ]
    return pd.DataFrame(items, columns=["quantity", "value"])


def write_all(frame: pd.DataFrame, out_dir) -> dict[str, str]:
    """Write every table as CSV and return what was written."""
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}

    builders = {
        "headline": headline,
        "classification_summary": classification_summary,
        "imputation_summary": imputation_summary,
        "mechanism_comparison": mechanism_comparison,
        "best_imputer_auroc": lambda f: best_imputer_table(f, "auroc"),
        "best_imputer_auprc": lambda f: best_imputer_table(f, "auprc"),
        "best_imputer_brier": lambda f: best_imputer_table(f, "brier"),
        "reconstruction_vs_downstream": reconstruction_vs_downstream,
        "preference_stability": preference_stability,
        "degradation_auroc": lambda f: degradation_table(f, "auroc"),
    }
    for name, builder in builders.items():
        try:
            table = builder(frame)
        except (ValueError, KeyError) as exc:
            written[name] = f"skipped: {exc}"
            continue
        path = out_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        written[name] = str(path)
    return written
