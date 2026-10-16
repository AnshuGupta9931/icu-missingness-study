"""Figures for the results chapter.

Conventions held across every figure, so they read as one set:

* Imputation method is the series identity and always gets the same colour,
  marker and line style - colour is never the only cue, which keeps the figures
  readable in greyscale print and under colour-vision deficiency.
* Shaded bands are +/- 1 standard deviation across seeds. Where bands overlap,
  the difference between those methods is within run-to-run noise and should not
  be reported as an effect.
* One y-axis per panel. Reconstruction error and AUROC are never plotted against
  a shared pair of axes: comparing them is the job of the scatter figure, where
  each is given its own axis.
* Panels that share a quantity share its scale, so panel-to-panel comparison is
  valid by eye.

Colours are the validated categorical palette: adjacent-pair separation holds
for the four-series line charts, and the scatter is restricted to three
simultaneous colours (mechanism) with the fourth factor split across panels,
which is what the all-pairs gate requires.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from .tables import _agg, test_rows  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#8a8982"
GRID = "#e6e5e1"

# Validated categorical slots 1-4 (blue, orange, aqua, yellow).
IMPUTER_STYLE: dict[str, dict] = {
    "median": {"color": "#2a78d6", "marker": "o", "linestyle": "-", "label": "Median"},
    "knn": {"color": "#eb6834", "marker": "s", "linestyle": "--", "label": "KNN"},
    "mice": {"color": "#1baf7a", "marker": "^", "linestyle": "-.", "label": "MICE"},
    "missforest": {"color": "#eda100", "marker": "D", "linestyle": ":", "label": "MissForest"},
}

# Slots 1-3 only: these three validate on the all-pairs gate used by scatters.
MECHANISM_STYLE: dict[str, dict] = {
    "mcar": {"color": "#2a78d6", "marker": "o", "label": "MCAR"},
    "mar": {"color": "#eb6834", "marker": "s", "label": "MAR"},
    "mnar": {"color": "#1baf7a", "marker": "^", "label": "MNAR"},
}

# Sequential blue ramp, light -> dark, for magnitude heatmaps.
SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]

MECHANISM_LABEL = {"mcar": "MCAR", "mar": "MAR", "mnar": "MNAR", "none": "No masking"}
CLASSIFIER_LABEL = {
    "logreg": "Logistic regression",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "lightgbm": "LightGBM",
}
METRIC_LABEL = {
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "f1": "F1 score",
    "precision": "Precision",
    "recall": "Recall",
    "brier": "Brier score",
    "log_loss": "Log loss",
    "imp_rmse_z": "Reconstruction RMSE (standardised units)",
    "imp_mae_z": "Reconstruction MAE (standardised units)",
    "imp_r2_z": "Reconstruction $R^2$ (standardised units)",
    "imp_rmse_raw": "Reconstruction RMSE (original units)",
    "imp_r2_raw": "Reconstruction $R^2$ (original units)",
}


def use_style() -> None:
    """Recessive axes, readable type, nothing competing with the data."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SOFT,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 6.5,
            "lines.markeredgewidth": 0.9,
            "font.size": 10,
            "figure.dpi": 130,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def _tidy(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", visible=False)


def _metric_label(metric: str) -> str:
    return METRIC_LABEL.get(metric, metric.replace("_", " "))


# ----------------------------------------------------------------------
# line charts: metric against missingness level
# ----------------------------------------------------------------------
def metric_vs_level(
    frame: pd.DataFrame,
    metric: str = "auroc",
    classifier: str | None = None,
    out_path: str | Path | None = None,
    show_baseline: bool = True,
    who_label: str | None = None,
):
    """`metric` against missingness level, one panel per mechanism.

    Parameters
    ----------
    classifier
        Restrict to one classifier, or None to average over all of them. The
        averaged version answers "what does missingness do?"; the per-classifier
        versions are what you read to see whether the answer depends on the
        model.
    """
    use_style()
    rows = test_rows(frame)
    base = rows[rows["mechanism"] == "none"]
    grid = rows[rows["mechanism"] != "none"]
    if classifier:
        grid = grid[grid["classifier"] == classifier]
        base = base[base["classifier"] == classifier]

    mechanisms = [m for m in ("mcar", "mar", "mnar") if m in set(grid["mechanism"])]
    if not mechanisms:
        raise ValueError("no mechanism rows to plot")

    summary = _agg(grid, ["mechanism", "level", "imputer"], [metric])
    fig, axes = plt.subplots(
        1, len(mechanisms), figsize=(4.0 * len(mechanisms), 3.9), sharey=True
    )
    axes = np.atleast_1d(axes)

    for ax, mechanism in zip(axes, mechanisms):
        panel = summary[summary["mechanism"] == mechanism]

        if show_baseline and not base.empty:
            ref = float(base[metric].mean())
            ax.axhline(ref, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
            if ax is axes[-1]:
                # Right-aligned and below the line: the curves fall away from
                # the reference as missingness grows, so this corner stays clear.
                ax.annotate(
                    "no artificial missingness",
                    xy=(0.98, ref),
                    xycoords=("axes fraction", "data"),
                    xytext=(0, -4),
                    textcoords="offset points",
                    ha="right",
                    va="top",
                    fontsize=8,
                    color=INK_MUTED,
                )

        for imputer, style in IMPUTER_STYLE.items():
            series = panel[panel["imputer"] == imputer].sort_values("level")
            if series.empty:
                continue
            x = series["level"].to_numpy() * 100
            mean = series[f"{metric}_mean"].to_numpy()
            sd = series[f"{metric}_std"].to_numpy()
            ax.fill_between(
                x, mean - sd, mean + sd, color=style["color"], alpha=0.14, linewidth=0, zorder=2
            )
            ax.plot(
                x,
                mean,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                markeredgecolor=SURFACE,  # 2px surface ring where marks overlap
                zorder=3,
                label=style["label"],
            )

        ax.set_title(MECHANISM_LABEL[mechanism])
        ax.set_xlabel("Missing values masked (%)")
        _tidy(ax)

    axes[0].set_ylabel(_metric_label(metric))
    handles = [
        Line2D(
            [], [],
            color=s["color"], marker=s["marker"], linestyle=s["linestyle"],
            markeredgecolor=SURFACE, label=s["label"],
        )
        for s in IMPUTER_STYLE.values()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07))

    if who_label is not None:
        who = who_label
    else:
        who = CLASSIFIER_LABEL.get(classifier, classifier) if classifier else "mean over classifiers"
    fig.suptitle(
        f"{_metric_label(metric)} against missingness level  -  {who}",
        fontsize=12, fontweight="bold", color=INK, y=1.02,
    )
    # Say exactly what the band covers: with no classifier fixed, the points
    # being averaged vary over classifiers as well as seeds.
    spread = "seeds" if classifier else "classifiers and seeds"
    fig.text(
        0.5, -0.14,
        f"Bands show +/- 1 SD across {spread}. Overlapping bands mean the gap is within run-to-run noise.",
        ha="center", fontsize=8, color=INK_MUTED,
    )
    return _finish(fig, out_path)


def reconstruction_vs_level(
    frame: pd.DataFrame, metric: str = "imp_rmse_z", out_path: str | Path | None = None
):
    """Reconstruction error against missingness level, one panel per mechanism.

    Imputation happens before any classifier is fitted, so these values are
    identical across the four classifiers. One is selected to avoid counting
    the same imputation four times, and the title says so rather than naming
    that classifier, which would imply a dependence that does not exist.
    """
    return metric_vs_level(
        frame, metric=metric, classifier=_any_classifier(frame), out_path=out_path,
        show_baseline=False,
        who_label="identical for every classifier",
    )


# ----------------------------------------------------------------------
# the central scatter: does reconstruction accuracy transfer?
# ----------------------------------------------------------------------
def reconstruction_vs_downstream(
    frame: pd.DataFrame,
    imputation_metric: str = "imp_rmse_z",
    downstream_metric: str = "auroc",
    out_path: str | Path | None = None,
):
    """Downstream performance against reconstruction error, one panel per imputer.

    The question the project is built around. If reconstruction accuracy
    translated directly into predictive utility, every panel would show a clean
    downward slope and the panels would line up with each other. Point size
    encodes the missingness level, so the within-panel trajectory from low to
    high missingness is visible too.

    Mechanism carries the colour here (three simultaneous hues, which is what
    the all-pairs colour gate allows) and imputer moves to the panel axis.
    """
    use_style()
    rows = test_rows(frame)
    rows = rows[rows["mechanism"] != "none"]
    summary = _agg(
        rows, ["mechanism", "level", "imputer", "classifier"],
        [imputation_metric, downstream_metric],
    )
    imp_col, down_col = f"{imputation_metric}_mean", f"{downstream_metric}_mean"
    summary = summary.dropna(subset=[imp_col, down_col])
    if summary.empty:
        raise ValueError("no rows with both metrics present")

    imputers = [i for i in IMPUTER_STYLE if i in set(summary["imputer"])]
    present = [m for m in MECHANISM_STYLE if m in set(summary["mechanism"])]
    fig, axes = plt.subplots(
        1, len(imputers), figsize=(3.3 * len(imputers), 3.6), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)

    levels = np.sort(summary["level"].unique())
    size_for = dict(zip(levels, np.linspace(18, 90, len(levels))))

    for ax, imputer in zip(axes, imputers):
        panel = summary[summary["imputer"] == imputer]
        for mechanism in present:
            style = MECHANISM_STYLE[mechanism]
            sub = panel[panel["mechanism"] == mechanism]
            if sub.empty:
                continue
            ax.scatter(
                sub[imp_col], sub[down_col],
                s=[size_for[lv] for lv in sub["level"]],
                color=style["color"], marker=style["marker"],
                edgecolor=SURFACE, linewidth=0.8, alpha=0.85, zorder=3,
                label=style["label"],
            )
        ax.set_title(IMPUTER_STYLE[imputer]["label"])
        _tidy(ax)
        ax.grid(axis="x", visible=True)

    axes[0].set_ylabel(_metric_label(downstream_metric))
    # One shared x label: the panels share the axis, and four copies of a long
    # label collide with each other.
    fig.supxlabel(_metric_label(imputation_metric), fontsize=9.5, color=INK_SOFT, y=-0.02)

    # Only mechanisms actually present get a legend entry, so a partial run
    # does not advertise a condition it never ran.
    handles = [
        Line2D([], [], color=MECHANISM_STYLE[m]["color"], marker=MECHANISM_STYLE[m]["marker"],
               linestyle="none", markeredgecolor=SURFACE, label=MECHANISM_STYLE[m]["label"])
        for m in present
    ]
    handles += [
        Line2D([], [], color=INK_MUTED, marker="o", linestyle="none",
               markersize=np.sqrt(size_for[levels[0]]), label=f"{levels[0]:.0%} missing"),
        Line2D([], [], color=INK_MUTED, marker="o", linestyle="none",
               markersize=np.sqrt(size_for[levels[-1]]), label=f"{levels[-1]:.0%} missing"),
    ]
    fig.legend(
        handles=handles, loc="lower center",
        ncol=len(handles), bbox_to_anchor=(0.5, -0.17),
    )
    fig.suptitle(
        "Does better reconstruction mean better prediction?",
        fontsize=12, fontweight="bold", color=INK, y=1.14,
    )
    fig.text(
        0.5, 1.02,
        "Each point is one (mechanism, level, classifier) condition, averaged over seeds. "
        "Point size is the missingness level.",
        ha="center", fontsize=8.5, color=INK_SOFT,
    )
    return _finish(fig, out_path)


# ----------------------------------------------------------------------
# heatmaps
# ----------------------------------------------------------------------
def metric_heatmap(
    frame: pd.DataFrame,
    metric: str = "auroc",
    classifier: str | None = None,
    out_path: str | Path | None = None,
):
    """Imputer x level grid of `metric`, one panel per mechanism.

    Sequential single-hue ramp: darker is a higher value. Every cell is
    labelled, so the numbers are readable without decoding the colour.
    """
    use_style()
    rows = test_rows(frame)
    rows = rows[rows["mechanism"] != "none"]
    if classifier:
        rows = rows[rows["classifier"] == classifier]

    summary = _agg(rows, ["mechanism", "level", "imputer"], [metric])
    col = f"{metric}_mean"
    mechanisms = [m for m in ("mcar", "mar", "mnar") if m in set(summary["mechanism"])]
    imputers = [i for i in IMPUTER_STYLE if i in set(summary["imputer"])]

    vmin, vmax = float(summary[col].min()), float(summary[col].max())
    cmap = ListedColormap(SEQUENTIAL_BLUE)
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(
        1, len(mechanisms), figsize=(3.9 * len(mechanisms), 3.1), sharey=True
    )
    axes = np.atleast_1d(axes)

    for ax, mechanism in zip(axes, mechanisms):
        panel = summary[summary["mechanism"] == mechanism]
        pivot = panel.pivot(index="imputer", columns="level", values=col).reindex(imputers)
        ax.imshow(pivot.to_numpy(), cmap=cmap, norm=norm, aspect="auto")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:.0%}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([IMPUTER_STYLE[i]["label"] for i in pivot.index])
        ax.set_title(MECHANISM_LABEL[mechanism])
        ax.set_xlabel("Masked (%)")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        span = max(vmax - vmin, 1e-12)
        for r, imputer in enumerate(pivot.index):
            for c, level in enumerate(pivot.columns):
                value = pivot.loc[imputer, level]
                if not np.isfinite(value):
                    continue
                dark = (value - vmin) / span > 0.55
                ax.text(
                    c, r, f"{value:.3f}", ha="center", va="center", fontsize=7.6,
                    color=SURFACE if dark else INK,
                )

    bar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes.tolist(),
        fraction=0.025, pad=0.02,
    )
    bar.set_label(_metric_label(metric), fontsize=9, color=INK_SOFT)
    bar.outline.set_visible(False)

    who = CLASSIFIER_LABEL.get(classifier, classifier) if classifier else "mean over classifiers"
    fig.suptitle(
        f"{_metric_label(metric)} by imputation method and missingness level  -  {who}",
        fontsize=12, fontweight="bold", color=INK, y=1.04,
    )
    return _finish(fig, out_path)


def best_imputer_heatmap(
    frame: pd.DataFrame, metric: str = "auroc", out_path: str | Path | None = None
):
    """Which imputer wins in each condition - the stability question, as a picture.

    A single colour filling a panel means the recommendation is stable for that
    mechanism; a patchwork means the preferred method depends on the level or
    the classifier. Cells carry the method name as text, so identity never rests
    on colour alone, and cells whose winning margin is inside one seed-level
    standard deviation are hatched: those are ties, not wins.
    """
    from .tables import best_imputer_table

    use_style()
    best = best_imputer_table(frame, metric)
    mechanisms = [m for m in ("mcar", "mar", "mnar") if m in set(best["mechanism"])]
    classifiers = [c for c in CLASSIFIER_LABEL if c in set(best["classifier"])]
    imputers = list(IMPUTER_STYLE)
    code = {name: i for i, name in enumerate(imputers)}
    cmap = ListedColormap([IMPUTER_STYLE[i]["color"] for i in imputers])

    # With a single seed there is no spread to compare a margin against, so no
    # cell can be called decisive and marking them all as ties would say more
    # about the run than about the methods.
    sd_available = bool(np.isfinite(best["best_seed_sd"]).any())

    fig, axes = plt.subplots(
        1, len(mechanisms), figsize=(4.3 * len(mechanisms), 2.9), sharey=True
    )
    axes = np.atleast_1d(axes)

    for ax, mechanism in zip(axes, mechanisms):
        panel = best[best["mechanism"] == mechanism]
        pivot = panel.pivot(index="classifier", columns="level", values="best_imputer")
        pivot = pivot.reindex(classifiers)
        solid = panel.pivot(index="classifier", columns="level", values="margin_exceeds_seed_sd")
        solid = solid.reindex(classifiers)

        grid_values = pivot.map(lambda v: code.get(v, np.nan) if isinstance(v, str) else np.nan)
        ax.imshow(
            grid_values.to_numpy(dtype=float), cmap=cmap, aspect="auto",
            vmin=-0.5, vmax=len(imputers) - 0.5,
        )

        for r, clf in enumerate(pivot.index):
            for c, level in enumerate(pivot.columns):
                winner = pivot.loc[clf, level]
                if not isinstance(winner, str):
                    continue
                decisive = bool(solid.loc[clf, level]) if sd_available else True
                if not decisive:
                    # Wash the cell toward the surface rather than hatching it:
                    # a tie should read as "faint", and hatching competes with
                    # the label that carries the identity.
                    ax.add_patch(
                        plt.Rectangle(
                            (c - 0.5, r - 0.5), 1, 1,
                            facecolor=SURFACE, alpha=0.55, linewidth=0,
                        )
                    )
                ax.text(
                    c, r, IMPUTER_STYLE[winner]["label"],
                    ha="center", va="center", fontsize=7.4,
                    color=_readable_on(IMPUTER_STYLE[winner]["color"]),
                    fontweight="bold" if decisive else "normal",
                )

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:.0%}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([CLASSIFIER_LABEL[c] for c in pivot.index])
        ax.set_title(MECHANISM_LABEL[mechanism])
        ax.set_xlabel("Masked (%)")
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        f"Best imputation method by condition (by {_metric_label(metric)})",
        fontsize=12, fontweight="bold", color=INK, y=1.14,
    )
    caption = (
        "Faded cells: the winner's margin is within one seed-level SD, so the condition is a tie."
        if sd_available
        else "Single seed: margins cannot be compared against run-to-run spread, so no cell is marked as a tie."
    )
    fig.text(0.5, 1.02, caption, ha="center", fontsize=8.5, color=INK_SOFT)
    return _finish(fig, out_path)


def _readable_on(hex_color: str) -> str:
    """Black or white label text, whichever the cell colour can carry.

    White on the yellow slot is close to unreadable; picking by luminance keeps
    every cell legible without hard-coding exceptions per colour.
    """
    r, g, b = (int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
    channels = [
        c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)
    ]
    luminance = 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]
    return INK if luminance > 0.42 else SURFACE


# ----------------------------------------------------------------------
# natural missingness profile
# ----------------------------------------------------------------------
def natural_missingness(profile: pd.DataFrame, out_path: str | Path | None = None, top: int = 25):
    """The cohort's own missingness, before any masking (proposal objective 1)."""
    use_style()
    data = profile.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.0, max(3.0, 0.26 * len(data))))

    colors = [
        IMPUTER_STYLE["median"]["color"] if m else INK_MUTED for m in data["maskable"]
    ]
    ax.barh(
        data["column"], data["frac_missing"] * 100,
        color=colors, height=0.68, zorder=3,
    )
    ax.set_xlabel("Missing in cohort (%)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    _tidy(ax)

    for name, value in zip(data["column"], data["frac_missing"] * 100):
        if value > 0.5:
            ax.annotate(
                f"{value:.1f}%", xy=(value, name), xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=8, color=INK_SOFT,
            )

    handles = [
        Line2D([], [], color=IMPUTER_STYLE["median"]["color"], marker="s", linestyle="none",
               label="Maskable (clinical measurement)"),
        Line2D([], [], color=INK_MUTED, marker="s", linestyle="none",
               label="Protected (complete by design)"),
    ]
    ax.legend(handles=handles, loc="lower right")
    ax.set_title(
        "Naturally occurring missingness in the cohort",
        fontsize=12, fontweight="bold", color=INK, loc="left",
    )
    return _finish(fig, out_path)


# ----------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------
def write_all(
    frame: pd.DataFrame,
    out_dir: str | Path,
    profile: pd.DataFrame | None = None,
    classifiers: Sequence[str] | None = None,
) -> dict[str, str]:
    """Render the whole figure set; records a reason for anything skipped."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    jobs: dict[str, callable] = {
        "auroc_vs_level": lambda p: metric_vs_level(frame, "auroc", None, p),
        "auprc_vs_level": lambda p: metric_vs_level(frame, "auprc", None, p),
        "brier_vs_level": lambda p: metric_vs_level(frame, "brier", None, p),
        "f1_vs_level": lambda p: metric_vs_level(frame, "f1", None, p),
        "reconstruction_rmse_vs_level": lambda p: reconstruction_vs_level(frame, "imp_rmse_z", p),
        "reconstruction_r2_vs_level": lambda p: reconstruction_vs_level(frame, "imp_r2_z", p),
        "reconstruction_vs_downstream": lambda p: reconstruction_vs_downstream(frame, out_path=p),
        "auroc_heatmap": lambda p: metric_heatmap(frame, "auroc", None, p),
        "best_imputer_auroc": lambda p: best_imputer_heatmap(frame, "auroc", p),
        "best_imputer_brier": lambda p: best_imputer_heatmap(frame, "brier", p),
    }
    for clf in classifiers or []:
        jobs[f"auroc_vs_level_{clf}"] = (
            lambda p, c=clf: metric_vs_level(frame, "auroc", c, p)
        )

    if profile is not None and not profile.empty:
        jobs["natural_missingness"] = lambda p: natural_missingness(profile, p)

    for name, job in jobs.items():
        path = out_dir / f"{name}.png"
        try:
            job(path)
            written[name] = str(path)
        except (ValueError, KeyError, IndexError) as exc:
            written[name] = f"skipped: {exc}"
        finally:
            plt.close("all")
    return written


def _any_classifier(frame: pd.DataFrame) -> str:
    """Reconstruction metrics repeat across classifiers; pick one to avoid double counting."""
    rows = test_rows(frame)
    return str(rows["classifier"].iloc[0])


def _finish(fig, out_path: str | Path | None):
    if out_path is not None:
        fig.savefig(out_path)
        plt.close(fig)
        return Path(out_path)
    return fig
