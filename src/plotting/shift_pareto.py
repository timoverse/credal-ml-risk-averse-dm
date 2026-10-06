"""Shift-Pareto plot: expected loss (y) vs CVaR loss (x) across CIFAR-10-C severities.

One curve per (method, alpha): six points, severity 0 (the clean test set) through 5, each
aggregation averaged over the corruptions evaluated at that severity. Severity is annotated
along the curve, so each method reads as a degradation trajectory in (CVaR, mean) space.

Data comes from the W&B cache (see wandb_cache.py), parsed from the shift/* summary keys
written by experiments/shift_risk_metric.py (kind="shift" rows). Pure DataFrame in, matplotlib
figure out.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import PLOTS_PATH
from plotting.wandb_cache import load_runs

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

# Per-method (fmt, color, label), so the same method
# reads identically across figures; an unlisted method falls back to the default colour cycle.
# The decision rule is deliberately absent: shift runs use one rule per method (and set sizes
# are rule-independent), so rows are selected by method only. Should several rules ever coexist
# in the cache, plot_shift_pareto draws them as separate rule-labeled curves.
_METHOD_STYLES: dict[str, tuple[str, str | None, str]] = {
    "efficient_credal_prediction": ("-^", "crimson", "ECP"),
    "credal_bnn": ("-h", "olive", "CredalBNN"),
    "credal_rl_multinomial": ("-^", "brown", "RL-Multinomial"),
    "credal_wrapper": ("-X", "orange", "CredalWrapper"),
    "base": ("-s", "black", "Base MLE"),
}

# Colormap shading a method's alpha-swept curves (light = small alpha, dark = large), keyed by the
# method's base color. Single-alpha methods keep the
# flat base color.
_METHOD_CMAPS: dict[str | None, str] = {
    "crimson": "Reds",
    "orange": "Oranges",
    "brown": "copper",
    "black": "Greys",
}


def _title_tags(df: pd.DataFrame) -> str:
    """Compact 'dataset=...' tag summarising the slice plotted; empty if df has no dataset column."""
    if "dataset" not in df.columns:
        return ""
    datasets = sorted(df["dataset"].dropna().unique().tolist())
    return f"dataset={','.join(map(str, datasets))}" if datasets else ""


def _pivot_to_wide(df: pd.DataFrame, loss: str) -> pd.DataFrame:
    """Long-form (one row per aggregation) -> wide-form (one row per cell, aggregations as columns).

    Filters to a single per-instance loss first. A cell is one (artifact, decision_rule,
    corruption, severity); this also drops the loss="efficiency" set-size rows.
    """
    sub = df[df["loss"] == loss]
    if sub.empty:
        return sub
    for col in ("method.train.alpha", "num_train"):
        if col not in sub.columns:
            sub = sub.assign(**{col: float("nan")})
    # Non-cvar_minimax rows have no beta; the sentinel keeps cvar_beta out of NaN territory in
    # the pivot index and keeps different-beta cvar_minimax runs on separate rows.
    sub = sub.assign(cvar_beta=sub["cvar_beta"].fillna(-1.0))
    return sub.pivot(
        index=[
            "run_id",
            "method.name",
            "method.train.alpha",
            "num_train",
            "seed",
            "decision_rule",
            "cvar_beta",
            "corruption",
            "severity",
        ],
        columns="aggregation",
        values="value",
    ).reset_index()


def _collapse_to_latest_run(df: pd.DataFrame, wide: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per cell identity, from the latest run_id.

    Re-evaluating the same artifact produces a fresh W&B run with the same identity but a new
    run_id; without this every re-run would add a duplicate point at each severity. Recency
    comes from created_at on the long-form df. No-op if created_at is
    missing (older caches).
    """
    if "created_at" not in df.columns:
        return wide
    identity_cols = [
        "method.name",
        "decision_rule",
        "seed",
        "num_train",
        "method.train.alpha",
        "cvar_beta",
        "corruption",
        "severity",
    ]
    runs = df[["run_id", "created_at"]].drop_duplicates(subset=["run_id"])
    annotated = wide.merge(runs, on="run_id", how="left")
    annotated = annotated.sort_values("created_at", kind="mergesort")
    return annotated.drop_duplicates(subset=identity_cols, keep="last").drop(columns=["created_at"])


def plot_shift_pareto(
    df: pd.DataFrame,
    methods: list[str] | None = None,
    alphas: list[float] | None = None,
    seeds: list[int] | None = None,
    aggregation_x: str = "cvar_0.1",
    aggregation_y: str = "mean",
    loss: str = "log_loss",
    ax: Axes | None = None,
) -> Axes:
    """Plot expected (aggregation_y) vs worst-case (aggregation_x) of loss across shift severities.

    Each (method, alpha, seed) contributes one curve of six severity points (0 = clean through 5),
    every point averaging the aggregations over the corruptions evaluated at that severity. A
    method absent from the cache is skipped with a warning, so the same call works before and
    after new shift runs are synced. Calls plt.show() at the end when ax is None (i.e. when this
    function created the figure); pass ax to compose subplots.

    Args:
        df: Long-form DataFrame from load_runs(). Each shift row is one
            shift/<corruption>/<severity>/<decision_rule>/<aggregation>/<loss> value.
        methods: Method names to plot. The decision rule is ignored (shift runs use one rule
            per method); if several rules coexist in the cache they become separate
            rule-labeled curves. None plots every method present in the cache.
        alphas: Restrict alpha-swept methods to these method.train.alpha values (one curve
            each). Methods without an alpha (e.g. credal_wrapper, base) are always kept.
            None keeps all alphas present.
        seeds: Restrict to these seeds. None = all seeds present in df.
        aggregation_x: Aggregation plotted on the x-axis (typically a CVaR variant).
        aggregation_y: Aggregation plotted on the y-axis (typically "mean").
        loss: Per-instance loss to plot.
        ax: Existing matplotlib Axes to draw onto. None creates a new figure and calls plt.show().

    Returns:
        The Axes the plot was drawn onto.

    Raises:
        ValueError: no shift rows match the filters, or nothing was plotted.
        KeyError: aggregation_x or aggregation_y is absent from the cached shift data.
    """
    import pandas as pd  # noqa: PLC0415

    created_own_fig = ax is None
    df = df[df["kind"] == "shift"]
    if seeds is not None:
        df = df[df["seed"].isin(seeds)]
    wide = _pivot_to_wide(df, loss=loss)
    if wide.empty:
        raise ValueError(f"No shift rows in cache match loss={loss!r} (and seeds={seeds}).")
    wide = _collapse_to_latest_run(df, wide)
    index_cols = (
        "run_id",
        "method.name",
        "method.train.alpha",
        "num_train",
        "seed",
        "decision_rule",
        "cvar_beta",
        "corruption",
        "severity",
    )
    for col in (aggregation_x, aggregation_y):
        if col not in wide.columns:
            available = sorted(c for c in wide.columns if c not in index_cols)
            raise KeyError(f"Aggregation {col!r} not in cached shift data for loss={loss!r}. Available: {available}")

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    if methods is None:
        methods = sorted(wide["method.name"].dropna().unique().tolist())

    plotted = False
    for method in methods:
        fmt, color, base_label = _METHOD_STYLES.get(method, ("-s", None, method))
        sub = wide[wide["method.name"] == method]
        if alphas is not None:
            # Keep NaN-alpha rows: single-artifact methods (credal_wrapper, base) have no alpha knob.
            sub = sub[sub["method.train.alpha"].isin(alphas) | sub["method.train.alpha"].isna()]
        if sub.empty:
            logger.warning("No shift rows for method=%r; skipping.", method)
            continue
        multi_rule = sub["decision_rule"].nunique() > 1
        # -1.0 is the no-beta sentinel from _pivot_to_wide; several real betas mean several
        # cvar_minimax calibrations of the same artifact, which must stay separate curves.
        multi_beta = sub["cvar_beta"].nunique() > 1
        multi_seed = sub["seed"].nunique() > 1
        multi_num_train = sub["num_train"].nunique(dropna=False) > 1
        # Shade alpha-swept curves within the method's hue: with several alphas one flat color
        # makes the curves indistinguishable.
        alpha_values = sorted(sub["method.train.alpha"].dropna().unique())
        cmap = plt.colormaps[_METHOD_CMAPS.get(color, "viridis")] if len(alpha_values) > 1 else None
        group_cols = ["method.train.alpha", "seed", "num_train", "decision_rule", "cvar_beta"]
        for (alpha, seed, num_train, rule, beta), group in sub.groupby(group_cols, dropna=False):
            curve_color = color
            if cmap is not None and pd.notna(alpha):
                t = (alpha - alpha_values[0]) / (alpha_values[-1] - alpha_values[0])
                curve_color = cmap(0.35 + 0.65 * t)  # floor keeps the smallest alpha visible
            # One severity point per curve: average each aggregation over that severity's corruptions.
            curve = (
                group.groupby("severity")[[aggregation_x, aggregation_y]].mean().reset_index().sort_values("severity")
            )
            parts = [base_label]
            if multi_rule:
                parts.append(str(rule))
            if multi_beta and beta != -1.0:
                parts.append(f"β={beta:g}")
            if pd.notna(alpha):
                parts.append(f"α={alpha:g}")
            if multi_seed:
                parts.append(f"seed={seed}")
            if multi_num_train:
                parts.append(f"n={'full' if pd.isna(num_train) else int(num_train)}")
            ax.plot(
                curve[aggregation_x],
                curve[aggregation_y],
                fmt,
                color=curve_color,
                markersize=6,
                label=", ".join(parts),
            )
            for _, row in curve.iterrows():
                ax.annotate(
                    f"s={int(row['severity'])}",
                    (row[aggregation_x], row[aggregation_y]),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                    color=curve_color,
                )
            plotted = True

    if not plotted:
        raise ValueError(f"No points to plot for methods={methods}, alphas={alphas}, seeds={seeds}, loss={loss!r}.")

    ax.set_xlabel(f"{aggregation_x} of {loss}")
    ax.set_ylabel(f"{aggregation_y} of {loss}")
    title = "Shift-Pareto (severity 0 = clean)"
    title_tags = _title_tags(df)
    if title_tags:
        title = f"{title}\n{title_tags}"
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    if created_own_fig:
        plt.tight_layout()
        plt.show()
    return ax


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Edit these and re-run from PyCharm. None on filter fields means "no constraint".
    DATASET: str | None = "cifar10"
    METHODS: list[str] | None = ["efficient_credal_prediction", "credal_rl_multinomial"]
    ALPHAS: list[float] | None = [0.6, 0.8, 0.95]  # None = all cached alphas
    SEEDS: list[int] | None = [1]
    # One figure per CVaR level tau; extend once more levels are evaluated in shift runs.
    AGGREGATIONS_X: list[str] = ["cvar_0.05", "cvar_0.1"]
    AGGREGATION_Y = "mean"
    LOSS = "log_loss"
    SAVE = True

    filters: dict[str, object] = {}
    if DATASET is not None:
        filters["dataset"] = DATASET
    df = load_runs(filters=filters or None)
    print(f"Loaded {len(df)} rows from {df['run_id'].nunique()} runs.")

    for aggregation_x in AGGREGATIONS_X:
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_shift_pareto(
            df,
            methods=METHODS,
            alphas=ALPHAS,
            seeds=SEEDS,
            aggregation_x=aggregation_x,
            aggregation_y=AGGREGATION_Y,
            loss=LOSS,
            ax=ax,
        )
        fig.tight_layout()
        if SAVE:
            plot_path = PLOTS_PATH / f"shift_pareto_{LOSS}_{aggregation_x}_{DATASET}.pdf"
            fig.savefig(plot_path, bbox_inches="tight")
            print(f"Plot saved to {plot_path}")
    plt.show()
