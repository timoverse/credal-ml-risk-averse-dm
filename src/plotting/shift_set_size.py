"""Set-size-under-shift plot: credal set size (y) vs corruption severity (x).

One curve per (method, alpha): within each seed the set size is averaged over the corruptions
evaluated at a severity, then the curve shows mean plus/minus one std across seeds (the std is
zero for a single seed). Set size is 1 - efficiency, the mean credal interval width, taken from
the shift/<corruption>/<severity>/set/efficiency summary keys written by
experiments/shift_set_size.py (probly's efficiency is 1 - mean interval width, so larger set
size = wider, less committed sets). Rows from the pre-split .../set/mean/efficiency keys are
ignored; re-run shift_set_size.py to repopulate.

Data comes from the W&B cache (see wandb_cache.py). Pure DataFrame in, matplotlib figure out.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import PLOTS_PATH
from plotting.paper_style import (
    BASELINE_BLUE,
    COLUMN_WIDTH,
    FP_REGULAR,
    METHODS,
    OURS,
    OURS_TAG,
    TEXT_WIDTH,
    Method,
    Sizes,
    band_style,
    font,
    halo,
    legend_below,
    line_style,
    ours_label,
    render_for_print,
    save,
    set_xlabel,
    set_ylabel,
    style_axes,
    tint,
    use_fira_mathtext,
)
from plotting.shift_pareto import _METHOD_STYLES
from plotting.wandb_cache import load_runs

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# This figure contrasts the credal baselines AS A GROUP with our predictor, like the coverage
# against efficiency figure: every baseline wears the baseline blue and is told apart by its marker,
# the marker it has in that figure, so a method looks the same in both. Six of them crowd into a
# narrow band, so the baselines are drawn lighter than our predictor: thinner lines and smaller
# markers, which also leaves the one red line standing out.
_BASELINE_MARKERS: dict[str, str] = {
    "credal_ensembling": "P",
    "credal_bnn": "^",
    "credal_wrapper": "*",
    "credal_dro": "X",
    "credal_relative_likelihood": "s",
    "efficient_credal_prediction": "D",
}
_BASELINE_LINE_SHRINK = 0.65
_BASELINE_MARKER_SHRINK = 0.8
# Methods outside the paper's registry (exploratory sweeps) share one neutral look and keep their
# shift_pareto names.
_FALLBACK_COLOR = "#9a9a9a"
_CURVE_DASH = (0, (2.8, 2.8))  # the accuracy linestyle
# Width the paper includes the figure at, per dataset: BloodMNIST is the main-text figure in one
# column, the others are appendix figures in the one-column layout.
_PRINT_WIDTHS: dict[str, float] = {"bloodmnist": 0.95 * COLUMN_WIDTH}
_PRINT_WIDTH_APPENDIX = 0.5 * TEXT_WIDTH
# Display names; unlisted datasets fall back to their cache name.
_DATASET_NAMES: dict[str, str] = {
    "bloodmnist": "BloodMNIST",
    "cifar10": "CIFAR-10",
}


# The ensemble has no alpha knob in the cache; the paper table pins it as alpha = 0.
_ALPHA_FREE_LABELS: dict[str, str] = {"credal_ensembling": "CreEns$_{0.0}$"}


def _method_style(method: str) -> Method:
    """A method's label and its color in this figure: ours its own, a credal baseline the shared blue.

    Methods the paper does not show get the neutral fallback.
    """
    if method in _BASELINE_MARKERS:
        return Method(METHODS[method].label, BASELINE_BLUE, _BASELINE_MARKERS[method])
    if method in METHODS:
        return METHODS[method]
    label = _METHOD_STYLES[method][2] if method in _METHOD_STYLES else method
    return Method(label, _FALLBACK_COLOR, "o")


def _collapse_to_latest_run(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one efficiency row per cell identity, from the latest run_id.

    Duplicates of a cell arise when the same identity is retrained: a fresh training run with a
    new run_id re-reports the same cell once shift_set_size runs on the new artifact. The latest
    row wins. Recency comes from created_at; no-op if it is missing.
    """
    if "created_at" not in df.columns:
        return df
    identity_cols = ["method.name", "seed", "num_train", "method.train.alpha", "corruption", "severity"]
    df = df.sort_values("created_at", kind="mergesort")
    return df.drop_duplicates(subset=identity_cols, keep="last")


def _line_kwargs(method: str, sizes: Sizes) -> dict:
    """ax.plot keyword arguments for a method's set-size curve (see paper_style.line_style).

    Ours is drawn as everywhere in the paper. A baseline keeps the paper-wide line but trades its
    own color for the shared baseline blue and takes its marker from the coverage figure.
    """
    if method == OURS:
        return line_style(method, sizes)
    if method in _BASELINE_MARKERS:
        width = _BASELINE_LINE_SHRINK * sizes.line
        return line_style(method, sizes) | {
            "color": BASELINE_BLUE,
            "marker": _BASELINE_MARKERS[method],
            "markersize": _BASELINE_MARKER_SHRINK * sizes.marker,
            "markeredgewidth": _BASELINE_LINE_SHRINK * sizes.marker_edge,
            "linewidth": width,
            "path_effects": halo(width, sizes),
        }
    return {
        "color": _FALLBACK_COLOR,
        "linewidth": sizes.line,
        "marker": "o",
        "markersize": sizes.marker,
        "solid_capstyle": "round",
        "path_effects": halo(sizes.line, sizes),
        "zorder": 3,
    }


def _direction_arrow(ax: Axes, arrow: str, sizes: Sizes) -> None:
    """Better-direction arrow, upright above the rotated y label.

    Anchored to the label's bbox at draw time, so it tracks any later re-layout. Computer Modern
    math for the thin arrow shape; Fira's own glyph is too heavy.
    """
    ax.annotate(
        arrow,
        xy=(0.5, 1.0),
        xycoords=ax.yaxis.label,
        xytext=(0, 3 * sizes.scale),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=sizes.label,
        math_fontfamily="cm",
    )


def plot_shift_set_size(  # noqa: PLR0913
    df: pd.DataFrame,
    methods: list[str] | None = None,
    alphas: list[float] | None = None,
    seeds: list[int] | None = None,
    plot_accuracy: bool = True,
    sizes: Sizes | None = None,
) -> Figure:
    """Plot credal set size against shift severity, averaged over corruptions and seeds.

    Each (method, alpha) contributes one curve of six severity points (0 = clean through 5).
    Within each seed the set size is first averaged over that severity's corruptions; the marker
    is the mean and the shaded band plus/minus one std across seeds. Methods without efficiency
    rows in the cache (non-credal, or not yet synced) are skipped with a warning.

    Args:
        df: Long-form DataFrame from load_runs(). Efficiency rows are kind="shift" with
            loss="efficiency" and the rule-independent sentinel decision_rule="set".
        methods: Method names to plot, styled as everywhere in the paper (paper_style.METHODS).
            None plots every method with efficiency rows in the cache.
        alphas: Restrict alpha-swept methods to these method.train.alpha values (one curve
            each). Methods without an alpha are always kept. None keeps all alphas present.
        seeds: Restrict to these seeds. None = all seeds present in df.
        plot_accuracy: Overlay the regular point-prediction accuracy (loss="accuracy" rows, one
            dashed curve per method in the method's color on a right-hand axis; the accuracy is
            alpha-free, so rows from different alpha artifacts are averaged). Accuracy curves get
            no legend entries of their own: the legend names each method once by color and two
            neutral style entries say which line style is the set size and which the accuracy.
            Skipped with a log message when no accuracy rows are cached (older sweeps did not
            record them).
        sizes: Type sizes and stroke widths. None uses the reference's.

    Returns:
        The assembled matplotlib Figure.

    Raises:
        ValueError: no efficiency rows match the filters, or nothing was plotted.
    """
    import pandas as pd  # noqa: PLC0415

    sizes = sizes or Sizes()
    use_fira_mathtext()
    df = df[(df["kind"] == "shift") & (df["decision_rule"] == "set") & (df["loss"].isin(["efficiency", "accuracy"]))]
    if "aggregation" in df.columns:
        # Legacy pre-split rows (shift/<c>/<s>/set/mean/efficiency) carry aggregation="mean";
        # current shift_set_size.py keys have none. Plot only current-form rows so vintages
        # cannot silently mix.
        df = df[df["aggregation"].isna()]
    if seeds is not None:
        df = df[df["seed"].isin(seeds)]
    for col in ("method.train.alpha", "num_train"):
        if col not in df.columns:
            df = df.assign(**{col: float("nan")})
    # The dedup identity has no loss column, so collapse per metric.
    acc_df = _collapse_to_latest_run(df[df["loss"] == "accuracy"])
    df = _collapse_to_latest_run(df[df["loss"] == "efficiency"])
    if df.empty:
        raise ValueError(
            f"No shift set-size rows in cache (seeds={seeds}). Run experiments/shift_set_size.py "
            "and re-sync the cache (python -m plotting.wandb_cache ...)."
        )
    # probly's efficiency is 1 - mean interval width; invert so the y-axis reads as a set size.
    df = df.assign(set_size=1.0 - df["value"])

    fig, ax = plt.subplots(figsize=(4.6, 2.3), constrained_layout=True)
    if methods is None:
        methods = sorted(df["method.name"].dropna().unique().tolist())

    plotted = False
    for method in methods:
        style = _method_style(method)
        sub = df[df["method.name"] == method]
        if alphas is not None:
            # Keep NaN-alpha rows: single-artifact methods (e.g. credal_wrapper) have no alpha knob.
            sub = sub[sub["method.train.alpha"].isin(alphas) | sub["method.train.alpha"].isna()]
        if sub.empty:
            logger.warning("No shift efficiency rows for method=%r; skipping.", method)
            continue
        multi_num_train = sub["num_train"].nunique(dropna=False) > 1
        # Shade alpha-swept curves within the method's hue (light = small alpha, full = large).
        alpha_values = sorted(sub["method.train.alpha"].dropna().unique())
        for (alpha, num_train), group in sub.groupby(["method.train.alpha", "num_train"], dropna=False):
            kwargs = _line_kwargs(method, sizes)
            if len(alpha_values) > 1 and pd.notna(alpha):
                t = (alpha - alpha_values[0]) / (alpha_values[-1] - alpha_values[0])
                kwargs["color"] = tint(style.color, 0.45 + 0.55 * t)  # floor keeps the smallest alpha visible
            # Corruption-average within each seed first, then mean +/- std across seeds.
            per_seed = group.groupby(["seed", "severity"])["set_size"].mean().reset_index()
            stats = per_seed.groupby("severity")["set_size"].agg(["mean", "std"]).reset_index().sort_values("severity")
            label = _ALPHA_FREE_LABELS.get(method, style.label)
            if pd.notna(alpha):
                label = f"{label}$_{{{alpha:g}}}$"  # bare alpha value as the index
            if multi_num_train:
                label = f"{label}, n={'full' if pd.isna(num_train) else int(num_train)}"
            if method == OURS:
                label = ours_label(label)
            ax.plot(stats["severity"], stats["mean"], label=label, **kwargs)
            # std is NaN for a single seed; a zero-width band then draws nothing.
            std = stats["std"].fillna(0.0)
            ax.fill_between(stats["severity"], stats["mean"] - std, stats["mean"] + std, **band_style(kwargs["color"]))
            plotted = True

    if not plotted:
        raise ValueError(f"No set sizes to plot for methods={methods}, alphas={alphas}, seeds={seeds}.")

    # Accuracy overlay: one dashed curve per method in the method's color on a right-hand axis,
    # corruption- and alpha-averaged within each seed, mean plus/minus one std across seeds. No
    # per-curve legend entries; the style legend inside the axes explains the dashes.
    accuracy_width = 0.85 * sizes.line
    has_accuracy = False
    if plot_accuracy:
        if acc_df.empty:
            logger.info("No accuracy rows cached (older sweeps); accuracy overlay skipped.")
        else:
            ax2 = ax.twinx()
            # Draw the set-size axes above the twin, so the primary curves' halos win at crossings.
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)
            for method in methods:
                color = _method_style(method).color
                sub = acc_df[acc_df["method.name"] == method]
                if alphas is not None:
                    sub = sub[sub["method.train.alpha"].isin(alphas) | sub["method.train.alpha"].isna()]
                if sub.empty:
                    continue
                per_seed = sub.groupby(["seed", "severity"])["value"].mean().reset_index()
                stats = per_seed.groupby("severity")["value"].agg(["mean", "std"]).reset_index().sort_values("severity")
                # A baseline's accuracy curve is thinned like its set-size curve.
                width = accuracy_width * (_BASELINE_LINE_SHRINK if method in _BASELINE_MARKERS else 1.0)
                ax2.plot(
                    stats["severity"],
                    stats["mean"],
                    color=color,
                    linewidth=width,
                    linestyle=_CURVE_DASH,
                    dash_capstyle="butt",
                    zorder=3,
                    path_effects=halo(width, sizes),
                )
                std = stats["std"].fillna(0.0)
                ax2.fill_between(
                    stats["severity"], stats["mean"] - std, stats["mean"] + std, **band_style(color, alpha=0.10)
                )
                has_accuracy = True
            set_ylabel(ax2, "accuracy", sizes)
            _direction_arrow(ax2, "$\\uparrow$", sizes)
            ax2.set_ylim(0.0, 1.0)
            style_axes(ax2, sizes)

    severities = sorted(int(s) for s in df["severity"].unique())
    ax.set_xticks(severities)
    ax.set_xticklabels(["clean" if s == 0 else str(s) for s in severities])  # as in the shift risk grid
    ax.margins(y=0.12)
    ax.autoscale_view()
    ax.set_ylim(bottom=0.0)
    ax.locator_params(axis="y", nbins=5)
    set_xlabel(ax, "corruption severity", sizes)
    set_ylabel(ax, "set size", sizes)
    # Diagonal arrow: under shift larger sets are the honest answer, while on clean data smaller
    # is better, so neither a plain up nor down arrow tells the truth.
    _direction_arrow(ax, "$\\nearrow$", sizes)
    style_axes(ax, sizes)

    entries = list(zip(*ax.get_legend_handles_labels(), strict=True))
    # Line-style key inside the axes at the left center: which style is the set size, which the
    # accuracy. Created before the method legend, which is a figure legend and leaves it in place.
    if has_accuracy:
        style_handles = [
            plt.Line2D([0], [0], color="0.35", linewidth=accuracy_width, linestyle=_CURVE_DASH, dash_capstyle="butt"),
            plt.Line2D(
                [0],
                [0],
                color="0.35",
                linewidth=sizes.line,
                marker="o",
                markersize=sizes.marker,
                solid_capstyle="round",
            ),
        ]
        style_leg = ax.legend(
            style_handles,
            ["accuracy", "set size"],
            loc="center left",
            bbox_to_anchor=(0.0, 0.62),
            frameon=True,
            fancybox=True,
            framealpha=1.0,
            facecolor="white",
            edgecolor="0.85",
            prop=font(FP_REGULAR, sizes.legend),
            handlelength=2.4,
            handletextpad=0.5,
        )
        style_leg.get_frame().set_linewidth(2 * sizes.spine)
    # Method legend below the axes: the baselines in up to two rows, ours set apart on the right.
    ours_entry = next((e for e in entries if e[1].endswith(OURS_TAG)), None)
    others = [e for e in entries if e is not ours_entry]
    ncol = len(others) if len(entries) <= 4 else -(-len(others) // 2)
    rows = [others[i : i + ncol] for i in range(0, len(others), ncol)]
    legend_below(fig, rows, ncol=ncol, sizes=sizes, ours=ours_entry, top=-0.01)
    return fig


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Edit these and re-run from PyCharm. None on filter fields means "no constraint".
    DATASETS: list[str] = ["bloodmnist", "cifar10"]  # the paper's main-text and appendix figure
    METHODS_TO_PLOT: list[str] | None = [
        "credal_ensembling",
        "credal_bnn",
        "credal_wrapper",
        "credal_dro",
        "credal_relative_likelihood",
        "efficient_credal_prediction",
        "credal_rl_multinomial",
    ]
    ALPHAS: list[float] | None = [0.95]  # None = all cached alphas
    SEEDS: list[int] | None = [1, 2, 3]  # None = average over all cached seeds
    PLOT_ACCURACY = True  # overlay maximax point accuracy (needs re-run sweeps; older caches lack the rows)
    SAVE = True

    for dataset in DATASETS:
        df = load_runs(filters={"dataset": dataset})
        print(f"{dataset}: loaded {len(df)} rows from {df['run_id'].nunique()} runs.")
        print_width = _PRINT_WIDTHS.get(dataset, _PRINT_WIDTH_APPENDIX)
        fig, sizes = render_for_print(
            lambda s, df=df: plot_shift_set_size(
                df, methods=METHODS_TO_PLOT, alphas=ALPHAS, seeds=SEEDS, plot_accuracy=PLOT_ACCURACY, sizes=s
            ),
            print_width,
        )
        if SAVE:
            save(fig, f"shift_set_size_{dataset}", PLOTS_PATH, print_width, sizes)
    plt.show()
