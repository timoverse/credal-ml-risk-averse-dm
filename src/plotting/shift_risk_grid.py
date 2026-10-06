"""Shift risk figure as a tau x metric grid: CVaR_tau (top row) and mean (bottom row) against severity.

A 2 x T grid, one column per CVaR level tau. Every panel plots the loss against the corruption
severity (0 = clean through 5), one line per arm, so each column answers "how does each method's
tail (top) and average (bottom) degrade as the data shifts, at this risk level". Values average
over the corruptions evaluated at a severity, then over seeds (band = plus/minus one std over
seeds). Rows share their y-axis, so the columns are directly comparable.

The arms, and where tau enters each, follow plotting/shift_risk_metric.py (_arm_tau): the
cvar_minimax arms (MLE + rule, CreWra + rule, CreWare + rule) are calibrated at tau, SQwash and
AdaCVaR are trained at tail fraction tau (one artifact per tau), and the MLE's CVaR is read at
tau while its mean does not depend on tau. A dotted line marks the constant uniform prediction,
whose CVaR and mean both equal ln K (log loss) or 1 - 1/K (Brier) at every severity -- the trivial
reference any tail-risk claim has to be read against.

This figure is the paper's style reference: its colors, type and strokes live in
plotting/paper_style.py, which every other figure draws from. The two MLE arms are both neutral
grays, the plain one darker, and are told apart by line style (solid vs dashed) and marker as well.

Run from src/:  python -m plotting.shift_risk_grid --dataset pathmnist bloodmnist
Single-column main-paper version (three taus, saved as ..._tau001-005-02):
                python -m plotting.shift_risk_grid --dataset pathmnist --taus 0.01 0.05 0.2
Appendix alpha ablation: python -m plotting.shift_risk_grid --alpha-ablation
Data comes from the W&B cache (plotting/wandb_cache.py), kind="shift" rows.
"""

from __future__ import annotations

import argparse
import logging
import sys
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import PLOTS_PATH
from plotting.paper_style import (
    COLUMN_WIDTH,
    FP_SEMIBOLD,
    METHODS,
    OUR_RULE,
    OURS,
    OURS_TAG,
    REFERENCE_DOTS,
    REFERENCE_GRAY,
    TEXT_WIDTH,
    Sizes,
    align_ylabels,
    band_style,
    legend_below,
    line_style,
    ours_label,
    render_for_print,
    save,
    set_header,
    set_ylabel,
    style_axes,
    tint,
    use_fira_mathtext,
)
from plotting.shift_risk_metric import _arm_tau, _filter_shift_rows
from plotting.wandb_cache import load_runs

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Arm -> (its method in paper_style.METHODS, label), in legend order. Keyed by
# (method.name, decision_rule).
_GRID_ARMS: dict[tuple[str, str], tuple[str, str]] = {
    ("base", "mle"): ("base", "MLE"),
    ("base", "cvar_minimax"): ("base+rule", f"MLE {OUR_RULE}"),
    ("sqwash", "mle"): ("sqwash", "SQwash"),
    ("adacvar", "mle"): ("adacvar", "AdaCVaR"),
    ("credal_wrapper", "cvar_minimax"): ("credal_wrapper", f"CreWra {OUR_RULE}"),
    (OURS, "cvar_minimax"): (OURS, ours_label("CreWare + rule")),
}
_OURS = (OURS, "cvar_minimax")
_NUM_CLASSES = {"pathmnist": 9, "bloodmnist": 8}
_TAUS = [0.01, 0.025, 0.05, 0.1, 0.2]
# Figure-fraction y of the legend's top edge: a little air under the x label.
_LEGEND_TOP = 0.0
_MAIN_TEXT_TAUS = 3  # up to this many columns the grid is the single-column main-text figure
# Plausibility levels of the CreWare alpha ablation, the grid of the coverage-efficiency appendix.
_ABLATION_ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0]


def _uniform_loss(loss: str, num_classes: int) -> float | None:
    """Loss of the uniform prediction, identical on every instance: ln K (log loss) or 1 - 1/K (Brier).

    Args:
        loss: Per-instance loss name.
        num_classes: Number of classes K.

    Returns:
        The constant loss, or None for a loss without a closed form here (no reference line).
    """
    if loss == "log_loss":
        return float(np.log(num_classes))
    if loss == "brier":
        # (1 - 1/K)^2 on the true class plus (K - 1) / K^2 on the others.
        return 1.0 - 1.0 / num_classes
    return None


def _arm_stats(sub: pd.DataFrame, method: str, rule: str, taus: list[float], alpha: float | None) -> pd.DataFrame:
    """Per (metric, tau, severity): mean and std over seeds of the corruption-averaged loss.

    Args:
        sub: Deduplicated shift rows of one dataset and loss.
        method: Arm method name.
        rule: Arm decision rule.
        taus: The figure's tau grid.
        alpha: method.train.alpha of the credal-relative-likelihood arm to keep (the paper's 0.95);
            ignored for the other arms, whose alpha is absent or is the tau coordinate itself.

    Returns:
        Columns metric, tau, severity, mean, std, and n_corr / n_corr_max: the fewest and most
        corruptions any seed averaged over (they differ when a seed has missing or NaN cells).
    """
    arm = sub[(sub["method.name"] == method) & (sub["decision_rule"] == rule)]
    if method == "credal_rl_multinomial" and alpha is not None:
        arm = arm[np.isclose(arm["method.train.alpha"].astype(float), alpha)]
    if arm.empty:
        return arm
    arm = _arm_tau(arm, method, rule, taus)
    arm = arm[arm["tau"].isin(taus)]
    per_seed = arm.groupby(["metric", "tau", "severity", "seed"])["value"].agg(["mean", "size"])
    stats = per_seed.groupby(["metric", "tau", "severity"]).agg(
        mean=("mean", "mean"), std=("mean", "std"), n_corr=("size", "min"), n_corr_max=("size", "max")
    )
    return stats.reset_index()


def _warn_short(stats: pd.DataFrame, dataset: str, name: str) -> None:
    """Warn when a seed averaged fewer corruptions than another (missing or NaN cells bias the mean).

    Args:
        stats: Output of _arm_stats for one line.
        dataset: Dataset name, for the message.
        name: Line name, for the message.
    """
    short = stats[stats["n_corr"] < stats["n_corr_max"]]
    if not short.empty:
        logger.warning(
            "%s %s: a seed averages fewer corruptions than the others (missing or NaN cells) at (tau, severity) %s",
            dataset,
            name,
            sorted({(float(t), int(s)) for t, s in zip(short["tau"], short["severity"], strict=True)}),
        )


def _new_grid(taus: list[float]) -> tuple[Figure, np.ndarray]:
    """Empty 2 x len(taus) grid with shared x and row-shared y, in the paper's fonts.

    Args:
        taus: CVaR levels, one column each.

    Returns:
        The figure and its (2, len(taus)) axes array.
    """
    use_fira_mathtext()
    fig, axes = plt.subplots(
        2, len(taus), figsize=(1.45 * len(taus), 3.1), sharex=True, sharey="row", constrained_layout=True
    )
    return fig, np.atleast_2d(axes)


def _draw_line(axes: np.ndarray, stats: pd.DataFrame, taus: list[float], label: str, style: dict) -> bool:
    """Draw one line (mean over seeds, plus/minus one std band) into every panel of the grid.

    Args:
        axes: (2, len(taus)) axes array from _new_grid.
        stats: Output of _arm_stats for the line.
        taus: CVaR levels, one column each.
        label: Legend label.
        style: ax.plot keyword arguments (paper_style.line_style); the bands sit below every line.

    Returns:
        Whether any point was drawn.
    """
    drawn = False
    for row, metric in enumerate(("cvar", "mean")):
        for col, tau in enumerate(taus):
            pts = stats[(stats["metric"] == metric) & (stats["tau"] == tau)].sort_values("severity")
            if pts.empty:
                continue
            ax = axes[row, col]
            ax.plot(pts["severity"], pts["mean"], label=label, **style)
            band = pts["std"].fillna(0.0)
            ax.fill_between(pts["severity"], pts["mean"] - band, pts["mean"] + band, **band_style(style["color"]))
            drawn = True
    return drawn


def _decorate_grid(  # noqa: PLR0913
    fig: Figure,
    axes: np.ndarray,
    taus: list[float],
    severities: list[int],
    dataset: str,
    loss: str,
    uniform: bool,
    sizes: Sizes,
) -> float:
    """Column headers, uniform reference, severity ticks and axis labels. No figure title.

    Args:
        fig: The grid figure.
        axes: Its (2, len(taus)) axes array.
        taus: CVaR levels, one column each.
        severities: Severities on the x-axis (0 = clean).
        dataset: Dataset name, for the uniform reference.
        loss: Per-instance loss plotted.
        uniform: Draw the constant uniform prediction as a dotted reference.
        sizes: Type sizes and stroke widths.

    Returns:
        Figure-fraction x of the centre of the panel columns, which the x label is centred on and
        the legend should be too.
    """
    uniform_level = _uniform_loss(loss, _NUM_CLASSES[dataset]) if uniform and dataset in _NUM_CLASSES else None
    # "Brier", not "Brier score": the longer y labels overrun the panel height and the two rows collide.
    loss_label = {"log_loss": "log loss", "brier": "Brier"}.get(loss, loss)
    for col, tau in enumerate(taus):
        set_header(axes[0, col], f"τ = {tau:g}", sizes)
        for row in range(2):
            ax: Axes = axes[row, col]
            if uniform_level is not None:
                ax.axhline(
                    uniform_level,
                    color=REFERENCE_GRAY,
                    linewidth=sizes.reference_line,
                    linestyle=REFERENCE_DOTS,
                    zorder=1,
                    label="uniform prediction",
                )
            ax.set_xticks(severities)
            ax.set_xticklabels(["clean" if s == 0 else str(s) for s in severities])
            ax.set_ylim(bottom=0.0)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.minorticks_off()
            style_axes(ax, sizes)
    set_ylabel(axes[0, 0], f"CVaR$_τ$ of {loss_label}", sizes)
    set_ylabel(axes[1, 0], f"mean {loss_label}", sizes)
    # The rows' tick labels differ in width (two digits against one), which would stagger the labels.
    align_ylabels(fig, axes[:, 0])
    xlabel = fig.supxlabel("corruption severity", fontproperties=FP_SEMIBOLD, fontsize=sizes.label)
    # The figure's own centre lies left of the panels', by half the margin the y labels take, and an
    # x label or a legend centred there sits visibly off the panels it belongs to.
    fig.canvas.draw()  # constrained_layout places the panels at draw time
    center = (axes[0, 0].get_position().x0 + axes[0, -1].get_position().x1) / 2
    xlabel.set_x(center)
    return center


def _legend_entries(ax: Axes) -> list[tuple]:
    """(handle, label) pairs of a panel in drawing order, one per label."""
    handles, labels = [], []
    for handle, lab in zip(*ax.get_legend_handles_labels(), strict=True):
        if lab not in labels:
            handles.append(handle)
            labels.append(lab)
    return list(zip(handles, labels, strict=True))


def plot_shift_risk_grid(
    df: pd.DataFrame,
    dataset: str,
    taus: list[float] | None = None,
    alpha: float = 0.95,
    seeds: list[int] | None = None,
    loss: str = "log_loss",
    uniform: bool = True,
    legend_rows: int = 2,
    sizes: Sizes | None = None,
) -> Figure:
    """Build the 2 x len(taus) grid: CVaR_tau (top) and mean (bottom) of the loss against severity.

    Args:
        df: Long-form cache rows (load_runs), already restricted to the dataset.
        dataset: Dataset name, for the uniform reference and the default title.
        taus: CVaR levels, one column each. None uses all five levels 0.01 to 0.2.
        alpha: Plausibility level of the CreWare arm.
        seeds: Restrict to these seeds. None uses all present.
        loss: Per-instance loss to plot.
        uniform: Draw the constant uniform prediction (CVaR = mean, see _uniform_loss) as a dotted reference.
        legend_rows: Rows of the baseline legend grid; two rows make it 2 x 3 with the uniform
            reference. Ours sits on a line of its own below the grid, under the middle column.
        sizes: Type sizes and stroke widths. None uses the reference's.

    Returns:
        The assembled matplotlib Figure.

    Raises:
        ValueError: nothing was plotted.
    """
    taus = taus or _TAUS
    sizes = sizes or Sizes()
    sub = _filter_shift_rows(df, seeds, None, loss)
    severities = sorted(int(s) for s in sub["severity"].dropna().unique())

    fig, axes = _new_grid(taus)
    plotted = False
    for (method, rule), (key, label) in _GRID_ARMS.items():
        stats = _arm_stats(sub, method, rule, taus, alpha)
        if stats.empty:
            logger.warning("No %s rows for arm %s/%s; skipping.", dataset, method, rule)
            continue
        _warn_short(stats, dataset, f"{method}/{rule}")
        plotted |= _draw_line(axes, stats, taus, label, line_style(key, sizes))
    if not plotted:
        raise ValueError(f"Nothing was plotted for {dataset}; check the cache rows.")
    center = _decorate_grid(fig, axes, taus, severities, dataset, loss, uniform, sizes)

    entries = _legend_entries(axes[0, 0])
    ours_entry = next((e for e in entries if e[1].endswith(OURS_TAG)), None)
    others = [e for e in entries if e is not ours_entry]
    ncol = -(-len(others) // legend_rows)
    rows = [others[i : i + ncol] for i in range(0, len(others), ncol)]
    # Ours under the middle column: the three arms that use our rule then stack in one column.
    legend_below(
        fig,
        rows,
        ncol=ncol,
        sizes=sizes,
        ours=ours_entry,
        ours_column=ncol // 2,
        top=_LEGEND_TOP,
        center=center,
    )
    return fig


def plot_shift_risk_alpha_grid(
    df: pd.DataFrame,
    dataset: str,
    taus: list[float] | None = None,
    alphas: list[float] | None = None,
    seeds: list[int] | None = None,
    loss: str = "log_loss",
    uniform: bool = True,
    legend_rows: int = 2,
    sizes: Sizes | None = None,
) -> Figure:
    """Alpha ablation on the same grid: CreWare + rule alone, one line per plausibility level alpha.

    Lines are shaded within CreWare's red by the rank of alpha, from a light tint (smallest) to
    the full color (largest). Rank rather than value spacing, because the grid crowds at the top
    (0.9, 0.95, 1), where a value-proportional shade would make three lines look identical.

    Args:
        df: Long-form cache rows (load_runs), already restricted to the dataset.
        dataset: Dataset name, for the uniform reference and the default title.
        taus: CVaR levels, one column each. None uses all five levels 0.01 to 0.2.
        alphas: Plausibility levels, one line each. None uses the ablation grid 0 to 1.
        seeds: Restrict to these seeds. None uses all present.
        loss: Per-instance loss to plot.
        uniform: Draw the constant uniform prediction as a dotted reference.
        legend_rows: Rows the alpha entries are split into (the uniform reference ends the first).
        sizes: Type sizes and stroke widths. None uses the reference's.

    Returns:
        The assembled matplotlib Figure.

    Raises:
        ValueError: nothing was plotted.
    """
    taus = taus or _TAUS
    alphas = sorted(alphas or _ABLATION_ALPHAS)
    sizes = sizes or Sizes()
    sub = _filter_shift_rows(df, seeds, None, loss)
    severities = sorted(int(s) for s in sub["severity"].dropna().unique())

    fig, axes = _new_grid(taus)
    plotted = False
    for rank, alpha in enumerate(alphas):
        stats = _arm_stats(sub, *_OURS, taus, alpha)
        if stats.empty:
            logger.warning("No %s rows for CreWare at alpha %g; skipping.", dataset, alpha)
            continue
        _warn_short(stats, dataset, f"CreWare alpha={alpha:g}")
        # The floor keeps the smallest alpha visible against the white surface.
        t = 0.35 + 0.65 * rank / max(len(alphas) - 1, 1)
        # Stronger shades on top, so the dense high-alpha end stays readable where lines cross.
        style = line_style(OURS, sizes) | {"color": tint(METHODS[OURS].color, t), "zorder": 3 + t}
        plotted |= _draw_line(axes, stats, taus, f"α = {alpha:g}", style)
    if not plotted:
        raise ValueError(f"Nothing was plotted for {dataset}; check the cache rows.")
    center = _decorate_grid(fig, axes, taus, severities, dataset, loss, uniform, sizes)

    # The alphas fill even rows in order; the uniform reference ends the first row, so its long
    # label widens only the last column instead of one in the middle of the alpha sequence.
    entries = _legend_entries(axes[0, 0])
    lines = [e for e in entries if e[1].startswith("α")]
    refs = [e for e in entries if not e[1].startswith("α")]
    per_row = -(-len(lines) // legend_rows)
    rows = [lines[i : i + per_row] for i in range(0, len(lines), per_row)]
    rows[0] += refs
    legend_below(
        fig,
        rows,
        ncol=max(len(row) for row in rows),
        sizes=sizes,
        top=_LEGEND_TOP,
        center=center,
    )
    return fig


def main() -> None:
    """Render the grid for each requested dataset to PLOTS_PATH as pdf and png."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", nargs="+", default=["pathmnist", "bloodmnist"])
    parser.add_argument("--taus", nargs="+", type=float, default=_TAUS)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--loss", default="log_loss")
    parser.add_argument("--no-uniform", action="store_true", help="omit the uniform-prediction reference line")
    parser.add_argument("--legend-rows", type=int, default=2, help="rows of the legend grid (ours gets its own line)")
    parser.add_argument(
        "--print-width",
        type=float,
        default=None,
        help="inches the paper includes the figure at; default 0.9 of a column for up to three taus "
        "(main text) and 0.8 of the text width beyond (appendix)",
    )
    parser.add_argument(
        "--alpha-ablation",
        nargs="*",
        type=float,
        default=None,
        metavar="ALPHA",
        help="draw the CreWare alpha ablation instead (bare flag: alphas 0 to 1 of the appendix grid)",
    )
    args = parser.parse_args()
    # A tau subset (e.g. the single-column main-paper version) gets its own file name, so it never
    # overwrites the full five-column grid. No dots in the tag (0.05 -> 005): LaTeX's graphicx can
    # mistake everything after the first dot of a file name for its extension.
    tau_tag = "" if args.taus == _TAUS else "_tau" + "-".join(f"{t:g}".replace(".", "") for t in args.taus)
    print_width = args.print_width or (0.9 * COLUMN_WIDTH if len(args.taus) <= _MAIN_TEXT_TAUS else 0.8 * TEXT_WIDTH)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    common = {"taus": args.taus, "loss": args.loss, "uniform": not args.no_uniform}
    for dataset in args.dataset:
        df = load_runs({"kind": "shift", "dataset": dataset})
        if args.alpha_ablation is not None:
            build = partial(
                plot_shift_risk_alpha_grid,
                df,
                dataset,
                alphas=args.alpha_ablation or None,
                legend_rows=args.legend_rows,
                **common,
            )
            stem = "shift_risk_alpha_grid"
        else:
            build = partial(plot_shift_risk_grid, df, dataset, alpha=args.alpha, legend_rows=args.legend_rows, **common)
            stem = "shift_risk_grid"
        fig, sizes = render_for_print(lambda sizes, build=build: build(sizes=sizes), print_width)
        save(fig, f"{stem}_{dataset}_{args.loss}{tau_tag}", PLOTS_PATH, print_width, sizes)
        plt.close(fig)


if __name__ == "__main__":
    main()
