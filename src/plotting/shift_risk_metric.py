"""Paper figure for the shift risk experiment: own-level CVaR and mean against tau, per severity.

A 1 x S row of panels, one per shift severity (0 = clean through 5), with twin y-axes in the
style of plotting/shift_set_size.py's set-size/accuracy overlay: solid curves on the left axis
show the CVaR_tau of each arm's predictions, dashed curves on the right axis the mean loss of
the same predictions, both against the CVaR level tau on a log x-axis. Values average over the
corruptions evaluated at that severity, then over seeds (band = plus/minus one std over seeds).
Styling follows the same driving-figure family: flat palette, Fira Sans, halos, legend below,
with a line-style key tying solid to CVaR and dashed to mean.

The arms differ in where tau enters, so the tau coordinate comes from a different column per arm:
cvar_minimax rows carry it as cvar_beta (calibration level); sqwash and adacvar carry it as
method.train.alpha (the tail fraction they trained on, one artifact per tau); MLE rows are
tau-independent, so their mean is a flat reference line and their cvar_<tau> aggregations spread
over the x-axis. This file complements plotting/risk_metric.py (the exploratory severity-shaded
single-panel view of the same rows): here severities are panels and the mle baselines are in.

Data comes from the W&B cache (see wandb_cache.py), parsed from the shift/* summary keys written
by experiments/shift_risk_metric.py (kind="shift" rows). A dummy-data generator with the same
schema renders the figure before the cluster runs land; flip DUMMY in __main__.
"""

from __future__ import annotations

import logging
import sys
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import PLOTS_PATH
from plotting.paper_style import (
    FP_REGULAR,
    FP_SEMIBOLD,
    OURS_TAG,
    Sizes,
    fira,
    legend_below,
    ours_label,
    style_axes,
)
from plotting.paper_style import halo as _halo
from plotting.paper_style import tint as _tint
from plotting.shift_set_size import _CURVE_DASH
from plotting.wandb_cache import load_runs

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# Style per arm, keyed by (method.name, decision_rule): (fmt, color, label, linewidth). An arm
# is a curve family; base appears twice (risk-neutral MLE reference and the singleton
# cvar_minimax ablation). Colors live in the shift_set_size palette family: ours wears the
# pink/red, the MLE reference the near-black gray, the ablation a lighter gray, and the two
# tail-trained baselines the teal/purple slots unused by the credal methods in this figure.
_ARM_STYLES: dict[tuple[str, str], tuple[str, str, str, float]] = {
    ("base", "mle"): ("o", "#454545", "MLE", 1.5),
    ("sqwash", "mle"): ("s", "#3bb5c3", "SQwash", 1.5),
    ("adacvar", "mle"): ("D", "#8f5fd6", "AdaCVaR", 1.5),
    ("base", "cvar_minimax"): ("v", "#9a9a9a", "MLE + CVaR-minimax", 1.5),
    ("credal_rl_multinomial", "cvar_minimax"): ("^", "#ea365b", "CreWare", 1.8),
}
_OURS = ("credal_rl_multinomial", "cvar_minimax")


# Type sizes: the paper-wide reference sizes (paper_style.Sizes), which were tuned on this
# experiment's single-column figure.
_SIZES = Sizes()
FS_LABEL = _SIZES.label
FS_TITLE = _SIZES.header
FS_TICK = _SIZES.tick
FS_LEGEND = _SIZES.legend

FP_THIN = fira("Thin")  # in-panel severity captions

# The paper-wide legend, with the shorter handle these figures' solid lines get by with.
_legend_below = partial(legend_below, handlelength=1.8)


def _panel_grid(
    n_panels: int, ncols: int | None, panel_w: float, panel_h: float, share: bool = True
) -> tuple[Figure, list[Axes]]:
    """Lay out n_panels, either in one row or in a grid of ncols columns.

    Args:
        n_panels: Number of severity panels.
        ncols: Columns in the grid. None puts every panel in a single row (the wide layout);
            2 with four severities gives the single-column quadrant layout.
        panel_w: Width per panel in inches.
        panel_h: Height per panel in inches.
        share: Share both axes across panels, so values are directly comparable between
            severities. False autoscales each panel, which fills the low-severity panels
            (whose losses are small) at the cost of cross-panel comparability.

    Returns:
        (figure, axes in panel order). Unused cells of a partly filled grid are hidden.
    """
    cols = ncols or n_panels
    rows = -(-n_panels // cols)  # ceil
    fig, axes_arr = plt.subplots(
        rows,
        cols,
        figsize=(panel_w * cols, panel_h * rows),
        sharex=share,
        sharey=share,
        constrained_layout=True,
    )
    flat: list[Axes] = list(np.atleast_1d(axes_arr).ravel())
    for ax in flat[n_panels:]:
        ax.set_visible(False)
    return fig, flat[:n_panels]


def _style_panel(ax: Axes) -> None:
    """Paper-wide panel chrome at the reference sizes; call once the ticks are final."""
    style_axes(ax, _SIZES)


def _arm_tau(sub: pd.DataFrame, method: str, rule: str, taus: list[float]) -> pd.DataFrame:
    """Attach the arm's tau coordinate and keep only its mean and own-level cvar rows.

    Args:
        sub: Shift rows of one (method, decision_rule) arm.
        method: Method name of the arm.
        rule: Decision rule of the arm.
        taus: The tau grid of the figure, used to broadcast tau-independent MLE means.

    Returns:
        Rows with columns metric ("mean" or "cvar") and tau added; other rows dropped.
    """
    import pandas as pd  # noqa: PLC0415

    if rule == "cvar_minimax":
        sub = sub.assign(tau=sub["cvar_beta"].astype(float))
        own = sub["aggregation"] == sub["tau"].map(lambda t: f"cvar_{t:g}")
        mean = sub["aggregation"] == "mean"
        return pd.concat([sub[mean].assign(metric="mean"), sub[own].assign(metric="cvar")], ignore_index=True)
    if method in ("sqwash", "adacvar"):
        # One artifact per tau: the training tail fraction is the tau coordinate, and each run
        # records the mean plus its own matched cvar aggregation.
        sub = sub.assign(tau=sub["method.train.alpha"].astype(float))
        own = sub["aggregation"] == sub["tau"].map(lambda t: f"cvar_{t:g}")
        mean = sub["aggregation"] == "mean"
        return pd.concat([sub[mean].assign(metric="mean"), sub[own].assign(metric="cvar")], ignore_index=True)
    # MLE: cvar rows get tau from the aggregation key; the tau-independent mean is replicated
    # across the figure's tau grid so it draws as a flat reference line.
    cvar = sub[sub["aggregation"].str.startswith("cvar_")].copy()
    cvar["tau"] = cvar["aggregation"].str.removeprefix("cvar_").astype(float)
    cvar["metric"] = "cvar"
    mean = sub[sub["aggregation"] == "mean"]
    means = pd.concat([mean.assign(tau=t, metric="mean") for t in taus], ignore_index=True)
    return pd.concat([means, cvar], ignore_index=True)


def _filter_shift_rows(
    df: pd.DataFrame,
    seeds: list[int] | None,
    corruptions: list[str] | None,
    loss: str,
) -> pd.DataFrame:
    """Filter to the loss's shift rows, apply seed/corruption filters, and dedup to latest runs.

    Args:
        df: Long-form DataFrame from load_runs() or _dummy_df().
        seeds: Restrict to these seeds. None keeps all.
        corruptions: Restrict to these corruption names (clean rows always kept). None keeps all.
        loss: Per-instance loss whose rows to keep.

    Returns:
        The filtered frame.

    Raises:
        ValueError: no rows survive the filters.
    """
    sub = df[(df["kind"] == "shift") & (df["loss"] == loss)]
    if seeds is not None:
        sub = sub[sub["seed"].isin(seeds)]
    if corruptions is not None:
        sub = sub[sub["corruption"].isin([*corruptions, "clean"])]
    for col in ("method.train.alpha", "num_train"):
        if col not in sub.columns:
            sub = sub.assign(**{col: float("nan")})
    if "created_at" in sub.columns:
        identity = [
            "method.name",
            "decision_rule",
            "method.train.alpha",
            "num_train",
            "seed",
            "cvar_beta",
            "corruption",
            "severity",
            "aggregation",
        ]
        sub = sub.sort_values("created_at", kind="mergesort").drop_duplicates(subset=identity, keep="last")
    if sub.empty:
        raise ValueError(f"No shift rows for loss={loss!r} with the given filters.")
    return sub


def plot_shift_risk_metric(
    df: pd.DataFrame,
    taus: list[float],
    alphas: list[float] | None = None,
    seeds: list[int] | None = None,
    severities: list[int] | None = None,
    corruptions: list[str] | None = None,
    loss: str = "log_loss",
    ncols: int | None = 2,
    share_axes: bool = True,
    title: str | None = "CVaR and mean under distribution shift",
) -> Figure:
    """Build the twin-axis tau-sweep figure: CVaR_tau (left, solid) and mean (right, dashed) vs tau.

    The appendix companion to plot_shift_risk_pareto: same rows, but tau is an axis rather than
    a marker shade, which is easier to read off at a specific tau.

    Each curve is one arm from _ARM_STYLES (the credal arm contributes one curve per selected
    alpha, labeled with the alpha as a subscript). Points average the aggregation over that
    severity's corruptions, then over seeds; the band is plus/minus one std over seeds (absent
    for a single seed). The left (CVaR) axis is shared across panels, and the right (mean) axes
    are unified to one range after plotting, so both metrics read consistently left to right.

    Args:
        df: Long-form DataFrame from load_runs() or _dummy_df().
        taus: CVaR levels on the x-axis, ascending.
        alphas: Restrict the alpha-swept credal arm to these method.train.alpha values. Arms
            without an alpha knob are always kept. None keeps all alphas present.
        seeds: Restrict to these seeds. None uses all seeds present.
        severities: Severity panels, in reading order. None uses all present, sorted.
        corruptions: Restrict to these corruption names (severity 0 rows, always named clean,
            are always kept). Use this to hold the corruption set fixed when the cache mixes
            protocols. None uses all present.
        loss: Per-instance loss to plot.
        ncols: Columns in the panel grid; None puts them all in one row (the wide layout).
        share_axes: Share both axes across panels so severities are directly comparable. False
            autoscales each panel, filling the low-severity ones at the cost of comparability.
        title: Figure title above the panel row. None omits it.

    Returns:
        The assembled matplotlib Figure.

    Raises:
        ValueError: no rows match the filters, or nothing was plotted.
    """
    import pandas as pd  # noqa: PLC0415

    if FP_REGULAR.get_file() is not None:
        # Route mathtext through Fira so the alpha subscripts match the surrounding text.
        plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Fira Sans", "mathtext.default": "regular"})

    sub = _filter_shift_rows(df, seeds, corruptions, loss)

    if severities is None:
        severities = sorted(int(s) for s in sub["severity"].dropna().unique())

    fig, axes = _panel_grid(len(severities), ncols, panel_w=1.9, panel_h=1.85, share=share_axes)
    twins: list[Axes] = [ax.twinx() for ax in axes]
    for ax, twin in zip(axes, twins, strict=True):
        # Draw the CVaR axes above the twin, so the solid curves' halos win at crossings.
        ax.set_zorder(twin.get_zorder() + 1)
        ax.patch.set_visible(False)

    for (method, rule), (marker, color, label, lw) in _ARM_STYLES.items():
        arm = sub[(sub["method.name"] == method) & (sub["decision_rule"] == rule)]
        if arm.empty:
            logger.warning("No rows for arm %s/%s; skipping.", method, rule)
            continue
        arm = _arm_tau(arm, method, rule, taus)
        arm = arm[arm["tau"].isin(taus)]
        # The credal arm draws one curve per selected alpha; everything else is a single curve
        # (their alpha is either absent or the tau coordinate itself).
        is_alpha_swept = rule == "cvar_minimax" and method != "base"
        if is_alpha_swept and alphas is not None:
            arm = arm[arm["method.train.alpha"].isin(alphas)]
        alpha_values = sorted(arm["method.train.alpha"].dropna().unique()) if is_alpha_swept else [float("nan")]
        for alpha_value in alpha_values:
            curve_rows = arm[arm["method.train.alpha"] == alpha_value] if pd.notna(alpha_value) else arm
            curve_label = label
            if pd.notna(alpha_value):
                curve_label = f"{label}$_{{{alpha_value:g}}}$"
            if (method, rule) == _OURS:
                curve_label = ours_label(curve_label)  # set in semibold in the legend
            # Average over corruptions per (severity, tau, seed), then stats over seeds.
            cells = curve_rows.groupby(["metric", "severity", "tau", "seed"], dropna=False)["value"].mean()
            stats = cells.groupby(["metric", "severity", "tau"]).agg(["mean", "std"]).reset_index()
            for col_idx, severity in enumerate(severities):
                cvar_pts = stats[(stats["metric"] == "cvar") & (stats["severity"] == severity)].sort_values("tau")
                if not cvar_pts.empty:
                    ax = axes[col_idx]
                    (line,) = ax.plot(
                        cvar_pts["tau"],
                        cvar_pts["mean"],
                        color=color,
                        linewidth=lw,
                        marker=marker,
                        markersize=3.5,
                        solid_capstyle="round",
                        zorder=4 if (method, rule) == _OURS else 3,
                        path_effects=_halo(lw),
                        label=curve_label,
                    )
                    band = cvar_pts["std"].fillna(0.0)
                    ax.fill_between(
                        cvar_pts["tau"],
                        cvar_pts["mean"] - band,
                        cvar_pts["mean"] + band,
                        color=color,
                        alpha=0.16,
                        linewidth=0,
                        zorder=2,
                    )
                mean_pts = stats[(stats["metric"] == "mean") & (stats["severity"] == severity)].sort_values("tau")
                if not mean_pts.empty:
                    twin = twins[col_idx]
                    twin.plot(
                        mean_pts["tau"],
                        mean_pts["mean"],
                        color=color,
                        linewidth=1.3,
                        linestyle=_CURVE_DASH,
                        dash_capstyle="butt",
                        zorder=3,
                        path_effects=_halo(1.3),
                    )
                    band = mean_pts["std"].fillna(0.0)
                    twin.fill_between(
                        mean_pts["tau"],
                        mean_pts["mean"] - band,
                        mean_pts["mean"] + band,
                        color=color,
                        alpha=0.10,
                        linewidth=0,
                        zorder=2,
                    )

    loss_label = {"log_loss": "Log loss", "brier": "Brier score"}.get(loss, loss)
    cols = ncols or len(severities)
    # In a narrow grid five tau labels collide, so label only the ends and the middle there.
    tick_taus = taus if cols >= len(severities) else [taus[0], taus[len(taus) // 2], taus[-1]]
    for idx, severity in enumerate(severities):
        axes[idx].set_title(
            "Clean" if severity == 0 else f"Severity {severity}", fontproperties=FP_REGULAR, fontsize=FS_TITLE
        )
        axes[idx].set_xscale("log")
        axes[idx].set_xticks(tick_taus)
        axes[idx].set_xticklabels([f"{t:g}" for t in tick_taus])
        axes[idx].minorticks_off()
        if idx % cols == 0:
            axes[idx].set_ylabel(f"CVaR$_τ$ {loss_label}", fontproperties=FP_SEMIBOLD, fontsize=FS_LABEL)
    axes[0].set_ylim(bottom=0.0)
    # One unified range for the right-hand mean axes; tick labels only on the rightmost column,
    # mirroring how sharey hides the interior left-hand labels.
    mean_top = max(twin.get_ylim()[1] for twin in twins)
    for idx, twin in enumerate(twins):
        twin.set_ylim(0.0, mean_top)
        twin.minorticks_off()
        last_in_row = idx % cols == cols - 1 or idx == len(twins) - 1
        if not last_in_row:
            twin.tick_params(labelright=False)
        elif idx % cols == cols - 1:
            twin.set_ylabel(f"Mean {loss_label}", fontproperties=FP_SEMIBOLD, fontsize=FS_LABEL, labelpad=4)
    fig.supxlabel("CVaR level τ", fontproperties=FP_SEMIBOLD, fontsize=FS_LABEL)
    for ax in (*axes, *twins):
        _style_panel(ax)
    if title:
        fig.suptitle(title, fontproperties=FP_REGULAR, fontsize=FS_TITLE + 1)

    # Legend below the figure: a line-style key (solid = CVaR, dashed = mean) then one entry
    # per arm, in _ARM_STYLES order. Anchored outside the constrained-layout area, so it cannot
    # collide with the supxlabel; savefig(bbox_inches="tight") includes it in the saved file.
    handles: list = []
    labels: list[str] = []
    for ax in axes:
        for handle, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            if lab not in labels:
                handles.append(handle)
                labels.append(lab)
    if not handles:
        raise ValueError("Nothing was plotted; check the arm filters against the cached rows.")
    style_handles = [
        plt.Line2D([0], [0], color="0.35", linewidth=1.5, solid_capstyle="round"),
        plt.Line2D([0], [0], color="0.35", linewidth=1.3, linestyle=_CURVE_DASH, dash_capstyle="butt"),
    ]
    style_labels = [f"CVaR$_τ$ {loss_label}", f"Mean {loss_label}"]
    # Key row, then the baselines over two rows; ours sits apart on the right, centred.
    entries = list(zip(handles, labels, strict=True))
    ours = next((e for e in entries if e[1].endswith(OURS_TAG)), None)
    others = [e for e in entries if e is not ours]
    split = -(-len(others) // 2)  # ceil, so the first baseline row is the longer one
    _legend_below(
        fig,
        [list(zip(style_handles, style_labels, strict=True)), others[:split], others[split:]],
        ncol=max(split, len(style_labels)),
        ours=ours,
    )
    return fig


def plot_shift_risk_pareto(
    df: pd.DataFrame,
    taus: list[float],
    alphas: list[float] | None = None,
    seeds: list[int] | None = None,
    severities: list[int] | None = None,
    corruptions: list[str] | None = None,
    loss: str = "log_loss",
    ncols: int | None = 2,
    share_axes: bool = True,
    title: str | None = "CVaR and mean under distribution shift",
) -> Figure:
    """Build the trade-off figure: mean loss (x) against own-level CVaR_tau (y), one panel per severity.

    The risk-return view, and the main paper figure: each arm is a path of len(taus) points
    through (mean, CVaR) space, marker shade encoding tau (dark = smallest, deepest tail; light
    = largest). Lower-left dominates on both objectives, so the frontier is read directly
    instead of compared across parallel curves. MLE traces a vertical path (its mean does not
    react to tau); the cvar_minimax arms bend as their calibration deepens, and the slope of
    that bend is the price of tail protection. Points average over the severity's corruptions
    then over seeds; thin crosshairs are plus/minus one std over seeds. All panels share both
    axes, so the cluster's march across severities is comparable.

    Args:
        df: Long-form DataFrame from load_runs() or _dummy_df().
        taus: CVaR levels traced along each path, ascending.
        alphas: Restrict the alpha-swept credal arm to these method.train.alpha values. Arms
            without an alpha knob are always kept. None keeps all alphas present.
        seeds: Restrict to these seeds. None uses all seeds present.
        severities: Severity panels, in reading order. None uses all present, sorted. Four
            severities with ncols=2 gives the single-column quadrant layout.
        corruptions: Restrict to these corruption names (clean rows always kept). None keeps all.
        loss: Per-instance loss to plot.
        ncols: Columns in the panel grid; None puts them all in one row (the wide layout).
        share_axes: Share both axes across panels so severities are directly comparable. False
            autoscales each panel, filling the low-severity ones at the cost of comparability.
        title: Figure title above the panel row. None omits it.

    Returns:
        The assembled matplotlib Figure.

    Raises:
        ValueError: no rows match the filters, or nothing was plotted.
    """
    import pandas as pd  # noqa: PLC0415

    if FP_REGULAR.get_file() is not None:
        plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Fira Sans", "mathtext.default": "regular"})

    sub = _filter_shift_rows(df, seeds, corruptions, loss)
    if severities is None:
        severities = sorted(int(s) for s in sub["severity"].dropna().unique())

    # Panel width is set so the two-column grid is at least as wide as the legend block below it
    # (~3.9 in at FS_LEGEND); narrower panels let the legend dictate the figure's cropped width.
    fig, axes = _panel_grid(len(severities), ncols, panel_w=2.0, panel_h=1.4, share=share_axes)

    # Marker shade along the path: darkest at the smallest tau (deepest tail), lighter as tau
    # grows; the floor keeps the largest tau clearly visible.
    shades = {tau: 1.0 - 0.55 * rank / max(1, len(taus) - 1) for rank, tau in enumerate(sorted(taus))}

    plotted = False
    for (method, rule), (marker, color, label, lw) in _ARM_STYLES.items():
        arm = sub[(sub["method.name"] == method) & (sub["decision_rule"] == rule)]
        if arm.empty:
            logger.warning("No rows for arm %s/%s; skipping.", method, rule)
            continue
        arm = _arm_tau(arm, method, rule, taus)
        arm = arm[arm["tau"].isin(taus)]
        is_alpha_swept = rule == "cvar_minimax" and method != "base"
        if is_alpha_swept and alphas is not None:
            arm = arm[arm["method.train.alpha"].isin(alphas)]
        alpha_values = sorted(arm["method.train.alpha"].dropna().unique()) if is_alpha_swept else [float("nan")]
        for alpha_value in alpha_values:
            curve_rows = arm[arm["method.train.alpha"] == alpha_value] if pd.notna(alpha_value) else arm
            curve_label = label
            if pd.notna(alpha_value):
                curve_label = f"{label}$_{{{alpha_value:g}}}$"
            if (method, rule) == _OURS:
                curve_label = ours_label(curve_label)
            cells = curve_rows.groupby(["metric", "severity", "tau", "seed"], dropna=False)["value"].mean()
            stats = cells.groupby(["metric", "severity", "tau"]).agg(["mean", "std"]).reset_index()
            for col_idx, severity in enumerate(severities):
                sev = stats[stats["severity"] == severity]
                path = (
                    sev[sev["metric"] == "mean"]
                    .merge(sev[sev["metric"] == "cvar"], on="tau", suffixes=("_mean", "_cvar"))
                    .sort_values("tau")
                )
                if path.empty:
                    continue
                ax = axes[col_idx]
                zorder = 4 if (method, rule) == _OURS else 3
                ax.plot(
                    path["mean_mean"],
                    path["mean_cvar"],
                    color=color,
                    linewidth=lw,
                    solid_capstyle="round",
                    zorder=zorder,
                    path_effects=_halo(lw),
                    label=curve_label,
                )
                ax.errorbar(
                    path["mean_mean"],
                    path["mean_cvar"],
                    xerr=path["std_mean"].fillna(0.0),
                    yerr=path["std_cvar"].fillna(0.0),
                    fmt="none",
                    ecolor=color,
                    elinewidth=0.6,
                    alpha=0.45,
                    zorder=zorder - 1,
                )
                for _, point in path.iterrows():
                    ax.plot(
                        point["mean_mean"],
                        point["mean_cvar"],
                        marker=marker,
                        markersize=4.5,
                        color=_tint(color, shades[float(point["tau"])]),
                        markeredgecolor=color,
                        markeredgewidth=0.5,
                        zorder=zorder + 1,
                    )
                plotted = True

    if not plotted:
        raise ValueError("Nothing was plotted; check the arm filters against the cached rows.")

    loss_label = {"log_loss": "log loss", "brier": "Brier score"}.get(loss, loss)
    # The severity caption lives inside its panel, bottom right: the paths run to the lower left,
    # so that corner is empty, and dropping the titles hands the freed strip back to the axes.
    for idx, severity in enumerate(severities):
        axes[idx].text(
            0.97,
            0.05,
            "clean" if severity == 0 else f"severity {severity}",
            transform=axes[idx].transAxes,
            ha="right",
            va="bottom",
            fontproperties=FP_SEMIBOLD,
            fontsize=0.8 * FS_TITLE,
            color="black",
            zorder=6,
        )
    # One y label for the whole grid, centred on the panels, mirroring the shared supxlabel.
    fig.supylabel(f"CVaR$_τ$ {loss_label}", fontproperties=FP_SEMIBOLD, fontsize=FS_LABEL)
    fig.supxlabel(f"mean {loss_label}", fontproperties=FP_SEMIBOLD, fontsize=FS_LABEL)
    for ax in axes:
        _style_panel(ax)
    if title:
        fig.suptitle(title, fontproperties=FP_REGULAR, fontsize=FS_TITLE + 1)

    handles: list = []
    labels: list[str] = []
    for ax in axes:
        for handle, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            if lab not in labels:
                handles.append(handle)
                labels.append(lab)
    # Tau shade key: the whole swept grid as one strip of neutral-gray swatches, dark (deepest
    # tail) to light, drawn as a single legend entry via HandlerTuple. Two lone dots read as two
    # categories rather than as an ordered sweep, and left the tail direction to be guessed.
    lo, hi = min(taus), max(taus)
    strip = tuple(
        plt.Line2D([0], [0], linestyle="none", marker="s", markersize=4.5, color=_tint("#454545", shades[tau]))
        for tau in sorted(taus)
    )
    tau_key = (strip, f"τ = {lo:g} → {hi:g}   (dark = deeper tail)")
    # Key row, then the baselines in two fixed columns -- the two MLE-based arms on the left, the
    # two tail-trained baselines in the middle -- with ours apart on the right, vertically centred.
    entries = list(zip(handles, labels, strict=True))
    ours = next((e for e in entries if e[1].endswith(OURS_TAG)), None)
    by_label = {lab: entry for entry in entries if (lab := entry[1]) and entry is not ours}
    baseline_rows = [["MLE", "SQwash"], ["MLE + CVaR-minimax", "AdaCVaR"]]
    rows = [[by_label[lab] for lab in row if lab in by_label] for row in baseline_rows]
    # Anything unexpected in the cache still gets a slot rather than vanishing from the legend.
    placed = {lab for row in baseline_rows for lab in row}
    leftover = [entry for entry in entries if entry is not ours and entry[1] not in placed]
    rows += [leftover[i : i + 2] for i in range(0, len(leftover), 2)]
    _legend_below(
        fig,
        rows,
        ncol=2,
        ours=ours,
        key=tau_key,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.35)},
    )
    return fig


def _dummy_df(taus: list[float], loss: str = "log_loss") -> pd.DataFrame:
    """Fabricate cache-schema shift rows for all five arms, for layout work before results land.

    Shapes are chosen to look plausible, not to mean anything: the mean loss grows with
    severity, the own-level CVaR grows as tau shrinks, tail-trained and decision-time arms
    trade a slightly worse mean for a better tail, and the credal arm trades most.

    Args:
        taus: CVaR levels of the fabricated sweep.
        loss: Loss name stamped on the rows.

    Returns:
        Long-form DataFrame with the columns plot_shift_risk_metric uses.
    """
    import pandas as pd  # noqa: PLC0415

    rng = np.random.default_rng(0)
    corruptions = ["gaussian_noise", "motion_blur", "brightness_down", "contrast_down", "jpeg_compression"]
    # (mean factor, cvar factor, hedge cost) per arm: the fabricated trade-off. The hedge cost
    # makes the mean deteriorate as tau shrinks (deeper calibration hedges harder), which is what
    # bends the pareto paths; the MLE reference has none, so its path stays vertical.
    arms = {
        ("base", "mle"): (1.00, 1.00, 0.00),
        ("sqwash", "mle"): (1.03, 0.94, 0.05),
        ("adacvar", "mle"): (1.05, 0.96, 0.07),
        ("base", "cvar_minimax"): (1.02, 0.90, 0.08),
        ("credal_rl_multinomial", "cvar_minimax"): (1.03, 0.80, 0.05),
    }
    rows = []
    for (method, rule), (mean_f, cvar_f, hedge) in arms.items():
        for severity in range(6):
            base_mean = (0.9 + 0.38 * severity) * mean_f
            cell_corruptions = ["clean"] if severity == 0 else corruptions
            for corruption in cell_corruptions:
                corr_offset = float(rng.normal(0.0, 0.06))
                for seed in (1, 2, 3):
                    noise = rng.normal(0.0, 0.02, size=2 * len(taus))
                    for tau_idx, tau in enumerate(taus):
                        # Tail amplification of the fabricated loss distribution at this tau, and
                        # the hedge cost the tau-adaptive arms pay on the mean as tau shrinks.
                        amplif = 1.0 + (1.1 + 0.25 * severity) * float(-np.log10(tau)) * cvar_f
                        depth = float(-np.log10(tau)) - float(-np.log10(max(taus)))
                        mean_value = base_mean * (1.0 + hedge * depth) * (1.0 + corr_offset + noise[2 * tau_idx])
                        cvar_value = base_mean * amplif * (1.0 + corr_offset + noise[2 * tau_idx + 1])
                        shared = {
                            "kind": "shift",
                            "loss": loss,
                            "method.name": method,
                            "decision_rule": rule,
                            "corruption": corruption,
                            "severity": severity,
                            "seed": seed,
                            "num_train": float("nan"),
                            "run_id": f"dummy_{method}_{rule}_{seed}_{tau:g}",
                            "created_at": "2026-07-27T00:00:00Z",
                            "method.train.alpha": (
                                tau
                                if method in ("sqwash", "adacvar")
                                else 0.95
                                if rule == "cvar_minimax" and method != "base"
                                else float("nan")
                            ),
                            "cvar_beta": tau if rule == "cvar_minimax" else float("nan"),
                        }
                        if method == "base" and rule == "mle":
                            # One tau-independent run: flat mean, one cvar row per level.
                            if tau_idx == 0:
                                rows.append({**shared, "aggregation": "mean", "value": mean_value})
                            rows.append({**shared, "aggregation": f"cvar_{tau:g}", "value": cvar_value})
                        else:
                            rows.append({**shared, "aggregation": "mean", "value": mean_value})
                            rows.append({**shared, "aggregation": f"cvar_{tau:g}", "value": cvar_value})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Edit these and re-run. DUMMY renders fabricated data for layout work; False pulls the cache.
    DUMMY = False
    DATASET = "cifar10"
    TAUS = [0.01, 0.025, 0.05, 0.1, 0.2]
    ALPHAS: list[float] | None = [0.95]  # curves per alpha for the credal arm; None = all cached
    SEEDS: list[int] | None = [1, 2, 3]
    # Four severities in two columns is the single-column paper layout: clean and 1 on the top
    # row, 3 and 5 below. None takes every severity present; NCOLS=None puts them in one row.
    SEVERITIES: list[int] | None = [0, 1, 3, 5]
    NCOLS: int | None = 2
    CORRUPTIONS: list[str] | None = None  # e.g. the 5-subset, to pin the protocol
    LOSS = "log_loss"
    STYLE = "pareto"  # "pareto" = mean vs CVaR trade-off paths; "tau_curves" = twin-axis tau sweep
    SHARE_AXES = False  # True puts every severity panel on one common scale
    TITLE: str | None = None  # the paper figure carries its description in the caption
    SAVE = True

    if DUMMY:
        df = _dummy_df(TAUS, loss=LOSS)
        print(f"Fabricated {len(df)} dummy rows.")
    else:
        df = load_runs(filters={"dataset": DATASET})
        print(f"Loaded {len(df)} rows from {df['run_id'].nunique()} runs.")

    plot_fn = plot_shift_risk_pareto if STYLE == "pareto" else plot_shift_risk_metric
    fig = plot_fn(
        df,
        taus=TAUS,
        alphas=ALPHAS,
        seeds=SEEDS,
        severities=SEVERITIES,
        corruptions=CORRUPTIONS,
        loss=LOSS,
        ncols=NCOLS,
        share_axes=SHARE_AXES,
        title=TITLE,
    )
    if SAVE:
        suffix = "dummy" if DUMMY else DATASET
        plot_path = PLOTS_PATH / f"shift_risk_metric_{STYLE}_{LOSS}_{suffix}.pdf"
        fig.savefig(plot_path, bbox_inches="tight")
        print(f"Plot saved to {plot_path}")
    plt.show()
