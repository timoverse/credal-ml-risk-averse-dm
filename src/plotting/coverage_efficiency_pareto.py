"""Coverage-efficiency trade-off on CIFAR-10, as a Pareto figure in the paper's house style.

Reads the CSV that plotting/coverage_efficiency_csv.py exports -- one row per (method, alpha,
seed) -- and draws coverage on x against efficiency on y, one curve per alpha-swept method and one
marker per method that has no alpha.

Both axes are ORIENTED SO THAT MORE IS BETTER, which is worth stating because it inverts the usual
conformal-prediction figure. Coverage is the fraction of test instances whose first-order target
distribution lies inside the predicted credal set. Efficiency is probly's set-size efficiency,
1 - mean(upper - lower) of the envelope, so a tight set scores HIGH -- it is not a set size, and
the y axis is not to be read as one. The ideal point is therefore the TOP RIGHT corner, and the
figure marks it, because a reader arriving from a coverage-vs-set-size plot will otherwise assume
the opposite.

The credal level alpha traces each swept method's own trade-off: alpha = 0 is a vacuous set
(coverage 1, efficiency 0, where all three swept methods necessarily meet) and alpha -> 1 shrinks
the set toward the point prediction. That shared origin is why the curves fan out from one corner.

Styling is imported from experiments/plot_runtime_bars rather than restated, so this figure cannot
drift from the rest of the paper's set: same Fira faces, same crimson-for-ours against blue
baselines, same hairline spines and 0.92 gridlines. Because six baselines cannot be told apart in
one hue, the hue carries the GROUP and the marker (plus a dash pattern on the swept ones)
separates methods within it -- the convention cost_sensitive/cost_sensitive_triage.py's line figure
already uses.

Run: uv run python src/plotting/coverage_efficiency_pareto.py [--dataset cifar10]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.transforms import ScaledTranslation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.plot_runtime_bars import (  # noqa: PLC2701
    _BASELINE_BLUE,
    _BASELINE_GRAY,
    _TICK_NUDGE_PT,
    _TRAIN_COLOR,
    FP_TICK,
    _lighten,
)
from experiments.plot_runtime_bars import FP_REGULAR as _SHARED_REGULAR
from experiments.plot_runtime_bars import FP_SEMIBOLD as _SHARED_SEMIBOLD
from paths import PLOTS_PATH, RESULTS_PATH

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.font_manager import FontProperties


def _with_dejavu_math(face: FontProperties) -> FontProperties:
    """A copy of a shared face whose mathtext is set in DejaVu, as this figure has always drawn it.

    The shared faces are pinned to the Fira math fontset (plotting/paper_style.py). This figure
    predates that: its alpha subscripts and the arrow of the shade key are DejaVu glyphs, and it is
    kept exactly as it was, so its faces keep the fontset it was drawn with.
    """
    legacy = face.copy()
    legacy.set_math_fontfamily("dejavusans")
    return legacy


FP_REGULAR = _with_dejavu_math(_SHARED_REGULAR)
FP_SEMIBOLD = _with_dejavu_math(_SHARED_SEMIBOLD)

# CSV method name -> the paper's short name. Taken from plot_runtime_bars' table so a method is
# called the same thing in the timing figure and here; "CreRL" is credal_relative_likelihood, NOT
# the kernel-evidence method, which is CreWare (see that module's note on the collision).
METHOD_LABELS = {
    "credal_ensembling": "CreEns",
    "credal_bnn": "CreBNN",
    "credal_wrapper": "CreWra",
    "credal_dro": "CreDRO",
    "credal_relative_likelihood": "CreRL",
    "efficient_credal_prediction": "EffCre",
    "credal_rl_multinomial": "CreWare",
}
OURS = "credal_rl_multinomial"
# Draw order: baselines first so ours lands on top of any overlap. Within that, the alpha-swept
# methods before the single-setting ones, which is also the legend's reading order.
METHOD_ORDER = (
    "credal_relative_likelihood",
    "efficient_credal_prediction",
    "credal_ensembling",
    "credal_bnn",
    "credal_wrapper",
    "credal_dro",
    OURS,
)
# One fixed marker per method, so a method keeps its shape if the figure is ever split or reordered.
# The dash pattern is only meaningful on the swept methods (the others are single points); it is
# kept short because a long dash reads as solid on a curve this size.
METHOD_MARKERS = {
    "credal_relative_likelihood": "s",
    "efficient_credal_prediction": "D",
    "credal_ensembling": "P",
    "credal_bnn": "^",
    "credal_wrapper": "*",
    "credal_dro": "X",
    OURS: "o",
}
# Legend order, which is NOT the draw order: the four fixed-set baselines first (two per column at
# ncol = 4), then the two swept baselines, then ours alone in the last column. Grouping the legend
# by KIND of method rather than by draw order means the columns say what the colours say -- gray
# for a method with one operating point, blue for one with a knob, crimson for ours.
LEGEND_ORDER = (
    "credal_ensembling",
    "credal_bnn",
    "credal_wrapper",
    "credal_dro",
    "credal_relative_likelihood",
    "efficient_credal_prediction",
    OURS,
)
METHOD_DASHES: dict[str, Any] = {
    "credal_relative_likelihood": (0, (2.2, 1.1)),
    "efficient_credal_prediction": (0, (0.9, 0.9)),
    OURS: "solid",
}
ALPHA_SWEPT = frozenset(METHOD_DASHES)
# alpha rides on the marker's SHADE, within the method's own hue: pale at alpha = 0, saturated at
# alpha = 1. Hue is already spoken for -- it carries the group (blue baselines, crimson ours)
# everywhere else in the paper -- and size is spoken for too, since a single-point method needs a
# bigger marker to be legible at all. Lightness is the one channel left, and it happens to encode
# the quantity in the right direction: a small alpha is a weak likelihood cut, hence a big vague
# set, and it draws as the faint marker. The saturated end is the tight set.
#
# 0.72, not 1.0: past roughly 0.8 the palest markers stop reading against the 0.92 gridlines, and
# the alpha = 0 point of every curve then looks like a gap rather than a measurement.
ALPHA_TINT_MAX = 0.72
# One multiplier on every line width and marker size, so the figure's weight can be retuned in one
# place instead of by editing seven numbers that then drift apart.
SIZE_SCALE = 1.4
# Per-method override of that scale. "*" is the one marker whose nominal size badly overstates the
# ink it lays down -- a five-pointed star is mostly the gaps between its points -- so CreWra needs
# roughly twice the nominal size of the others to read as their equal on the page.
MARKER_SCALE = {"credal_wrapper": 2.0}


def _tint(color: str, amount: float) -> str:
    """`color` blended `amount` of the way into white, as a hex STRING.

    Hex rather than the RGB triple _lighten hands back: matplotlib's stubs type a Line2D colour as
    a string, so splatting a tuple through a kwargs dict fails the type check even though it is a
    perfectly good colour at runtime. cost_sensitive_triage.py pays the same conversion for the
    same reason, and keeping one type throughout means tinted and flat hues stay interchangeable.
    """
    return to_hex(_lighten(color, amount))


def read_rows(path: Path) -> list[dict[str, Any]]:
    """The raw per-seed CSV as typed rows; a blank alpha means the method has no credal level."""
    with path.open(newline="") as handle:
        return [
            {
                "method": row["method"],
                "alpha": None if row["alpha"] in ("", None) else float(row["alpha"]),
                "seed": float(row["seed"]),
                "coverage": float(row["coverage"]),
                "efficiency": float(row["efficiency"]),
            }
            for row in csv.DictReader(handle)
        ]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, float | None]]]:
    """Seed mean and standard error of both metrics, per (method, alpha), sorted by alpha.

    SEM rather than the std the _agg export carries: every other figure in this paper reports the
    seed mean with +-1 standard error, and mixing the two across a figure set would make spreads
    look three times larger here than in the bar figures for the same number of seeds. It is
    recomputed from the raw per-seed rows with ddof=1 so it matches cost_sensitive_triage's agg() exactly.
    """
    grouped: dict[tuple[str, float | None], dict[str, list[float]]] = defaultdict(
        lambda: {"coverage": [], "efficiency": []}
    )
    for row in rows:
        bucket = grouped[(str(row["method"]), row["alpha"])]
        bucket["coverage"].append(float(row["coverage"]))
        bucket["efficiency"].append(float(row["efficiency"]))

    out: dict[str, list[dict[str, float | None]]] = defaultdict(list)
    for (method, alpha), bucket in grouped.items():
        point: dict[str, float | None] = {"alpha": alpha}
        for metric in ("coverage", "efficiency"):
            values = np.asarray(bucket[metric])
            point[metric] = float(values.mean())
            point[f"{metric}_sem"] = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
        point["n_seeds"] = float(len(bucket["coverage"]))
        out[method].append(point)
    for series in out.values():
        # None sorts first: a method with no alpha has a single point, so the key never collides.
        series.sort(key=lambda p: (p["alpha"] is not None, p["alpha"] or 0.0))
    return out


def dominated_by(series: dict[str, list[dict[str, float | None]]], method: str) -> dict[str, int]:
    """How many of `method`'s points another method strictly dominates (both metrics >=, one >).

    Printed rather than drawn. With both axes oriented "more is better", one curve sitting above
    and right of another is a real ordering and not a plotting artefact, so it is worth stating in
    numbers next to a figure that shows it in ink.
    """
    mine = [p for p in series.get(method, []) if p["coverage"] is not None]
    counts: dict[str, int] = {}
    for other, points in series.items():
        if other == method:
            continue
        counts[METHOD_LABELS.get(other, other)] = sum(
            any(
                q["coverage"] >= p["coverage"]  # ty: ignore[unsupported-operator]
                and q["efficiency"] >= p["efficiency"]  # ty: ignore[unsupported-operator]
                and (q["coverage"] > p["coverage"] or q["efficiency"] > p["efficiency"])  # ty: ignore[unsupported-operator]
                for q in points
            )
            for p in mine
        )
    return counts


class _AlphaRamp(HandlerBase):
    """Legend swatch: a rounded pill running pale to saturated, the marker-shade key.

    Built the way plot_runtime_bars' _SplitPill is -- overlapping patches clipped to a pill outline,
    in the handle box's own point coordinates -- because a legend handle cannot hold an image and a
    second colorbar axes would need layout space this figure does not have.

    Drawn in OURS' crimson rather than a neutral gray. The shade rule applies within every method's
    own hue, so no single hue is literally representative; showing it in the crimson makes the key
    double as a pointer to our curve, which is the same trick the runtime figure's highlight legend
    uses for its intensity key.
    """

    def __init__(self, color: str, tint_max: float, steps: int = 28, thickness: float = 4.2) -> None:
        super().__init__()
        self._color, self._tint_max, self._steps, self._thickness = color, tint_max, steps, thickness

    def create_artists(  # noqa: PLR0913, PLR0917
        self,
        legend,  # noqa: ANN001, ARG002
        orig_handle,  # noqa: ANN001, ARG002
        xdescent,  # noqa: ANN001
        ydescent,  # noqa: ANN001
        width,  # noqa: ANN001
        height,  # noqa: ANN001
        fontsize,  # noqa: ANN001, ARG002
        trans,  # noqa: ANN001
    ) -> list:
        """Draw the ramp, clipped to a rounded pill."""
        x0, th = -xdescent, self._thickness
        y0 = -ydescent + (height - th) / 2
        pill = FancyBboxPatch((x0, y0), width, th, boxstyle=f"round,pad=0,rounding_size={th / 2}")
        clip = pill.get_path()
        step = width / self._steps
        artists: list = []
        for index in range(self._steps):
            # Saturated at the LEFT, pale at the right, i.e. alpha running 1 -> 0. That is the
            # order the data appears in on the axes: alpha near 1 gives a tight set at low
            # coverage (left), alpha near 0 a vacuous one at coverage 1 (right). A ramp running the
            # other way would have the reader mapping the key to the plot back to front.
            fraction = 1.0 - index / (self._steps - 1)
            # +0.6 of overlap: abutting rectangles leave hairline seams once the renderer
            # antialiases them, which reads as banding rather than as a ramp.
            artists.append(
                Rectangle(
                    (x0 + index * step, y0),
                    step + 0.6,
                    th,
                    facecolor=_tint(self._color, self._tint_max * (1.0 - fraction)),
                    edgecolor="none",
                )
            )
        for artist in artists:
            artist.set_transform(trans)
            artist.set_clip_path(clip, trans)
        return artists


def _legend_label(method: str) -> str:
    """The method's name for the legend.

    The alpha-swept methods carry an "alpha" subscript, so the legend says which three have a knob
    without the reader having to notice which entries are curves. Ours carries the paper's tag and
    is set in the semibold face as a whole, name and tag, the way every figure of the paper marks
    our method (plot() picks the face).
    """
    name = METHOD_LABELS[method]
    if method in ALPHA_SWEPT:
        name = rf"{name}$_\alpha$"
    return f"{name} [ours]" if method == OURS else name


def _style(method: str, solid_baselines: bool = False) -> dict[str, Any]:
    """Line kwargs for one method: crimson and heavier for ours, blue for every baseline.

    A method with no alpha is a SINGLE marker, so it gets a larger one with a white edge. At the
    curve methods' size those four shapes are indistinguishable mush, and unlike a curve point they
    have no neighbours to be read as a trend with -- the marker is the entire series.
    """
    is_ours = method == OURS
    is_curve = method in ALPHA_SWEPT
    # Three hues, one per KIND of method: crimson for ours, blue for a baseline with an alpha knob,
    # gray for one with a single fixed operating point. The gray is what makes the four fixed-set
    # baselines findable -- as blue they read as stray points dropped off the two blue curves that
    # pass straight through them, which is exactly where CreEns, CreWra and CreDRO sit.
    if is_ours:
        color = _TRAIN_COLOR
    else:
        color = _BASELINE_BLUE if is_curve else _BASELINE_GRAY
    base_size = (4.0 if is_ours else 3.4) if is_curve else 5.4
    return {
        "color": color,
        "marker": METHOD_MARKERS[method],
        "markersize": base_size * MARKER_SCALE.get(method, SIZE_SCALE),
        "linewidth": (1.5 if is_ours else 1.0) * SIZE_SCALE,
        # solid_baselines drops the dash patterns and leans on marker shape alone. It was tried and
        # NOT adopted: the two swept baselines cross at about (0.5, 0.87), and drawn solid the
        # reader has to identify a square against a diamond at 5pt to know which curve they came in
        # on. The dash pattern settles that without a second look, which is worth more than the
        # redundancy it costs. Kept as a flag so the comparison can be re-made rather than
        # re-argued -- but the paper figure is the dashed one, and it is the default.
        "linestyle": ("solid" if solid_baselines else METHOD_DASHES.get(method, "none"))
        if method in ALPHA_SWEPT
        else "none",
        "markeredgewidth": (0.0 if is_curve else 0.5) * SIZE_SCALE,
        "markeredgecolor": "white",
        "zorder": 5 if is_ours else 3,
    }


def _mark_ideal_corner(ax: Axes) -> None:
    """Say which way is better, since this figure's y axis inverts the usual convention.

    Efficiency is 1 - mean envelope width, so the good direction is UP, not down as it would be
    were the axis a set size. A reader who does not notice reads every curve backwards.

    The arrow sits in the BOTTOM LEFT, not beside the ideal corner it points at: the top right is
    where EffCre's high-coverage points live, and an annotation there landed on the very curve the
    reader is meant to be comparing. The low-coverage, low-efficiency corner is the one region no
    method occupies -- nothing is both loose and uncovered -- so it is the only free space, and the
    arrow still points the way it must.
    """
    ax.annotate(
        "",
        xy=(0.225, 0.225),
        xytext=(0.075, 0.075),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "color": "0.55", "linewidth": 0.7, "shrinkA": 0, "shrinkB": 0},
    )
    # Beyond the arrowhead rather than beside the shaft, so the arrow points AT the word: the
    # annotation then reads as one gesture ("this way -> better") instead of as a label with a
    # decorative rule next to it. Set at the axis labels' size, since it names a direction on the
    # axes and is not a caption about the data.
    ax.text(
        0.222,
        0.232,
        "better",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontproperties=FP_SEMIBOLD,
        fontsize=7,
        color="0.45",
    )


def plot(series: dict[str, list[dict[str, float | None]]], dataset: str, solid_baselines: bool = False) -> None:
    """Draw and save the trade-off figure."""
    plt.rcParams.update(
        {
            "axes.linewidth": 0.3125,
            "axes.edgecolor": "black",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Fira Sans:regular",
            "mathtext.bf": "Fira Sans:semibold",
        }
    )
    fig_w, fig_h = 3.35, 2.55
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes((0.145, 0.145, 0.835, 0.835))

    drawn: dict[str, Any] = {}
    for method in METHOD_ORDER:
        points = series.get(method)
        if not points:
            continue
        style = _style(method, solid_baselines)
        x = np.array([p["coverage"] for p in points], dtype=float)
        y = np.array([p["efficiency"] for p in points], dtype=float)
        xerr = np.array([p["coverage_sem"] for p in points], dtype=float)
        yerr = np.array([p["efficiency_sem"] for p in points], dtype=float)
        # Error bars in BOTH axes: every point is a seed mean of two measured quantities, and a
        # y-only bar would silently claim coverage was measured without error. Drawn under the
        # markers in the whisker gray the bar figures use.
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="none",
            ecolor="0.25",
            elinewidth=0.6 * SIZE_SCALE,
            capsize=1.1 * SIZE_SCALE,
            capthick=0.6 * SIZE_SCALE,
            zorder=style["zorder"] - 1,
        )
        if method in ALPHA_SWEPT:
            # Line and markers are drawn separately: ax.plot paints every marker of a call in one
            # colour, and the whole point here is that they differ. The line becomes a guide at a
            # fixed mid tint, the markers carry alpha in their shade.
            ax.plot(
                x,
                y,
                color=style["color"],
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
                zorder=style["zorder"] - 0.5,
            )
            alphas = np.array([p["alpha"] for p in points], dtype=float)
            ax.scatter(
                x,
                y,
                marker=style["marker"],
                # scatter sizes in points SQUARED, plot in points; squaring keeps a marker the same
                # physical size as the single-point methods' plot markers beside it.
                s=style["markersize"] ** 2,
                c=[_tint(style["color"], ALPHA_TINT_MAX * (1.0 - a)) for a in alphas],
                linewidths=0.4 * SIZE_SCALE,
                edgecolors="white",
                zorder=style["zorder"],
            )
            # The swatch mirrors what is drawn: full-hue line, and a marker at the saturated end
            # of that method's own ramp, which is its alpha = 1 point.
            # Spelled out rather than splatted from `style`: that dict is typed Any, and matplotlib's
            # stubs then try the tuple dash pattern against every numeric keyword in turn.
            handle = plt.Line2D(
                [],
                [],
                color=style["color"],
                marker=style["marker"],
                markersize=style["markersize"],
                markerfacecolor=style["color"],
                markeredgewidth=0,
                linewidth=style["linewidth"],
                linestyle=style["linestyle"],
            )
        else:
            (handle,) = ax.plot(x, y, **style)
        drawn[method] = handle

    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(True, color="0.92", lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("coverage", fontproperties=FP_SEMIBOLD, fontsize=7, labelpad=2.5)
    ax.set_ylabel("efficiency", fontproperties=FP_SEMIBOLD, fontsize=7, labelpad=3)
    ax.tick_params(axis="both", width=0.3125, length=2.5, color="black", pad=1.2)
    # BOTH axes in Book, the weight plot_runtime_bars uses for tick labels. Thin was tried on both
    # and reads washed-out once the PDF is placed in the document -- which is the finding that
    # module's own comment records ("Thin reads washed-out gray"). Thin survives there only on the y
    # NUMBERS, and only because that figure's x ticks are method names carrying the row's weight;
    # a figure whose every tick is a bare number has nothing to trade that recession against.
    nudge = ScaledTranslation(0, _TICK_NUDGE_PT / 72, fig.dpi_scale_trans)
    for label in ax.get_xticklabels():
        label.set_fontproperties(FP_TICK)
        label.set_fontsize(6.5)
        label.set_color("0.15")
    for label in ax.get_yticklabels():
        label.set_fontproperties(FP_TICK)
        label.set_fontsize(6.5)
        label.set_color("0.15")
        label.set_transform(label.get_transform() + nudge)
    _mark_ideal_corner(ax)

    legend_font = FP_REGULAR.copy()
    legend_font.set_size(6.0)
    # The alpha key gets its OWN row above the method legend rather than an eighth swatch inside
    # it: it explains what moves a point ALONG a curve, which is a different kind of fact from what
    # a method swatch says, and folding it in would read as an eighth method. Same separation the
    # shift_risk_metric pareto figure gives its tau key.
    key_handle = plt.Line2D([], [])
    key_legend = fig.legend(
        [key_handle],
        [r"marker shade:  $\alpha$ = 1  $\rightarrow$  0"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.005),
        bbox_transform=fig.transFigure,
        frameon=False,
        prop=legend_font,
        handlelength=2.6,
        handletextpad=0.6,
        borderpad=0.0,
        handler_map={key_handle: _AlphaRamp(_TRAIN_COLOR, ALPHA_TINT_MAX)},
    )
    for text in key_legend.get_texts():
        text.set_color("0.45")
    # Rebuilt in LEGEND_ORDER, not draw order. matplotlib fills legend columns COLUMN-major, so
    # seven entries at ncol = 4 give columns of 2, 2, 2, 1 -- which lands the four gray fixed-set
    # baselines in the first two columns, the two blue swept ones in the third, and ours alone in
    # the fourth, exactly the grouping the colours carry.
    legend = fig.legend(
        [drawn[m] for m in LEGEND_ORDER if m in drawn],
        [_legend_label(m) for m in LEGEND_ORDER if m in drawn],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.075),
        bbox_transform=fig.transFigure,
        ncol=4,
        frameon=False,
        prop=legend_font,
        handlelength=1.9,
        handletextpad=0.5,
        columnspacing=1.1,
        borderpad=0.0,
    )
    ours_font = FP_SEMIBOLD.copy()
    ours_font.set_size(6.0)
    for text, method in zip(legend.get_texts(), [m for m in LEGEND_ORDER if m in drawn], strict=True):
        text.set_color("0.15")
        if method == OURS:
            text.set_fontproperties(ours_font)

    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    stem = f"coverage_efficiency_pareto_{dataset}" + ("_solid" if solid_baselines else "")
    for suffix in ("pdf", "png"):
        fig.savefig(PLOTS_PATH / f"{stem}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {PLOTS_PATH / stem}.pdf and .png")


def main() -> None:
    """Read the per-seed CSV, aggregate, report the dominance counts, draw the figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--csv", type=Path, default=None, help="defaults to results/coverage_efficiency_<dataset>.csv")
    parser.add_argument(
        "--solid-baselines",
        action="store_true",
        help="Draw the swept baselines solid instead of dashed/dotted; writes a _solid variant.",
    )
    args = parser.parse_args()

    path = args.csv or RESULTS_PATH / f"coverage_efficiency_{args.dataset}.csv"
    if not path.exists():
        raise SystemExit(f"{path} does not exist; export it with plotting/coverage_efficiency_csv.py first")
    series = aggregate(read_rows(path))

    print(f"{'method':<28}{'points':>7}{'alphas':>8}")
    for method in METHOD_ORDER:
        points = series.get(method, [])
        alphas = [p["alpha"] for p in points if p["alpha"] is not None]
        print(f"{METHOD_LABELS.get(method, method):<28}{len(points):>7}{len(alphas):>8}")
    counts = dominated_by(series, OURS)
    total = len(series.get(OURS, []))
    print(f"\nOf {METHOD_LABELS[OURS]}'s {total} operating points, strictly dominated by:")
    for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<12} {count}/{total}")
    plot(series, args.dataset, args.solid_baselines)


if __name__ == "__main__":
    main()
