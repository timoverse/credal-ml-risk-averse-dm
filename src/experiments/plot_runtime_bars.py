"""Paper figure for the timing table: train and test seconds per credal method as paired bars.

Reads the csv `src/experiments/timing.py` writes (one row per method, `train_s_seed<s>` and
`test_s_seed<s>` columns) and draws one red train bar and one blue test bar per method, with the
std over runs as a whisker. `--csv` defaults to `results/timing`, the directory of real per-seed
measurements; `--placeholder` draws invented numbers for styling work and says so on stdout.

Train and test differ by orders of magnitude (minutes of training vs. seconds of inference), so
the y-axis is logarithmic by default; `--linear` switches it off.

The paper's figure is the `--broken-y` layout (plots/runtime_bars_broken.*): train over test, the
baselines in one panel and ours in its own. It is the one layout here drawn in the paper's shared
figure style (plotting/paper_style.py), with its type sized for the width the paper includes it at.
The other layouts are earlier variants and keep their own look.

Run:  uv run python src/experiments/plot_runtime_bars.py --broken-y
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib.transforms import ScaledTranslation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# FP_SEMIBOLD and FP_REGULAR are the paper's faces, re-exported under the names this module always
# had: cost_sensitive/cost_sensitive_triage.py imports them from here.
from plotting.paper_style import (
    BASELINE_BLUE,
    FP_LIGHT,
    FP_REGULAR,
    FP_SEMIBOLD,
    METHODS,
    OURS,
    TEXT_WIDTH,
    Sizes,
    ours_label,
    render_for_print,
    save,
    use_fira_mathtext,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# The paper's two colors for "ours against the baselines as a group" (plotting/paper_style.py):
# CreWare's red and the blue every baseline wears.
_TRAIN_COLOR = METHODS[OURS].color
_TEST_COLOR = BASELINE_BLUE
# --light-test variant: test bars are the TRAIN crimson blended 55% into white, so the pair reads
# as one hue at two intensities instead of two hues
_TEST_COLOR_LIGHT = "#f5a4b5"

# --method-colors variant: one hue per method, taken from plotting/shift_pareto.py's _METHOD_STYLES
# so a method reads the same here as in the pareto/set-size figures. Three of the seven timed
# methods have no entry there (credal_ensembling, credal_dro, credal_relative_likelihood) and get
# hues of their own.
_METHOD_COLORS: dict[str, str] = {
    "CreEns": "navy",  # credal_ensembling
    "CreBNN": "olive",  # credal_bnn
    "CreWra": "orange",  # credal_wrapper
    "CreDRO": "teal",  # credal_dro
    "CreRL": "purple",  # credal_relative_likelihood
    "EffCre": "crimson",  # efficient_credal_prediction
    "CreWare": "brown",  # credal_rl_multinomial
}
_TEST_TINT = 0.55  # how far a test bar is blended into white from its method's hue
# corner rounding, in INCHES so every bar rounds identically regardless of which panel it is in.
# A bar is ~0.23in wide, so this is roughly a quarter of its width -- past ~0.10 they turn into pills.
_BAR_RADIUS_IN = 0.06

# --highlight-ours variant: only our row keeps the crimson, every baseline recedes into one gray.
# Light enough that ours clearly pops, but not so light that the 55%-white test tint stops reading
# as a bar -- below roughly #a5a5a5 the test half turns into a barely-there ghost.
_BASELINE_GRAY = "#9b9b9b"
# the baselines now carry the blue of the train/test pair instead of gray: a hue reads as a
# deliberate choice where gray reads as "unstyled", and blue vs crimson is the same opposition the
# driving figure already uses. Ours keeps _TRAIN_COLOR and gets its own panel.
_BASELINE_BLUE = _TEST_COLOR
_OURS = "CreWare"
# in the split layout ours sits alone under its own panel, so it can carry weight the shared row
# could not: the whole label is semibold (mathtext.bf), which is what sets it apart from the six.
_OURS_LABEL = r"$\mathbf{CreWare}_{\,\mathbf{0.\!95}}$ $\mathbf{[ours]}$"
# The paper layout's version of that label: the paper-wide ours tag, set whole in FP_SEMIBOLD. The
# subscript needs no \mathbf of its own -- under use_fira_mathtext math is drawn in the face of the
# text around it.
_OURS_TICK = ours_label(r"CreWare$_{\,0.\!95}$")

# The paper layout (_plot_broken) is included at this width, and its type is sized to print like
# the reference figure's there (paper_style.render_for_print).
_PRINT_WIDTH_BROKEN = 0.9 * TEXT_WIDTH
# Strokes and pads paper_style.Sizes has no name for, in points at the reference scale; the paper
# layout multiplies each by sizes.scale, like every other stroke.
_GRID_WIDTH = 0.5  # the light y grid
_WHISKER_WIDTH = 0.6  # the +-1 std whisker
_X_TICK_PAD = 1.2  # x tick to method name
_Y_TICK_PAD = 3.5  # y tick to y number (matplotlib's default)
_LABEL_PAD = 3.0  # y numbers to y label

# table row order. The alpha rides as a subscript on the method name and "[ours]" sits inline after
# it, so every label is a single line and the figure keeps no space for a second one. The last slot
# is the widest label; it can afford to run past the axes on the right, but watch the LEFT edge --
# it is centred, so growing this string eats into EffCre. Mathtext gives a bare "." TeX's punctuation
# spacing,
# which opened a visible gap on both sides. Only the RIGHT one is negated (0.\!95): the period's ink
# sits low and left, so it needs the natural space before it to look centred -- kerning both sides
# glues it onto the leading zero. Do NOT reach for {.} (it makes the gap WIDER) and do not double
# the \! (at 7pt the point vanishes into the digits and it reads "095"). The leading \, is a thin
# space holding the subscript off the method name, which otherwise butts straight into the zero.
_ROWS: tuple[tuple[str, str], ...] = (
    ("CreEns", r"CreEns$_{\,0.\!0}$"),
    ("CreBNN", "CreBNN"),
    ("CreWra", "CreWra"),
    ("CreDRO", "CreDRO"),
    ("CreRL", r"CreRL$_{\,0.\!95}$"),
    ("EffCre", r"EffCre$_{\,0.\!95}$"),
    ("CreWare", r"CreWare$_{\,0.\!95}$ [$\mathbf{ours}$]"),
)

# stand-ins until timing.py has run: (train mean, train std, test mean, test std) in seconds
_PLACEHOLDER: dict[str, tuple[float, float, float, float]] = {
    "CreEns": (8620.0, 210.0, 41.8, 1.9),
    "CreBNN": (3050.0, 145.0, 88.4, 6.2),
    "CreWra": (8620.0, 210.0, 6.4, 0.4),
    "CreDRO": (640.0, 31.0, 3.1, 0.2),
    "CreRL": (1450.0, 96.0, 27.3, 2.4),
    "EffCre": (212.0, 9.0, 1.2, 0.1),
    "CreWare": (378.0, 17.0, 2.3, 0.2),
}


def _fira(weight: str) -> FontProperties:
    """Fira Sans at a named weight, straight from the TeX Live font tree (registered as fallback)."""
    hits = sorted(Path("/usr/local/texlive").glob("*/texmf-dist/fonts/opentype/public/fira"))
    if not hits:
        return FontProperties(family="sans-serif")
    path = hits[-1] / f"FiraSans-{weight}.otf"
    fontManager.addfont(str(path))
    return FontProperties(fname=path)


# The two faces below are NOT the paper's: its tick labels are set in paper_style.FP_LIGHT, which is
# what the paper layout (_plot_broken) uses. Book and Thin stay for the older layouts in this file
# and because cost_sensitive/cost_sensitive_triage.py imports both names.
FP_TICK = _fira("Book")  # tick labels: lighter than Regular, but Thin reads washed-out gray
# the y numbers are the one place Thin works: they are a reference scale, not content, and letting
# them recede puts the weight on the method names. The method names themselves stay at Book.
FP_THIN = _fira("Thin")
# y numbers nudged up off their tick. Measured against the rendered png, matplotlib's placement is
# already centred to within ~0.1pt, so this is an OPTICAL correction, not a fix for a layout bug:
# digits read as low when their ink is centred, because the eye weights the flat baseline against
# the open counters above it. 0.65pt overshot visibly; past ~0.5 they start to look high.
_TICK_NUDGE_PT = 0.35


def _lighten(color: str, amount: float) -> tuple[float, float, float]:
    """`color` blended `amount` of the way into white -- the test bar's tint of its method hue."""
    r, g, b = (c + (1.0 - c) * amount for c in to_rgb(color))
    return (r, g, b)


class _SplitPill(HandlerBase):
    """Legend swatch: one rounded pill cut on a 45-degree diagonal, `left` hue into `right` hue.

    The split layout colours the baselines blue and ours crimson, so a single-hue swatch would name
    only half the figure. Two separate swatches would say there are two *quantities*; one pill in two
    colours says there is one quantity (train, or test) shown in both groups.

    Everything is built in the handle box's own coordinates, which are points -- so "45 degrees" is
    an honest 45 degrees on the page, and the cut is drawn by clipping two overlapping half-plane
    polygons to the pill outline rather than by trying to describe the two halves as paths.
    """

    def __init__(self, left: str | tuple, right: str | tuple, thickness: float = 4.2) -> None:
        super().__init__()
        self._left, self._right, self._thickness = left, right, thickness

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
        x0, th = -xdescent, self._thickness
        y0 = -ydescent + (height - th) / 2  # centred on the handle box, whatever height it gets
        pill = FancyBboxPatch((x0, y0), width, th, boxstyle=f"round,pad=0,rounding_size={th / 2}")
        clip = pill.get_path()  # the outline both halves are clipped to; never drawn itself

        # the cut: x = xc + (y - yc), extended well past the pill so the polygons cover its corners
        xc, yc = x0 + width / 2, y0 + th / 2
        lo, hi = y0 - th, y0 + 2 * th
        x_lo, x_hi = xc + (lo - yc), xc + (hi - yc)
        halves = (
            ([(x0 - th, lo), (x_lo, lo), (x_hi, hi), (x0 - th, hi)], self._left),
            ([(x_lo, lo), (x0 + width + th, lo), (x0 + width + th, hi), (x_hi, hi)], self._right),
        )
        artists: list = [Polygon(v, closed=True, facecolor=c, edgecolor="none") for v, c in halves]
        # a hairline of page-white along the cut, so the two hues meet at a crisp edge instead of
        # bleeding into one another at this size
        artists.append(Line2D([x_lo, x_hi], [lo, hi], color="white", lw=0.7, solid_capstyle="butt"))
        for a in artists:
            a.set_transform(trans)
            a.set_clip_path(clip, trans)
        return artists


def _read_csv(path: Path) -> dict[str, tuple[float, float, float, float]]:
    """(train mean, train std, test mean, test std) per method, in seconds.

    `path` is either one timing.py csv or a DIRECTORY of them. The cluster runs split the sweep by
    method and by seed (results/timing/<group>_seed<n>.csv, one seed column each), so a directory is
    read as one logical table: every file contributes its seeds to whichever methods it names, and a
    method may appear in several files. Std is over seeds, population-style, matching the single-file
    path -- with three seeds it is an indication of spread, not a real error bar.
    """
    files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    train: dict[str, list[float]] = {}
    test: dict[str, list[float]] = {}
    for file in files:
        with file.open(newline="") as f:
            for row in csv.DictReader(f):
                method = row["method"]
                for prefix, bucket in (("train_s_seed", train), ("test_s_seed", test)):
                    vals = [float(v) for k, v in row.items() if k.startswith(prefix) and v not in (None, "")]
                    bucket.setdefault(method, []).extend(vals)
    return {
        m: (float(np.mean(train[m])), float(np.std(train[m])), float(np.mean(test[m])), float(np.std(test[m])))
        for m in train
    }


def _axes_aspect(ax) -> float:  # noqa: ANN001
    """Physical width/height ratio of an axes, so axes-coordinate roundings can be made circular."""
    pos = ax.get_position()
    fig = ax.figure
    return (pos.width * fig.get_figwidth()) / (pos.height * fig.get_figheight())


def _rounded_bar(
    ax,  # noqa: ANN001
    x0: float,
    x1: float,
    top: float,
    color: str | tuple[float, float, float],
    radius_in: float,
) -> None:
    """A bar from the axis floor to `top` with rounded corners of `radius_in` INCHES.

    Drawn in axes coordinates (not data), so the corner radius stays a fixed physical size whatever
    the y-scale does -- on a log axis a data-space rounding would be squashed at the bottom. The
    radius is given in inches rather than axes units because the split layout has two panels of very
    different widths: the same axes-unit radius would round the wide baseline bars four times as hard
    as ours, and the two panels have to look like the same chart.
    """
    to_axes = ax.transData + ax.transAxes.inverted()
    (ax0, _), (ax1, ay1) = to_axes.transform([(x0, top), (x1, top)])
    aspect = _axes_aspect(ax)
    radius = radius_in / (ax.get_position().width * ax.figure.get_figwidth())
    # start below the floor and clip, so only the TOP corners stay rounded. The underhang has to
    # clear the corner arc, which is `radius * aspect` tall in axes-y once mutation_aspect has
    # stretched it -- a fixed 0.04 was enough for a hairline rounding but rounds the bottoms visibly
    # as soon as the radius grows.
    drop = radius * aspect + 0.02
    ax.add_patch(
        FancyBboxPatch(
            (ax0, -drop),
            ax1 - ax0,
            ay1 + drop,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            transform=ax.transAxes,
            mutation_aspect=aspect,  # same physical corner radius in x and y
            linewidth=0,
            facecolor=color,
            zorder=2,
        )
    )


def _build_broken(
    stats: dict[str, tuple[float, float, float, float]],
    table: tuple[tuple[str, str], ...],
    log: bool,
    sizes: Sizes,
) -> Figure:
    """Draw the broken layout (see _plot_broken) with the given type sizes and stroke widths.

    The panel geometry is fixed in inches and does not depend on `sizes`; the type, the hairlines
    and the pads do, so that they print like the reference figure's at the width the paper includes
    this one at. The figure is drawn much narrower than it is included, hence a scale well below 1.
    """
    use_fira_mathtext()
    scale = sizes.scale
    # horizontal geometry is the highlight layout's except for the pitch, which that layout had to
    # size for a PAIR of bars per method and this one does not.
    n_base = len(table) - 1
    unit_in = 0.7007
    # one bar per slot, a touch narrower than the footprint the train/test pair used to fill (0.19):
    # a solid block that wide reads as heavy in a panel this short.
    half = 0.16
    # slot pitch = the bar plus the air beside it. The air is 0.75x the 0.447 it started at; written
    # as bar+gap rather than as a number so the next squeeze is a gap edit, not a re-derivation.
    # (The method names stopped binding when the type was sized for print: they are far narrower
    # than a slot now.)
    pitch = 2 * half + 0.447 * 0.75
    left_in, right_in, gap_in = 0.369, 0.070, 0.161
    base_span, ours_span = (n_base - 1) * pitch + 0.83, 0.95
    fig_w = left_in + (base_span + ours_span) * unit_in + gap_in + right_in
    # vertical, also in inches. The train panel stays the taller of the two -- its bars carry the
    # result, the test row only has to show that inference is cheap for everyone -- but only at
    # 1.5x, not the 2.4x it started at: the extra height was resolution nothing needed, and this is
    # the cheapest vertical space in the figure to give back to the page.
    bottom_in, h_test, gap_v, top_in = 0.26, 0.34, 0.115, 0.05
    h_train = 1.5 * h_test
    fig_h = bottom_in + h_test + gap_v + h_train + top_in
    fig = plt.figure(figsize=(fig_w, fig_h))

    cols = (
        (left_in / fig_w, base_span * unit_in / fig_w, tuple(range(n_base)), pitch),
        ((left_in + base_span * unit_in + gap_in) / fig_w, ours_span * unit_in / fig_w, (len(table) - 1,), 1.0),
    )
    rows_geom = (((bottom_in + h_test + gap_v) / fig_h, h_train / fig_h), (bottom_in / fig_h, h_test / fig_h))
    grid = [[fig.add_axes((x0, y0, w, h)) for x0, w, _, _ in cols] for y0, h in rows_geom]

    trains = np.array([stats[k][0] for k, _ in table])
    train_sd = np.array([stats[k][1] for k, _ in table])
    tests = np.array([stats[k][2] for k, _ in table])
    test_sd = np.array([stats[k][3] for k, _ in table])
    train_colors = [_TRAIN_COLOR if k == _OURS else _BASELINE_BLUE for k, _ in table]
    test_colors = [_lighten(c, _TEST_TINT) for c in train_colors]

    # (values, sd, colours, y-limits, label) per row -- top row is train, bottom is test.
    # Log per panel is the default now that CreBNN is in: it trains ~26x and infers ~51x above the
    # cheapest method, so on a linear scale every other bar collapses to a sliver. Each panel snaps
    # to whole decades around its own range -- two decades each, so both rows read the same way, and
    # unlike the shared axis this figure replaced there is no empty band inside either one.
    def _lims(v: np.ndarray, sd: np.ndarray) -> tuple[float, float]:
        if not log:
            return 0.0, float((v + sd).max() * 1.16)
        # a decade BELOW the smallest value, not at it: a log bar is measured from the floor, so a
        # floor snapped to the minimum leaves the cheapest method -- ours -- as a two-pixel stub.
        # One decade of headroom under it costs a gridline and makes every bar a readable height.
        return 10 ** (np.floor(np.log10(v.min())) - 1), 10 ** np.ceil(np.log10((v + sd).max()))

    quantities = (
        (trains, train_sd, train_colors, _lims(trains, train_sd), "train (s)"),
        (tests, test_sd, test_colors, _lims(tests, test_sd), "test (s)"),
    )

    for row, (*_, lims, ylabel) in zip(grid, quantities, strict=True):
        for a, (_, _, rows, p_) in zip(row, cols, strict=True):
            if log:
                a.set_yscale("log")
                a.yaxis.set_minor_locator(plt.NullLocator())  # 8 dashes a decade in a 0.5in panel
            a.set_ylim(*lims)
            a.grid(True, axis="y", color="0.92", lw=_GRID_WIDTH * scale)
            a.set_axisbelow(True)
            # open axes: the bars stand on the bottom spine and the left one carries the scale
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                a.spines[side].set_linewidth(sizes.spine)
                a.spines[side].set_color("black")
            a.set_xlim(-0.45, (n_base - 1) * pitch + 0.38) if p_ != 1.0 else a.set_xlim(-0.475, 0.475)
            a.set_xticks(np.arange(len(rows), dtype=float) * p_)
        row[1].spines["left"].set_visible(False)  # the scale is read off the left panel
        row[1].tick_params(axis="y", which="both", length=0, labelleft=False)
        row[0].set_ylabel(ylabel, fontproperties=FP_SEMIBOLD, fontsize=sizes.label, labelpad=_LABEL_PAD * scale)
        if log:
            # explicit decades: matplotlib's LogLocator thins a panel this short to every OTHER
            # decade, which left the train row with two labels and an unlabelled gridline between
            lo_e, hi_e = int(np.floor(np.log10(lims[0]))), int(np.round(np.log10(lims[1])))
            row[0].set_yticks([10.0**e for e in range(lo_e, hi_e + 1)])
        else:
            row[0].yaxis.set_major_locator(plt.MaxNLocator(3))

        # 14306 is four digits of noise on a tick; the reader needs the magnitude, not the seconds.
        # The floor tick goes unlabelled: bars are measured FROM it, so its value says nothing, and
        # dropping it stops the train row's floor from colliding with the test row's ceiling.
        def _fmt(v: float, _pos: int, floor: float = lims[0]) -> str:
            if log and np.isclose(v, floor):
                return ""
            if v == 0:
                return "0"
            return f"{v / 1000:g}k" if v >= 1000 else f"{v:g}"

        row[0].yaxis.set_major_formatter(plt.FuncFormatter(_fmt))
        row[1].set_yticks(row[0].get_yticks())  # same gridlines in both panels of a row
        row[1].set_ylim(row[0].get_ylim())

    fig.canvas.draw()  # bars are placed in axes coords, so the layout has to be final first
    whisker_width = _WHISKER_WIDTH * scale
    for row, (vals, sds, colors, _, _) in zip(grid, quantities, strict=True):
        for a, (_, _, rows, p_) in zip(row, cols, strict=True):
            for slot, i in enumerate(rows):
                x, mu, sd = slot * p_, vals[i], sds[i]
                _rounded_bar(a, x - half, x + half, mu, colors[i], radius_in=_BAR_RADIUS_IN)
                a.plot([x, x], [mu - sd, mu + sd], color="0.25", lw=whisker_width, zorder=3)  # +-1 std whisker
                a.plot([x - half * 0.4, x + half * 0.4], [mu + sd] * 2, color="0.25", lw=whisker_width, zorder=3)
                a.plot([x - half * 0.4, x + half * 0.4], [mu - sd] * 2, color="0.25", lw=whisker_width, zorder=3)

    for a in grid[0]:  # the method names belong to the bottom row only; the top row shares them
        a.tick_params(axis="x", length=0, labelbottom=False)
    for a, (_, _, rows, _) in zip(grid[1], cols, strict=True):
        a.set_xticklabels([_OURS_TICK if table[i][0] == _OURS else table[i][1] for i in rows])
        a.tick_params(axis="x", width=sizes.spine, length=sizes.tick_length, color="black", pad=_X_TICK_PAD * scale)
    for a in (*grid[0], *grid[1]):
        a.tick_params(axis="y", width=sizes.spine, length=sizes.tick_length, color="black", pad=_Y_TICK_PAD * scale)
        for label in (*a.get_xticklabels(), *a.get_yticklabels()):
            label.set_fontproperties(FP_LIGHT)
            label.set_fontsize(sizes.tick)
            label.set_color("black")
            label.set_multialignment("center")
    grid[1][1].tick_params(axis="y", which="both", length=0)
    grid[0][1].tick_params(axis="y", which="both", length=0)
    # ours: the whole label semibold, as wherever the paper names our method. Black like every other
    # label: the bars above it already carry the red.
    ours_tick = grid[1][1].get_xticklabels()[0]
    ours_tick.set_fontproperties(FP_SEMIBOLD)
    ours_tick.set_fontsize(sizes.tick)

    nudge = ScaledTranslation(0, _TICK_NUDGE_PT * scale / 72, fig.dpi_scale_trans)
    for a in (grid[0][0], grid[1][0]):  # only the left column carries y numbers
        for label in a.get_yticklabels():
            label.set_transform(label.get_transform() + nudge)

    # matplotlib places a y-label just outside its own tick labels, so "train (s)" is pushed further
    # left than "test (s)" by the width difference between "100k" and "100" -- two labels that should
    # read as one column, visibly staggered. Pin both to the leftmost of the two. (Done by hand:
    # Figure.align_ylabels, and so paper_style.align_ylabels, skips axes placed with add_axes.)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # ty:ignore[unresolved-attribute]
    labels = [row[0].yaxis.label for row in grid]
    target = min(lbl.get_window_extent(renderer).x0 for lbl in labels)
    for a, lbl in zip((row[0] for row in grid), labels, strict=True):
        anchor_x = lbl.get_transform().transform(lbl.get_position())[0]
        shift = target - lbl.get_window_extent(renderer).x0  # keep each label's own anchor offset
        x_fig = fig.transFigure.inverted().transform((anchor_x + shift, 0))[0]
        pos = a.get_position()
        a.yaxis.set_label_coords(x_fig, pos.y0 + pos.height / 2, transform=fig.transFigure)
    return fig


def _plot_broken(
    stats: dict[str, tuple[float, float, float, float]],
    path: Path,
    table: tuple[tuple[str, str], ...] = _ROWS,
    log: bool = True,
) -> None:
    """The highlight figure with the y-axis BROKEN: train on top, test below, one scale each.

    Train and test sit ~3.5 empty decades apart, so on one shared axis the test bars are stubs and
    the log scale spends most of its height on nothing. Two stacked panels give each quantity its
    own scale -- which is also the honest form of a twin y-axis, since the panel break makes
    "different scales" a visible fact rather than a colour-to-axis mapping the reader must infer.

    Two consequences fall out of the split. Each method now has ONE bar per panel, drawn in exactly
    the footprint the train/test pair used to occupy, so the figure keeps its width. And the legend
    disappears: the row is the quantity, so the y-labels say what the swatches used to.

    This is the layout the paper uses, so it wears the paper's shared figure style
    (plotting/paper_style.py): its type is sized to print like the reference figure's at
    _PRINT_WIDTH_BROKEN, and the printed label size is logged on saving. Written as pdf and png
    under the stem of `path`.
    """
    # Three passes, one more than the default: the figure is small and its tight bounding box ends
    # on the y labels, so the saved width moves with the type more than a wide grid's does, and two
    # passes leave the labels at 6.32 pt against the reference's 6.28.
    fig, sizes = render_for_print(lambda s: _build_broken(stats, table, log, s), _PRINT_WIDTH_BROKEN, passes=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    save(fig, path.stem, path.parent, _PRINT_WIDTH_BROKEN, sizes)
    plt.close(fig)


def _plot(  # noqa: PLR0913
    stats: dict[str, tuple[float, float, float, float]],
    path: Path,
    log: bool,
    test_color: str,
    scheme: str = "duo",
    table: tuple[tuple[str, str], ...] = _ROWS,
) -> None:
    """Paired train/test bars per method: wide and short, styled like the driving figure.

    Three colour schemes. "duo" is the original one-red-one-blue split. The other two are
    monochrome -- the hue belongs to the METHOD and the test bar is a tint of it, so each pair
    reads as one thing at two intensities: "method" gives every method its shift_pareto hue,
    "highlight" gives only ours the crimson and greys out the baselines.
    """
    plt.rcParams.update(
        {
            "axes.linewidth": 0.3125,  # 1.25x the driving figure's hairline spines
            "axes.edgecolor": "black",
            # the "[ours]" tick label goes through mathtext, so the math font has to match the ticks
            "mathtext.fontset": "custom",
            "mathtext.rm": "Fira Sans:regular",
            "mathtext.bf": "Fira Sans:semibold",
        }
    )
    # 0.81x the original 6.6in width (two rounds of 0.9x). Height is 0.10in shorter than the two-line
    # era: the tick labels went from two lines to one, so the bottom margin gives that line back to
    # the page instead of to whitespace. Labels do NOT shrink with the figure, so each width cut eats
    # straight into the gaps between them -- "CreWare 0.95 [ours]" is the binding constraint.
    # "highlight" breaks the x-axis: the six baselines share one panel and ours stands alone in its
    # own, so the comparison reads as "the field" vs "us" instead of seven peers in a row. Its
    # geometry is driven in INCHES off a fixed inches-per-data-unit, and the figure width FOLLOWS
    # from it -- that is what lets the slot pitch tighten without the bars changing width.
    split = scheme == "highlight"
    n_base = len(table) - 1 if split else len(table)
    left, bottom, right, height = 0.069, 0.258, 0.987, 0.70
    fig_h = 1.65
    if split:
        unit_in = 0.7007  # inches per x data unit -- the bar width anchor; do not tune casually
        left_in, right_in, gap_in = 0.369, 0.070, 0.161  # margins and the between-panel break
        # pitch is the slot-to-slot distance. At 1.0 the panel is as wide as it was before ours was
        # cut out of it; 0.767 squeezes it to 0.8x by taking the difference out of the whitespace
        # BETWEEN pairs only. It cannot go much lower: at 0.76 the gap between two methods equals
        # the gap inside a train/test pair, and the pairing stops reading.
        pitch = 0.767
        base_span, ours_span = (n_base - 1) * pitch + 0.83, 0.95
        fig_w = left_in + (base_span + ours_span) * unit_in + gap_in + right_in
        fig = plt.figure(figsize=(fig_w, fig_h))
        base_w, ours_w = base_span * unit_in / fig_w, ours_span * unit_in / fig_w
        ax = fig.add_axes((left_in / fig_w, bottom, base_w, height))
        ax_ours = fig.add_axes(((left_in + base_span * unit_in + gap_in) / fig_w, bottom, ours_w, height))
        panels = ((ax, range(n_base), pitch), (ax_ours, [len(table) - 1], 1.0))
    else:
        pitch = 1.0
        fig = plt.figure(figsize=(5.35, fig_h))
        ax = fig.add_axes((left, bottom, right - left, height))  # same axes box in INCHES as at height 1.75
        ax_ours, panels = None, ((ax, range(len(table)), pitch),)

    gap = 0.05  # sliver between a method's two bars, taken out of the bars, not out of the slot
    # bar half-width, at 0.9x the original 0.164. Only the BARS get thinner: the slot pitch and the
    # panel spans are unchanged, so the figure keeps its width and the difference goes to whitespace.
    half = (0.189 - gap / 2) * 0.9
    offset = half + gap / 2  # bar centres, pushed apart by the gap
    trains = np.array([stats[k][0] for k, _ in table])
    train_sd = np.array([stats[k][1] for k, _ in table])
    tests = np.array([stats[k][2] for k, _ in table])
    test_sd = np.array([stats[k][3] for k, _ in table])

    for a, *_ in panels:
        if log:
            a.set_yscale("log")
            a.set_ylim(10 ** np.floor(np.log10(min(tests.min(), 1.0))), 10 ** np.ceil(np.log10(trains.max() * 1.6)))
        else:
            a.set_ylim(0, (trains + train_sd).max() * 1.18)
        a.grid(True, axis="y", color="0.92", lw=0.5)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    # asymmetric on purpose: the left pad was as wide as a bar and read as dead space against the
    # spine, while the right pad still has to absorb "CreWare 0.95 [ours]" overhanging its slot.
    # A bar's outer edge sits 0.353 from its slot centre, so -0.45 leaves it a ~0.1 shoulder.
    ax.set_xlim(-0.45, (n_base - 1) * pitch + 0.38)
    if ax_ours is not None:
        ax_ours.set_xlim(-0.475, 0.475)  # one slot, trimmed to the bar pair plus a hair
        # no background wash: the panel break, the crimson bars and the bold label already say
        # "ours", and a tint underneath them only muddied the colour
        ax_ours.spines["left"].set_visible(False)  # the y-scale is read off the left panel
        # both, not just major: a log axis puts minor ticks on the spineless left edge too, and they
        # read as stray dashes floating in the panel
        ax_ours.tick_params(axis="y", which="both", length=0, labelleft=False)

    if scheme == "method":
        train_colors = [_METHOD_COLORS[k] for k, _ in table]
        test_colors = [_lighten(c, _TEST_TINT) for c in train_colors]
    elif scheme == "highlight":
        train_colors = [_TRAIN_COLOR if k == _OURS else _BASELINE_BLUE for k, _ in table]
        test_colors = [_lighten(c, _TEST_TINT) for c in train_colors]
    else:
        train_colors = [_TRAIN_COLOR] * len(table)
        test_colors = [test_color] * len(table)

    fig.canvas.draw()  # the bars are placed in axes coords, so the layout has to be final first
    for a, rows, p_ in panels:
        for slot, i in enumerate(rows):
            for x, mu, sd, color in (
                (slot * p_ - offset, trains[i], train_sd[i], train_colors[i]),
                (slot * p_ + offset, tests[i], test_sd[i], test_colors[i]),
            ):
                _rounded_bar(a, x - half, x + half, mu, color, radius_in=_BAR_RADIUS_IN)
                a.plot([x, x], [mu - sd, mu + sd], color="0.25", lw=0.6, zorder=3)  # +-1 std whisker
                a.plot([x - half * 0.4, x + half * 0.4], [mu + sd] * 2, color="0.25", lw=0.6, zorder=3)
                a.plot([x - half * 0.4, x + half * 0.4], [mu - sd] * 2, color="0.25", lw=0.6, zorder=3)

    ax.set_ylabel("runtime (s)", fontproperties=FP_SEMIBOLD, fontsize=7, labelpad=3)
    for a, rows, p_ in panels:
        a.set_xticks(np.arange(len(list(rows)), dtype=float) * p_)
        a.set_xticklabels([_OURS_LABEL if split and table[i][0] == _OURS else table[i][1] for i in rows])
        # x only: a blanket tick_params would hand the ours panel its y ticks back, undoing the
        # length=0 above and leaving stub dashes hanging in the gap between the panels
        a.tick_params(axis="x", width=0.3125, length=2.5, color="black", pad=1.2)
        for label in (*a.get_xticklabels(), *a.get_yticklabels()):
            label.set_fontproperties(FP_TICK)
            label.set_fontsize(6.5)  # matches the legend; buys back the gap the width cuts ate
            label.set_color("0.15")
            label.set_multialignment("center")  # the alpha line centres under the method name
    ax.tick_params(axis="y", width=0.3125, length=2.5, color="black")
    if ax_ours is not None:
        ax_ours.get_xticklabels()[0].set_color(_TRAIN_COLOR)

    legend_font = FP_REGULAR.copy()
    legend_font.set_size(6.5)
    # in the monochrome schemes the hue belongs to the METHOD, so the legend can only key on
    # intensity: solid = train, tinted = test. "method" shows that in neutral gray (no one hue is
    # representative); "highlight" shows it in ours' crimson, which doubles as a pointer to our row.
    handler_map = {}
    if scheme == "method":
        swatches: tuple = ("0.35", _lighten("0.35", _TEST_TINT))
        handles = [plt.Line2D([0], [0], color=c, lw=4, solid_capstyle="round") for c in swatches]
    elif scheme == "highlight":
        # one pill per entry, diagonally cut blue-into-crimson: it names the intensity (solid =
        # train, tinted = test) for both groups at once instead of picking a side. The handles are
        # bare proxies -- _SplitPill draws everything -- so the map is keyed on the proxy INSTANCE.
        handles = [plt.Line2D([], []) for _ in range(2)]
        handler_map = {
            handles[0]: _SplitPill(_BASELINE_BLUE, _TRAIN_COLOR),
            handles[1]: _SplitPill(_lighten(_BASELINE_BLUE, _TEST_TINT), _lighten(_TRAIN_COLOR, _TEST_TINT)),
        }
    else:
        swatches = (_TRAIN_COLOR, test_color)
        handles = [plt.Line2D([0], [0], color=c, lw=4, solid_capstyle="round") for c in swatches]
    # anchored in FIGURE coords in the split layout: "upper center" of the left panel is off-centre
    # for the figure as a whole once ours has its own panel on the right.
    anchor = {"bbox_to_anchor": (0.53, 1.0), "bbox_transform": fig.transFigure} if split else {}
    ax.legend(
        handles,
        ["train", "test"],
        loc="upper center",
        ncol=2,
        frameon=False,
        prop=legend_font,
        # 2.2 in the split layout: the pill is twice as long, which gives the 45-degree cut room to
        # read as a diagonal rather than as a notch. The other schemes keep the plain line swatch.
        handlelength=2.2 if split else 1.1,
        handletextpad=0.8,
        columnspacing=1.2,
        borderpad=0.0,
        handler_map=handler_map,
        **anchor,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {path} and {path.with_suffix('.png')}")


def main() -> None:
    """Draw the timing bars from a timing.py csv, or from placeholders if there is none yet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/timing"),
        help="timing.py csv or a directory of them; the real measurements are the default.",
    )
    parser.add_argument(
        "--placeholder",
        action="store_true",
        help="Draw invented numbers instead of the csv. Styling work only -- never for the paper.",
    )
    parser.add_argument("--out", type=Path, default=Path("plots/runtime_bars.pdf"))
    parser.add_argument("--linear", action="store_true", help="Linear y-axis instead of log.")
    parser.add_argument(
        "--light-test", action="store_true", help="Pale red test bars; writes runtime_bars_light.* by default."
    )
    parser.add_argument(
        "--method-colors",
        action="store_true",
        help="One shift_pareto hue per method, test as its tint; writes runtime_bars_methods.* by default.",
    )
    parser.add_argument(
        "--highlight-ours",
        action="store_true",
        help="Crimson for ours, gray baselines, test as a tint; writes runtime_bars_highlight.* by default.",
    )
    parser.add_argument(
        "--broken-y",
        action="store_true",
        help="The paper's layout: train and test on stacked panels, one scale each; writes runtime_bars_broken.*.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")  # paper_style.save logs the printed type size

    table = _ROWS
    if args.placeholder:
        print("--placeholder: drawing invented numbers, NOT the measurements")
        stats = _PLACEHOLDER
    else:
        # the csv is the default and a missing one is fatal: a figure that silently falls back to
        # invented numbers is how invented numbers end up in a paper.
        if not args.csv.exists():
            raise SystemExit(f"{args.csv} does not exist; pass --csv or --placeholder")
        stats = _read_csv(args.csv)
        # a method with no timings is DROPPED, not fatal: the sweep is split across cluster jobs of
        # very different lengths (CreBNN is ~5h a seed), so the figure has to be drawable while some
        # are still running. Anything dropped is named on stdout -- never silently.
        missing = [k for k, _ in table if k not in stats]
        if missing:
            print(f"no timings for {', '.join(missing)} -- dropping from the figure")
            table = tuple((k, lab) for k, lab in table if k in stats)
        if not any(k == _OURS for k, _ in table):
            raise SystemExit(f"{args.csv} has no timings for {_OURS}, which the figure is built around")
    if args.broken_y:
        scheme, stem = "broken", "runtime_bars_broken.pdf"
    elif args.method_colors:
        scheme, stem = "method", "runtime_bars_methods.pdf"
    elif args.highlight_ours:
        scheme, stem = "highlight", "runtime_bars_highlight.pdf"
    elif args.light_test:
        scheme, stem = "duo", "runtime_bars_light.pdf"
    else:
        scheme, stem = "duo", "runtime_bars.pdf"
    out = args.out
    if out == Path("plots/runtime_bars.pdf"):  # each variant gets its own default name
        out = out.with_name(stem)
    if scheme == "broken":
        _plot_broken(stats, out, table=table, log=not args.linear)
        return
    _plot(
        stats,
        out,
        log=not args.linear,
        test_color=_TEST_COLOR_LIGHT if args.light_test else _TEST_COLOR,
        scheme=scheme,
        table=table,
    )


if __name__ == "__main__":
    main()
