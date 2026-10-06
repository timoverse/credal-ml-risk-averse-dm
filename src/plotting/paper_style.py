"""The one look every figure of the paper shares: fonts, type sizes, strokes, and a color per method.

The reference is the risk-averse shift grid (plotting/shift_risk_grid.py) as it prints in the main
text: Fira Sans, semibold lower-case axis labels, light tick labels, a hairline box around every
panel with no grid, haloed lines with one marker shape per method, and a frameless legend below the
panels with ours set apart in semibold. Every other figure draws from this module instead of
restating those choices, so they cannot drift apart again.

Two things live here and nowhere else:

- METHODS: the color, marker and line style of each method. A method wears the same color in every
  figure it appears in, and no two methods share one, so a color means the same thing on every page.
- Sizes: type sizes and stroke widths. They are the reference figure's numbers times a scale, and
  the scale exists because figures are drawn at different zooms: the reference is drawn 1.59x its
  printed width (REFERENCE_ZOOM), so its 10 pt labels print at 6.3 pt. A figure drawn at another
  zoom has to scale its type to print at the same size; render_for_print() measures the zoom from
  the saved width and the width the paper includes the figure at, and hands back the matching Sizes.

Palette: MLE gray, AdaCVaR amber, CreWra violet and CreWare red are the reference grid's. The others
were chosen with the dataviz skill's validate_palette.js against the methods they share a figure
with (--pairs all, white surface): the seven credal predictors clear the CVD target (worst pair
9.1) and the normal-vision floor (16.0). CreEns is a deliberately chroma-poor steel, the neutral of
that family as gray is for the MLE. Markers back the colors up, so identity never rides on hue
alone.

Blue is BASELINE_BLUE. It is the one color every baseline wears in the figures that contrast the
baselines as a group with our method (the triage bars and line curves, the runtime bars, coverage
against efficiency), and it is the own color of the two risk-averse baselines where each method
has one: SQwash in the shift and triage figures, CeSoR in the driving figure. They never share a
figure.

The MLE under our rule is a lighter gray than the plain MLE, as well as dashed, so the two can be
told apart in a bar. With it and the blue, the shift grid's six colors clear the same checks (CVD
8.8, normal vision 16.4), and so do the driving figure's four.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.text import Annotation

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# The figures are laid out by measuring rendered text, and that measurement depends on the canvas:
# an interactive backend scales the figure dpi by the display's pixel ratio (2 on a Retina screen),
# which moves every element by a fraction of a pixel. Agg renders the same on every machine, so it
# is the default here; set MPLBACKEND to choose another backend, e.g. to get plt.show() windows.
if "MPLBACKEND" not in os.environ:
    matplotlib.use("Agg")

# \linewidth of the AISTATS template in inches: one column of the main text, and the full text
# block (figure* in the main text, every figure of the one-column appendix).
COLUMN_WIDTH = 3.25
TEXT_WIDTH = 6.75
# Saved width of the reference figure over the width it is included at (0.9 of a column).
REFERENCE_ZOOM = 4.661 / (0.9 * COLUMN_WIDTH)
_SAVE_PAD_IN = 0.1  # matplotlib's default pad_inches under bbox_inches="tight"


def fira(weight: str) -> FontProperties:
    """Fira Sans at a named weight from the TeX Live font tree; plain sans-serif when unavailable."""
    hits = sorted(Path("/usr/local/texlive").glob("*/texmf-dist/fonts/opentype/public/fira"))
    if not hits:
        return FontProperties(family="sans-serif")
    path = hits[-1] / f"FiraSans-{weight}.otf"
    fontManager.addfont(str(path))
    # A FontProperties fixes its math fontset when it is CREATED, from the rcParams of that moment,
    # so it is pinned here: left to the default, bold mathtext is drawn in DejaVu whatever
    # use_fira_mathtext sets afterwards.
    return FontProperties(fname=path, math_fontfamily="custom")


FP_SEMIBOLD = fira("SemiBold")  # axis labels, the ours entry
FP_REGULAR = fira("Regular")  # legend, panel headers, annotations
FP_LIGHT = fira("Light")  # tick labels


@dataclass(frozen=True)
class Sizes:
    """Type sizes (pt) and stroke widths (pt) of a figure: the reference's numbers times a scale.

    Attributes:
        scale: 1 for a figure drawn at the reference zoom; see render_for_print.
    """

    scale: float = 1.0

    @property
    def label(self) -> float:
        """Axis labels."""
        return 10.0 * self.scale

    @property
    def tick(self) -> float:
        """Tick labels, and in-panel annotations that read as data."""
        return 8.0 * self.scale

    @property
    def legend(self) -> float:
        """Legend entries."""
        return 9.0 * self.scale

    @property
    def header(self) -> float:
        """Panel and column headers."""
        return 10.0 * self.scale

    @property
    def line(self) -> float:
        """Width of a method's line."""
        return 1.4 * self.scale

    @property
    def line_ours(self) -> float:
        """Width of our method's line."""
        return 1.8 * self.scale

    @property
    def marker(self) -> float:
        """Marker size on a line."""
        return 3.2 * self.scale

    @property
    def marker_edge(self) -> float:
        """Width of a marker's outline, which adds to its size."""
        return 1.0 * self.scale

    @property
    def halo(self) -> float:
        """How much wider than its line the white halo under it is."""
        return 1.0 * self.scale

    @property
    def spine(self) -> float:
        """Width of the panel box and of the ticks."""
        return 0.15 * self.scale

    @property
    def tick_length(self) -> float:
        """Length of a tick."""
        return 2.5 * self.scale

    @property
    def reference_line(self) -> float:
        """Width of a dotted reference line."""
        return 0.9 * self.scale


@dataclass(frozen=True)
class Method:
    """How one method is drawn wherever it appears.

    Attributes:
        label: Name as the paper writes it, without alpha subscript or the ours tag.
        color: Its color in every figure.
        marker: Its marker in every line plot.
        linestyle: Solid unless the method shares its color with a sibling.
    """

    label: str
    color: str
    marker: str
    linestyle: object = "-"


OURS = "credal_rl_multinomial"
# Every baseline, in the figures that set the baselines as a group against ours; and the own color
# of SQwash and of CeSoR.
BASELINE_BLUE = "#4286de"
# Keyed by method.name of the W&B cache where one exists. "base+rule" is the MLE under our decision
# rule: a lighter gray than the plain MLE's, dashed, with its own marker. Ours takes the circle; the
# two MLE arms take the triangles, up for the plain one and down under the rule.
METHODS: dict[str, Method] = {
    "base": Method("MLE", "#454545", "^"),
    "base+rule": Method("MLE + rule", "#999999", "v", (0, (3.0, 1.6))),
    "sqwash": Method("SQwash", BASELINE_BLUE, "s"),
    "adacvar": Method("AdaCVaR", "#eda100", "D"),
    "cesor": Method("CeSoR", BASELINE_BLUE, "d"),
    "credal_ensembling": Method("CreEns", "#849098", "P"),
    "credal_bnn": Method("CreBNN", "#008300", "p"),
    "credal_wrapper": Method("CreWra", "#4a3aa7", "X"),
    "credal_dro": Method("CreDRO", "#9bc53d", "*"),
    "credal_relative_likelihood": Method("CreRL", "#a05195", "h"),
    "efficient_credal_prediction": Method("EffCre", "#3cc2ef", ">"),
    OURS: Method("CreWare", "#ea365b", "o"),
}
REFERENCE_GRAY = "0.55"  # reference lines that are not a method (e.g. the uniform prediction)
REFERENCE_DOTS = (0, (1, 1.6))
BAND_ALPHA = 0.14  # the plus/minus band under a line
# The panel box and its ticks. Their width (Sizes.spine) is already under a screen pixel at any
# sensible zoom, and a PDF viewer never draws a line thinner than one pixel, so how heavy the box
# looks on screen is set by its darkness, not its width: a mid gray reads as half the black line.
FRAME_COLOR = "0.5"
OURS_TAG = "[ours]"
# Our decision rule applied to somebody else's predictor: the rule is ours, the predictor is not,
# so only the rule is semibold (mathtext bold, see use_fira_mathtext).
OUR_RULE = r"$\mathbf{+}$ $\mathbf{rule}$"


def ours_label(label: str) -> str:
    """Our method's legend or tick label: the name followed by the ours tag."""
    return f"{label} {OURS_TAG}"


def use_fira_mathtext() -> None:
    """Route mathtext through Fira Sans, upright, so subscripts and Greek match the text around them."""
    if FP_REGULAR.get_file() is None:
        return
    plt.rcParams.update(
        {
            "mathtext.fontset": "custom",
            "mathtext.rm": "Fira Sans",
            "mathtext.it": "Fira Sans",
            "mathtext.bf": "Fira Sans:semibold",
            "mathtext.default": "regular",
        }
    )


def font(face: FontProperties, size: float) -> FontProperties:
    """A copy of a face at a size, for the prop= arguments that take no separate fontsize."""
    sized = face.copy()
    sized.set_size(size)
    return sized


def halo(linewidth: float, sizes: Sizes | None = None) -> list:
    """The white halo path effect for a line of the given width."""
    return [pe.withStroke(linewidth=linewidth + (sizes or Sizes()).halo, foreground="white")]


def tint(color: str, t: float) -> tuple[float, float, float]:
    """Blend a color towards white: t=1 is the full color, smaller t a lighter tint of the same hue."""
    r, g, b = to_rgb(color)
    return (1.0 - t + t * r, 1.0 - t + t * g, 1.0 - t + t * b)


def line_style(method: str, sizes: Sizes) -> dict:
    """Keyword arguments for ax.plot that draw a method's line the way every figure does.

    Args:
        method: Key into METHODS.
        sizes: The figure's sizes.

    Returns:
        color, linestyle, linewidth, marker, markersize, cap styles, halo and zorder (ours on top).
    """
    style = METHODS[method]
    ours = method == OURS
    linewidth = sizes.line_ours if ours else sizes.line
    return {
        "color": style.color,
        "linestyle": style.linestyle,
        "linewidth": linewidth,
        "marker": style.marker,
        "markersize": sizes.marker,
        "markeredgewidth": sizes.marker_edge,
        "dash_capstyle": "butt",
        "solid_capstyle": "round",
        "path_effects": halo(linewidth, sizes),
        "zorder": 4 if ours else 3,
    }


def band_style(color: object, alpha: float = BAND_ALPHA) -> dict:
    """Keyword arguments for ax.fill_between that draw the plus/minus band under a line."""
    return {"color": color, "alpha": alpha, "linewidth": 0, "zorder": 2}


def style_axes(ax: Axes, sizes: Sizes) -> None:
    """Panel chrome: a gray hairline box and outward ticks, light black tick labels, no grid.

    Call once the ticks are final; tick labels created later would miss the font.

    Args:
        ax: Axes to style.
        sizes: The figure's sizes.
    """
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(sizes.spine)
        spine.set_color(FRAME_COLOR)
    ax.grid(visible=False)
    ax.tick_params(which="both", width=sizes.spine, length=sizes.tick_length, color=FRAME_COLOR)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontproperties(FP_LIGHT)
        label.set_fontsize(sizes.tick)
        label.set_color("black")


def set_xlabel(ax: Axes, text: str, sizes: Sizes) -> None:
    """Semibold x label at the figure's label size. Labels are lower case, apart from names."""
    ax.set_xlabel(text, fontproperties=FP_SEMIBOLD, fontsize=sizes.label)


def set_ylabel(ax: Axes, text: str, sizes: Sizes) -> None:
    """Semibold y label at the figure's label size. Labels are lower case, apart from names."""
    ax.set_ylabel(text, fontproperties=FP_SEMIBOLD, fontsize=sizes.label)


def set_header(ax: Axes, text: str, sizes: Sizes) -> None:
    """A panel or column header: what a panel shows when the axes cannot say it, never a figure title."""
    ax.set_title(text, fontproperties=FP_REGULAR, fontsize=sizes.header)


def align_ylabels(fig: Figure, axes: Iterable[Axes]) -> None:
    """Put the y labels of stacked panels on one vertical line, whatever the width of their ticks."""
    fig.align_ylabels(list(axes))


def legend_below(  # noqa: PLR0913
    fig: Figure,
    rows: list[list[tuple]],
    ncol: int,
    sizes: Sizes | None = None,
    ours: tuple | None = None,
    key: tuple | None = None,
    handler_map: dict | None = None,
    handlelength: float = 2.4,
    top: float = 0.018,
    ours_column: int | None = None,
    center: float = 0.5,
) -> None:
    """Draw a frameless legend below the figure: a grid of rows, with ours set apart from it.

    Matplotlib fills a legend column-major, so a desired row layout has to be transposed first;
    this takes the rows as written and does that, padding short rows with invisible entries.
    The ours entry is a SECOND legend rather than another grid cell, because matplotlib would
    place a leftover cell in the first row instead of centring it against both; it is measured
    off the drawn grid and centred vertically on it. A re-layout after this call (e.g.
    tight_layout) would leave that placement stale.

    Args:
        fig: Figure to attach the legend to.
        rows: One list of (handle, label) pairs per legend row, in reading order.
        ncol: Entries per row; rows shorter than this are padded.
        sizes: The figure's sizes. None uses the reference's.
        ours: Optional (handle, label) drawn apart from the grid: to its right, vertically centred,
            or on a line of its own below it (ours_column).
        key: Optional (handle, label) drawn as its own centred legend above the grid -- a scale
            key that reads as an aside rather than as one more method entry, and whose label
            length cannot distort the grid's column widths.
        handler_map: Extra legend handlers, e.g. {tuple: HandlerTuple(...)} for a swatch strip.
        handlelength: Handle width in font units; raise it to fit a multi-marker handle.
        top: Figure-fraction y of the legend's top edge. The default sits just inside the figure:
            constrained_layout leaves a slack band under a supxlabel, and anchoring below y=0
            stacks the legend on top of that for a large dead gap.
        ours_column: Put the ours entry on a line of its own below the grid, under this column,
            handle under handle. It is laid out as one more row of the same legend, so the row
            spacing is the grid's own in every backend; only its tag hangs out of the column to the
            right, so the column stays as narrow as the entries above it. The label before the tag
            must have no descenders, since the tag is set on its bounding box. None hangs the entry
            to the right of the grid instead.
        center: Figure-fraction x the legend is centred on. The default is the figure's centre; a
            figure whose panels sit off-centre (a wide y label on the left) passes their centre.
    """
    sizes = sizes or Sizes()
    blank = (plt.Line2D([], [], linestyle="none"), "")
    legend_font = font(FP_REGULAR, sizes.legend)
    ours_font = font(FP_SEMIBOLD, sizes.legend)
    beside = ours is not None and ours_column is None
    if ours is not None and ours_column is not None:
        row = [blank] * ncol
        row[ours_column] = (ours[0], ours[1].removesuffix(OURS_TAG).rstrip())
        rows = [*rows, row]
    padded = [row + [blank] * (ncol - len(row)) for row in rows]
    # Transpose: matplotlib walks down each column, so column c is every row's c-th entry.
    ordered = [padded[r][c] for c in range(ncol) for r in range(len(padded))]
    if key is not None:
        key_legend = fig.legend(
            [key[0]],
            [key[1]],
            loc="upper center",
            bbox_to_anchor=(center, top),
            bbox_transform=fig.transFigure,
            frameon=False,
            prop=legend_font,
            handlelength=2.6,  # a key handle is a strip of swatches, not a single line sample
            handletextpad=0.5,
            borderpad=0.0,
            handler_map=handler_map,
        )
        # Stack the grid under it; the key's own height is unknown until it has been laid out.
        fig.canvas.draw()
        top = key_legend.get_window_extent().transformed(fig.transFigure.inverted()).y0
    grid = fig.legend(
        [h for h, _ in ordered],
        [lab for _, lab in ordered],
        loc="upper left" if beside else "upper center",
        bbox_to_anchor=(0.0, top) if beside else (center, top),
        bbox_transform=fig.transFigure,
        ncol=ncol,
        frameon=False,
        prop=legend_font,
        columnspacing=1.2,
        handlelength=handlelength,
        handletextpad=0.5,
        labelspacing=0.4,
        borderpad=0.0,
        handler_map=handler_map,
    )
    for text in grid.get_texts():
        if text.get_text().endswith(OURS_TAG):
            text.set_fontproperties(ours_font)
    if ours is None:
        return
    if ours_column is not None:
        # The last row's entry in that column, column-major. Its tag is drawn from the label's right
        # edge at draw time, so it follows the legend wherever a backend lays it out. Bottom on
        # bottom puts the two on one baseline (same face, and neither reaches below a "p"). Drawn
        # after the legend: an artist drawn before it would read the label's place from the
        # previous draw, which savefig's tight bounding box has moved by then.
        label = grid.get_texts()[(ours_column + 1) * len(padded) - 1]
        label.set_fontproperties(ours_font)
        fig.add_artist(
            Annotation(
                f" {OURS_TAG}",
                xy=(1, 0),
                xycoords=label,
                xytext=(0, 0),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontproperties=ours_font,
                annotation_clip=False,
                zorder=grid.get_zorder() + 1,
            )
        )
        return
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    box = grid.get_window_extent().transformed(inv)
    # Hang the ours entry off the grid's right edge at its vertical midpoint.
    gap = 0.02
    leg = fig.legend(
        [ours[0]],
        [ours[1]],
        loc="center left",
        bbox_to_anchor=(box.x1 + gap, (box.y0 + box.y1) / 2),
        bbox_transform=fig.transFigure,
        frameon=False,
        prop=legend_font,
        handlelength=handlelength,
        handletextpad=0.5,
    )
    for text in leg.get_texts():
        text.set_fontproperties(ours_font)
    # Centre grid and ours as one block. Left as drawn, the pair runs off the right edge, and
    # savefig(bbox_inches="tight") then widens the canvas on that side only.
    fig.canvas.draw()
    ours_box = leg.get_window_extent().transformed(inv)
    shift = center - (ours_box.x1 - box.x0) / 2 - box.x0
    grid.set_bbox_to_anchor((box.x0 + shift, box.y1), transform=fig.transFigure)
    leg.set_bbox_to_anchor((box.x1 + gap + shift, (box.y0 + box.y1) / 2), transform=fig.transFigure)
    if key is not None:
        # Centre the key on the grid+ours block rather than on the figure: the block's own centre
        # is what the eye reads as the legend's axis, and it is only known once both are placed.
        key_box = key_legend.get_window_extent().transformed(inv)
        key_legend.set_bbox_to_anchor(
            ((box.x0 + shift + ours_box.x1 + shift) / 2, key_box.y1), transform=fig.transFigure
        )


def saved_width(fig: Figure) -> float:
    """Width in inches of the figure as savefig(bbox_inches="tight") writes it."""
    fig.canvas.draw()
    return fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * _SAVE_PAD_IN  # ty: ignore[unresolved-attribute]


def render_for_print(build: Callable[[Sizes], Figure], print_width: float, passes: int = 2) -> tuple[Figure, Sizes]:
    """Build a figure whose type prints at the reference's size when included at print_width.

    The scale depends on the saved width, and the saved width on the type (labels and the legend
    push the tight bounding box out), so the figure is built, measured and rebuilt. Two passes
    settle it to well under a percent.

    Args:
        build: Draws the figure with the given sizes and returns it.
        print_width: Width in inches the paper includes the figure at, e.g. 0.9 * COLUMN_WIDTH.
        passes: Measure-and-rebuild rounds.

    Returns:
        The figure and the sizes it was drawn with.
    """
    sizes = Sizes()
    for _ in range(passes):
        fig = build(sizes)
        sizes = Sizes(saved_width(fig) / print_width / REFERENCE_ZOOM)
        plt.close(fig)
    return build(sizes), sizes


def save(fig: Figure, stem: str, directory: Path, print_width: float | None = None, sizes: Sizes | None = None) -> None:
    """Write the figure as pdf and png, and log the size its axis labels print at.

    Args:
        fig: Figure to save.
        stem: File name without extension.
        directory: Where to write.
        print_width: Width in inches the paper includes the figure at; with sizes, the printed
            label size is logged so a figure that drifted from the reference's 6.3 pt shows up.
        sizes: The sizes the figure was drawn with.
    """
    for ext in ("pdf", "png"):
        fig.savefig(directory / f"{stem}.{ext}", bbox_inches="tight", dpi=200)
    note = ""
    if print_width is not None and sizes is not None:
        width = saved_width(fig)
        note = f" ({width:.2f} in wide, labels print at {sizes.label * print_width / width:.2f} pt)"
    logger.info("Saved %s.pdf and .png%s", directory / stem, note)
