"""Landmark-compression ablation figure: what shrinking CreWare's reference costs and buys.

Three panels against one x axis, the landmark count M that experiments/landmark_ablation.py
sweeps. M is the only moving part of that experiment -- encoder, whitener, bandwidth and alpha all
stay fixed from the trained artifact -- so every curve here is the same method at a different
reference size, and the leftmost point of each is the untouched 45000-point reference.

(a) OOD AUROC, split near (cifar100, tin) against far (mnist, svhn, textures, places365). This is
    what compression COSTS.
(b) inference time, the seconds of one credal test pass. This is what it BUYS, and it is the reason
    the ablation exists: the pass is a fixed encoder forward plus an evidence term that scales with
    the reference, so only the second term is recoverable and the curve must flatten, not fall to
    zero.
(c) seconds of the k-means fit itself -- a ONE-OFF cost paid at fit time, not per inference, so it
    is on its own panel rather than added to (b). Absent for the full reference, which is why that
    curve starts one point later than the others.

x is logarithmic: M is a genuine count spanning 45x, and its sweep values are already roughly
log-spaced, so a linear axis would pile four of the six points into one fifth of the panel.

The look is the paper's shared figure style (plotting/paper_style.py). There are no baselines in
this figure -- every series is ours -- so both lines wear CreWare's red: near-OOD the full color,
solid, with the method's circle; far-OOD a light tint of it, dashed, with a square.

Run: uv run python src/plotting/landmark_ablation.py
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import PLOTS_PATH, RESULTS_PATH
from plotting.paper_style import (
    FP_REGULAR,
    METHODS,
    OURS,
    TEXT_WIDTH,
    Sizes,
    band_style,
    legend_below,
    line_style,
    render_for_print,
    save,
    set_xlabel,
    set_ylabel,
    style_axes,
    tint,
    use_fira_mathtext,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

# One csv per artifact seed, as experiments/landmark_ablation.py writes them into results/.
RESULTS_GLOB = "landmark_ablation_*_seed*.csv"
_STEM = "landmark_ablation_cifar10"
# The six OpenOOD cifar10 sets in the paper's column order, near-OOD first, then far-OOD (see
# plotting/ood_table._PAPER_COLUMN_ORDER and experiments/landmark_ablation.OOD_DATASETS).
OOD_NAMES = ("cifar100", "tin", "mnist", "svhn", "textures", "places365")
NEAR_OOD = ("cifar100", "tin")

# The paper includes the figure at the full text width (one-column appendix).
_PRINT_WIDTH = 1.0 * TEXT_WIDTH
# Hand-placed panels, in inches: three equal boxes with room between them for the next panel's
# y numbers and label, and the x labels in the strip below.
_FIG_SIZE = (7.0, 1.72)
_PANEL_SIZE = (1.70, 1.13)
_PANEL_LEFTS = (0.50, 2.86, 5.22)
_PANEL_BOTTOM = 0.36
# Far-OOD is the same method on another family of sets, so it takes the same hue at a lower
# intensity: the tint the runtime figure gives its test bars.
_FAR_TINT = 0.45
_FAR_DASHES = (0, (2.2, 1.1))
_FULL_BAND_COLOR = "#e8e8e8"
_FULL_TAG_COLOR = "0.45"
# Tick and label pads at matplotlib's defaults, in points at the reference scale. paper_style
# leaves them unscaled, which is invisible near the reference zoom but not at this figure's; scaled
# by sizes.scale they print as they do in the reference figure.
_TICK_PAD = 3.5
_LABEL_PAD = 4.0


def read_dir(directory: Path) -> list[dict[str, Any]]:
    """Every sweep row across the per-artifact-seed csvs in `directory`.

    The baseline row leaves kmeans_seed and kmeans_s empty (nothing was compressed), so both are
    read as None rather than coerced -- a 0.0 there would land in panel (c) as a real measurement
    of a fit that never happened.

    Args:
        directory: Where the csvs of experiments/landmark_ablation.py live.

    Returns:
        One dict per csv row, with the artifact seed taken from the file name.
    """
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob(RESULTS_GLOB)):
        seed = int(path.stem.split("seed")[-1])
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "artifact_seed": seed,
                        "m_landmarks": int(row["m_landmarks"]),
                        "kmeans_seed": None if row["kmeans_seed"] in ("", None) else int(row["kmeans_seed"]),
                        "kmeans_s": None if row["kmeans_s"] in ("", None) else float(row["kmeans_s"]),
                        "test_s": float(row["test_s"]),
                        **{f"auroc_{n}": float(row[f"auroc_{n}"]) for n in OOD_NAMES},
                    }
                )
    return rows


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean and standard error over every seed at each landmark count, ascending in M.

    Both seed axes are pooled: an artifact seed and a k-means seed are both just repetitions of the
    same cell, and the figure's band is meant to say "how much does this number move if you run it
    again", which is over both. The baseline has only artifact seeds, so its band is over three
    values and the compressed cells' over nine -- stated here because it is invisible on the
    figure, where they look alike.

    Args:
        rows: Output of read_dir.

    Returns:
        One dict per landmark count: m_landmarks, n_cells, and <metric> / <metric>_sem for the
        test time, the k-means time, the per-set AUROCs and their near and far means (None where a
        metric is undefined, i.e. the k-means time of the full reference).
    """
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["m_landmarks"]].append(row)

    metrics = ["test_s", "kmeans_s", "auroc_near", "auroc_far", *(f"auroc_{n}" for n in OOD_NAMES)]
    out: list[dict[str, Any]] = []
    for m_landmarks, cells in sorted(grouped.items()):
        for cell in cells:
            near = [cell[f"auroc_{n}"] for n in OOD_NAMES if n in NEAR_OOD]
            far = [cell[f"auroc_{n}"] for n in OOD_NAMES if n not in NEAR_OOD]
            cell["auroc_near"], cell["auroc_far"] = float(np.mean(near)), float(np.mean(far))
        point: dict[str, Any] = {"m_landmarks": m_landmarks, "n_cells": len(cells)}
        for metric in metrics:
            values = [c[metric] for c in cells if c[metric] is not None]
            if not values:
                point[metric], point[f"{metric}_sem"] = None, None
                continue
            array = np.asarray(values, dtype=float)
            point[metric] = float(array.mean())
            point[f"{metric}_sem"] = float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else 0.0
        out.append(point)
    return out


def _series(points: list[dict[str, Any]], metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x, mean and sem arrays for one metric, dropping the landmark counts where it is undefined."""
    kept = [p for p in points if p[metric] is not None]
    return (
        np.array([p["m_landmarks"] for p in kept], dtype=float),
        np.array([p[metric] for p in kept], dtype=float),
        np.array([p[f"{metric}_sem"] for p in kept], dtype=float),
    )


def _line_styles(sizes: Sizes) -> tuple[dict[str, Any], dict[str, Any]]:
    """ax.plot keyword arguments of the near-OOD and of the far-OOD line, both our method's.

    Args:
        sizes: The figure's sizes.

    Returns:
        The near style (paper_style.line_style of ours) and the far style derived from it: a
        light tint of the same red, dashed, with a square, drawn below the near line.
    """
    near = line_style(OURS, sizes)
    far = near | {
        "color": tint(METHODS[OURS].color, _FAR_TINT),
        "linestyle": _FAR_DASHES,
        "marker": "s",
        "zorder": near["zorder"] - 0.5,
    }
    return near, far


def _line(ax: Axes, points: list[dict[str, Any]], metric: str, style: dict[str, Any]) -> Line2D:
    """One metric as a line with a plus/minus one SEM band; returns the handle for the legend."""
    x, y, sem = _series(points, metric)
    ax.fill_between(x, y - sem, y + sem, **band_style(style["color"]))
    (line,) = ax.plot(x, y, **style)
    return line


def _mark_full_reference(ax: Axes, m_full: float, sizes: Sizes) -> None:
    """Shade the uncompressed reference, the point every other point is a compression of.

    It is the first point of each curve rather than a separate series, so without a mark it reads
    as just another sweep value -- when it is in fact the only one that is not an approximation,
    and the number every AUROC loss on the panel is measured against.
    """
    ax.axvspan(m_full * 0.80, m_full * 1.22, color=_FULL_BAND_COLOR, zorder=0, linewidth=0)
    ax.text(
        m_full,
        0.965,
        "full",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontproperties=FP_REGULAR,
        fontsize=sizes.tick,
        color=_FULL_TAG_COLOR,
        zorder=6,
    )


def _finish_panel(ax: Axes, ylabel: str, x_ticks: np.ndarray, sizes: Sizes) -> None:
    """Axes, labels and chrome of one panel, once its lines are drawn.

    Args:
        ax: The panel.
        ylabel: Its y label.
        x_ticks: The swept landmark counts, which are the x ticks.
        sizes: The figure's sizes.
    """
    ax.set_xscale("log")
    ax.set_xticks(x_ticks)
    # Plain counts, not matplotlib's 10^n: the sweep values are 1000 ... 45000, none of them a
    # power of ten, so scientific tick labels would name numbers that are not in the experiment.
    ax.set_xticklabels([f"{int(v / 1000)}k" if v >= 1000 else f"{int(v)}" for v in x_ticks])
    ax.xaxis.set_minor_locator(plt.NullLocator())  # a decade of minor ticks in a 1.7in panel is noise
    # DESCENDING: the full reference on the left, harder compression to the right. A numeric axis
    # normally grows rightward, and this one deliberately does not, because M is not the subject --
    # the intervention is, and the intervention is compression. Read left to right the figure is
    # then one sentence: start from the exact method, throw reference away, watch AUROC give and
    # inference time fall. Ascending, the same curves have to be read backwards to say it.
    ax.set_xlim(max(x_ticks) * 1.35, min(x_ticks) / 1.35)
    low, high = ax.get_ylim()
    ax.set_ylim(low, low + (high - low) * 1.14)  # headroom for the "full" tag
    _mark_full_reference(ax, float(max(x_ticks)), sizes)
    set_xlabel(ax, "reference instances", sizes)
    set_ylabel(ax, ylabel, sizes)
    ax.xaxis.labelpad = ax.yaxis.labelpad = _LABEL_PAD * sizes.scale
    ax.tick_params(axis="both", pad=_TICK_PAD * sizes.scale)
    style_axes(ax, sizes)


def plot_landmark_ablation(points: list[dict[str, Any]], sizes: Sizes | None = None) -> Figure:
    """Build the three-panel ablation figure: OOD AUROC, inference time and k-means fit against M.

    Args:
        points: Output of aggregate, one entry per landmark count.
        sizes: Type sizes and stroke widths. None uses the reference's.

    Returns:
        The assembled matplotlib Figure.
    """
    sizes = sizes or Sizes()
    use_fira_mathtext()
    fig_w, fig_h = _FIG_SIZE
    fig = plt.figure(figsize=_FIG_SIZE)
    box = (_PANEL_BOTTOM / fig_h, _PANEL_SIZE[0] / fig_w, _PANEL_SIZE[1] / fig_h)
    axes = [fig.add_axes((left / fig_w, *box)) for left in _PANEL_LEFTS]
    x_ticks = np.array([p["m_landmarks"] for p in points], dtype=float)
    near_style, far_style = _line_styles(sizes)

    near = _line(axes[0], points, "auroc_near", near_style)
    far = _line(axes[0], points, "auroc_far", far_style)
    _line(axes[1], points, "test_s", near_style)
    _line(axes[2], points, "kmeans_s", near_style)
    for ax, ylabel in zip(axes, ("OOD AUROC", "inference time (s)", "k-means fit (s)"), strict=True):
        _finish_panel(ax, ylabel, x_ticks, sizes)

    entries = [(near, "near-OOD (CIFAR-100, TIN)"), (far, "far-OOD (MNIST, SVHN, Textures, Places365)")]
    # The x labels live inside the figure's bottom strip, so the legend hangs from its lower edge.
    legend_below(fig, [entries], ncol=len(entries), sizes=sizes, top=0.0)
    return fig


def _print_tradeoff(points: list[dict[str, Any]]) -> None:
    """Print the numbers behind the figure, full reference first, with speed-up and near-OOD loss."""
    m_full = max(p["m_landmarks"] for p in points)
    baseline = next(p for p in points if p["m_landmarks"] == m_full)
    print(f"\n{'M':>7}{'cells':>7}{'near':>9}{'far':>9}{'test_s':>9}{'kmeans_s':>10}{'speedup':>9}{'d near':>9}")
    for point in sorted(points, key=lambda p: -p["m_landmarks"]):
        kmeans = "--" if point["kmeans_s"] is None else f"{point['kmeans_s']:.1f}"
        print(
            f"{point['m_landmarks']:>7}{point['n_cells']:>7}{point['auroc_near']:>9.4f}{point['auroc_far']:>9.4f}"
            f"{point['test_s']:>9.3f}{kmeans:>10}"
            f"{baseline['test_s'] / point['test_s']:>8.2f}x{point['auroc_near'] - baseline['auroc_near']:>9.4f}"
        )


def main() -> None:
    """Read the sweep csvs, aggregate over seeds, report the trade-off, draw the figure."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_PATH, help="Directory of the sweep csvs.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rows = read_dir(args.results_dir)
    if not rows:
        raise SystemExit(f"No {RESULTS_GLOB} under {args.results_dir}. Run experiments/landmark_ablation.py first.")
    points = aggregate(rows)
    _print_tradeoff(points)

    fig, sizes = render_for_print(lambda s: plot_landmark_ablation(points, s), _PRINT_WIDTH)
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    save(fig, _STEM, PLOTS_PATH, _PRINT_WIDTH, sizes)
    plt.close(fig)


if __name__ == "__main__":
    main()
