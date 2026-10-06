"""Setup figure for slides: the mountain-road map with NO agent routes.

Run: uv run python src/experiments/plot_setup_map.py  ->  plots/credal_driving_map.{pdf,png}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # put src/ on the path

from driving.gridworld import default_road  # noqa: E402
from experiments.credal_driving_curve import _draw_car  # noqa: E402  # reuse the car glyph


def main() -> None:
    """Draw the bare map (road / mountain / hazard, start car, goal star) and save it."""
    road = default_road(p_hazard=0.1)
    cmap = ListedColormap(["#e9eef2", "#5b6470", "#d6604d"])  # road, mountain (wall), hazard

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.imshow(road.grid, cmap=cmap, vmin=0, vmax=2, aspect="auto")

    # faint grid lines to read it as a grid-world MDP
    for x in range(road.width):
        ax.axvline(x + 0.5, color="white", lw=0.4, alpha=0.5)
    for y in range(road.height):
        ax.axhline(y + 0.5, color="white", lw=0.4, alpha=0.5)

    ax.scatter(*road.goal[::-1], marker="*", c="gold", s=320, edgecolors="black", zorder=9)
    _draw_car(ax, road.start)

    # the two moves that change position from the start cell: up and right (down/left bump the border).
    # (x=col, y=row); the car sits at road.start, up = decreasing row.
    move = "#1f3b73"
    sy, sx = road.start  # (row, col)
    ax.add_patch(
        FancyArrowPatch(
            (sx, sy - 0.45), (sx, sy - 1.1), arrowstyle="-|>", mutation_scale=15, lw=2.2, color=move, zorder=10
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (sx + 0.45, sy), (sx + 1.1, sy), arrowstyle="-|>", mutation_scale=15, lw=2.2, color=move, zorder=10
        )
    )

    # crop off the outer wall ring -> thin frame (same convention as the curve-figure snapshots)
    ax.set_xlim(0.5, road.width - 1.5)
    ax.set_ylim(road.height - 1.5, 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#444444")

    handles = [
        Patch(facecolor="#e9eef2", edgecolor="#444444", label="road (free)"),
        Patch(facecolor="#5b6470", label="mountain (wall)"),
        Patch(facecolor="#d6604d", label="hazard  (p = 0.1)"),
        plt.Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="gold",
            markeredgecolor="black",
            markersize=15,
            label="goal",
            lw=0,
        ),
        plt.Line2D(
            [0], [0], marker="s", color="w", markerfacecolor="#1f3b73", markersize=11, label="start (car)", lw=0
        ),
        plt.Line2D([0], [0], color="#1f3b73", lw=2.2, label="move  a ∈ {↑,↓,←,→}"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)

    out = Path(__file__).resolve().parents[2] / "plots"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "credal_driving_map.pdf", bbox_inches="tight")
    fig.savefig(out / "credal_driving_map.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
