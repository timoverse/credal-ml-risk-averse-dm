"""Self-driving car on a mountain road: how the agents route past an under-observed hazard.

A car must reach its destination; the short route crosses a hazardous stretch (rockfall / flooding)
with an unknown crash probability, the long detour goes safely around the mountain. Each agent plans
from a fixed offline dataset that has observed the hazard a limited number of times (the x-axis), so
the epistemic uncertainty over the hazard never resolves itself away. The shortcut is tuned to be
EV-GOOD (worth it on average) but tail-risky.

The three swept metrics (mean reward, population CVaR reward pooled over seeds x episodes, mean
accident rate) are shown per agent vs. the number of hazardous-cell observations, with road-map
snapshots of the routes taken beside them:

- `mle` (risk-neutral, value iteration on p_mle): always takes the shortcut -- best mean reward,
  worst tail.
- `aleatoric_cvar` (min(1, p_mle/beta) in value iteration): a per-cell CVaR surrogate on the point
  estimate; reacts once it has observed crashes, then avoids. Computed, not drawn.
- `credal_minimax` (value iteration on the upper credal bound p_high): epistemic robustness -- avoids
  while the hazard is under-observed, then aligns with the MLE once p_high shrinks below threshold.
  Computed, not drawn.
- `cvar_minimax` / `cvar_mle` (cvar_agent.py): the cost-sensitive CVaR-minimax rule, planned per cell
  by budget-augmented value iteration -- it minimises the CVaR of the *return* under the worst case
  of the per-cell credal intervals [p_low, p_high] or under the single MLE distribution (p_mle),
  jointly with the VaR threshold v. The credal one is the robust risk-averse agent (always detours);
  the MLE one walks into the under-observed hazard.

Run:            uv run python src/experiments/credal_driving_curve.py --setting {default|twin_peaks}
Re-plot only:   uv run python src/experiments/credal_driving_curve.py --setting <name> --plot-only
                (reuses the pickled results of that setting's last full run; no sweep)
One column:     add `--layout onecol` to --plot-only; writes <plot_path stem>_onecol.pdf/.png next to
                the wide figure, which is left untouched

Type, strokes and the color of each agent come from plotting/paper_style.py, the look every figure
of the paper shares; the type is sized per layout so that it prints at the reference's size at the
width the paper includes the figure at (see `_LAYOUTS` and `_plot`).
"""

from __future__ import annotations

import argparse
import logging
import pickle
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.path as mpath
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.gridspec import GridSpec
from matplotlib.mathtext import MathTextParser
from matplotlib.offsetbox import DrawingArea
from matplotlib.patches import Circle, FancyBboxPatch, PathPatch
from matplotlib.textpath import TextPath
from matplotlib.ticker import NullLocator, ScalarFormatter
from omegaconf import OmegaConf
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # put src/ on the path

from driving.agents import AGENTS, crash_belief  # noqa: E402
from driving.cvar_agent import cvar_route  # noqa: E402
from driving.evaluate import cvar_of_returns, evaluate_route  # noqa: E402
from driving.gridworld import HAZARD, WALL, build_road  # noqa: E402
from driving.offline import collect_observations  # noqa: E402
from driving.planner import intended_route, value_iteration  # noqa: E402
from plotting.paper_style import (  # noqa: E402
    COLUMN_WIDTH,
    FP_LIGHT,
    FP_REGULAR,
    FP_SEMIBOLD,
    METHODS,
    OUR_RULE,
    OURS,
    OURS_TAG,
    REFERENCE_GRAY,
    REFERENCE_ZOOM,
    TEXT_WIDTH,
    Sizes,
    align_ylabels,
    band_style,
    font,
    line_style,
    ours_label,
    set_xlabel,
    set_ylabel,
    style_axes,
    use_fira_mathtext,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.figure import Figure
    from matplotlib.text import Text

logger = logging.getLogger(__name__)

# the analytical agents (AGENTS) plus two reward-CVaR-minimax agents (budget-augmented value
# iteration, v optimised with the policy): cvar_minimax over the credal intervals and cvar_mle over
# the single-distribution MLE (p_mle).
ORDER = (*AGENTS, "cvar_minimax", "cvar_mle")
# each reward-CVaR agent's (lower, upper) belief grid keys the augmented value iteration plans against
_CVAR_INTERVAL = {"cvar_minimax": ("p_low", "p_high"), "cvar_mle": ("p_mle", "p_mle")}

# Each drawn agent's method in paper_style.METHODS, which fixes its color, marker and line style in
# every figure of the paper: the MLE's dark gray, ours in red, and the MLE ablation of our rule in a
# lighter gray, dashed and with its own marker. credal_minimax (EV-robust) and aleatoric_cvar (the
# per-cell CVaR surrogate) are still computed but not drawn.
_METHOD = {"mle": "base", "cvar_mle": "base+rule", "cvar_minimax": OURS}
_COLORS = {agent: METHODS[key].color for agent, key in _METHOD.items()}
# dash and gap, in points at scale 1, of the vertical guides that tie the snapshots to their
# observation count. In points rather than in line widths, matplotlib's unit: the guides are as
# thin as the panel box, and their dashes must not shrink with that hairline.
_GUIDE_DASH_PT = (2.1, 1.4)
_ARROW_GAP = 1.2  # gap between a y label and its tick labels, in label sizes; holds the direction arrow
_LABELS = {
    "mle": METHODS["base"].label,
    # the rule is ours, the MLE under it is not: only the rule part is semibold (paper_style.OUR_RULE)
    "cvar_mle": f"{METHODS['base'].label} {OUR_RULE}",
    "cvar_minimax": ours_label("Credal + rule"),
}
_NEON_ORANGE = "#ff6d00"
_XTICKS = (1, 2, 4, 8, 20, 50)
# the drawn agents, in the legend's order: baseline -> ablation -> ours
_LEGEND_ORDER = ("mle", "cvar_mle", "cvar_minimax")


def _crosses_hazard(route: list, road) -> bool:  # noqa: ANN001
    """Whether a route steps on any hazard (hazard) cell."""
    return any(road.grid[cell] == HAZARD for cell in route)


def _route_classes(routes: list, road) -> list[tuple[float, bool, list]]:  # noqa: ANN001
    """Split routes into behaviour classes (crosses hazard vs. detours).

    Different seeds give different offline datasets and therefore different policies, so an agent in
    the transition region is a *mix* of behaviours -- a single route would misrepresent the crash
    rate. Returns (fraction_of_seeds, crosses_hazard, representative_route) per class that occurs,
    so the snapshot can draw each behaviour with opacity proportional to how often it is taken.
    """
    crosses = [_crosses_hazard(r, road) for r in routes]
    out = []
    for is_cross in (True, False):
        pool = [tuple(r) for r, c in zip(routes, crosses, strict=True) if c == is_cross]
        if pool:
            # the most common route of the class; ties (e.g. every high-k MLE weave being unique)
            # resolve to the first seed's route. The weaving through individually-observed hazard
            # cells is deliberate to show: the MLE searches the cheapest-LOOKING thread by p_mle.
            rep = list(Counter(pool).most_common(1)[0][0])
            out.append((len(pool) / len(routes), is_cross, rep))
    return out


# metric key in evaluate_route -> (results key, axis label) for the three swept curves.
# crash_rate and mean_return are means of per-seed values (identical to pooled means, equal episode
# counts). cvar_return is the POPULATION CVaR: returns pooled over all seeds x episodes, one shared
# VaR threshold across seeds -- the paper's marginal CVaR. A mean of
# per-seed CVaRs would let lucky seeds offset unlucky seeds' tails (the instance-wise aggregation
# the paper argues against) and is always at least as favourable as the population CVaR.
_METRICS = {
    "crash_rate": ("crash_rate", "mean accident rate"),
    "mean_return": ("mean_reward", "mean reward"),
    "cvar_return": ("cvar_reward", "CVaR reward"),
}


def run(cfg, setting: str) -> dict:  # noqa: ANN001
    """Sweep hazard observations: per-agent accident rate / mean reward / CVaR reward + route snapshots."""
    c = cfg.settings[setting]
    road = build_road(c.world, p_hazard=c.p_hazard)
    rewards = {"step_cost": c.step_cost, "goal_reward": c.goal_reward, "crash_penalty": c.crash_penalty}
    s_max = None if c.s_max is None else int(c.s_max)
    plan = partial(_cvar_routes, road, beta=c.beta, s_max=s_max, v_step=float(c.cvar_v_step), rewards=rewards)
    snapshot_k = {int(k) for k in c.snapshot_k}
    swept = {res_key: {r: [] for r in ORDER} for res_key, _ in _METRICS.values()}
    swept_std = {res_key: {r: [] for r in ORDER} for res_key, _ in _METRICS.values()}
    snapshots: dict[int, dict[str, list]] = {}

    with ProcessPoolExecutor() as executor:
        for k_hazard in c.k_hazard_values:
            seed_grids = [
                collect_observations(
                    road,
                    n_case=c.n_case,
                    k_hazard=int(k_hazard),
                    alpha=c.alpha,
                    rng=np.random.default_rng(cfg.seed + s),
                ).predict_cell_grid(road)
                for s in range(c.n_seeds)
            ]
            # the reward-CVaR plans are the sweep's hot loop and independent across seeds
            cvar_routes = list(executor.map(plan, seed_grids))
            acc = {m: {r: [] for r in ORDER} for m in _METRICS if m != "cvar_return"}
            seed_returns: dict[str, list] = {r: [] for r in ORDER}  # raw rollout returns, pooled below
            routes_seen = {r: [] for r in ORDER}
            for s, grids in enumerate(seed_grids):
                for rule in ORDER:
                    if rule in _CVAR_INTERVAL:
                        route = cvar_routes[s][rule]
                    else:
                        belief = crash_belief(rule, grids, beta=c.beta)
                        policy, _ = value_iteration(road, belief, gamma=c.gamma, **rewards)
                        route = intended_route(road, policy)
                    metrics = evaluate_route(
                        road,
                        route,
                        n_episodes=cfg.eval.n_episodes,
                        cvar_beta=c.beta,
                        rng=np.random.default_rng(7000 + s),
                        **rewards,
                    )
                    for m in acc:
                        acc[m][rule].append(metrics[m])
                    seed_returns[rule].append(metrics["returns"])
                    routes_seen[rule].append(route)
            for m, (res_key, _) in _METRICS.items():
                for rule in ORDER:
                    if m == "cvar_return":
                        # population CVaR: one tail over all seeds x episodes -- a single pooled
                        # statistic per (agent, k), so it is drawn without a band (std stays 0)
                        swept[res_key][rule].append(cvar_of_returns(np.concatenate(seed_returns[rule]), c.beta))
                        swept_std[res_key][rule].append(0.0)
                    else:
                        swept[res_key][rule].append(float(np.mean(acc[m][rule])))
                        swept_std[res_key][rule].append(float(np.std(acc[m][rule])))
            if int(k_hazard) in snapshot_k:
                # store the RAW per-seed routes; the behaviour classes and the representative route
                # drawn per class are a pure plotting decision (_route_classes, applied in _build)
                snapshots[int(k_hazard)] = {r: [list(map(tuple, rt)) for rt in routes_seen[r]] for r in ORDER}
            logger.info("k=%d: crash rate %s", k_hazard, {r: round(swept["crash_rate"][r][-1], 3) for r in ORDER})

    results = {
        "setting": setting,
        "road": road,
        "k_values": list(c.k_hazard_values),
        "beta": c.beta,
        "snapshots": snapshots,
        "snapshot_k": [int(k) for k in c.snapshot_k],
        **swept,
        **{f"{res_key}_std": swept_std[res_key] for res_key, _ in _METRICS.values()},
    }
    _save_results(results, c.plot_path)
    _plot(results, c.plot_path)
    return results


def _results_cache(plot_path: str) -> Path:
    """Where the pickled sweep results live (next to the figure, so --plot-only can reuse them)."""
    return Path(plot_path).with_suffix(".pkl")


def _save_results(results: dict, plot_path: str) -> None:
    """Pickle the sweep results so plot styling can be iterated without re-running the sweep."""
    cache = _results_cache(plot_path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as f:
        pickle.dump(results, f)


def _cvar_routes(road, grids: dict, *, beta: float, s_max: int | None, v_step: float, rewards: dict) -> dict:  # noqa: ANN001
    """One seed's route per reward-CVaR agent, planned from its own belief grids (a pool task)."""
    return {
        rule: cvar_route(road, grids[lo], grids[hi], beta=beta, s_max=s_max, v_step=v_step, **rewards)
        for rule, (lo, hi) in _CVAR_INTERVAL.items()
    }


def _lane_offset(xs: np.ndarray, ys: np.ndarray, d: float) -> tuple[np.ndarray, np.ndarray]:
    """Offset a polyline perpendicular to its local direction: parallel driving lanes.

    Unlike a constant diagonal shift, the lane follows the route around corners at a fixed
    lateral distance, so it never drifts into the mountains alongside a hugging route.
    """
    t = np.stack([np.gradient(xs), np.gradient(ys)], axis=1)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    return xs - d * t[:, 1], ys + d * t[:, 0]


def _chaikin(xs: np.ndarray, ys: np.ndarray, iters: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Smooth a polyline by Chaikin corner-cutting, so the route reads as a natural driving line."""
    pts = list(zip(xs, ys, strict=True))
    for _ in range(iters):
        new = [pts[0]]
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:], strict=True):
            new.append((0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1))
            new.append((0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1))
        new.append(pts[-1])
        pts = new
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def _route_mask(ax, cx, cy, sc=1.0) -> None:  # noqa: ANN001
    """An invisible white cover over a cell, hiding the route-lane ends behind the car/goal badge.

    Both the start and the goal sit on WHITE checker cells, so the mask itself does not show.
    """
    ax.add_patch(
        FancyBboxPatch(
            (cx - 0.42 * sc, cy - 0.42 * sc),
            0.84 * sc,
            0.84 * sc,
            boxstyle=f"round,pad=0,rounding_size={0.18 * sc}",
            linewidth=0,
            facecolor="white",
            zorder=7.4,
        )
    )


def _draw_car(ax, cell, sc=1.0, color="#1f3b73", lw=1.0) -> None:  # noqa: ANN001
    """A side-view car filling the start cell; big enough that every route lane starts behind it.

    `sc` scales the badge's cell-unit geometry, so it keeps its visual size on big maps; `lw` scales
    its outlines (the figure passes `Sizes.scale`, other callers keep the full weight).
    """
    cy, cx = cell
    _route_mask(ax, cx, cy, sc)
    ax.add_patch(  # cabin, tucked behind the body (wide enough to cover the outer route lanes)
        FancyBboxPatch(
            (cx - 0.31 * sc, cy - 0.40 * sc),
            0.62 * sc,
            0.30 * sc,
            boxstyle=f"round,pad=0,rounding_size={0.12 * sc}",
            linewidth=0.5 * lw,
            edgecolor="black",
            facecolor=color,
            zorder=7.8,
        )
    )
    for wx in (cx - 0.26 * sc, cx + 0.06 * sc):  # windows
        ax.add_patch(
            FancyBboxPatch(
                (wx, cy - 0.35 * sc),
                0.20 * sc,
                0.17 * sc,
                boxstyle=f"round,pad=0,rounding_size={0.05 * sc}",
                linewidth=0.3 * lw,
                edgecolor="black",
                facecolor="#a8c6e8",
                zorder=8.1,
            )
        )
    ax.add_patch(  # body with rounded bonnet and boot
        FancyBboxPatch(
            (cx - 0.46 * sc, cy - 0.14 * sc),
            0.92 * sc,
            0.30 * sc,
            boxstyle=f"round,pad=0,rounding_size={0.11 * sc}",
            linewidth=0.5 * lw,
            edgecolor="black",
            facecolor=color,
            zorder=8,
        )
    )
    for wx in (cx - 0.26 * sc, cx + 0.26 * sc):  # wheels with light hubcaps
        ax.add_patch(
            Circle(
                (wx, cy + 0.20 * sc), 0.13 * sc, facecolor="#222222", edgecolor="black", linewidth=0.4 * lw, zorder=8.2
            )
        )
        ax.add_patch(Circle((wx, cy + 0.20 * sc), 0.052 * sc, facecolor="#d9d9d9", edgecolor="none", zorder=8.3))


def _plot_metric(  # noqa: PLR0913
    ax,  # noqa: ANN001
    k_values: list,
    data: dict,
    data_std: dict,
    ylabel: str,
    snaps: list,
    sizes: Sizes,
    show_x: bool = True,
    clip: tuple | None = None,
    yticks: tuple | None = None,
    ycenter: float | None = None,
    xticks: tuple = _XTICKS,
    band: bool = True,
) -> None:
    """One swept-metric panel vs. the number of hazardous-cell observations (log x).

    Central line per agent, with a +-1 std shaded band across seeds where the metric is a per-seed
    mean (mean reward, crash rate); the population CVaR is one pooled statistic per (agent, k), so
    its panel is drawn without a band (`band=False`). A white halo keeps the lines readable.
    `show_x=False` hides the x tick labels for the upper panels of a shared-x stack.
    """
    k = np.asarray(k_values, dtype=float)
    if band:  # bands first, under every line
        for rule in _LEGEND_ORDER:
            mean = np.asarray(data[rule])
            std = np.asarray(data_std[rule])
            lo, hi = mean - std, mean + std
            if clip is not None:
                lo, hi = np.clip(lo, *clip), np.clip(hi, *clip)
            ax.fill_between(k, lo, hi, **band_style(_COLORS[rule]))
    for rule in _LEGEND_ORDER:
        ax.plot(k, data[rule], **line_style(_METHOD[rule], sizes))
    ax.margins(y=0.14)  # keep the markers clear of the top/bottom borders
    if ycenter is not None:  # symmetric y-range around a reference value
        y0, y1 = ax.get_ylim()
        half = max(ycenter - y0, y1 - ycenter)
        ax.set_ylim(ycenter - half, ycenter + half)
    ax.set_xscale("log")
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    set_ylabel(ax, ylabel, sizes)
    # the pad has to clear the direction arrow that is drawn into this gap (see _build)
    ax.yaxis.labelpad = _ARROW_GAP * sizes.label
    if yticks is not None:
        ax.set_yticks(yticks)
    else:
        ax.locator_params(axis="y", nbins=4)  # few y-ticks so the short panels stay uncluttered
    guide_dash = (0, tuple(v * sizes.scale / sizes.spine for v in _GUIDE_DASH_PT))
    for x in snaps:  # mark the observation counts shown as snapshots beside the curves
        ax.axvline(x, color=REFERENCE_GRAY, lw=sizes.spine, ls=guide_dash, zorder=1)
    if show_x:
        set_xlabel(ax, "hazard cell observations (k)", sizes)
    else:
        ax.tick_params(labelbottom=False)
    style_axes(ax, sizes)
    # after style_axes, which sets the ticks: matplotlib's 3.5pt pad is for full-size type, and
    # under the short ticks of the x-axis even its scaled value leaves a visible gap
    ax.tick_params(axis="y", pad=3.5 * sizes.scale)
    ax.tick_params(axis="x", pad=1.8 * sizes.scale)


def _block_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box (r0, r1, c0, c1), inclusive, of the True cells of `mask`."""
    rows, cols = np.where(mask)
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def _scale(road) -> float:  # noqa: ANN001
    """Cell-unit scale factor: 1.0 on the original 13-wide map, growing with the grid size."""
    return max(road.grid.shape) / 13.0


_MAP_REF_PT = 48.0  # map height (pt) the point-sized snapshot glyphs were tuned on


def _view_span(road, sc: float) -> tuple[float, float]:  # noqa: ANN001
    """Width and height, in data units, of the view a snapshot sets (see `_plot_snapshot`)."""
    if sc == 1.0:
        return road.width - 1.0, road.height - 1.0 + 0.18
    m = 0.55 * sc
    return road.width - 1.0 + 2 * m, road.height - 1.0 + 2 * m


def _fade_to_white(ax, band_frac: float = 0.11, ease: float = 1.35) -> None:  # noqa: ANN001
    """Lay a white veil over the finished snapshot, opaque at the border and clear in the middle.

    Drawn ON TOP of the map rather than baked into the checkerboard's alpha, so EVERYTHING fades
    together -- grid, mountains, hazard zone and the route lines all dissolve into the page instead
    of the routes running crisply off the edge of a fading grid. It sits below the car and goal
    badges (zorder 8/9), which stay solid wherever they land.

    `band_frac` is the ramp width as a fraction of the shorter view side, `ease` its curve (>1 holds
    the white in longer near the border). These two are the knobs for how soft the edge reads.
    """
    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = sorted(ax.get_ylim())
    n = 240  # sampled fine and drawn bilinear: the veil must be smooth, not per cell
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y1, y0, n)  # top row first, to match imshow's default origin="upper"
    gx, gy = np.meshgrid(xs, ys)
    dist = np.minimum.reduce([gx - x0, x1 - gx, gy - y0, y1 - gy])
    band = band_frac * min(x1 - x0, y1 - y0)
    alpha = 1.0 - np.clip(dist / band, 0.0, 1.0) ** ease
    veil = np.ones((n, n, 4))
    veil[..., 3] = alpha
    ax.imshow(veil, aspect="auto", interpolation="bilinear", extent=(x0, x1, y0, y1), zorder=7)


def _glyph_scale(ax, road, sc: float) -> float:  # noqa: ANN001
    """Shrink factor for the snapshot's POINT-sized glyphs, from the map's drawn height.

    The badge and the hazard `!` are sized in points, so they do not follow the map when the figure
    geometry changes -- on a short figure they grow relative to the cells and start to poke out of
    the view. Their height (aspect-locked, so possibly width-bound) against the geometry they were
    tuned on gives one factor that keeps them proportional to the world. They are pictograms that
    belong to the map, so they take this factor and not the type scale of `Sizes`.
    """
    fig = ax.figure
    pos = ax.get_position()  # the gridspec cell; apply_aspect has not run yet
    w_span, h_span = _view_span(road, sc)
    drawn_pt = min(pos.height * fig.get_figheight(), pos.width * fig.get_figwidth() * h_span / w_span) * 72
    return float(np.clip(drawn_pt / _MAP_REF_PT, 0.7, 1.0))


def _draw_road(ax, road, gs: float = 1.0, lw: float = 1.0) -> None:  # noqa: ANN001
    """The map background: checkerboard tarmac, the mountain block(s), and the hazard zone.

    The grid shows as a white / almost-white checkerboard. Each mountain (connected wall block) is
    a rounded white patch with a black border sitting ON TOP of the checkerboard -- visibly not
    part of the drivable grid. The hazard stretch is marked like a danger zone in a game:
    neon-orange border, a faint dotted orange floor, and an exclamation mark in the middle.
    All cell-unit geometry scales with `_scale(road)` so big maps keep the same visual weight;
    `gs` shrinks the point-sized glyphs with the drawn map and `lw` scales the outlines (`Sizes.scale`).
    """
    h, w = road.grid.shape
    sc = _scale(road)
    if sc == 1.0:
        # upsampled checkerboard with a per-pixel alpha that fades the gridworld out towards the
        # axes border (no hard outline; the map dissolves into the white margin)
        s = 8
        rr, cc = np.mgrid[0 : h * s, 0 : w * s]
        cell_r, cell_c = rr // s, cc // s
        checker = ((cell_r + cell_c) % 2)[..., None]
        rgb = np.where(checker == 0, to_rgb("#ffffff"), to_rgb("#e6e9ef"))
        # the fade towards the border is NOT baked in here any more -- `_fade_to_white` lays a white
        # veil over the finished map so the routes dissolve with the grid (see there)
        alpha = np.ones(rr.shape)
        ring = (cell_r == 0) | (cell_r == h - 1) | (cell_c == 0) | (cell_c == w - 1)
        alpha[ring] = 0.0  # the cropped outer wall ring is pure margin
        rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
        ax.imshow(rgba, aspect="auto", interpolation="nearest", extent=(-0.5, w - 0.5, h - 0.5, -0.5), zorder=0)
    else:
        # one crisp VECTOR quad per world cell (image pixels alias/blur at ~2px per cell): the
        # very fine texture is deliberate -- it shows the size of the world. The alpha fade is
        # per cell and only over the outermost few cells.
        rr, cc = np.mgrid[0:h, 0:w]
        checker = ((rr + cc) % 2)[..., None]
        rgb = np.where(checker == 0, to_rgb("#ffffff"), to_rgb("#e6e9ef"))
        # the fade lives in `_fade_to_white`, not in these cells' alpha
        alpha = np.ones(rr.shape)
        alpha[(rr == 0) | (rr == h - 1) | (cc == 0) | (cc == w - 1)] = 0.0  # cropped wall ring
        rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
        mesh = ax.pcolormesh(
            np.arange(w + 1) - 0.5,
            np.arange(h + 1) - 0.5,
            np.zeros((h, w)),
            zorder=0,
            antialiased=False,
        )
        mesh.set_array(None)  # drop the scalar mapping; the per-cell RGBA below must win at draw
        mesh.set_color(rgba.reshape(-1, 4))

    interior_wall = road.grid == WALL
    interior_wall[0, :] = interior_wall[-1, :] = False  # the outer ring is cropped, not drawn
    interior_wall[:, 0] = interior_wall[:, -1] = False
    if sc == 1.0:  # the published small map: one rounded rectangle per mountain block
        blocks, n_blocks = ndimage.label(interior_wall)
        for b in range(1, n_blocks + 1):
            r0, r1, c0, c1 = _block_bbox(blocks == b)
            ax.add_patch(
                FancyBboxPatch(
                    (c0 - 0.42, r0 - 0.42),
                    (c1 - c0) + 0.84,
                    (r1 - r0) + 0.84,
                    boxstyle="round,pad=0,rounding_size=0.45",
                    linewidth=0.88 * lw,
                    edgecolor="black",
                    facecolor="white",
                    zorder=2,
                )
            )
    else:
        # big maps have blob-shaped mountains: render the smoothed wall mask as filled contours,
        # which follows ANY shape (U-forms, notches) with naturally rounded edges. The level sits
        # slightly above 0.5, pulling the drawn edge ~half a cell INTO the mountain so routes
        # hugging the boundary keep visible clearance.
        smooth = ndimage.gaussian_filter(interior_wall.astype(np.float64), sigma=2.0)
        ax.contourf(np.arange(w), np.arange(h), smooth, levels=[0.58, 2.0], colors=["white"], zorder=2)
        ax.contour(
            np.arange(w), np.arange(h), smooth, levels=[0.58], colors=["black"], linewidths=0.88 * lw, zorder=2.05
        )

    r0, r1, c0, c1 = _block_bbox(road.grid == HAZARD)
    pad = 0.44  # true cell units, like the mountain padding: the zone must not swallow the pass
    # a hazard stretch at the map's left wall bleeds out towards the panel border (view starts at
    # x=0, as on the default map); an interior hazard zone gets a symmetric box
    x0 = 0.06 if c0 <= 1 else c0 - pad
    y0 = r0 - pad
    wdt, hgt = (c1 + pad) - x0, (r1 - r0) + 2 * pad
    dots_x = np.arange(x0 + 0.22 * sc, x0 + wdt - 0.1 * sc, 0.38 * sc)  # spacing ~ constant in pt
    dots_y = np.arange(y0 + 0.22 * sc, y0 + hgt - 0.1 * sc, 0.38 * sc)
    mx, my = np.meshgrid(dots_x, dots_y)
    ax.scatter(mx, my, s=0.7, c=_NEON_ORANGE, alpha=0.45, lw=0, zorder=1.6)
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            wdt,
            hgt,
            boxstyle=f"round,pad=0,rounding_size={0.18 * sc}",
            linewidth=0.96 * lw,
            edgecolor=_NEON_ORANGE,
            facecolor=to_rgba(_NEON_ORANGE, 0.06),
            zorder=1.5,
        )
    )
    ax.text(
        x0 + wdt / 2,
        (r0 + r1) / 2,
        "!",
        color=_NEON_ORANGE,
        fontproperties=FP_SEMIBOLD,
        fontsize=(11 if sc == 1.0 else 8) * gs,  # sized to fit inside the big-map hazard belt
        ha="center",
        va="center",
        zorder=5,
        path_effects=[pe.withStroke(linewidth=2.2 * gs, foreground="white")],
    )


def _draw_goal(ax, cell, sc=1.0, gs: float = 1.0, lw: float = 1.0) -> None:  # noqa: ANN001
    """A green checkmark on a white badge at the goal cell; the badge hides the route line ends.

    Both are point-sized scatter markers, so they stay circular / undistorted regardless of the
    panel's (non-square) data aspect; `gs` keeps them proportional to the map (see `_glyph_scale`)
    and `lw` scales the badge's outline.
    """
    gy, gx = cell
    # big maps: smaller badge (the car glyph is the size anchor) and a tighter mask, so neither
    # pokes past the badge or the panel border
    badge, mark, mark_lw, mask_sc = (70, 22, 1.2, sc) if sc == 1.0 else (30, 9, 0.9, 0.55 * sc)
    badge, mark, mark_lw = badge * gs**2, mark * gs**2, mark_lw * gs * lw  # scatter sizes are areas
    _route_mask(ax, gx, gy, mask_sc)
    ax.scatter([gx], [gy], s=badge, facecolor="white", edgecolor="black", linewidths=0.6 * gs * lw, zorder=8)
    check = mpath.Path([(-0.7, 0.05), (-0.15, -0.5), (0.75, 0.6)], [mpath.Path.MOVETO] + [mpath.Path.LINETO] * 2)
    ax.scatter([gx], [gy], s=mark, marker=check, facecolor="none", edgecolor="#3aa85f", linewidths=mark_lw, zorder=9)


def _plot_snapshot(ax, road, classes, k_label, sizes: Sizes) -> None:  # noqa: ANN001
    """A snapshot of the map with each agent's route(s); opacity ~ fraction of seeds taking them.

    `classes[rule]` is a list of (fraction, crosses_hazard, route) from `_route_classes`, so an agent
    that is split across seeds (e.g. the MLE mid-transition) draws BOTH its routes, the faint one
    being the minority behaviour. Three lanes: MLE, MLE + rule (dashed as in the metric panels),
    and ours.
    """
    sc = _scale(road)
    gs = _glyph_scale(ax, road, sc)  # keeps the point-sized glyphs proportional to the drawn map
    lane_lw = 1.0 * sizes.scale  # one width for the three lanes, which run side by side
    _draw_road(ax, road, gs, sizes.scale)
    # lane separation: the published small map keeps its diagonal shifts; big maps use true
    # perpendicular lanes (ours on the side pointing AWAY from the mountain a detour hugs)
    legacy = {"mle": -0.2, "cvar_mle": 0.0, "cvar_minimax": 0.2}
    lanes = {"mle": 0.13 * sc, "cvar_mle": 0.0, "cvar_minimax": -0.13 * sc}
    for rule in _LEGEND_ORDER:
        for frac, _is_cross, route in classes[rule]:
            xs = np.array([cell[1] for cell in route], dtype=float)
            ys = np.array([cell[0] for cell in route], dtype=float)
            if sc == 1.0:
                xs, ys = xs + legacy[rule], ys + legacy[rule]
            else:
                xs, ys = _lane_offset(xs, ys, lanes[rule])
            sx, sy = _chaikin(xs, ys)
            # opacity is simply the fraction of seeds taking this route
            ax.plot(
                sx,
                sy,
                color=_COLORS[rule],
                lw=lane_lw,
                linestyle=METHODS[_METHOD[rule]].linestyle,
                alpha=frac,
                solid_capstyle="round",
                dash_capstyle="butt",
                zorder={"mle": 3.5, "cvar_mle": 4, "cvar_minimax": 6}[rule],
            )
    _draw_goal(ax, road.goal, sc, gs, sizes.scale)
    _draw_car(ax, road.start, sc, lw=sizes.scale)
    # the tag sits in the lower-right corner of the map. Big maps pull it in from the corner, clear
    # of the wide fade-out band. On the small map the detours hug the mountain, which leaves the
    # outermost road row free: the tag stands in it, flush with the right edge of the checkerboard
    # and with its foot at the bottom of the view, so the view needs no extra margin for it.
    w_span, h_span = _view_span(road, sc)
    if sc == 1.0:
        k_x, k_y = w_span - 0.55, h_span - 0.18 - 0.1
        k_ha, k_va, k_transform = "right", "bottom", ax.transData
    else:
        k_x, k_y, k_ha, k_va, k_transform = 0.93, 0.085, "right", "bottom", ax.transAxes
    ax.text(
        k_x,
        k_y,
        f"k = {k_label}",
        transform=k_transform,
        ha=k_ha,
        va=k_va,
        fontproperties=FP_SEMIBOLD,
        fontsize=sizes.tick,
        color="0.45",
        zorder=9,
    )
    # the outer wall ring stays inside the view as a white margin around the faded gridworld;
    # big maps add a real margin so the car/goal badges sit at the end of the WORLD, then the
    # checkerboard fades out, then comes the panel border
    if sc == 1.0:
        ax.set_xlim(0.0, w_span)
        ax.set_ylim(h_span - 0.18, -0.18)
    else:
        m = 0.55 * sc
        ax.set_xlim(-m, w_span - m)
        ax.set_ylim(h_span - m, -m)
    # the view is final, so the veil can size itself to it. The small map is only 11 cells across and
    # its routes and hazard belt run right along the border, so it gets a narrower band than the big
    # map, where there is empty world to dissolve into
    _fade_to_white(ax, band_frac=0.055 if sc == 1.0 else 0.11)
    ax.set_aspect("equal", adjustable="box")  # square grid cells (the panel shrinks to fit)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():  # the map floats free; no frame around the snapshot
        spine.set_visible(False)


# value-label format and better-direction arrow per summarised metric
_SUMMARY_FMT = {"mean_reward": "{:.1f}", "cvar_reward": "{:.1f}", "crash_rate": "{:.2f}"}
_BETTER = {"mean_reward": "↑", "cvar_reward": "↑", "crash_rate": "↓"}
_SUMMARY_TITLE = {"mean_reward": "reward", "cvar_reward": "CVaR", "crash_rate": "crash rate"}

# snapshot rows: (data k, displayed k, note-border alpha, note title, note body). The middle row
# shows the k=4 routes under the k=8 label -- the mid-transition split carries the message better.
_SNAP_ROWS_DEFAULT = (
    (
        1,
        1,
        0.9,
        "High EU",
        "baselines drive into the barely observed hazard;\nonly ours plans against epistemic uncertainty.",
    ),
    (
        4,
        8,
        0.6,
        "Fading EU",
        "baselines only detour after seeing crashes;\nours stays safe without ever observing one.",
    ),
    (
        20,
        20,
        0.35,
        "Low EU",
        "once the hazard is known, the baselines catch up\nto the safe route ours took from the start.",
    ),
)
_SNAP_ROWS_TWIN_PEAKS = (
    (
        1,
        1,
        0.9,
        "High EU",
        "baselines drive into the barely observed hazard;\nonly ours plans against epistemic uncertainty.",
    ),
    (
        90,
        90,
        0.6,
        "Fading EU",
        "baselines only detour after seeing crashes;\nours stays safe without ever observing one.",
    ),
    (
        400,
        400,
        0.35,
        "Low EU",
        "once the hazard is known, the baselines catch up\nto the safe route ours took from the start.",
    ),
)

# per-setting figure styling: snapshot rows and the curve panels' fixed y-geometry (empty dicts =
# let matplotlib pick). The default setting's values are tuned to its published figure.
_STYLE: dict[str, dict] = {
    "default": {
        "snap_rows": _SNAP_ROWS_DEFAULT,
        "yticks": {"mean_reward": (14, 17, 20), "cvar_reward": (-20, 0, 20), "crash_rate": (0.0, 0.2, 0.4)},
        "ycenters": {"mean_reward": 17.0, "cvar_reward": 0.0, "crash_rate": 0.2},
        "xticks": _XTICKS,
        "summary_bar_frac": 0.62,  # bar width as a fraction of the panel; 1-2 digit labels fit the rest
    },
    "twin_peaks": {
        "snap_rows": _SNAP_ROWS_TWIN_PEAKS,
        # same three-tick rhythm as the default setting (auto-ticks gave this one only two per panel,
        # which made the two figures read differently side by side); ycenter mirrors the range about
        # the middle tick, so these have to be symmetric triples
        "yticks": {"mean_reward": (164, 167, 170), "cvar_reward": (-50, 50, 150), "crash_rate": (0.0, 0.1, 0.2)},
        "ycenters": {"mean_reward": 167.0, "cvar_reward": 50.0, "crash_rate": 0.1},
        "xticks": (1, 4, 20, 100, 400),
        "summary_bar_frac": 0.52,  # shorter than default so the 3-digit rewards keep a clear label column
    },
}


def _axes_aspect(ax) -> float:  # noqa: ANN001
    """Physical width/height ratio of an axes, so axes-coordinate roundings can be made circular."""
    pos = ax.get_position()
    fig = ax.figure
    return (pos.width * fig.get_figwidth()) / (pos.height * fig.get_figheight())


# height of the sticker note as a fraction of its cell, centred in it. Was 0.80; 0.64 is the 0.8x
# footprint. The type inside does NOT scale with it -- shrink much further and the body will spill.
_NOTE_H = 0.64


# the one-column layout's notes (keyed by note title) say only what the BASELINES do at each level,
# on two lines of <= 25 characters for its narrow boxes, without closing punctuation. Our agent
# takes the same safe route in all three snapshots, so it is said once, in a banner under them.
_NOTE_BODY_ONECOL = {
    "High EU": "baselines drive into the\nbarely observed hazard",
    "Fading EU": "baselines only detour\nafter seeing crashes",
    "Low EU": "once the hazard is known,\nthe baselines detour too",
}
# That banner: lower case like every annotation, "ours" semibold as in the legend's tag.
_OURS_BANNER = r"$\mathbf{ours}$ takes the safe route at every EU level"

# figure geometry per layout, hand-tuned in inches. "wide" is the two-column figure* (the appendix's
# twin-peaks figure; the small map shares it). "onecol" packs the content into ONE column: curve
# panels ~0.3x as wide, the map column trimmed to the maps, notes about half as wide (see
# `_NOTE_BODY_ONECOL`), and the legend's row set tighter to fit beside the x-label.
# The type is NOT part of the geometry: `print_width` is the width the paper includes the layout
# at, and `_plot` sizes all type and strokes so that they print there like the reference figure's
# (paper_style.Sizes). A panel that the type no longer fits is a geometry problem to solve here.
# The wide layout's part headings sit in the strip above the panels' `top`, `title_dy` label sizes
# above them; the strip is as deep as the headings need at the label size, the panels get the rest.
# The one-column layout has no headings: its panels and snapshot rows run to the top of the canvas,
# and a slim fourth row under its snapshot rows holds the banner about our agent.
_LAYOUTS: dict[str, dict] = {
    "wide": {
        "figsize": (7.9, 2.03),
        "print_width": TEXT_WIDTH,
        "outer": {"width_ratios": [1.3507, 1.0993], "wspace": 0.04, "bottom": 0.11, "top": 0.934},
        "left_ratios": [1.0, 0.28],  # curve panel : sweep-mean bars
        # the snapshot rows claim a strip above and below the metric panels (see `_build`). The map
        # column is wider than the (height-limited) maps, and the note column takes from that slack
        # what its 48-character lines need at the tick size.
        "right": {"width_ratios": [0.67, 1.089], "hspace": 0.028, "wspace": 0.04, "bottom": 0.070, "top": 0.966},
        "note": {},  # `_draw_note` defaults
        "note_bodies": None,  # the two-line notes stored with the snapshot rows
        "legend": {},  # the row's default spacing (see `legend` in `_build`)
        "titles": (
            "A) Evaluation across hazard-observation levels",
            "B) Illustration and interpretation at three EU levels",
        ),
        "title_dy": 0.65,  # the headings' foot above the snapshot rows; the divider overshoots by twice this
        "ours_banner": None,  # its notes say what our agent does
        "pad_inches": 0.1,  # matplotlib's default tight-bbox padding
    },
    "onecol": {
        # widths in inches on the 3.48in canvas: 0.38 y-label gutter | 0.80 curves | 0.60 bars |
        # 0.08 divider gap | 0.62 maps | 0.02 | 0.97 notes. The gutter is what the y labels, their
        # arrows and the tick labels take at this layout's type size, and the notes are as wide as
        # their longest line at the tick size needs (`_NOTE_BODY_ONECOL`).
        "figsize": (3.48, 2.1),
        "print_width": COLUMN_WIDTH,
        "outer": {
            "width_ratios": [1.40, 1.61],
            "wspace": 0.0532,
            "left": 0.1092,
            "right": 0.9971,
            "bottom": 0.11,
            "top": 1.0,
        },
        "left_ratios": [0.80, 0.60],
        # same top and bottom as the metric panels: the legend's upper row sits right under this
        # block, so it must not dip below the floor. The slim fourth row is the banner's.
        "right": {
            "width_ratios": [0.62, 0.97],
            "height_ratios": [1.0, 1.0, 1.0, 0.3],
            "hspace": 0.08,
            "wspace": 0.0252,
            "bottom": 0.11,
            "top": 1.0,
        },
        # same physical box height and corners as wide; heading and body sit a little lower in it,
        # which balances the space above the heading against the space under the body
        "note": {"h": 0.71, "rounding": 0.13, "title_y": 0.31, "body_y": -0.167},
        "note_bodies": _NOTE_BODY_ONECOL,
        # the row has only the room right of the x-label, so it is set tighter than in wide and
        # without the legend's own padding: it ends flush with the note boxes
        "legend": {
            "columnspacing": 1.0,
            "handlelength": 1.5,
            "handletextpad": 0.5,
            "borderpad": 0.0,
            "borderaxespad": 0.0,
        },
        "titles": None,
        "ours_banner": _OURS_BANNER,
        "pad_inches": 0.015,  # the default 0.1in per side would cost 5% of a column's width
    },
}


def _draw_note(  # noqa: PLR0913
    ax,  # noqa: ANN001
    title: str,
    body: str,
    border_alpha: float,
    sizes: Sizes,
    *,
    h: float = _NOTE_H,
    title_y: float = 0.35,
    body_y: float = -0.125,
    linespacing: float = 1.5,
    rounding: float = 0.08,
) -> None:
    """A sticker-style note next to a snapshot.

    A filled rounded box in the hazard orange, fading out as the epistemic uncertainty (EU)
    vanishes: a bold headline and a centred dark-orange body. The keyword knobs are per-layout
    (see `_LAYOUTS`); their defaults are the wide figure's.
    """
    ax.set_axis_off()
    # the box keeps its width and its vertical centre and only gets shorter: `h` is the single
    # knob, and the two text baselines below are derived from it so the type stays put relative to the
    # shrinking box rather than drifting towards its edges.
    y0 = 0.5 - h / 2
    ax.add_patch(
        FancyBboxPatch(
            (0.01, y0),
            0.98,
            h,
            boxstyle=f"round,pad=0.02,rounding_size={rounding}",
            transform=ax.transAxes,
            mutation_aspect=_axes_aspect(ax),  # same physical corner radius in x and y
            linewidth=0.72 * sizes.scale,
            edgecolor=to_rgba(_NEON_ORANGE, border_alpha),
            facecolor=to_rgba(_NEON_ORANGE, 0.12 * border_alpha),
            clip_on=False,
        )
    )
    # -- the knobs to hand-tune this note: leading and the two baselines. The type is the figure's
    # (heading at the legend size, body at the tick size), so what has to give when a line does not
    # fit is the LINE BREAK, which lives with the text itself (_SNAP_ROWS_DEFAULT /
    # _SNAP_ROWS_TWIN_PEAKS / _NOTE_BODY_ONECOL), or the note column's width in `_LAYOUTS`.
    ax.text(
        0.5,
        0.5 + title_y * h,  # title baseline, as a fraction of the box height
        title,
        transform=ax.transAxes,
        va="center",
        ha="center",
        fontproperties=FP_SEMIBOLD,
        fontsize=sizes.legend,
        color="#a34a00",
    )
    ax.text(
        0.5,
        0.5 + body_y * h,  # body block centre, as a fraction of the box height
        body,
        transform=ax.transAxes,
        va="center",
        ha="center",
        ma="center",
        fontproperties=FP_REGULAR,
        fontsize=sizes.tick,
        color="#a34a00",
        linespacing=linespacing,  # leading between the wrapped lines
    )


def _draw_ours_banner(ax, text: str, sizes: Sizes) -> None:  # noqa: ANN001
    """The one statement about OUR agent: a slim strip under the snapshot rows, in its red.

    Same sticker look as the notes, but spanning maps and notes and wearing the color of our route
    instead of the hazard orange, so it reads as the summary of all three rows.
    """
    ax.set_axis_off()
    ours = _COLORS["cvar_minimax"]
    ax.add_patch(
        FancyBboxPatch(
            (0.012, 0.15),
            0.982,  # right edge flush with the note boxes' (their pad overhangs the cell a little)
            0.70,
            boxstyle="round,pad=0.012,rounding_size=0.035",
            transform=ax.transAxes,
            mutation_aspect=_axes_aspect(ax),  # same physical corner radius in x and y
            linewidth=0.72 * sizes.scale,  # the notes' edge
            edgecolor=to_rgba(ours, 0.9),
            facecolor=to_rgba(ours, 0.10),
            clip_on=False,
        )
    )
    # Centred by its INK, from the top of the ascenders to the foot of the descender, so the gap to
    # the box is the same above and below. va="center" centres the line box instead, whose descent
    # is deeper than this line's one "y" (text 0.45pt high); centring only the ascender-to-baseline
    # band ignores that "y" (0.55pt low). Here the two agree with the x-height band on the middle.
    plain = re.sub(r"\$\\mathbf\{(.*?)\}\$", r"\1", text)
    ink = TextPath((0, 0), plain, size=sizes.tick, prop=FP_REGULAR).get_extents()
    cell_pt = ax.get_position().height * ax.figure.get_figheight() * 72
    label = ax.text(
        0.5,
        0.5 - (ink.y0 + ink.y1) / 2 / cell_pt,
        text,
        transform=ax.transAxes,
        va="baseline",
        ha="center",
        fontproperties=FP_REGULAR,
        fontsize=sizes.tick,  # the notes' body size
        color=ours,
    )
    drop = mtransforms.ScaledTranslation(0, -_mathtext_rise(label) / 72, ax.figure.dpi_scale_trans)
    label.set_transform(label.get_transform() + drop)


def _tip_rounded_bar(base: float, tip: float, y: float, h: float) -> mpath.Path:
    """A horizontal bar path with sharp corners at the base and rounded corners at the tip."""
    r = min(h / 2, abs(tip - base) / 2, 0.05)
    sgn = 1.0 if tip >= base else -1.0
    xr = tip - sgn * r
    yb, yt = y - h / 2, y + h / 2
    p = mpath.Path
    verts = [(base, yb), (xr, yb), (tip, yb), (tip, yb + r), (tip, yt - r), (tip, yt), (xr, yt), (base, yt), (base, yb)]
    codes = [p.MOVETO, p.LINETO, p.CURVE3, p.CURVE3, p.LINETO, p.CURVE3, p.CURVE3, p.LINETO, p.CLOSEPOLY]
    return mpath.Path(verts, codes)


def _plot_metric_summary(ax, results: dict, res_key: str, sizes: Sizes, bar_frac: float = 0.62) -> None:  # noqa: ANN001
    """Boxed sparkline bars attached to a curve panel: the metric averaged over all k, per agent.

    Everything is drawn in axes coordinates (the panel is near-square, so the rounded tips stay
    round): bars use the left 62% of the width, the value labels right-align in the rest, and a
    `mean over k` header with the better-direction arrow names the column. Bars grow from 0
    (sharp base, rounded tip), so negative sweep means extend leftwards from an inner zero line.
    The best value per metric is emphasised semibold in its agent's color; the others stay light.
    """
    rules = list(_LEGEND_ORDER)  # top-to-bottom like the legend, ours at the bottom
    vals = [float(np.mean(results[res_key][r])) for r in rules]
    lo, hi = min(0.0, *vals), max(0.0, *vals)
    span = (hi - lo) or 1.0
    best = max(vals) if _BETTER[res_key] == "↑" else min(vals)
    left_pad = 0.06 if lo < 0 else 0.0  # keep leftward (negative) bars off the shared border
    x0 = left_pad + (0.0 - lo) / span * bar_frac  # axes-fraction position of the zero line
    bar_h = 0.15
    ax.text(
        0.96,
        0.955,
        f"ø {_SUMMARY_TITLE[res_key]} {_BETTER[res_key]}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontproperties=FP_SEMIBOLD,
        fontsize=sizes.tick,
        color="0.35",
    )
    if lo < 0:  # inner zero line, so the leftward (negative) bars read correctly
        ax.plot(
            [x0, x0],
            [0.065, 0.755],
            transform=ax.transAxes,
            color="0.85",
            lw=0.5 * sizes.scale,
            zorder=1,
            clip_on=False,
        )
    for i, (rule, v) in enumerate(zip(rules, vals, strict=True)):
        y = 0.635 - i * 0.225  # three lanes, clear of the header above and the bottom border below
        wv = v / span * bar_frac
        is_best = v == best
        if abs(wv) > 0.012:  # a zero bar (our crash rate) is just its label
            ax.add_patch(
                PathPatch(
                    _tip_rounded_bar(x0, x0 + wv, y, bar_h),
                    transform=ax.transAxes,
                    linewidth=0,
                    facecolor=_COLORS[rule],
                    zorder=3,
                )
            )
        ax.text(
            0.96,
            y,
            _SUMMARY_FMT[res_key].format(v).replace("-", "\N{MINUS SIGN}"),  # as the tick labels set it
            transform=ax.transAxes,
            va="center",
            ha="right",
            fontproperties=FP_SEMIBOLD if is_best else FP_LIGHT,
            fontsize=sizes.tick,
            color=_COLORS[rule] if is_best else "0.25",
        )
    ax.set_xticks([])
    ax.set_yticks([])
    style_axes(ax, sizes)  # the same hairline box as the curve panel it is attached to


def _legend_handles(sizes: Sizes) -> list:
    """One legend handle per agent in `_LEGEND_ORDER`: the lines as the panels draw them."""
    return [plt.Line2D([0], [0], **line_style(_METHOD[rule], sizes)) for rule in _LEGEND_ORDER]


_HANDLE_LENGTH = 1.8  # legend handle length, in legend-font sizes


def _mathtext_rise(text: Text) -> float:
    """Points by which the pdf sets a mathtext label above its baseline; 0 for a plain label.

    A vector backend places the glyphs of a mathtext string from the top of its box, whose height
    it has rounded up to a whole point, so the string rides up to a point high beside the plain
    labels of its row -- how much depends on the type size. The string's first glyph stands on the
    baseline and carries exactly that offset. (The png rounds to a pixel instead, so there the
    corrected label can end up a fraction of a pixel low.)
    """
    if "$" not in text.get_text():
        return 0.0
    parsed = MathTextParser("path").parse(text.get_text(), 72, text.get_fontproperties())
    return float(parsed.glyphs[0][4])


def _build(results: dict, layout: str, sizes: Sizes) -> Figure:  # noqa: PLR0915
    """Compact figure with the swept metrics and route snapshots (`layout`: see `_LAYOUTS`).

    Part A: three metric panels stacked on a shared log x-axis (mean reward, CVaR reward, accident
    rate), each with its sweep-mean bars attached. Part B: one route snapshot per marked observation
    count with a note on what it shows. One shared legend at the bottom; dashed verticals in the
    metric panels mark the observation counts of the snapshots. All type and strokes come from
    `sizes` (see `_plot`); the geometry is the layout's and does not depend on them.
    """
    use_fira_mathtext()
    k_values = results["k_values"]
    panels = [
        ("mean_reward", "mean reward"),
        ("cvar_reward", "CVaR reward"),
        ("crash_rate", "crash rate"),
    ]
    clips = {"crash_rate": (0.0, 1.0)}
    style = _STYLE[results.get("setting", "default")]  # old pickles predate the setting key
    lay = _LAYOUTS[layout]
    snap_rows = style["snap_rows"]
    vline_ks = [label_k for _, label_k, *_ in snap_rows]
    # wider and ~20% shorter than the original 7.2x2.8: the curve panels and the (aspect-locked)
    # map snapshots squeeze vertically, and the freed width goes to the note boxes
    fig = plt.figure(figsize=lay["figsize"])
    # rather than the original 1.45/1.00: the right block takes ~10% more width so the maps can grow
    # (see gs_right) WITHOUT eating into the note column, whose text wrap is fixed
    # top leaves the strip the part headings need (they may poke out of the canvas; the figure is
    # saved by its tight bounding box). bottom stays put -- the x-axis label and the legend live
    # under it.
    outer = GridSpec(1, 2, figure=fig, **lay["outer"])
    # each metric row: the curve panel with its sweep-mean bar panel directly attached
    gs_left = outer[0].subgridspec(len(panels), 2, width_ratios=lay["left_ratios"], wspace=0.0, hspace=0.12)
    # each snapshot row: the map with its note box. The maps are aspect-locked and HEIGHT-limited
    # (they shrink horizontally to fit their row), so the ROW HEIGHT is what sizes them -- widening
    # the map column alone does nothing. This is its own GridSpec rather than a subgridspec of
    # outer[1] so the rows can claim the strip above the metric panels and squeeze the gaps between
    # them: top/bottom/hspace here are the map-size knobs.
    right_box = outer[1].get_position(fig)
    # hspace: equal gaps between the three rows, the only vertical slack left. bottom: the bottom row
    # (map AND note) sits below the metric panels' floor.
    n_right_rows = len(snap_rows) + (lay["ours_banner"] is not None)  # the banner takes a row of its own
    gs_right = GridSpec(n_right_rows, 2, figure=fig, left=right_box.x0, right=right_box.x1, **lay["right"])
    metric_axes: list = []
    summary_axes: list = []
    for i, (res_key, ylabel) in enumerate(panels):
        ax = fig.add_subplot(gs_left[i, 0], sharex=metric_axes[0] if metric_axes else None)
        _plot_metric(
            ax,
            k_values,
            results[res_key],
            results[f"{res_key}_std"],
            ylabel,
            vline_ks,
            sizes,
            show_x=(i == len(panels) - 1),
            clip=clips.get(res_key),
            yticks=style["yticks"].get(res_key),
            ycenter=style["ycenters"].get(res_key),
            xticks=style["xticks"],
            band=(res_key != "cvar_reward"),  # the pooled population CVaR is a single statistic
        )
        metric_axes.append(ax)
        sax = fig.add_subplot(gs_left[i, 1])
        _plot_metric_summary(sax, results, res_key, sizes, style.get("summary_bar_frac", 0.62))
        summary_axes.append(sax)
    align_ylabels(fig, metric_axes)  # varying tick widths (minus signs) must not shift the labels
    snap_axes: list = []
    note_axes: list = []
    for i, (data_k, label_k, note_alpha, note_title, note_body) in enumerate(snap_rows):
        max_ = fig.add_subplot(gs_right[i, 0])
        # behaviour classes + representative routes are derived here, at plot time, from the raw
        # per-seed routes in the results
        classes = {r: _route_classes(routes, results["road"]) for r, routes in results["snapshots"][data_k].items()}
        _plot_snapshot(max_, results["road"], classes, label_k, sizes)
        snap_axes.append(max_)
        nax = fig.add_subplot(gs_right[i, 1])
        if lay["note_bodies"] is not None:
            note_body = lay["note_bodies"][note_title]
        _draw_note(nax, note_title, note_body, note_alpha, sizes, **lay["note"])
        note_axes.append(nax)
    if lay["ours_banner"] is not None:
        _draw_ours_banner(fig.add_subplot(gs_right[-1, :]), lay["ours_banner"], sizes)

    handles = _legend_handles(sizes)
    labels = [_LABELS[r] for r in _LEGEND_ORDER]
    x_div = (max(a.get_position().x1 for a in summary_axes) + min(a.get_position().x0 for a in snap_axes)) / 2
    # one row sharing the x-label's line, right-aligned with the snapshot block: it costs the figure
    # no line of its own, which is what the paper's page budget cares about
    fig.canvas.draw()  # realise the x-label extent the legend row is centred on
    to_fig = fig.transFigure.inverted()
    xlabel_bb = metric_axes[-1].xaxis.label.get_window_extent()
    xlabel_mid = to_fig.transform((0, (xlabel_bb.y0 + xlabel_bb.y1) / 2))[1]
    # the x-label has no descenders, so its ink centre reads a touch high against the legend row;
    # nudge the whole row down onto the optical line
    xlabel_mid -= 0.115 * sizes.label / 72 / fig.get_figheight()
    note_right = max(a.get_position().x1 for a in note_axes)

    legend_kw = {"columnspacing": 1.4, "handlelength": _HANDLE_LENGTH, "handletextpad": 0.6} | lay["legend"]
    leg = fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(note_right, xlabel_mid),
        ncol=len(labels),
        frameon=False,
        prop=font(FP_REGULAR, sizes.legend),
        handleheight=1.2,
        **legend_kw,
    )
    for text in leg.get_texts():
        if text.get_text().endswith(OURS_TAG):  # ours is semibold as a whole
            text.set_fontproperties(font(FP_SEMIBOLD, sizes.legend))
    # Put every entry on the baseline of the first one (the mathtext label is deeper than the plain
    # ones, and each entry is a column of its own).
    fig.canvas.draw()
    texts, handle_boxes = leg.get_texts(), leg.findobj(DrawingArea)
    baselines = [text.get_transform().transform(text.get_position())[1] for text in texts]
    # the labels' ink centre sits about a third of the font size above their baseline, and matplotlib
    # baseline-aligns the handles a touch above that -- drop them onto the ink centre
    handle_drop = -0.06 * sizes.legend / 72
    for i, (text, box) in enumerate(zip(texts, handle_boxes, strict=True)):
        lift = (baselines[0] - baselines[i]) / fig.dpi
        text_shift = mtransforms.ScaledTranslation(0, lift - _mathtext_rise(text) / 72, fig.dpi_scale_trans)
        text.set_transform(text.get_transform() + text_shift)
        handle_shift = mtransforms.ScaledTranslation(0, lift + handle_drop, fig.dpi_scale_trans)
        for artist in box.get_children():
            artist.set_transform(artist.get_transform() + handle_shift)

    # unrotated math-font direction arrows sitting NEXT TO the vertical y-labels
    fig.canvas.draw()  # realise the aligned y-label positions first
    arrows = {"mean_reward": "$\\uparrow$", "cvar_reward": "$\\uparrow$", "crash_rate": "$\\downarrow$"}
    for ax, (res_key, _) in zip(metric_axes, panels, strict=True):
        inv = ax.transAxes.inverted()
        x_label_right = inv.transform((ax.yaxis.label.get_window_extent().x1, 0))[0]
        ax.text(
            x_label_right + 0.001,  # hugging the label, inside the labelpad gap before the ticks
            0.5,
            arrows[res_key],
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=sizes.label,
            color="black",
            # DejaVu's thin math arrow, not Fira's heavy one: the glyph the other figures' direction
            # arrows end up with too (under paper_style's mathtext settings their "cm" falls back to it)
            math_fontfamily="dejavusans",
        )

    # A | B split: a light divider between the bar column and the snapshots, as tall as what stands
    # on either side of it.
    snap_left = min(a.get_position().x0 for a in snap_axes)
    top_y = max(metric_axes[0].get_position().y1, snap_axes[0].get_position().y1)
    bottom_y = metric_axes[-1].get_position().y0
    divider_top = top_y
    if lay["titles"] is not None:
        # Part headings: A starts flush with the y labels, the left edge of the figure; B with the
        # snapshots.
        label_left = to_fig.transform((min(ax.yaxis.label.get_window_extent().x0 for ax in metric_axes), 0))[0]
        title_dy = lay["title_dy"] * sizes.label / 72 / fig.get_figheight()
        divider_top = top_y + 2 * title_dy
        for x, title in zip((label_left, snap_left), lay["titles"], strict=True):
            fig.text(
                x, top_y + title_dy, title, fontproperties=FP_SEMIBOLD, fontsize=sizes.label, ha="left", va="bottom"
            )
    fig.add_artist(
        plt.Line2D(
            [x_div, x_div], [bottom_y, divider_top], transform=fig.transFigure, color="0.8", lw=0.6 * sizes.scale
        )
    )
    return fig


def _saved_width(fig: Figure, pad: float) -> float:
    """Width in inches of the figure as savefig(bbox_inches="tight", pad_inches=pad) writes it."""
    fig.canvas.draw()
    return fig.get_tightbbox(fig.canvas.get_renderer()).width + 2 * pad  # ty: ignore[unresolved-attribute]


def _render_for_print(build: Callable[[Sizes], Figure], print_width: float, pad: float) -> tuple[Figure, Sizes]:
    """paper_style.render_for_print for a figure saved with `pad` inches around its tight bounding box.

    The shared function measures the saved width with matplotlib's default padding of 0.1in, which
    the one-column layout cannot afford (see `_LAYOUTS`); this is the same measure-and-rebuild loop
    with the padding the layout is saved with. It also runs until the scale has settled instead of
    for two rounds: the y labels are the figure's left edge here, so the saved width follows the
    type more closely than in a figure with a fixed canvas, and two rounds leave the type 2% large.
    """
    sizes = Sizes(1.0)
    for _ in range(_MAX_SIZE_PASSES):
        fig = build(sizes)
        settled = Sizes(_saved_width(fig, pad) / print_width / REFERENCE_ZOOM)
        plt.close(fig)
        done = abs(settled.scale / sizes.scale - 1.0) < _SIZE_TOLERANCE
        sizes = settled
        if done:
            break
    return build(sizes), sizes


_MAX_SIZE_PASSES = 8
_SIZE_TOLERANCE = 5e-4  # relative change of the scale between two rounds: under 0.005pt on a label


def _plot(results: dict, path: str, layout: str = "wide") -> None:
    """Draw the figure in a layout and write it as pdf and png (see `_build`).

    The type is sized for the width the paper includes the layout at: its axis labels print at the
    size of the reference figure's (paper_style), which the log line reports.
    """
    lay = _LAYOUTS[layout]
    print_width, pad = lay["print_width"], lay["pad_inches"]
    fig, sizes = _render_for_print(partial(_build, results, layout), print_width, pad)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(out.with_suffix(ext), dpi=200, bbox_inches="tight", pad_inches=pad)
    width = _saved_width(fig, pad)
    logger.info(
        "Saved %s and .png (%.2f in wide, labels print at %.2f pt)",
        out.with_suffix(".pdf"),
        width,
        sizes.label * print_width / width,
    )
    plt.close(fig)


def main() -> None:
    """Load config, run the sweep (or re-plot the pickled results), and print per-agent crash rates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", default="default", help="which settings block of the config to run")
    parser.add_argument("--plot-only", action="store_true", help="re-plot the pickled results of the last full run")
    parser.add_argument(
        "--layout",
        default="wide",
        choices=list(_LAYOUTS),
        help="figure geometry for --plot-only; non-default layouts are saved under a _<layout> suffix",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = OmegaConf.load(Path(__file__).resolve().parents[2] / "configs" / "credal_driving.yaml")
    if args.setting not in cfg.settings:
        raise SystemExit(f"unknown setting {args.setting!r}; choose from {list(cfg.settings)}")
    plot_path = cfg.settings[args.setting].plot_path
    if args.plot_only:
        with _results_cache(plot_path).open("rb") as f:
            results = pickle.load(f)  # noqa: S301 -- our own cache, written by run()
        out = Path(plot_path)
        if args.layout != "wide":  # never overwrite the published wide figure
            out = out.with_name(f"{out.stem}_{args.layout}{out.suffix}")
        _plot(results, str(out), args.layout)
        return
    if args.layout != "wide":
        raise SystemExit("--layout only applies to --plot-only (a full run always plots the wide figure)")
    res = run(cfg, args.setting)
    print("hazard_obs:", res["k_values"])
    for res_key, _ in _METRICS.values():
        print(f"--- {res_key} ---")
        for rule in ORDER:
            print(f"  {rule:16s} {[round(v, 2) for v in res[res_key][rule]]}")


if __name__ == "__main__":
    main()
