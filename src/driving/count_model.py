"""Per-cell count-based credal danger model: Bernoulli relative-likelihood alpha-cut.

Each cell accrues (n_safe, n_crash) from interactive experience. The credal interval on the
crash probability is the relative-likelihood region {p : L(p) - L(p_mle) >= log(alpha)}, which is
data-local: an unvisited cell is [0, 1] and the interval shrinks as safe observations accrue.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import brentq

from driving.gridworld import WALL

if TYPE_CHECKING:
    from driving.gridworld import RoadMap


def _loglik(n_safe: int, n_crash: int, p: float) -> float:
    """Bernoulli log-likelihood n_crash*log(p) + n_safe*log(1-p), guarding the zero-count terms."""
    val = 0.0
    if n_crash > 0:
        val += n_crash * np.log(p)
    if n_safe > 0:
        val += n_safe * np.log1p(-p)
    return val


def credal_interval(n_safe: int, n_crash: int, alpha: float) -> tuple[float, float]:
    """Relative-likelihood alpha-cut interval [p_low, p_high] for crash probability from counts.

    Edge cases are closed-form; the interior case roots L(p) - L(p_mle) = log(alpha) on each side
    of the MLE. The objective is concave with a single peak at p_mle, so exactly one root lies in
    each of (0, p_mle) and (p_mle, 1).
    """
    n = n_safe + n_crash
    if n == 0:
        return 0.0, 1.0  # no information: maximal uncertainty
    if n_crash == 0:
        return 0.0, 1.0 - alpha ** (1.0 / n_safe)  # all safe
    if n_safe == 0:
        return alpha ** (1.0 / n_crash), 1.0  # all crash
    p_mle = n_crash / n
    target = _loglik(n_safe, n_crash, p_mle) + np.log(alpha)

    def g(p: float) -> float:
        return _loglik(n_safe, n_crash, p) - target

    lo = brentq(g, 1e-12, p_mle)
    hi = brentq(g, p_mle, 1.0 - 1e-12)
    return float(lo), float(hi)


class CountModel:
    """Per-cell crash/safe counts -> per-cell credal interval grids."""

    def __init__(self, road: RoadMap, alpha: float) -> None:
        """Start every cell at zero counts (so every interval is [0, 1])."""
        self.alpha = alpha
        self.n_safe = np.zeros(road.grid.shape, dtype=np.int64)
        self.n_crash = np.zeros(road.grid.shape, dtype=np.int64)

    def observe(self, cell: tuple[int, int], crashed: bool) -> None:
        """Record one entry into `cell` as a crash or a safe passage."""
        if crashed:
            self.n_crash[cell] += 1
        else:
            self.n_safe[cell] += 1

    def predict_cell_grid(self, road: RoadMap) -> dict[str, np.ndarray]:
        """Return the p_mle and p_high grids (H, W); 0 on wall cells (never entered).

        p_mle for an unvisited cell is 0 (an optimistic 'safe until seen otherwise' assumption, not a
        true MLE); p_high is the upper edge of the relative-likelihood credal interval ([0, 1] when
        unvisited). These are the two beliefs the agents plan with (point estimate vs. credal bound).

        The interval solve runs once per UNIQUE (n_safe, n_crash) pair, not per cell: all well-cased
        safe cells share one pair and the few hazard cells realize a handful more.
        """
        free = road.grid != WALL
        n = self.n_safe + self.n_crash
        p_mle = np.zeros(road.grid.shape, dtype=np.float64)
        np.divide(self.n_crash, n, out=p_mle, where=(n > 0) & free)
        p_high = np.zeros(road.grid.shape, dtype=np.float64)
        pairs = np.unique(np.stack([self.n_safe[free], self.n_crash[free]], axis=1), axis=0)
        for ns, nc in pairs:
            _, hi = credal_interval(int(ns), int(nc), self.alpha)
            p_high[free & (self.n_safe == ns) & (self.n_crash == nc)] = hi
        return {"p_mle": p_mle, "p_high": p_high}
