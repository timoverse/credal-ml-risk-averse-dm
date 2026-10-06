"""Agents = decision rules reducing per-cell evidence to a scalar crash belief.

Three offline-RL decision rules, differing only in how they treat the under-observed hazard:

- `mle`: the point estimate p_mle. Unseen cells -> 0 (optimistic). Cannot react to a hazard it has
  not sampled.
- `aleatoric_cvar`: the paper's risk-averse agent -- CVaR of a Bernoulli loss under the point
  estimate, min(1, p_mle/beta). Still blind to the unseen: CVaR of a zero point estimate is zero,
  so aleatoric risk-aversion gives no protection against danger that was never observed.
- `credal_minimax`: Gamma-maximin on the Bernoulli relative-likelihood interval, i.e. the upper
  bound p_high. Pessimistic about the unseen ([0,1] -> 1) but calibrated and threshold-free: the
  interval shrinks smoothly with data (1 sample -> [0,0.85], 5 -> [0,0.4], ...).
"""

from __future__ import annotations

import numpy as np

AGENTS: tuple[str, ...] = ("mle", "aleatoric_cvar", "credal_minimax")


def crash_belief(
    rule: str,
    grids: dict[str, np.ndarray],
    *,
    beta: float,
) -> np.ndarray:
    """Per-cell crash belief grid for the given decision rule.

    Args:
        rule: One of AGENTS.
        grids: dict with the p_mle and p_high grids (from CountModel.predict_cell_grid).
        beta: CVaR tail level for aleatoric_cvar.

    Returns:
        A (H, W) crash-belief grid.

    Raises:
        ValueError: rule is not in AGENTS.
    """
    if rule == "mle":
        return grids["p_mle"]
    if rule == "aleatoric_cvar":
        return np.minimum(1.0, grids["p_mle"] / beta)
    if rule == "credal_minimax":
        return grids["p_high"]
    raise ValueError(f"Unknown agent {rule!r}; choose from {AGENTS}.")
