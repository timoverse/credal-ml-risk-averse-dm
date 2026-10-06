"""Evaluate an intended route under the true hazard: crash rate, mean return, CVaR of return.

evaluate_route also hands back the raw per-episode returns so the sweep can pool them across seeds
into one population return distribution (the paper's population CVaR couples all seeds through a
single VaR threshold, so it cannot be assembled from per-seed CVaR summaries).
"""

from __future__ import annotations

import numpy as np

from driving.gridworld import HAZARD, RoadMap


def cvar_of_returns(returns: np.ndarray, beta: float) -> float:
    """Mean of the worst beta-fraction (lowest returns), the left-tail risk measure.

    At least one return is always included so the measure is defined for any beta in (0, 1].
    """
    sorted_returns = np.sort(returns)
    k = max(1, int(np.ceil(beta * len(sorted_returns))))
    return float(sorted_returns[:k].mean())


def _rollout_once(
    road: RoadMap,
    route: list[tuple[int, int]],
    *,
    step_cost: float,
    goal_reward: float,
    crash_penalty: float,
    rng: np.random.Generator,
) -> tuple[float, bool]:
    """One episode along `route` under true crash probabilities; returns (return, crashed)."""
    total = 0.0
    for cell in route[1:]:  # route[0] is the start (already there)
        if road.grid[cell] == HAZARD and rng.random() < road.p_true[cell]:
            return total - crash_penalty, True
        total -= step_cost
    if route[-1] == road.goal:
        total += goal_reward
    return total, False


def evaluate_route(
    road: RoadMap,
    route: list[tuple[int, int]],
    *,
    n_episodes: int,
    step_cost: float,
    goal_reward: float,
    crash_penalty: float,
    cvar_beta: float,
    rng: np.random.Generator,
) -> dict[str, float | np.ndarray]:
    """Roll out `route` n_episodes times; return crash_rate, mean_return, cvar_return, raw returns.

    cvar_return is the CVaR within this rollout batch only (aleatoric, conditional on the route);
    the population CVaR over seeds must be computed by pooling the raw `returns` across seeds.
    """
    returns = np.empty(n_episodes, dtype=np.float64)
    crashes = 0
    for i in range(n_episodes):
        ret, crashed = _rollout_once(
            road,
            route,
            step_cost=step_cost,
            goal_reward=goal_reward,
            crash_penalty=crash_penalty,
            rng=rng,
        )
        returns[i] = ret
        crashes += int(crashed)
    return {
        "crash_rate": crashes / n_episodes,
        "mean_return": float(returns.mean()),
        "cvar_return": cvar_of_returns(returns, cvar_beta),
        "returns": returns,
    }
