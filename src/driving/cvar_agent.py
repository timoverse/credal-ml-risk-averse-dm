"""Cost-sensitive CVaR-minimax agent: robust (worst-case-credal) CVaR of the RETURN, planned per cell.

The loss is the trajectory COST (the negative return), so the crash penalty drives the decision --
exactly like the cost the planner uses for the other agents. The CVaR is taken over the return
distribution of a single drive (aleatoric crash randomness along the route), NOT over seeds.

The policy is found by BUDGET-AUGMENTED value iteration (Bauerle-Ott): the Rockafellar-Uryasev dual
CVaR_beta(C) = min_v [ v + (1/beta) E[(C - v)+] ] is dynamic-programmable once the state is augmented
with the accrued cost. For a fixed VaR threshold v the augmented value iteration minimises
E_belief[(C - v)+] under the belief crash probabilities (p_high for the credal Gamma-maximin agent,
p_mle for the single-distribution MLE agent). The threshold v is calibrated once on a HOLD-OUT set
(extra seeds, disjoint from the test seeds) to minimise the realised population CVaR of the return
under the TRUE dynamics -- the reward-distribution analogue of decision_rules.calibrate_var_threshold.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from driving.planner import neighbor_maps

if TYPE_CHECKING:
    from driving.gridworld import RoadMap

Route = list

# CVaR is tail-only: it is indifferent to outcomes below the VaR threshold v. With a large goal_reward
# every survive-to-goal cost sits far below v, so the truncated cost (C - v)+ is 0 for all crash-free
# routes -- the detour and a pointless 60-step ramble tie at 0 and argmin picks arbitrarily. An
# infinitesimal mean-cost tiebreak eps*E[C] breaks the tie toward efficient goal-reaching without
# changing the tail-driven shortcut-vs-detour decision (eps*dC is negligible against the 1/beta-scaled
# tail). It is consistent with true CVaR minimisation, which prefers the detour for its lower cost.
_MEAN_TIEBREAK = 1e-3


def cvar_value_iteration(
    road: RoadMap,
    crash_prob: np.ndarray,
    v: float,
    *,
    step_cost: float,
    goal_reward: float,
    crash_penalty: float,
    s_max: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Budget-augmented value iteration for the truncated cost E[(C - v)+] at threshold v.

    State is (cell, s) where s = number of steps taken so far (accrued cost = s * step_cost); s
    strictly increases per move so the augmented state is a DAG in s and one backward pass suffices.
    The cost convention matches the evaluator: every step costs step_cost; a crash on entry to a cell
    costs crash_penalty but not that step's step_cost; reaching the goal adds goal_reward.

    Returns (value, policy): value[s, r, c] = min expected (C - v)+ from (cell, s); policy[s, r, c] =
    greedy action there. beta does not enter -- for fixed v the inner problem is just min E[(C - v)+].
    """
    h, w = road.grid.shape
    nr, nc, is_goal = neighbor_maps(road)
    p_nbr = crash_prob[nr, nc]  # (A, H, W) belief crash prob of the neighbour reached by each action
    if s_max is None:
        s_max = len(road.free_cells())  # exact bound: a non-looping route visits each cell once
    # a smaller cap (config s_max, ~2.5x the detour) keeps the augmented state tractable on big
    # worlds; routes longer than the cap keep value `large`, i.e. are forbidden -- sound as long
    # as the cap comfortably exceeds every sensible route
    large = 1e12

    value = np.full((s_max + 1, h, w), large, dtype=np.float64)
    policy = np.zeros((s_max, h, w), dtype=np.intp)
    for s in range(s_max - 1, -1, -1):
        cost_s = s * step_cost
        goal_c = cost_s + step_cost - goal_reward  # total cost of arriving at the goal here
        crash_c = cost_s + crash_penalty  # total cost of crashing on entry to the neighbour
        goal_term = max(goal_c - v, 0.0) + _MEAN_TIEBREAK * goal_c  # arrive at goal (deterministic)
        crash_term = max(crash_c - v, 0.0) + _MEAN_TIEBREAK * crash_c
        cont = value[s + 1][nr, nc]  # (A, H, W) survive-and-continue value at the neighbour
        q = np.where(is_goal, goal_term, p_nbr * crash_term + (1.0 - p_nbr) * cont)
        value[s] = q.min(axis=0)
        policy[s] = q.argmin(axis=0)
    return value, policy


def cvar_route(
    road: RoadMap,
    crash_prob: np.ndarray,
    v: float,
    *,
    step_cost: float,
    goal_reward: float,
    crash_penalty: float,
    s_max: int | None = None,
) -> Route:
    """The route the budget-augmented CVaR policy drives (absent crashes) from start to goal at v."""
    _, policy = cvar_value_iteration(
        road, crash_prob, v, step_cost=step_cost, goal_reward=goal_reward, crash_penalty=crash_penalty, s_max=s_max
    )
    route = [road.start]
    cell, s = road.start, 0
    for _ in range(policy.shape[0]):
        if cell == road.goal:
            break
        nxt = road.neighbor(cell, int(policy[s][cell]))
        if nxt == cell:  # stuck against a wall: no progress
            break
        route.append(nxt)
        cell, s = nxt, s + 1
    return route


def _return_distribution(
    road: RoadMap,
    route: Route,
    crash_prob: np.ndarray,
    *,
    step_cost: float,
    goal_reward: float,
    crash_penalty: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact (returns, probabilities) of a route under per-cell crash beliefs (first crash ends it)."""
    returns, probs, survive = [], [], 1.0
    for t, cell in enumerate(route[1:], start=1):
        q = float(crash_prob[cell])
        probs.append(survive * q)
        returns.append(-(t - 1) * step_cost - crash_penalty)
        survive *= 1.0 - q
    returns.append(-(len(route) - 1) * step_cost + (goal_reward if route[-1] == road.goal else 0.0))
    probs.append(survive)
    return np.asarray(returns), np.asarray(probs)


def _shortest_path(road: RoadMap) -> Route:
    """Shortest start->goal route (BFS, hazards allowed) -- the most hazard-exposed route on the map."""
    prev: dict[tuple[int, int], tuple[int, int]] = {road.start: road.start}
    queue: deque[tuple[int, int]] = deque([road.start])
    while queue:
        cell = queue.popleft()
        if cell == road.goal:
            break
        for action in range(len(road.actions)):
            nxt = road.neighbor(cell, action)
            if nxt != cell and nxt not in prev:
                prev[nxt] = cell
                queue.append(nxt)
    route, cell = [], road.goal
    while True:
        route.append(cell)
        if cell == prev[cell]:
            break
        cell = prev[cell]
    return route[::-1]


def _v_max(road: RoadMap, rewards: dict) -> float:
    """Top of the v grid: the largest cost that can occur with positive probability.

    The VaR threshold v lives in the cost tail, so the grid must reach the worst realizable cost --
    a crash at the deepest hazard on the most-exposed (shortest) route under the true dynamics. Using
    crash_penalty alone undershoots this (it is the CHEAPEST crash, on the first step), capping v below
    the crash tail so it could never flip a route. Mirrors decision_rules.calibrate_var_threshold, which
    tops its grid at the largest realized loss.
    """
    ret, prob = _return_distribution(road, _shortest_path(road), road.p_true, **rewards)
    cost = -ret
    return float(cost[prob > 0.0].max())


def calibrate_v(
    holdout_grids: list[dict],
    road: RoadMap,
    *,
    beta: float,
    rewards: dict,
    num_grid: int,
    prob_key: str = "p_high",
    s_max: int | None = None,
) -> float:
    """Calibrate the VaR threshold v on the hold-out by minimising the realised population CVaR of return.

    For each grid value v, every hold-out seed's budget-augmented CVaR policy is planned under the
    belief `prob_key` (p_high = the worst-case credal probability; p_mle = the single-distribution
    MLE credal set) at that v; the realised return distribution of the route it drives under the TRUE
    crash probabilities is pooled, and F(v) = v + (1/beta) E_pop[(cost - v)+] is evaluated. The
    lowest-F(v) v is returned.
    """
    best_v, best_f = 0.0, float("inf")
    for v in np.linspace(0.0, _v_max(road, rewards), num_grid):
        excess, total = 0.0, 0.0
        for grids in holdout_grids:
            route = cvar_route(road, grids[prob_key], float(v), s_max=s_max, **rewards)
            ret, prob = _return_distribution(road, route, road.p_true, **rewards)  # realised, true dynamics
            excess += float((prob * np.clip(-ret - v, 0.0, None)).sum())
            total += 1.0
        f = float(v) + excess / (beta * total)
        if f < best_f:
            best_f, best_v = f, float(v)
    return best_v
