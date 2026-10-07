"""Cost-sensitive CVaR-minimax agent: robust (worst-case-credal) CVaR of the RETURN, planned per cell.

The loss is the trajectory COST (the negative return), so the crash penalty drives the decision --
exactly like the cost the planner uses for the other agents. The CVaR is taken over the return
distribution of a single drive (aleatoric crash randomness along the route), NOT over seeds.

The agent minimises the Rockafellar-Uryasev form under the worst case of the per-cell credal
intervals, jointly over the policy and the VaR threshold v:

    min_policy min_v [ v + (1/beta) sup_q E_q[(C - v)+] ],   q_cell in [p_low_cell, p_high_cell].

For a fixed v the inner problem is solved by BUDGET-AUGMENTED value iteration (Bauerle-Ott): the
state is augmented with the accrued cost, which makes the truncated cost (C - v)+ dynamic-programmable.
The outer minimisation scans v over the attainable costs. Everything is computed from the agent's own
beliefs (the offline counts) -- the true crash probabilities never enter the plan. The MLE agent is the
same planner on the degenerate intervals [p_mle, p_mle].

Exchanging min_v and sup_q makes the objective an upper bound on the worst-case CVaR sup_q CVaR_q(C).
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from driving.planner import neighbor_maps

if TYPE_CHECKING:
    from driving.gridworld import RoadMap

Route = list

# CVaR is tail-only: it is indifferent to outcomes below the VaR threshold v, so routes that differ
# only below v tie in the truncated cost (C - v)+ and argmin would pick arbitrarily among them. An
# infinitesimal mean-cost tiebreak eps*E[C] breaks such ties toward the cheaper route without
# changing the tail-driven decision.
_MEAN_TIEBREAK = 1e-6
# thresholds planned per vectorized pass; bounds the memory of the (s, v, H, W) policy table
_V_CHUNK = 16


def _policies(
    road: RoadMap,
    p_low: np.ndarray,
    p_high: np.ndarray,
    vs: np.ndarray,
    *,
    step_cost: float,
    goal_reward: float,
    crash_penalty: float,
    s_max: int,
) -> np.ndarray:
    """Budget-augmented value iteration for the worst-case truncated cost, for each threshold in `vs`.

    State is (cell, s) where s = number of steps taken so far (accrued cost = s * step_cost); s
    strictly increases per move so the augmented state is a DAG in s and one backward pass suffices.
    The cost convention matches the evaluator: every step costs step_cost; a crash on entry to a cell
    costs crash_penalty but not that step's step_cost; reaching the goal adds goal_reward.

    The backup is linear in the crash probability of the entered cell, so its worst case over the
    cell's interval sits at an endpoint: p_high where crashing now costs more than continuing, p_low
    where a later (costlier) crash is the greater threat.

    Returns policy[s, i, r, c] = greedy action at (cell, s) for threshold vs[i]. beta does not enter --
    for fixed v the inner problem is just min sup E[(C - v)+].
    """
    h, w = road.grid.shape
    nr, nc, is_goal = neighbor_maps(road)
    lo, hi = p_low[nr, nc], p_high[nr, nc]  # (A, H, W) interval of the neighbour reached by each action
    v = vs[:, None, None, None]
    # routes longer than the cap keep value `large`, i.e. are forbidden
    large = 1e12

    value = np.full((len(vs), h, w), large, dtype=np.float64)
    policy = np.zeros((s_max, len(vs), h, w), dtype=np.int8)
    for s in range(s_max - 1, -1, -1):
        goal_c = (s + 1) * step_cost - goal_reward  # total cost of arriving at the goal here
        crash_c = s * step_cost + crash_penalty  # total cost of crashing on entry to the neighbour
        goal_term = np.maximum(goal_c - v, 0.0) + _MEAN_TIEBREAK * goal_c  # arrive at goal (deterministic)
        crash_term = np.maximum(crash_c - v, 0.0) + _MEAN_TIEBREAK * crash_c
        cont = value[:, nr, nc]  # (V, A, H, W) survive-and-continue value at the neighbour
        excess = crash_term - cont
        q = np.where(is_goal, goal_term, cont + np.where(excess > 0.0, hi, lo) * excess)
        value = q.min(axis=1)
        policy[s] = q.argmin(axis=1)
    return policy


def _trace(road: RoadMap, policy: np.ndarray) -> Route:
    """The route a budget-augmented policy[s, r, c] drives (absent crashes) from the start."""
    route = [road.start]
    cell = road.start
    for s in range(policy.shape[0]):
        if cell == road.goal:
            break
        nxt = road.neighbor(cell, int(policy[s][cell]))
        if nxt == cell:  # stuck against a wall: no progress
            break
        route.append(nxt)
        cell = nxt
    return route


def _worst_truncated_cost(
    road: RoadMap,
    route: Route,
    p_low: np.ndarray,
    p_high: np.ndarray,
    v: float,
    *,
    step_cost: float,
    goal_reward: float,
    crash_penalty: float,
) -> float:
    """Exact sup over the per-cell intervals of E[(C - v)+] along `route` (first crash ends it)."""
    if route[-1] != road.goal:
        return float("inf")
    n_steps = len(route) - 1
    worst = max(n_steps * step_cost - goal_reward - v, 0.0)
    for t in range(n_steps - 1, 0, -1):  # route[t] is entered on step t, after t - 1 paid steps
        excess = max((t - 1) * step_cost + crash_penalty - v, 0.0) - worst
        worst += (p_high[route[t]] if excess > 0.0 else p_low[route[t]]) * excess
    return worst


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


def cvar_route(
    road: RoadMap,
    p_low: np.ndarray,
    p_high: np.ndarray,
    *,
    beta: float,
    step_cost: float,
    goal_reward: float,
    crash_penalty: float,
    s_max: int | None = None,
    v_step: float = 1.0,
) -> Route:
    """The route minimising min_v [ v + (1/beta) sup E[(C - v)+] ] over the per-cell intervals.

    The threshold v is scanned upwards from the smallest attainable cost in steps of `v_step`. The
    objective at any v is at least v, so the scan stops once v exceeds the best objective found: no
    larger threshold can improve on it. With integer costs and v_step=1 the grid holds every
    attainable cost and the scan is exact; a coarser grid is within v_step of the optimum (the
    objective of a fixed route rises with slope at most 1 above its minimiser).

    Args:
        road: The road providing the geometry.
        p_low: (H, W) lower crash probabilities (equal to p_high for a point belief).
        p_high: (H, W) upper crash probabilities.
        beta: CVaR tail level.
        step_cost: Cost of one step.
        goal_reward: Reward for reaching the goal.
        crash_penalty: Cost of a crash.
        s_max: Route-length cap; None = number of free cells (a non-looping route visits each once).
        v_step: Grid spacing of the threshold scan.

    Returns:
        The planned route from start to goal.

    Raises:
        ValueError: no route reaches the goal within s_max steps.
    """
    rewards = {"step_cost": step_cost, "goal_reward": goal_reward, "crash_penalty": crash_penalty}
    if s_max is None:
        s_max = len(road.free_cells())
    v_min = min((len(_shortest_path(road)) - 1) * step_cost - goal_reward, crash_penalty)
    v_top = (s_max - 1) * step_cost + crash_penalty  # the largest cost any admissible route can incur
    grid = np.arange(v_min, v_top + v_step, v_step)
    best_route, best_obj = None, float("inf")
    for start in range(0, len(grid), _V_CHUNK):
        vs = grid[start : start + _V_CHUNK]
        if vs[0] > best_obj:
            break
        policy = _policies(road, p_low, p_high, vs, s_max=s_max, **rewards)
        for i, v in enumerate(vs):
            route = _trace(road, policy[:, i])
            obj = float(v) + _worst_truncated_cost(road, route, p_low, p_high, float(v), **rewards) / beta
            if obj < best_obj - 1e-9:
                best_route, best_obj = route, obj
    if best_route is None:
        raise ValueError(f"no route reaches the goal within s_max={s_max} steps")
    return best_route
