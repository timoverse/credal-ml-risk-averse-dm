"""Value iteration over the hazardous road for a given per-cell crash belief.

The agent's belief `crash_prob[cell]` is the probability it assigns to crashing when
entering `cell`. Entering the goal ends the episode with `goal_reward`; entering a cell
with belief p yields a terminal crash (reward -crash_penalty) with probability p, else a
normal step (reward -step_cost). Agents differ only in the belief grid they pass in.

With `gamma=1.0` this is a proper stochastic-shortest-path problem (terminal goal/crash,
positive step cost, so every non-terminating policy strictly loses value) and the sweep
converges; big worlds use gamma=1 so the shortcut-vs-detour comparison is undistorted.
"""

from __future__ import annotations

import numpy as np

from driving.gridworld import WALL, RoadMap


def neighbor_maps(road: RoadMap) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-action neighbour row/col maps and a goal mask, (A, H, W); cached on the road.

    Every free cell's neighbour is itself free (a wall bump returns the same cell), so these indices
    always land on valid free cells. Wall cells are left at 0 and never read (no free cell steps onto
    a wall).
    """
    cached = getattr(road, "_neighbor_maps", None)
    if cached is not None:
        return cached
    h, w = road.grid.shape
    n_actions = len(road.actions)
    nr = np.zeros((n_actions, h, w), dtype=np.intp)
    nc = np.zeros((n_actions, h, w), dtype=np.intp)
    for a in range(n_actions):
        for cell in road.free_cells():
            nxt = road.neighbor(cell, a)
            nr[a][cell] = nxt[0]
            nc[a][cell] = nxt[1]
    is_goal = (nr == road.goal[0]) & (nc == road.goal[1])
    road._neighbor_maps = (nr, nc, is_goal)  # noqa: SLF001 -- our own per-road cache slot
    return nr, nc, is_goal


def value_iteration(
    road: RoadMap,
    crash_prob: np.ndarray,
    *,
    step_cost: float,
    goal_reward: float,
    crash_penalty: float,
    gamma: float = 0.99,
    tol: float = 1e-6,
    max_iters: int = 10_000,
) -> tuple[dict[tuple[int, int], int], np.ndarray]:
    """Return (policy, value): greedy action per non-goal free cell and the state-value grid.

    Vectorized Bellman backups over the (A, H, W) neighbour maps; wall cells are pinned to 0 and
    never read (free cells only ever step onto free cells).
    """
    nr, nc, is_goal = neighbor_maps(road)
    p_nbr = crash_prob[nr, nc]  # (A, H, W) belief crash prob of the neighbour reached by each action
    free = road.grid != WALL
    value = np.zeros(road.grid.shape, dtype=np.float64)

    def backup(v: np.ndarray) -> np.ndarray:
        cont = -step_cost + gamma * v[nr, nc]
        return np.where(is_goal, goal_reward, p_nbr * (-crash_penalty) + (1.0 - p_nbr) * cont)

    for _ in range(max_iters):
        new_value = np.where(free, backup(value).max(axis=0), 0.0)
        new_value[road.goal] = 0.0  # absorbing goal: the reward accrues on entry, not at the goal
        delta = float(np.abs(new_value - value).max())
        value = new_value
        if delta < tol:
            break

    q = backup(value)
    policy = {c: int(q[(slice(None), *c)].argmax()) for c in road.free_cells() if c != road.goal}
    return policy, value


def intended_route(road: RoadMap, policy: dict[tuple[int, int], int]) -> list[tuple[int, int]]:
    """The believed-best path from start to goal, following the greedy policy.

    Ignores stochastic crashes (those are sampled only at evaluation). Stops at the goal,
    on a self-loop, or after height*width steps as a safety bound.
    """
    route = [road.start]
    cell = road.start
    for _ in range(road.height * road.width):
        if cell == road.goal:
            break
        nxt = road.neighbor(cell, policy[cell])
        if nxt == cell:  # stuck against a wall: no progress
            break
        route.append(nxt)
        cell = nxt
    return route
