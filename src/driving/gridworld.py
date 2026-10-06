"""The credal hazardous road: a deterministic grid MDP with under-sampled hazards.

Cells are FREE, WALL, or HAZARD. Entering a hazard cell triggers a crash with the
cell's true probability (handled by the planner/evaluator, not here). Geometry only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FREE, WALL, HAZARD = 0, 1, 2

# Action index -> (drow, dcol). 4-action worlds: up/down/left/right; 8-action worlds add diagonals.
ACTIONS_4: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
ACTIONS_8: tuple[tuple[int, int], ...] = (*ACTIONS_4, (-1, -1), (-1, 1), (1, -1), (1, 1))
ACTIONS = ACTIONS_4  # backward-compat name; the default world keeps the 4-action set

_DEFAULT_LAYOUT = [
    "#############",
    "#G..........#",
    "#...........#",
    "#..#######..#",
    "#HH#######..#",
    "#HH#######..#",
    "#HH#######..#",
    "#..#######..#",
    "#...........#",
    "#S..........#",
    "#############",
]


@dataclass
class RoadMap:
    """A grid road: cell types, start/goal, and per-cell true crash probabilities."""

    grid: np.ndarray  # (H, W) of {FREE, WALL, HAZARD}
    start: tuple[int, int]
    goal: tuple[int, int]
    p_true: np.ndarray  # (H, W) true crash prob; 0 except on hazard cells
    actions: tuple[tuple[int, int], ...] = ACTIONS_4  # the car's move set (4- or 8-directional)
    # planner cache slot for the (A, H, W) neighbour maps, filled by planner.neighbor_maps
    _neighbor_maps: tuple[np.ndarray, np.ndarray, np.ndarray] | None = field(default=None, repr=False)

    @property
    def height(self) -> int:
        """Number of rows."""
        return int(self.grid.shape[0])

    @property
    def width(self) -> int:
        """Number of columns."""
        return int(self.grid.shape[1])

    def in_bounds(self, r: int, c: int) -> bool:
        """Whether (r, c) lies inside the grid."""
        return 0 <= r < self.height and 0 <= c < self.width

    def is_wall(self, r: int, c: int) -> bool:
        """Whether (r, c) is out of bounds or a wall."""
        return not self.in_bounds(r, c) or self.grid[r, c] == WALL

    def neighbor(self, cell: tuple[int, int], action: int) -> tuple[int, int]:
        """Resulting cell after taking `action`; stays put if it would hit a wall/border.

        A diagonal move additionally requires BOTH orthogonal neighbours to be free -- the car
        cannot cut a corner past a wall.
        """
        dr, dc = self.actions[action]
        nr, nc = cell[0] + dr, cell[1] + dc
        if self.is_wall(nr, nc):
            return cell
        if dr != 0 and dc != 0 and (self.is_wall(cell[0] + dr, cell[1]) or self.is_wall(cell[0], cell[1] + dc)):
            return cell
        return (nr, nc)

    def free_cells(self) -> list[tuple[int, int]]:
        """All non-wall cells (FREE or HAZARD), row-major."""
        return [(r, c) for r in range(self.height) for c in range(self.width) if self.grid[r, c] != WALL]


def default_road(p_hazard: float = 0.25) -> RoadMap:
    """Build the mountain road: a central wall (the mountain) with a hazardous direct road.

    The short route runs straight up the left side of the mountain through a hazard stretch (e.g.
    rockfall / flooding); the safe detour drives around the right side of the mountain. The start is
    bottom-left, the destination top-left.
    """
    h = len(_DEFAULT_LAYOUT)
    w = len(_DEFAULT_LAYOUT[0])
    grid = np.full((h, w), FREE, dtype=np.int64)
    p_true = np.zeros((h, w), dtype=np.float64)
    start = goal = None
    for r, row in enumerate(_DEFAULT_LAYOUT):
        for c, ch in enumerate(row):
            if ch == "#":
                grid[r, c] = WALL
            elif ch == "H":
                grid[r, c] = HAZARD
                p_true[r, c] = p_hazard
            elif ch == "S":
                start = (r, c)
            elif ch == "G":
                goal = (r, c)
    if start is None or goal is None:
        raise ValueError("Layout must contain exactly one S and one G.")
    return RoadMap(grid=grid, start=start, goal=goal, p_true=p_true)


def twin_peaks_road(p_hazard: float = 0.012) -> RoadMap:
    """Build the twin-peaks world: two blob-shaped mountains, a hazardous pass, detours outside.

    A 100x120 grid for an 8-directional car. The mountains are unions of overlapping ellipses --
    the right one bigger and with a notched top -- so the world reads naturally rather than as
    rectangles. The short route runs straight up the pass between the mountains through a hazard
    belt that seals the pass's full width for 12 rows; the safe detours go around the outside of
    either mountain, the LEFT one being the shorter (correct) detour. Start bottom-center,
    destination top-center. Shortcut 91 steps, left detour ~137 steps, right detour ~155 steps;
    the ~46-step premium of the left detour balances the 12-row hazard traverse (see the config).
    """
    h, w = 100, 120
    grid = np.full((h, w), FREE, dtype=np.int64)
    grid[0, :] = grid[-1, :] = WALL
    grid[:, 0] = grid[:, -1] = WALL
    rr, cc = np.mgrid[0:h, 0:w]

    def ellipse(cy: float, cx: float, ry: float, rx: float) -> np.ndarray:
        return ((rr - cy) / ry) ** 2 + ((cc - cx) / rx) ** 2 <= 1.0

    # left mountain (smaller): a two-lobe blob, envelope ~rows 24..74, cols 8..50
    left = ellipse(40, 29, 16, 21) | ellipse(60, 28, 14, 19)
    # right mountain (bigger -- the WRONG detour side): three lobes, envelope ~rows 21..77,
    # cols 70..113, with a concave notch in the upper edge (open to the outside)
    right = ellipse(38, 90, 17, 20) | ellipse(60, 92, 17, 19) | ellipse(46, 102, 13, 11)
    right &= ~ellipse(28, 97, 8, 6)
    grid[left | right] = WALL
    for r in range(44, 56):  # seal the pass: every free cell between the mountains is hazardous
        c0 = c1 = 60
        while grid[r, c0 - 1] == FREE:
            c0 -= 1
        while grid[r, c1 + 1] == FREE:
            c1 += 1
        grid[r, c0 : c1 + 1] = HAZARD
    p_true = np.where(grid == HAZARD, p_hazard, 0.0)
    return RoadMap(grid=grid, start=(95, 60), goal=(4, 60), p_true=p_true, actions=ACTIONS_8)


# world name (config `settings.<name>.world`) -> builder
ROADS = {"default": default_road, "twin_peaks": twin_peaks_road}


def build_road(name: str, *, p_hazard: float) -> RoadMap:
    """Build a registered world by name with the given true hazard probability.

    Raises:
        ValueError: `name` is not a registered world.
    """
    if name not in ROADS:
        raise ValueError(f"unknown world {name!r}; choose from {sorted(ROADS)}")
    return ROADS[name](p_hazard=p_hazard)
