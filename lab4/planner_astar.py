"""
planner_astar.py — A* pathfinding on a 2D grid.
Supports 4- and 8-connectivity with diagonal movement costs.
"""
from __future__ import annotations
import heapq
import math
from typing import Iterable


def _is_free(cell) -> bool:
    """Check if grid cell is free/traversable."""
    if cell is None:
        return False
    if isinstance(cell, (int, float)):
        return cell == 0
    if isinstance(cell, str):
        return cell not in ["#", "X", "1"]
    return bool(cell) is False


_DIAG = math.sqrt(2)
_DIRS_8 = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, _DIAG), (-1, 1, _DIAG), (1, -1, _DIAG), (1, 1, _DIAG),
]
_DIRS_4 = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]


def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    """Chebyshev distance (admissible for 8-connectivity)."""
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return max(dr, dc) + (_DIAG - 1) * min(dr, dc)


def inflate_grid(grid, radius: int = 1) -> list[list[int]]:
    """
    Grow obstacles by `radius` cells (Minkowski sum).
    Returns a new grid with inflated obstacles.
    """
    rows, cols = len(grid), len(grid[0])
    inflated = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if not _is_free(grid[r][c]):
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < rows and 0 <= cc < cols:
                            inflated[rr][cc] = 1
    return inflated


def astar(grid, start: tuple[int, int], goal: tuple[int, int],
          eight_connected: bool = True) -> list[tuple[int, int]]:
    """
    A* pathfinding on a 2D grid.
    Returns list of cells from start to goal, or [] if no path.
    """
    rows, cols = len(grid), len(grid[0])
    dirs = _DIRS_8 if eight_connected else _DIRS_4

    sr, sc = start
    gr, gc = goal
    if not _is_free(grid[sr][sc]) or not _is_free(grid[gr][gc]):
        return []

    frontier: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(frontier, (0.0, start))

    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    cost_so_far: dict[tuple[int, int], float] = {start: 0.0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break

        r, c = current
        for dr, dc, step_cost in dirs:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < rows and 0 <= cc < cols):
                continue
            if not _is_free(grid[rr][cc]):
                continue

            # For diagonal moves, check that both cardinal neighbors are free
            if dr != 0 and dc != 0:
                if not _is_free(grid[r + dr][c]) or not _is_free(grid[r][c + dc]):
                    continue

            new_cost = cost_so_far[current] + step_cost
            nb = (rr, cc)
            if nb not in cost_so_far or new_cost < cost_so_far[nb]:
                cost_so_far[nb] = new_cost
                priority = new_cost + _heuristic(nb, goal)
                heapq.heappush(frontier, (priority, nb))
                came_from[nb] = current

    if goal not in came_from:
        return []

    # Reconstruct path
    path = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = came_from[cur]
    path.reverse()
    return path
