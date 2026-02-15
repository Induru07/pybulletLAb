from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence
import numpy as np

Grid = List[List[int]]  # 1 = mur, 0 = vide




def load_grid(path: str | Path) -> Grid:
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    grid: Grid = []
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        if " " in s or "\t" in s:
            row = [int(x) for x in s.replace("\t", " ").split() if x in ("0", "1")]
        else:
            row = [int(ch) for ch in s if ch in ("0", "1")]
        if row:
            grid.append(row)
    if not grid:
        raise ValueError(f"Map vide: {path}")
    w = len(grid[0])
    if any(len(r) != w for r in grid):
        raise ValueError("Map non rectangulaire.")
    return grid


@dataclass(frozen=True)
class GridMapConfig:
    cell_size: float = 0.5   # mètres par cellule
    max_range: float = 6.0   # portée max du scan (m)
    step: float = 0.05       # pas de ray-marching (m)



class GridMapLidar:
    """
    “Lidar” simulé sur une grille 0/1.
    Convertit (x,y) monde (mètres) -> indices cellule, et raycast jusqu’au mur.
    """
    def __init__(self, grid: Grid, cfg: GridMapConfig = GridMapConfig()):
        self.grid = grid
        self.grid_np = np.asarray(grid, dtype=np.uint8)
        self.cfg = cfg
        self.rows = len(grid)
        self.cols = len(grid[0])

        self.world_w = self.cols * cfg.cell_size
        self.world_h = self.rows * cfg.cell_size
        self.origin_x = -self.world_w / 2.0
        self.origin_y = -self.world_h / 2.0

    def _world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        cx = int((x - self.origin_x) / self.cfg.cell_size)
        cy = int((y - self.origin_y) / self.cfg.cell_size)
        return cy, cx  # row, col

    def _is_wall(self, x: float, y: float) -> bool:
        r, c = self._world_to_cell(x, y)
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            return True  # hors map = mur
        return bool(self.grid_np[r, c] == 1)

    def _is_wall_points(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        r = ((y - self.origin_y) / self.cfg.cell_size).astype(np.int32)
        c = ((x - self.origin_x) / self.cfg.cell_size).astype(np.int32)

        outside = (r < 0) | (r >= self.rows) | (c < 0) | (c >= self.cols)
        hit = outside.copy()

        inside = ~outside
        if np.any(inside):
            hit[inside] = self.grid_np[r[inside], c[inside]] == 1
        return hit

    def is_free(self, x: float, y: float) -> bool:
        return not self._is_wall(x, y)

    def scan(self, x: float, y: float, theta: float, angles: Sequence[float]) -> np.ndarray:
        """
        Retourne un tableau distances (m) pour chaque angle relatif dans angles.
        """
        a = np.asarray(angles, dtype=np.float32)
        n = int(a.shape[0])
        if n == 0:
            return np.empty((0,), dtype=np.float32)

        ang = theta + a
        cos_a = np.cos(ang).astype(np.float32, copy=False)
        sin_a = np.sin(ang).astype(np.float32, copy=False)

        dists = np.full(n, self.cfg.max_range, dtype=np.float32)
        active = np.ones(n, dtype=bool)
        active_idx = np.arange(n, dtype=np.int32)

        dist = 0.0
        while dist < self.cfg.max_range and np.any(active):
            dist += self.cfg.step

            idx = active_idx[active]
            px = x + dist * cos_a[idx]
            py = y + dist * sin_a[idx]
            hit = self._is_wall_points(px, py)

            if np.any(hit):
                hit_idx = idx[hit]
                dists[hit_idx] = dist
                active[hit_idx] = False

        return dists

    def scan_particles(self, poses: np.ndarray, angles: Sequence[float], out: np.ndarray | None = None) -> np.ndarray:
        poses_arr = np.asarray(poses, dtype=np.float32)
        a = np.asarray(angles, dtype=np.float32)
        n = int(poses_arr.shape[0])
        k = int(a.shape[0])

        if out is None:
            dists = np.full((n, k), self.cfg.max_range, dtype=np.float32)
        else:
            dists = out
            dists.fill(self.cfg.max_range)

        if n == 0 or k == 0:
            return dists

        x0 = poses_arr[:, 0:1]
        y0 = poses_arr[:, 1:2]
        theta = poses_arr[:, 2:3]

        ang = theta + a[None, :]
        cos_a = np.cos(ang).astype(np.float32, copy=False)
        sin_a = np.sin(ang).astype(np.float32, copy=False)

        active = np.ones((n, k), dtype=bool)
        dist = 0.0
        while dist < self.cfg.max_range and np.any(active):
            dist += self.cfg.step
            px = x0 + dist * cos_a
            py = y0 + dist * sin_a
            hit = self._is_wall_points(px, py)
            new_hits = active & hit
            if np.any(new_hits):
                dists[new_hits] = dist
                active[new_hits] = False

        return dists
