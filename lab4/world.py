from __future__ import annotations
import pybullet as p
from shared.utils.grid_map import load_grid, GridMapLidar, GridMapConfig
from shared.utils.maze_builder import MazeBuilder, MazePhysicsConfig


def build_world(cid: int, map_path: str, cell_size: float = 0.5):
    grid = load_grid(map_path)

    maze = MazeBuilder(cid, MazePhysicsConfig(cell_size=cell_size))
    maze.add_ground()
    maze.build(grid)

    lidar = GridMapLidar(grid, GridMapConfig(cell_size=cell_size))
    return grid, lidar

def cell_center_to_world(row: int, col: int, rows: int, cols: int, cell_size: float) -> tuple[float, float]:
    world_w = cols * cell_size
    world_h = rows * cell_size
    origin_x = -world_w / 2.0
    origin_y = -world_h / 2.0
    x = origin_x + (col + 0.5) * cell_size
    y = origin_y + (row + 0.5) * cell_size
    return x, y

def world_to_cell(x: float, y: float, rows: int, cols: int, cell_size: float) -> tuple[int, int]:
    """convert world coord (x,y) -> (row,col) """
    world_w = cols * cell_size
    world_h = rows * cell_size
    origin_x = -world_w / 2.0
    origin_y = -world_h / 2.0

    col = int((x - origin_x) / cell_size)
    row = int((y - origin_y) / cell_size)

    # clamp dans la grille
    row = max(0, min(rows - 1, row))
    col = max(0, min(cols - 1, col))
    return row, col
