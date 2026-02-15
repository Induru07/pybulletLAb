from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import pybullet as p
import pybullet_data


@dataclass(frozen=True)
class MazePhysicsConfig:
    cell_size: float = 0.5
    wall_height: float = 0.25
    friction: float = 1.0
    restitution: float = 0.05
    rgba: Tuple[float, float, float, float] = (0.9, 0.9, 0.9, 1.0)


class MazeBuilder:
    def __init__(self, cid: int, cfg: MazePhysicsConfig = MazePhysicsConfig()):
        self.cid = cid
        self.cfg = cfg

    def add_ground(self) -> int:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.cid)
        p.changeDynamics(plane_id, -1,
                         lateralFriction=self.cfg.friction,
                         restitution=self.cfg.restitution,
                         physicsClientId=self.cid)
        return plane_id

    def build(self, grid: list[list[int]], base_z: float = 0.0) -> List[int]:
        rows = len(grid)
        cols = len(grid[0])
        cs = self.cfg.cell_size
        h = self.cfg.wall_height

        world_w = cols * cs
        world_h = rows * cs
        origin_x = -world_w / 2.0
        origin_y = -world_h / 2.0

        half_extents = (cs / 2.0, cs / 2.0, h / 2.0)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.cid)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=self.cfg.rgba, physicsClientId=self.cid)

        ids: List[int] = []
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] != 1:
                    continue
                x = origin_x + (c + 0.5) * cs
                y = origin_y + (r + 0.5) * cs
                z = base_z + h / 2.0
                wall_id = p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=vis,
                    basePosition=(x, y, z),
                    physicsClientId=self.cid
                )
                p.changeDynamics(wall_id, -1,
                                 lateralFriction=self.cfg.friction,
                                 restitution=self.cfg.restitution,
                                 physicsClientId=self.cid)
                ids.append(wall_id)
        return ids
