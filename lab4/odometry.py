from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class OdomConfig:
    wheel_radius: float = 0.165
    wheel_base: float = 0.55


class DiffDriveOdometry:
    def __init__(self, cfg: OdomConfig):
        self.cfg = cfg
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self._last_left = None
        self._last_right = None



    def reset(self, x=0.0, y=0.0, theta=0.0):
        self.x, self.y, self.theta = x, y, theta
        self._last_left = None
        self._last_right = None

    def update_from_wheels(self, left_angle: float, right_angle: float):
        # Premier appel : on initialise et on SORT
        if self._last_left is None or self._last_right is None:
            self._last_left = left_angle
            self._last_right = right_angle
            return self.x, self.y, self.theta, 0.0, 0.0  # dC, dT = 0

        # À partir d'ici, Pylance sait que _last_left/_last_right sont des float
        last_left: float = self._last_left
        last_right: float = self._last_right

        dL = (left_angle - last_left) * self.cfg.wheel_radius
        dR = (right_angle - last_right) * self.cfg.wheel_radius

        self._last_left = left_angle
        self._last_right = right_angle

        dC = 0.5 * (dL + dR)
        dT = (dR - dL) / self.cfg.wheel_base

        self.theta = (self.theta + dT + math.pi) % (2 * math.pi) - math.pi
        self.x += dC * math.cos(self.theta)
        self.y += dC * math.sin(self.theta)

        return self.x, self.y, self.theta, dC, dT

