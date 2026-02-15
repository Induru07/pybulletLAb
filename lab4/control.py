# lab4/control.py
from __future__ import annotations
import pybullet as p


class KeyboardTeleop:
    def __init__(self):
        self.v = 0.0
        self.w = 0.0
        self.v_step = 0.6
        self.w_step = 1.5

        self.v_max = 3.5
        self.w_max = 3.5

    def update(self, cid: int):
        keys = p.getKeyboardEvents(physicsClientId=cid)

        def down(k):
            return (k in keys) and (keys[k] & p.KEY_IS_DOWN)

        # Support AZERTY + QWERTY + flèches
        forward = down(ord('w')) or down(ord('z')) or down(p.B3G_UP_ARROW)
        backward = down(ord('s')) or down(p.B3G_DOWN_ARROW)
        left = down(ord('a')) or down(ord('q')) or down(p.B3G_LEFT_ARROW)
        right = down(ord('d')) or down(p.B3G_RIGHT_ARROW)

        # Si aucune touche, on freine doucement
        if not (forward or backward):
            self.v *= 0.85
        if not (left or right):
            self.w *= 0.80

        if forward:
            self.v += self.v_step
        if backward:
            self.v -= self.v_step
        if left:
            self.w += self.w_step
        if right:
            self.w -= self.w_step

        # clamp
        self.v = max(min(self.v, self.v_max), -self.v_max)
        self.w = max(min(self.w, self.w_max), -self.w_max)

        return float(self.v), float(self.w)
