"""
robot.py — Husky robot wrapper for PyBullet.
Provides differential-drive control (v, w) and pose/velocity readout.
"""
from __future__ import annotations
import math
import pybullet as p
import pybullet_data


class HuskyRobot:
    """
    Husky robot from pybullet_data.
    Control: linear velocity v (m/s) + angular velocity w (rad/s).
    """

    def __init__(self, cid: int, start_pos=(0, 0, 0.1),
                 start_yaw=0.0, scale: float = 1.0):
        self.cid = cid
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)

        orn = p.getQuaternionFromEuler((0, 0, start_yaw))
        scaled_pos = (start_pos[0], start_pos[1], start_pos[2] * scale)
        self.body_id = p.loadURDF(
            "husky/husky.urdf", scaled_pos, orn,
            globalScaling=scale, physicsClientId=cid)

        # Find wheel joints by name
        self.wheel_joints = []
        for j in range(p.getNumJoints(self.body_id, physicsClientId=cid)):
            info = p.getJointInfo(self.body_id, j, physicsClientId=cid)
            name = info[1].decode("utf-8")
            if "wheel" in name:
                self.wheel_joints.append(j)

        if len(self.wheel_joints) < 2:
            raise RuntimeError("Could not find Husky wheel joints.")

        # Differential-drive parameters (tuned for stability)
        self.wheel_radius = 0.165 * scale
        self.wheel_base = 0.55 * scale
        self.drive_force = 50000.0

        # For odometry
        self._last_joint_pos = None

        print(f"[HUSKY] wheels: {len(self.wheel_joints)} joints, "
              f"radius={self.wheel_radius:.3f}, base={self.wheel_base:.3f}")

    def set_cmd_vel(self, v: float, w: float):
        """Set linear (m/s) and angular (rad/s) velocity command."""
        # Differential drive: v, w -> left/right wheel angular velocities
        v_l = v - (w * self.wheel_base / 2.0)
        v_r = v + (w * self.wheel_base / 2.0)

        wl = v_l / self.wheel_radius
        wr = v_r / self.wheel_radius

        force = self.drive_force

        for j in self.wheel_joints:
            name = p.getJointInfo(self.body_id, j, physicsClientId=self.cid)[1].decode("utf-8")
            target = wl if "left" in name else wr
            p.setJointMotorControl2(
                self.body_id, j,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=target,
                force=force,
                positionGain=0.0,
                velocityGain=1.0,
                physicsClientId=self.cid)

    def get_pose(self) -> tuple[float, float, float]:
        """Return (x, y, yaw) ground-truth pose."""
        pos, orn = p.getBasePositionAndOrientation(
            self.body_id, physicsClientId=self.cid)
        yaw = p.getEulerFromQuaternion(orn)[2]
        return float(pos[0]), float(pos[1]), float(yaw)

    def get_velocity(self) -> tuple[float, float]:
        """Return (v_linear, w_angular) from physics engine."""
        lin_vel, ang_vel = p.getBaseVelocity(
            self.body_id, physicsClientId=self.cid)
        v = math.hypot(lin_vel[0], lin_vel[1])
        w = ang_vel[2]
        return float(v), float(w)

    def get_wheel_angles(self) -> list[float]:
        """Return current wheel joint angles."""
        angles = []
        for j in self.wheel_joints:
            st = p.getJointState(self.body_id, j, physicsClientId=self.cid)
            angles.append(st[0])
        return angles

    def get_left_right_wheel_angles(self) -> tuple[float, float]:
        """Return average (left_angle, right_angle) from all wheels."""
        left_angles = []
        right_angles = []
        for j in self.wheel_joints:
            name = p.getJointInfo(self.body_id, j, physicsClientId=self.cid)[1].decode("utf-8")
            st = p.getJointState(self.body_id, j, physicsClientId=self.cid)
            if "left" in name:
                left_angles.append(st[0])
            else:
                right_angles.append(st[0])
        left_avg = sum(left_angles) / max(1, len(left_angles))
        right_avg = sum(right_angles) / max(1, len(right_angles))
        return left_avg, right_avg
