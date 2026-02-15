"""
agent.py — RobotAgent class for multi-robot simulation.
Encapsulates: Robot hardware interface, PF localization, Navigator, JobManager.
"""
from __future__ import annotations

import math
import numpy as np
from typing import Optional, Tuple, List, Dict

from lab4.config import SimConfig
from lab4.robot import HuskyRobot
from lab4.navigator import Navigator, NavConfig
from lab4.odometry import DiffDriveOdometry, OdomConfig
from lab4.job_manager import JobManager
from lab4.slam import OccupancyGridMapper
from lab4.world import cell_center_to_world, world_to_cell
from shared.utils.particle_filter import ParticleFilter, PFConfig
from shared.utils.grid_map import GridMapLidar
from shared.utils.logger import RunLogger
from shared.utils.spawn import find_first_open_area_top


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _resample_near_spawn(pf: ParticleFilter, lidar: GridMapLidar,
                         sx: float, sy: float,
                         std_xy: float, std_th: float,
                         max_tries: int = 50) -> None:
    """Resample particles that landed inside walls near the spawn point."""
    bad = np.array([not lidar.is_free(float(px), float(py))
                    for px, py in pf.p[:, :2]])
    tries = 0
    while bad.any() and tries < max_tries:
        n_bad = int(bad.sum())
        pf.p[bad, 0] = np.random.normal(sx, std_xy, size=n_bad)
        pf.p[bad, 1] = np.random.normal(sy, std_xy, size=n_bad)
        pf.p[bad, 2] = np.random.normal(0.0, std_th, size=n_bad)
        pf.p[:, 2] = _wrap(pf.p[:, 2])
        bad = np.array([not lidar.is_free(float(px), float(py))
                        for px, py in pf.p[:, :2]])
        tries += 1
    if bad.any():
        print(f"[WARN] {int(bad.sum())} particles still in walls after resample.")


def _fill_particle_scans(pf: ParticleFilter, lidar: GridMapLidar,
                         angles: np.ndarray, out: np.ndarray) -> None:
    """Compute simulated lidar scans for all particles."""
    if hasattr(lidar, "scan_particles"):
        lidar.scan_particles(pf.p, angles, out=out)
    else:
        for i in range(pf.cfg.n):
            out[i, :] = lidar.scan(
                float(pf.p[i, 0]), float(pf.p[i, 1]),
                float(pf.p[i, 2]), angles)


class RobotAgent:
    def __init__(self, agent_id: int, cid: int, start_pose: Tuple[float, float, float],
                 grid: List[List[float]], lidar: GridMapLidar, cfg: SimConfig,
                 logger: Optional['RunLogger'] = None):
        self.id = agent_id
        self.cid = cid
        self.cfg = cfg
        self.grid = grid
        self.lidar = lidar
        self.logger = logger
        self.rows = len(grid)
        self.cols = len(grid[0])
        
        sx, sy, sth = start_pose
        self.start_pose = start_pose

        # ── Robot Hardware ──
        # Auto-scale: Husky at scale=1.0 is ~0.695m wide.
        # Target: robot width ≈ 60% of cell_size for clearance in corridors.
        robot_scale = max(0.2, min(0.6, (cfg.cell_size * 0.60) / 0.695))
        self.robot = HuskyRobot(cid, start_pos=(sx, sy, 0.1), start_yaw=sth, scale=robot_scale)
        
        # ── Particle Filter ──
        pf_cfg = PFConfig(
            n=cfg.n_particles,
            trans_noise_per_m=0.10,
            rot_noise_per_rad=0.08,
            trans_noise_min=0.002,
            rot_noise_min=0.002,
            meas_std=0.3,
        )
        self.pf = ParticleFilter(pf_cfg)
        self.pf.init_gaussian(sx, sy, sth, std_xy=cfg.pf_init_std_xy, std_th=cfg.pf_init_std_th)
        _resample_near_spawn(self.pf, lidar, sx, sy, std_xy=cfg.pf_init_std_xy, std_th=cfg.pf_init_std_th)
        
        # Precompute ray angles
        K = max(4, cfg.n_lidar_rays)
        self.angles = np.linspace(-math.pi, math.pi, K, endpoint=False).astype(np.float32)
        self.z_hat_buf = np.empty((cfg.n_particles, K), dtype=np.float32)

        # ── Navigator ──
        # Scale safety radii to robot size
        nav_safety = max(0.12, robot_scale * 0.55)
        nav_slowdown = max(0.25, robot_scale * 1.1)
        self.nav = Navigator(grid, NavConfig(
            cell_size=cfg.cell_size,
            safety_radius=nav_safety,
            slowdown_radius=nav_slowdown,
        ))

        # ── Job Manager ──
        self.job_manager: Optional[JobManager] = None
        if cfg.n_jobs > 0:
            self.job_manager = JobManager(grid)
            self.job_manager.generate_random_jobs(cfg.n_jobs, seed=(cfg.seed + agent_id) if cfg.seed else None)
        
        # ── SLAM (Phase 6) ──
        self.mapper: Optional[OccupancyGridMapper] = None
        if cfg.slam_enabled:
            self.mapper = OccupancyGridMapper(self.rows, self.cols, cfg.cell_size)

        # ── Odometry (velocity-based, avoids wheel slippage) ──
        self.odom = DiffDriveOdometry(OdomConfig(
            wheel_radius=self.robot.wheel_radius,
            wheel_base=self.robot.wheel_base,
        ))
        self.odom.reset(sx, sy, sth)

        # ── State ──
        self.last_plan_t = -1e9
        self.stall_count = 0
        self.stall_ref_t = 0.0
        self.stall_ref_x = sx
        self.stall_ref_y = sy
        self.stall_ref_th = sth
        self.backup_until = -1.0
        self.emergency_forward_until = -1.0
        self.was_degraded = False
        self.gt_usage_time = 0.0
        self.total_resample_count = 0
        self.active = True  # If finished all jobs
        self._debug_frame = 0  # throttle GUI drawing

        # Warmup PF
        self._pf_warmup()

        # Set initial goal
        self._setup_initial_goal()

    def _pf_warmup(self):
        sx, sy, sth = self.start_pose
        for _ in range(5):
            z = self.lidar.scan(sx, sy, sth, self.angles)
            _fill_particle_scans(self.pf, self.lidar, self.angles, self.z_hat_buf)
            self.pf.update(z, self.z_hat_buf)
            self.pf.resample()

    def draw_debug_info(self, pf_err: float):
        """Draw GT, PF, and Error in PyBullet GUI."""
        if self.cid < 0: return # Headless/Direct mode check? No, cid is always valid but GUI might not be.
        import pybullet as p
        
        x, y, z = self.robot.get_pose()
        # Text above robot
        text = f"ID:{self.id}\nGT:({x:.1f},{y:.1f})\nErr:{pf_err:.2f}m"
        p.addUserDebugText(text, [x, y, 1.5], textColorRGB=[0, 0, 0], textSize=1.0, lifeTime=0.1, physicsClientId=self.cid)
        
        # Draw PF mean?
        # Draw PF mean as a Red Cross
        px, py, _ = self.pf.estimate()
        # p.addUserDebugText("PF", [px, py, 0.5], textColorRGB=[1, 0, 0], textSize=0.8, lifeTime=0.1, physicsClientId=self.cid)
        s = 0.3 # Size of cross
        p.addUserDebugLine([px-s, py, 0.05], [px+s, py, 0.05], [1, 0, 0], lineWidth=3, lifeTime=0.1, physicsClientId=self.cid)
        p.addUserDebugLine([px, py-s, 0.05], [px, py+s, 0.05], [1, 0, 0], lineWidth=3, lifeTime=0.1, physicsClientId=self.cid)
        
        # Also draw some particles (every 20th one)
        step = max(1, self.pf.p.shape[0] // 20)
        for i in range(0, self.pf.p.shape[0], step):
             ppx, ppy = self.pf.p[i, 0], self.pf.p[i, 1]
             p.addUserDebugLine([ppx, ppy, 0.05], [ppx, ppy, 0.1], [0, 1, 0], lineWidth=2, lifeTime=0.1, physicsClientId=self.cid)


    def _setup_initial_goal(self):
        if self.job_manager:
            tgt = self.job_manager.next_target()
            if tgt:
                (r, c), mode = tgt
                print(f"[Agent {self.id}] Starting: {mode} at ({r},{c})")
                self.nav.set_goal_cell(r, c)
            else:
                self.active = False
                return

        elif self.cfg.goal_mode == "random":
             self._pick_random_goal()

    def _pick_random_goal(self):
        """Pick a random free cell as goal."""
        # Simple rejection sampling
        for _ in range(100):
            r = np.random.randint(1, self.rows - 1)
            c = np.random.randint(1, self.cols - 1)
            if self.grid[r][c] == 0:
                # Check distance?
                # For now just set it.
                print(f"[Agent {self.id}] New random goal: ({r},{c})")
                self.nav.set_goal_cell(r, c)
                sx, sy, _ = self.robot.get_pose()  # Use current pose
                self.nav.plan_from_pose(sx, sy)
                return
        print(f"[Agent {self.id}] Failed to pick random goal.")


    def update(self, dt: float, sim_t: float, dynamic_obstacles: List[Tuple[float, float]]) -> Dict:
        """
        Run one control cycle.
        Returns metrics dict (e.g. pf_error, etc).
        """
        if not self.active:
            self.robot.set_cmd_vel(0, 0)
            return {}

        x_gt, y_gt, th_gt = self.robot.get_pose()
        pf_err = 0.0
        
        # ── Stall detection ──
        if (sim_t - self.stall_ref_t) >= 3.0:
            dist_moved = math.hypot(x_gt - self.stall_ref_x, y_gt - self.stall_ref_y)
            rot_moved = abs(_wrap(th_gt - self.stall_ref_th))
            
            # Check if moving command was non-zero? We don't track last command easily here.
            # Assume if active we should move.
            if dist_moved < 0.15 and rot_moved < 0.2: # Check both translation and rotation
                self.stall_count += 1
                if self.stall_count >= 2:
                    print(f"[Agent {self.id}] stall #{self.stall_count} -> backup+turn")
                    self.backup_until = sim_t + 1.0
                    self.stall_count = 0
                else:
                    print(f"[Agent {self.id}] stall detected -> force replan")
                self.last_plan_t = -1e9
            else:
                self.stall_count = 0
            self.stall_ref_t = sim_t
            self.stall_ref_x = x_gt
            self.stall_ref_y = y_gt
            self.stall_ref_th = th_gt

        # ── Control Logic ──
        v, w = 0.0, 0.0
        
        if sim_t < self.backup_until:
             v, w = -0.2, -0.5
        elif sim_t < self.emergency_forward_until:
             v, w = 0.9, 0.0
        else:
            # ── Velocity-based odometry PF predict ──
            # Use physics-engine velocity to avoid wheel-slippage overestimate
            v_meas, w_meas = self.robot.get_velocity()
            dC = v_meas * dt  # linear displacement this step
            dT = w_meas * dt  # angular change this step

            # Predict particles using velocity-derived displacement
            self.pf.predict(dC, dT)

            # ── PF Update (Lidar measurement) ──
            did_resample = False
            if (sim_t % self.cfg.scan_period) < dt:
                z = self.lidar.scan(x_gt, y_gt, th_gt, self.angles)
                _fill_particle_scans(self.pf, self.lidar, self.angles, self.z_hat_buf)
                self.pf.update(z, self.z_hat_buf)
                self.pf.resample()
                did_resample = True
                self.total_resample_count += 1
                self.last_scan = z
            else:
                self.last_scan = None

            # Estimate
            x_pf, y_pf, th_pf = self.pf.estimate()
            pf_err = math.hypot(x_pf - x_gt, y_pf - y_gt)
            neff_val = self.pf.neff()

            # ── Log PF data ──
            if self.logger and did_resample:
                self.logger.write_row(
                    f"pf_agent{self.id}.csv",
                    ["t", "x_pf", "y_pf", "th_pf", "x_gt", "y_gt", "th_gt",
                     "err_xy", "neff", "resample_count"],
                    [f"{sim_t:.4f}", x_pf, y_pf, th_pf, x_gt, y_gt, th_gt,
                     pf_err, neff_val, self.total_resample_count],
                )
            
            # Degraded mode logic
            if self.was_degraded:
                use_gt = pf_err > self.cfg.pf_degraded_exit
            else:
                use_gt = pf_err > self.cfg.pf_degraded_enter
            
            if use_gt and not self.was_degraded:
                print(f"[Agent {self.id}] degraded: PF err {pf_err:.2f}m -> using GT")
                self.was_degraded = True
            elif not use_gt and self.was_degraded:
                print(f"[Agent {self.id}] back to PF control")
                self.was_degraded = False
            
            if use_gt:
                x_ctrl, y_ctrl, th_ctrl = x_gt, y_gt, th_gt
                self.gt_usage_time += dt
            else:
                x_ctrl, y_ctrl, th_ctrl = x_pf, y_pf, th_pf
            
            # SLAM Map Update
            if self.mapper and self.last_scan is not None:
                self.mapper.update(x_ctrl, y_ctrl, th_ctrl, self.last_scan, self.angles)
            
            # Set dynamic obstacles
            self.nav.set_dynamic_obstacles(dynamic_obstacles)

            # Replan
            goal_dist_gt = float('inf')
            if self.nav.goal_cell:
                 gr, gc = self.nav.goal_cell
                 # Need conversion.
                 gx, gy = cell_center_to_world(gr, gc, self.rows, self.cols, self.cfg.cell_size)
                 goal_dist_gt = math.hypot(gx - x_gt, gy - y_gt)

            if goal_dist_gt <= self.nav.cfg.goal_tol_m:
                 # Goal reached
                 if self.job_manager:
                     self.job_manager.complete_target()
                     next_tgt = self.job_manager.next_target()
                     if next_tgt:
                         (r, c), mode = next_tgt
                         print(f"[Agent {self.id}] Next: {mode} at ({r},{c})")
                         self.nav.set_goal_cell(r, c)
                         self.nav.plan_from_pose(x_gt, y_gt)
                         self.last_plan_t = sim_t
                         self.stall_count = 0
                         self.stall_ref_t = sim_t
                         self.stall_ref_x, self.stall_ref_y, self.stall_ref_th = x_gt, y_gt, th_gt
                     else:
                         print(f"[Agent {self.id}] ALL JOBS DONE.")
                         self.active = False
                         self.robot.set_cmd_vel(0, 0)
                         return {"pf_err": pf_err}
                 else:
                     # Single goal done OR random mode loop
                     if self.cfg.goal_mode == "random":
                         self._pick_random_goal()
                         self.last_plan_t = sim_t
                     else:
                         self.active = False # Single goal done
                         return {"pf_err": pf_err}
            
            # Plan
            if (sim_t - self.last_plan_t) >= self.cfg.replan_interval and goal_dist_gt > 2.0:
                 self.nav.plan_from_pose(x_ctrl, y_ctrl)
                 self.last_plan_t = sim_t
            
            # Compute Cmd
            # Phase 2b: Last mile GT
            if goal_dist_gt < 1.5:
                 # Use GT for control near goal
                 x_ctrl, y_ctrl, th_ctrl = x_gt, y_gt, th_gt
            
            scan = None # Lidar scan for reactive avoidance?
            # We need to run dedicated nav scan?
            # In Phase 2c, simulation ran nav scan at 10Hz.
            # Here update is called every dt.
            # We can replicate nav scan logic inside compute_cmd if we pass scan data?
            # Need to call lidar.scan() for reactive layer.
            # But lidar.scan is on static map. It doesn't see other robots.
            # Reactive avoidance in Navigator uses explicit scan data.
            # Simulation.py passed `nav_scan`.
            # I should do `nav_scan` here.
            # But wait, Navigator uses `scan` args.
            
            # For pure static wall avoidance, lidar scan is enough.
            nav_scan_vals = self.lidar.scan(x_gt, y_gt, th_gt, self.angles) # K rays
            # Navigator expects dict {'front': ..., 'left': ...} ?
            # Wait, Navigator `compute_cmd` uses `scan.get('front')`.
            # I need to process rays into 'front', 'left', 'right'.
            # Or Navigator docs?
            # Let's check Navigator again.
            
            # Phase 2c implementation details?
            # I didn't verify how scan was passed in Phase 2c.
            # Simulation loop did `nav_scan = ...`.
            # I need to replicate that.
            
            # Simplified:
            # front: average of center rays
            # left: average of left rays
            # right: average of right rays
            
            # Or just pass raw array? Navigator handles dict.
            # I'll create the dict.
            
            # Map [-pi, pi] to indices.
            # 0 is back? -pi.
            # front is 0 rad? NO.
            # K rays from -pi to pi.
            # Center of array is 0 rad (front).
            # 3K/4 -> +pi/2 (left)
            # So mid=K/2 is front.
            # Improved Obstacle Avoidance: Average multiple rays
            K = len(self.angles)
            # Rays are -pi to pi. Mid is 0 (front). 
            # We want to scan a sector around Front, Left, Right.
            
            def get_sector_avg(center_idx, half_width):
                start = max(0, center_idx - half_width)
                end = min(K, center_idx + half_width + 1)
                chunk = nav_scan_vals[start:end]
                if len(chunk) == 0: return 10.0
                return float(np.min(chunk)) # Use min for safer avoidance
            
            hw = max(1, K // 12) # +/- 1 ray for 16 rays (~22 deg).
            
            # Front (mid)
            front_idx = K // 2
            front_dist = get_sector_avg(front_idx, hw)
            
            # Right (-pi/2 -> K/4)
            right_idx = K // 4
            right_dist = get_sector_avg(right_idx, hw)
            
            # Left (+pi/2 -> 3K/4)
            left_idx = int(3 * K / 4)
            left_dist = get_sector_avg(left_idx, hw)

            scan_dict = {
                'front': front_dist, 
                'left': left_dist,
                'right': right_dist
            }
            v, w, done = self.nav.compute_cmd(x_ctrl, y_ctrl, th_ctrl, scan=scan_dict)
        
        self.robot.set_cmd_vel(v, w)
        
        # GUI Debug (throttle to every 10th frame to avoid crushing FPS)
        if not self.cfg.direct:
             self._debug_frame += 1
             if self._debug_frame % 10 == 0:
                 self.draw_debug_info(pf_err)
             
        return {"pf_err": pf_err}

    def get_pose(self):
        return self.robot.get_pose()
