"""
simulation.py — Clean simulation engine for the AMR warehouse project.
Handles: world setup, multi-robot management, human obstacles, logging.
"""
from __future__ import annotations

import math
import json
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import numpy as np
import pybullet as p

from shared.utils.grid_map import load_grid, GridMapLidar, GridMapConfig
from shared.utils.logger import RunLogger
from shared.utils.maze_builder import MazeBuilder, MazePhysicsConfig
from shared.utils.spawn import find_first_open_area_top
from lab4.agent import RobotAgent
from lab4.human import SimulatedHuman
from lab4.config import SimConfig
from lab4.world import cell_center_to_world


class Simulation:
    """
    Main simulation engine for the AMR warehouse project.
    Manages multiple robots (RobotAgents) and humans.
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        if cfg.seed is not None:
            np.random.seed(cfg.seed)

        # ── Connect PyBullet ──
        self.cid = p.connect(p.DIRECT if cfg.direct else p.GUI)
        if not cfg.direct:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
            p.resetDebugVisualizerCamera(
                cameraDistance=7.0, cameraYaw=45, cameraPitch=-55,
                cameraTargetPosition=(0, 0, 0), physicsClientId=self.cid)

        p.resetSimulation(physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)

        self.dt = 1.0 / float(cfg.hz)
        p.setTimeStep(self.dt, physicsClientId=self.cid)

        # ── Build world ──
        self.grid = load_grid(cfg.map_path)
        maze = MazeBuilder(self.cid, MazePhysicsConfig(cell_size=cfg.cell_size, friction=cfg.friction))
        maze.add_ground()
        maze.build(self.grid)
        self.lidar = GridMapLidar(self.grid, GridMapConfig(cell_size=cfg.cell_size))
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])

        # ── Agents (Phase 5) ──
        self.agents: List[RobotAgent] = []
        n_robots = max(1, cfg.n_robots)
        print(f"[SIM] Spawning {n_robots} robots...")

        # ── Logger (create early so agents can use it) ──
        self.logger = RunLogger.create("shared/data")
        
        for i in range(n_robots):
            spawn_r, spawn_c = 0, 0
            if i == 0:
                try:
                     spawn_r, spawn_c = find_first_open_area_top(self.grid, free_block=6, margin=1)
                except RuntimeError:
                     try:
                         spawn_r, spawn_c = find_first_open_area_top(self.grid, free_block=4, margin=1)
                     except RuntimeError:
                         try:
                             spawn_r, spawn_c = find_first_open_area_top(self.grid, free_block=2, margin=0)
                         except RuntimeError:
                             # Tight maze: find first free cell scanning top-to-bottom
                             found = False
                             for rr in range(1, self.rows - 1):
                                 for cc in range(1, self.cols - 1):
                                     if self.grid[rr][cc] == 0:
                                         spawn_r, spawn_c = rr, cc
                                         found = True
                                         break
                                 if found:
                                     break
            else:
                 spawn_r = self.rng.integers(1, self.rows-2)
                 spawn_c = self.rng.integers(1, self.cols-2)
                 while self.grid[spawn_r][spawn_c] != 0:
                      spawn_r = self.rng.integers(1, self.rows-2)
                      spawn_c = self.rng.integers(1, self.cols-2)

            # Detect corridor direction for spawn yaw
            spawn_yaw = self._detect_corridor_yaw(spawn_r, spawn_c)

            sx, sy = cell_center_to_world(spawn_r, spawn_c, self.rows, self.cols, cfg.cell_size)
            
            # Create agent
            agent = RobotAgent(i, self.cid, (sx, sy, spawn_yaw), self.grid, self.lidar, cfg,
                               logger=self.logger)
            self.agents.append(agent)

        # ── Humans (Phase 4) ──
        self.humans: List[SimulatedHuman] = []
        if cfg.n_humans > 0:
            print(f"[SIM] Spawning {cfg.n_humans} humans...")
            for i in range(cfg.n_humans):
                # Retry logic for spawning humans away from robots is tricky with multiple robots.
                # Simplified: just spawn random free.
                hr = self.rng.integers(1, self.rows-2)
                hc = self.rng.integers(1, self.cols-2)
                if self.grid[hr][hc] == 0:
                     hx, hy = cell_center_to_world(hr, hc, self.rows, self.cols, cfg.cell_size)
                     self.humans.append(SimulatedHuman(
                             self.cid, hx, hy, self.grid, cfg.cell_size, seed=i))

        print(f"MAP: {cfg.map_path}")
        print(f"Logs in: {self.logger.run_dir}")

        # ── Draw Pick/Drop markers in GUI ──
        if not cfg.direct:
            self._draw_job_markers()

        # ── Results ──
        self.result = {
            "success": False,
            "sim_time": 0.0,
            "distance": 0.0,
            "mean_pf_error": 0.0,
            "gt_usage_pct": 0.0,
        }

    def _detect_corridor_yaw(self, r: int, c: int) -> float:
        """Detect corridor direction at (r,c) and return a yaw aligned with it."""
        free_h = 0  # horizontal free neighbors
        free_v = 0  # vertical free neighbors
        for dc in [-1, 1]:
            cc = c + dc
            if 0 <= cc < self.cols and self.grid[r][cc] == 0:
                free_h += 1
        for dr in [-1, 1]:
            rr = r + dr
            if 0 <= rr < self.rows and self.grid[rr][c] == 0:
                free_v += 1
        if free_h > free_v:
            return 0.0            # horizontal corridor → face east (0 rad)
        elif free_v > free_h:
            return math.pi / 2    # vertical corridor → face north (π/2)
        return 0.0                # open area or corner → default east

    def shutdown(self):
        p.disconnect(self.cid)

    def _draw_job_markers(self):
        """Draw pick (blue) and drop (red) markers in the PyBullet GUI."""
        for agent in self.agents:
            if agent.job_manager is None:
                continue
            # Draw current job + queued jobs
            all_jobs = []
            if agent.job_manager.current_job:
                all_jobs.append(agent.job_manager.current_job)
            all_jobs.extend(agent.job_manager.jobs)

            for job in all_jobs:
                # Pick marker — blue sphere
                pr, pc = job.pick_cell
                px, py = cell_center_to_world(pr, pc, self.rows, self.cols, self.cfg.cell_size)
                vs_pick = p.createVisualShape(
                    p.GEOM_SPHERE, radius=0.15,
                    rgbaColor=[0, 0.4, 1, 0.8], physicsClientId=self.cid)
                p.createMultiBody(baseMass=0, baseVisualShapeIndex=vs_pick,
                                  basePosition=[px, py, 0.15], physicsClientId=self.cid)
                p.addUserDebugText(f"P{job.id}", [px, py, 0.5],
                                   textColorRGB=[0, 0.3, 1], textSize=1.2,
                                   lifeTime=0, physicsClientId=self.cid)

                # Drop marker — red sphere
                dr, dc = job.drop_cell
                dx, dy = cell_center_to_world(dr, dc, self.rows, self.cols, self.cfg.cell_size)
                vs_drop = p.createVisualShape(
                    p.GEOM_SPHERE, radius=0.15,
                    rgbaColor=[1, 0.2, 0, 0.8], physicsClientId=self.cid)
                p.createMultiBody(baseMass=0, baseVisualShapeIndex=vs_drop,
                                  basePosition=[dx, dy, 0.15], physicsClientId=self.cid)
                p.addUserDebugText(f"D{job.id}", [dx, dy, 0.5],
                                   textColorRGB=[1, 0.2, 0], textSize=1.2,
                                   lifeTime=0, physicsClientId=self.cid)

    def run(self) -> dict:
        """Run the simulation. Returns a result dict."""
        cfg = self.cfg
        sim_t = 0.0
        
        # Tracking metrics
        total_pf_errors = []
        last_poses = {}
        total_distance = 0.0
        
        for agent in self.agents:
            ax, ay, _ = agent.get_pose()
            last_poses[agent.id] = (ax, ay)
        
        while True:
            # ── Time-out check ──
            if cfg.autotest and sim_t >= cfg.max_sim_s:
                break
            if cfg.nav_enabled and sim_t >= cfg.max_sim_s:
                print(f"[NAV] timeout at {sim_t:.1f}s")
                break

            # ── Update Humans ──
            human_positions = []
            for h in self.humans:
                h.update(self.dt)
                hx, hy, _, _ = h.get_pose()
                human_positions.append((hx, hy))

            # ── Update Agents ──
            # Basic traffic coordination: Robot sees OTHER robots + humans as obstacles
            
            # First collect all robot poses
            agent_poses = []
            for agent in self.agents:
                ax, ay, _ = agent.get_pose()
                agent_poses.append((ax, ay))
            
            # Now update each agent
            all_done = True
            for i, agent in enumerate(self.agents):
                if not agent.active:
                    continue
                all_done = False
                
                # Construct obstacles list for this agent
                # Humans + other robots
                obstacles = list(human_positions)
                for j, (ax, ay) in enumerate(agent_poses):
                    if i != j:
                        obstacles.append((ax, ay))
                
                # Update agent
                metrics = agent.update(self.dt, sim_t, obstacles)
                
                if "pf_err" in metrics:
                    total_pf_errors.append(metrics["pf_err"])
                
                # Update distance
                ax_new, ay_new, _ = agent.get_pose()
                lx, ly = last_poses[agent.id]
                dist = math.hypot(ax_new - lx, ay_new - ly)
                total_distance += dist
                last_poses[agent.id] = (ax_new, ay_new)

            # ── Step Physics ──
            p.stepSimulation(physicsClientId=self.cid)
            if not cfg.direct:
                time.sleep(self.dt)
            
            sim_t += self.dt
            
            # Check completion
            if all_done:
                print(f"[SIM] All agents finished at {sim_t:.2f}s")
                self.result["success"] = True
                break

        self.result["sim_time"] = sim_t
        self.result["distance"] = total_distance
        if total_pf_errors:
            self.result["mean_pf_error"] = float(np.mean(total_pf_errors))

        # ── Compute GT usage percentage ──
        total_gt_time = sum(a.gt_usage_time for a in self.agents)
        n_active = max(1, len(self.agents))
        if sim_t > 0:
            self.result["gt_usage_pct"] = 100.0 * total_gt_time / (sim_t * n_active)

        # ── Save mission summary JSON ──
        summary = {
            "success": self.result["success"],
            "sim_time": round(sim_t, 3),
            "distance": round(total_distance, 3),
            "mean_pf_error": round(self.result["mean_pf_error"], 5),
            "gt_usage_pct": round(self.result.get("gt_usage_pct", 0), 2),
            "n_robots": cfg.n_robots,
            "n_humans": cfg.n_humans,
            "n_jobs": cfg.n_jobs,
            "n_particles": cfg.n_particles,
            "map": cfg.map_path,
            "seed": cfg.seed,
            "agents": [],
        }
        for agent in self.agents:
            summary["agents"].append({
                "id": agent.id,
                "active": agent.active,
                "gt_usage_time": round(agent.gt_usage_time, 3),
                "total_resample_count": agent.total_resample_count,
            })
        summary_path = self.logger.run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[SIM] Summary saved to {summary_path}")
            
        # ── Save SLAM Maps ──
        for agent in self.agents:
            if hasattr(agent, "mapper") and agent.mapper:
                outfile = f"slam_map_agent{agent.id}.txt"
                print(f"[SIM] Saving SLAM map to {outfile}...")
                agent.mapper.save(outfile)

        return self.result
