"""
job_manager.py — Manages a queue of pick-and-drop jobs for the AMR.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Deque
from collections import deque
from lab4.planner_astar import inflate_grid

@dataclass
class Job:
    id: int
    pick_cell: Tuple[int, int]
    drop_cell: Tuple[int, int]
    status: str = "PENDING"  # PENDING, PICKING, DROPPING, COMPLETED


class JobManager:
    """
    Manages a queue of jobs.
    Flow:
    1. generate_random_jobs(N) -> populates queue
    2. next_target() -> (r, c, "PICK") or (r, c, "DROP")
    3. complete_target() -> advances state (PICK -> DROP or DROP -> NEXT JOB)
    """

    def __init__(self, grid: List[List[float]]):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.inflated = inflate_grid(grid, radius=1)
        self.jobs: Deque[Job] = deque()
        self.current_job: Optional[Job] = None
        self.state: str = "IDLE"  # IDLE, PICKING, DROPPING

    def generate_random_jobs(self, n: int, seed: Optional[int] = None):
        """Generate N random valid jobs, then optimize order via nearest-neighbor."""
        if seed is not None:
            random.seed(seed)
        
        raw_jobs = []
        for i in range(n):
            pick = self._find_free_cell()
            drop = self._find_free_cell()
            while drop == pick:
                drop = self._find_free_cell()
            raw_jobs.append(Job(id=i+1, pick_cell=pick, drop_cell=drop))

        # ── Nearest-neighbor heuristic for job ordering ──
        ordered = self._optimize_order_nn(raw_jobs)

        for job in ordered:
            self.jobs.append(job)
            print(f"[JOB] Generated Job {job.id}: Pick {job.pick_cell} -> Drop {job.drop_cell}")

    @staticmethod
    def _cell_dist(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance between two grid cells."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _optimize_order_nn(self, jobs: List[Job]) -> List[Job]:
        """
        Nearest-neighbor heuristic: starting from (0,0), always pick the job
        whose pick cell is closest to the current position (last drop cell).
        Reduces total travel distance for multi-job missions.
        """
        if len(jobs) <= 1:
            return jobs

        remaining = list(jobs)
        ordered: List[Job] = []
        current_pos = (0, 0)  # Start position (top-left of grid)

        while remaining:
            best_idx = 0
            best_dist = float("inf")
            for i, job in enumerate(remaining):
                d = self._cell_dist(current_pos, job.pick_cell)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            chosen = remaining.pop(best_idx)
            ordered.append(chosen)
            current_pos = chosen.drop_cell  # After drop, go to next pick

        print(f"[JOB] Order optimized (nearest-neighbor): "
              f"{[j.id for j in ordered]}")
        return ordered

    def next_target(self) -> Optional[Tuple[Tuple[int, int], str]]:
        """
        Returns (target_cell, mode) where mode is "PICK" or "DROP".
        Returns None if all jobs are done.
        """
        if self.current_job is None:
            if not self.jobs:
                return None
            self.current_job = self.jobs.popleft()
            self.current_job.status = "PICKING"
            self.state = "PICKING"
            return (self.current_job.pick_cell, "PICK")
        
        if self.state == "PICKING":
            return (self.current_job.pick_cell, "PICK")
        elif self.state == "DROPPING":
            return (self.current_job.drop_cell, "DROP")
        
        return None

    def complete_target(self):
        """Call when the robot reaches the current target."""
        if self.current_job is None:
            return

        if self.state == "PICKING":
            print(f"[JOB] Picked up Job {self.current_job.id}")
            self.state = "DROPPING"
            self.current_job.status = "DROPPING"
        elif self.state == "DROPPING":
            print(f"[JOB] Dropped off Job {self.current_job.id}")
            self.current_job.status = "COMPLETED"
            self.current_job = None
            self.state = "IDLE"

    def all_done(self) -> bool:
        return (not self.jobs) and (self.current_job is None)

    def _find_free_cell(self) -> Tuple[int, int]:
        """Find a random free cell (value 0)."""
        while True:
            r = random.randint(0, self.rows - 1)
            c = random.randint(0, self.cols - 1)
            # Use inflated grid to ensure target is not too close to wall
            if self.inflated[r][c] == 0:
                return (r, c)
