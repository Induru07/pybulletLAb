"""
slam.py — Simple Occupancy Grid Mapping.
"""
from __future__ import annotations

import math
import numpy as np
from pathlib import Path
from typing import List, Tuple

from lab4.world import world_to_cell


def bresenham(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """Yields integer coordinates on the line from (x0, y0) to (x1, y1)."""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x, y))
    return points


class OccupancyGridMapper:
    """
    Builds an occupancy grid using log-odds updates from Lidar scans.
    """
    def __init__(self, rows: int, cols: int, cell_size: float):
        self.rows = rows
        self.cols = cols
        self.cell_size = cell_size
        
        # Log-odds grid. Init to 0 (p=0.5).
        self.log_odds = np.zeros((rows, cols), dtype=np.float32)
        
        # Hyperparameters
        self.l_occ = 0.8
        self.l_free = -0.4
        self.l_max = 3.5
        self.l_min = -3.5

    def update(self, x: float, y: float, th: float, dists: np.ndarray, angles: np.ndarray):
        """
        Update grid with a new scan.
        x, y, th: robot pose (estimated or GT)
        dists: 1D array of ranges
        angles: 1D array of relative angles
        """
        # Robot cell
        rx, ry = world_to_cell(x, y, self.rows, self.cols, self.cell_size)
        
        start_cell = (rx, ry)
        
        N = len(dists)
        # Precompute sensor model updates in bulk? No way, Bresenham is per ray.
        # But we can optimize by only processing unique cells.
        
        # Identify free and occ cells for this scan
        free_cells = set()
        occ_cells = set()
        
        for i in range(N):
            r = dists[i]
            # Max range check handled by scan() usually, but here:
            # If r is max_range, we assume free space up to there, but no obstacle.
            # Usually 'max_range' means 'no return'.
            
            # Let's assume r < max_range means obstacle.
            # Calculating end point
            ang = th + angles[i]
            ex = x + r * math.cos(ang)
            ey = y + r * math.sin(ang)
            
            er, ec = world_to_cell(ex, ey, self.rows, self.cols, self.cell_size)
            
            # Trace line
            line = bresenham(rx, ry, er, ec)
            
            # If hit something (r < max? we don't know max here easily unless passed).
            # Assume valid hits.
            
            # All cells except last are free
            for (cr, cc) in line[:-1]:
                if 0 <= cr < self.rows and 0 <= cc < self.cols:
                    free_cells.add((cr, cc))
            
            last_r, last_c = line[-1]
            if 0 <= last_r < self.rows and 0 <= last_c < self.cols:
                 occ_cells.add((last_r, last_c))

        # Update grid
        # Vectorized update is hard with arbitrary sets.
        # Just iterate cells.
        
        # Avoid conflict: if a cell is both free and occ in same scan?
        # Usually occ wins (at end of ray).
        # We process free first, then occ.
        # Although Bresenham ensures endpoints are handled last.
        # But rays overlap.
        # A cell might be 'free' for one ray and 'occ' for another?
        # Typically "occupied" dominates or logic "hit > miss".
        
        # Simplified:
        for (r, c) in free_cells:
            if (r, c) not in occ_cells: # Only clean free
                 self.log_odds[r, c] += self.l_free
        
        for (r, c) in occ_cells:
            self.log_odds[r, c] += self.l_occ
            
        # Clamp
        np.clip(self.log_odds, self.l_min, self.l_max, out=self.log_odds)

    def save(self, path: str):
        """Save grid to text file (0=free, 1=occ, like input map format)."""
        # Convert log-odds to probabilties
        # p = 1 - 1 / (1 + exp(l))
        # threshold p > 0.5 (l > 0) as occupied
        
        arr = np.where(self.log_odds > 0, 1, 0)
        
        with open(path, 'w') as f:
            for r in range(self.rows):
                line = "".join(str(arr[r, c]) for c in range(self.cols))
                f.write(line + "\n")
