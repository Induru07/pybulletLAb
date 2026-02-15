"""
Generate warehouse-style occupancy grid maps.
Usage:
    python -m shared.utils.gen_warehouse_map [--size small|medium|big] [--out PATH]
"""
from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path


def rect(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, val: int = 1):
    """Fill rectangle [r0:r1, c0:c1] with val (0=free, 1=obstacle)."""
    grid[r0:r1, c0:c1] = val


# --------------------------------------------------------------------------- #
#  warehouse_small  (30 x 50)
# --------------------------------------------------------------------------- #
def make_warehouse_small(rows: int = 30, cols: int = 50) -> np.ndarray:
    """
    Small warehouse: outer walls, rack columns with 4-cell aisles,
    cross-aisle in the middle, loading/staging zones in corners.
    """
    g = np.zeros((rows, cols), dtype=int)

    # Outer walls
    g[0, :] = 1; g[-1, :] = 1; g[:, 0] = 1; g[:, -1] = 1

    # Storage racks — vertical columns with wide aisles
    rack_top, rack_bot = 3, rows - 3
    rack_w, aisle_w = 2, 4

    c = 10
    while c + rack_w < cols - 4:
        rect(g, rack_top, c, rack_bot, c + rack_w, 1)
        # Cross gaps every 5 rows for passages
        for r in range(rack_top + 2, rack_bot - 2, 5):
            rect(g, r, c, r + 2, c + rack_w, 0)
        c += rack_w + aisle_w

    # Main cross-aisle (horizontal, middle)
    mid = rows // 2
    rect(g, mid - 1, 1, mid + 2, cols - 1, 0)

    # Loading zone (bottom-left)
    rect(g, rows - 6, 2, rows - 2, 8, 0)

    # Staging zone (top-left)
    rect(g, 2, 2, 6, 8, 0)

    return g


# --------------------------------------------------------------------------- #
#  warehouse_medium  (50 x 80)
# --------------------------------------------------------------------------- #
def make_warehouse_medium(rows: int = 50, cols: int = 80) -> np.ndarray:
    """
    Medium warehouse: office block, charging zone, many rack rows,
    two cross-aisles, chicane passages, scattered pallets.
    """
    g = np.zeros((rows, cols), dtype=int)

    # Outer walls
    g[0, :] = 1; g[-1, :] = 1; g[:, 0] = 1; g[:, -1] = 1

    # Office block (top-left)
    rect(g, 2, 2, 10, 14, 1)

    # Charging zone (bottom-right)
    rect(g, rows - 12, cols - 16, rows - 2, cols - 2, 1)
    # Access corridor to charging
    rect(g, rows - 12, cols - 26, rows - 8, cols - 16, 0)

    # Storage racks area
    rack_top, rack_bot = 5, rows - 14
    rack_left, rack_right = 20, cols - 18
    rack_w, aisle_w = 2, 3
    gap_every = 8

    c = rack_left
    rack_idx = 0
    while c + rack_w < rack_right:
        rect(g, rack_top, c, rack_bot, c + rack_w, 1)
        for r in range(rack_top + 4, rack_bot - 3, gap_every):
            offset = 0 if rack_idx % 2 == 0 else 3
            rect(g, r + offset, c, r + offset + 2, c + rack_w, 0)
        c += rack_w + aisle_w
        rack_idx += 1

    # Two cross-aisles
    mid1 = rows // 3
    mid2 = 2 * rows // 3
    rect(g, mid1 - 1, rack_left - 2, mid1 + 2, rack_right + 2, 0)
    rect(g, mid2 - 1, rack_left - 2, mid2 + 2, rack_right + 2, 0)

    # Chicane (narrow passage left side)
    rect(g, 14, rack_left - 5, 18, rack_left - 1, 1)
    rect(g, 18, rack_left - 8, 22, rack_left - 4, 1)

    # Scattered pallets
    pallets = [(12, 8), (20, 10), (rows - 18, 8), (16, cols - 10), (28, cols - 8)]
    for r, c_p in pallets:
        if 1 <= r < rows - 2 and 1 <= c_p < cols - 2:
            rect(g, r, c_p, r + 2, c_p + 2, 1)

    return g


# --------------------------------------------------------------------------- #
#  warehouse_big  (80 x 120)  — original
# --------------------------------------------------------------------------- #
def make_warehouse_big(rows: int = 80, cols: int = 120) -> np.ndarray:
    """
    Large warehouse with office block, charging zone, many rack rows,
    cross-aisles, chicanes, and scattered pallets.
    """
    g = np.zeros((rows, cols), dtype=int)

    # Outer walls
    g[0, :] = 1; g[-1, :] = 1; g[:, 0] = 1; g[:, -1] = 1

    # Office block (top-left)
    rect(g, 2, 2, 14, 18, 1)

    # Charging/maintenance block (bottom-right)
    rect(g, rows - 16, cols - 22, rows - 2, cols - 2, 1)
    rect(g, rows - 16, cols - 40, rows - 10, cols - 22, 0)

    # Storage racks
    rack_top, rack_bot = 6, rows - 22
    rack_left, rack_right = 28, cols - 28
    aisle_w, rack_w, gap_every = 3, 3, 10

    c = rack_left
    rack_idx = 0
    while c + rack_w < rack_right:
        rect(g, rack_top, c, rack_bot, c + rack_w, 1)
        for r in range(rack_top + 6, rack_bot - 6, gap_every):
            if rack_idx % 2 == 0:
                rect(g, r, c, r + 2, c + rack_w, 0)
            else:
                rect(g, r + 3, c, r + 5, c + rack_w, 0)
        c += rack_w + aisle_w
        rack_idx += 1

    # Cross-aisles
    rect(g, rows // 2 - 2, rack_left - 2, rows // 2 + 2, rack_right + 2, 0)
    rect(g, rows - 30, rack_left - 2, rows - 26, rack_right + 2, 0)

    # Chicanes
    rect(g, 20, rack_left - 6, 26, rack_left - 2, 1)
    rect(g, 26, rack_left - 10, 32, rack_left - 6, 1)
    rect(g, 14, rack_left - 10, 20, rack_left - 6, 1)
    rect(g, 34, rack_right + 2, 40, rack_right + 6, 1)
    rect(g, 40, rack_right + 6, 46, rack_right + 10, 1)

    # Pallets
    pallets = [
        (18, 10), (24, 14), (30, 8),
        (22, cols - 16), (28, cols - 12), (16, cols - 10),
    ]
    for r, c_p in pallets:
        rect(g, r, c_p, r + 2, c_p + 2, 1)

    return g


# --------------------------------------------------------------------------- #
#  I/O
# --------------------------------------------------------------------------- #
def save_txt(grid: np.ndarray, path: str | Path):
    """Save grid as lines of 0/1 characters."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in range(grid.shape[0]):
            f.write("".join(str(x) for x in grid[r]) + "\n")
    print(f"Wrote {p}  ({grid.shape[0]}x{grid.shape[1]})")


BUILDERS = {
    "small": make_warehouse_small,
    "medium": make_warehouse_medium,
    "big": make_warehouse_big,
}


def main():
    parser = argparse.ArgumentParser(description="Generate warehouse maps")
    parser.add_argument("--size", choices=list(BUILDERS.keys()), default=None,
                        help="Which size to generate (omit = all)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path (only with --size)")
    args = parser.parse_args()

    maps_dir = Path(__file__).resolve().parents[1] / "maps"

    if args.size:
        g = BUILDERS[args.size]()
        out = args.out or str(maps_dir / f"warehouse_{args.size}.txt")
        save_txt(g, out)
    else:
        for name, builder in BUILDERS.items():
            g = builder()
            save_txt(g, maps_dir / f"warehouse_{name}.txt")


if __name__ == "__main__":
    main()
