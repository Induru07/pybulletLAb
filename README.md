# AMR Warehouse Simulation — PyBulletLabs (Lab 4)

Autonomous Mobile Robot (AMR) warehouse simulation built with **PyBullet**.  
Demonstrates planning, localization, SLAM, multi-robot coordination, and human-aware navigation in warehouse-like environments.

---

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [CLI Parameters](#cli-parameters)
4. [Scenarios](#scenarios)
5. [Maps](#maps)
6. [Architecture](#architecture)
7. [Outputs & Logs](#outputs--logs)
8. [Batch Testing](#batch-testing)

---

## Installation

```bash
# 1. Clone / extract the project
cd PyBulletLabs

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install numpy pybullet matplotlib
```

**Requirements:** Python 3.10+, NumPy, PyBullet, Matplotlib (for analysis).

---

## Quick Start

```bash
cd PyBulletLabs

# Teleop mode (keyboard control, GUI)
python -m lab4.main

# Autonomous navigation to a random goal (GUI)
python -m lab4.main --map warehouse_small --nav --goal_random

# Headless mode — fast, no GUI
python -m lab4.main --map warehouse_small --nav --goal_random --direct --seed 42

# Multi-robot + humans
python -m lab4.main --map warehouse_small --nav --goal_random --robots 3 --humans 2

# Multi-job (pick & drop missions)
python -m lab4.main --map warehouse_medium --nav --jobs 5 --robots 2 --humans 1

# SLAM mode (build map from scratch)
python -m lab4.main --map warehouse_small --nav --goal_random --slam --direct
```

---

## CLI Parameters

### World
| Parameter       | Default | Description                      |
|-----------------|---------|----------------------------------|
| `--map`         | (interactive) | Map file name, filename, or path |
| `--cell_size`   | 0.5     | Grid cell size (meters)          |
| `--hz`          | 50.0    | Physics simulation frequency     |
| `--friction`    | 0.05    | Ground friction coefficient      |

### Particle Filter
| Parameter       | Default | Description                      |
|-----------------|---------|----------------------------------|
| `--n`           | 500     | Number of particles              |
| `--scan_period` | 0.4     | Lidar scan period (seconds)      |
| `--angles_n`    | 16      | Number of lidar rays             |

### Navigation
| Parameter       | Default | Description                      |
|-----------------|---------|----------------------------------|
| `--nav`         | false   | Enable autonomous navigation     |
| `--goal_rc`     | —       | Goal as `row,col`                |
| `--goal_top`    | false   | Goal at top of map               |
| `--goal_random` | false   | Random goal (loops)              |
| `--goal_margin` | 1       | Margin from walls for goal       |
| `--replan_s`    | 2.0     | Replan interval (seconds)        |

### Speeds
| Parameter       | Default | Description                      |
|-----------------|---------|----------------------------------|
| `--v_straight`  | 3.5     | Straight-line velocity (m/s)     |
| `--w_turn`      | 3.5     | Turning angular velocity (rad/s) |
| `--v_curve`     | 2.5     | Curve linear velocity (m/s)      |
| `--w_curve`     | 2.5     | Curve angular velocity (rad/s)   |

### Emergency Recovery
| Parameter             | Default | Description                      |
|-----------------------|---------|----------------------------------|
| `--emergency_idle_s`  | 5.0     | Idle before emergency (seconds)  |
| `--emergency_forward_s`| 1.2    | Forward burst duration (seconds) |
| `--emergency_pf_runs` | 3       | PF re-init attempts              |

### Advanced Features
| Parameter   | Default | Description                          |
|-------------|---------|--------------------------------------|
| `--jobs`    | 0       | Number of pick/drop jobs (Phase 3)   |
| `--humans`  | 0       | Number of simulated humans (Phase 4) |
| `--robots`  | 1       | Number of robots (Phase 5)           |
| `--slam`    | false   | Enable SLAM mapping (Phase 6)        |

### Mode / Display
| Parameter    | Default | Description                          |
|--------------|---------|--------------------------------------|
| `--direct`   | false   | Headless mode (no GUI)               |
| `--seed`     | None    | Random seed for reproducibility      |
| `--autotest` | false   | Autotest mode                        |
| `--max_sim_s`| 60.0    | Maximum simulation time (seconds)    |

---

## Scenarios

### Company 1 — Replace Human-Driven Vehicles
```bash
python -m lab4.main --map warehouse_medium --nav --jobs 5 --robots 2 --direct --seed 1
```
Multiple AMRs autonomously execute pick-and-drop jobs without any humans present.

### Company 2 — Robots with Human Awareness
```bash
python -m lab4.main --map warehouse_medium --nav --jobs 3 --robots 2 --humans 3 --direct --seed 1
```
Robots navigate while detecting and avoiding simulated pedestrians (stop/slow/reroute).

---

## Maps

Available warehouse maps in `shared/maps/`:

| Map | Size | Description |
|-----|------|-------------|
| `warehouse_small.txt`   | 30×50 | Small warehouse with racks and aisles |
| `warehouse_medium.txt`  | 50×80 | Medium warehouse with office, charging zone, chicanes |
| `warehouse_big.txt`     | 80×120 | Large warehouse with multiple zones |
| `maze_lab4.txt`         | —      | Lab maze for testing |
| `maze_realistic.txt`    | —      | Realistic maze |
| `maze_realistic_x4.txt` | —      | Large realistic maze (4x scale) |

Generate new maps:
```bash
python -m shared.utils.gen_warehouse_map --size medium --out shared/maps/my_warehouse.txt
```

---

## Architecture

```
lab4/
├── main.py             # CLI entry point, argument parsing
├── config.py           # SimConfig dataclass (all parameters)
├── simulation.py       # Simulation engine (world, agents, humans, loop)
├── agent.py            # RobotAgent (PF + Navigator + JobManager + SLAM)
├── robot.py            # HuskyRobot (PyBullet URDF, differential drive)
├── navigator.py        # A* path planning + pure-pursuit following
├── planner_astar.py    # A* algorithm with grid inflation
├── control.py          # Keyboard teleop controller
├── odometry.py         # Differential-drive odometry
├── job_manager.py      # Multi-job pick/drop queue manager
├── human.py            # Simulated human pedestrian agent
├── slam.py             # Occupancy grid SLAM (log-odds + Bresenham)
├── world.py            # World coordinate conversions

shared/
├── maps/               # .txt grid maps (0=free, 1=wall)
├── data/               # Run logs (timestamped folders)
└── utils/
    ├── grid_map.py         # Grid loader + simulated lidar
    ├── maze_builder.py     # PyBullet wall builder from grid
    ├── particle_filter.py  # Monte Carlo Localization (MCL)
    ├── spawn.py            # Spawn point finder
    ├── logger.py           # CSV/JSON run logger
    ├── map_picker.py       # Interactive/direct map selector
    └── gen_warehouse_map.py# Warehouse map generator
```

### Control Pipeline
```
Odometry → PF Predict → Lidar Scan → PF Update → PF Estimate
    → A* Plan → Waypoint Follower → (v, w) → Husky Differential Drive
```

---

## Outputs & Logs

Each run creates a timestamped folder under `shared/data/run_YYYYMMDD_HHMMSS/`:

- **`pf.csv`** — Particle filter: `t, x_pf, y_pf, th_pf, x_gt, y_gt, th_gt, err_xy, neff, resample_count`
- **`odometry.csv`** — Odometry: `t, x_odo, y_odo, th_odo, x_gt, y_gt, th_gt, v_cmd, w_cmd`
- **`summary.json`** — Mission summary: success, time, distance, PF error, GT usage %

Console output at end of run:
```
=== RESULT ===
  success:       True
  sim_time:      42.30s
  distance:      18.75m
  mean_pf_error: 0.152m
  gt_usage:      3.2%
```

---

## Batch Testing

Run multiple missions automatically and generate analysis:

```bash
# Run 10 missions on warehouse_small with 2 robots + 1 human
python -m lab4.batch_runner --map warehouse_small --nav --goal_random \
    --robots 2 --humans 1 --runs 10 --direct --seed 100

# Generate plots from collected data
python -m lab4.analyze_runs --data_dir shared/data
```

See `lab4/batch_runner.py` and `lab4/analyze_runs.py` for details.

---

## License

Academic project — ESIEA Lab 4 (AMR Warehouse Automation).
