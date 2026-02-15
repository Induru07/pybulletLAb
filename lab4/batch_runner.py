"""
batch_runner.py — Run N simulation missions automatically and collect results.
Usage:
    python -m lab4.batch_runner --map warehouse_small --nav --goal_random \
        --robots 2 --humans 1 --runs 10 --direct --seed 100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from shared.utils.map_picker import pick_map
from lab4.config import SimConfig
from lab4.simulation import Simulation


def run_batch(base_cfg: SimConfig, n_runs: int, base_seed: int) -> list[dict]:
    """Run n_runs simulations with incrementing seeds. Returns list of result dicts."""
    results = []

    for i in range(n_runs):
        seed = base_seed + i
        cfg = SimConfig(**{**base_cfg.__dict__, "seed": seed, "direct": True})

        print(f"\n{'='*60}")
        print(f"  RUN {i+1}/{n_runs}  (seed={seed})")
        print(f"{'='*60}")

        sim = Simulation(cfg)
        t0 = time.time()
        try:
            result = sim.run()
            wall_time = time.time() - t0
            result["seed"] = seed
            result["run_index"] = i
            result["wall_time"] = wall_time
            result["run_dir"] = str(sim.logger.run_dir)
            results.append(result)

            print(f"  -> success={result['success']}, "
                  f"sim_t={result['sim_time']:.1f}s, "
                  f"dist={result['distance']:.2f}m, "
                  f"pf_err={result['mean_pf_error']:.3f}m, "
                  f"wall={wall_time:.1f}s")
        except Exception as e:
            print(f"  -> FAILED: {e}")
            results.append({
                "success": False, "sim_time": 0, "distance": 0,
                "mean_pf_error": 0, "seed": seed, "run_index": i,
                "error": str(e),
            })
        finally:
            sim.shutdown()

    return results


def print_summary(results: list[dict]):
    """Print aggregate statistics."""
    n = len(results)
    successes = sum(1 for r in results if r.get("success"))
    times = [r["sim_time"] for r in results if r.get("success")]
    dists = [r["distance"] for r in results if r.get("success")]
    pf_errs = [r["mean_pf_error"] for r in results if r.get("success")]

    print(f"\n{'='*60}")
    print(f"  BATCH SUMMARY  ({n} runs)")
    print(f"{'='*60}")
    print(f"  Success rate:    {successes}/{n} ({100*successes/n:.0f}%)")
    if times:
        print(f"  Sim time (avg):  {sum(times)/len(times):.2f}s")
        print(f"  Distance (avg):  {sum(dists)/len(dists):.2f}m")
        print(f"  PF error (avg):  {sum(pf_errs)/len(pf_errs):.4f}m")
        print(f"  Sim time (min):  {min(times):.2f}s")
        print(f"  Sim time (max):  {max(times):.2f}s")
    print()


def main():
    parser = argparse.ArgumentParser(description="Batch AMR Simulation Runner")

    # ── Same args as main.py ──
    parser.add_argument("--map", type=str, default="warehouse_small")
    parser.add_argument("--cell_size", type=float, default=0.5)
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--friction", type=float, default=0.05)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--scan_period", type=float, default=0.4)
    parser.add_argument("--angles_n", type=int, default=36)
    parser.add_argument("--nav", action="store_true")
    parser.add_argument("--goal_rc", type=str, default=None)
    parser.add_argument("--goal_top", action="store_true")
    parser.add_argument("--goal_random", action="store_true")
    parser.add_argument("--goal_margin", type=int, default=1)
    parser.add_argument("--replan_s", type=float, default=2.0)
    parser.add_argument("--emergency_idle_s", type=float, default=5.0)
    parser.add_argument("--emergency_forward_s", type=float, default=1.2)
    parser.add_argument("--emergency_pf_runs", type=int, default=3)
    parser.add_argument("--max_sim_s", type=float, default=60.0)
    parser.add_argument("--v_straight", type=float, default=3.5)
    parser.add_argument("--w_turn", type=float, default=3.5)
    parser.add_argument("--v_curve", type=float, default=2.5)
    parser.add_argument("--w_curve", type=float, default=2.5)
    parser.add_argument("--curve_s", type=float, default=15.0)
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument("--humans", type=int, default=0)
    parser.add_argument("--robots", type=int, default=1)
    parser.add_argument("--slam", action="store_true")

    # ── Batch-specific ──
    parser.add_argument("--runs", type=int, default=5, help="Number of runs")
    parser.add_argument("--seed", type=int, default=42, help="Base seed")
    parser.add_argument("--out", type=str, default="batch_results.json",
                        help="Output JSON file for results")

    args = parser.parse_args()

    # ── Resolve map ──
    map_path = pick_map("shared/maps", direct=args.map)

    # ── Goal mode ──
    goal_mode = "random"
    goal_rc = None
    if args.goal_rc:
        goal_mode = "rc"
        rr, cc = args.goal_rc.split(",")
        goal_rc = (int(rr), int(cc))
    elif args.goal_top:
        goal_mode = "top"

    # ── Build base config ──
    base_cfg = SimConfig(
        map_path=map_path,
        cell_size=args.cell_size,
        hz=args.hz,
        friction=args.friction,
        n_particles=args.n,
        scan_period=args.scan_period,
        n_lidar_rays=args.angles_n,
        nav_enabled=args.nav,
        goal_mode=goal_mode,
        goal_rc=goal_rc,
        goal_margin=args.goal_margin,
        replan_interval=args.replan_s,
        emergency_idle_s=args.emergency_idle_s,
        emergency_forward_s=args.emergency_forward_s,
        emergency_pf_runs=args.emergency_pf_runs,
        autotest=False,
        max_sim_s=args.max_sim_s,
        v_straight=args.v_straight,
        w_turn=args.w_turn,
        v_curve=args.v_curve,
        w_curve=args.w_curve,
        curve_s=args.curve_s,
        direct=True,
        seed=args.seed,
        n_jobs=args.jobs,
        n_humans=args.humans,
        n_robots=args.robots,
        slam_enabled=args.slam,
    )

    # ── Run batch ──
    results = run_batch(base_cfg, args.runs, args.seed)

    # ── Save results ──
    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to: {out_path}")

    # ── Print summary ──
    print_summary(results)


if __name__ == "__main__":
    main()
