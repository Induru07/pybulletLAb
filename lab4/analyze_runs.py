"""
analyze_runs.py — Load run logs and generate analysis plots.
Usage:
    python -m lab4.analyze_runs --data_dir shared/data
    python -m lab4.analyze_runs --batch batch_results.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for headless
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not installed — text-only analysis.")


def load_pf_csv(path: Path) -> Dict[str, np.ndarray]:
    """Load pf.csv into dict of arrays."""
    data: Dict[str, list] = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, []).append(float(v))
    return {k: np.array(v) for k, v in data.items()}


def load_odometry_csv(path: Path) -> Dict[str, np.ndarray]:
    """Load odometry.csv into dict of arrays."""
    data: Dict[str, list] = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, []).append(float(v))
    return {k: np.array(v) for k, v in data.items()}


def analyze_single_run(run_dir: Path, save_plots: bool = True):
    """Analyze a single run directory."""
    print(f"\n--- Analyzing: {run_dir.name} ---")

    pf_path = run_dir / "pf.csv"
    odo_path = run_dir / "odometry.csv"

    if pf_path.exists():
        pf = load_pf_csv(pf_path)
        t = pf.get("t", np.array([]))
        err = pf.get("err_xy", np.array([]))
        neff = pf.get("neff", np.array([]))

        if len(err) > 0:
            print(f"  PF Error:  mean={np.mean(err):.4f}m, "
                  f"max={np.max(err):.4f}m, "
                  f"final={err[-1]:.4f}m")
        if len(neff) > 0:
            print(f"  Neff:      mean={np.mean(neff):.1f}, "
                  f"min={np.min(neff):.1f}")

        if HAS_MPL and save_plots and len(t) > 0 and len(err) > 0:
            fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

            axes[0].plot(t, err, "r-", linewidth=0.8, label="PF Error (m)")
            axes[0].set_ylabel("Localization Error (m)")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            if len(neff) == len(t):
                axes[1].plot(t, neff, "b-", linewidth=0.8, label="Neff")
                axes[1].set_ylabel("Effective Particles")
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)

            axes[-1].set_xlabel("Time (s)")
            fig.suptitle(f"PF Analysis — {run_dir.name}")
            fig.tight_layout()
            out = run_dir / "pf_analysis.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"  Plot saved: {out}")

    if odo_path.exists():
        odo = load_odometry_csv(odo_path)
        x_gt = odo.get("x_gt", np.array([]))
        y_gt = odo.get("y_gt", np.array([]))

        if HAS_MPL and save_plots and len(x_gt) > 1:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.plot(x_gt, y_gt, "b-", linewidth=0.8, label="GT Trajectory")

            # Plot PF trajectory if available
            if pf_path.exists():
                pf = load_pf_csv(pf_path)
                x_pf = pf.get("x_pf", np.array([]))
                y_pf = pf.get("y_pf", np.array([]))
                if len(x_pf) > 1:
                    ax.plot(x_pf, y_pf, "r--", linewidth=0.6, alpha=0.7, label="PF Estimate")

            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_aspect("equal")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.suptitle(f"Trajectory — {run_dir.name}")
            fig.tight_layout()
            out = run_dir / "trajectory.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"  Plot saved: {out}")


def analyze_batch(batch_file: Path, save_plots: bool = True):
    """Analyze batch results JSON."""
    with open(batch_file, "r") as f:
        results = json.load(f)

    n = len(results)
    successes = [r for r in results if r.get("success")]
    failures = [r for r in results if not r.get("success")]

    print(f"\n{'='*60}")
    print(f"  BATCH ANALYSIS  ({batch_file.name})")
    print(f"{'='*60}")
    print(f"  Total runs:      {n}")
    print(f"  Successes:       {len(successes)} ({100*len(successes)/n:.0f}%)")
    print(f"  Failures:        {len(failures)}")

    if successes:
        times = [r["sim_time"] for r in successes]
        dists = [r["distance"] for r in successes]
        pf_errs = [r["mean_pf_error"] for r in successes]

        print(f"\n  Simulation Time:")
        print(f"    mean:  {np.mean(times):.2f}s")
        print(f"    std:   {np.std(times):.2f}s")
        print(f"    min:   {np.min(times):.2f}s")
        print(f"    max:   {np.max(times):.2f}s")

        print(f"\n  Distance Traveled:")
        print(f"    mean:  {np.mean(dists):.2f}m")
        print(f"    std:   {np.std(dists):.2f}m")

        print(f"\n  PF Localization Error:")
        print(f"    mean:  {np.mean(pf_errs):.4f}m")
        print(f"    std:   {np.std(pf_errs):.4f}m")
        print(f"    worst: {np.max(pf_errs):.4f}m")

        if HAS_MPL and save_plots:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))

            seeds = [r.get("seed", i) for i, r in enumerate(successes)]

            axes[0].bar(range(len(times)), times, color="steelblue")
            axes[0].set_title("Simulation Time (s)")
            axes[0].set_xlabel("Run")
            axes[0].axhline(np.mean(times), color="red", linestyle="--", linewidth=1)

            axes[1].bar(range(len(dists)), dists, color="seagreen")
            axes[1].set_title("Distance Traveled (m)")
            axes[1].set_xlabel("Run")
            axes[1].axhline(np.mean(dists), color="red", linestyle="--", linewidth=1)

            axes[2].bar(range(len(pf_errs)), pf_errs, color="coral")
            axes[2].set_title("Mean PF Error (m)")
            axes[2].set_xlabel("Run")
            axes[2].axhline(np.mean(pf_errs), color="red", linestyle="--", linewidth=1)

            fig.suptitle(f"Batch Results — {n} runs ({len(successes)} success)")
            fig.tight_layout()
            out = batch_file.parent / "batch_analysis.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"\n  Plot saved: {out}")

    # Also analyze individual run directories if available
    for r in results:
        rd = r.get("run_dir")
        if rd and Path(rd).exists():
            analyze_single_run(Path(rd), save_plots=save_plots)


def find_recent_runs(data_dir: Path, n: int = 5) -> List[Path]:
    """Find the N most recent run directories."""
    runs = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
                  reverse=True)
    return runs[:n]


def main():
    parser = argparse.ArgumentParser(description="Analyze AMR simulation results")
    parser.add_argument("--data_dir", type=str, default="shared/data",
                        help="Directory containing run folders")
    parser.add_argument("--batch", type=str, default=None,
                        help="Batch results JSON file")
    parser.add_argument("--last", type=int, default=5,
                        help="Analyze N most recent runs")
    parser.add_argument("--no_plots", action="store_true",
                        help="Disable plot generation")

    args = parser.parse_args()
    save_plots = not args.no_plots

    if args.batch:
        analyze_batch(Path(args.batch), save_plots=save_plots)
    else:
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"Data directory not found: {data_dir}")
            return

        runs = find_recent_runs(data_dir, args.last)
        if not runs:
            print("No run directories found.")
            return

        print(f"Analyzing {len(runs)} most recent runs from {data_dir}")
        for rd in runs:
            analyze_single_run(rd, save_plots=save_plots)


if __name__ == "__main__":
    main()
