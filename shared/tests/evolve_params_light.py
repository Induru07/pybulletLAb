import time
import random
import csv
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "shared" / "data"
PY = "python"


def newest_run_dir(before_names: set[str]) -> Path:
    after = sorted([p for p in DATA_DIR.iterdir() if p.is_dir() and p.name.startswith("run_")])
    # idéal: trouver le nouveau dossier créé
    for p in reversed(after):
        if p.name not in before_names:
            return p
    # fallback: dernier run
    if not after:
        raise RuntimeError("Aucune run dans shared/data")
    return after[-1]


def score_from_pf_csv(run_dir: Path, threshold: float = 0.5, k_consec: int = 5) -> dict:
    pf_path = run_dir / "pf.csv"
    if not pf_path.exists():
        return {"ok": False, "fitness": 1e9, "reason": "pf.csv missing"}

    times = []
    errs = []

    with pf_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "t" not in row or "err_xy" not in row:
                return {"ok": False, "fitness": 1e9, "reason": "columns missing"}
            try:
                t = float(row["t"])
                e = float(row["err_xy"])
            except ValueError:
                continue
            times.append(t)
            errs.append(e)

    if not errs:
        return {"ok": False, "fitness": 1e9, "reason": "no rows"}

    # RMSE
    rmse = (sum(e * e for e in errs) / len(errs)) ** 0.5
    final_err = errs[-1]
    min_err = min(errs)

    # convergence: threshold pendant k_consec scans
    conv_t = None
    consec = 0
    for t, e in zip(times, errs):
        if e <= threshold:
            consec += 1
            if consec >= k_consec:
                conv_t = t
                break
        else:
            consec = 0

    converged = conv_t is not None

    return {
        "ok": True,
        "rmse": rmse,
        "final_err": final_err,
        "min_err": min_err,
        "converged": converged,
        "conv_t": conv_t,
    }


def run_one(cfg: dict, threshold: float = 0.5, k_consec: int = 5) -> tuple[float, str, Path]:
    before = {p.name for p in DATA_DIR.iterdir() if p.is_dir()}

    cmd = [
        PY, "-m", "lab4.main",
        "--autotest",
        "--direct",
        "--map", "shared/maps/maze_realistic.txt",
        "--max_sim_s", str(cfg["max_sim_s"]),
        "--curve_s", str(cfg["curve_s"]),
        "--scan_period", str(cfg["scan_period"]),
        "--angles_n", str(cfg["angles_n"]),
        "--n", str(cfg["n"]),
        "--friction", str(cfg["friction"]),
        "--v_straight", str(cfg["v_straight"]),
        "--w_turn", str(cfg["w_turn"]),
        "--v_curve", str(cfg["v_curve"]),
        "--w_curve", str(cfg["w_curve"]),
        "--seed", str(cfg["seed"]),
        "--hz", str(cfg["hz"]),
    ]

    t0 = time.time()
    subprocess.run(cmd, cwd=str(ROOT), check=True, capture_output=True, text=True)
    wall = time.time() - t0

    run_dir = newest_run_dir(before)
    stats = score_from_pf_csv(run_dir, threshold=threshold, k_consec=k_consec)

    if not stats["ok"]:
        fitness = 1e9
        summary = f"FAILED {stats.get('reason')} wall={wall:.2f}s"
        return fitness, summary, run_dir

    rmse = stats["rmse"]
    final_err = stats["final_err"]
    converged = stats["converged"]
    conv_t = stats["conv_t"]

    # Fitness: plus petit = meilleur
    # - pénalité énorme si pas convergé
    # - pénalité temps (wall + convergence tardive)
    fitness = (
        rmse
        + 0.7 * final_err
        + (60.0 if not converged else 0.0)
        + 0.15 * wall
        + (0.2 * (conv_t - stats["conv_t"]) if False else 0.0)  # noop, gardé pour clarté
    )

    # Option: pénaliser convergence lente si converged
    if converged:
        # temps relatif: on soustrait le début de la run
        t_start = stats["conv_t"]  # déjà absolu; on simplifie en pénalisant légèrement
        fitness += 0.05 * (t_start % 1000)

    summary = (
        f"fit={fitness:.2f} rmse={rmse:.2f} final={final_err:.2f} "
        f"conv={converged} wall={wall:.2f}s K={cfg['angles_n']} N={cfg['n']} scan={cfg['scan_period']}"
    )
    return fitness, summary, run_dir


def random_cfg(seed: int) -> dict:
    return {
        "seed": seed,
        "max_sim_s": 45.0,
        "curve_s": random.choice([10.0, 12.0, 15.0]),         # rognable
        "angles_n": random.choice([ 20, 24, 32]),
        "n": random.choice([140, 180, 200]),
        "friction": random.choice([0.6, 0.8, 0.9, 1.0, 1.2]),
        "v_straight": random.choice([1.0, 1.3, 1.6]),
        "w_turn": random.choice([0.9, 1.2, 1.5, 1.8]),
        "v_curve": random.choice([0.7, 1.0, 1.2]),
        "w_curve": random.choice([0.5, 0.8, 1.1]),
        "hz": random.choice([90.0, 120.0]),                   # sim speed
        "scan_period": random.choice([0.6, 0.8, 0.9, 1.0]),

    }


def mutate(cfg: dict) -> dict:
    c = cfg.copy()

    # mutations petites autour des valeurs actuelles
    def jitter(x, lo, hi, step):
        x2 = x + random.choice([-step, 0.0, step])
        return max(lo, min(hi, x2))

    # on change 1 param la plupart du temps
    if random.random() < 0.7:
        k = random.choice(["scan_period", "friction", "v_straight", "w_turn", "v_curve", "w_curve"])
    else:
        k = random.choice(["angles_n", "n", "hz"])

    if k == "scan_period":
        c[k] = jitter(float(c[k]), 0.5, 1.1, 0.1)
    elif k == "friction":
        c[k] = jitter(float(c[k]), 0.6, 1.3, 0.1)
    elif k == "v_straight":
        c[k] = jitter(float(c[k]), 1.0, 1.8, 0.1)
    elif k == "w_turn":
        c[k] = jitter(float(c[k]), 0.9, 2.0, 0.1)
    elif k == "v_curve":
        c[k] = jitter(float(c[k]), 0.6, 1.3, 0.1)
    elif k == "w_curve":
        c[k] = jitter(float(c[k]), 0.5, 1.3, 0.1)
    elif k == "angles_n":
        c[k] = random.choice([16, 20, 24])
    elif k == "n":
        c[k] = random.choice([140, 180, 200])
    elif k == "hz":
        c[k] = random.choice([90.0, 120.0])

    c["seed"] = random.randint(0, 10_000_000)
    return c



def evolve(budget_s: float = 3200.0, pop: int = 8, elite: int = 3):
    random.seed(0)
    start = time.time()
    population = [random_cfg(1000 + i) for i in range(pop)]
    best = None

    gen = 0
    while (time.time() - start) < budget_s:
        gen += 1
        print(f"\n=== GEN {gen} ===")
        scored = []

        for i, cfg in enumerate(population):
            if (time.time() - start) >= budget_s:
                break
            try:
                fit, summary, run_dir = run_one(cfg)
                print(f"{i+1}/{len(population)} {summary} run={run_dir.name}")
                scored.append((fit, cfg))
                if best is None or fit < best[0]:
                    best = (fit, cfg)
            except subprocess.CalledProcessError as e:
                print("Run failed:", e)

        scored.sort(key=lambda x: x[0])
        elites = [cfg for _, cfg in scored[:elite]] if scored else []
        if not elites:
            population = [random_cfg(2000 + i) for i in range(pop)]
            continue

        population = elites[:]
        while len(population) < pop:
            if random.random() < 0.10:   # 10% exploration
                population.append(random_cfg(random.randint(0, 10_000_000)))
            else:
                population.append(mutate(random.choice(elites)))


    print("\n=== BEST ===")
    if best:
        print("fitness =", best[0])
        print("cfg =", best[1])


if __name__ == "__main__":
    evolve(budget_s=1800.0, pop=8, elite=3)
