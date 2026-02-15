from __future__ import annotations

from pathlib import Path
import csv
import math
from typing import Dict, List, Optional, Tuple

# Optionnel: graphique (si matplotlib installé)
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def latest_run_dir(data_dir: Path) -> Path:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable: {data_dir}")
    runs = sorted([p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("run_")])
    if not runs:
        raise FileNotFoundError(f"Aucune run trouvée dans: {data_dir}")
    return runs[-1]


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def get_float(row: Dict[str, str], key: str) -> Optional[float]:
    v = row.get(key, "")
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def analyze_convergence(
    pf_rows: List[Dict[str, str]],
    pos_thresh: float = 0.50,   # mètres
    streak_needed: int = 5      # nb de scans consécutifs
) -> Tuple[Optional[float], List[float], List[float], List[float]]:
    """
    Retourne:
      - t_converge (ou None)
      - t_list
      - err_xy_list
      - err_th_list (si possible, sinon NaN)
    """
    t_list: List[float] = []
    err_xy_list: List[float] = []
    err_th_list: List[float] = []

    consec = 0
    t_converge: Optional[float] = None

    for row in pf_rows:
        t = get_float(row, "t")
        if t is None:
            continue

        # Si err_xy existe déjà dans le CSV
        err_xy = get_float(row, "err_xy")

        # Sinon on le recalcule via pf vs gt
        if err_xy is None:
            x_pf = get_float(row, "x_pf")
            y_pf = get_float(row, "y_pf")
            x_gt = get_float(row, "x_gt")
            y_gt = get_float(row, "y_gt")
            if None in (x_pf, y_pf, x_gt, y_gt):
                continue
            err_xy = math.sqrt((x_pf - x_gt) ** 2 + (y_pf - y_gt) ** 2)

        # Angle error (optionnel)
        th_pf = get_float(row, "th_pf")
        th_gt = get_float(row, "th_gt")
        if None in (th_pf, th_gt):
            err_th = float("nan")
        else:
            err_th = abs(wrap_angle(th_pf - th_gt))

        t_list.append(t)
        err_xy_list.append(float(err_xy))
        err_th_list.append(float(err_th))

        # Convergence rule: err < thresh streak_needed fois
        if err_xy < pos_thresh:
            consec += 1
        else:
            consec = 0

        if t_converge is None and consec >= streak_needed:
            t_converge = t

    return t_converge, t_list, err_xy_list, err_th_list


def rmse(values: List[float]) -> float:
    vals = [v for v in values if not (math.isnan(v) or math.isinf(v))]
    if not vals:
        return float("nan")
    return math.sqrt(sum(v * v for v in vals) / len(vals))


def main():
    root = Path(__file__).resolve().parents[2]  # PyBulletLabs/
    data_dir = root / "shared" / "data"

    run_dir = latest_run_dir(data_dir)
    pf_path = run_dir / "pf.csv"

    rows = read_csv_dicts(pf_path)

    # Tu peux ajuster ces paramètres
    POS_THRESH = 0.50      # m
    STREAK = 5             # scans consécutifs

    t_conv, t_list, err_xy, err_th = analyze_convergence(rows, POS_THRESH, STREAK)

    print("\n=== Analyse Convergence Particle Filter ===")
    print("Run:", run_dir.name)
    print("Fichier:", pf_path)
    print(f"Nb de scans (lignes pf.csv): {len(err_xy)}")
    if err_xy:
        print(f"Erreur finale: {err_xy[-1]:.3f} m")
        print(f"Erreur min/max: {min(err_xy):.3f} / {max(err_xy):.3f} m")
        print(f"RMSE position: {rmse(err_xy):.3f} m")
    if t_conv is None:
        print(f"Convergence: NON atteinte (seuil={POS_THRESH}m sur {STREAK} scans)")
    else:
        # temps depuis le premier scan
        t0 = t_list[0] if t_list else t_conv
        print(f"Convergence: OUI")
        print(f" - seuil: {POS_THRESH} m")
        print(f" - streak: {STREAK} scans")
        print(f" - temps de convergence (absolu): t={t_conv:.2f}s")
        print(f" - temps depuis 1er scan: {(t_conv - t0):.2f}s")

    # Graphique optionnel
    if HAS_MPL and t_list:
        plt.figure()
        plt.plot(t_list, err_xy)
        plt.axhline(POS_THRESH, linestyle="--")
        if t_conv is not None:
            plt.axvline(t_conv, linestyle="--")
        plt.xlabel("t (s)")
        plt.ylabel("Erreur position err_xy (m)")
        plt.title(f"Convergence PF - {run_dir.name}")
        plt.show()
    else:
        if not HAS_MPL:
            print("\n(Info) matplotlib non installé -> pas de graphique.")
            print("Installe-le avec: python -m pip install matplotlib")


if __name__ == "__main__":
    main()
