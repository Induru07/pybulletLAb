from __future__ import annotations
from pathlib import Path

def upscale_map(in_path: Path, out_path: Path, k: int) -> None:
    lines = in_path.read_text(encoding="utf-8").splitlines()
    rows = []
    for raw in lines:
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        # support "0101" ou "0 1 0 1"
        if " " in s or "\t" in s:
            row = [ch for ch in s.replace("\t", " ").split() if ch in ("0", "1")]
        else:
            row = [ch for ch in s if ch in ("0", "1")]
        if row:
            rows.append(row)

    if not rows:
        raise ValueError("Map vide/illisible")

    w = len(rows[0])
    if any(len(r) != w for r in rows):
        raise ValueError("Map non rectangulaire")

    out_lines = []
    for r in rows:
        # chaque char répété k fois horizontalement
        expanded_row = "".join(ch * k for ch in r)
        # chaque ligne répétée k fois verticalement
        out_lines.extend([expanded_row] * k)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"OK: {in_path} -> {out_path} (x{k})")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--k", type=int, required=True, choices=[2,3,4,5,6,8,10])
    args = p.parse_args()

    upscale_map(Path(args.in_path), Path(args.out_path), args.k)
