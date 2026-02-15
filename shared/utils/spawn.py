from __future__ import annotations
from typing import List, Tuple

Grid = List[List[int]]  # 1=wall, 0=free


def find_first_open_area_top(grid: Grid, free_block: int = 6, margin: int = 1) -> Tuple[int, int]:
    """
    Cherche depuis le haut (row=0 vers le bas) le premier bloc free_block x free_block rempli de 0.
    Retourne (row, col) du centre du bloc trouvé.

    free_block: taille du bloc libre requis (en cellules). 6 marche bien pour Husky si map x4.
    margin: ignore les bords (souvent murs)
    """
    rows = len(grid)
    cols = len(grid[0])

    # Parcours du haut vers le bas
    for r in range(margin, rows - free_block - margin + 1):
        for c in range(margin, cols - free_block - margin + 1):
            ok = True
            for rr in range(r, r + free_block):
                # early break rapide
                if 1 in grid[rr][c:c + free_block]:
                    ok = False
                    break
            if ok:
                center_r = r + free_block // 2
                center_c = c + free_block // 2
                return center_r, center_c

    raise RuntimeError("Aucune zone libre suffisamment grande trouvée en haut de la map. Diminue free_block ou change la map.")
