from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import csv
from typing import Iterable, Sequence


@dataclass
class RunLogger:
    run_dir: Path

    @staticmethod
    def create(base_dir: str | Path) -> "RunLogger":
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        run_dir = base / stamp
        run_dir.mkdir(parents=True, exist_ok=True)
        return RunLogger(run_dir=run_dir)

    def write_row(self, filename: str, header: Sequence[str], row: Sequence[object]) -> None:
        path = self.run_dir / filename
        write_header = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(list(header))
            w.writerow(list(row))
