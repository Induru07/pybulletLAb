"""
map_picker.py — Select a map from the shared/maps directory.
Supports interactive mode (GUI) and direct mode (CLI / headless).
"""
from __future__ import annotations
from pathlib import Path


def pick_map(maps_dir: str = "shared/maps", default: str | None = None,
             direct: str | None = None) -> str:
    """
    Return the path to a .txt map file.

    Args:
        maps_dir: directory containing .txt map files.
        default:  filename used when the user presses Enter.
        direct:   if provided, resolve this name directly (no interactive prompt).
                  Accepts a full path, a filename, or a bare name like 'warehouse_small'.
    """
    d = Path(maps_dir)
    if not d.exists():
        raise FileNotFoundError(f"Maps directory not found: {d.resolve()}")

    # --- direct / headless mode ---
    if direct is not None:
        # Full path given
        p = Path(direct)
        if p.is_file():
            return str(p.as_posix())
        # Try inside maps_dir
        p = d / direct
        if p.is_file():
            return str(p.as_posix())
        # Try adding .txt
        p = d / (direct + ".txt")
        if p.is_file():
            return str(p.as_posix())
        raise FileNotFoundError(f"Map not found: '{direct}' (searched in {d.resolve()})")

    # --- interactive mode ---
    files = sorted([f for f in d.glob("*.txt") if f.is_file()])
    if not files:
        raise FileNotFoundError(f"No .txt files found in: {d.resolve()}")

    print("\nChoose a map:")
    for i, f in enumerate(files, 1):
        print(f"  {i}) {f.name}")

    if default:
        print(f"\nEnter = default: {default}")

    while True:
        choice = input("\nMap number: ").strip()
        if choice == "" and default:
            return str((d / default).as_posix())
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(files):
                return str(files[idx - 1].as_posix())
        print("Invalid choice. Try again.")
