from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"

def find_cleanup_targets() -> List[Tuple[Path, Path]]:
    """
    Returns list of (src, dst) moves to reduce clutter without deleting data.

    - Moves legacy `classification_summary_wide.csv` to outputs/_legacy/
    - Moves v1 ensemble outputs (outputs/ensemble/) to outputs/_archive/
    - Moves older (non-calibrated) ensemble_v2 run folders to outputs/ensemble_v2/_archive/
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    moves: List[Tuple[Path, Path]] = []

    # Legacy wide summary
    wide = OUTPUTS_DIR / "classification_summary_wide.csv"
    if wide.exists():
        moves.append((wide, OUTPUTS_DIR / "_legacy" / f"classification_summary_wide_{ts}.csv"))

    # Ensemble v1 outputs
    ens_v1 = OUTPUTS_DIR / "ensemble"
    if ens_v1.exists() and ens_v1.is_dir():
        moves.append((ens_v1, OUTPUTS_DIR / "_archive" / f"ensemble_v1_{ts}"))

    # Older ensemble_v2 runs (keep calibrated runs by default)
    ens_v2 = OUTPUTS_DIR / "ensemble_v2"
    if ens_v2.exists():
        for child in ens_v2.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if not name.startswith("models-"):
                continue
            if "__cal-" in name:
                continue
            moves.append((child, ens_v2 / "_archive" / f"{name}_{ts}"))

    return moves

def main() -> None:
    parser = argparse.ArgumentParser(description="Move legacy/old outputs into archive folders (no deletion).")
    parser.add_argument("--apply", action="store_true", help="Actually move files (default is dry-run).")
    args = parser.parse_args()

    moves = find_cleanup_targets()
    if not moves:
        print("No cleanup targets found.")
        return

    print("Planned moves:")
    for src, dst in moves:
        print(f"- {src} -> {dst}")

    if not args.apply:
        print("\nDry-run only. Re-run with `--apply` to perform these moves.")
        return

    for src, dst in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

    print("\nDone.")

if __name__ == "__main__":
    main()