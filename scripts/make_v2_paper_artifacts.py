"""Generate combined v2 paper figures and summary tables from saved outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v2.plotting import generate_all_figures


def _collect(pattern_root: Path, filename: str) -> pd.DataFrame:
    frames = []
    for path in pattern_root.rglob(filename):
        rel_parent = path.parent.relative_to(pattern_root)
        if "paper_artifacts" in rel_parent.parts:
            continue
        df = pd.read_csv(path)
        if "run" in df.columns:
            df = df.drop(columns=["run"])
        df.insert(0, "run", rel_parent.as_posix())
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create v2 paper tables/figures from output directories.")
    parser.add_argument("--root", type=Path, default=Path("outputs/v2"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/v2/paper_artifacts"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for filename, out_name in [
        ("subject_metrics.csv", "combined_subject_metrics.csv"),
        ("aggregate_metrics.csv", "combined_aggregate_metrics.csv"),
        ("coverage_sweep.csv", "combined_coverage_sweep.csv"),
        ("ablation_summary.csv", "combined_ablation_summary.csv"),
        ("risk_coverage.csv", "risk_coverage.csv"),
        ("reliability_diagram_bins.csv", "reliability_diagram_bins.csv"),
    ]:
        df = _collect(args.root, filename)
        if not df.empty:
            df.to_csv(args.out_dir / out_name, index=False)

    generate_all_figures(args.out_dir)
    print(f"Wrote paper artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
