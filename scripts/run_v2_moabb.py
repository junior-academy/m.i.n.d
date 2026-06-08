"""Run v2 external MOABB evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v2.config import OUTPUT_DIR
from v2.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run M.I.N.D. v2 MOABB evaluation.")
    parser.add_argument("--datasets", nargs="+", default=["Cho2017", "PhysionetMI"])
    parser.add_argument("--subject-limit", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--out-dir", default=str(OUTPUT_DIR / "moabb_external"))
    args = parser.parse_args()
    run_experiment(
        model="all",
        use_ea=True,
        calibrate=True,
        ensemble="deep",
        ensemble_weights="validation",
        post_ensemble_calibration=True,
        uncertainty_score="max-prob",
        coverage_target=0.60,
        include_moabb=True,
        moabb_datasets=args.datasets,
        moabb_subject_limit=args.subject_limit,
        out_dir=Path(args.out_dir),
        epochs=args.epochs,
    )
    print(f"Wrote MOABB v2 outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
