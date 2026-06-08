"""Reproduce the v2 headline BCI IV 2a comparison runs."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v2.config import OUTPUT_DIR
from v2.runner import run_experiment


def main() -> None:
    base = OUTPUT_DIR / "headline_bci_iv_2a"
    configs = [
        ("classical_lda", dict(model="lda", calibrate=False, ensemble="none", ensemble_weights="equal", post=False)),
        ("deep_equal", dict(model="all", calibrate=True, ensemble="deep", ensemble_weights="equal", post=False)),
        ("deep_validation_weighted", dict(model="all", calibrate=True, ensemble="deep", ensemble_weights="validation", post=False)),
        ("deep_equal_postcal", dict(model="all", calibrate=True, ensemble="deep", ensemble_weights="equal", post=True)),
        ("hybrid_equal", dict(model="all", calibrate=True, ensemble="hybrid", ensemble_weights="equal", post=False)),
    ]
    for name, cfg in configs:
        out_dir = base / name
        run_experiment(
            model=cfg["model"],
            use_ea=True,
            calibrate=cfg["calibrate"],
            ensemble=cfg["ensemble"],
            ensemble_weights=cfg["ensemble_weights"],
            post_ensemble_calibration=cfg["post"],
            uncertainty_score="max-prob",
            coverage_target=0.60,
            include_moabb=False,
            moabb_datasets=[],
            moabb_subject_limit=None,
            include_bci_iiia=False,
            out_dir=out_dir,
        )
    print(f"Wrote headline runs under {base}")


if __name__ == "__main__":
    main()
