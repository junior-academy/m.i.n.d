"""Held-out T->E selective evaluation for the main research question."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BASE_MODELS,
    COMPARISON_MODEL,
    EPOCH_DIR,
    OPERATING_COVERAGE,
    OUTPUT_DIR,
    SUBJECT_IDS,
    THRESHOLD_GRID,
)
from decode import accuracy, equal_soft_vote, fit_predict_subject


def _load_pair(epoch_dir: Path, subject: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = f"A{subject:02d}T"
    test = f"A{subject:02d}E"
    paths = {
        "X_train": epoch_dir / f"X_{train}.npy",
        "y_train": epoch_dir / f"y_{train}.npy",
        "X_test": epoch_dir / f"X_{test}.npy",
        "y_test": epoch_dir / f"y_{test}.npy",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing preprocessed T/E arrays. Run `python preprocess.py` first.\n"
            + "\n".join(missing)
        )
    return (
        np.load(paths["X_train"]),
        np.load(paths["y_train"]),
        np.load(paths["X_test"]),
        np.load(paths["y_test"]),
    )


def _mask_for_coverage(proba: np.ndarray, coverage: float) -> tuple[np.ndarray, float]:
    conf = np.max(proba, axis=1)
    n_keep = int(np.ceil(float(coverage) * conf.size))
    n_keep = min(max(n_keep, 1), conf.size)
    order = np.argsort(conf)[::-1]
    mask = np.zeros(conf.size, dtype=bool)
    mask[order[:n_keep]] = True
    threshold = float(conf[order[n_keep - 1]])
    return mask, threshold


def _threshold_rows(subject: int, y_true: np.ndarray, name: str, proba: np.ndarray) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    conf = np.max(proba, axis=1)
    pred = np.argmax(proba, axis=1)
    for threshold in THRESHOLD_GRID:
        mask = conf >= float(threshold)
        rows.append(
            {
                "subject": subject,
                "decoder": name,
                "threshold": float(threshold),
                "coverage": float(mask.mean()),
                "risk": float(np.mean(pred[mask] != y_true[mask])) if mask.any() else float("nan"),
                "accuracy": float(np.mean(pred[mask] == y_true[mask])) if mask.any() else float("nan"),
            }
        )
    return rows


def run(epoch_dir: Path = EPOCH_DIR, out_dir: Path = OUTPUT_DIR, operating_coverage: float = OPERATING_COVERAGE) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    subject_rows = []
    curve_rows = []

    for subject in SUBJECT_IDS:
        X_train, y_train, X_test, y_test = _load_pair(epoch_dir, subject)
        pred = fit_predict_subject(
            subject=subject,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_names=BASE_MODELS,
        )
        p_ens = equal_soft_vote(pred.proba)
        p_lda = pred.proba[COMPARISON_MODEL]

        ens_mask, ens_threshold = _mask_for_coverage(p_ens, operating_coverage)
        lda_mask, lda_threshold = _mask_for_coverage(p_lda, operating_coverage)
        ens_acc = accuracy(pred.y_true, p_ens, ens_mask)
        lda_acc = accuracy(pred.y_true, p_lda, lda_mask)

        subject_rows.append(
            {
                "subject": subject,
                "n_train": int(X_train.shape[0]),
                "n_test": int(X_test.shape[0]),
                "coverage": float(operating_coverage),
                "ensemble_threshold_at_coverage": ens_threshold,
                "lda_threshold_at_coverage": lda_threshold,
                "ensemble_acc_matched_coverage": ens_acc,
                "lda_acc_matched_coverage": lda_acc,
                "delta_ensemble_minus_lda": ens_acc - lda_acc,
                "ensemble_acc_all": accuracy(pred.y_true, p_ens),
                "lda_acc_all": accuracy(pred.y_true, p_lda),
            }
        )
        curve_rows.extend(_threshold_rows(subject, pred.y_true, "ensemble", p_ens))
        curve_rows.extend(_threshold_rows(subject, pred.y_true, "LDA", p_lda))

    subject_df = pd.DataFrame(subject_rows).sort_values("subject")
    curve_df = pd.DataFrame(curve_rows).sort_values(["decoder", "subject", "threshold"])
    subject_df.to_csv(out_dir / "subject_results.csv", index=False)
    curve_df.to_csv(out_dir / "risk_coverage.csv", index=False)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "evaluation": "fit feature extractors and decoders on A0xT, score once on A0xE",
                "base_models": BASE_MODELS,
                "ensemble": "equal soft vote",
                "comparison": COMPARISON_MODEL,
                "operating_coverage": operating_coverage,
                "threshold_grid": THRESHOLD_GRID,
            },
            indent=2,
        )
    )
    return out_dir / "subject_results.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run held-out-session selective evaluation.")
    parser.add_argument("--epoch-dir", type=Path, default=EPOCH_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--coverage", type=float, default=OPERATING_COVERAGE)
    args = parser.parse_args()
    result = run(epoch_dir=args.epoch_dir, out_dir=args.out_dir, operating_coverage=args.coverage)
    print(f"Wrote {result}")


if __name__ == "__main__":
    main()
