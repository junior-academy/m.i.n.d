"""Held-out T->E selective evaluation for the main research question."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from adaptation import adapt_train_test
from config import (
    ALIGNMENT_EPS,
    BASE_MODELS,
    COMPARISON_MODEL,
    EPOCH_DIR,
    OPERATING_COVERAGE,
    OUTPUT_DIR,
    SESSION_ADAPTATION,
    SUBJECT_IDS,
    THRESHOLD_GRID,
)
from decode import accuracy, equal_soft_vote, fit_predict_subject
from reliability import metrics_row, reliability_rows, risk_coverage_rows, write_reliability_diagrams


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


def run(
    epoch_dir: Path = EPOCH_DIR,
    out_dir: Path = OUTPUT_DIR,
    operating_coverage: float = OPERATING_COVERAGE,
    adaptation: str = SESSION_ADAPTATION,
    include_eegnet: bool = False,
    eegnet_epochs: int = 60,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    subject_rows = []
    curve_rows = []
    metric_rows = []
    diagram_rows = []

    for subject in SUBJECT_IDS:
        X_train, y_train, X_test, y_test = _load_pair(epoch_dir, subject)
        X_train_eval, X_test_eval = adapt_train_test(
            X_train,
            X_test,
            method=adaptation,
            eps=ALIGNMENT_EPS,
        )
        pred = fit_predict_subject(
            subject=subject,
            X_train=X_train_eval,
            y_train=y_train,
            X_test=X_test_eval,
            y_test=y_test,
            model_names=BASE_MODELS,
        )
        p_ens = equal_soft_vote(pred.proba)
        p_lda = pred.proba[COMPARISON_MODEL]
        extra_probas: dict[str, np.ndarray] = {}
        if include_eegnet:
            from eegnet import EEGNetConfig, fit_predict_eegnet

            eeg_y_true, p_eegnet = fit_predict_eegnet(
                X_train=X_train_eval,
                y_train=y_train,
                X_test=X_test_eval,
                y_test=y_test,
                config=EEGNetConfig(epochs=eegnet_epochs),
            )
            if not np.array_equal(eeg_y_true, pred.y_true):
                raise RuntimeError("EEGNet label encoding does not match the classical decoder encoding")
            extra_probas["EEGNet"] = p_eegnet

        ens_mask, ens_threshold = _mask_for_coverage(p_ens, operating_coverage)
        lda_mask, lda_threshold = _mask_for_coverage(p_lda, operating_coverage)
        ens_acc = accuracy(pred.y_true, p_ens, ens_mask)
        lda_acc = accuracy(pred.y_true, p_lda, lda_mask)
        extra_subject_cols = {}
        for extra_name, extra_proba in extra_probas.items():
            extra_mask, extra_threshold = _mask_for_coverage(extra_proba, operating_coverage)
            extra_acc = accuracy(pred.y_true, extra_proba, extra_mask)
            key = extra_name.lower()
            extra_subject_cols[f"{key}_threshold_at_coverage"] = extra_threshold
            extra_subject_cols[f"{key}_acc_matched_coverage"] = extra_acc
            extra_subject_cols[f"delta_ensemble_minus_{key}"] = ens_acc - extra_acc
            extra_subject_cols[f"{key}_acc_all"] = accuracy(pred.y_true, extra_proba)

        subject_rows.append(
            {
                "subject": subject,
                "n_train": int(X_train.shape[0]),
                "n_test": int(X_test.shape[0]),
                "adaptation": adaptation,
                "coverage": float(operating_coverage),
                "ensemble_threshold_at_coverage": ens_threshold,
                "lda_threshold_at_coverage": lda_threshold,
                "ensemble_acc_matched_coverage": ens_acc,
                "lda_acc_matched_coverage": lda_acc,
                "delta_ensemble_minus_lda": ens_acc - lda_acc,
                "ensemble_acc_all": accuracy(pred.y_true, p_ens),
                "lda_acc_all": accuracy(pred.y_true, p_lda),
                **extra_subject_cols,
            }
        )
        decoder_outputs = {"ensemble": p_ens, "LDA": p_lda, **extra_probas}
        for name, proba in decoder_outputs.items():
            curve_rows.extend(
                risk_coverage_rows(
                    dataset="BCI_IV_2a",
                    subject=subject,
                    decoder=name,
                    y_true=pred.y_true,
                    proba=proba,
                    thresholds=THRESHOLD_GRID,
                )
            )
            metric_rows.append(
                metrics_row(
                    dataset="BCI_IV_2a",
                    subject=subject,
                    decoder=name,
                    y_true=pred.y_true,
                    proba=proba,
                )
            )
            diagram_rows.extend(
                reliability_rows(
                    dataset="BCI_IV_2a",
                    subject=subject,
                    decoder=name,
                    y_true=pred.y_true,
                    proba=proba,
                )
            )

    subject_df = pd.DataFrame(subject_rows).sort_values("subject")
    curve_df = pd.DataFrame(curve_rows).sort_values(["decoder", "subject", "threshold"])
    metrics_df = pd.DataFrame(metric_rows).sort_values(["decoder", "subject"])
    reliability_df = pd.DataFrame(diagram_rows).sort_values(["decoder", "subject", "bin"])
    subject_df.to_csv(out_dir / "subject_results.csv", index=False)
    curve_df.to_csv(out_dir / "risk_coverage.csv", index=False)
    metrics_df.to_csv(out_dir / "reliability_metrics.csv", index=False)
    reliability_df.to_csv(out_dir / "reliability_diagram_bins.csv", index=False)
    write_reliability_diagrams(reliability_df, out_dir)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "evaluation": "fit feature extractors and decoders on A0xT, score once on A0xE",
                "session_adaptation": adaptation,
                "adaptation_label_use": "A0xE labels are used only for final scoring",
                "alignment_eps": ALIGNMENT_EPS,
                "base_models": BASE_MODELS,
                "ensemble": "equal soft vote",
                "comparison": COMPARISON_MODEL,
                "eegnet": "included as optional comparison" if include_eegnet else "not run",
                "eegnet_epochs": eegnet_epochs if include_eegnet else None,
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
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--coverage", type=float, default=OPERATING_COVERAGE)
    parser.add_argument("--adaptation", default=SESSION_ADAPTATION, choices=("euclidean", "euclidean_alignment", "none"))
    parser.add_argument("--include-eegnet", action="store_true", help="Add optional EEGNet deep-learning comparison.")
    parser.add_argument("--eegnet-epochs", type=int, default=60)
    args = parser.parse_args()
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = OUTPUT_DIR if args.adaptation != "none" else OUTPUT_DIR.parent / "heldout_session_no_adaptation"
    result = run(
        epoch_dir=args.epoch_dir,
        out_dir=out_dir,
        operating_coverage=args.coverage,
        adaptation=args.adaptation,
        include_eegnet=args.include_eegnet,
        eegnet_epochs=args.eegnet_epochs,
    )
    print(f"Wrote {result}")


if __name__ == "__main__":
    main()
