"""External MOABB validation for the M.I.N.D. reliability claim.

This script intentionally reuses the same calibrated LDA/SVM/FBCSP decoder used
for the held-out BCI IV 2a run. The goal is not to chase a new state of the art;
it is to ask whether selective, calibrated probability voting improves
reliability on additional motor-imagery datasets.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from adaptation import adapt_train_test
from config import (
    ALIGNMENT_EPS,
    BASE_MODELS,
    COMPARISON_MODEL,
    OPERATING_COVERAGE,
    OUTPUT_DIR,
    SESSION_ADAPTATION,
    THRESHOLD_GRID,
)
from decode import accuracy, equal_soft_vote, fit_predict_subject
from reliability import metrics_row, reliability_rows, risk_coverage_rows, write_reliability_diagrams

DEFAULT_DATASETS = ("Cho2017", "PhysionetMI")


def _prepare_external_data_env() -> None:
    base = Path(__file__).resolve().parent
    fake_home = base / ".mne_home"
    data_dir = base / "data" / "moabb"
    (fake_home / ".mne").mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(fake_home))
    os.environ.setdefault("MNE_DATA", str(data_dir))
    os.environ.setdefault("MNE_LOGGING_LEVEL", "WARNING")


def _import_moabb():
    _prepare_external_data_env()
    try:
        import moabb
        from moabb import datasets as moabb_datasets
        from moabb.paradigms import LeftRightImagery
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MOABB is required for external validation. Install it with "
            "`python3 -m pip install moabb` or install the updated requirements."
        ) from exc
    moabb.set_log_level("warning")
    return moabb_datasets, LeftRightImagery


def _dataset_instance(name: str):
    moabb_datasets, _ = _import_moabb()
    try:
        cls = getattr(moabb_datasets, name)
    except AttributeError as exc:
        available = [item for item in dir(moabb_datasets) if item[:1].isupper()]
        raise ValueError(f"Unknown MOABB dataset {name!r}. Available examples: {available[:25]}") from exc
    return cls()


def _select_subjects(dataset, limit: int | None) -> list[int]:
    subjects = list(getattr(dataset, "subject_list", []))
    if limit is not None:
        subjects = subjects[:limit]
    if not subjects:
        raise ValueError(f"{dataset.__class__.__name__} did not expose a subject_list")
    return subjects


def _split_subject(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Prefer session/run holdout; fall back to stratified trial split."""

    for column in ("session", "run"):
        if column in meta.columns:
            values = list(pd.Series(meta[column]).dropna().unique())
            if len(values) > 1:
                train_values = set(values[:-1])
                train_mask = meta[column].isin(train_values).to_numpy()
                test_mask = ~train_mask
                if train_mask.any() and test_mask.any():
                    return X[train_mask], y[train_mask], X[test_mask], y[test_mask], f"held_out_{column}"

    indices = np.arange(y.shape[0])
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.35,
        random_state=random_state,
        stratify=y,
    )
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx], "stratified_trial_split"


def _mask_for_coverage(proba: np.ndarray, coverage: float) -> tuple[np.ndarray, float]:
    conf = np.max(proba, axis=1)
    n_keep = int(np.ceil(float(coverage) * conf.size))
    n_keep = min(max(n_keep, 1), conf.size)
    order = np.argsort(conf)[::-1]
    mask = np.zeros(conf.size, dtype=bool)
    mask[order[:n_keep]] = True
    threshold = float(conf[order[n_keep - 1]])
    return mask, threshold


def _load_left_right_dataset(dataset_name: str, subjects: Iterable[int]):
    _, LeftRightImagery = _import_moabb()
    dataset = _dataset_instance(dataset_name)
    paradigm = LeftRightImagery(fmin=8.0, fmax=30.0, resample=250.0)
    X, y, meta = paradigm.get_data(dataset=dataset, subjects=list(subjects))
    return np.asarray(X, dtype=np.float32), np.asarray(y), pd.DataFrame(meta)


def _run_dataset(
    *,
    dataset_name: str,
    subjects_per_dataset: int | None,
    operating_coverage: float,
    adaptation: str,
    random_state: int,
    include_eegnet: bool,
    eegnet_epochs: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    dataset = _dataset_instance(dataset_name)
    subjects = _select_subjects(dataset, subjects_per_dataset)
    X, y, meta = _load_left_right_dataset(dataset_name, subjects)

    subject_rows = []
    curve_rows = []
    metric_rows = []
    diagram_rows = []

    for subject in subjects:
        subject_mask = meta["subject"].astype(str).to_numpy() == str(subject)
        X_sub = X[subject_mask]
        y_sub = y[subject_mask]
        meta_sub = meta.loc[subject_mask].reset_index(drop=True)
        if X_sub.shape[0] < 20 or np.unique(y_sub).size < 2:
            continue

        X_train, y_train, X_test, y_test, split = _split_subject(
            X_sub,
            y_sub,
            meta_sub,
            random_state=random_state,
        )
        if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
            continue

        X_train_eval, X_test_eval = adapt_train_test(
            X_train,
            X_test,
            method=adaptation,
            eps=ALIGNMENT_EPS,
        )
        pred = fit_predict_subject(
            subject=int(subject),
            X_train=X_train_eval,
            y_train=y_train,
            X_test=X_test_eval,
            y_test=y_test,
            model_names=BASE_MODELS,
        )
        p_ens = equal_soft_vote(pred.proba)
        p_cmp = pred.proba[COMPARISON_MODEL]
        extra_probas: dict[str, np.ndarray] = {}
        if include_eegnet:
            from eegnet import EEGNetConfig, fit_predict_eegnet

            eeg_y_true, p_eegnet = fit_predict_eegnet(
                X_train=X_train_eval,
                y_train=y_train,
                X_test=X_test_eval,
                y_test=y_test,
                config=EEGNetConfig(epochs=eegnet_epochs, random_state=random_state),
            )
            if not np.array_equal(eeg_y_true, pred.y_true):
                raise RuntimeError("EEGNet label encoding does not match the classical decoder encoding")
            extra_probas["EEGNet"] = p_eegnet
        ens_mask, ens_threshold = _mask_for_coverage(p_ens, operating_coverage)
        cmp_mask, cmp_threshold = _mask_for_coverage(p_cmp, operating_coverage)
        ens_acc = accuracy(pred.y_true, p_ens, ens_mask)
        cmp_acc = accuracy(pred.y_true, p_cmp, cmp_mask)
        extra_subject_cols = {}
        for extra_name, extra_proba in extra_probas.items():
            extra_mask, extra_threshold = _mask_for_coverage(extra_proba, operating_coverage)
            extra_acc = accuracy(pred.y_true, extra_proba, extra_mask)
            key = extra_name.lower()
            extra_subject_cols[f"{key}_threshold_at_coverage"] = extra_threshold
            extra_subject_cols[f"{key}_acc_matched_coverage"] = extra_acc
            extra_subject_cols[f"delta_ensemble_minus_{key}"] = ens_acc - extra_acc
            extra_subject_cols[f"{key}_acc_all"] = accuracy(pred.y_true, extra_proba)

        ens_metrics = metrics_row(
            dataset=dataset_name,
            subject=subject,
            decoder="ensemble",
            y_true=pred.y_true,
            proba=p_ens,
        )
        cmp_metrics = metrics_row(
            dataset=dataset_name,
            subject=subject,
            decoder=COMPARISON_MODEL,
            y_true=pred.y_true,
            proba=p_cmp,
        )
        subject_rows.append(
            {
                "dataset": dataset_name,
                "subject": subject,
                "split": split,
                "n_train": int(X_train.shape[0]),
                "n_test": int(X_test.shape[0]),
                "adaptation": adaptation,
                "coverage": float(operating_coverage),
                "ensemble_threshold_at_coverage": ens_threshold,
                f"{COMPARISON_MODEL.lower()}_threshold_at_coverage": cmp_threshold,
                "ensemble_acc_matched_coverage": ens_acc,
                f"{COMPARISON_MODEL.lower()}_acc_matched_coverage": cmp_acc,
                f"delta_ensemble_minus_{COMPARISON_MODEL.lower()}": ens_acc - cmp_acc,
                "ensemble_ece": ens_metrics["ece"],
                f"{COMPARISON_MODEL.lower()}_ece": cmp_metrics["ece"],
                f"delta_ece_ensemble_minus_{COMPARISON_MODEL.lower()}": ens_metrics["ece"] - cmp_metrics["ece"],
                "ensemble_brier": ens_metrics["brier"],
                f"{COMPARISON_MODEL.lower()}_brier": cmp_metrics["brier"],
                f"delta_brier_ensemble_minus_{COMPARISON_MODEL.lower()}": ens_metrics["brier"] - cmp_metrics["brier"],
                **extra_subject_cols,
            }
        )
        decoder_outputs = {"ensemble": p_ens, COMPARISON_MODEL: p_cmp, **extra_probas}
        for name, proba in decoder_outputs.items():
            curve_rows.extend(
                risk_coverage_rows(
                    dataset=dataset_name,
                    subject=subject,
                    decoder=name,
                    y_true=pred.y_true,
                    proba=proba,
                    thresholds=THRESHOLD_GRID,
                )
            )
            metric_rows.append(
                metrics_row(
                    dataset=dataset_name,
                    subject=subject,
                    decoder=name,
                    y_true=pred.y_true,
                    proba=proba,
                )
            )
            diagram_rows.extend(
                reliability_rows(
                    dataset=dataset_name,
                    subject=subject,
                    decoder=name,
                    y_true=pred.y_true,
                    proba=proba,
                )
            )

    return subject_rows, curve_rows, metric_rows, diagram_rows


def summarize(subject_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    delta_acc_col = f"delta_ensemble_minus_{COMPARISON_MODEL.lower()}"
    delta_ece_col = f"delta_ece_ensemble_minus_{COMPARISON_MODEL.lower()}"
    delta_brier_col = f"delta_brier_ensemble_minus_{COMPARISON_MODEL.lower()}"
    summary = (
        subject_df.groupby("dataset", as_index=False)
        .agg(
            n_subjects=("subject", "nunique"),
            mean_delta_acc_matched_coverage=(delta_acc_col, "mean"),
            median_delta_acc_matched_coverage=(delta_acc_col, "median"),
            mean_delta_ece=(delta_ece_col, "mean"),
            mean_delta_brier=(delta_brier_col, "mean"),
            mean_ensemble_acc_matched_coverage=("ensemble_acc_matched_coverage", "mean"),
            mean_comparison_acc_matched_coverage=(f"{COMPARISON_MODEL.lower()}_acc_matched_coverage", "mean"),
            mean_ensemble_ece=("ensemble_ece", "mean"),
            mean_comparison_ece=(f"{COMPARISON_MODEL.lower()}_ece", "mean"),
            mean_ensemble_brier=("ensemble_brier", "mean"),
            mean_comparison_brier=(f"{COMPARISON_MODEL.lower()}_brier", "mean"),
        )
        .sort_values("dataset")
    )
    summary.to_csv(out_dir / "summary_by_dataset.csv", index=False)
    return summary


def run(
    *,
    datasets: tuple[str, ...] = DEFAULT_DATASETS,
    out_dir: Path = OUTPUT_DIR.parent / "moabb_external",
    subjects_per_dataset: int | None = 5,
    operating_coverage: float = OPERATING_COVERAGE,
    adaptation: str = SESSION_ADAPTATION,
    random_state: int = 42,
    include_eegnet: bool = False,
    eegnet_epochs: int = 60,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    subject_rows: list[dict] = []
    curve_rows: list[dict] = []
    metric_rows: list[dict] = []
    diagram_rows: list[dict] = []

    for dataset_name in datasets:
        ds_subjects, ds_curves, ds_metrics, ds_diagrams = _run_dataset(
            dataset_name=dataset_name,
            subjects_per_dataset=subjects_per_dataset,
            operating_coverage=operating_coverage,
            adaptation=adaptation,
            random_state=random_state,
            include_eegnet=include_eegnet,
            eegnet_epochs=eegnet_epochs,
        )
        subject_rows.extend(ds_subjects)
        curve_rows.extend(ds_curves)
        metric_rows.extend(ds_metrics)
        diagram_rows.extend(ds_diagrams)

    subject_df = pd.DataFrame(subject_rows).sort_values(["dataset", "subject"])
    curve_df = pd.DataFrame(curve_rows).sort_values(["dataset", "decoder", "subject", "threshold"])
    metrics_df = pd.DataFrame(metric_rows).sort_values(["dataset", "decoder", "subject"])
    reliability_df = pd.DataFrame(diagram_rows).sort_values(["dataset", "decoder", "subject", "bin"])

    subject_df.to_csv(out_dir / "subject_results.csv", index=False)
    curve_df.to_csv(out_dir / "risk_coverage.csv", index=False)
    metrics_df.to_csv(out_dir / "reliability_metrics.csv", index=False)
    reliability_df.to_csv(out_dir / "reliability_diagram_bins.csv", index=False)
    if not reliability_df.empty:
        write_reliability_diagrams(reliability_df, out_dir)
    if not subject_df.empty:
        summarize(subject_df, out_dir)

    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "evaluation": "MOABB LeftRightImagery external validation",
                "datasets": datasets,
                "subjects_per_dataset": subjects_per_dataset,
                "base_models": BASE_MODELS,
                "ensemble": "equal soft vote",
                "comparison": COMPARISON_MODEL,
                "eegnet": "included as optional comparison" if include_eegnet else "not run",
                "eegnet_epochs": eegnet_epochs if include_eegnet else None,
                "operating_coverage": operating_coverage,
                "threshold_grid": THRESHOLD_GRID,
                "adaptation": adaptation,
                "random_state": random_state,
                "note": "MOABB downloads raw datasets on first run; labels are used only for training and final scoring.",
            },
            indent=2,
        )
    )
    return out_dir / "summary_by_dataset.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MOABB external reliability validation.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR.parent / "moabb_external")
    parser.add_argument("--subjects-per-dataset", type=int, default=5)
    parser.add_argument("--all-subjects", action="store_true")
    parser.add_argument("--coverage", type=float, default=OPERATING_COVERAGE)
    parser.add_argument("--adaptation", default=SESSION_ADAPTATION, choices=("euclidean", "euclidean_alignment", "none"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--include-eegnet", action="store_true", help="Add optional EEGNet deep-learning comparison.")
    parser.add_argument("--eegnet-epochs", type=int, default=60)
    args = parser.parse_args()

    limit = None if args.all_subjects else args.subjects_per_dataset
    result = run(
        datasets=tuple(args.datasets),
        out_dir=args.out_dir,
        subjects_per_dataset=limit,
        operating_coverage=args.coverage,
        adaptation=args.adaptation,
        random_state=args.random_state,
        include_eegnet=args.include_eegnet,
        eegnet_epochs=args.eegnet_epochs,
    )
    print(f"Wrote {result}")


if __name__ == "__main__":
    main()
