"""Metrics and output table builders for v2 selective prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from reliability import brier_score, expected_calibration_error, reliability_rows

from .config import COVERAGE_POINTS, CURVE_COVERAGES
from .uncertainty import mask_for_coverage, normalize_proba


def _accuracy(y_true: np.ndarray, proba: np.ndarray, mask: np.ndarray | None = None) -> float:
    pred = normalize_proba(proba).argmax(axis=1)
    if mask is None:
        mask = np.ones(y_true.size, dtype=bool)
    if not mask.any():
        return float("nan")
    return float(np.mean(pred[mask] == y_true[mask]))


def _risk(y_true: np.ndarray, proba: np.ndarray, mask: np.ndarray) -> float:
    acc = _accuracy(y_true, proba, mask)
    return float(1.0 - acc) if np.isfinite(acc) else float("nan")


def subject_metric_row(
    *,
    dataset: str,
    subject: str | int,
    split: str,
    model: str,
    y_true: np.ndarray,
    proba: np.ndarray,
    uncertainty_score: str,
    coverage_target: float,
    use_ea: bool,
    calibrated: bool,
) -> dict:
    proba = normalize_proba(proba)
    pred = proba.argmax(axis=1)
    mask, cutoff = mask_for_coverage(proba, coverage_target, uncertainty_score)
    return {
        "dataset": dataset,
        "subject": subject,
        "split": split,
        "model": model,
        "use_ea": bool(use_ea),
        "calibrated": bool(calibrated),
        "uncertainty_score": uncertainty_score,
        "coverage_target": float(coverage_target),
        "score_cutoff": cutoff,
        "n_trials": int(y_true.size),
        "accuracy_all": float(np.mean(pred == y_true)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "brier": brier_score(y_true, proba),
        "ece": expected_calibration_error(y_true, proba),
        "selective_accuracy": _accuracy(y_true, proba, mask),
        "selective_risk": _risk(y_true, proba, mask),
        "coverage_observed": float(mask.mean()),
    }


def risk_coverage_rows(
    *,
    dataset: str,
    subject: str | int,
    model: str,
    y_true: np.ndarray,
    proba: np.ndarray,
    uncertainty_score: str,
) -> list[dict]:
    rows = []
    for coverage in CURVE_COVERAGES:
        mask, cutoff = mask_for_coverage(proba, coverage, uncertainty_score)
        rows.append(
            {
                "dataset": dataset,
                "subject": subject,
                "model": model,
                "uncertainty_score": uncertainty_score,
                "coverage": float(mask.mean()),
                "target_coverage": coverage,
                "score_cutoff": cutoff,
                "risk": _risk(y_true, proba, mask),
                "accuracy": _accuracy(y_true, proba, mask),
                "n_covered": int(mask.sum()),
            }
        )
    return rows


def coverage_sweep_rows(
    *,
    dataset: str,
    subject: str | int,
    model: str,
    y_true: np.ndarray,
    proba: np.ndarray,
    uncertainty_score: str,
) -> list[dict]:
    rows = []
    for coverage in COVERAGE_POINTS:
        mask, cutoff = mask_for_coverage(proba, coverage, uncertainty_score)
        rows.append(
            {
                "dataset": dataset,
                "subject": subject,
                "model": model,
                "uncertainty_score": uncertainty_score,
                "target_coverage": coverage,
                "coverage": float(mask.mean()),
                "score_cutoff": cutoff,
                "selective_accuracy": _accuracy(y_true, proba, mask),
                "selective_risk": _risk(y_true, proba, mask),
            }
        )
    return rows


def reliability_bin_rows(dataset: str, subject: str | int, model: str, y_true: np.ndarray, proba: np.ndarray) -> list[dict]:
    return reliability_rows(dataset=dataset, subject=subject, decoder=model, y_true=y_true, proba=proba)


def aggregate_metrics(subject_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "model", "use_ea", "calibrated", "uncertainty_score", "coverage_target"]
    metrics = [
        "accuracy_all",
        "macro_f1",
        "brier",
        "ece",
        "selective_accuracy",
        "selective_risk",
        "coverage_observed",
    ]
    return (
        subject_df.groupby(group_cols, as_index=False)
        .agg(**{f"mean_{m}": (m, "mean") for m in metrics}, n_subjects=("subject", "nunique"))
        .sort_values(["dataset", "model"])
    )


def ablation_summary(subject_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, ds_df in subject_df.groupby("dataset"):
        for model, model_df in ds_df.groupby("model"):
            base = model_df[model_df["uncertainty_score"] == "max-prob"]
            if not base.empty:
                rows.append(
                    {
                        "dataset": dataset,
                        "ablation": "model",
                        "setting": model,
                        "mean_selective_accuracy": base["selective_accuracy"].mean(),
                        "mean_ece": base["ece"].mean(),
                        "mean_brier": base["brier"].mean(),
                    }
                )
    return pd.DataFrame(rows)


def write_tables(
    out_dir: Path,
    subject_rows: list[dict],
    curve_rows: list[dict],
    reliability_rows_: list[dict],
    coverage_rows: list[dict],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    subject_df = pd.DataFrame(subject_rows)
    curve_df = pd.DataFrame(curve_rows)
    reliability_df = pd.DataFrame(reliability_rows_)
    coverage_df = pd.DataFrame(coverage_rows)
    subject_df.to_csv(out_dir / "subject_metrics.csv", index=False)
    aggregate_metrics(subject_df).to_csv(out_dir / "aggregate_metrics.csv", index=False)
    ablation_summary(subject_df).to_csv(out_dir / "ablation_summary.csv", index=False)
    curve_df.to_csv(out_dir / "risk_coverage.csv", index=False)
    reliability_df.to_csv(out_dir / "reliability_diagram_bins.csv", index=False)
    coverage_df.to_csv(out_dir / "coverage_sweep.csv", index=False)
