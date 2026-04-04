"""
Ensemble pipeline for motor imagery BCI (BCIC IV 2a).

Loads per-subject `X_*.npy` / `y_*.npy`, trains LDA/SVM/RF with CSP features, and
produces a weighted soft-voting ensemble with an optional confidence threshold.

Outputs:
- outputs/ensemble/ensemble_subject_results.csv
- outputs/ensemble/predictions_subject_<id>.csv
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

try:
    from mne.decoding import CSP  # type: ignore
except ModuleNotFoundError:
    # Fall back to a minimal local CSP implementation so the repo runs without mne installed.
    from csp import CSP

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "analysis_results"
BASELINE_RESULTS_CSV = BASE_DIR / "outputs" / "classification_results.csv"

def subject_key(path: Path):
    nums = re.findall(r"\d+", path.stem)
    if nums:
        return int(nums[0])
    return path.stem

def find_xy_files(data_dir: Path) -> List[Tuple[Path, Path]]:
    x_files = sorted([p for p in data_dir.glob("X_*.npy")], key=subject_key)
    y_files = sorted([p for p in data_dir.glob("y_*.npy")], key=subject_key)
    if len(x_files) != len(y_files):
        raise ValueError(f"Mismatch: {len(x_files)} X files but {len(y_files)} y files")
    return list(zip(x_files, y_files))

def load_subject(x_path: Path, y_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    X = np.load(x_path); y = np.load(y_path)
    le = LabelEncoder()
    y = le.fit_transform(y)
    return X, y

def build_models(random_state: int) -> Dict[str, Pipeline]:
    return {
        "LDA": Pipeline(
            [
                ("csp", CSP(n_components=4, reg=None, log=True, norm_trace=False)),
                ("clf", LinearDiscriminantAnalysis()),
            ]
        ),
        "SVM": Pipeline(
            [
                ("csp", CSP(n_components=4, reg=None, log=True, norm_trace=False)),
                ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)),
            ]
        ),
        "RF": Pipeline(
            [
                ("csp", CSP(n_components=4, reg=None, log=True, norm_trace=False)),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

def _normalize_weights(weights: Dict[str, float], model_names: Iterable[str]) -> Dict[str, float]:
    w = {k: float(weights.get(k, 0.0)) for k in model_names}
    total = sum(max(0.0, v) for v in w.values())
    if total <= 0:
        return {k: 1.0 / len(w) for k in w}
    return {k: max(0.0, v) / total for k, v in w.items()}

def weights_from_baseline(subject_id: int) -> Optional[Dict[str, float]]:
    if not BASELINE_RESULTS_CSV.exists():
        return None
    df = pd.read_csv(BASELINE_RESULTS_CSV)
    row = df[df["subject"] == subject_id]
    if row.empty:
        return None
    row = row.iloc[0]
    return {
        "LDA": float(row.get("LDA_mean_acc", np.nan)),
        "SVM": float(row.get("SVM_mean_acc", np.nan)),
        "RF": float(row.get("RF_mean_acc", np.nan)),
    }

def oof_predict_proba(X: np.ndarray, y: np.ndarray, models: Dict[str, Pipeline], n_splits: int, random_state: int) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    classes = np.unique(y)
    n_classes = len(classes)
    proba = {name: np.full((len(y), n_classes), np.nan, dtype=float) for name in models}

    for train_idx, test_idx in skf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[test_idx]
        for name, model in models.items():
            model.fit(X_train, y_train)
            p = model.predict_proba(X_test)
            if p.shape[1] != n_classes:
                raise RuntimeError(f"{name} predict_proba returned {p.shape[1]} classes; expected {n_classes}")
            proba[name][test_idx] = p

    # Validate all filled
    for name, p in proba.items():
        if np.isnan(p).any():
            raise RuntimeError(f"OOF probabilities not fully populated for {name}")
    return proba, classes

def ensemble_from_proba(proba: Dict[str, np.ndarray], weights: Dict[str, float], threshold: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model_names = list(proba.keys())
    w = _normalize_weights(weights, model_names)
    p_ens = np.zeros_like(next(iter(proba.values())))
    for name in model_names:
        p_ens += w[name] * proba[name]
    max_prob = p_ens.max(axis=1)
    pred = p_ens.argmax(axis=1)
    confident = max_prob >= threshold
    pred_thresholded = pred.copy()
    pred_thresholded[~confident] = -1
    return p_ens, pred_thresholded, max_prob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=(BASE_DIR / "outputs" / "ensemble"))
    parser.add_argument("--subjects", type=str, default="all", help="Comma-separated subject ids (e.g., 1,2,3) or 'all'")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.55, help="Confidence threshold on max ensemble probability")
    parser.add_argument("--weights", choices=["equal", "baseline"], default="baseline")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_xy_files(args.data_dir)
    if args.subjects != "all":
        wanted = {int(s.strip()) for s in args.subjects.split(",") if s.strip()}
        pairs = [(x, y) for x, y in pairs if int(subject_key(x)) in wanted]
        if not pairs:
            raise SystemExit(f"No subjects matched: {sorted(wanted)}")

    models = build_models(random_state=args.random_state)
    subject_rows = []

    for x_path, y_path in pairs:
        subject_id = int(subject_key(x_path))
        print(f"[ensemble] Subject {subject_id}: loading {x_path.name} / {y_path.name}")
        X, y = load_subject(x_path, y_path)

        proba, classes = oof_predict_proba(
            X=X,
            y=y,
            models=models,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )

        baseline_w = weights_from_baseline(subject_id) if args.weights == "baseline" else None
        weights = baseline_w or {"LDA": 1.0, "SVM": 1.0, "RF": 1.0}
        weights = _normalize_weights(weights, models.keys())

        p_ens, pred_thresh, max_prob = ensemble_from_proba(proba, weights, threshold=args.threshold)
        pred_no_thresh = p_ens.argmax(axis=1)

        acc_all = accuracy_score(y, pred_no_thresh)
        confident_mask = pred_thresh != -1
        coverage = float(confident_mask.mean())
        acc_conf = accuracy_score(y[confident_mask], pred_thresh[confident_mask]) if confident_mask.any() else float("nan")

        # per-model accuracies from OOF
        per_model_acc = {}
        for name, p in proba.items():
            per_model_acc[f"{name}_oof_acc"] = accuracy_score(y, p.argmax(axis=1))

        pred_df = pd.DataFrame(
            {
                "subject": subject_id,
                "trial_index": np.arange(len(y)),
                "y_true": y,
                "lda_pred": proba["LDA"].argmax(axis=1),
                "svm_pred": proba["SVM"].argmax(axis=1),
                "rf_pred": proba["RF"].argmax(axis=1),
                "ensemble_pred": pred_no_thresh,
                "ensemble_pred_thresholded": pred_thresh,
                "ensemble_max_prob": max_prob,
                "ensemble_confident": confident_mask.astype(int),
            }
        )
        for i, cls in enumerate(classes):
            pred_df[f"p_ens_c{int(cls)}"] = p_ens[:, i]
        pred_df.to_csv(args.out_dir / f"predictions_subject_{subject_id}.csv", index=False)

        row = {
            "subject": subject_id,
            "n_trials": int(X.shape[0]),
            "n_channels": int(X.shape[1]),
            "n_timepoints": int(X.shape[2]),
            "n_classes": int(len(classes)),
            "threshold": float(args.threshold),
            "ensemble_acc_all": float(acc_all),
            "ensemble_acc_confident": float(acc_conf),
            "ensemble_coverage": float(coverage),
            "weights": json.dumps(weights, sort_keys=True),
            **per_model_acc,
        }
        subject_rows.append(row)
        print(
            f"[ensemble] Subject {subject_id}: acc_all={acc_all:.3f} acc_conf={acc_conf:.3f} coverage={coverage:.3f} weights={weights}"
        )

    results_df = pd.DataFrame(subject_rows).sort_values("subject")
    results_df.to_csv(args.out_dir / "ensemble_subject_results.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "ensemble_mean_acc_all": float(results_df["ensemble_acc_all"].mean()),
                "ensemble_sd_acc_all": float(results_df["ensemble_acc_all"].std()),
                "ensemble_mean_acc_confident": float(results_df["ensemble_acc_confident"].mean()),
                "ensemble_sd_acc_confident": float(results_df["ensemble_acc_confident"].std()),
                "ensemble_mean_coverage": float(results_df["ensemble_coverage"].mean()),
                "ensemble_sd_coverage": float(results_df["ensemble_coverage"].std()),
            }
        ]
    )
    summary.to_csv(args.out_dir / "ensemble_summary.csv", index=False)
    print(f"[ensemble] Wrote {args.out_dir / 'ensemble_subject_results.csv'}")
    print(f"[ensemble] Wrote {args.out_dir / 'ensemble_summary.csv'}")


if __name__ == "__main__":
    main()
