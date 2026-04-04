"""
Ensemble v2: model selection + weight strategies + threshold sweep.

Key additions vs v1:
- Drop weaker models (e.g., use only LDA+SVM)
- Use equal weights, per-subject baseline weights, or global baseline weights
- Sweep thresholds without re-fitting models (reuses OOF probabilities)

Outputs (under `--out-dir`):
- config.json
- subject_results.csv (includes default threshold metrics)
- threshold_metrics.csv (long-form per-subject per-threshold)
- predictions_subject_<id>.csv (per-trial preds + ensemble probs)
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
    X = np.load(x_path)
    y = np.load(y_path)
    le = LabelEncoder()
    y = le.fit_transform(y)
    return X, y


def build_models(random_state: int) -> Dict[str, Pipeline]:
    csp_kwargs = dict(n_components=4, reg=None, log=True, norm_trace=False)
    return {
        "LDA": Pipeline(
            [
                ("csp", CSP(**csp_kwargs)),
                ("clf", LinearDiscriminantAnalysis()),
            ]
        ),
        "SVM": Pipeline(
            [
                ("csp", CSP(**csp_kwargs)),
                ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)),
            ]
        ),
        "RF": Pipeline(
            [
                ("csp", CSP(**csp_kwargs)),
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


def _parse_models(arg: str) -> List[str]:
    models = [m.strip().upper() for m in arg.split(",") if m.strip()]
    if not models:
        raise ValueError("--models must not be empty")
    allowed = {"LDA", "SVM", "RF"}
    bad = [m for m in models if m not in allowed]
    if bad:
        raise ValueError(f"Unknown model(s) {bad}; allowed: {sorted(allowed)}")
    # keep user order, remove duplicates
    out = []
    for m in models:
        if m not in out:
            out.append(m)
    return out


def _parse_thresholds(arg: str) -> List[float]:
    thresholds = [float(t.strip()) for t in arg.split(",") if t.strip()]
    if not thresholds:
        raise ValueError("--threshold-grid must not be empty")
    for t in thresholds:
        if t < 0 or t > 1:
            raise ValueError(f"Thresholds must be in [0,1]; got {t}")
    return sorted(set(thresholds))


def _normalize_weights(weights: Dict[str, float], model_names: Iterable[str]) -> Dict[str, float]:
    w = {k: float(weights.get(k, 0.0)) for k in model_names}
    total = sum(max(0.0, v) for v in w.values())
    if total <= 0:
        return {k: 1.0 / len(w) for k in w}
    return {k: max(0.0, v) / total for k, v in w.items()}


def baseline_subject_weights(subject_id: int) -> Optional[Dict[str, float]]:
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


def baseline_global_weights() -> Optional[Dict[str, float]]:
    if not BASELINE_RESULTS_CSV.exists():
        return None
    df = pd.read_csv(BASELINE_RESULTS_CSV)
    needed = {"LDA_mean_acc", "SVM_mean_acc", "RF_mean_acc"}
    if not needed.issubset(df.columns):
        return None
    return {
        "LDA": float(df["LDA_mean_acc"].mean()),
        "SVM": float(df["SVM_mean_acc"].mean()),
        "RF": float(df["RF_mean_acc"].mean()),
    }


def oof_predict_proba(
    X: np.ndarray,
    y: np.ndarray,
    models: Dict[str, Pipeline],
    n_splits: int,
    random_state: int,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
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

    for name, p in proba.items():
        if np.isnan(p).any():
            raise RuntimeError(f"OOF probabilities not fully populated for {name}")
    return proba, classes


def ensemble_proba(proba: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    model_names = list(proba.keys())
    w = _normalize_weights(weights, model_names)
    p_ens = np.zeros_like(next(iter(proba.values())))
    for name in model_names:
        p_ens += w[name] * proba[name]
    return p_ens


def threshold_metrics(y: np.ndarray, p_ens: np.ndarray, threshold: float) -> Dict[str, float]:
    pred_all = p_ens.argmax(axis=1)
    acc_all = float(accuracy_score(y, pred_all))

    max_prob = p_ens.max(axis=1)
    confident = max_prob >= threshold
    coverage = float(confident.mean())
    if confident.any():
        acc_conf = float(accuracy_score(y[confident], pred_all[confident]))
    else:
        acc_conf = float("nan")

    return {
        "threshold": float(threshold),
        "ensemble_acc_all": acc_all,
        "ensemble_acc_confident": acc_conf,
        "ensemble_coverage": coverage,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=(BASE_DIR / "outputs" / "ensemble_v2"))
    parser.add_argument("--run-name", type=str, default="", help="Optional name (creates a subfolder under out-dir)")
    parser.add_argument("--subjects", type=str, default="all", help="Comma-separated subject ids (e.g., 1,2,3) or 'all'")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--models", type=str, default="LDA,SVM,RF", help="Comma-separated subset: LDA,SVM,RF")
    parser.add_argument(
        "--weights",
        choices=["equal", "baseline_subject", "baseline_global"],
        default="baseline_subject",
    )
    parser.add_argument("--threshold", type=float, default=0.55, help="Default threshold for subject_results.csv")
    parser.add_argument(
        "--threshold-grid",
        type=str,
        default="0.0,0.55,0.6,0.65,0.7",
        help="Comma-separated thresholds for threshold_metrics.csv",
    )
    args = parser.parse_args()

    if not (0 <= args.threshold <= 1):
        raise SystemExit("--threshold must be in [0,1]")
    thresholds = _parse_thresholds(args.threshold_grid)

    model_list = _parse_models(args.models)
    all_models = build_models(random_state=args.random_state)
    models = {name: all_models[name] for name in model_list}

    run_suffix = args.run_name.strip()
    if not run_suffix:
        run_suffix = f"models-{'_'.join(model_list)}__weights-{args.weights}"
    out_dir = args.out_dir / run_suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_xy_files(args.data_dir)
    if args.subjects != "all":
        wanted = {int(s.strip()) for s in args.subjects.split(",") if s.strip()}
        pairs = [(x, y) for x, y in pairs if int(subject_key(x)) in wanted]
        if not pairs:
            raise SystemExit(f"No subjects matched: {sorted(wanted)}")

    global_w = baseline_global_weights() if args.weights == "baseline_global" else None
    subject_rows = []
    threshold_rows = []

    config = {
        "models": model_list,
        "weights": args.weights,
        "threshold": args.threshold,
        "threshold_grid": thresholds,
        "n_splits": args.n_splits,
        "random_state": args.random_state,
        "data_dir": str(args.data_dir),
        "baseline_results_csv": str(BASELINE_RESULTS_CSV),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    for x_path, y_path in pairs:
        subject_id = int(subject_key(x_path))
        print(f"[ensemble_v2] Subject {subject_id}: {x_path.name} / {y_path.name}")
        X, y = load_subject(x_path, y_path)

        proba, classes = oof_predict_proba(
            X=X,
            y=y,
            models=models,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )

        if args.weights == "equal":
            weights = {name: 1.0 for name in models.keys()}
        elif args.weights == "baseline_global":
            if global_w is None:
                raise SystemExit(f"Missing baseline file for global weights: {BASELINE_RESULTS_CSV}")
            weights = global_w
        else:
            w = baseline_subject_weights(subject_id)
            if w is None:
                raise SystemExit(f"Missing baseline file/subject for weights: {BASELINE_RESULTS_CSV} subject={subject_id}")
            weights = w
        weights = _normalize_weights(weights, models.keys())

        p_ens = ensemble_proba(proba, weights)
        pred_all = p_ens.argmax(axis=1)
        max_prob = p_ens.max(axis=1)

        # per-model accuracies from OOF
        per_model_acc = {f"{name}_oof_acc": float(accuracy_score(y, p.argmax(axis=1))) for name, p in proba.items()}

        # Default threshold metrics for subject_results.csv
        default_metrics = threshold_metrics(y=y, p_ens=p_ens, threshold=args.threshold)

        subject_rows.append(
            {
                "subject": subject_id,
                "n_trials": int(X.shape[0]),
                "n_channels": int(X.shape[1]),
                "n_timepoints": int(X.shape[2]),
                "n_classes": int(len(classes)),
                "weights": json.dumps(weights, sort_keys=True),
                **per_model_acc,
                **default_metrics,
            }
        )

        # Threshold sweep metrics
        for t in thresholds:
            m = threshold_metrics(y=y, p_ens=p_ens, threshold=t)
            threshold_rows.append({"subject": subject_id, **m})

        # Per-trial outputs
        pred_df = pd.DataFrame(
            {
                "subject": subject_id,
                "trial_index": np.arange(len(y)),
                "y_true": y,
                "ensemble_pred": pred_all,
                "ensemble_max_prob": max_prob,
            }
        )
        for name, p in proba.items():
            pred_df[f"{name.lower()}_pred"] = p.argmax(axis=1)
        for i, cls in enumerate(classes):
            pred_df[f"p_ens_c{int(cls)}"] = p_ens[:, i]
        pred_df.to_csv(out_dir / f"predictions_subject_{subject_id}.csv", index=False)

    pd.DataFrame(subject_rows).sort_values("subject").to_csv(out_dir / "subject_results.csv", index=False)
    pd.DataFrame(threshold_rows).sort_values(["subject", "threshold"]).to_csv(out_dir / "threshold_metrics.csv", index=False)
    print(f"[ensemble_v2] Wrote {out_dir / 'subject_results.csv'}")
    print(f"[ensemble_v2] Wrote {out_dir / 'threshold_metrics.csv'}")


if __name__ == "__main__":
    main()

