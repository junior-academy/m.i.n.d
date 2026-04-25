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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

try:
    from tqdm.auto import tqdm  # type: ignore

    _tqdm_write = tqdm.write
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(it=None, **_kwargs):  # type: ignore
        return it if it is not None else []

    def _tqdm_write(msg: str) -> None:  # type: ignore
        print(msg)

try:
    from mne.decoding import CSP  # type: ignore
except ModuleNotFoundError:
    from csp import CSP

from fbcsp import FBCSPFeatures

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

def _parse_bands_arg(arg: str) -> List[Tuple[float, float]]:
    """
    Parse a band string like: "8-12,12-16,16-20,20-30" -> [(8,12), ...]
    """
    if not arg.strip():
        raise ValueError("--bands must not be empty")
    bands: List[Tuple[float, float]] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            raise ValueError(f"Invalid band '{part}'; expected 'lo-hi' (e.g., 8-12)")
        lo_s, hi_s = part.split("-", 1)
        lo = float(lo_s.strip())
        hi = float(hi_s.strip())
        if lo <= 0 or hi <= 0 or hi <= lo:
            raise ValueError(f"Invalid band '{part}'; expected 0 < lo < hi")
        bands.append((lo, hi))
    if not bands:
        raise ValueError("--bands must not be empty")
    return bands

def build_models(
    *,
    random_state: int,
    features: str,
    n_components: int,
    sfreq: float,
    bands: Sequence[Tuple[float, float]],
    include_bandpower: bool,
) -> Dict[str, Pipeline]:
    if features not in {"csp", "fbcsp"}:
        raise ValueError(f"Unknown features='{features}'")

    def make_feat():
        if features == "csp":
            return CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
        return FBCSPFeatures(
            sfreq=float(sfreq),
            bands=tuple((float(lo), float(hi)) for lo, hi in bands),
            n_components=int(n_components),
            include_bandpower=bool(include_bandpower),
        )

    return {
        "LDA": Pipeline([("feat", make_feat()), ("clf", LinearDiscriminantAnalysis())]),
        "SVM": Pipeline(
            [("feat", make_feat()), ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True))]
        ),
        "RF": Pipeline(
            [
                ("feat", make_feat()),
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

def _fit_for_fold(
    *,
    model_name: str,
    base_model: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    tune: str,
    tune_cv: int,
    random_state: int,
    calibrate: str,
    calibration_cv: int,
) -> object:
    estimator = clone(base_model)

    if tune == "small" and model_name == "SVM":
        inner_cv = StratifiedKFold(n_splits=tune_cv, shuffle=True, random_state=random_state)
        param_grid = {
            "feat__n_components": [4, 6, 8],
            "clf__C": [0.5, 1.0, 2.0],
            "clf__gamma": ["scale", 0.1, 0.01],
        }
        gs = GridSearchCV(
            estimator,
            param_grid=param_grid,
            scoring="accuracy",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
        )
        gs.fit(X_train, y_train)
        estimator = clone(gs.best_estimator_)

    if calibrate != "none" and model_name in {"SVM", "RF"}:
        cal = CalibratedClassifierCV(estimator, method=calibrate, cv=calibration_cv)
        cal.fit(X_train, y_train)
        return cal

    estimator.fit(X_train, y_train)
    return estimator

def oof_predict_proba(
    *,
    X: np.ndarray,
    y: np.ndarray,
    models: Dict[str, Pipeline],
    n_splits: int,
    random_state: int,
    tune: str,
    tune_cv: int,
    calibrate: str,
    calibration_cv: int,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    classes = np.unique(y)
    n_classes = len(classes)
    proba = {name: np.full((len(y), n_classes), np.nan, dtype=float) for name in models}

    splits = list(skf.split(X, y))
    for train_idx, test_idx in tqdm(splits, total=n_splits, desc="OOF CV folds", leave=False):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[test_idx]
        for name, model in models.items():
            fitted = _fit_for_fold(
                model_name=name,
                base_model=model,
                X_train=X_train,
                y_train=y_train,
                tune=tune,
                tune_cv=tune_cv,
                random_state=random_state,
                calibrate=calibrate,
                calibration_cv=calibration_cv,
            )
            p = fitted.predict_proba(X_test)
            if p.shape[1] != n_classes:
                raise RuntimeError(f"{name} predict_proba returned {p.shape[1]} classes; expected {n_classes}")
            proba[name][test_idx] = p

    for name, p in proba.items():
        if np.isnan(p).any():
            raise RuntimeError(f"OOF probabilities not fully populated for {name}")
    return proba, classes

def oof_stacking_proba(
    *,
    proba: Dict[str, np.ndarray],
    y: np.ndarray,
    model_order: Sequence[str],
    n_splits: int,
    random_state: int,
) -> np.ndarray:
    X_meta = np.hstack([proba[name] for name in model_order])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    classes = np.unique(y)
    n_classes = int(classes.size)
    meta = np.full((len(y), n_classes), np.nan, dtype=float)
    for train_idx, test_idx in skf.split(X_meta, y):
        X_tr, y_tr = X_meta[train_idx], y[train_idx]
        X_te = X_meta[test_idx]
        lr = LogisticRegression(max_iter=1000, solver="lbfgs")
        lr.fit(X_tr, y_tr)
        meta[test_idx] = lr.predict_proba(X_te)
    if np.isnan(meta).any():
        raise RuntimeError("OOF stacking probabilities not fully populated")
    return meta

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

def _t_confidence_interval(values: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = int(values.size)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    if n < 2:
        return {"n": n, "mean": mean, "sd": sd, "ci_low": mean, "ci_high": mean}
    from scipy import stats

    sem = sd / (n**0.5)
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    half = float(tcrit * sem)
    return {"n": n, "mean": mean, "sd": sd, "ci_low": mean - half, "ci_high": mean + half}

def _paired_ttest(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a = a[mask]
    b = b[mask]
    n = int(a.size)
    if n < 2:
        return {"n": n, "t_stat": float("nan"), "p_value": float("nan")}
    from scipy import stats

    t_stat, p_value = stats.ttest_rel(a, b)
    return {"n": n, "t_stat": float(t_stat), "p_value": float(p_value)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=(BASE_DIR / "outputs" / "ensemble_v2"))
    parser.add_argument("--run-name", type=str, default="", help="Optional name (creates a subfolder under out-dir)")
    parser.add_argument("--subjects", type=str, default="all", help="Comma-separated subject ids (e.g., 1,2,3) or 'all'")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--preset",
        choices=["main", "ablation_rf", "custom"],
        default="main",
        help="main=LDA+SVM (recommended), ablation_rf=LDA+SVM+RF, custom=use --models",
    )
    parser.add_argument("--models", type=str, default="LDA,SVM", help="Comma-separated subset: LDA,SVM,RF")
    parser.add_argument(
        "--weights",
        choices=["equal", "baseline_subject", "baseline_global"],
        default="baseline_subject",
    )
    parser.add_argument(
        "--features",
        choices=["csp", "fbcsp"],
        default="csp",
        help="Feature extractor: csp (default) or fbcsp (filter-bank CSP).",
    )
    parser.add_argument("--n-components", type=int, default=4, help="CSP/FBCSP components per band (default: 4).")
    parser.add_argument(
        "--bands",
        type=str,
        default="8-12,12-16,16-20,20-30",
        help="FBCSP sub-bands as 'lo-hi' pairs, comma-separated (default: 8-12,12-16,16-20,20-30).",
    )
    parser.add_argument(
        "--sfreq",
        type=float,
        default=250.0,
        help="Sampling frequency (Hz) for FBCSP bandpass filters (default: 250).",
    )
    parser.add_argument(
        "--include-bandpower",
        action="store_true",
        help="When --features fbcsp, also concatenate per-channel bandpower features per band.",
    )
    parser.add_argument(
        "--tune",
        choices=["none", "small"],
        default="none",
        help="Per-subject tuning (nested CV): small tunes SVM (feat__n_components, C, gamma).",
    )
    parser.add_argument("--tune-cv", type=int, default=3, help="Inner CV folds for --tune small (default: 3).")
    parser.add_argument(
        "--ensemble-method",
        choices=["softvote", "stacking"],
        default="softvote",
        help="Ensembling method: softvote (weighted soft voting) or stacking (logreg on base probabilities).",
    )
    parser.add_argument(
        "--calibrate",
        choices=["none", "sigmoid", "isotonic"],
        default="sigmoid",
        help="Calibrate probabilities for SVM/RF (recommended: sigmoid).",
    )
    parser.add_argument("--calibration-cv", type=int, default=3, help="CV folds for CalibratedClassifierCV")
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

    if args.preset == "main":
        model_list = ["LDA", "SVM"]
    elif args.preset == "ablation_rf":
        model_list = ["LDA", "SVM", "RF"]
    else:
        model_list = _parse_models(args.models)
    bands = _parse_bands_arg(args.bands)
    all_models = build_models(
        random_state=args.random_state,
        features=args.features,
        n_components=int(args.n_components),
        sfreq=float(args.sfreq),
        bands=bands,
        include_bandpower=bool(args.include_bandpower),
    )
    models = {name: all_models[name] for name in model_list}

    run_suffix = args.run_name.strip()
    if not run_suffix:
        run_suffix = (
            f"models-{'_'.join(model_list)}__weights-{args.weights}"
            f"__feat-{args.features}"
            f"__ens-{args.ensemble_method}"
            f"__tune-{args.tune}"
            f"__cal-{args.calibrate}"
        )
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
        "preset": args.preset,
        "models": model_list,
        "weights": args.weights,
        "features": args.features,
        "n_components": int(args.n_components),
        "bands": bands,
        "sfreq": float(args.sfreq),
        "include_bandpower": bool(args.include_bandpower),
        "tune": args.tune,
        "tune_cv": int(args.tune_cv),
        "ensemble_method": args.ensemble_method,
        "calibrate": args.calibrate,
        "calibration_cv": args.calibration_cv,
        "threshold": args.threshold,
        "threshold_grid": thresholds,
        "n_splits": args.n_splits,
        "random_state": args.random_state,
        "data_dir": str(args.data_dir),
        "baseline_results_csv": str(BASELINE_RESULTS_CSV),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))

    baseline_best_acc_by_subject: Optional[pd.Series] = None
    if BASELINE_RESULTS_CSV.exists():
        baseline_df = pd.read_csv(BASELINE_RESULTS_CSV)
        if {"subject", "best_acc"}.issubset(baseline_df.columns):
            baseline_best_acc_by_subject = baseline_df.set_index("subject")["best_acc"]

    for x_path, y_path in tqdm(pairs, total=len(pairs), desc="Subjects", unit="subj"):
        subject_id = int(subject_key(x_path))
        _tqdm_write(f"[ensemble_v2] Subject {subject_id}: {x_path.name} / {y_path.name}")
        X, y = load_subject(x_path, y_path)

        proba, classes = oof_predict_proba(
            X=X,
            y=y,
            models=models,
            n_splits=args.n_splits,
            random_state=args.random_state,
            tune=args.tune,
            tune_cv=args.tune_cv,
            calibrate=args.calibrate,
            calibration_cv=args.calibration_cv,
        )

        if args.ensemble_method == "stacking":
            weights = {}
            p_ens = oof_stacking_proba(
                proba=proba,
                y=y,
                model_order=model_list,
                n_splits=args.n_splits,
                random_state=args.random_state,
            )
        else:
            if args.weights == "equal":
                weights = {name: 1.0 for name in models.keys()}
            elif args.weights == "baseline_global":
                if global_w is None:
                    raise SystemExit(f"Missing baseline file for global weights: {BASELINE_RESULTS_CSV}")
                weights = global_w
            else:
                w = baseline_subject_weights(subject_id)
                if w is None:
                    raise SystemExit(
                        f"Missing baseline file/subject for weights: {BASELINE_RESULTS_CSV} subject={subject_id}"
                    )
                weights = w
            weights = _normalize_weights(weights, models.keys())
            p_ens = ensemble_proba(proba, weights)
        pred_all = p_ens.argmax(axis=1)
        max_prob = p_ens.max(axis=1)

        per_model_acc = {f"{name}_oof_acc": float(accuracy_score(y, p.argmax(axis=1))) for name, p in proba.items()}

        default_metrics = threshold_metrics(y=y, p_ens=p_ens, threshold=args.threshold)

        best_single_acc = (
            float(baseline_best_acc_by_subject.loc[subject_id])
            if baseline_best_acc_by_subject is not None and subject_id in baseline_best_acc_by_subject.index
            else float("nan")
        )
        subject_rows.append(
            {
                "subject": subject_id,
                "n_trials": int(X.shape[0]),
                "n_channels": int(X.shape[1]),
                "n_timepoints": int(X.shape[2]),
                "n_classes": int(len(classes)),
                "weights": json.dumps(weights, sort_keys=True),
                "best_single_acc": best_single_acc,
                **per_model_acc,
                **default_metrics,
            }
        )

        for t in thresholds:
            m = threshold_metrics(y=y, p_ens=p_ens, threshold=t)
            threshold_rows.append({"subject": subject_id, **m})

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

    subject_results_df = pd.DataFrame(subject_rows).sort_values("subject")
    threshold_metrics_df = pd.DataFrame(threshold_rows).sort_values(["subject", "threshold"])

    subject_results_df["delta_all_vs_best"] = subject_results_df["ensemble_acc_all"] - subject_results_df["best_single_acc"]
    subject_results_df["delta_conf_vs_best"] = (
        subject_results_df["ensemble_acc_confident"] - subject_results_df["best_single_acc"]
    )

    subject_results_df.to_csv(out_dir / "subject_results.csv", index=False)
    threshold_metrics_df.to_csv(out_dir / "threshold_metrics.csv", index=False)

    summary_rows = []
    for col in ["ensemble_acc_all", "ensemble_acc_confident", "ensemble_coverage", "best_single_acc", "delta_all_vs_best", "delta_conf_vs_best"]:
        stats = _t_confidence_interval(subject_results_df[col].to_numpy(dtype=float))
        summary_rows.append(
            {
                "metric": col,
                "n": stats["n"],
                "mean": stats["mean"],
                "sd": stats["sd"],
                "ci_low_95": stats["ci_low"],
                "ci_high_95": stats["ci_high"],
            }
        )

    t_all = _paired_ttest(
        subject_results_df["ensemble_acc_all"].to_numpy(dtype=float),
        subject_results_df["best_single_acc"].to_numpy(dtype=float),
    )
    t_conf = _paired_ttest(
        subject_results_df["ensemble_acc_confident"].to_numpy(dtype=float),
        subject_results_df["best_single_acc"].to_numpy(dtype=float),
    )
    summary_rows.append(
        {
            "metric": "paired_ttest_all_vs_best",
            "n": t_all["n"],
            "mean": float("nan"),
            "sd": float("nan"),
            "ci_low_95": float("nan"),
            "ci_high_95": float("nan"),
            "t_stat": t_all["t_stat"],
            "p_value": t_all["p_value"],
        }
    )
    summary_rows.append(
        {
            "metric": "paired_ttest_confident_vs_best",
            "n": t_conf["n"],
            "mean": float("nan"),
            "sd": float("nan"),
            "ci_low_95": float("nan"),
            "ci_high_95": float("nan"),
            "t_stat": t_conf["t_stat"],
            "p_value": t_conf["p_value"],
        }
    )
    pd.DataFrame(summary_rows).to_csv(out_dir / "run_summary.csv", index=False)
    print(f"[ensemble_v2] Wrote {out_dir / 'subject_results.csv'}")
    print(f"[ensemble_v2] Wrote {out_dir / 'threshold_metrics.csv'}")
    print(f"[ensemble_v2] Wrote {out_dir / 'run_summary.csv'}")

if __name__ == "__main__":
    main()
