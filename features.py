from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # stratified k-fold cross-validation
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # lda
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier  # random forest classifier

try:
    from mne.decoding import CSP  # type: ignore
except ModuleNotFoundError:
    # Fall back to a minimal local CSP implementation so the repo runs without mne installed.
    from csp import CSP  # common spatial patterns for feature extraction

from fbcsp import FBCSPFeatures

try:
    from tqdm.auto import tqdm  # type: ignore

    _tqdm_write = tqdm.write
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(it=None, **_kwargs):  # type: ignore
        return it if it is not None else []

    def _tqdm_write(msg: str) -> None:  # type: ignore
        print(msg)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "analysis_results"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents = True, exist_ok = True)

N_SPLITS = 5
RANDOM_STATE = 42

def subject_key(path):
    """Extract subject key from filename"""
    nums = re.findall(r"\d+", path.stem)
    if nums:
        return int(nums[0])
    return path.stem

def find_xy_files(data_dir):
    """Find all X and y files in the data directory"""
    x_files = sorted([p for p in data_dir.glob("X_*.npy")], key=subject_key)
    y_files = sorted([p for p in data_dir.glob("y_*.npy")], key=subject_key)

    if len(x_files) != len(y_files):
        raise ValueError(f"Mismatch: {len(x_files)} X files but {len(y_files)} y files")
    pairs = list(zip(x_files, y_files))
    return pairs

def load_subject(x_path, y_path):
    """Load X and y for a single subject"""
    X = np.load(x_path)
    y = np.load(y_path)

    le = LabelEncoder()
    y = le.fit_transform(y)

    return X, y
    
def _parse_bands_arg(arg: str) -> List[Tuple[float, float]]:
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
    features: str,
    n_components: int,
    sfreq: float,
    bands: Sequence[Tuple[float, float]],
    include_bandpower: bool,
    random_state: int,
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
        "SVM": Pipeline([("feat", make_feat()), ("clf", SVC(kernel="rbf", C=1.0, gamma="scale"))]),
        "RF": Pipeline(
            [
                ("feat", make_feat()),
                ("clf", RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)),
            ]
        ),
    }

def evaluate_subject(
    *,
    X: np.ndarray,
    y: np.ndarray,
    models: Dict[str, Pipeline],
    n_splits: int,
    random_state: int,
    tune: str,
    tune_cv: int,
):
    """Evaluate all models for a single subject using stratified k-fold cross-validation"""
    results = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for name, model in models.items():
        scores: List[float] = []
        splits = list(skf.split(X, y))
        for train_idx, test_idx in tqdm(splits, total=n_splits, desc=f"{name} CV", leave=False):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            estimator = clone(model)
            if tune == "small" and name == "SVM":
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
                estimator = gs.best_estimator_
            else:
                estimator.fit(X_train, y_train)

            y_pred = estimator.predict(X_test)
            scores.append(float(accuracy_score(y_test, y_pred)))

        results[f"{name}_mean_acc"] = float(np.mean(scores))
        results[f"{name}_std_acc"] = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-splits", type=int, default=N_SPLITS)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
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
    parser.add_argument("--sfreq", type=float, default=250.0, help="Sampling frequency (Hz) for FBCSP (default: 250).")
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
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_xy_files(args.data_dir)
    bands = _parse_bands_arg(args.bands)
    models = build_models(
        features=args.features,
        n_components=int(args.n_components),
        sfreq=float(args.sfreq),
        bands=bands,
        include_bandpower=bool(args.include_bandpower),
        random_state=int(args.random_state),
    )

    all_results = []
    for x_path, y_path in tqdm(pairs, total=len(pairs), desc="Subjects", unit="subj"):
        subject_id = subject_key(x_path)
        _tqdm_write(f"[features] Evaluating Subject {subject_id}...")

        X, y = load_subject(x_path, y_path)
        class_counts = dict(zip(*np.unique(y, return_counts=True)))

        _tqdm_write(f"[features] X shape: {X.shape} | y shape: {y.shape} | class distribution: {class_counts}")

        try:
            results = evaluate_subject(
                X=X,
                y=y,
                models=models,
                n_splits=int(args.n_splits),
                random_state=int(args.random_state),
                tune=args.tune,
                tune_cv=int(args.tune_cv),
            )
        except Exception as e:
            print(f"Error evaluating Subject {subject_id}: {e}")
            continue

        results["subject"] = subject_id
        results["n_trials"] = X.shape[0]
        results["n_channels"] = X.shape[1]
        results["n_timepoints"] = X.shape[2]
        results["n_classes"] = len(np.unique(y))
        all_results.append(results)

    results_df = pd.DataFrame(all_results)
    if results_df.empty:
        print("No results to save.")
        return
    
    mean_cols = [c for c in results_df.columns if c.endswith("_mean_acc")]
    results_df["best_model"] = results_df[mean_cols].idxmax(axis=1).str.replace("_mean_acc", "", regex=False)
    results_df["best_acc"] = results_df[mean_cols].max(axis=1)

    results_df = results_df.sort_values("subject")
    results_df.to_csv(args.out_dir / "classification_results.csv", index=False)

    mean_cols = {
        "LDA": "LDA_mean_acc",
        "SVM": "SVM_mean_acc",
        "RF": "RF_mean_acc",
    }
    sd_cols = {
        "LDA": "LDA_mean_acc",
        "SVM": "SVM_mean_acc",
        "RF": "RF_mean_acc",
    }

    best_single_mean = float(results_df["best_acc"].mean())
    best_single_sd = float(results_df["best_acc"].std())

    summary_long = pd.DataFrame(
        [
            {"model": "LDA", "mean_acc": float(results_df[mean_cols["LDA"]].mean()), "sd_acc": float(results_df[sd_cols["LDA"]].std())},
            {"model": "SVM", "mean_acc": float(results_df[mean_cols["SVM"]].mean()), "sd_acc": float(results_df[sd_cols["SVM"]].std())},
            {"model": "RF", "mean_acc": float(results_df[mean_cols["RF"]].mean()), "sd_acc": float(results_df[sd_cols["RF"]].std())},
            {"model": "Best-Single", "mean_acc": best_single_mean, "sd_acc": best_single_sd},
        ]
    )
    summary_long.to_csv(args.out_dir / "classification_summary.csv", index=False)

    summary_wide = pd.DataFrame(
        [
            {
                "LDA_mean": float(results_df["LDA_mean_acc"].mean()),
                "SVM_mean": float(results_df["SVM_mean_acc"].mean()),
                "RF_mean": float(results_df["RF_mean_acc"].mean()),
                "LDA_subject_std": float(results_df["LDA_mean_acc"].std()),
                "SVM_subject_std": float(results_df["SVM_mean_acc"].std()),
                "RF_subject_std": float(results_df["RF_mean_acc"].std()),
                "BestSingle_mean": best_single_mean,
                "BestSingle_subject_std": best_single_sd,
            }
        ]
    )
    summary_wide.to_csv(args.out_dir / "classification_summary_wide.csv", index=False)
    print(f"Results saved to {args.out_dir / 'classification_results.csv'}")
    print(f"Summary saved to {args.out_dir / 'classification_summary.csv'}")

if __name__ == "__main__":
    main()
