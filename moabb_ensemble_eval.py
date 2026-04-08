from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def _require(pkg: str, hint: str):
    try:
        return __import__(pkg)
    except ModuleNotFoundError as e:
        raise SystemExit(f"Missing dependency: {pkg}\n\nInstall:\n  {hint}\n") from e


def _parse_csv_floats(arg: str) -> List[float]:
    out: List[float] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise SystemExit("threshold grid must not be empty")
    return out


def _parse_csv_ints(arg: str) -> Optional[List[int]]:
    arg = arg.strip()
    if not arg:
        return None
    out: List[int] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out or None


def _parse_csv_strs(arg: str) -> List[str]:
    arg = arg.strip()
    if not arg:
        return []
    return [p.strip() for p in arg.split(",") if p.strip()]


def _softmax_safe(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.nanmax(x, axis=1, keepdims=True)
    ex = np.exp(np.clip(x, -50, 50))
    denom = np.sum(ex, axis=1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return ex / denom


def _normalize_weights(w: Dict[str, float]) -> Dict[str, float]:
    s = float(sum(max(0.0, v) for v in w.values()))
    if s <= 0:
        n = len(w)
        return {k: 1.0 / n for k in w}
    return {k: max(0.0, v) / s for k, v in w.items()}


def _paired_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    if d.size < 2:
        return float("nan")
    sd = float(np.std(d, ddof=1))
    return float(np.mean(d) / sd) if sd > 0 else float("nan")


def _ci95(xs: np.ndarray) -> Tuple[float, float]:
    xs = np.asarray(xs, dtype=float)
    xs = xs[np.isfinite(xs)]
    if xs.size < 2:
        return float("nan"), float("nan")
    mean = float(xs.mean())
    se = float(xs.std(ddof=1) / np.sqrt(xs.size))
    tcrit = float(stats.t.ppf(0.975, df=xs.size - 1))
    return mean - tcrit * se, mean + tcrit * se


@dataclass
class OOFProba:
    subject: int
    y_true: np.ndarray  # (n_trials,)
    probas: Dict[str, np.ndarray]  # model -> (n_trials, n_classes)
    classes: np.ndarray  # (n_classes,) integer-encoded class ids (0..K-1)
    class_labels: Optional[List[str]] = None  # original labels if available (e.g., strings)


def _make_base_estimators(models: List[str], random_state: int):
    _require("sklearn", "python3 -m pip install scikit-learn")
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    est = {}
    for m in models:
        if m == "LDA":
            est[m] = LDA()
        elif m == "SVM":
            est[m] = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=random_state)
        elif m == "RF":
            est[m] = RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=1,
                random_state=random_state,
                n_jobs=1,
            )
        else:
            raise SystemExit(f"Unknown model: {m}")
    return est


def _fit_oof_probas_for_subject(
    X: np.ndarray,
    y: np.ndarray,
    subject: int,
    models: List[str],
    n_splits: int,
    random_state: int,
    calibrate: str,
    calibration_cv: int,
    n_components: int,
) -> OOFProba:
    _require("mne", "python3 -m pip install mne")
    _require("sklearn", "python3 -m pip install scikit-learn")

    from mne.decoding import CSP
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder

    X = np.asarray(X, dtype=float)
    y_raw = np.asarray(y)
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    # Store original labels for reference (useful when MOABB returns strings).
    class_labels = [str(c) for c in le.classes_.tolist()]
    classes = np.arange(int(le.classes_.size), dtype=int)
    n_classes = int(classes.size)

    base = _make_base_estimators(models=models, random_state=random_state)
    probas: Dict[str, np.ndarray] = {m: np.full((X.shape[0], n_classes), np.nan, dtype=float) for m in models}

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for fold, (tr, te) in enumerate(cv.split(X, y_enc), start=1):
        X_tr, y_tr = X[tr], y_enc[tr]
        X_te, y_te = X[te], y_enc[te]
        # CSP works on (n_trials, n_channels, n_times) directly in MNE.
        for m in models:
            clf = base[m]
            if calibrate != "none" and m in ("SVM", "RF"):
                clf = CalibratedClassifierCV(clf, method=calibrate, cv=calibration_cv)

            pipe = Pipeline(
                [
                    ("csp", CSP(n_components=n_components, reg="ledoit_wolf", log=True, norm_trace=False)),
                    ("clf", clf),
                ]
            )
            pipe.fit(X_tr, y_tr)
            p = pipe.predict_proba(X_te)

            # Align columns to `classes` ordering.
            if hasattr(pipe[-1], "classes_"):
                cls = pipe[-1].classes_
            else:
                cls = classes
            aligned = np.full((p.shape[0], n_classes), np.nan, dtype=float)
            for j, c in enumerate(cls):
                idx = int(c)
                aligned[:, idx] = p[:, j]
            probas[m][te] = aligned

    return OOFProba(subject=subject, y_true=y_enc.astype(int), probas=probas, classes=classes, class_labels=class_labels)


def _model_acc(oof: OOFProba, model: str) -> float:
    p = oof.probas[model]
    ok = np.isfinite(p).all(axis=1)
    if not ok.any():
        return float("nan")
    pred = np.argmax(p[ok], axis=1)
    yt = oof.y_true[ok]
    return float((pred == yt).mean())


def _compute_best_single(oof: OOFProba, models: List[str]) -> Tuple[str, float]:
    best_m = models[0]
    best_acc = -1.0
    for m in models:
        acc = _model_acc(oof, m)
        if np.isfinite(acc) and acc > best_acc:
            best_acc = acc
            best_m = m
    return best_m, float(best_acc)


def _ensemble_proba(oof: OOFProba, models: List[str], weights: Dict[str, float]) -> np.ndarray:
    w = _normalize_weights({m: float(weights.get(m, 0.0)) for m in models})
    ps = [oof.probas[m] * w[m] for m in models]
    p = np.sum(ps, axis=0)
    # Some models can yield non-normalized probas due to numerical issues; renormalize.
    row_sums = np.sum(p, axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return p / row_sums


def _threshold_metrics_for_subject(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: List[float],
) -> pd.DataFrame:
    pred = np.argmax(proba, axis=1)
    maxp = np.max(proba, axis=1)
    acc_all = float((pred == y_true).mean())

    rows = []
    for t in thresholds:
        mask = maxp >= float(t)
        cov = float(mask.mean())
        if mask.any():
            acc_conf = float((pred[mask] == y_true[mask]).mean())
        else:
            acc_conf = float("nan")
        rows.append(
            {
                "threshold": float(t),
                "ensemble_acc_all": acc_all,
                "ensemble_acc_confident": acc_conf,
                "ensemble_coverage": cov,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MOABB-powered ensemble evaluation.\n\n"
            "Loads an MI dataset via MOABB, computes per-subject out-of-fold probabilities for base models "
            "(CSP + classifier), and then evaluates ensemble variants across a threshold grid.\n\n"
            "Outputs are written in the same 'grid CSV' style used by the M.I.N.D dashboard."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="bnci2014_001",
        choices=["bnci2014_001", "physionetmi"],
        help="MOABB dataset (default: bnci2014_001).",
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=4,
        choices=[2, 3, 4],
        help="Number of classes for MotorImagery paradigm (default: 4).",
    )
    parser.add_argument(
        "--events",
        type=str,
        default="",
        help="Comma-separated events to include. Recommended for PhysionetMI 2-class: left_hand,right_hand.",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default="",
        help="Comma-separated subject ids. Empty = dataset default subject_list.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("m.i.n.d/outputs/moabb_ensemble"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-components", type=int, default=6)
    parser.add_argument(
        "--models",
        type=str,
        default="LDA,SVM,RF",
        help="Comma-separated base models subset: LDA,SVM,RF (default: LDA,SVM,RF).",
    )
    parser.add_argument(
        "--calibrate",
        type=str,
        default="sigmoid",
        choices=["none", "sigmoid", "isotonic"],
        help="Probability calibration for SVM/RF (default: sigmoid).",
    )
    parser.add_argument("--calibration-cv", type=int, default=3)
    parser.add_argument(
        "--threshold-grid",
        type=str,
        default="0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00",
        help="Comma-separated thresholds for confident-mode metrics.",
    )
    parser.add_argument(
        "--mne-data",
        type=Path,
        default=None,
        help="Optional MNE_DATA directory for caching dataset downloads.",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    _require("moabb", "python3 -m pip install moabb")
    _require("mne", "python3 -m pip install mne")
    _require("sklearn", "python3 -m pip install scikit-learn")

    if args.mne_data is not None:
        os.environ["MNE_DATA"] = str(args.mne_data.expanduser().resolve())

    from moabb.datasets import BNCI2014_001, PhysionetMI
    from moabb.paradigms import MotorImagery

    thresholds = _parse_csv_floats(args.threshold_grid)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    subjects = _parse_csv_ints(args.subjects)
    events = _parse_csv_strs(args.events)

    if args.dataset == "physionetmi" and args.n_classes == 2 and not events:
        events = ["left_hand", "right_hand"]

    ds = PhysionetMI() if args.dataset == "physionetmi" else BNCI2014_001()
    if subjects is not None:
        ds.subject_list = subjects

    if events:
        paradigm = MotorImagery(events=events, n_classes=min(args.n_classes, len(events)))
    else:
        paradigm = MotorImagery(n_classes=args.n_classes)

    X, y, meta = paradigm.get_data(dataset=ds, subjects=ds.subject_list, return_epochs=False)
    meta = meta.reset_index(drop=True)
    if "subject" not in meta.columns:
        raise SystemExit("MOABB metadata missing 'subject' column.")

    out_root = args.out_dir / args.dataset
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = {
        "dataset": args.dataset,
        "n_classes": args.n_classes,
        "events": events,
        "subjects": ds.subject_list,
        "n_splits": args.n_splits,
        "random_state": args.random_state,
        "n_components": args.n_components,
        "models": models,
        "calibrate": args.calibrate,
        "calibration_cv": args.calibration_cv,
        "threshold_grid": thresholds,
    }
    (out_root / "config.json").write_text(json.dumps(cfg, indent=2))

    # 1) Compute OOF probas for each subject and model.
    oofs: List[OOFProba] = []
    for subj in sorted(meta["subject"].unique().tolist()):
        idx = meta.index[meta["subject"] == subj].to_numpy()
        oof = _fit_oof_probas_for_subject(
            X=X[idx],
            y=y[idx],
            subject=int(subj),
            models=models,
            n_splits=int(args.n_splits),
            random_state=int(args.random_state),
            calibrate=str(args.calibrate),
            calibration_cv=int(args.calibration_cv),
            n_components=int(args.n_components),
        )
        oofs.append(oof)

    # 2) Baselines table (best-single per subject).
    baseline_rows = []
    per_subject_accs: Dict[int, Dict[str, float]] = {}
    for oof in oofs:
        accs = {m: _model_acc(oof, m) for m in models}
        per_subject_accs[oof.subject] = accs
        best_m, best_acc = _compute_best_single(oof, models=models)
        baseline_rows.append(
            {
                "subject": oof.subject,
                **{f"{m}_acc": accs[m] for m in models},
                "best_model": best_m,
                "best_acc": best_acc,
                "n_trials": int(oof.y_true.size),
                "n_classes": int(oof.classes.size),
            }
        )
    baseline_df = pd.DataFrame(baseline_rows).sort_values("subject")
    baseline_path = out_root / "classification_results.csv"
    baseline_df.to_csv(baseline_path, index=False)

    # 3) Global weights (computed from per-subject model accuracies).
    global_w = {m: float(np.nanmean([per_subject_accs[s][m] for s in per_subject_accs])) for m in models}
    global_w = _normalize_weights(global_w)

    # 4) Evaluate ensemble variants across threshold grid.
    grid_dir = out_root / "ensemble_v2"
    grid_dir.mkdir(parents=True, exist_ok=True)

    variants: List[Tuple[str, List[str], str]] = [
        ("LDA_SVM_equal", ["LDA", "SVM"], "equal"),
        ("LDA_SVM_subject", ["LDA", "SVM"], "baseline_subject"),
        ("LDA_SVM_RF_equal", ["LDA", "SVM", "RF"], "equal"),
    ]
    # Filter variants that reference missing models.
    variants = [v for v in variants if all(m in models for m in v[1])]

    all_grids: Dict[str, pd.DataFrame] = {}
    for name, ens_models, weight_mode in variants:
        rows = []
        for oof in oofs:
            if weight_mode == "equal":
                w = {m: 1.0 for m in ens_models}
            elif weight_mode == "baseline_subject":
                w = {m: float(per_subject_accs[oof.subject].get(m, 0.0)) for m in ens_models}
            elif weight_mode == "baseline_global":
                w = {m: float(global_w.get(m, 0.0)) for m in ens_models}
            else:
                raise SystemExit(f"Unknown weight mode: {weight_mode}")

            p_ens = _ensemble_proba(oof, models=ens_models, weights=w)
            tm = _threshold_metrics_for_subject(oof.y_true, p_ens, thresholds=thresholds)
            tm.insert(0, "subject", int(oof.subject))
            tm["ensemble_name"] = name
            rows.append(tm)

        grid = pd.concat(rows, axis=0, ignore_index=True)
        grid_path = grid_dir / f"{name}_grid.csv"
        grid.to_csv(grid_path, index=False)
        all_grids[name] = grid

    # 5) Stats tests vs best-single (per threshold, per grid file).
    stats_rows = []
    for name, grid in all_grids.items():
        # Join best_acc per subject.
        grid2 = grid.merge(baseline_df[["subject", "best_acc"]], on="subject", how="left")
        for thr, sub in grid2.groupby("threshold"):
            a = pd.to_numeric(sub["ensemble_acc_confident"], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(sub["best_acc"], errors="coerce").to_numpy(dtype=float)
            mask = ~(np.isnan(a) | np.isnan(b))
            a2, b2 = a[mask], b[mask]
            n = int(a2.size)
            if n >= 2:
                t = stats.ttest_rel(a2, b2)
                lev = stats.levene(a2, b2, center="median")
                t_stat, t_p = float(t.statistic), float(t.pvalue)
                lev_stat, lev_p = float(lev.statistic), float(lev.pvalue)
                d = _paired_cohens_d(a2, b2)
            else:
                t_stat = t_p = lev_stat = lev_p = d = float("nan")

            stats_rows.append(
                {
                    "grid_file": f"{name}_grid.csv",
                    "ensemble_name": name,
                    "threshold": float(thr),
                    "n_subjects_used": n,
                    "mean_coverage": float(pd.to_numeric(sub["ensemble_coverage"], errors="coerce").mean()),
                    "mean_ens_conf_acc": float(np.nanmean(a)),
                    "mean_best_single_acc": float(np.nanmean(b)),
                    "mean_diff_conf_minus_best": float(np.nanmean(a - b)),
                    "paired_t_stat": t_stat,
                    "paired_t_pvalue": t_p,
                    "levene_stat": lev_stat,
                    "levene_pvalue": lev_p,
                    "paired_cohens_d": d,
                }
            )
    stats_df = pd.DataFrame(stats_rows).sort_values(["ensemble_name", "threshold"])
    stats_path = grid_dir / "stats_tests_confident_vs_best.csv"
    stats_df.to_csv(stats_path, index=False)

    # 6) Run summary per variant at an operating point (use threshold closest to 0.60 if present).
    op = 0.60
    run_rows = []
    for name, grid in all_grids.items():
        thrs = sorted(grid["threshold"].unique().tolist())
        chosen = min(thrs, key=lambda t: abs(float(t) - op)) if thrs else op
        sub = grid[grid["threshold"] == chosen]
        for metric in ["ensemble_acc_all", "ensemble_acc_confident", "ensemble_coverage"]:
            xs = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
            n = int(np.isfinite(xs).sum())
            mean = float(np.nanmean(xs))
            sd = float(np.nanstd(xs, ddof=1)) if n >= 2 else float("nan")
            ci_low, ci_high = _ci95(xs)
            run_rows.append(
                {
                    "ensemble_name": name,
                    "threshold": float(chosen),
                    "metric": metric,
                    "n": n,
                    "mean": mean,
                    "sd": sd,
                    "ci_low_95": ci_low,
                    "ci_high_95": ci_high,
                }
            )
    run_df = pd.DataFrame(run_rows)
    run_path = grid_dir / "run_summary.csv"
    run_df.to_csv(run_path, index=False)

    print("MOABB ensemble eval complete.")
    print(f"- dataset={args.dataset} n_subjects={len(oofs)} n_classes={int(np.unique(y).size)}")
    print(f"- wrote: {baseline_path}")
    print(f"- wrote: {stats_path}")
    print(f"- wrote: {run_path}")
    print(f"- wrote grids: {grid_dir}")


if __name__ == "__main__":
    main()
