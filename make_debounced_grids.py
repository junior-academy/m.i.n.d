from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gating import DebounceConfig, debounced_gate, toggle_rate, wrong_fire_rate_all


BASE_DIR = Path(__file__).resolve().parent


def _prob_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.startswith("p_ens_c")]

    def _key(c: str) -> int:
        try:
            return int(c.replace("p_ens_c", ""))
        except Exception:
            return 10**9

    return sorted(cols, key=_key)


def _latest_dir(root: Path, glob_pat: str) -> Optional[Path]:
    cands = [p for p in root.glob(glob_pat) if p.is_dir()]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def _load_subject_predictions(run_dir: Path, subject: int) -> pd.DataFrame:
    p = run_dir / f"predictions_subject_{subject}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing predictions CSV: {p}")
    df = pd.read_csv(p)
    needed = {"trial_index", "ensemble_pred", "ensemble_max_prob"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{p} missing columns: {sorted(missing)}")
    if "y_true" not in df.columns:
        raise ValueError(f"{p} missing column y_true (needed to compute debounced accuracy)")
    cols = _prob_cols(df)
    if not cols:
        raise ValueError(f"{p} missing p_ens_c* probability columns")
    df = df.sort_values("trial_index").reset_index(drop=True)
    return df


def _thresholds_from_run(run_dir: Path) -> List[float]:
    tm = run_dir / "threshold_metrics.csv"
    if not tm.exists():
        raise FileNotFoundError(f"Missing threshold_metrics.csv in {run_dir} (needed to infer threshold grid)")
    df = pd.read_csv(tm)
    if "threshold" not in df.columns:
        raise ValueError(f"{tm} missing column threshold")
    thrs = sorted(set(float(x) for x in pd.to_numeric(df["threshold"], errors="raise").tolist()))
    return thrs


def _debounced_metrics_for_subject(
    *,
    df_pred: pd.DataFrame,
    thresholds: List[float],
    off_gap: float,
    k: int,
    n: int,
) -> pd.DataFrame:
    prob_cols = _prob_cols(df_pred)
    p_ens = df_pred[prob_cols].to_numpy(dtype=float)
    y_true = pd.to_numeric(df_pred["y_true"], errors="raise").to_numpy(dtype=int)
    y_hat = p_ens.argmax(axis=1).astype(int)

    rows: List[Dict[str, float]] = []
    for t in thresholds:
        t_on = float(t)
        t_off = float(max(0.0, t_on - float(off_gap)))
        cfg = DebounceConfig(t_on=t_on, t_off=t_off, k=int(k), n=int(n))
        fired, latched = debounced_gate(p_ens=p_ens, y_hat=y_hat, cfg=cfg)
        # Fill preds for fired trials using latched; non-fired irrelevant for confident acc.
        y_pred = np.where(fired, latched, -1)

        acc_all = float(np.mean(y_hat == y_true))
        cov = float(np.mean(fired))
        if fired.any():
            acc_conf = float(np.mean(y_pred[fired] == y_true[fired]))
        else:
            acc_conf = float("nan")

        rows.append(
            {
                "threshold": float(t_on),
                "ensemble_acc_all": acc_all,
                "ensemble_acc_confident": acc_conf,
                "ensemble_coverage": cov,
                "toggle_rate": toggle_rate(fired),
                "wrong_fire_rate_all": wrong_fire_rate_all(y_true=y_true, y_pred=y_pred, fired=fired),
                "t_on": float(t_on),
                "t_off": float(t_off),
                "k": float(k),
                "n": float(n),
            }
        )
    return pd.DataFrame(rows)


def _process_run(
    *,
    run_dir: Path,
    subjects: List[int],
    out_grid_path: Path,
    off_gap: float,
    k: int,
    n: int,
) -> None:
    thresholds = _thresholds_from_run(run_dir)

    frames = []
    actual_subjects: List[int] = []
    for s in subjects:
        if (run_dir / f"predictions_subject_{int(s)}.csv").exists():
            actual_subjects.append(int(s))
    if not actual_subjects:
        # Fallback: discover from filenames.
        for p in sorted(run_dir.glob("predictions_subject_*.csv")):
            try:
                sid = int(p.stem.split("_")[-1])
                actual_subjects.append(sid)
            except Exception:
                continue
    actual_subjects = sorted(set(actual_subjects))

    for s in actual_subjects:
        df_pred = _load_subject_predictions(run_dir, subject=int(s))
        sub = _debounced_metrics_for_subject(df_pred=df_pred, thresholds=thresholds, off_gap=off_gap, k=k, n=n)
        sub.insert(0, "subject", int(s))
        frames.append(sub)

    out = pd.concat(frames, ignore_index=True).sort_values(["subject", "threshold"])

    # Persist debounced threshold metrics inside the run folder for traceability.
    out.to_csv(run_dir / "threshold_metrics_debounced.csv", index=False)

    # Also write a dashboard-style grid CSV at a stable path.
    out_grid_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_grid_path, index=False)

    print(f"[debounce] wrote {run_dir / 'threshold_metrics_debounced.csv'}")
    print(f"[debounce] wrote {out_grid_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate debounced (hysteresis + k-of-n) threshold grid CSVs from existing predictions_subject_*.csv.\n\n"
            "This does NOT retrain any models. It post-processes OOF per-trial probabilities into a more stable gate "
            "and exports dashboard-ready *_debounced_grid.csv files."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=(BASE_DIR / "outputs" / "ensemble_v2"),
        help="Root outputs directory containing models-* run folders (default: m.i.n.d/outputs/ensemble_v2).",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default="1,2,3,4,5,6,7,8,9",
        help="Comma-separated subjects to include (default: 1..9).",
    )
    parser.add_argument("--off-gap", type=float, default=0.05, help="t_off = max(0, t_on - off_gap) (default: 0.05).")
    parser.add_argument("--k", type=int, default=3, help="k in k-of-n confirmation (default: 3).")
    parser.add_argument("--n", type=int, default=5, help="n in k-of-n confirmation (default: 5).")
    parser.add_argument(
        "--off-gap-grid",
        type=str,
        default="",
        help="Optional comma-separated off-gap values to sweep (e.g., '0.05,0.10'). Overrides --off-gap when set.",
    )
    parser.add_argument(
        "--k-grid",
        type=str,
        default="",
        help="Optional comma-separated k values to sweep (e.g., '3,4'). Overrides --k when set.",
    )
    parser.add_argument(
        "--n-grid",
        type=str,
        default="",
        help="Optional comma-separated n values to sweep (e.g., '5,7'). Overrides --n when set.",
    )
    parser.add_argument(
        "--also-3a",
        action="store_true",
        help="Also generate debounced grids for validation_3a outputs (writes under m.i.n.d/outputs/validation_3a/).",
    )
    args = parser.parse_args()

    subjects = [int(x.strip()) for x in args.subjects.split(",") if x.strip()]
    root = args.root

    off_gaps = (
        [float(x.strip()) for x in args.off_gap_grid.split(",") if x.strip()]
        if args.off_gap_grid.strip()
        else [float(args.off_gap)]
    )
    ks = [int(x.strip()) for x in args.k_grid.split(",") if x.strip()] if args.k_grid.strip() else [int(args.k)]
    ns = [int(x.strip()) for x in args.n_grid.split(",") if x.strip()] if args.n_grid.strip() else [int(args.n)]

    # 2a: map stable filenames -> latest run dir patterns
    specs: List[Tuple[str, str]] = [
        ("LDA_SVM_equal_debounced_grid.csv", "models-LDA_SVM__weights-equal*"),
        ("LDA_SVM_baseline_subject_debounced_grid.csv", "models-LDA_SVM__weights-baseline_subject*"),
        ("LDA_SVM_RF_global_debounced_grid.csv", "models-LDA_SVM_RF__weights-baseline_global*"),
    ]

    for out_name, pat in specs:
        run_dir = _latest_dir(root, pat)
        if run_dir is None:
            print(f"[debounce] skip (no run dir): {pat}")
            continue
        frames: List[pd.DataFrame] = []
        for off_gap in off_gaps:
            for k in ks:
                for n in ns:
                    thresholds = _thresholds_from_run(run_dir)

                    actual_subjects: List[int] = []
                    for s in subjects:
                        if (run_dir / f"predictions_subject_{int(s)}.csv").exists():
                            actual_subjects.append(int(s))
                    if not actual_subjects:
                        for p in sorted(run_dir.glob("predictions_subject_*.csv")):
                            try:
                                sid = int(p.stem.split("_")[-1])
                                actual_subjects.append(sid)
                            except Exception:
                                continue
                    actual_subjects = sorted(set(actual_subjects))

                    for s in actual_subjects:
                        df_pred = _load_subject_predictions(run_dir, subject=int(s))
                        sub = _debounced_metrics_for_subject(
                            df_pred=df_pred, thresholds=thresholds, off_gap=float(off_gap), k=int(k), n=int(n)
                        )
                        sub.insert(0, "subject", int(s))
                        frames.append(sub)

        out = pd.concat(frames, ignore_index=True).sort_values(["subject", "threshold", "t_off", "k", "n"])
        out.to_csv(run_dir / "threshold_metrics_debounced.csv", index=False)
        out_grid_path = root / out_name
        out_grid_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_grid_path, index=False)
        print(f"[debounce] wrote {run_dir / 'threshold_metrics_debounced.csv'}")
        print(f"[debounce] wrote {out_grid_path}")

    if args.also_3a:
        val_root = BASE_DIR / "outputs" / "validation_3a" / "ensemble_v2"
        if val_root.exists():
            # IIIa only has LDA+SVM variants in this repo.
            specs_3a: List[Tuple[str, str]] = [
                ("LDA_SVM_equal_debounced_grid.csv", "models-LDA_SVM__weights-equal*"),
                ("LDA_SVM_baseline_subject_debounced_grid.csv", "models-LDA_SVM__weights-baseline_subject*"),
            ]
            for out_name, pat in specs_3a:
                run_dir = _latest_dir(val_root, pat)
                if run_dir is None:
                    print(f"[debounce] (3a) skip (no run dir): {pat}")
                    continue
                frames: List[pd.DataFrame] = []
                for off_gap in off_gaps:
                    for k in ks:
                        for n in ns:
                            thresholds = _thresholds_from_run(run_dir)

                            actual_subjects: List[int] = []
                            for s in subjects:
                                if (run_dir / f"predictions_subject_{int(s)}.csv").exists():
                                    actual_subjects.append(int(s))
                            if not actual_subjects:
                                for p in sorted(run_dir.glob("predictions_subject_*.csv")):
                                    try:
                                        sid = int(p.stem.split("_")[-1])
                                        actual_subjects.append(sid)
                                    except Exception:
                                        continue
                            actual_subjects = sorted(set(actual_subjects))

                            for s in actual_subjects:
                                df_pred = _load_subject_predictions(run_dir, subject=int(s))
                                sub = _debounced_metrics_for_subject(
                                    df_pred=df_pred,
                                    thresholds=thresholds,
                                    off_gap=float(off_gap),
                                    k=int(k),
                                    n=int(n),
                                )
                                sub.insert(0, "subject", int(s))
                                frames.append(sub)

                out = pd.concat(frames, ignore_index=True).sort_values(["subject", "threshold", "t_off", "k", "n"])
                out.to_csv(run_dir / "threshold_metrics_debounced.csv", index=False)
                out_grid_path = val_root / out_name
                out_grid_path.parent.mkdir(parents=True, exist_ok=True)
                out.to_csv(out_grid_path, index=False)
                print(f"[debounce] (3a) wrote {run_dir / 'threshold_metrics_debounced.csv'}")
                print(f"[debounce] (3a) wrote {out_grid_path}")
        else:
            print("[debounce] validation_3a not found; skipping")


if __name__ == "__main__":
    main()
