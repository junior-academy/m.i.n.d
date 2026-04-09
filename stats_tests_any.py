from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def _paired_cohens_d(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size < 2:
        return float("nan")
    sd = float(diffs.std(ddof=1))
    if sd == 0.0:
        return float("nan")
    return float(diffs.mean() / sd)


def _t_ci_mean(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(values.size)
    if n == 0:
        return float("nan"), float("nan")
    mean = float(values.mean())
    if n < 2:
        return mean, mean
    sd = float(values.std(ddof=1))
    sem = sd / (n**0.5)
    try:
        from scipy import stats  # type: ignore

        tcrit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    except Exception:
        tcrit_table = {
            1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
            11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
            21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
        }
        tcrit = float(tcrit_table.get(n - 1, 1.96))

    half = float(tcrit * sem)
    return mean - half, mean + half


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute paired t-tests + Levene's tests comparing ensemble confident accuracy vs baseline best-single "
            "for an arbitrary threshold_metrics.csv + baseline classification_results.csv."
        )
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to baseline classification_results.csv (must include columns: subject,best_acc).",
    )
    parser.add_argument(
        "--threshold-metrics",
        type=Path,
        required=True,
        help="Path to ensemble threshold_metrics.csv (must include columns: subject,threshold,ensemble_acc_confident,ensemble_coverage).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="Optional label stored as ensemble_name in the output.",
    )
    parser.add_argument(
        "--grid-file",
        type=str,
        default="",
        help=(
            "Optional grid file identifier to write into the output as `grid_file` "
            "(e.g., 'LDA_SVM_equal_grid.csv'). This makes the CSV dashboard-ready."
        ),
    )
    args = parser.parse_args()

    if not args.baseline.exists():
        raise SystemExit(f"Missing baseline file: {args.baseline}")
    if not args.threshold_metrics.exists():
        raise SystemExit(f"Missing threshold_metrics file: {args.threshold_metrics}")

    base_df = pd.read_csv(args.baseline)
    if not {"subject", "best_acc"}.issubset(base_df.columns):
        raise SystemExit(f"{args.baseline} missing columns subject,best_acc")
    base_df = base_df[["subject", "best_acc"]].copy()
    base_df["subject"] = pd.to_numeric(base_df["subject"], errors="raise").astype(int)
    base_df["best_acc"] = pd.to_numeric(base_df["best_acc"], errors="raise")

    tm = pd.read_csv(args.threshold_metrics)
    required = {"subject", "threshold", "ensemble_acc_confident", "ensemble_coverage"}
    missing = required - set(tm.columns)
    if missing:
        raise SystemExit(f"{args.threshold_metrics} missing columns: {sorted(missing)}")

    tm = tm[list(required)].copy()
    tm["subject"] = pd.to_numeric(tm["subject"], errors="raise").astype(int)
    tm["threshold"] = pd.to_numeric(tm["threshold"], errors="raise")
    tm["ensemble_acc_confident"] = pd.to_numeric(tm["ensemble_acc_confident"], errors="coerce")
    tm["ensemble_coverage"] = pd.to_numeric(tm["ensemble_coverage"], errors="raise")

    df = tm.merge(base_df, on="subject", how="left")
    df["mean_diff_conf_minus_best"] = df["ensemble_acc_confident"] - df["best_acc"]

    label = args.label.strip() or args.threshold_metrics.parent.name
    grid_file = args.grid_file.strip()

    rows = []
    for thr, sub in df.groupby("threshold"):
        a = sub["ensemble_acc_confident"].to_numpy(dtype=float)
        b = sub["best_acc"].to_numpy(dtype=float)
        mask = ~(np.isnan(a) | np.isnan(b))
        a2, b2 = a[mask], b[mask]

        n = int(a2.size)
        diffs = a2 - b2
        if n >= 2:
            try:
                from scipy import stats  # type: ignore

                t = stats.ttest_rel(a2, b2)
                lev = stats.levene(a2, b2, center="median")
                t_stat, t_p = float(t.statistic), float(t.pvalue)
                lev_stat, lev_p = float(lev.statistic), float(lev.pvalue)
            except Exception:
                t_stat = t_p = lev_stat = lev_p = float("nan")
        else:
            t_stat = t_p = lev_stat = lev_p = float("nan")

        d = _paired_cohens_d(diffs)
        ci_low, ci_high = _t_ci_mean(diffs, alpha=0.05)

        rows.append(
            {
                "grid_file": grid_file,
                "ensemble_name": label,
                "threshold": float(thr),
                "n_subjects_used": n,
                "mean_coverage": float(sub["ensemble_coverage"].mean()),
                "mean_ens_conf_acc": float(np.nanmean(a)),
                "mean_best_single_acc": float(np.nanmean(b)),
                "mean_diff_conf_minus_best": float(np.nanmean(a - b)),
                "diff_ci_low_95": float(ci_low),
                "diff_ci_high_95": float(ci_high),
                "paired_cohens_d": float(d),
                "paired_t_stat": t_stat,
                "paired_t_pvalue": t_p,
                "levene_stat": lev_stat,
                "levene_pvalue": lev_p,
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["threshold"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
