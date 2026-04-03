from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
ENS_DIR = OUTPUTS_DIR / "ensemble_v2"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute paired t-tests + Levene's tests comparing ensemble confident accuracy vs baseline best-single."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=(ENS_DIR / "stats_tests_confident_vs_best.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    baseline_path = OUTPUTS_DIR / "classification_results.csv"
    if not baseline_path.exists():
        raise SystemExit(f"Missing baseline results: {baseline_path}")
    base_df = pd.read_csv(baseline_path)
    if not {"subject", "best_acc"}.issubset(base_df.columns):
        raise SystemExit(f"{baseline_path} missing columns subject,best_acc")
    base_df = base_df[["subject", "best_acc"]].copy()
    base_df["subject"] = pd.to_numeric(base_df["subject"], errors="raise").astype(int)
    base_df["best_acc"] = pd.to_numeric(base_df["best_acc"], errors="raise")

    grid_paths = sorted(ENS_DIR.glob("*_grid.csv"))
    if not grid_paths:
        raise SystemExit(f"No *_grid.csv found in {ENS_DIR}. Run ensemble_v2 and/or plot_ensembles first.")

    rows = []
    for p in grid_paths:
        df = pd.read_csv(p)
        required = {"subject", "threshold", "ensemble_acc_confident", "ensemble_coverage"}
        missing = required - set(df.columns)
        if missing:
            raise SystemExit(f"{p} missing columns: {sorted(missing)}")

        df = df.merge(base_df, on="subject", how="left")
        df["threshold"] = pd.to_numeric(df["threshold"], errors="raise")
        df["ensemble_acc_confident"] = pd.to_numeric(df["ensemble_acc_confident"], errors="coerce")
        df["ensemble_coverage"] = pd.to_numeric(df["ensemble_coverage"], errors="raise")

        ens_name = df["ensemble_name"].iloc[0] if "ensemble_name" in df.columns else p.stem

        for thr, sub in df.groupby("threshold"):
            a = sub["ensemble_acc_confident"].to_numpy(dtype=float)
            b = sub["best_acc"].to_numpy(dtype=float)
            mask = ~(np.isnan(a) | np.isnan(b))
            a2, b2 = a[mask], b[mask]

            n = int(a2.size)
            mean_cov = float(sub["ensemble_coverage"].mean())
            mean_conf = float(np.nanmean(a))
            mean_best = float(np.nanmean(b))
            mean_diff = float(np.nanmean(a - b))

            if n >= 2:
                t = stats.ttest_rel(a2, b2)
                lev = stats.levene(a2, b2, center="median")
                t_stat, t_p = float(t.statistic), float(t.pvalue)
                lev_stat, lev_p = float(lev.statistic), float(lev.pvalue)
            else:
                t_stat = t_p = lev_stat = lev_p = float("nan")

            rows.append(
                {
                    "grid_file": p.name,
                    "ensemble_name": ens_name,
                    "threshold": float(thr),
                    "n_subjects_used": n,
                    "mean_coverage": mean_cov,
                    "mean_ens_conf_acc": mean_conf,
                    "mean_best_single_acc": mean_best,
                    "mean_diff_conf_minus_best": mean_diff,
                    "paired_t_stat": t_stat,
                    "paired_t_pvalue": t_p,
                    "levene_stat": lev_stat,
                    "levene_pvalue": lev_p,
                }
            )

    out_df = pd.DataFrame(rows).sort_values(["ensemble_name", "threshold"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

