from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = BASE_DIR / "outputs"
KEY_DIR = OUTPUTS_DIR / "key_numbers"


def load_baseline_results() -> pd.DataFrame:
    path = OUTPUTS_DIR / "classification_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline per-subject results: {path}")
    df = pd.read_csv(path)
    required = {"subject", "LDA_mean_acc", "SVM_mean_acc", "RF_mean_acc", "best_model", "best_acc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df = df.copy()
    df["subject"] = df["subject"].astype(int)
    for c in ["LDA_mean_acc", "SVM_mean_acc", "RF_mean_acc", "best_acc"]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    df["best_model"] = df["best_model"].astype(str)
    return df.sort_values("subject")


def baseline_summary_from_results(results_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, col in [("LDA", "LDA_mean_acc"), ("SVM", "SVM_mean_acc"), ("RF", "RF_mean_acc")]:
        rows.append(
            {
                "model": model,
                "mean_acc": float(results_df[col].mean()),
                "sd_acc": float(results_df[col].std()),
            }
        )
    rows.append(
        {
            "model": "Best-Single",
            "mean_acc": float(results_df["best_acc"].mean()),
            "sd_acc": float(results_df["best_acc"].std()),
        }
    )
    return pd.DataFrame(rows)


def load_ensemble_grids() -> pd.DataFrame:
    grid_paths = sorted((OUTPUTS_DIR / "ensemble_v2").glob("*_grid.csv"))
    if not grid_paths:
        raise FileNotFoundError(f"No ensemble grid CSVs found in {(OUTPUTS_DIR / 'ensemble_v2')}")

    frames = []
    for p in grid_paths:
        df = pd.read_csv(p)
        if "ensemble_name" not in df.columns:
            df["ensemble_name"] = p.stem
        required = {"subject", "threshold", "ensemble_acc_all", "ensemble_acc_confident", "ensemble_coverage", "ensemble_name"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{p} missing columns: {sorted(missing)}")
        df = df.copy()
        df["subject"] = df["subject"].astype(int)
        df["threshold"] = pd.to_numeric(df["threshold"], errors="raise")
        df["ensemble_acc_all"] = pd.to_numeric(df["ensemble_acc_all"], errors="raise")
        df["ensemble_acc_confident"] = pd.to_numeric(df["ensemble_acc_confident"], errors="coerce")
        df["ensemble_coverage"] = pd.to_numeric(df["ensemble_coverage"], errors="raise")
        df["ensemble_name"] = df["ensemble_name"].astype(str)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_stats_tests() -> Optional[pd.DataFrame]:
    path = OUTPUTS_DIR / "ensemble_v2" / "stats_tests_confident_vs_best.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    needed = {"ensemble_name", "threshold", "paired_t_pvalue", "levene_pvalue", "mean_diff_conf_minus_best", "mean_coverage"}
    if not needed.issubset(df.columns):
        return None
    df = df.copy()
    df["threshold"] = pd.to_numeric(df["threshold"], errors="raise")
    return df


def select_grid_at_threshold(ens_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    sub = ens_df[ens_df["threshold"] == threshold].copy()
    if sub.empty:
        available = sorted(ens_df["threshold"].unique().tolist())
        raise ValueError(f"No ensemble grid rows for threshold={threshold}. Available: {available}")
    return sub


def main() -> None:
    parser = argparse.ArgumentParser(description="Update outputs/key_numbers from latest baseline + ensemble_v2 grids.")
    parser.add_argument("--threshold", type=float, default=0.65, help="Operating threshold to snapshot in key_numbers.")
    args = parser.parse_args()

    KEY_DIR.mkdir(parents=True, exist_ok=True)

    baseline_results = load_baseline_results()
    baseline_summary = baseline_summary_from_results(baseline_results)
    best_single_mean = float(baseline_summary[baseline_summary["model"] == "Best-Single"]["mean_acc"].iloc[0])

    ens_df = load_ensemble_grids()
    ens_at_thr = select_grid_at_threshold(ens_df, threshold=args.threshold)
    stats_df = load_stats_tests()

    # --- Per-subject table ---
    per_subject = pd.DataFrame(
        {
            "subject": baseline_results["subject"].astype(int),
            "LDA Acc": baseline_results["LDA_mean_acc"],
            "SVM Acc": baseline_results["SVM_mean_acc"],
            "RF Acc": baseline_results["RF_mean_acc"],
            "Best Single": baseline_results["best_model"],
            "Best Single Acc": baseline_results["best_acc"],
        }
    )

    # Add ensemble columns (confident acc + coverage) for each ensemble_name at chosen threshold
    for ens_name in sorted(ens_at_thr["ensemble_name"].unique()):
        block = ens_at_thr[ens_at_thr["ensemble_name"] == ens_name][
            ["subject", "ensemble_acc_confident", "ensemble_coverage"]
        ].copy()
        block = block.rename(
            columns={
                "ensemble_acc_confident": f"{ens_name} Conf Acc @ {args.threshold}",
                "ensemble_coverage": f"{ens_name} Coverage @ {args.threshold}",
            }
        )
        per_subject = per_subject.merge(block, on="subject", how="left")
        per_subject[f"{ens_name} (Conf - Best) @ {args.threshold}"] = (
            per_subject[f"{ens_name} Conf Acc @ {args.threshold}"] - per_subject["Best Single Acc"]
        )

    per_subject.insert(0, "Subject", per_subject["subject"].map(lambda s: f"S{int(s)}"))
    per_subject = per_subject.drop(columns=["subject"])

    per_subject_path = KEY_DIR / "key_numbers_per_subject.csv"
    per_subject.to_csv(per_subject_path, index=False)

    # --- Summary table ---
    summary_rows: List[Tuple[str, object]] = []
    for _, row in baseline_summary.iterrows():
        summary_rows.append((f"Mean {row['model']} accuracy", float(row["mean_acc"])))
        summary_rows.append((f"SD {row['model']} accuracy (across subjects)", float(row["sd_acc"])))

    summary_rows.append(("Operating threshold (max prob)", float(args.threshold)))

    # Ensemble summaries at threshold
    grouped = (
        ens_at_thr.groupby("ensemble_name", as_index=False)
        .agg(
            mean_conf_acc=("ensemble_acc_confident", "mean"),
            mean_cov=("ensemble_coverage", "mean"),
        )
        .sort_values("ensemble_name")
    )
    for _, row in grouped.iterrows():
        summary_rows.append((f"{row['ensemble_name']} mean confident accuracy @ {args.threshold}", float(row["mean_conf_acc"])))
        summary_rows.append((f"{row['ensemble_name']} mean coverage @ {args.threshold}", float(row["mean_cov"])))
        summary_rows.append((f"{row['ensemble_name']} mean (conf - Best) @ {args.threshold}", float(row["mean_conf_acc"] - best_single_mean)))

        if stats_df is not None:
            hit = stats_df[(stats_df["ensemble_name"] == row["ensemble_name"]) & (stats_df["threshold"] == args.threshold)]
            if not hit.empty:
                r = hit.iloc[0]
                summary_rows.append((f"{row['ensemble_name']} paired t-test p (conf vs Best) @ {args.threshold}", float(r["paired_t_pvalue"])))
                summary_rows.append((f"{row['ensemble_name']} Levene p (conf vs Best) @ {args.threshold}", float(r["levene_pvalue"])))

    # Placeholders for still-missing metrics
    summary_rows.extend(
        [
            ("Confusion matrix (4x4)", ""),
            ("Cohen's kappa", ""),
            ("Latency", ""),
        ]
    )

    summary_df = pd.DataFrame({"Metric": [r[0] for r in summary_rows], "Value": [r[1] for r in summary_rows]})
    summary_path = KEY_DIR / "key_numbers_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # --- Claim bank ---
    # Choose the best "main" candidate at this threshold by mean diff (conf - best), if available.
    claim_rows = []
    if not grouped.empty:
        grouped = grouped.copy()
        grouped["mean_diff"] = grouped["mean_conf_acc"] - best_single_mean
        best_row = grouped.sort_values("mean_diff", ascending=False).iloc[0]
        claim_rows.append(
            (
                f"At threshold {args.threshold}, some ensemble outperformed Best-Single on confident trials (mean)",
                "Yes" if best_row["mean_diff"] > 0 else "No",
                f"{best_row['ensemble_name']} mean_diff={float(best_row['mean_diff']):.6f}",
            )
        )
        claim_rows.append(
            (
                f"At threshold {args.threshold}, confident accuracy beats Best-Single while maintaining >=0.4 coverage",
                "Yes"
                if (best_row["mean_diff"] > 0 and float(best_row["mean_cov"]) >= 0.4)
                else "No",
                f"{best_row['ensemble_name']} mean_conf={float(best_row['mean_conf_acc']):.6f}, mean_cov={float(best_row['mean_cov']):.6f}",
            )
        )

    claim_rows.extend(
        [
            ("System shows agreement beyond chance", "Missing", "Cohen's kappa"),
            ("System is fast enough for practical use", "Missing", "Latency"),
        ]
    )
    claims_df = pd.DataFrame({"Claim": [c[0] for c in claim_rows], "Supported?": [c[1] for c in claim_rows], "Metric that proves it": [c[2] for c in claim_rows]})
    claims_path = KEY_DIR / "key_numbers_claim_bank.csv"
    claims_df.to_csv(claims_path, index=False)

    print(f"Wrote {per_subject_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {claims_path}")


if __name__ == "__main__":
    main()
