"""Wilcoxon inference for the pre-registered held-out operating point."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from config import OUTPUT_DIR


def summarize(subject_results: Path, out_path: Path) -> Path:
    df = pd.read_csv(subject_results)
    required = {
        "subject",
        "coverage",
        "ensemble_acc_matched_coverage",
        "lda_acc_matched_coverage",
        "delta_ensemble_minus_lda",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{subject_results} missing columns: {sorted(missing)}")

    deltas = df["delta_ensemble_minus_lda"].to_numpy(dtype=float)
    deltas = deltas[np.isfinite(deltas)]
    n = int(deltas.size)
    if n < 4:
        stat = p_value = float("nan")
        note = "non-inferential: fewer than 4 paired subjects"
    else:
        test = stats.wilcoxon(deltas, zero_method="wilcox", alternative="greater")
        stat = float(test.statistic)
        p_value = float(test.pvalue)
        note = "Wilcoxon signed-rank, one-sided ensemble > LDA; N=9 is limited."

    summary = pd.DataFrame(
        [
            {
                "n_subjects": n,
                "coverage": float(df["coverage"].iloc[0]),
                "mean_ensemble_acc": float(df["ensemble_acc_matched_coverage"].mean()),
                "mean_lda_acc": float(df["lda_acc_matched_coverage"].mean()),
                "mean_delta_ensemble_minus_lda": float(np.mean(deltas)) if n else float("nan"),
                "median_delta_ensemble_minus_lda": float(np.median(deltas)) if n else float("nan"),
                "wilcoxon_statistic": stat,
                "wilcoxon_p_greater": p_value,
                "note": note,
            }
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    df[["subject", "delta_ensemble_minus_lda"]].to_csv(out_path.parent / "per_subject_deltas.csv", index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize held-out matched-coverage statistics.")
    parser.add_argument("--subject-results", type=Path, default=OUTPUT_DIR / "subject_results.csv")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "headline_stats.csv")
    args = parser.parse_args()
    out = summarize(args.subject_results, args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
