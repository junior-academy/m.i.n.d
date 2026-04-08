from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


def _paired_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for paired samples (mean(diff) / sd(diff))."""
    d = a - b
    sd = float(np.std(d, ddof=1)) if d.size >= 2 else float("nan")
    return float(np.mean(d) / sd) if sd and np.isfinite(sd) else float("nan")


def _all_pairs(items: list[str]) -> Iterable[tuple[str, str]]:
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            yield items[i], items[j]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute paired t-tests + Levene's tests for MOABB evaluation results.\n\n"
            "Inputs are typically produced by `m.i.n.d/moabb_eval_physionet.py`:\n"
            "  - results_raw.csv (fold-level)\n"
            "  - results_per_subject.csv (pipeline x subject mean)\n\n"
            "This script compares pipelines using per-subject mean scores."
        )
    )
    parser.add_argument(
        "--results-per-subject",
        type=Path,
        required=True,
        help="Path to results_per_subject.csv (columns: pipeline,subject,score_mean).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output CSV path (pipeline comparison stats).",
    )
    parser.add_argument(
        "--pipelines",
        type=str,
        default="",
        help="Optional comma-separated pipeline names to include (default: all in file).",
    )
    args = parser.parse_args()

    if not args.results_per_subject.exists():
        raise SystemExit(f"Missing file: {args.results_per_subject}")

    df = pd.read_csv(args.results_per_subject)
    required = {"pipeline", "subject", "score_mean"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{args.results_per_subject} missing columns: {sorted(missing)}")

    df = df[list(required)].copy()
    df["pipeline"] = df["pipeline"].astype(str)
    df["subject"] = pd.to_numeric(df["subject"], errors="raise").astype(int)
    df["score_mean"] = pd.to_numeric(df["score_mean"], errors="coerce")

    if args.pipelines.strip():
        keep = {p.strip() for p in args.pipelines.split(",") if p.strip()}
        df = df[df["pipeline"].isin(keep)].copy()

    pipelines = sorted(df["pipeline"].unique().tolist())
    if len(pipelines) < 2:
        raise SystemExit("Need at least 2 pipelines to compare.")

    wide = df.pivot_table(index="subject", columns="pipeline", values="score_mean", aggfunc="mean")

    rows = []
    for p1, p2 in _all_pairs(pipelines):
        a = wide[p1].to_numpy(dtype=float)
        b = wide[p2].to_numpy(dtype=float)
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

        rows.append(
            {
                "pipeline_a": p1,
                "pipeline_b": p2,
                "n_subjects_used": n,
                "mean_a": float(np.nanmean(a2)),
                "mean_b": float(np.nanmean(b2)),
                "mean_diff_a_minus_b": float(np.nanmean(a2 - b2)),
                "paired_t_stat": t_stat,
                "paired_t_pvalue": t_p,
                "levene_stat": lev_stat,
                "levene_pvalue": lev_p,
                "paired_cohens_d": d,
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["pipeline_a", "pipeline_b"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

