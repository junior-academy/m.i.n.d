from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from stats_utils import levene_two_groups, paired_ttest_rel


def _paired_cohens_d(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size < 2:
        return float("nan")
    mean = float(diffs.mean())
    sd = float(diffs.std(ddof=1))
    if sd == 0.0:
        if mean == 0.0:
            return 0.0
        return float("inf") if mean > 0 else float("-inf")
    return float(mean / sd)


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
            1: 12.706,
            2: 4.303,
            3: 3.182,
            4: 2.776,
            5: 2.571,
            6: 2.447,
            7: 2.365,
            8: 2.306,
            9: 2.262,
            10: 2.228,
            11: 2.201,
            12: 2.179,
            13: 2.160,
            14: 2.145,
            15: 2.131,
            16: 2.120,
            17: 2.110,
            18: 2.101,
            19: 2.093,
            20: 2.086,
            21: 2.080,
            22: 2.074,
            23: 2.069,
            24: 2.064,
            25: 2.060,
            26: 2.056,
            27: 2.052,
            28: 2.048,
            29: 2.045,
            30: 2.042,
        }
        tcrit = float(tcrit_table.get(n - 1, 1.96))
    half = float(tcrit * sem)
    return mean - half, mean + half


def _find_baseline_for_ensemble_dir(ensemble_v2_dir: Path) -> Path | None:
    """
    Heuristic mapping:
    - outputs/ensemble_v2 -> outputs/classification_results.csv
    - outputs/<dataset>/ensemble_v2 -> outputs/<dataset>/classification_results.csv or outputs/<dataset>/baselines/classification_results.csv
    """
    candidates = [
        ensemble_v2_dir.parent / "baselines" / "classification_results.csv",
        ensemble_v2_dir.parent / "classification_results.csv",
        ensemble_v2_dir.parent.parent / "classification_results.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def compute_stats_for_folder(*, ensemble_v2_dir: Path, baseline_path: Path) -> pd.DataFrame:
    base_df = pd.read_csv(baseline_path)
    if not {"subject", "best_acc"}.issubset(base_df.columns):
        raise SystemExit(f"{baseline_path} missing columns subject,best_acc")
    base_df = base_df[["subject", "best_acc"]].copy()
    base_df["subject"] = pd.to_numeric(base_df["subject"], errors="raise").astype(int)
    base_df["best_acc"] = pd.to_numeric(base_df["best_acc"], errors="raise")

    grid_paths = sorted(ensemble_v2_dir.glob("*_grid.csv"))
    if not grid_paths:
        raise SystemExit(f"No *_grid.csv found in {ensemble_v2_dir}")

    rows: list[dict] = []
    for p in grid_paths:
        df = pd.read_csv(p)
        required = {"subject", "threshold", "ensemble_acc_confident", "ensemble_coverage"}
        missing = required - set(df.columns)
        if missing:
            raise SystemExit(f"{p} missing columns: {sorted(missing)}")

        df = df.merge(base_df, on="subject", how="left")
        df["subject"] = pd.to_numeric(df["subject"], errors="raise").astype(int)
        df["threshold"] = pd.to_numeric(df["threshold"], errors="raise")
        df["ensemble_acc_confident"] = pd.to_numeric(df["ensemble_acc_confident"], errors="coerce")
        df["ensemble_coverage"] = pd.to_numeric(df["ensemble_coverage"], errors="raise")
        df["best_acc"] = pd.to_numeric(df["best_acc"], errors="coerce")

        ens_name = df["ensemble_name"].iloc[0] if "ensemble_name" in df.columns else p.stem.replace("_grid", "")

        for thr, sub in df.groupby("threshold"):
            a = sub["ensemble_acc_confident"].to_numpy(dtype=float)
            b = sub["best_acc"].to_numpy(dtype=float)
            mask = ~(np.isnan(a) | np.isnan(b))
            a2, b2 = a[mask], b[mask]
            diffs = a2 - b2
            n = int(diffs.size)

            if n >= 2:
                try:
                    from scipy import stats  # type: ignore

                    t = stats.ttest_rel(a2, b2)
                    lev = stats.levene(a2, b2, center="median")
                    t_stat, t_p = float(t.statistic), float(t.pvalue)
                    lev_stat, lev_p = float(lev.statistic), float(lev.pvalue)
                except Exception:
                    t = paired_ttest_rel(a2, b2)
                    lev = levene_two_groups(a2, b2)
                    t_stat, t_p = float(t.statistic), float(t.pvalue)
                    lev_stat, lev_p = float(lev.statistic), float(lev.pvalue)
            else:
                t_stat = t_p = lev_stat = lev_p = float("nan")

            d = _paired_cohens_d(diffs)
            ci_low, ci_high = _t_ci_mean(diffs, alpha=0.05)

            rows.append(
                {
                    "grid_file": p.name,
                    "ensemble_name": ens_name,
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

    return pd.DataFrame(rows).sort_values(["ensemble_name", "threshold"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate stats_tests_confident_vs_best.csv for every m.i.n.d/outputs/**/ensemble_v2 folder.\n"
            "This fixes dashboard NA/NaN issues when grid files or columns change."
        )
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=(Path(__file__).resolve().parent / "outputs"),
        help="Root outputs folder to scan (default: m.i.n.d/outputs).",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually write files (otherwise just prints what it would do).",
    )
    args = parser.parse_args()

    outputs_root = args.outputs_root
    if not outputs_root.exists():
        raise SystemExit(f"Missing outputs root: {outputs_root}")

    ensemble_dirs = sorted(p for p in outputs_root.rglob("ensemble_v2") if p.is_dir())
    if not ensemble_dirs:
        raise SystemExit(f"No ensemble_v2 dirs found under: {outputs_root}")

    for d in ensemble_dirs:
        baseline = _find_baseline_for_ensemble_dir(d)
        if baseline is None:
            print(f"[skip] {d} (no baseline classification_results.csv found)")
            continue
        out_path = d / "stats_tests_confident_vs_best.csv"
        print(f"[stats] {d}")
        print(f"       baseline={baseline}")
        df = compute_stats_for_folder(ensemble_v2_dir=d, baseline_path=baseline)
        if args.write:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            print(f"       wrote={out_path} ({len(df)} rows)")
        else:
            print(f"       would_write={out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
