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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Report per-threshold metrics (coverage/conf-acc/all-acc) + paired diff vs Best-Single with effect size + CI.\n"
            "If the grid includes debounced stability columns, also report wrong-fire/toggle/safe-fire and select a policy\n"
            "under a safety constraint (wrong-fire <= alpha)."
        )
    )
    parser.add_argument("--grid", type=Path, required=True, help="Path to *_grid.csv or *_debounced_grid.csv.")
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to classification_results.csv (must include subject,best_acc).",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="Safety constraint for debounced policy selection.")
    parser.add_argument("--out", type=Path, default=None, help="Optional output CSV path.")
    args = parser.parse_args()

    if not args.grid.exists():
        raise SystemExit(f"Missing grid: {args.grid}")
    if not args.baseline.exists():
        raise SystemExit(f"Missing baseline: {args.baseline}")

    base = pd.read_csv(args.baseline)
    if not {"subject", "best_acc"}.issubset(base.columns):
        raise SystemExit(f"{args.baseline} missing columns subject,best_acc")
    base = base[["subject", "best_acc"]].copy()
    base["subject"] = pd.to_numeric(base["subject"], errors="raise").astype(int)
    base["best_acc"] = pd.to_numeric(base["best_acc"], errors="raise")

    df = pd.read_csv(args.grid)
    required = {"subject", "threshold", "ensemble_acc_all", "ensemble_acc_confident", "ensemble_coverage"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{args.grid} missing columns: {sorted(missing)}")

    df = df.copy()
    df["subject"] = pd.to_numeric(df["subject"], errors="raise").astype(int)
    df["threshold"] = pd.to_numeric(df["threshold"], errors="raise")
    df["ensemble_acc_all"] = pd.to_numeric(df["ensemble_acc_all"], errors="raise")
    df["ensemble_acc_confident"] = pd.to_numeric(df["ensemble_acc_confident"], errors="coerce")
    df["ensemble_coverage"] = pd.to_numeric(df["ensemble_coverage"], errors="raise")

    has_debounced = "wrong_fire_rate_all" in df.columns and "toggle_rate" in df.columns
    if has_debounced:
        df["wrong_fire_rate_all"] = pd.to_numeric(df["wrong_fire_rate_all"], errors="coerce")
        df["toggle_rate"] = pd.to_numeric(df["toggle_rate"], errors="coerce")
        df["safe_fire"] = df["ensemble_coverage"] - df["wrong_fire_rate_all"]
        for c in ["t_on", "t_off", "k", "n"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.merge(base, on="subject", how="left")
    df["diff_conf_minus_best"] = df["ensemble_acc_confident"] - df["best_acc"]

    rows: list[dict] = []
    group_cols = ["threshold"]
    if has_debounced:
        for c in ["t_off", "k", "n"]:
            if c in df.columns:
                group_cols.append(c)

    for key, sub in df.groupby(group_cols):
        thr = float(key[0]) if isinstance(key, tuple) else float(key)
        a = sub["ensemble_acc_confident"].to_numpy(dtype=float)
        b = sub["best_acc"].to_numpy(dtype=float)
        mask = ~(np.isnan(a) | np.isnan(b))
        diffs = a[mask] - b[mask]
        n_used = int(diffs.size)

        ci_lo, ci_hi = _t_ci_mean(diffs, alpha=0.05)
        d = _paired_cohens_d(diffs)

        row = {
            "threshold": float(thr),
            "n_subjects_used": n_used,
            "best_single_mean": float(np.nanmean(sub["best_acc"].to_numpy(dtype=float))),
            "mean_conf_acc": float(np.nanmean(a)),
            "mean_coverage": float(np.nanmean(sub["ensemble_coverage"].to_numpy(dtype=float))),
            "mean_all_acc": float(np.nanmean(sub["ensemble_acc_all"].to_numpy(dtype=float))),
            "mean_diff_conf_minus_best": float(np.nanmean(sub["diff_conf_minus_best"].to_numpy(dtype=float))),
            "diff_ci_low_95": float(ci_lo),
            "diff_ci_high_95": float(ci_hi),
            "paired_cohens_d": float(d),
        }

        if has_debounced:
            row.update(
                {
                    "mean_wrong_fire_rate_all": float(np.nanmean(sub["wrong_fire_rate_all"].to_numpy(dtype=float))),
                    "mean_toggle_rate": float(np.nanmean(sub["toggle_rate"].to_numpy(dtype=float))),
                    "mean_safe_fire": float(np.nanmean(sub["safe_fire"].to_numpy(dtype=float))),
                    "t_on": float(thr),
                    "t_off": float(np.nanmean(sub["t_off"].to_numpy(dtype=float))) if "t_off" in sub.columns else float("nan"),
                    "k": float(np.nanmean(sub["k"].to_numpy(dtype=float))) if "k" in sub.columns else float("nan"),
                    "n": float(np.nanmean(sub["n"].to_numpy(dtype=float))) if "n" in sub.columns else float("nan"),
                }
            )

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)

    if has_debounced:
        feasible = out[np.isfinite(out["mean_wrong_fire_rate_all"]) & (out["mean_wrong_fire_rate_all"] <= float(args.alpha))]
        pick_pool = feasible if not feasible.empty else out
        pick = pick_pool.sort_values(["mean_safe_fire", "mean_wrong_fire_rate_all", "mean_toggle_rate"], ascending=[False, True, True]).iloc[0]
        print(f"[policy] alpha={args.alpha:.3f} feasible={not feasible.empty} selected_t_on={float(pick['threshold']):.2f}")
        print(
            f"         safe_fire={float(pick['mean_safe_fire']):.3f} wrong_fire={float(pick['mean_wrong_fire_rate_all']):.3f} "
            f"coverage={float(pick['mean_coverage']):.3f} toggle={float(pick['mean_toggle_rate']):.3f}"
        )

    # Print a compact table to stdout
    show_cols = [
        "threshold",
    ]
    if has_debounced:
        for c in ["t_off", "k", "n"]:
            if c in out.columns:
                show_cols.append(c)
    show_cols += [
        "mean_conf_acc",
        "mean_coverage",
        "mean_all_acc",
        "mean_diff_conf_minus_best",
        "paired_cohens_d",
        "diff_ci_low_95",
        "diff_ci_high_95",
    ]
    if has_debounced:
        show_cols += ["mean_safe_fire", "mean_wrong_fire_rate_all", "mean_toggle_rate"]
    print(out[show_cols].to_string(index=False))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)
        print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
