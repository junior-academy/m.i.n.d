from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def configure_matplotlib_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def _save_fig(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")


def _mean(xs: np.ndarray) -> float:
    xs = np.asarray(xs, dtype=float)
    xs = xs[np.isfinite(xs)]
    return float(xs.mean()) if xs.size else float("nan")


def _heatmap(ax, data: pd.DataFrame, title: str, cmap: str = "viridis", vmin=None, vmax=None):
    arr = data.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels([str(x) for x in data.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels([str(y) for y in data.index])
    ax.set_title(title)
    return im


def load_baseline_results(path: Path) -> Tuple[pd.DataFrame, float]:
    df = pd.read_csv(path)
    if not {"subject", "best_acc"}.issubset(df.columns):
        raise SystemExit(f"{path} missing required columns: subject,best_acc")
    df = df[["subject", "best_acc"]].copy()
    df["subject"] = pd.to_numeric(df["subject"], errors="raise").astype(int)
    df["best_acc"] = pd.to_numeric(df["best_acc"], errors="raise")
    return df, float(df["best_acc"].mean())


def load_grids(grid_dir: Path) -> pd.DataFrame:
    paths = sorted(grid_dir.glob("*_grid.csv"))
    if not paths:
        raise SystemExit(f"No *_grid.csv found in {grid_dir}")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        required = {"subject", "threshold", "ensemble_acc_all", "ensemble_acc_confident", "ensemble_coverage"}
        missing = required - set(df.columns)
        if missing:
            raise SystemExit(f"{p} missing columns: {sorted(missing)}")
        # Keep optional stability-controller columns if present (for *_debounced_grid.csv).
        optional = []
        for c in ["toggle_rate", "wrong_fire_rate_all"]:
            if c in df.columns:
                optional.append(c)
        keep = list(required) + optional
        df = df[keep].copy()
        df["ensemble_name"] = p.stem.replace("_grid", "")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["subject"] = pd.to_numeric(out["subject"], errors="raise").astype(int)
    out["threshold"] = pd.to_numeric(out["threshold"], errors="raise")
    out["ensemble_acc_all"] = pd.to_numeric(out["ensemble_acc_all"], errors="raise")
    out["ensemble_acc_confident"] = pd.to_numeric(out["ensemble_acc_confident"], errors="coerce")
    out["ensemble_coverage"] = pd.to_numeric(out["ensemble_coverage"], errors="raise")
    if "toggle_rate" in out.columns:
        out["toggle_rate"] = pd.to_numeric(out["toggle_rate"], errors="coerce")
    if "wrong_fire_rate_all" in out.columns:
        out["wrong_fire_rate_all"] = pd.to_numeric(out["wrong_fire_rate_all"], errors="coerce")
    out["ensemble_name"] = out["ensemble_name"].astype(str)
    return out


def maybe_write_stability_plots(ens_df: pd.DataFrame, out_dir: Path) -> None:
    """
    If stability-controller columns exist (toggle_rate, wrong_fire_rate_all) we create additional plots.

    Expected source: *_debounced_grid.csv files produced by make_debounced_grids.py, which include:
    - toggle_rate
    - wrong_fire_rate_all
    """
    required = {"toggle_rate", "wrong_fire_rate_all"}
    if not required.issubset(set(ens_df.columns)):
        return

    configure_matplotlib_style()
    import matplotlib.pyplot as plt

    df = ens_df.copy()
    df["toggle_rate"] = pd.to_numeric(df["toggle_rate"], errors="coerce")
    df["wrong_fire_rate_all"] = pd.to_numeric(df["wrong_fire_rate_all"], errors="coerce")
    df["safe_fire"] = df["ensemble_coverage"] - df["wrong_fire_rate_all"]

    agg = (
        df.groupby(["ensemble_name", "threshold"])
        .agg(
            mean_toggle=("toggle_rate", "mean"),
            mean_wrong_fire=("wrong_fire_rate_all", "mean"),
            mean_safe_fire=("safe_fire", "mean"),
            mean_cov=("ensemble_coverage", "mean"),
        )
        .reset_index()
    )

    # 8) Toggle rate vs threshold
    fig, ax = plt.subplots()
    for name, sub in agg.groupby("ensemble_name"):
        ax.plot(sub["threshold"], sub["mean_toggle"], marker="o", label=name)
    ax.set_xlabel("Threshold (t_on)")
    ax.set_ylabel("Mean toggle rate")
    ax.set_title("Stability (Debounced): Toggle Rate vs Threshold")
    ax.legend()
    _save_fig(fig, out_dir / "stability_toggle_rate_vs_threshold.png")
    plt.close(fig)

    # 9) Wrong-fire vs threshold
    fig, ax = plt.subplots()
    for name, sub in agg.groupby("ensemble_name"):
        ax.plot(sub["threshold"], sub["mean_wrong_fire"], marker="o", label=name)
    ax.set_xlabel("Threshold (t_on)")
    ax.set_ylabel("Mean wrong-fire rate (all trials)")
    ax.set_title("Safety (Debounced): Wrong-Fire vs Threshold")
    ax.legend()
    _save_fig(fig, out_dir / "stability_wrong_fire_vs_threshold.png")
    plt.close(fig)

    # 10) Safe-fire vs threshold
    fig, ax = plt.subplots()
    for name, sub in agg.groupby("ensemble_name"):
        ax.plot(sub["threshold"], sub["mean_safe_fire"], marker="o", label=name)
    ax.set_xlabel("Threshold (t_on)")
    ax.set_ylabel("Mean safe-fire (coverage − wrong-fire)")
    ax.set_title("Usefulness (Debounced): Safe-Fire vs Threshold")
    ax.legend()
    _save_fig(fig, out_dir / "stability_safe_fire_vs_threshold.png")
    plt.close(fig)


def load_stats_tests(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"grid_file", "threshold", "paired_t_pvalue", "mean_diff_conf_minus_best", "mean_coverage"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{path} missing columns: {sorted(missing)}")
    df = df.copy()
    df["threshold"] = pd.to_numeric(df["threshold"], errors="raise")
    for c in ["paired_t_pvalue", "mean_diff_conf_minus_best", "mean_coverage"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the 7 keeper plots for an arbitrary ensemble grid directory + baseline results."
    )
    parser.add_argument("--baseline-results", type=Path, required=True, help="classification_results.csv with best_acc.")
    parser.add_argument("--grid-dir", type=Path, required=True, help="Directory containing *_grid.csv files.")
    parser.add_argument("--stats-tests", type=Path, required=True, help="stats_tests_confident_vs_best.csv")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for PNGs.")
    parser.add_argument("--threshold", type=float, default=0.60, help="Operating threshold for per-subject plot.")
    parser.add_argument("--heatmap-grid", type=str, default="LDA_SVM_RF_equal_grid.csv",
                        help="Which grid_file to use for the heatmap plot (default: LDA_SVM_RF_equal_grid.csv).")
    args = parser.parse_args()

    configure_matplotlib_style()
    import matplotlib.pyplot as plt

    base_df, best_single_mean = load_baseline_results(args.baseline_results)
    ens_df = load_grids(args.grid_dir)
    stats_df = load_stats_tests(args.stats_tests)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Aggregates
    agg = (
        ens_df.groupby(["ensemble_name", "threshold"])
        .agg(
            mean_cov=("ensemble_coverage", "mean"),
            mean_conf=("ensemble_acc_confident", "mean"),
            mean_all=("ensemble_acc_all", "mean"),
        )
        .reset_index()
    )

    # 1) Tradeoff: conf acc vs coverage
    fig, ax = plt.subplots()
    for name, sub in agg.groupby("ensemble_name"):
        ax.plot(sub["mean_cov"], sub["mean_conf"], marker="o", label=name)
    ax.axhline(best_single_mean, linestyle="--", color="orange", label="Best-Single mean")
    ax.set_xlabel("Mean coverage")
    ax.set_ylabel("Mean confident accuracy")
    ax.set_title("Accuracy–Coverage Tradeoff")
    ax.legend()
    _save_fig(fig, out_dir / "tradeoff_conf_acc_vs_coverage.png")
    plt.close(fig)

    # 2) Confident acc vs threshold
    fig, ax = plt.subplots()
    for name, sub in agg.groupby("ensemble_name"):
        ax.plot(sub["threshold"], sub["mean_conf"], marker="o", label=name)
    ax.axhline(best_single_mean, linestyle="--", color="orange", label="Best-Single mean")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Mean confident accuracy")
    ax.set_title("Ensemble Accuracy (Confident Trials) vs Threshold")
    ax.legend()
    _save_fig(fig, out_dir / "ensemble_accuracy_confident_vs_threshold.png")
    plt.close(fig)

    # 3) Paired t-test p-value vs threshold (use stats_tests file)
    fig, ax = plt.subplots()
    for grid_file, sub in stats_df.groupby("grid_file"):
        ax.plot(sub["threshold"], sub["paired_t_pvalue"], marker="o", label=grid_file.replace("_grid.csv", ""))
    ax.axhline(0.05, linestyle="--", color="magenta", label="p=0.05")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Paired t-test p-value")
    ax.set_title("Paired t-test p-value vs Threshold")
    ax.set_ylim(0, 1.0)
    ax.legend()
    _save_fig(fig, out_dir / "paired_ttest_pvalue_vs_threshold.png")
    plt.close(fig)

    # 4) Mean (Conf Acc − Best) vs threshold
    fig, ax = plt.subplots()
    for grid_file, sub in stats_df.groupby("grid_file"):
        ax.plot(sub["threshold"], sub["mean_diff_conf_minus_best"], marker="o", label=grid_file.replace("_grid.csv", ""))
    ax.axhline(0.0, linestyle="--", color="gray")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Mean(Conf Acc − Best)")
    ax.set_title("Mean(Conf Acc − Best) vs Threshold")
    ax.legend()
    _save_fig(fig, out_dir / "mean_diff_conf_vs_best_vs_threshold.png")
    plt.close(fig)

    # 5) Per-subject confident accuracy at threshold
    t0 = float(args.threshold)
    # Snap to nearest threshold present
    thrs = sorted(ens_df["threshold"].unique().tolist())
    chosen = min(thrs, key=lambda t: abs(float(t) - t0)) if thrs else t0

    # Choose a "main" grid: prefer LDA_SVM_subject if present, else first
    main_name = "LDA_SVM_subject"
    if main_name not in set(ens_df["ensemble_name"].unique()):
        main_name = sorted(ens_df["ensemble_name"].unique().tolist())[0]
    sub = ens_df[(ens_df["ensemble_name"] == main_name) & (ens_df["threshold"] == chosen)].merge(base_df, on="subject", how="left")
    fig, ax = plt.subplots(figsize=(9, 4))
    xs = sub.sort_values("subject")
    ax.plot(xs["subject"], xs["ensemble_acc_confident"], marker="o", label=f"{main_name} conf acc")
    ax.plot(xs["subject"], xs["best_acc"], marker="o", label="Best-Single")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Per-Subject Confident Accuracy @ t={chosen:.2f} ({main_name})")
    ax.legend()
    _save_fig(fig, out_dir / f"per_subject_conf_acc_t{chosen:.2f}.png")
    plt.close(fig)

    # 6) Heatmap: ΔConf vs Best by Subject/Threshold (for requested grid_file)
    heat = ens_df.merge(base_df, on="subject", how="left")
    heat["diff"] = heat["ensemble_acc_confident"] - heat["best_acc"]
    heat_name = args.heatmap_grid.replace("_grid.csv", "").replace("_grid", "")
    heat_sub = heat[heat["ensemble_name"] == heat_name]
    if not heat_sub.empty:
        piv = heat_sub.pivot_table(index="subject", columns="threshold", values="diff", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(10, 4))
        im = _heatmap(ax, piv, title=f"ΔConf vs Best ({heat_name})", cmap="coolwarm")
        fig.colorbar(im, ax=ax, shrink=0.85)
        _save_fig(fig, out_dir / "heatmap_ablation_global_diff_conf_minus_best.png")
        plt.close(fig)

    # 7) Coverage vs threshold
    fig, ax = plt.subplots()
    for name, sub in agg.groupby("ensemble_name"):
        ax.plot(sub["threshold"], sub["mean_cov"], marker="o", label=name)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Mean coverage")
    ax.set_title("Ensemble Coverage vs Threshold")
    ax.legend()
    _save_fig(fig, out_dir / "ensemble_coverage_vs_threshold.png")
    plt.close(fig)

    # Optional: stability-controller plots (debounced gate)
    maybe_write_stability_plots(ens_df=ens_df, out_dir=out_dir)

    print(f"Wrote visuals to {out_dir}")


if __name__ == "__main__":
    main()
