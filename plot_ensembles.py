from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

def load_baseline_summary(path: Path) -> Tuple[pd.DataFrame, Dict[str, float], float]:
    baseline_df = pd.read_csv(path)
    cols = set(baseline_df.columns)

    if {"model", "mean_acc"}.issubset(cols):
        baseline_df = baseline_df.copy()
        baseline_df["model"] = baseline_df["model"].astype(str)
        baseline_df["mean_acc"] = pd.to_numeric(baseline_df["mean_acc"], errors="raise")

        baseline_means = dict(zip(baseline_df["model"], baseline_df["mean_acc"]))
        best_single_rows = baseline_df[
            baseline_df["model"].str.lower().isin({"best-single", "best_single", "best single"})
        ]
        if best_single_rows.empty:
            raise ValueError(
                'Baseline summary must contain a "Best-Single" row in the `model` column (or use the wide-format summary).'
            )
        best_single_mean = float(best_single_rows.iloc[0]["mean_acc"])
        return baseline_df, baseline_means, best_single_mean

    required_wide = {"LDA_mean", "SVM_mean", "RF_mean"}
    if not required_wide.issubset(cols):
        raise ValueError(
            f"Baseline summary at {path} must be either long-form (model, mean_acc) or wide-form "
            f"({sorted(required_wide)}). Got columns: {list(baseline_df.columns)}"
        )

    row = baseline_df.iloc[0]
    lda_mean = float(row["LDA_mean"])
    svm_mean = float(row["SVM_mean"])
    rf_mean = float(row["RF_mean"])

    lda_sd = float(row["LDA_subject_std"]) if "LDA_subject_std" in cols else float("nan")
    svm_sd = float(row["SVM_subject_std"]) if "SVM_subject_std" in cols else float("nan")
    rf_sd = float(row["RF_subject_std"]) if "RF_subject_std" in cols else float("nan")

    results_path = BASE_DIR / "outputs" / "classification_results.csv"
    if results_path.exists():
        results_df = pd.read_csv(results_path)
        if "best_acc" not in results_df.columns:
            raise ValueError(f"Expected column `best_acc` in {results_path}; got {list(results_df.columns)}")
        best_single_mean = float(pd.to_numeric(results_df["best_acc"], errors="raise").mean())
        best_single_sd = float(pd.to_numeric(results_df["best_acc"], errors="raise").std())
    else:
        raise ValueError(
            f"Wide baseline summary {path} does not include Best-Single. "
            f"To plot it, provide a long-form baseline summary with a Best-Single row, "
            f"or add {results_path} so it can be computed."
        )

    baseline_df_long = pd.DataFrame(
        [
            {"model": "LDA", "mean_acc": lda_mean, "sd_acc": lda_sd},
            {"model": "SVM", "mean_acc": svm_mean, "sd_acc": svm_sd},
            {"model": "RF", "mean_acc": rf_mean, "sd_acc": rf_sd},
            {"model": "Best-Single", "mean_acc": best_single_mean, "sd_acc": best_single_sd},
        ]
    )
    baseline_means = dict(zip(baseline_df_long["model"], baseline_df_long["mean_acc"]))
    return baseline_df_long, baseline_means, best_single_mean


def load_ensemble_grids() -> pd.DataFrame:
    ensemble_v2_dir = BASE_DIR / "outputs" / "ensemble_v2"

    def _latest_dir(root: Path, glob_pat: str) -> Optional[Path]:
        cands = [p for p in root.glob(glob_pat) if p.is_dir()]
        if not cands:
            return None
        cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return cands[0]

    def _load_grid_with_fallback(
        grid_path: Path,
        fallback_run_glob: str,
        ensemble_name: str,
    ) -> pd.DataFrame:
        """
        Prefer the newest per-run `threshold_metrics.csv` under outputs/ensemble_v2/models-*.

        Why: grid files are "stable filenames" for the dashboard, but it is easy for them to become stale
        if you rerun experiments (new models-* folder) with a different threshold grid. Using the newest
        run's `threshold_metrics.csv` ensures the stable *_grid.csv files always reflect the latest run.

        If no run folder exists, fall back to reading the existing stable *_grid.csv.
        """
        run_dir = _latest_dir(ensemble_v2_dir, fallback_run_glob)
        if run_dir is not None:
            tm = run_dir / "threshold_metrics.csv"
            if tm.exists():
                df = pd.read_csv(tm)
                required = {"subject", "threshold", "ensemble_acc_all", "ensemble_acc_confident", "ensemble_coverage"}
                missing = required - set(df.columns)
                if missing:
                    raise ValueError(f"{tm} missing columns: {sorted(missing)}")
                df = df[list(required)].copy()
                df["ensemble_name"] = ensemble_name
                grid_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(grid_path, index=False)
                return df

        if not grid_path.exists():
            raise FileNotFoundError(
                f"Missing stable grid CSV: {grid_path}\n"
                f"Also no matching run folder found for pattern: {fallback_run_glob}\n"
                f"Run ensemble_v2, then rerun plot_ensembles to generate the dashboard grids."
            )

        df = pd.read_csv(grid_path)
        df["ensemble_name"] = ensemble_name
        return df

    grid_specs = [
        (
            ensemble_v2_dir / "LDA_SVM_equal_grid.csv",
            "models-LDA_SVM__weights-equal*__ens-softvote*",
            "Main: LDA+SVM (equal)",
        ),
        (
            ensemble_v2_dir / "LDA_SVM_RF_global_grid.csv",
            "models-LDA_SVM_RF__weights-baseline_global*__ens-softvote*",
            "Ablation: LDA+SVM+RF (global)",
        ),
        (
            ensemble_v2_dir / "LDA_SVM_baseline_subject_grid.csv",
            "models-LDA_SVM__weights-baseline_subject*__ens-softvote*",
            "Main: LDA+SVM (subj-weights)",
        ),
        (
            ensemble_v2_dir / "LDA_SVM_stacking_grid.csv",
            "models-LDA_SVM__*__ens-stacking*",
            "Stacking: LDA+SVM (meta-learner)",
        ),
    ]

    frames = []
    for grid_path, fallback_glob, ensemble_name in grid_specs:
        frames.append(_load_grid_with_fallback(grid_path, fallback_glob, ensemble_name))

    ens_df = pd.concat(frames, ignore_index=True)
    required = {"subject", "threshold", "ensemble_acc_all", "ensemble_acc_confident", "ensemble_coverage", "ensemble_name"}
    missing = required - set(ens_df.columns)
    if missing:
        raise ValueError(f"Ensemble grids missing required columns: {sorted(missing)}")

    ens_df["threshold"] = pd.to_numeric(ens_df["threshold"], errors="raise")
    ens_df["ensemble_acc_all"] = pd.to_numeric(ens_df["ensemble_acc_all"], errors="raise")
    ens_df["ensemble_acc_confident"] = pd.to_numeric(ens_df["ensemble_acc_confident"], errors="coerce")
    ens_df["ensemble_coverage"] = pd.to_numeric(ens_df["ensemble_coverage"], errors="raise")

    return ens_df


def load_debounced_grids() -> pd.DataFrame:
    """
    Load debounced stability-controller grids produced by `make_debounced_grids.py`.
    These are stored as top-level `*_debounced_grid.csv` files in outputs/ensemble_v2/.
    """
    ensemble_v2_dir = BASE_DIR / "outputs" / "ensemble_v2"
    paths = sorted(ensemble_v2_dir.glob("*_debounced_grid.csv"))
    if not paths:
        return pd.DataFrame()

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        required = {
            "subject",
            "threshold",
            "ensemble_acc_all",
            "ensemble_acc_confident",
            "ensemble_coverage",
            "toggle_rate",
            "wrong_fire_rate_all",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{p} missing columns: {sorted(missing)}")
        df = df[list(required)].copy()
        df["ensemble_name"] = p.stem.replace("_grid", "")
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["subject"] = pd.to_numeric(out["subject"], errors="raise").astype(int)
    out["threshold"] = pd.to_numeric(out["threshold"], errors="raise")
    for c in [
        "ensemble_acc_all",
        "ensemble_acc_confident",
        "ensemble_coverage",
        "toggle_rate",
        "wrong_fire_rate_all",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["ensemble_name"] = out["ensemble_name"].astype(str)
    return out

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
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")

def _heatmap(ax, data, x_labels, y_labels, title: str, cmap: str = "viridis", vmin=None, vmax=None):
    import numpy as np

    arr = data.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels([str(x) for x in x_labels], rotation=45, ha="right")
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels([str(y) for y in y_labels])
    ax.set_title(title)
    # annotate
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7, color="white" if val > (vmin or 0) + 0.5 * ((vmax or val) - (vmin or 0)) else "black")
    return im


def load_stats_tests(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "ensemble_name",
        "threshold",
        "paired_t_pvalue",
        "levene_pvalue",
        "mean_diff_conf_minus_best",
        "mean_coverage",
        "n_subjects_used",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Stats tests CSV at {path} missing columns: {sorted(missing)}")

    df = df.copy()
    df["threshold"] = pd.to_numeric(df["threshold"], errors="raise")
    for c in ["paired_t_pvalue", "levene_pvalue", "mean_diff_conf_minus_best", "mean_coverage", "n_subjects_used"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ensemble_name"] = df["ensemble_name"].astype(str)
    return df


def load_key_numbers() -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    key_dir = BASE_DIR / "outputs" / "key_numbers"
    per_subject_path = key_dir / "key_numbers_per_subject.csv"
    summary_path = key_dir / "key_numbers_summary.csv"
    if not per_subject_path.exists():
        raise FileNotFoundError(f"Missing key numbers per-subject CSV: {per_subject_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing key numbers summary CSV: {summary_path}")

    per_subject = pd.read_csv(per_subject_path)
    summary = pd.read_csv(summary_path)
    if not {"Metric", "Value"}.issubset(summary.columns):
        raise ValueError(f"{summary_path} must include columns Metric,Value; got {list(summary.columns)}")

    op = summary[summary["Metric"].astype(str).str.contains("Operating threshold", case=False, na=False)]
    if op.empty:
        raise ValueError(f"Could not find 'Operating threshold' row in {summary_path}")
    operating_threshold = float(pd.to_numeric(op.iloc[0]["Value"], errors="raise"))
    return per_subject, summary, operating_threshold


def _extract_key_number_cols(per_subject_df: pd.DataFrame, operating_threshold: float) -> Tuple[List[str], List[str]]:
    suffix = f"@ {operating_threshold}"
    conf_cols = [c for c in per_subject_df.columns if c.endswith(f"Conf Acc {suffix}")]
    cov_cols = [c for c in per_subject_df.columns if c.endswith(f"Coverage {suffix}")]
    return conf_cols, cov_cols


def load_baseline_per_subject() -> pd.DataFrame:
    path = BASE_DIR / "outputs" / "classification_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline per-subject results: {path}")
    df = pd.read_csv(path)
    if not {"subject", "best_acc"}.issubset(df.columns):
        raise ValueError(f"{path} missing columns subject,best_acc; got {list(df.columns)}")
    df = df.copy()
    df["subject"] = pd.to_numeric(df["subject"], errors="raise").astype(int)
    df["best_acc"] = pd.to_numeric(df["best_acc"], errors="raise")
    return df[["subject", "best_acc"]].sort_values("subject")


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as e:
        raise SystemExit(
            "matplotlib is required to generate plots. Install it (e.g. `pip install matplotlib`) and rerun."
        ) from e

    visuals_dir = BASE_DIR / "outputs" / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)

    baseline_path = BASE_DIR / "outputs" / "classification_summary.csv"
    baseline_df, baseline_means, best_single_mean = load_baseline_summary(baseline_path)

    ens_df = load_ensemble_grids()
    deb_df = load_debounced_grids()
    grouped = (
        ens_df.groupby(["ensemble_name", "threshold"], as_index=False)
        .agg(
            mean_acc_all=("ensemble_acc_all", "mean"),
            mean_acc_conf=("ensemble_acc_confident", "mean"),
            mean_cov=("ensemble_coverage", "mean"),
        )
        .sort_values(["ensemble_name", "threshold"])
    )

    configure_matplotlib_style()

    # (1) Ensemble Accuracy (Confident Trials) vs Threshold
    fig, ax = plt.subplots()
    for ensemble_name in grouped["ensemble_name"].unique():
        sub = grouped[grouped["ensemble_name"] == ensemble_name].sort_values("threshold")
        ax.plot(sub["threshold"], sub["mean_acc_conf"], marker="o", linewidth=2, label=ensemble_name)

    ax.axhline(
        best_single_mean,
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
        color="black",
        label=f"Best-Single baseline ({best_single_mean:.3f})",
    )

    ax.set_title("Ensemble Accuracy (Confident Trials) vs Threshold")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Mean accuracy (confident trials)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _save_fig(fig, visuals_dir / "ensemble_accuracy_confident_vs_threshold.png")
    plt.close(fig)

    # (2) Ensemble Coverage vs Threshold
    fig, ax = plt.subplots()
    for ensemble_name in grouped["ensemble_name"].unique():
        sub = grouped[grouped["ensemble_name"] == ensemble_name].sort_values("threshold")
        ax.plot(sub["threshold"], sub["mean_cov"], marker="o", linewidth=2, label=ensemble_name)

    ax.set_title("Ensemble Coverage vs Threshold")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Mean coverage (fraction confident)")
    ax.set_ylim(0, 1)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _save_fig(fig, visuals_dir / "ensemble_coverage_vs_threshold.png")
    plt.close(fig)

    # (3) Accuracy–Coverage Tradeoff (Pareto curve per ensemble)
    trade = (
        ens_df.groupby(["ensemble_name", "threshold"], as_index=False)
        .agg(mean_conf=("ensemble_acc_confident", "mean"), mean_cov=("ensemble_coverage", "mean"))
        .sort_values(["ensemble_name", "threshold"])
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    for ensemble_name in sorted(trade["ensemble_name"].unique()):
        sub = trade[trade["ensemble_name"] == ensemble_name]
        ax.plot(sub["mean_cov"], sub["mean_conf"], marker="o", linewidth=2, label=ensemble_name)
        for _, r in sub.iterrows():
            ax.text(r["mean_cov"], r["mean_conf"], f"{r['threshold']:.2f}", fontsize=7, ha="left", va="bottom")
    ax.axhline(best_single_mean, linestyle="--", linewidth=1.2, alpha=0.8, color="black", label="Best-Single mean")
    ax.set_title("Accuracy–Coverage Tradeoff (Mean across subjects)")
    ax.set_xlabel("Mean coverage")
    ax.set_ylabel("Mean confident accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    _save_fig(fig, visuals_dir / "tradeoff_conf_acc_vs_coverage.png")
    plt.close(fig)

    # (4/5) Stats-test visuals (paired t-test + mean delta), using precomputed CSV
    stats_path = BASE_DIR / "outputs" / "ensemble_v2" / "stats_tests_confident_vs_best.csv"
    if stats_path.exists():
        stats_df = load_stats_tests(stats_path)

        # Map stats-test ensemble names to the display names used in grid plots.
        name_map = {
            "LDA_SVM_RF_global_grid": "Ablation: LDA+SVM+RF (global)",
            "LDA_SVM_baseline_subject_grid": "Main: LDA+SVM (subj-weights)",
            "LDA_SVM_equal_grid": "Main: LDA+SVM (equal)",
            "Main: LDA+SVM (equal)": "Main: LDA+SVM (equal)",
        }
        stats_df = stats_df.copy()
        stats_df["ensemble_name"] = stats_df["ensemble_name"].map(lambda x: name_map.get(x, x))

        # (4) Paired t-test p-value vs threshold
        fig, ax = plt.subplots()
        for ensemble_name in stats_df["ensemble_name"].unique():
            sub = stats_df[stats_df["ensemble_name"] == ensemble_name].sort_values("threshold")
            ax.plot(sub["threshold"], sub["paired_t_pvalue"], marker="o", linewidth=2, label=ensemble_name)
        ax.axhline(0.05, linestyle="--", linewidth=1.2, alpha=0.8, color="black", label="p=0.05")
        ax.set_yscale("log")
        ax.set_title("Paired t-test p-value vs Threshold (Confident Acc vs Best-Single)")
        ax.set_xlabel("Confidence threshold")
        ax.set_ylabel("p-value (log scale)")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        _save_fig(fig, visuals_dir / "paired_ttest_pvalue_vs_threshold.png")
        plt.close(fig)

        # (5) Mean (Confident Acc − Best-Single) vs threshold
        fig, ax = plt.subplots()
        for ensemble_name in stats_df["ensemble_name"].unique():
            sub = stats_df[stats_df["ensemble_name"] == ensemble_name].sort_values("threshold")
            ax.plot(sub["threshold"], sub["mean_diff_conf_minus_best"], marker="o", linewidth=2, label=ensemble_name)
        ax.axhline(0.0, linestyle="--", linewidth=1.2, alpha=0.8, color="black", label="no improvement")
        ax.set_title("Mean (Confident Acc − Best-Single) vs Threshold")
        ax.set_xlabel("Confidence threshold")
        ax.set_ylabel("Mean accuracy difference")
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        _save_fig(fig, visuals_dir / "mean_diff_conf_vs_best_vs_threshold.png")
        plt.close(fig)

    # (6) Per-Subject Confident Accuracy at Threshold 0.60
    try:
        baseline_best = load_baseline_per_subject()
        ens_raw = ens_df.copy()
        ens_raw["subject"] = pd.to_numeric(ens_raw["subject"], errors="raise").astype(int)
        ens_raw = ens_raw.merge(baseline_best, on="subject", how="left")

        thr = 0.60
        snap = ens_raw[ens_raw["threshold"] == thr].copy()
        if not snap.empty:
            fig, ax = plt.subplots(figsize=(11, 5))
            x = [f"S{s}" for s in sorted(snap["subject"].unique().tolist())]
            best_by_subject = baseline_best.sort_values("subject")["best_acc"].to_numpy(dtype=float)
            ax.plot(x, best_by_subject, marker="o", linewidth=2, color="black", label="Best-Single")
            for ens_name in sorted(snap["ensemble_name"].unique()):
                vals = (
                    snap[snap["ensemble_name"] == ens_name]
                    .sort_values("subject")["ensemble_acc_confident"]
                    .to_numpy(dtype=float)
                )
                ax.plot(x, vals, marker="o", linewidth=2, label=ens_name)
            ax.set_title("Per-Subject Confident Accuracy at Threshold 0.60")
            ax.set_xlabel("Subject")
            ax.set_ylabel("Accuracy (confident trials)")
            ax.set_ylim(0, 1)
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
            _save_fig(fig, visuals_dir / "per_subject_conf_acc_t0.60.png")
            plt.close(fig)
    except Exception:
        pass

    # (7) Heatmap: Ablation (global) — (Conf Acc − Best-Single) by Subject/Threshold
    try:
        baseline_best = load_baseline_per_subject()
        ens_raw = ens_df.copy()
        ens_raw["subject"] = pd.to_numeric(ens_raw["subject"], errors="raise").astype(int)
        ens_raw = ens_raw.merge(baseline_best, on="subject", how="left")
        ens_raw["diff_conf_minus_best"] = ens_raw["ensemble_acc_confident"] - ens_raw["best_acc"]

        ablation_name = "Ablation: LDA+SVM+RF (global)"
        sub = ens_raw[ens_raw["ensemble_name"] == ablation_name].copy()
        if not sub.empty:
            thresholds = sorted(sub["threshold"].unique().tolist())
            subjects = sorted(sub["subject"].unique().tolist())
            pivot_diff = (
                sub.pivot_table(index="subject", columns="threshold", values="diff_conf_minus_best", aggfunc="mean")
                .reindex(index=subjects, columns=thresholds)
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            vmax = float(
                max(
                    abs(pivot_diff.min().min(skipna=True)),
                    abs(pivot_diff.max().max(skipna=True)),
                )
            ) if pivot_diff.size else 0.2
            im = _heatmap(
                ax,
                pivot_diff,
                thresholds,
                [f"S{s}" for s in subjects],
                "Heatmap: Ablation (global) — (Conf Acc − Best-Single) by Subject/Threshold",
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            _save_fig(fig, visuals_dir / "heatmap_ablation_global_diff_conf_minus_best.png")
            plt.close(fig)
    except Exception:
        pass

    # (8) Stability controller plots (debounced)
    try:
        if not deb_df.empty:
            deb = deb_df.copy()
            deb["safe_fire"] = deb["ensemble_coverage"] - deb["wrong_fire_rate_all"]
            agg2 = (
                deb.groupby(["ensemble_name", "threshold"], as_index=False)
                .agg(
                    mean_toggle=("toggle_rate", "mean"),
                    mean_wrong_fire=("wrong_fire_rate_all", "mean"),
                    mean_safe_fire=("safe_fire", "mean"),
                )
                .sort_values(["ensemble_name", "threshold"])
            )

            fig, ax = plt.subplots()
            for name in agg2["ensemble_name"].unique():
                sub = agg2[agg2["ensemble_name"] == name].sort_values("threshold")
                ax.plot(sub["threshold"], sub["mean_toggle"], marker="o", linewidth=2, label=name)
            ax.set_title("Stability (Debounced): Toggle Rate vs Threshold")
            ax.set_xlabel("Threshold (t_on)")
            ax.set_ylabel("Mean toggle rate")
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
            _save_fig(fig, visuals_dir / "stability_toggle_rate_vs_threshold.png")
            plt.close(fig)

            fig, ax = plt.subplots()
            for name in agg2["ensemble_name"].unique():
                sub = agg2[agg2["ensemble_name"] == name].sort_values("threshold")
                ax.plot(sub["threshold"], sub["mean_wrong_fire"], marker="o", linewidth=2, label=name)
            ax.set_title("Safety (Debounced): Wrong-Fire vs Threshold")
            ax.set_xlabel("Threshold (t_on)")
            ax.set_ylabel("Mean wrong-fire rate (all trials)")
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
            _save_fig(fig, visuals_dir / "stability_wrong_fire_vs_threshold.png")
            plt.close(fig)

            fig, ax = plt.subplots()
            for name in agg2["ensemble_name"].unique():
                sub = agg2[agg2["ensemble_name"] == name].sort_values("threshold")
                ax.plot(sub["threshold"], sub["mean_safe_fire"], marker="o", linewidth=2, label=name)
            ax.set_title("Usefulness (Debounced): Safe-Fire vs Threshold")
            ax.set_xlabel("Threshold (t_on)")
            ax.set_ylabel("Mean safe-fire (coverage − wrong-fire)")
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
            _save_fig(fig, visuals_dir / "stability_safe_fire_vs_threshold.png")
            plt.close(fig)
    except Exception:
        pass


if __name__ == "__main__":
    main()
