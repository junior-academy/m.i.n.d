from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent


def load_baseline_summary(path: Path) -> Tuple[pd.DataFrame, Dict[str, float], float]:
    baseline_df = pd.read_csv(path)
    cols = set(baseline_df.columns)

    # Preferred "long" format (as described in the prompt)
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

    # Back-compat "wide" format produced by this repo (single row with LDA_mean/SVM_mean/RF_mean)
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

    # SD columns are optional in wide form; include if present.
    lda_sd = float(row["LDA_subject_std"]) if "LDA_subject_std" in cols else float("nan")
    svm_sd = float(row["SVM_subject_std"]) if "SVM_subject_std" in cols else float("nan")
    rf_sd = float(row["RF_subject_std"]) if "RF_subject_std" in cols else float("nan")

    # Best-Single isn't present in the wide summary; compute from per-subject results if available.
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

    def _load_grid_with_fallback(
        grid_path: Path,
        fallback_threshold_metrics_path: Path,
        ensemble_name: str,
    ) -> pd.DataFrame:
        """
        Prefer the explicit *_grid.csv files described in the prompt.
        Fall back to ensemble_v2 per-run `threshold_metrics.csv` if the grid file doesn't exist.
        If fallback is used, write the grid CSV to the expected location for next time.
        """
        if grid_path.exists():
            df = pd.read_csv(grid_path)
            df["ensemble_name"] = ensemble_name
            return df

        if not fallback_threshold_metrics_path.exists():
            raise FileNotFoundError(
                f"Missing ensemble grid CSV: {grid_path}\n"
                f"Also missing fallback threshold metrics CSV: {fallback_threshold_metrics_path}\n"
                f"Generate grids by running ensemble_v2 (or export the *_grid.csv files), then rerun."
            )

        df = pd.read_csv(fallback_threshold_metrics_path)
        # expected columns in threshold_metrics.csv
        required = {"subject", "threshold", "ensemble_acc_all", "ensemble_acc_confident", "ensemble_coverage"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Fallback threshold metrics at {fallback_threshold_metrics_path} missing columns {sorted(missing)}"
            )

        df = df[list(required)].copy()
        df["ensemble_name"] = ensemble_name
        # Persist to the requested grid filename so future runs follow the documented structure.
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(grid_path, index=False)
        return df

    grid_specs = [
        (
            ensemble_v2_dir / "LDA_SVM_equal_grid.csv",
            ensemble_v2_dir / "models-LDA_SVM__weights-equal" / "threshold_metrics.csv",
            "Main: LDA+SVM (equal)",
        ),
        (
            ensemble_v2_dir / "LDA_SVM_RF_global_grid.csv",
            ensemble_v2_dir / "models-LDA_SVM_RF__weights-baseline_global" / "threshold_metrics.csv",
            "Ablation: LDA+SVM+RF (global)",
        ),
        (
            ensemble_v2_dir / "LDA_SVM_baseline_subject_grid.csv",
            ensemble_v2_dir / "models-LDA_SVM__weights-baseline_subject" / "threshold_metrics.csv",
            "Main: LDA+SVM (subj-weights)",
        ),
    ]

    frames = []
    for grid_path, fallback_path, ensemble_name in grid_specs:
        frames.append(_load_grid_with_fallback(grid_path, fallback_path, ensemble_name))

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

    # (a) Mean accuracy on all trials vs threshold
    fig, ax = plt.subplots()
    for ensemble_name in grouped["ensemble_name"].unique():
        sub = grouped[grouped["ensemble_name"] == ensemble_name].sort_values("threshold")
        ax.plot(sub["threshold"], sub["mean_acc_all"], marker="o", linewidth=2, label=ensemble_name)

    # Baseline horizontal lines
    def _baseline_label(model_name: str, mean_acc: float) -> str:
        return f"{model_name} baseline ({mean_acc:.3f})"

    for model in ["LDA", "SVM", "RF"]:
        if model in baseline_means:
            ax.axhline(
                baseline_means[model],
                linestyle="--",
                linewidth=1.2,
                alpha=0.8,
                label=_baseline_label(model, float(baseline_means[model])),
            )

    ax.axhline(
        best_single_mean,
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
        color="black",
        label=_baseline_label("Best-Single", best_single_mean),
    )

    ax.set_title("Ensemble Accuracy (All Trials) vs Threshold")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Mean accuracy (all trials)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(visuals_dir / "ensemble_accuracy_all_vs_threshold.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # (b) Mean accuracy on confident trials vs threshold
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
        label=_baseline_label("Best-Single", best_single_mean),
    )

    ax.set_title("Ensemble Accuracy (Confident Trials) vs Threshold")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Mean accuracy (confident trials)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(visuals_dir / "ensemble_accuracy_confident_vs_threshold.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # (c) Mean coverage vs threshold
    fig, ax = plt.subplots()
    for ensemble_name in grouped["ensemble_name"].unique():
        sub = grouped[grouped["ensemble_name"] == ensemble_name].sort_values("threshold")
        ax.plot(sub["threshold"], sub["mean_cov"], marker="o", linewidth=2, label=ensemble_name)

    ax.set_title("Ensemble Coverage vs Threshold")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("Mean coverage (fraction confident)")
    ax.set_ylim(0, 1)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(visuals_dir / "ensemble_coverage_vs_threshold.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
