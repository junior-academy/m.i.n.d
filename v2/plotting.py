"""Publication-quality plots for v2 saved output tables."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ORDER = [
    "v1_lda",
    "eegnet_uncalibrated",
    "eegnet_calibrated",
    "spatial_uncalibrated",
    "spatial_calibrated",
    "deep_ensemble_validation_weighted",
]

MODEL_LABELS = {
    "v1_lda": "v1 LDA",
    "eegnet_uncalibrated": "EEGNet",
    "eegnet_calibrated": "EEGNet + temp.",
    "spatial_uncalibrated": "Spatial-attn.",
    "spatial_calibrated": "Spatial-attn. + temp.",
    "deep_ensemble_validation_weighted": "Deep ensemble",
}

MODEL_COLORS = {
    "v1_lda": "#6E6E6E",
    "eegnet_uncalibrated": "#8EC9E8",
    "eegnet_calibrated": "#0072B2",
    "spatial_uncalibrated": "#7BC87C",
    "spatial_calibrated": "#009E73",
    "deep_ensemble_validation_weighted": "#D55E00",
}


def _weighted_reliability_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate subject-bin rows into trial-weighted reliability bins."""

    rows = []
    for bin_id, bin_df in df.groupby("bin"):
        valid = bin_df.dropna(subset=["confidence", "accuracy"])
        count = float(valid["count"].sum())
        if count <= 0:
            continue
        confidence = float((valid["confidence"] * valid["count"]).sum() / count)
        accuracy = float((valid["accuracy"] * valid["count"]).sum() / count)
        rows.append(
            {
                "bin": bin_id,
                "confidence": confidence,
                "accuracy": accuracy,
                "count": int(count),
                "gap": abs(accuracy - confidence),
            }
        )
    return pd.DataFrame(rows)


def _setup_matplotlib(out_dir: Path):
    mpl_dir = out_dir / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.fontsize": 6.5,
            "figure.dpi": 180,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return plt


def _save(fig, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))


def _model_sort_key(model: str) -> int:
    try:
        return MODEL_ORDER.index(model)
    except ValueError:
        return len(MODEL_ORDER)


def _ordered_models(df: pd.DataFrame, column: str = "model") -> list[str]:
    return sorted(df[column].dropna().unique().tolist(), key=_model_sort_key)


def _display_name(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("_", " "))


def _pretty_dataset(dataset: str) -> str:
    names = {
        "BCI_IV_2a": "BCI Competition IV 2a",
        "Cho2017": "Cho et al. 2017",
        "PhysionetMI": "PhysioNet MI",
        "BNCI2014_001": "BNCI 2014-001",
    }
    return names.get(dataset, dataset.replace("_", " "))


def _select_largest_runs(df: pd.DataFrame, subject_col: str = "subject") -> pd.DataFrame:
    """Keep the largest run per dataset when combined smoke/full outputs coexist."""

    if df.empty or "run" not in df.columns or "dataset" not in df.columns:
        return df.copy()
    kept = []
    for dataset, ds_df in df.groupby("dataset"):
        if subject_col in ds_df.columns:
            run_sizes = ds_df.groupby("run")[subject_col].nunique()
        elif "n_subjects" in ds_df.columns:
            run_sizes = ds_df.groupby("run")["n_subjects"].max()
        else:
            run_sizes = ds_df.groupby("run").size()
        best_run = run_sizes.sort_values(ascending=False).index[0]
        kept.append(ds_df[ds_df["run"] == best_run])
    return pd.concat(kept, ignore_index=True)


def _mean_sem(series: pd.Series) -> tuple[float, float]:
    values = series.dropna().astype(float)
    if values.empty:
        return float("nan"), float("nan")
    sem = values.std(ddof=1) / np.sqrt(values.size) if values.size > 1 else 0.0
    return float(values.mean()), float(sem)


def write_summary_tables(out_dir: Path) -> None:
    subject_path = out_dir / "combined_subject_metrics.csv"
    coverage_path = out_dir / "combined_coverage_sweep.csv"
    if not subject_path.exists():
        subject_path = out_dir / "subject_metrics.csv"
    if subject_path.exists():
        subject_df = _select_largest_runs(pd.read_csv(subject_path))
        rows = []
        for cols, group in subject_df.groupby(["dataset", "model"], dropna=False):
            dataset, model = cols
            acc, acc_sem = _mean_sem(group["accuracy_all"])
            sel, sel_sem = _mean_sem(group["selective_accuracy"])
            ece, ece_sem = _mean_sem(group["ece"])
            brier, brier_sem = _mean_sem(group["brier"])
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "label": _display_name(model),
                    "n_subjects": group["subject"].nunique(),
                    "accuracy_all_mean": acc,
                    "accuracy_all_sem": acc_sem,
                    "selective_accuracy_mean": sel,
                    "selective_accuracy_sem": sel_sem,
                    "ece_mean": ece,
                    "ece_sem": ece_sem,
                    "brier_mean": brier,
                    "brier_sem": brier_sem,
                }
            )
        pd.DataFrame(rows).sort_values(["dataset", "model"], key=lambda s: s.map(_model_sort_key) if s.name == "model" else s).to_csv(
            out_dir / "headline_summary_table.csv", index=False
        )

    if coverage_path.exists():
        coverage_df = _select_largest_runs(pd.read_csv(coverage_path))
        rows = []
        for cols, group in coverage_df.groupby(["dataset", "model", "target_coverage"], dropna=False):
            dataset, model, coverage = cols
            acc, acc_sem = _mean_sem(group["selective_accuracy"])
            risk, risk_sem = _mean_sem(group["selective_risk"])
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "label": _display_name(model),
                    "target_coverage": coverage,
                    "selective_accuracy_mean": acc,
                    "selective_accuracy_sem": acc_sem,
                    "selective_risk_mean": risk,
                    "selective_risk_sem": risk_sem,
                }
            )
        pd.DataFrame(rows).sort_values(["dataset", "target_coverage", "model"], key=lambda s: s.map(_model_sort_key) if s.name == "model" else s).to_csv(
            out_dir / "coverage_sweep_summary_table.csv", index=False
        )


def plot_headline_performance(subject_df: pd.DataFrame, out_dir: Path) -> None:
    plt = _setup_matplotlib(out_dir)
    plot_dir = out_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    subject_df = _select_largest_runs(subject_df)
    metrics = [
        ("accuracy_all", "All-trial accuracy", False),
        ("selective_accuracy", "Selective accuracy\n60% coverage", False),
        ("ece", "ECE", True),
        ("brier", "Brier score", True),
    ]
    for dataset, ds_df in subject_df.groupby("dataset"):
        models = _ordered_models(ds_df)
        fig, axes = plt.subplots(1, 4, figsize=(7.2, 1.95), sharex=False)
        for ax, (metric, title, lower_better) in zip(axes, metrics):
            means, sems, colors = [], [], []
            for model in models:
                mean, sem = _mean_sem(ds_df.loc[ds_df["model"] == model, metric])
                means.append(mean)
                sems.append(sem)
                colors.append(MODEL_COLORS.get(model, "0.35"))
            x = np.arange(len(models))
            ax.bar(x, means, yerr=sems, color=colors, edgecolor="0.2", linewidth=0.35, capsize=2.0)
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels([_display_name(m) for m in models], rotation=45, ha="right")
            if lower_better:
                ax.set_ylim(0, max(means) * 1.25 if means else 1)
            else:
                ax.set_ylim(0, 1.0)
            ax.grid(axis="y", color="0.9", linewidth=0.5)
        fig.suptitle(f"{_pretty_dataset(dataset)}: cross-session decoding and reliability", y=1.04, fontsize=9)
        fig.tight_layout(w_pad=0.8)
        _save(fig, plot_dir / f"figure_1_headline_performance_{dataset}")
        plt.close(fig)


def plot_risk_coverage(curve_df: pd.DataFrame, out_dir: Path) -> None:
    plt = _setup_matplotlib(out_dir)
    plot_dir = out_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    curve_df = _select_largest_runs(curve_df)
    for dataset, ds_df in curve_df.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(3.35, 2.55))
        for model in _ordered_models(ds_df):
            model_df = ds_df[ds_df["model"] == model]
            grouped = model_df.groupby("target_coverage", as_index=False).agg(risk=("risk", "mean"))
            ax.plot(
                grouped["target_coverage"],
                grouped["risk"],
                marker="o",
                markersize=3.0,
                linewidth=1.25,
                color=MODEL_COLORS.get(model, "0.35"),
                label=_display_name(model),
            )
        ax.set_xlabel("Coverage")
        ax.set_ylabel("Selective risk")
        ax.set_title(f"{_pretty_dataset(dataset)}: risk-coverage")
        ax.set_xlim(0, 1.02)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="0.9", linewidth=0.5)
        ax.legend(frameon=False, ncol=1, loc="upper left")
        fig.tight_layout()
        _save(fig, plot_dir / f"figure_2_risk_coverage_{dataset}")
        _save(fig, plot_dir / f"risk_coverage_{dataset}")
        plt.close(fig)


def plot_reliability(reliability_df: pd.DataFrame, out_dir: Path) -> None:
    plt = _setup_matplotlib(out_dir)
    plot_dir = out_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    reliability_df = _select_largest_runs(reliability_df)
    for dataset, ds_df in reliability_df.groupby("dataset"):
        fig, (ax, count_ax) = plt.subplots(
            2,
            1,
            figsize=(3.55, 3.15),
            height_ratios=[3.0, 1.0],
            sharex=True,
        )
        max_count = max(float(ds_df["count"].max()), 1.0)
        for model in sorted(ds_df["decoder"].dropna().unique(), key=_model_sort_key):
            grouped = _weighted_reliability_bins(ds_df[ds_df["decoder"] == model])
            if grouped.empty:
                continue
            sizes = 12.0 + 58.0 * np.sqrt(grouped["count"] / max_count)
            color = MODEL_COLORS.get(model, "0.35")
            ax.plot(
                grouped["confidence"],
                grouped["accuracy"],
                linewidth=1.05,
                color=color,
                alpha=0.92,
                label=_display_name(model),
            )
            ax.scatter(
                grouped["confidence"],
                grouped["accuracy"],
                s=sizes,
                color=color,
                edgecolor="white",
                linewidth=0.35,
                zorder=3,
            )
            count_ax.plot(
                grouped["confidence"],
                grouped["count"],
                marker="o",
                markersize=2.1,
                linewidth=0.9,
                color=color,
                alpha=0.8,
            )
        ax.plot([0, 1], [0, 1], "--", color="0.45", linewidth=0.8)
        ax.set_ylabel("Empirical accuracy")
        ax.set_title(f"{_pretty_dataset(dataset)}: trial-weighted reliability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        ax.grid(color="0.9", linewidth=0.5)
        ax.legend(frameon=False, ncol=1, loc="upper left")
        count_ax.set_xlabel("Predicted confidence")
        count_ax.set_ylabel("Trials/bin")
        count_ax.set_yscale("log")
        count_ax.grid(axis="y", color="0.9", linewidth=0.5)
        fig.tight_layout(h_pad=0.35)
        _save(fig, plot_dir / f"figure_3_reliability_support_{dataset}")
        _save(fig, plot_dir / f"reliability_{dataset}")
        plt.close(fig)


def plot_subject_deltas(subject_df: pd.DataFrame, out_dir: Path) -> None:
    plt = _setup_matplotlib(out_dir)
    plot_dir = out_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    subject_df = _select_largest_runs(subject_df)
    baseline = "v1_lda"
    highlight = ["eegnet_calibrated", "spatial_calibrated", "deep_ensemble_validation_weighted"]
    for dataset, ds_df in subject_df.groupby("dataset"):
        if baseline not in ds_df["model"].unique():
            continue
        metrics = [("accuracy_all", "All-trial accuracy delta"), ("selective_accuracy", "Selective accuracy delta")]
        fig, axes = plt.subplots(1, 2, figsize=(6.25, 2.35), sharey=True)
        for ax, (metric, title) in zip(axes, metrics):
            base = ds_df[ds_df["model"] == baseline][["subject", metric]].rename(columns={metric: "baseline"})
            y_positions = []
            y_labels = []
            for idx, model in enumerate([m for m in highlight if m in ds_df["model"].unique()]):
                merged = ds_df[ds_df["model"] == model][["subject", metric]].merge(base, on="subject")
                merged["delta"] = merged[metric] - merged["baseline"]
                x = merged.sort_values("subject")["delta"].to_numpy()
                y = np.full(x.shape, idx, dtype=float) + np.linspace(-0.16, 0.16, num=max(len(x), 1))
                ax.scatter(x, y, s=16, color=MODEL_COLORS.get(model, "0.35"), alpha=0.85, edgecolor="white", linewidth=0.25)
                ax.plot([float(np.nanmean(x)), float(np.nanmean(x))], [idx - 0.28, idx + 0.28], color=MODEL_COLORS.get(model, "0.35"), linewidth=1.6)
                y_positions.append(idx)
                y_labels.append(_display_name(model))
            ax.axvline(0, color="0.45", linewidth=0.8, linestyle="--")
            ax.set_title(title)
            ax.set_xlabel("Model - v1 LDA")
            ax.grid(axis="x", color="0.9", linewidth=0.5)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(y_labels)
        fig.suptitle(f"{_pretty_dataset(dataset)}: subject-paired gains", y=1.04, fontsize=9)
        fig.tight_layout(w_pad=0.9)
        _save(fig, plot_dir / f"figure_4_subject_deltas_{dataset}")
        plt.close(fig)


def plot_coverage_sweep(coverage_df: pd.DataFrame, out_dir: Path) -> None:
    plt = _setup_matplotlib(out_dir)
    plot_dir = out_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    coverage_df = _select_largest_runs(coverage_df)
    for dataset, ds_df in coverage_df.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(3.45, 2.45))
        for model in _ordered_models(ds_df):
            grouped = ds_df[ds_df["model"] == model].groupby("target_coverage", as_index=False).agg(
                selective_accuracy=("selective_accuracy", "mean")
            )
            ax.plot(
                grouped["target_coverage"],
                grouped["selective_accuracy"],
                marker="o",
                markersize=3.0,
                linewidth=1.2,
                color=MODEL_COLORS.get(model, "0.35"),
                label=_display_name(model),
            )
        ax.set_xlabel("Target coverage")
        ax.set_ylabel("Selective accuracy")
        ax.set_title(f"{_pretty_dataset(dataset)}: operating points")
        ax.set_xlim(0.34, 0.86)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", color="0.9", linewidth=0.5)
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        _save(fig, plot_dir / f"figure_5_coverage_sweep_{dataset}")
        plt.close(fig)


def generate_all_figures(out_dir: Path) -> None:
    curve_path = out_dir / "risk_coverage.csv"
    rel_path = out_dir / "reliability_diagram_bins.csv"
    subject_path = out_dir / "combined_subject_metrics.csv"
    coverage_path = out_dir / "combined_coverage_sweep.csv"
    if not subject_path.exists():
        subject_path = out_dir / "subject_metrics.csv"
    if not coverage_path.exists():
        coverage_path = out_dir / "coverage_sweep.csv"

    write_summary_tables(out_dir)
    if subject_path.exists():
        subject_df = pd.read_csv(subject_path)
        plot_headline_performance(subject_df, out_dir)
        plot_subject_deltas(subject_df, out_dir)
    if curve_path.exists():
        plot_risk_coverage(pd.read_csv(curve_path), out_dir)
    if rel_path.exists():
        plot_reliability(pd.read_csv(rel_path), out_dir)
    if coverage_path.exists():
        plot_coverage_sweep(pd.read_csv(coverage_path), out_dir)
