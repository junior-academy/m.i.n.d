"""CLI and orchestration for M.I.N.D. v2 experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder

from adaptation import adapt_train_test

from gating import DebounceConfig, debounced_gate, toggle_rate, wrong_fire_rate_all

from .config import ALIGNMENT_EPS, BCI_IIIA_SUBJECT_IDS, DEFAULT_COVERAGE, DEEP_EPOCHS, OUTPUT_DIR, RANDOM_SEED, SUBJECT_IDS
from .datasets import SubjectDataset, iter_bci_iiia_subjects, iter_bci_iv_2a_subjects, iter_moabb_subjects
from .evaluation import coverage_sweep_rows, reliability_bin_rows, risk_coverage_rows, subject_metric_row, write_tables
from .models import (
    ModelOutput,
    equal_ensemble,
    fit_deep_model_variants,
    fit_v1_lda,
    validation_weighted_ensemble,
)
from .plotting import generate_all_figures


def _prepare_runtime_env(out_dir: Path) -> None:
    cache_dir = out_dir / ".cache"
    mpl_dir = out_dir / ".mplconfig"
    cache_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MNE_LOGGING_LEVEL", "WARNING")


def _encode_subject_labels(data: SubjectDataset) -> SubjectDataset:
    le = LabelEncoder()
    y_train = le.fit_transform(data.y_train)
    y_test = le.transform(data.y_test)
    return SubjectDataset(
        dataset=data.dataset,
        subject=data.subject,
        X_train=data.X_train,
        y_train=y_train,
        X_test=data.X_test,
        y_test=y_test,
        split=data.split,
    )


def _prepare_data(data: SubjectDataset, use_ea: bool) -> SubjectDataset:
    data = _encode_subject_labels(data)
    if not use_ea:
        return data
    X_train, X_test = adapt_train_test(
        data.X_train,
        data.X_test,
        method="euclidean",
        eps=ALIGNMENT_EPS,
    )
    return SubjectDataset(data.dataset, data.subject, X_train, data.y_train, X_test, data.y_test, data.split)


def _requested_deep_models(model: str) -> list[str]:
    if model == "all":
        return ["eegnet", "spatial"]
    if model in {"eegnet", "spatial"}:
        return [model]
    return []


def _model_outputs_for_subject(
    data: SubjectDataset,
    *,
    model: str,
    calibrate: bool,
    ensemble: str,
    ensemble_weights: str,
    post_ensemble_calibration: bool,
    epochs: int,
    seed: int,
) -> list[ModelOutput]:
    outputs: list[ModelOutput] = []
    if model in {"lda", "all"} or ensemble == "hybrid":
        outputs.append(
            fit_v1_lda(
                subject=data.subject,
                X_train=data.X_train,
                y_train=data.y_train,
                X_test=data.X_test,
                y_test=data.y_test,
            )
        )

    deep_outputs: list[ModelOutput] = []
    ensemble_deep_members: list[ModelOutput] = []
    for deep_name in _requested_deep_models(model):
        print(f"[v2] subject={data.subject} fitting {deep_name} epochs={epochs}", flush=True)
        variants = fit_deep_model_variants(
            name=deep_name,
            X_train=data.X_train,
            y_train=data.y_train,
            X_test=data.X_test,
            y_test=data.y_test,
            include_calibrated=calibrate,
            epochs=epochs,
            seed=seed,
        )
        deep_outputs.extend(variants)
        ensemble_deep_members.append(variants[-1])
    outputs.extend(deep_outputs)

    if ensemble in {"deep", "hybrid"}:
        members = ensemble_deep_members if ensemble == "deep" else [out for out in outputs if out.name == "v1_lda"] + ensemble_deep_members
        if len(members) >= 2:
            if ensemble_weights == "validation":
                outputs.append(
                    validation_weighted_ensemble(
                        f"{ensemble}_ensemble_validation_weighted",
                        members,
                        post_calibrate=post_ensemble_calibration,
                    )
                )
            else:
                outputs.append(
                    equal_ensemble(
                        f"{ensemble}_ensemble_equal",
                        members,
                        post_calibrate=post_ensemble_calibration,
                    )
                )
    return outputs


def run_experiment(
    *,
    model: str,
    use_ea: bool,
    calibrate: bool,
    ensemble: str,
    ensemble_weights: str,
    post_ensemble_calibration: bool,
    uncertainty_score: str,
    coverage_target: float,
    include_moabb: bool,
    moabb_datasets: list[str],
    moabb_subject_limit: int | None,
    include_bci_iiia: bool,
    out_dir: Path,
    subjects: tuple[int, ...] = SUBJECT_IDS,
    bci_iiia_subjects: tuple[str, ...] = BCI_IIIA_SUBJECT_IDS,
    epochs: int = DEEP_EPOCHS,
    seed: int = RANDOM_SEED,
) -> Path:
    """Run a v2 experiment and write paper-ready CSV/plot outputs."""

    _prepare_runtime_env(out_dir)
    np.random.seed(seed)
    datasets = iter_bci_iv_2a_subjects(subjects)
    if include_bci_iiia:
        datasets.extend(iter_bci_iiia_subjects(bci_iiia_subjects, random_state=seed))
    if include_moabb:
        datasets.extend(iter_moabb_subjects(moabb_datasets, moabb_subject_limit, random_state=seed))

    subject_rows: list[dict] = []
    curve_rows: list[dict] = []
    rel_rows: list[dict] = []
    coverage_rows: list[dict] = []
    controller_rows: list[dict] = []

    for raw_data in datasets:
        print(f"[v2] dataset={raw_data.dataset} subject={raw_data.subject} split={raw_data.split}", flush=True)
        data = _prepare_data(raw_data, use_ea=use_ea)
        outputs = _model_outputs_for_subject(
            data,
            model=model,
            calibrate=calibrate,
            ensemble=ensemble,
            ensemble_weights=ensemble_weights,
            post_ensemble_calibration=post_ensemble_calibration,
            epochs=epochs,
            seed=seed,
        )
        for out in outputs:
            subject_rows.append(
                subject_metric_row(
                    dataset=data.dataset,
                    subject=data.subject,
                    split=data.split,
                    model=out.name,
                    y_true=out.y_true,
                    proba=out.proba,
                    uncertainty_score=uncertainty_score,
                    coverage_target=coverage_target,
                    use_ea=use_ea,
                    calibrated=out.calibrated,
                )
            )
            curve_rows.extend(
                risk_coverage_rows(
                    dataset=data.dataset,
                    subject=data.subject,
                    model=out.name,
                    y_true=out.y_true,
                    proba=out.proba,
                    uncertainty_score=uncertainty_score,
                )
            )
            coverage_rows.extend(
                coverage_sweep_rows(
                    dataset=data.dataset,
                    subject=data.subject,
                    model=out.name,
                    y_true=out.y_true,
                    proba=out.proba,
                    uncertainty_score=uncertainty_score,
                )
            )
            rel_rows.extend(reliability_bin_rows(data.dataset, data.subject, out.name, out.y_true, out.proba))
            for cfg_name, cfg in (
                ("balanced_0.60_0.55_3of5", DebounceConfig(t_on=0.60, t_off=0.55, k=3, n=5)),
                ("strict_0.80_0.75_3of5", DebounceConfig(t_on=0.80, t_off=0.75, k=3, n=5)),
            ):
                fired, latched = debounced_gate(p_ens=out.proba, cfg=cfg)
                wrong_fire = wrong_fire_rate_all(out.y_true, latched, fired)
                coverage = float(fired.mean())
                controller_rows.append(
                    {
                        "dataset": data.dataset,
                        "subject": data.subject,
                        "split": data.split,
                        "model": out.name,
                        "controller": cfg_name,
                        "t_on": cfg.t_on,
                        "t_off": cfg.t_off,
                        "k": cfg.k,
                        "n": cfg.n,
                        "coverage": coverage,
                        "wrong_fire_rate_all": wrong_fire,
                        "safe_fire": coverage - wrong_fire,
                        "toggle_rate": toggle_rate(fired),
                    }
                )

    write_tables(out_dir, subject_rows, curve_rows, rel_rows, coverage_rows, controller_rows)
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "model": model,
                "use_ea": use_ea,
                "calibrate": calibrate,
                "ensemble": ensemble,
                "ensemble_weights": ensemble_weights,
                "post_ensemble_calibration": post_ensemble_calibration,
                "uncertainty_score": uncertainty_score,
                "coverage_target": coverage_target,
                "include_moabb": include_moabb,
                "moabb_datasets": moabb_datasets,
                "moabb_subject_limit": moabb_subject_limit,
                "include_bci_iiia": include_bci_iiia,
                "bci_iiia_subjects": list(bci_iiia_subjects),
                "epochs": epochs,
                "seed": seed,
            },
            indent=2,
        )
    )
    try:
        generate_all_figures(out_dir)
    except Exception as exc:
        (out_dir / "plotting_error.txt").write_text(str(exc))
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run M.I.N.D. v2 reliability-aware experiments.")
    parser.add_argument("--model", choices=("lda", "eegnet", "spatial", "all"), default="all")
    parser.add_argument("--use-ea", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--ensemble", choices=("none", "deep", "hybrid"), default="deep")
    parser.add_argument("--ensemble-weights", choices=("equal", "validation"), default="equal")
    parser.add_argument("--post-ensemble-calibration", action="store_true")
    parser.add_argument("--uncertainty-score", choices=("max-prob", "entropy", "margin"), default="max-prob")
    parser.add_argument("--coverage-target", type=float, default=DEFAULT_COVERAGE)
    parser.add_argument("--include-moabb", action="store_true")
    parser.add_argument("--moabb-datasets", nargs="+", default=["Cho2017", "PhysionetMI"])
    parser.add_argument("--moabb-subject-limit", type=int, default=None)
    parser.add_argument("--include-bci-iiia", action="store_true")
    parser.add_argument("--bci-iiia-subjects", nargs="+", default=list(BCI_IIIA_SUBJECT_IDS))
    parser.add_argument("--subject-limit", type=int, default=None, help="Limit BCI IV 2a subjects for smoke tests.")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR / "custom")
    parser.add_argument("--epochs", type=int, default=DEEP_EPOCHS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    subjects = SUBJECT_IDS if args.subject_limit is None else SUBJECT_IDS[: args.subject_limit]
    result = run_experiment(
        model=args.model,
        use_ea=args.use_ea,
        calibrate=args.calibrate,
        ensemble=args.ensemble,
        ensemble_weights=args.ensemble_weights,
        post_ensemble_calibration=args.post_ensemble_calibration,
        uncertainty_score=args.uncertainty_score,
        coverage_target=args.coverage_target,
        include_moabb=args.include_moabb,
        moabb_datasets=args.moabb_datasets,
        moabb_subject_limit=args.moabb_subject_limit,
        include_bci_iiia=args.include_bci_iiia,
        out_dir=args.out_dir,
        subjects=subjects,
        bci_iiia_subjects=tuple(args.bci_iiia_subjects),
        epochs=args.epochs,
        seed=args.seed,
    )
    print(f"Wrote {result}")


if __name__ == "__main__":
    main()
