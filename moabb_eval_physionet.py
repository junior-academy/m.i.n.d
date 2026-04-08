from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path


def _require(pkg: str, hint: str):
    try:
        return __import__(pkg)
    except ModuleNotFoundError as e:
        raise SystemExit(f"Missing dependency: {pkg}\n\nInstall:\n  {hint}\n") from e


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a MOABB within-session Motor Imagery evaluation using CSP + (LDA, SVM).\n\n"
            "This is intended for external validation/generalization beyond BCI Competition IV 2a.\n"
            "MOABB will download/cache the dataset via MNE's data directory."
        )
    )
    parser.add_argument("--out-dir", type=Path, default=Path("m.i.n.d/outputs/moabb/physionetmi"))
    parser.add_argument(
        "--dataset",
        type=str,
        default="physionetmi",
        choices=["physionetmi", "bnci2014_001", "lee2019_mi", "alexmi"],
        help="MOABB dataset to evaluate (default: physionetmi).",
    )
    parser.add_argument("--n-classes", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument(
        "--events",
        type=str,
        default="",
        help=(
            "Comma-separated event names to include (e.g., left_hand,right_hand). "
            "If omitted, defaults to dataset-appropriate events (PhysionetMI+2 classes: left_hand,right_hand)."
        ),
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="accuracy",
        help=(
            "MOABB scoring metric (default: accuracy). "
            "If you use >2 classes, prefer accuracy; roc_auc needs a multi-class variant."
        ),
    )
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="simple",
        choices=["simple", "moabb"],
        help=(
            "Evaluation engine (default: simple). "
            "simple=use MOABB for data loading, then run sklearn CV with accuracy. "
            "moabb=use MOABB's WithinSessionEvaluation."
        ),
    )
    parser.add_argument("--cv-splits", type=int, default=5, help="CV splits for --eval-mode simple (default: 5).")
    parser.add_argument("--subjects", type=str, default="",
                        help="Comma-separated subject ids to run (e.g., 1,2,3). Empty = MOABB default subset.")
    parser.add_argument("--n-components", type=int, default=6, help="CSP components (default: 6).")
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--svm-gamma", type=str, default="scale")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel jobs inside MOABB (default: 1).")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if cached results exist.")
    parser.add_argument(
        "--mne-data",
        type=Path,
        default=None,
        help=(
            "Optional directory to use as MNE_DATA for caching/downloading (e.g., m.i.n.d/data/mne_data). "
            "If set, this overrides the current process MNE_DATA."
        ),
    )
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=120,
        help="HTTP read timeout (seconds) used by pooch downloader when fetching dataset files (default: 120).",
    )
    parser.add_argument(
        "--download-retries",
        type=int,
        default=5,
        help="HTTP retries for dataset downloads (default: 5).",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    _require("moabb", "python3 -m pip install moabb")
    _require("pyriemann", "python3 -m pip install pyriemann")
    _require("mne", "python3 -m pip install mne")
    _require("sklearn", "python3 -m pip install scikit-learn")
    _require("pooch", "python3 -m pip install pooch")
    _require("requests", "python3 -m pip install requests")

    if args.mne_data is not None:
        os.environ["MNE_DATA"] = str(args.mne_data.expanduser().resolve())

    # Increase download timeout + robustness for slow PhysioNet responses.
    import pooch  # type: ignore
    import requests  # type: ignore
    from requests.adapters import HTTPAdapter  # type: ignore
    from urllib3.util.retry import Retry  # type: ignore

    try:
        retry = Retry(
            total=int(args.download_retries),
            connect=int(args.download_retries),
            read=int(args.download_retries),
            status=int(args.download_retries),
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "HEAD"),
            raise_on_status=False,
        )
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.mount("http://", HTTPAdapter(max_retries=retry))
        pooch.core.DEFAULT_DOWNLOADER = pooch.HTTPDownloader(
            progressbar=True,
            timeout=int(args.download_timeout),
            session=session,
        )
    except Exception:
        # Best-effort only; keep going with pooch defaults.
        pass

    from moabb.datasets import AlexMI, BNCI2014_001, Lee2019_MI, PhysionetMI
    from moabb.paradigms import MotorImagery

    from mne.decoding import CSP
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold

    if args.subjects.strip():
        subjects = [int(s.strip()) for s in args.subjects.split(",") if s.strip()]
    else:
        subjects = None

    if args.dataset == "physionetmi":
        ds = PhysionetMI()
    elif args.dataset == "bnci2014_001":
        ds = BNCI2014_001()
    elif args.dataset == "lee2019_mi":
        ds = Lee2019_MI()
    elif args.dataset == "alexmi":
        ds = AlexMI()
    else:
        raise SystemExit(f"Unknown dataset: {args.dataset}")
    if subjects is not None:
        # MOABB subjects are 1-indexed for this dataset.
        ds.subject_list = subjects

    # Avoid PhysionetMI mixing multiple event sets (hands/feet/left/right/rest) by selecting events explicitly.
    # If you don't, MOABB can "choose from all possible events" and you may end up with >2 classes.
    if args.events.strip():
        events = [e.strip() for e in args.events.split(",") if e.strip()]
    else:
        if args.dataset == "physionetmi" and args.n_classes == 2:
            events = ["left_hand", "right_hand"]
        else:
            events = []

    if events:
        paradigm = MotorImagery(events=events, n_classes=min(args.n_classes, len(events)))
    else:
        paradigm = MotorImagery(n_classes=args.n_classes)

    pipelines = {
        "CSP+LDA": Pipeline(
            [
                ("csp", CSP(n_components=args.n_components, reg="ledoit_wolf", log=True, norm_trace=False)),
                ("clf", LDA()),
            ]
        ),
        "CSP+SVM": Pipeline(
            [
                ("csp", CSP(n_components=args.n_components, reg="ledoit_wolf", log=True, norm_trace=False)),
                ("clf", SVC(C=args.svm_c, kernel="rbf", gamma=args.svm_gamma, probability=True)),
            ]
        ),
    }

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_mode == "moabb":
        from moabb.evaluations import WithinSessionEvaluation

        # Note: some MOABB versions may still default to roc_auc internally.
        # If you see roc_auc multiclass errors, use --eval-mode simple.
        try:
            evaluation = WithinSessionEvaluation(
                paradigm=paradigm,
                datasets=[ds],
                overwrite=bool(args.overwrite),
                random_state=args.random_state,
                n_jobs=args.n_jobs,
                scoring=args.scoring,
            )
        except TypeError:
            evaluation = WithinSessionEvaluation(
                paradigm=paradigm,
                datasets=[ds],
                overwrite=bool(args.overwrite),
                random_state=args.random_state,
                n_jobs=args.n_jobs,
            )

        results = evaluation.process(pipelines)

        raw_csv = out_dir / "results_raw.csv"
        results.to_csv(raw_csv, index=False)

        summary = (
            results.groupby(["pipeline"])
            .agg(score_mean=("score", "mean"), score_sd=("score", "std"), n_subjects=("subject", "nunique"))
            .reset_index()
            .sort_values("score_mean", ascending=False)
        )
    else:
        # Simple evaluation: MOABB for (download+epoching+labels), then sklearn CV with accuracy.
        X, y, meta = paradigm.get_data(dataset=ds, subjects=ds.subject_list, return_epochs=False)
        meta = meta.reset_index(drop=True)

        rows = []
        cv = StratifiedKFold(n_splits=int(args.cv_splits), shuffle=True, random_state=args.random_state)

        for subj in sorted(meta["subject"].unique().tolist()):
            idx = meta.index[meta["subject"] == subj].to_numpy()
            Xs = X[idx]
            ys = y[idx]
            for pipe_name, pipe in pipelines.items():
                for fold, (tr, te) in enumerate(cv.split(Xs, ys), start=1):
                    pipe.fit(Xs[tr], ys[tr])
                    pred = pipe.predict(Xs[te])
                    acc = float((pred == ys[te]).mean())
                    rows.append(
                        {
                            "dataset": args.dataset,
                            "subject": int(subj),
                            "pipeline": pipe_name,
                            "fold": int(fold),
                            "score": acc,
                            "n_train": int(tr.size),
                            "n_test": int(te.size),
                        }
                    )

        import pandas as pd

        results = pd.DataFrame(rows)
        raw_csv = out_dir / "results_raw.csv"
        results.to_csv(raw_csv, index=False)

        summary = (
            results.groupby(["pipeline"])
            .agg(score_mean=("score", "mean"), score_sd=("score", "std"), n_subjects=("subject", "nunique"))
            .reset_index()
            .sort_values("score_mean", ascending=False)
        )
    summary_csv = out_dir / "results_summary.csv"
    summary.to_csv(summary_csv, index=False)

    # Per-subject mean scores (useful for paired stats tests).
    per_subject = (
        results.groupby(["pipeline", "subject"])
        .agg(score_mean=("score", "mean"), score_sd=("score", "std"), n_folds=("score", "count"))
        .reset_index()
        .sort_values(["pipeline", "subject"])
    )
    per_subject_csv = out_dir / "results_per_subject.csv"
    per_subject.to_csv(per_subject_csv, index=False)

    print("MOABB eval complete.")
    print(f"- MNE_DATA={os.environ.get('MNE_DATA', '(not set; using MNE config/default)')}")
    print(f"- Dataset={args.dataset}")
    print(f"- eval_mode={args.eval_mode}")
    if events:
        print(f"- events={','.join(events)}")
    print(f"- download_timeout_s={args.download_timeout}")
    print(f"- download_retries={args.download_retries}")
    print(f"- Wrote: {raw_csv}")
    print(f"- Wrote: {summary_csv}")
    print(f"- Wrote: {per_subject_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
