"""Dataset loading utilities for BCI IV 2a, BCI IIIa, and MOABB."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import BASE_DIR, BCI_IIIA_DATA_DIR, EPOCH_DIR, MOABB_DATA_DIR, RANDOM_SEED

BCI_IIIA_EVENT_ID = {
    "left_hand": 769,
    "right_hand": 770,
    "feet": 771,
    "tongue": 772,
}


@dataclass
class SubjectDataset:
    dataset: str
    subject: str | int
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    split: str


def load_bci_iv_2a_subject(subject: int, epoch_dir: Path = EPOCH_DIR) -> SubjectDataset:
    """Load one preprocessed BCI IV 2a subject in A0xT -> A0xE form."""

    train = f"A{subject:02d}T"
    test = f"A{subject:02d}E"
    paths = {
        "X_train": epoch_dir / f"X_{train}.npy",
        "y_train": epoch_dir / f"y_{train}.npy",
        "X_test": epoch_dir / f"X_{test}.npy",
        "y_test": epoch_dir / f"y_{test}.npy",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing preprocessed arrays:\n" + "\n".join(missing))
    return SubjectDataset(
        dataset="BCI_IV_2a",
        subject=subject,
        X_train=np.load(paths["X_train"]).astype(np.float32),
        y_train=np.load(paths["y_train"]),
        X_test=np.load(paths["X_test"]).astype(np.float32),
        y_test=np.load(paths["y_test"]),
        split="A0xT_to_A0xE",
    )


def iter_bci_iv_2a_subjects(subjects: Iterable[int]) -> list[SubjectDataset]:
    return [load_bci_iv_2a_subject(int(subject)) for subject in subjects]


def load_bci_iiia_subject(
    subject: str,
    data_dir: Path = BCI_IIIA_DATA_DIR,
    random_state: int = RANDOM_SEED,
) -> SubjectDataset:
    """Load one BCI Competition III Dataset IIIa subject from its GDF file.

    IIIa is distributed as one labeled recording per subject. To keep it
    external to BCI IV 2a without pretending it has a paired T/E session, this
    loader uses a stratified trial split and labels the split explicitly.
    """

    _prepare_moabb_env()
    try:
        import mne
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("MNE is required to load BCI IIIa GDF files.") from exc

    path = data_dir / f"{subject}.gdf"
    if not path.exists():
        raise FileNotFoundError(f"Missing BCI IIIa GDF file: {path}")

    raw = mne.io.read_raw_gdf(path, preload=True, verbose=False)
    raw.pick("eeg")
    raw.filter(
        8.0,
        30.0,
        method="iir",
        iir_params={"order": 5, "ftype": "butter"},
        verbose=False,
    )
    events, annotation_ids = mne.events_from_annotations(raw, verbose=False)
    event_id = {name: annotation_ids[str(code)] for name, code in BCI_IIIA_EVENT_ID.items() if str(code) in annotation_ids}
    if len(event_id) < 2:
        raise ValueError(f"{subject}: expected at least two motor-imagery event classes, found {event_id}")
    cue_events = events[np.isin(events[:, 2], list(event_id.values()))]
    code_by_name = BCI_IIIA_EVENT_ID
    name_by_event = {value: name for name, value in event_id.items()}
    labels = np.asarray([code_by_name[name_by_event[int(event)]] for event in cue_events[:, 2]], dtype=int)
    epochs = mne.Epochs(
        raw,
        cue_events,
        event_id,
        tmin=0.0,
        tmax=4.5,
        baseline=None,
        preload=True,
        event_repeated="drop",
        verbose=False,
    )
    X = epochs.get_data(copy=True).astype(np.float32)
    y = labels[epochs.selection]
    idx = np.arange(y.size)
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.35,
        random_state=random_state,
        stratify=y,
    )
    return SubjectDataset(
        dataset="BCI_IIIa",
        subject=subject,
        X_train=X[train_idx],
        y_train=y[train_idx],
        X_test=X[test_idx],
        y_test=y[test_idx],
        split="stratified_trial_split",
    )


def iter_bci_iiia_subjects(subjects: Iterable[str], random_state: int = RANDOM_SEED) -> list[SubjectDataset]:
    return [load_bci_iiia_subject(str(subject), random_state=random_state) for subject in subjects]


def _prepare_moabb_env() -> None:
    fake_home = BASE_DIR / ".mne_home"
    (fake_home / ".mne").mkdir(parents=True, exist_ok=True)
    MOABB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("_MNE_FAKE_HOME_DIR", str(fake_home))
    os.environ.setdefault("MNE_DATA", str(MOABB_DATA_DIR))
    os.environ.setdefault("MNE_LOGGING_LEVEL", "WARNING")


def _import_moabb():
    _prepare_moabb_env()
    try:
        import moabb
        from moabb import datasets as moabb_datasets
        from moabb.paradigms import LeftRightImagery
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("MOABB is required for --include-moabb runs.") from exc
    moabb.set_log_level("warning")
    return moabb_datasets, LeftRightImagery


def _dataset_instance(name: str):
    moabb_datasets, _ = _import_moabb()
    try:
        return getattr(moabb_datasets, name)()
    except AttributeError as exc:
        raise ValueError(f"Unknown MOABB dataset: {name}") from exc


def _split_moabb_subject(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    for column in ("session", "run"):
        if column in meta.columns:
            values = list(pd.Series(meta[column]).dropna().unique())
            if len(values) > 1:
                train_values = set(values[:-1])
                train_mask = meta[column].isin(train_values).to_numpy()
                test_mask = ~train_mask
                if train_mask.any() and test_mask.any():
                    return X[train_mask], y[train_mask], X[test_mask], y[test_mask], f"held_out_{column}"
    idx = np.arange(y.size)
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.35,
        random_state=random_state,
        stratify=y,
    )
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx], "stratified_trial_split"


def iter_moabb_subjects(
    dataset_names: list[str],
    subject_limit: int | None,
    random_state: int = RANDOM_SEED,
) -> list[SubjectDataset]:
    """Load MOABB LeftRightImagery subjects as train/test subject datasets."""

    _, LeftRightImagery = _import_moabb()
    out: list[SubjectDataset] = []
    for dataset_name in dataset_names:
        dataset = _dataset_instance(dataset_name)
        subjects = list(getattr(dataset, "subject_list", []))
        if subject_limit is not None:
            subjects = subjects[:subject_limit]
        paradigm = LeftRightImagery(fmin=8.0, fmax=30.0, resample=250.0)
        X, y, meta = paradigm.get_data(dataset=dataset, subjects=subjects)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        meta = pd.DataFrame(meta)
        for subject in subjects:
            mask = meta["subject"].astype(str).to_numpy() == str(subject)
            if mask.sum() < 20:
                continue
            X_sub = X[mask]
            y_sub = y[mask]
            meta_sub = meta.loc[mask].reset_index(drop=True)
            if np.unique(y_sub).size < 2:
                continue
            X_train, y_train, X_test, y_test, split = _split_moabb_subject(
                X_sub,
                y_sub,
                meta_sub,
                random_state=random_state,
            )
            if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
                continue
            out.append(
                SubjectDataset(
                    dataset=dataset_name,
                    subject=subject,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    split=split,
                )
            )
    return out
