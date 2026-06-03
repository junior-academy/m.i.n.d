"""Preprocess BCI Competition IV 2a GDF files into T/E epoch arrays."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import mne
import numpy as np

from config import (
    DATA_DIR,
    EPOCH_DIR,
    FILTER_HIGH,
    FILTER_LOW,
    ICA_N_COMPONENTS,
    RANDOM_SEED,
    SUBJECTS_TEST,
    SUBJECTS_TRAIN,
    TMAX,
    TMIN,
)

EVENT_ID = {
    "Left Hand": 769,
    "Right Hand": 770,
    "Feet": 771,
    "Tongue": 772,
}


def _mark_eog(raw: mne.io.BaseRaw) -> None:
    eog = [ch for ch in raw.ch_names if "EOG" in ch.upper()]
    if not eog and raw.info["nchan"] >= 25:
        eog = raw.ch_names[-3:]
    if eog:
        raw.set_channel_types({ch: "eog" for ch in eog})


def preprocess_subject(subject: str, data_dir: Path = DATA_DIR, out_dir: Path = EPOCH_DIR) -> None:
    """Create ``X_<subject>.npy`` and ``y_<subject>.npy`` for one T or E file."""

    gdf_path = data_dir / f"{subject}.gdf"
    if not gdf_path.exists():
        raise FileNotFoundError(f"Missing raw file: {gdf_path}")

    print(f"[preprocess] {subject}")
    raw = mne.io.read_raw_gdf(gdf_path, preload=True, verbose=False)
    _mark_eog(raw)
    raw.filter(
        FILTER_LOW,
        FILTER_HIGH,
        method="iir",
        iir_params={"order": 5, "ftype": "butter"},
        verbose=False,
    )

    ica = mne.preprocessing.ICA(
        n_components=ICA_N_COMPONENTS,
        random_state=RANDOM_SEED,
        max_iter="auto",
        verbose=False,
    )
    ica.fit(raw, verbose=False)
    eog_picks = mne.pick_types(raw.info, eog=True)
    if len(eog_picks) > 0:
        eog_name = raw.ch_names[int(eog_picks[0])]
        bads, _ = ica.find_bads_eog(raw, ch_name=eog_name, verbose=False)
        ica.exclude = list(bads)
    ica.apply(raw, verbose=False)

    raw.pick_types(eeg=True, eog=False, stim=False, misc=False)
    events, annotation_ids = mne.events_from_annotations(raw, verbose=False)
    epoch_event_id = {name: annotation_ids[str(code)] for name, code in EVENT_ID.items()}
    epochs = mne.Epochs(
        raw,
        events,
        epoch_event_id,
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,
        preload=True,
        reject={"eeg": 100e-6},
        event_repeated="drop",
        verbose=False,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"X_{subject}.npy", epochs.get_data(copy=True))
    np.save(out_dir / f"y_{subject}.npy", epochs.events[:, 2])


def preprocess_many(subjects: Iterable[str], data_dir: Path = DATA_DIR, out_dir: Path = EPOCH_DIR) -> None:
    for subject in subjects:
        preprocess_subject(subject, data_dir=data_dir, out_dir=out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess BCI IV 2a train and evaluation sessions.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=EPOCH_DIR)
    parser.add_argument("--sessions", choices=("all", "T", "E"), default="all")
    args = parser.parse_args()

    if args.sessions == "T":
        subjects = SUBJECTS_TRAIN
    elif args.sessions == "E":
        subjects = SUBJECTS_TEST
    else:
        subjects = (*SUBJECTS_TRAIN, *SUBJECTS_TEST)
    preprocess_many(subjects, data_dir=args.data_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
