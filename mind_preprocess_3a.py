from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from tqdm.auto import tqdm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(it=None, **_kwargs):  # type: ignore
        return it if it is not None else []


def _require_mne():
    try:
        import mne  # type: ignore

        return mne
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing dependency: mne\n\n"
            "Install it in your current environment:\n"
            "  python -m pip install mne\n"
        ) from e


def _infer_4class_event_id(event_id: Dict[str, int], counts: Dict[str, int]) -> Dict[str, int]:
    """
    Infer the 4 motor imagery class events for BCI Competition III dataset IIIa.

    Preferred mapping uses standard cue codes 769-772 (same as BCIC IV 2a).
    Fallback: pick the 4 most common non-rest event keys.
    """
    # Preferred: standard BCI cue codes used in several Graz datasets.
    preferred = ["769", "770", "771", "772"]
    if all(k in event_id for k in preferred):
        return {
            "Left Hand": event_id["769"],
            "Right Hand": event_id["770"],
            "Feet": event_id["771"],
            "Tongue": event_id["772"],
        }

    # Try name-based keys if present.
    lower = {k.lower(): k for k in event_id.keys()}
    name_map = {}
    for want, candidates in [
        ("Left Hand", ["left", "hand left", "lh"]),
        ("Right Hand", ["right", "hand right", "rh"]),
        ("Feet", ["feet", "foot", "both feet"]),
        ("Tongue", ["tongue"]),
    ]:
        found = None
        for c in candidates:
            for k_low, k in lower.items():
                if c in k_low:
                    found = k
                    break
            if found:
                break
        if found:
            name_map[want] = event_id[found]

    if len(name_map) == 4:
        return name_map

    # Fallback: pick top-4 by count excluding obvious non-class markers.
    # Many GDF files include "rest"/"trial start"/boundary markers; drop common ones.
    exclude = {"0", "1023", "768", "783", "boundary"}
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    picked: List[str] = []
    for k, _ in ranked:
        kl = k.lower()
        if k in exclude or kl in exclude:
            continue
        if "boundary" in kl or "rest" in kl:
            continue
        picked.append(k)
        if len(picked) == 4:
            break
    if len(picked) != 4:
        raise ValueError(
            f"Could not infer 4 class events. Available keys: {sorted(event_id.keys())[:40]} (showing first 40)."
        )

    # We don't know the semantic order; keep a stable ordering for reproducibility.
    picked = sorted(picked)
    return {f"Class{i+1}:{picked[i]}": event_id[picked[i]] for i in range(4)}


def preprocess_gdf(
    *,
    gdf_path: Path,
    out_dir: Path,
    subject_key: str,
    l_freq: float,
    h_freq: float,
    tmin: float,
    tmax: float,
    reject_uV: float,
) -> Tuple[Path, Path]:
    mne = _require_mne()

    print(f"\nProcessing {subject_key} ({gdf_path.name})...")
    raw = mne.io.read_raw_gdf(str(gdf_path), preload=True)

    # Some BCI IIIa files may not carry correct channel types. Mark all channels as EEG.
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})

    raw.filter(l_freq, h_freq, method="iir", iir_params=dict(order=5, ftype="butter"))

    raw.pick_types(eeg=True, eog=False, stim=False, misc=False)
    print(f"Remaining channels: {raw.info['nchan']}")

    events, event_id = mne.events_from_annotations(raw)
    # Count events per key (for inference + debugging)
    inv = {v: k for k, v in event_id.items()}
    counts: Dict[str, int] = {}
    for e in events[:, 2]:
        k = inv.get(int(e), str(int(e)))
        counts[k] = counts.get(k, 0) + 1

    epoch_event_id = _infer_4class_event_id(event_id, counts)
    print("Using epoch events:", epoch_event_id)

    epochs = mne.Epochs(
        raw,
        events,
        epoch_event_id,
        tmin=float(tmin),
        tmax=float(tmax),
        baseline=None,
        preload=True,
        reject={"eeg": float(reject_uV) * 1e-6} if reject_uV > 0 else None,
        event_repeated="drop",
    )

    X = epochs.get_data()
    y = epochs.events[:, 2]
    print(f"X shape: {X.shape}, y shape: {y.shape}, classes: {np.unique(y)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    x_path = out_dir / f"X_{subject_key}.npy"
    y_path = out_dir / f"y_{subject_key}.npy"
    np.save(x_path, X)
    np.save(y_path, y)
    print(f"Saved: {x_path.name}, {y_path.name}")
    return x_path, y_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess BCI Competition III dataset IIIa (.gdf) into X/y NumPy.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "BCICIV_3a_gdf",
        help="Folder containing IIIa .gdf files (default: m.i.n.d/data/BCICIV_3a_gdf).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "analysis_results_3a",
        help="Output folder for X_*.npy and y_*.npy (default: m.i.n.d/analysis_results_3a).",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default="k3b,k6b,l1b",
        help="Comma-separated subject keys (basename without .gdf). Default: k3b,k6b,l1b",
    )
    parser.add_argument("--l-freq", type=float, default=8.0)
    parser.add_argument("--h-freq", type=float, default=30.0)
    parser.add_argument("--tmin", type=float, default=0.0)
    parser.add_argument("--tmax", type=float, default=4.0)
    parser.add_argument(
        "--reject-uv",
        type=float,
        default=100.0,
        help="Epoch reject threshold in microvolts (0 disables). Default: 100.",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    out_dir: Path = args.out_dir
    subjects = [s.strip() for s in str(args.subjects).split(",") if s.strip()]
    if not subjects:
        raise SystemExit("--subjects must not be empty")

    if not data_dir.exists():
        raise SystemExit(f"Missing data dir: {data_dir}")

    for sk in tqdm(subjects, total=len(subjects), desc="Preprocessing (IIIa)", unit="subj"):
        gdf = data_dir / f"{sk}.gdf"
        if not gdf.exists():
            print(f"Missing {gdf}; skipping")
            continue
        try:
            preprocess_gdf(
                gdf_path=gdf,
                out_dir=out_dir,
                subject_key=sk,
                l_freq=float(args.l_freq),
                h_freq=float(args.h_freq),
                tmin=float(args.tmin),
                tmax=float(args.tmax),
                reject_uV=float(args.reject_uv),
            )
        except Exception as e:
            print(f"Error processing {sk}: {e}")

    print(f"\nIIIa preprocessing complete. Outputs saved in: {out_dir}")


if __name__ == "__main__":
    main()
