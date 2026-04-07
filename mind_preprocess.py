import mne
import numpy as np
import os
from config import *

try:
    from tqdm.auto import tqdm  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    def tqdm(it=None, **_kwargs):  # type: ignore
        return it if it is not None else []

def _save_channel_selection_visual(raw_before, raw_after, subject: str):
    """
    Save a simple visual report showing channel-type counts and kept EEG channel names.
    Writes to PLOTS_PATH (analysis_results/plots).
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    def _type_counts(raw):
        types = raw.get_channel_types()
        out = {}
        for t in types:
            out[t] = out.get(t, 0) + 1
        return out

    before_counts = _type_counts(raw_before)
    after_counts = _type_counts(raw_after)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    # Bar chart of counts
    all_types = sorted(set(before_counts) | set(after_counts))
    before_vals = [before_counts.get(t, 0) for t in all_types]
    after_vals = [after_counts.get(t, 0) for t in all_types]
    x = np.arange(len(all_types))
    width = 0.35
    axes[0].bar(x - width / 2, before_vals, width, label="Before pick_types")
    axes[0].bar(x + width / 2, after_vals, width, label="After pick_types")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(all_types, rotation=0)
    axes[0].set_ylabel("Channel count")
    axes[0].set_title(f"{subject}: Channel types before/after EEG-only selection")
    axes[0].legend(fontsize=8, frameon=False)

    # Text panel listing kept channel names
    kept = raw_after.ch_names
    axes[1].axis("off")
    text = "Kept channels (EEG-only):\n" + ", ".join(kept)
    axes[1].text(0.0, 1.0, text, va="top", ha="left", fontsize=9, wrap=True)

    os.makedirs(PLOTS_PATH, exist_ok=True)
    out_path = os.path.join(PLOTS_PATH, f"channel_selection_{subject}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def preprocess_subject(subject):
    """Preprocess a single subject's data and save X, y arrays"""
    print(f"\nProcessing {subject}...")
    
    # Load raw data
    raw = mne.io.read_raw_gdf(f'{DATA_PATH}/{subject}.gdf', preload=True)
    raw_before_pick = raw.copy()

    # Ensure EOG channels are correctly typed so we can drop them later.
    # Some GDF readers may import all channels as EEG unless we explicitly set types.
    eog_candidates = [ch for ch in raw.ch_names if "EOG" in ch.upper()]
    if eog_candidates:
        raw.set_channel_types({ch: "eog" for ch in eog_candidates})
    else:
        # Fallback for BCIC IV 2a: last 3 channels are EOG (22 EEG + 3 EOG = 25 total).
        if raw.info["nchan"] >= 25:
            assumed_eog = raw.ch_names[-3:]
            raw.set_channel_types({ch: "eog" for ch in assumed_eog})
            print(f"Warning: No 'EOG' channels found by name; assuming last 3 are EOG: {assumed_eog}")
    
    # Butterworth Bandpass filter (8-30 Hz)
    # isolates Mu & Beta rhythms relevant for motor imagery
    raw.filter(FILTER_LOW, FILTER_HIGH, method='iir',
               iir_params=dict(order=5, ftype='butter'))
    
    # ICA artifact removal
    ica = mne.preprocessing.ICA(n_components=ICA_N_COMPONENTS, random_state=RANDOM_SEED, max_iter='auto')
    ica.fit(raw)
    ica.exclude = []
    
    # Auto-detect EOG artifacts (use any available EOG channel)
    eog_picks = mne.pick_types(raw.info, eog=True)
    if len(eog_picks) > 0:
        eog_name = raw.ch_names[int(eog_picks[0])]
        eog_indices, _ = ica.find_bads_eog(raw, ch_name=eog_name)
        ica.exclude.extend(eog_indices)
    else:
        print("Warning: No EOG channels available for ICA EOG artifact detection.")
    
    print(f"Excluded {len(ica.exclude)} ICA components (eye/muscle artifacts)")
    
    ica.apply(raw)
    
    # Pick only EEG channels (drop EOG and other non-EEG)
    raw.pick_types(eeg=True, eog=False, stim=False, misc=False)
    print(f"Remaining channels: {raw.info['nchan']}, names: {raw.ch_names}")
    _save_channel_selection_visual(raw_before_pick, raw, subject)
    
    # Epoching around cues (769-772), 4.5s window
    events, event_id = mne.events_from_annotations(raw)
    
    epoch_event_id = {
        'Left Hand': event_id['769'],
        'Right Hand': event_id['770'],
        'Feet': event_id['771'],
        'Tongue': event_id['772']
    }
    
    epochs = mne.Epochs(
        raw, events, epoch_event_id,
        tmin=TMIN, tmax=TMAX,
        baseline=None,
        preload=True,
        reject={'eeg': 100e-6},
        event_repeated='drop'
    )
    
    print(f"Class distribution:\n{epochs}")
    
    # Extract X (n_trials, n_channels, n_times) and y (labels)
    X = epochs.get_data()
    y = epochs.events[:, 2]
    
    print(f"X shape: {X.shape}, y shape: {y.shape}, classes: {np.unique(y)}")
    
    # Save preprocessed data
    np.save(f'{OUTPUT_PATH}/X_{subject}.npy', X)
    np.save(f'{OUTPUT_PATH}/y_{subject}.npy', y)
    print("Saved X and y.")

if __name__ == '__main__':
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    for subject in tqdm(SUBJECTS_TRAIN, desc="Preprocessing (2a)", unit="subj"):
        try:
            preprocess_subject(subject)
        except Exception as e:
            print(f"Error processing {subject}: {e}")
    
    print("\nPreprocessing complete. Preprocessed data saved in 'analysis_results' folder.")
