import mne
import numpy as np
import os
from config import *

def preprocess_subject(subject):
    """Preprocess a single subject's data and save X, y arrays"""
    print(f"\nProcessing {subject}...")
    
    # Load raw data
    raw = mne.io.read_raw_gdf(f'{DATA_PATH}/{subject}.gdf', preload=True)

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
    for subject in SUBJECTS_TRAIN:
        try:
            preprocess_subject(subject)
        except Exception as e:
            print(f"Error processing {subject}: {e}")
    
    print("\nPreprocessing complete. Preprocessed data saved in 'analysis_results' folder.")
