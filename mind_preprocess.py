import mne
import numpy as np
import os
from config import *

def preprocess_subject(subject):
    """Preprocess a single subject's data and save X, y arrays"""
    print(f"\nProcessing {subject}...")
    
    # Load raw data
    raw = mne.io.read_raw_gdf(f'{DATA_PATH}/{subject}.gdf', preload=True)
    
    # Butterworth Bandpass filter (8-30 Hz)
    # isolates Mu & Beta rhythms relevant for motor imagery
    raw.filter(FILTER_LOW, FILTER_HIGH, method='iir',
               iir_params=dict(order=5, ftype='butter'))
    
    # ICA artifact removal
    ica = mne.preprocessing.ICA(n_components=ICA_N_COMPONENTS, random_state=RANDOM_SEED, max_iter='auto')
    ica.fit(raw)
    ica.exclude = []
    
    # Auto-detect EOG artifacts
    eog_indices, _ = ica.find_bads_eog(raw, ch_name='EOG-left')
    ica.exclude.extend(eog_indices)
    
    print(f"Excluded {len(ica.exclude)} ICA components (eye/muscle artifacts)")
    
    ica.apply(raw)
    
    # Pick only EEG channels
    raw.pick(['eeg'])
    
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