import mne
import numpy as np
import os
from config import *

os.makedirs('analysis_results', exist_ok=True)

# loading data
raw = mne.io.read_raw_gdf('data/A09T.gdf', preload=True)
print(f"Loaded: {len(raw.ch_names)} channels @ {raw.info['sfreq']} Hz")
print("Channels:", raw.ch_names)

# 5-th order butterworth bandpass filter (8-30 Hz)
# isolates Mu & Beta rhythms relevant for motor imagery
raw.filter(8.0, 30.0, method='iir',
           iir_params=dict(order=5, ftype='butter'))

# ica artifact removal (20 components, auto-detect EOG)
ica = mne.preprocessing.ICA(n_components=ICA_N_COMPONENTS, random_state=42, max_iter='auto')
ica.fit(raw)
ica.exclude = []

# auto-detect eye blink (EOG) components
# electrooculography are unwanted artifacts from electrical signals from eye movements/blinks
eog_indices, _ = ica.find_bads_eog (raw, ch_name='EOG-left')
ica.exclude.extend(eog_indices)
print(f"Excluded {len(ica.exclude)} ICA components (eye/muscle artifacts)")

ica.apply(raw)

# Drop EOG channels, keep only 22 EEG channels
raw.pick_types(eeg=True, eog=False)

# Epoch around class cues (769–772), 4.5s window, no baseline correction
events, _ = mne.events_from_annotations(raw)

event_id = {'Left Hand': 769, 'Right Hand': 770, 'Feet': 771, 'Tongue': 772}

epochs = mne.Epochs(
    raw, events, event_id,
    tmin=0.0, tmax=4.5,
    baseline=None,
    preload=True,
    reject={'eeg': 100e-6}   # drop trials with extreme artifacts
)

print(f"\nClass distribution:")
print(epochs)

# extracting arrays for machine learning
X = epochs.get_data()       # (n_trials, 22, 1125)
y = epochs.events[:, 2]     # integer labels

print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Classes: {np.unique(y)}")

# each array is a voltage reading (in volts) of one of the 22 EEG electrodes at one point in time.
# this is after all of the data cleaning and preprocessing.
np.save('analysis_results/X_A09T.npy', X)

# these are the labels (769, 770, 771, 772) corresponding to the class of motor imagery for each trial.
# 769 = left hand, 770 = right hand, 771 = feet, 772 = tongue
np.save('analysis_results/y_A09T.npy', y)
print("Saved X and y.")