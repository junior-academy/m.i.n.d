"""
Main analysis script for motor imagery BCI data
BCI Competition 2008 - Graz dataset A
"""
def main():
    import mne
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    from datetime import datetime

    # Create output directories
    output_dir = 'analysis_results'
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    print("=" * 60)
    print("Motor Imagery BCI Analysis")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Load your data
    file_path = 'data/A09T.gdf'
    raw = mne.io.read_raw_gdf(file_path, preload=True)

    print(f"\nLoaded file: {file_path}")
    print(f"Duration: {raw.times[-1]:.2f} seconds")
    print(f"Sampling rate: {raw.info['sfreq']} Hz")
    print(f"Channels: {len(raw.ch_names)}")

    # Apply preprocessing
    print("\nApplying preprocessing filters...")
    raw.filter(7, 35, fir_design='firwin')
    raw.notch_filter(60)

    # Get events and mapping
    events, event_id = mne.events_from_annotations(raw)

    # Event mapping
    stimulus_desc = '768'
    stimulus_code = event_id[stimulus_desc]
    class_descriptions = ['769', '770', '771', '772']
    class_names = {
        '769': 'Left Hand',
        '770': 'Right Hand',
        '771': 'Both Feet',
        '772': 'Tongue'
    }
    class_codes = {desc: event_id[desc] for desc in class_descriptions}

    print(f"\nEvent mapping:")
    for desc, code in event_id.items():
        if desc in class_descriptions:
            print(f"  {desc} -> {code} ({class_names[desc]})")
        else:
            print(f"  {desc} -> {code}")

    # Extract stimulus events
    stimulus_events = events[events[:, 2] == stimulus_code]
    print(f"\nFound {len(stimulus_events)} stimulus events")

    # Match stimuli with class events (occur 2 seconds later)
    sample_rate = raw.info['sfreq']
    two_seconds_samples = int(2 * sample_rate)

    epoch_class = []
    matched_stimuli = []

    for stim_event in stimulus_events:
        stim_time = stim_event[0]
        expected_class_time = stim_time + two_seconds_samples
        
        found_class = None
        for desc, code in class_codes.items():
            matching = events[(events[:, 0] == expected_class_time) & (events[:, 2] == code)]
            if len(matching) > 0:
                found_class = desc
                break
        
        if found_class:
            epoch_class.append(found_class)
            matched_stimuli.append(stim_event)
        else:
            epoch_class.append('unknown')

    # Filter valid trials
    valid_indices = [i for i, cls in enumerate(epoch_class) if cls != 'unknown']
    matched_stimuli = [matched_stimuli[i] for i in valid_indices]
    epoch_class = [epoch_class[i] for i in valid_indices]

    print(f"Valid trials: {len(epoch_class)}")
    print("\nClass distribution:")
    for desc in class_descriptions:
        count = epoch_class.count(desc)
        print(f"  {class_names[desc]}: {count} trials")

    # Create epochs (0-4 seconds after cue)
    tmin, tmax = 0, 4
    epochs = mne.Epochs(raw, np.array(matched_stimuli), tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, event_repeated='merge')

    print(f"\nEpochs shape: {epochs.get_data().shape}")

    # Add class labels to metadata
    epochs.metadata = pd.DataFrame({'class': epoch_class}, index=epochs.events[:, 0])
    epochs.metadata['class_name'] = epochs.metadata['class'].map(class_names)