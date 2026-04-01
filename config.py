"""
Configuration parameters for motor imagery BCI analysis
"""
import numpy as np
import os

# ICA components:
ICA_N_COMPONENTS = 20

# File paths
DATA_PATH = 'data'
OUTPUT_PATH = 'analysis_results'
PLOTS_PATH = os.path.join(OUTPUT_PATH, 'plots')

# Dataset parameters
SUBJECTS_TRAIN = ['A01T', 'A02T', 'A03T', 'A04T', 'A05T', 'A06T', 'A07T', 'A08T', 'A09T']
SUBJECTS_TEST = ['A01E', 'A02E', 'A03E', 'A04E', 'A05E', 'A06E', 'A07E', 'A08E', 'A09E']

# Preprocessing parameters
FILTER_LOW = 8      # Hz - high-pass filter
FILTER_HIGH = 30    # Hz - low-pass filter
NOTCH_FREQ = 50     # Hz - power line noise (50 in Europe)

# Epoch parameters
TMIN = 0            # seconds - start time relative to cue
TMAX = 4.5          # seconds - end time relative to cue
CLASS_DESCRIPTIONS = ['769', '770', '771', '772']
CLASS_NAMES = {
    '769': 'Left Hand',
    '770': 'Right Hand',
    '771': 'Both Feet',
    '772': 'Tongue'
}

# Classification parameters
CSP_N_COMPONENTS = 4
CV_FOLDS = 5
RANDOM_SEED = 42

# Motor cortex channels
MOTOR_CHANNELS = ['EEG-C3', 'EEG-C4', 'EEG-Cz']

# Time-frequency parameters
TF_FREQS = np.arange(4, 30, 2)  # 4-30 Hz in 2 Hz steps
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(PLOTS_PATH, exist_ok=True)