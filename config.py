"""Configuration for the held-out-session motor-imagery BCI evaluation."""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "BCICIV_2a_gdf"
EPOCH_DIR = BASE_DIR / "analysis_results"
OUTPUT_DIR = BASE_DIR / "outputs" / "heldout_session"

SUBJECT_IDS = tuple(range(1, 10))
SUBJECTS_TRAIN = tuple(f"A{i:02d}T" for i in SUBJECT_IDS)
SUBJECTS_TEST = tuple(f"A{i:02d}E" for i in SUBJECT_IDS)

RANDOM_SEED = 42

FILTER_LOW = 8.0
FILTER_HIGH = 30.0
NOTCH_FREQ = 50.0
TMIN = 0.0
TMAX = 4.5
ICA_N_COMPONENTS = 20

SFREQ = 250.0
FBCSP_BANDS = ((8.0, 12.0), (12.0, 16.0), (16.0, 20.0), (20.0, 30.0))
CSP_N_COMPONENTS = 4
INCLUDE_BANDPOWER = True

BASE_MODELS = ("LDA", "SVM")
COMPARISON_MODEL = "LDA"
CALIBRATION_METHOD = "sigmoid"
CALIBRATION_CV = 3

OPERATING_COVERAGE = 0.60
THRESHOLD_GRID = tuple(round(x / 100, 2) for x in range(0, 101, 5))
