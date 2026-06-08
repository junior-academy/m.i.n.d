"""Configuration defaults for M.I.N.D. v2 experiments."""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
EPOCH_DIR = BASE_DIR / "analysis_results"
OUTPUT_DIR = BASE_DIR / "outputs" / "v2"
MOABB_DATA_DIR = BASE_DIR / "data" / "moabb"

SUBJECT_IDS = tuple(range(1, 10))
RANDOM_SEED = 42
ALIGNMENT_EPS = 1e-6

SFREQ = 250.0
DEFAULT_COVERAGE = 0.60
COVERAGE_POINTS = (0.40, 0.60, 0.80)
CURVE_COVERAGES = tuple(round(x / 100, 2) for x in range(5, 101, 5))

DEEP_EPOCHS = 60
DEEP_BATCH_SIZE = 32
DEEP_LR = 1e-3
VALIDATION_SIZE = 0.20

