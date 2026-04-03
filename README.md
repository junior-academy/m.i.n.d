# M.I.N.D - Mental Interpretation Network for Decision Making

This folder contains the end-to-end analysis pipeline for Project **M.I.N.D.** on **BCI Competition IV Dataset 2a**:

- `mind_preprocess.py`: GDF → filtered/ICA-cleaned epochs → `X_*.npy` / `y_*.npy`
- `features.py`: baseline CSP+{LDA,SVM,RF} evaluation → baseline CSVs
- `ensemble_v2.py`: ensemble experiments (LDA+SVM main, RF ablation, calibration, threshold sweeps)
- `plot_ensembles.py`: generates the core figures used in writeups/slides
- `update_key_numbers.py`: regenerates the “Key Numbers to Track” tables from current outputs

All generated artifacts go to:
- `analysis_results/` (NumPy arrays + preprocessing plots)
- `outputs/` (CSVs + visuals)

## Quickstart

From the repo root:

### 0) Create an environment + install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# IMPORTANT: pin numpy<2 for GDF loading reliability
python -m pip install "numpy<2"
python -m pip install -r m.i.n.d/requirements.txt
```

Optional (avoids Matplotlib cache warnings on some machines):

```bash
export MPLCONFIGDIR=/tmp/mplconfig
```

### 1) Preprocess (GDF → X/y)

Make sure the dataset is present at:
- `m.i.n.d/data/BCICIV_2a_gdf/A01T.gdf` … `A09T.gdf`

Run preprocessing for all training subjects:

```bash
python3 m.i.n.d/mind_preprocess.py
```

Verify the saved arrays have **22 EEG channels**:

```bash
python3 - <<'PY'
import numpy as np
X = np.load('m.i.n.d/analysis_results/X_A01T.npy')
print(X.shape)  # expected: (288, 22, n_timepoints)
PY
```

Preprocessing also saves a channel-selection report image per subject to:
- `m.i.n.d/analysis_results/plots/channel_selection_<subject>.png`

### 2) Run baselines (CSP + LDA/SVM/RF)

```bash
python3 m.i.n.d/features.py
```

Outputs:
- `m.i.n.d/outputs/classification_results.csv` (per-subject)
- `m.i.n.d/outputs/classification_summary.csv` (long-form: LDA/SVM/RF/Best-Single rows)
- `m.i.n.d/outputs/classification_summary_wide.csv` (back-compat)

### 3) Run ensemble experiments (v2)

The “main” system is **LDA+SVM**. The **RF** variant is treated as an ablation.

Example: run the main system with calibration and a threshold grid including `0.75`:

```bash
python3 m.i.n.d/ensemble_v2.py \
  --preset main \
  --weights baseline_subject \
  --calibrate sigmoid \
  --threshold 0.75 \
  --threshold-grid 0.0,0.55,0.60,0.65,0.70,0.75
```

Example: run the RF ablation:

```bash
python3 m.i.n.d/ensemble_v2.py \
  --preset ablation_rf \
  --weights baseline_global \
  --calibrate sigmoid \
  --threshold 0.75 \
  --threshold-grid 0.0,0.55,0.60,0.65,0.70,0.75
```

Each run writes a new folder under:
- `m.i.n.d/outputs/ensemble_v2/models-.../`

including:
- `subject_results.csv`
- `threshold_metrics.csv`
- `run_summary.csv` (means/95% CI + paired tests vs Best-Single at the operating threshold)

### 4) Export grid CSVs + run stats tests (for plotting)

`plot_ensembles.py` reads the three grid CSVs:
- `m.i.n.d/outputs/ensemble_v2/LDA_SVM_equal_grid.csv`
- `m.i.n.d/outputs/ensemble_v2/LDA_SVM_baseline_subject_grid.csv`
- `m.i.n.d/outputs/ensemble_v2/LDA_SVM_RF_global_grid.csv`

If they’re missing, `plot_ensembles.py` will fall back to per-run `threshold_metrics.csv` and write them.

To compute paired t-tests and Levene’s test per threshold (confident accuracy vs Best‑Single):

```bash
python3 m.i.n.d/stats_tests.py
```

Outputs:
- `m.i.n.d/outputs/ensemble_v2/stats_tests_confident_vs_best.csv`

### 5) Update Key Numbers

Pick an operating threshold (example `0.65`) and regenerate key tables:

```bash
python3 m.i.n.d/update_key_numbers.py --threshold 0.65
```

Outputs:
- `m.i.n.d/outputs/key_numbers/key_numbers_per_subject.csv`
- `m.i.n.d/outputs/key_numbers/key_numbers_summary.csv`
- `m.i.n.d/outputs/key_numbers/key_numbers_claim_bank.csv`

### 6) Regenerate visuals

```bash
python3 m.i.n.d/plot_ensembles.py
```

This script generates the “keep” set of figures into:
- `m.i.n.d/outputs/visuals/`

## Cleaning old outputs

To reduce clutter without deleting data:

```bash
python3 m.i.n.d/clean_outputs.py       # dry-run
python3 m.i.n.d/clean_outputs.py --apply
```

## Troubleshooting

### `OverflowError: Python integer 256 out of bounds for uint8` when reading `.gdf`
This is typically resolved by pinning NumPy to `<2` in your environment:

```bash
python -m pip install "numpy<2"
```

### Matplotlib cache warnings
Set a writable cache directory:

```bash
export MPLCONFIGDIR=/tmp/mplconfig
```

## Dataset Information
This analysis uses the **BCI Competition 2008 – Graz data set A** (also known as BCI Competition IV dataset 2a).

### Dataset Details
- **Subjects**: 9 subjects (A01T - A09T for training, A01E - A09E for evaluation)
- **Tasks**: 4-class motor imagery (Left hand, Right hand, Both feet, Tongue)
- **Channels**: 22 EEG channels + 3 EOG channels
- **Sampling rate**: 250 Hz
- **Trials**: 288 per subject (72 per class)
- **Runs**: 6 runs with short breaks

### Experimental Protocol
Each trial follows this timeline:
- **t = 0s**: Fixation cross + acoustic warning
- **t = 2s**: Visual cue appears (arrow direction)
- **t = 3.25s**: Cue disappears
- **t = 6s**: Trial ends

### File Naming Convention
- **AXXT.gdf**: Training data (use for classifier development)
- **AXXE.gdf**: Evaluation data (use for testing)

## How to Download Other Subjects

### Option 1: Download from BCI Competition Website
Visit: http://www.bbci.de/competition/iv/
- Download "Dataset 2a" (Graz dataset A)
- Files: A01T.gdf through A09T.gdf (training)
- Files: A01E.gdf through A09E.gdf (evaluation)

### Option 2: Download via Python (check `download_data.py`)

### Option 3: Manual Download
1. Go to: http://www.bbci.de/competition/iv/
2. Register for a free account (required for dataset access)
3. Download "Dataset 2a" (approximately 2.5 GB total)
4. Extract all .gdf files to the same folder
