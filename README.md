# JR-Academy 6857 — M.I.N.D Runbook (Complete)

This repo contains:

- `m.i.n.d/` — the full EEG pipeline (BCI IV 2a main experiments + validation + MOABB tooling + Pygame demo)
- `mind-dashboard/` — a Next.js dashboard that renders results from `m.i.n.d/outputs/**`

This README is a **command-by-command runbook**. It’s intentionally exhaustive.

---

## 0) One-time setup

### 0.1 Python environment (recommended)

Use Python **3.10+** (3.11 recommended). If you use `conda`, activate your env first.

```bash
python3 -V
python3 -m pip install --upgrade pip
python3 -m pip install -r m.i.n.d/requirements.txt
```

Sanity check:

```bash
python3 - <<'PY'
import numpy, pandas, sklearn, mne
print("ok")
PY
```

Optional (avoids Matplotlib cache permission warnings on some machines):

```bash
export MPLCONFIGDIR="$(pwd)/.mplcache"
mkdir -p "$MPLCONFIGDIR"
```

### 0.2 Node / dashboard (optional)

```bash
cd mind-dashboard
npm install
```

### 0.3 Data locations (expected)

BCI Competition IV 2a (GDF):

- `m.i.n.d/data/BCICIV_2a_gdf/A01T.gdf` … `A09T.gdf`

BCI Competition IIIa (GDF):

- `m.i.n.d/data/BCICIV_3a_gdf/*.gdf`

PhysioNet EEG Motor Imagery (MOABB/MNE cache; EDF):

- recommended cache root: `m.i.n.d/data/mne_data/`

---

## 1) BCI Competition IV 2a (MAIN) — full pipeline

### 1.1 Preprocess (GDF → epochs → `analysis_results/`)

```bash
python3 m.i.n.d/mind_preprocess.py
```

Expected outputs:

- `m.i.n.d/analysis_results/X_A01T.npy` … `X_A09T.npy`
- `m.i.n.d/analysis_results/y_A01T.npy` … `y_A09T.npy`
- plots under `m.i.n.d/analysis_results/plots/`

Verify channel count (expected **22 EEG channels**):

```bash
python3 - <<'PY'
import numpy as np
X = np.load("m.i.n.d/analysis_results/X_A01T.npy")
print("X_A01T shape:", X.shape)  # (n_trials, 22, n_timepoints)
PY
```

### 1.2 Baselines (CSP/FBCSP → LDA/SVM/RF) → `m.i.n.d/outputs/*.csv`

Fast/default:

```bash
python3 m.i.n.d/features.py
```

Publication-grade (recommended): **FBCSP + bandpower**

```bash
python3 m.i.n.d/features.py --features fbcsp --include-bandpower
```

Optional: nested CV tuning (slower):

```bash
python3 m.i.n.d/features.py --features fbcsp --include-bandpower --tune small
```

Key outputs:

- `m.i.n.d/outputs/classification_results.csv` (per-subject `best_acc`)
- `m.i.n.d/outputs/classification_summary.csv`
- `m.i.n.d/outputs/classification_summary_wide.csv`

### 1.3 Threshold grid (all thresholds > 0.50, step 0.05)

```bash
THR_GRID="$(python3 - <<'PY'
print(",".join(f"{t:.2f}" for t in [0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00]))
PY
)"
echo "$THR_GRID"
```

### 1.4 Ensembles (writes run folders under `m.i.n.d/outputs/ensemble_v2/`)

#### 1.4A Main system: LDA + SVM (recommended)

```bash
python3 m.i.n.d/ensemble_v2.py \
  --preset main \
  --tune small \
  --features fbcsp \
  --include-bandpower \
  --calibrate sigmoid \
  --threshold 0.60 \
  --threshold-grid "$THR_GRID"
```

#### 1.4B Ablation: LDA + SVM + RF

```bash
python3 m.i.n.d/ensemble_v2.py \
  --preset ablation_rf \
  --tune small \
  --features fbcsp \
  --include-bandpower \
  --calibrate sigmoid \
  --threshold 0.60 \
  --threshold-grid "$THR_GRID"
```

#### 1.4C Optional: stacking meta-learner (slow)

```bash
python3 m.i.n.d/ensemble_v2.py \
  --preset main \
  --ensemble-method stacking \
  --tune small \
  --features fbcsp \
  --include-bandpower \
  --calibrate sigmoid \
  --threshold 0.60 \
  --threshold-grid "$THR_GRID"
```

Each run folder contains:

- `subject_results.csv`
- `threshold_metrics.csv`
- `run_summary.csv`
- per-subject prediction CSVs like `predictions_subject_7.csv` (if enabled in that preset/run)

### 1.5 Stats tests + key numbers + keeper visuals

Paired t-test + Levene (confident acc vs best-single, across thresholds):

```bash
python3 m.i.n.d/stats_tests.py
```

Update “key numbers” at the operating threshold (example `0.60`):

```bash
python3 m.i.n.d/update_key_numbers.py --threshold 0.60
```

Generate keeper plots:

```bash
python3 m.i.n.d/plot_ensembles.py
```

Expected outputs:

- `m.i.n.d/outputs/ensemble_v2/stats_tests_confident_vs_best.csv`
- `m.i.n.d/outputs/key_numbers/key_numbers_summary.csv`
- `m.i.n.d/outputs/visuals/*.png`

### 1.6 Pygame demo (no CLI needed; it has an in-app menu)

```bash
python3 m.i.n.d/visualizer_pygame.py
```

Controls (in-app):

- `ENTER` start selected patient/run-set
- `M` return to menu
- `SPACE` play/pause
- `←/→` step trials
- `[` / `]` threshold down/up (hold SHIFT for bigger step)
- `TAB` next run (when multiple runs are loaded)

---

## 2) BCI Competition IIIa (VALIDATION) — full pipeline

### 2.1 Preprocess IIIa (GDF → `analysis_results_3a/`)

```bash
python3 m.i.n.d/mind_preprocess_3a.py
```

Expected outputs:

- `m.i.n.d/analysis_results_3a/X_*.npy`
- `m.i.n.d/analysis_results_3a/y_*.npy`

### 2.2 Baselines on IIIa

```bash
python3 m.i.n.d/features.py \
  --data-dir m.i.n.d/analysis_results_3a \
  --out-dir m.i.n.d/outputs/validation_3a/baselines \
  --features fbcsp \
  --include-bandpower
```

### 2.3 Ensemble on IIIa (recommended: avoid nested tuning on tiny n=3)

```bash
python3 m.i.n.d/ensemble_v2.py \
  --data-dir m.i.n.d/analysis_results_3a \
  --out-dir m.i.n.d/outputs/validation_3a/ensemble_v2 \
  --preset main \
  --weights equal \
  --tune none \
  --features fbcsp \
  --include-bandpower \
  --calibrate sigmoid \
  --threshold 0.60 \
  --threshold-grid "$THR_GRID"
```

### 2.4 IIIa stats tests (per-threshold) for each run (equal + baseline_subject)

```bash
python3 m.i.n.d/stats_tests_any.py \
  --baseline m.i.n.d/outputs/validation_3a/baselines/classification_results.csv \
  --threshold-metrics m.i.n.d/outputs/validation_3a/ensemble_v2/models-LDA_SVM__weights-equal__feat-fbcsp__ens-softvote__tune-none__cal-sigmoid/threshold_metrics.csv \
  --out m.i.n.d/outputs/validation_3a/ensemble_v2/stats_tests_confident_vs_best__equal.csv \
  --label "BCI IIIa: LDA+SVM (equal)"

python3 m.i.n.d/stats_tests_any.py \
  --baseline m.i.n.d/outputs/validation_3a/baselines/classification_results.csv \
  --threshold-metrics m.i.n.d/outputs/validation_3a/ensemble_v2/models-LDA_SVM__weights-baseline_subject__feat-fbcsp__ens-softvote__tune-none__cal-sigmoid/threshold_metrics.csv \
  --out m.i.n.d/outputs/validation_3a/ensemble_v2/stats_tests_confident_vs_best__baseline_subject.csv \
  --label "BCI IIIa: LDA+SVM (subj-weights)"
```

Note: with **n=3 subjects**, p-values will often be non-significant; at very high thresholds, coverage can drop so low that p-values can become `NaN`.

### 2.5 Generate IIIa keeper visuals (separate from 2a)

Create dashboard-style `*_grid.csv` files (these are just renamed copies of `threshold_metrics.csv`):

```bash
cp m.i.n.d/outputs/validation_3a/ensemble_v2/models-LDA_SVM__weights-equal__feat-fbcsp__ens-softvote__tune-none__cal-sigmoid/threshold_metrics.csv \
  m.i.n.d/outputs/validation_3a/ensemble_v2/LDA_SVM_equal_grid.csv

cp m.i.n.d/outputs/validation_3a/ensemble_v2/models-LDA_SVM__weights-baseline_subject__feat-fbcsp__ens-softvote__tune-none__cal-sigmoid/threshold_metrics.csv \
  m.i.n.d/outputs/validation_3a/ensemble_v2/LDA_SVM_baseline_subject_grid.csv
```

Combine stats into a single CSV (used by the plotter):

```bash
python3 - <<'PY'
import pandas as pd
from pathlib import Path
base = Path("m.i.n.d/outputs/validation_3a/ensemble_v2")
out = base / "stats_tests_confident_vs_best.csv"
df = pd.concat([
    pd.read_csv(base / "stats_tests_confident_vs_best__equal.csv"),
    pd.read_csv(base / "stats_tests_confident_vs_best__baseline_subject.csv"),
], ignore_index=True)
df.to_csv(out, index=False)
print("Wrote", out)
PY
```

Generate the 7 keeper plots:

```bash
python3 m.i.n.d/plot_ensembles_any.py \
  --baseline-results m.i.n.d/outputs/validation_3a/baselines/classification_results.csv \
  --grid-dir m.i.n.d/outputs/validation_3a/ensemble_v2 \
  --stats-tests m.i.n.d/outputs/validation_3a/ensemble_v2/stats_tests_confident_vs_best.csv \
  --out-dir m.i.n.d/outputs/validation_3a/visuals \
  --threshold 0.60 \
  --heatmap-grid LDA_SVM_equal_grid.csv
```

---

## 3) Dashboard (Next.js) — local run + syncing artifacts

### 3.1 Put demo video in place

```bash
ls -la mind-dashboard/public/demo/demo.mp4
```

### 3.2 Sync latest artifacts into `mind-dashboard/public/**`

From repo root:

```bash
cd mind-dashboard
node scripts/sync_data.mjs
```

What this copies:

- 2a CSVs → `mind-dashboard/public/mind_data/`
- 3a CSVs → `mind-dashboard/public/mind_data/validation_3a/`
- 2a visuals → `mind-dashboard/public/visuals/`
- 3a visuals → `mind-dashboard/public/visuals/validation_3a/`

### 3.3 Run the dashboard

```bash
cd mind-dashboard
npm run dev
```

Production build check:

```bash
cd mind-dashboard
npm run build
```

---

## 4) External validation via MOABB (optional, recommended for generalization story)

### 4.1 Install MOABB (if missing)

```bash
python3 -m pip install moabb
python3 - <<'PY'
import moabb
print("moabb", moabb.__version__)
PY
```

### 4.2 MOABB PhysionetMI (2-class) — baseline pipelines (CSP+LDA vs CSP+SVM)

This uses a robust download wrapper and stores results under:

- `m.i.n.d/outputs/moabb/physionetmi/`

Recommended: start small (1 subject), then scale up.

```bash
bash m.i.n.d/scripts/run_moabb_physionet.sh \
  --dataset physionetmi \
  --subjects 1 \
  --n-classes 2 \
  --events left_hand,right_hand \
  --scoring accuracy \
  --n-jobs 1 \
  --mne-data m.i.n.d/data/mne_data \
  --download-timeout 600 \
  --download-retries 10
```

Then compute pipeline stats:

```bash
python3 m.i.n.d/moabb_stats_tests.py \
  --results-per-subject m.i.n.d/outputs/moabb/physionetmi/results_per_subject.csv \
  --out m.i.n.d/outputs/moabb/physionetmi/stats_tests_pipelines.csv
```

### 4.3 MOABB ensemble evaluation (Option 1): compare ensembles + threshold gating

This writes under:

- `m.i.n.d/outputs/moabb_ensemble/<dataset>/...`

PhysionetMI (2-class):

```bash
bash m.i.n.d/scripts/run_moabb_ensemble.sh \
  --dataset physionetmi \
  --subjects 1,2,3,4,5,6,7,8,9,10 \
  --events left_hand,right_hand \
  --threshold-grid "$THR_GRID" \
  --out-dir m.i.n.d/outputs/moabb_ensemble
```

BNCI2014_001 (4-class):

```bash
bash m.i.n.d/scripts/run_moabb_ensemble.sh \
  --dataset bnci2014_001 \
  --subjects 1,2,3,4,5,6,7,8,9 \
  --threshold-grid "$THR_GRID" \
  --out-dir m.i.n.d/outputs/moabb_ensemble
```

Then sync the dashboard artifacts:

```bash
cd mind-dashboard
node scripts/sync_data.mjs
npm run dev
```

---

## 5) “Run everything overnight” (Mac-friendly)

### 5.1 Keep the Mac awake (safe to turn off monitors)

```bash
caffeinate -dimsu &
```

### 5.2 Full 2a + 3a pipeline in one background job

From repo root:

```bash
nohup bash -lc '
set -euo pipefail

export MPLCONFIGDIR="$(pwd)/.mplcache"
mkdir -p "$MPLCONFIGDIR"

THR_GRID="$(python3 - <<PY
print(",".join(f"{t:.2f}" for t in [0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00]))
PY
)"

echo "[1/9] 2a preprocess"
python3 m.i.n.d/mind_preprocess.py

echo "[2/9] 2a baselines (FBCSP+bandpower)"
python3 m.i.n.d/features.py --features fbcsp --include-bandpower

echo "[3/9] 2a main ensemble"
python3 m.i.n.d/ensemble_v2.py --preset main --tune small --features fbcsp --include-bandpower --calibrate sigmoid --threshold 0.60 --threshold-grid "$THR_GRID"

echo "[4/9] 2a ablation ensemble"
python3 m.i.n.d/ensemble_v2.py --preset ablation_rf --tune small --features fbcsp --include-bandpower --calibrate sigmoid --threshold 0.60 --threshold-grid "$THR_GRID"

echo "[5/9] 2a stats + key numbers + visuals"
python3 m.i.n.d/stats_tests.py
python3 m.i.n.d/update_key_numbers.py --threshold 0.60
python3 m.i.n.d/plot_ensembles.py

echo "[6/9] 3a preprocess"
python3 m.i.n.d/mind_preprocess_3a.py

echo "[7/9] 3a baselines"
python3 m.i.n.d/features.py --data-dir m.i.n.d/analysis_results_3a --out-dir m.i.n.d/outputs/validation_3a/baselines --features fbcsp --include-bandpower

echo "[8/9] 3a main ensemble (equal weights, no tuning)"
python3 m.i.n.d/ensemble_v2.py --data-dir m.i.n.d/analysis_results_3a --out-dir m.i.n.d/outputs/validation_3a/ensemble_v2 --preset main --weights equal --tune none --features fbcsp --include-bandpower --calibrate sigmoid --threshold 0.60 --threshold-grid "$THR_GRID"

echo "[9/9] 3a stats (equal + subj-weights)"
python3 m.i.n.d/stats_tests_any.py --baseline m.i.n.d/outputs/validation_3a/baselines/classification_results.csv --threshold-metrics m.i.n.d/outputs/validation_3a/ensemble_v2/models-LDA_SVM__weights-equal__feat-fbcsp__ens-softvote__tune-none__cal-sigmoid/threshold_metrics.csv --out m.i.n.d/outputs/validation_3a/ensemble_v2/stats_tests_confident_vs_best__equal.csv --label "BCI IIIa: LDA+SVM (equal)"
python3 m.i.n.d/stats_tests_any.py --baseline m.i.n.d/outputs/validation_3a/baselines/classification_results.csv --threshold-metrics m.i.n.d/outputs/validation_3a/ensemble_v2/models-LDA_SVM__weights-baseline_subject__feat-fbcsp__ens-softvote__tune-none__cal-sigmoid/threshold_metrics.csv --out m.i.n.d/outputs/validation_3a/ensemble_v2/stats_tests_confident_vs_best__baseline_subject.csv --label "BCI IIIa: LDA+SVM (subj-weights)"

echo "[post] 3a visuals"
cp m.i.n.d/outputs/validation_3a/ensemble_v2/models-LDA_SVM__weights-equal__feat-fbcsp__ens-softvote__tune-none__cal-sigmoid/threshold_metrics.csv m.i.n.d/outputs/validation_3a/ensemble_v2/LDA_SVM_equal_grid.csv
cp m.i.n.d/outputs/validation_3a/ensemble_v2/models-LDA_SVM__weights-baseline_subject__feat-fbcsp__ens-softvote__tune-none__cal-sigmoid/threshold_metrics.csv m.i.n.d/outputs/validation_3a/ensemble_v2/LDA_SVM_baseline_subject_grid.csv
python3 - <<PY
import pandas as pd
from pathlib import Path
base = Path("m.i.n.d/outputs/validation_3a/ensemble_v2")
out = base / "stats_tests_confident_vs_best.csv"
df = pd.concat([
    pd.read_csv(base / "stats_tests_confident_vs_best__equal.csv"),
    pd.read_csv(base / "stats_tests_confident_vs_best__baseline_subject.csv"),
], ignore_index=True)
df.to_csv(out, index=False)
print("Wrote", out)
PY
python3 m.i.n.d/plot_ensembles_any.py --baseline-results m.i.n.d/outputs/validation_3a/baselines/classification_results.csv --grid-dir m.i.n.d/outputs/validation_3a/ensemble_v2 --stats-tests m.i.n.d/outputs/validation_3a/ensemble_v2/stats_tests_confident_vs_best.csv --out-dir m.i.n.d/outputs/validation_3a/visuals --threshold 0.60 --heatmap-grid LDA_SVM_equal_grid.csv

echo "[done]"
' > overnight_run.log 2>&1 &
```

Monitor:

```bash
tail -f overnight_run.log
```

---

## 6) Git / storage notes (important)

- **Do not commit datasets** (GDF/EDF) or large caches.
- GitHub blocks files **>100MB**. Videos like `mind-dashboard/public/demo/demo.mp4` at **1.5MB** are fine.

---

## 7) Quick “did it work?” checklist

2a:

```bash
ls -la m.i.n.d/analysis_results/X_A01T.npy
ls -la m.i.n.d/outputs/classification_results.csv
ls -la m.i.n.d/outputs/ensemble_v2/stats_tests_confident_vs_best.csv
ls -la m.i.n.d/outputs/visuals
```

3a:

```bash
ls -la m.i.n.d/analysis_results_3a
ls -la m.i.n.d/outputs/validation_3a/baselines/classification_results.csv
ls -la m.i.n.d/outputs/validation_3a/visuals
```

dashboard:

```bash
cd mind-dashboard
node scripts/sync_data.mjs
npm run dev
```
