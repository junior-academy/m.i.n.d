
---

## 1) BCI Competition IV 2a (MAIN) — Full Pipeline

### 1.1 Preprocess — GDF → Epochs → `analysis_results/`

```bash
python3 m.i.n.d/mind_preprocess.py
```

**Expected outputs:**
- `m.i.n.d/analysis_results/X_A01T.npy` … `X_A09T.npy`
- `m.i.n.d/analysis_results/y_A01T.npy` … `y_A09T.npy`
- Plots under `m.i.n.d/analysis_results/plots/`

**Verify channel count** (expected **22 EEG channels**):

```bash
python3 - <<'PY'
import numpy as np
X = np.load("m.i.n.d/analysis_results/X_A01T.npy")
print("X_A01T shape:", X.shape)  # (n_trials, 22, n_timepoints)
PY
```

---

### 1.2 Baselines — CSP/FBCSP → LDA/SVM/RF → `m.i.n.d/outputs/*.csv`

**Fast / default:**
```bash
python3 m.i.n.d/features.py
```

**Publication-grade (recommended) — FBCSP + bandpower:**
```bash
python3 m.i.n.d/features.py --features fbcsp --include-bandpower
```

**Optional — nested CV tuning (slower):**
```bash
python3 m.i.n.d/features.py --features fbcsp --include-bandpower --tune small
```

**Key outputs:**

| File | Description |
|---|---|
| `m.i.n.d/outputs/classification_results.csv` | Per-subject `best_acc` |
| `m.i.n.d/outputs/classification_summary.csv` | Summary stats |
| `m.i.n.d/outputs/classification_summary_wide.csv` | Wide format |

---

### 1.3 Threshold Grid (all thresholds > 0.50, step 0.05)

```bash
THR_GRID="$(python3 - <<'PY'
print(",".join(f"{t:.2f}" for t in [0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00]))
PY
)"
echo "$THR_GRID"
```

---

### 1.4 Ensembles — `m.i.n.d/outputs/ensemble_v2/`

#### 1.4A — Main System: LDA + SVM ✅ recommended

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

#### 1.4B — Ablation: LDA + SVM + RF

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

#### 1.4C — Optional: Stacking Meta-Learner (slow)

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

**Each run folder contains:**
- `subject_results.csv`
- `threshold_metrics.csv`
- `run_summary.csv`
- Per-subject prediction CSVs like `predictions_subject_7.csv` *(if enabled in that preset/run)*

---

### 1.5 Stats Tests + Key Numbers + Keeper Visuals

**Paired t-test + Levene** (confident acc vs best-single, across thresholds):
```bash
python3 m.i.n.d/stats_tests.py
```

**Update key numbers** at the operating threshold (example `0.60`):
```bash
python3 m.i.n.d/update_key_numbers.py --threshold 0.60
```

**Generate keeper plots:**
```bash
python3 m.i.n.d/plot_ensembles.py
```

**Expected outputs:**

| File | Description |
|---|---|
| `m.i.n.d/outputs/ensemble_v2/stats_tests_confident_vs_best.csv` | Stat test results |
| `m.i.n.d/outputs/key_numbers/key_numbers_summary.csv` | Key metrics |
| `m.i.n.d/outputs/visuals/*.png` | Keeper plots |

---

### 1.6 Pygame Demo

> No CLI arguments needed — the app has an in-app menu.

```bash
python3 m.i.n.d/visualizer_pygame.py
```

**Controls:**

| Key | Action |
|---|---|
| `ENTER` | Start selected patient/run-set |
| `M` | Return to menu |
| `SPACE` | Play / Pause |
| `← / →` | Step trials |
| `[ / ]` | Threshold down / up *(hold `SHIFT` for bigger step)* |
| `TAB` | Next run *(when multiple runs are loaded)* |

---

## 2) BCI Competition IIIa (VALIDATION) — Full Pipeline

### 2.1 Preprocess IIIa — GDF → `analysis_results_3a/`

```bash
python3 m.i.n.d/mind_preprocess_3a.py
```

**Expected outputs:**
- `m.i.n.d/analysis_results_3a/X_*.npy`
- `m.i.n.d/analysis_results_3a/y_*.npy`

---

### 2.2 Baselines on IIIa

```bash
python3 m.i.n.d/features.py \
  --data-dir m.i.n.d/analysis_results_3a \
  --out-dir m.i.n.d/outputs/validation_3a/baselines \
  --features fbcsp \
  --include-bandpower
```

---

### 2.3 Ensemble on IIIa

> **Recommended:** avoid nested tuning on tiny n=3

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

---

### 2.4 IIIa Stats Tests (per-threshold)

Run for both `equal` and `baseline_subject` weight schemes:

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

> ⚠️ With **n=3 subjects**, p-values will often be non-significant. At very high thresholds, coverage can drop so low that p-values become `NaN`.

---

### 2.5 Generate IIIa Keeper Visuals

**Create dashboard-style `*_grid.csv` files** (renamed copies of `threshold_metrics.csv`):

```bash
cp m.i.n.d/outputs/validation_3a/ensemble_v2/models-LDA_SVM__weights-equal__feat-fbcsp__ens-softvote__tune-none__cal-sigmoid/threshold_metrics.csv \
  m.i.n.d/outputs/validation_3a/ensemble_v2/LDA_SVM_equal_grid.csv

cp m.i.n.d/outputs/validation_3a/ensemble_v2/models-LDA_SVM__weights-baseline_subject__feat-fbcsp__ens-softvote__tune-none__cal-sigmoid/threshold_metrics.csv \
  m.i.n.d/outputs/validation_3a/ensemble_v2/LDA_SVM_baseline_subject_grid.csv
```

**Combine stats into a single CSV** (used by the plotter):

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

**Generate the 7 keeper plots:**

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

## 3) External Validation via MOABB *(optional, recommended for generalization story)*

### 3.1 Install MOABB

```bash
python3 -m pip install moabb
python3 - <<'PY'
import moabb
print("moabb", moabb.__version__)
PY
```

---

### 3.2 PhysionetMI (2-class) — Baseline Pipelines (CSP+LDA vs CSP+SVM)

Results stored under `m.i.n.d/outputs/moabb/physionetmi/`.

> **Recommended:** start with 1 subject, then scale up.

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

**Then compute pipeline stats:**

```bash
python3 m.i.n.d/moabb_stats_tests.py \
  --results-per-subject m.i.n.d/outputs/moabb/physionetmi/results_per_subject.csv \
  --out m.i.n.d/outputs/moabb/physionetmi/stats_tests_pipelines.csv
```

---

### 3.3 MOABB Ensemble Evaluation — Compare Ensembles + Threshold Gating

Results written under `m.i.n.d/outputs/moabb_ensemble/<dataset>/...`

**PhysionetMI (2-class):**
```bash
bash m.i.n.d/scripts/run_moabb_ensemble.sh \
  --dataset physionetmi \
  --subjects 1,2,3,4,5,6,7,8,9,10 \
  --events left_hand,right_hand \
  --threshold-grid "$THR_GRID" \
  --out-dir m.i.n.d/outputs/moabb_ensemble
```

**BNCI2014_001 (4-class):**
```bash
bash m.i.n.d/scripts/run_moabb_ensemble.sh \
  --dataset bnci2014_001 \
  --subjects 1,2,3,4,5,6,7,8,9 \
  --threshold-grid "$THR_GRID" \
  --out-dir m.i.n.d/outputs/moabb_ensemble
```

---

## 4) Run Everything Overnight *(Mac-friendly)*

### 4.1 Keep the Mac Awake

> Safe to turn off monitors.

```bash
caffeinate -dimsu &
```

---

### 4.2 Full 2a + 3a Pipeline in One Background Job

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

**Monitor progress:**
```bash
tail -f overnight_run.log
```

---

## 5) Git / Storage Notes *(important)*

> ⚠️ **Do not commit datasets** (GDF/EDF) or large caches.

- GitHub blocks files **> 100 MB**
- Demo videos and large media should be stored externally or in the dashboard repo

---

## 6) Quick "Did It Work?" Checklist

**2a:**
```bash
ls -la m.i.n.d/analysis_results/X_A01T.npy
ls -la m.i.n.d/outputs/classification_results.csv
ls -la m.i.n.d/outputs/ensemble_v2/stats_tests_confident_vs_best.csv
ls -la m.i.n.d/outputs/visuals
```

**3a:**
```bash
ls -la m.i.n.d/analysis_results_3a
ls -la m.i.n.d/outputs/validation_3a/baselines/classification_results.csv
ls -la m.i.n.d/outputs/validation_3a/visuals
```
