# M.I.N.D Held-Out Session Evaluation

This repo now answers one question:

> When a motor-imagery BCI is allowed to abstain on low-confidence trials, does a calibrated LDA+SVM ensemble stay more reliable than a single model when tested on a different recording session?

## Main Claim

The headline evaluation is subject-paired and session-held-out:

- Fit preprocessing outputs, FBCSP feature extractors, probability calibration, LDA, and SVM on `A0xT`.
- Apply label-free Euclidean alignment to reduce session covariance shift. `A0xE` labels are not used for alignment.
- Score exactly once on the paired `A0xE` session.
- Compare the equal-weight LDA+SVM soft vote against pre-specified LDA.
- Compare both decoders at matched coverage, not ensemble-confident trials against single-model all-trials.
- Report per-subject deltas and a Wilcoxon signed-rank test. With `n = 9`, inference is explicitly limited.

## Reproduce

Place BCI Competition IV 2a GDF files in `data/BCICIV_2a_gdf/`, then run:

```bash
python download_true_labels.py
./run_all.sh
```

The `A0xE.gdf` files contain unknown cues (`783`) rather than class labels. The true labels are published separately on the BCI Competition IV results page, and `download_true_labels.py` normalizes them into `data/BCICIV_2a_gdf/true_labels/A01E.csv` through `A09E.csv`.

Outputs are written to `outputs/heldout_session/`:

- `subject_results.csv`: one row per subject at the pre-registered coverage.
- `risk_coverage.csv`: threshold sweep for ensemble and LDA, used descriptively.
- `reliability_metrics.csv`: accuracy, mean confidence, ECE, and Brier score.
- `reliability_diagram_bins.csv`: top-label confidence-bin data for reliability diagrams.
- `reliability_diagrams/*.png`: plotted reliability diagrams when matplotlib is installed.
- `headline_stats.csv`: Wilcoxon summary.
- `per_subject_deltas.csv`: paired deltas used by the test.

The paper-style writeup is in `writeup.tex`. To run the older cold-transfer baseline for comparison, use:

```bash
python selective_eval.py --adaptation none
python stats.py \
  --subject-results outputs/heldout_session_no_adaptation/subject_results.csv \
  --out outputs/heldout_session_no_adaptation/headline_stats.csv
```

To add EEGNet as a modern deep-learning comparison, install PyTorch and pass
the optional flag:

```bash
python -m pip install -r requirements-eegnet.txt
python selective_eval.py --include-eegnet --eegnet-epochs 60
```

## MOABB External Validation

Run a small two-dataset external check:

```bash
python moabb_eval.py \
  --datasets Cho2017 PhysionetMI \
  --subjects-per-dataset 1 \
  --out-dir outputs/moabb_external
```

Run a broader validation once the first-download cache is populated:

```bash
python moabb_eval.py \
  --datasets Cho2017 PhysionetMI \
  --subjects-per-dataset 5 \
  --out-dir outputs/moabb_external_5subj
```

To include EEGNet in MOABB outputs:

```bash
python moabb_eval.py \
  --datasets Cho2017 PhysionetMI \
  --subjects-per-dataset 5 \
  --include-eegnet \
  --eegnet-epochs 60 \
  --out-dir outputs/moabb_external_5subj_eegnet
```

MOABB outputs mirror the held-out run:

- `summary_by_dataset.csv`: per-dataset mean deltas at matched coverage.
- `subject_results.csv`: subject-level matched-coverage accuracy, ECE, and Brier deltas.
- `risk_coverage.csv`: threshold sweep for selective prediction.
- `reliability_metrics.csv`: all-trial accuracy, confidence, ECE, and Brier.
- `reliability_diagram_bins.csv` and `reliability_diagrams/*.png`: reliability diagrams.

The current smoke subset in `outputs/moabb_external/` uses one subject each
from `Cho2017` and `PhysionetMI`. In that subset, matched-coverage accuracy and
Brier score improve for the ensemble on both datasets, while ECE is worse. That
is reported as a calibration limitation, not hidden: the average of calibrated
base decoders can still need ensemble-level recalibration.

## What Was Removed

The old within-session CV path, Random Forest main-model runs, stacking meta-learner, baseline-weighted ensembles, Pygame demo, dashboard assets, threshold-by-threshold significance sweep, committed NumPy arrays, large generated PNGs, caches, and dated backups were removed from the research path.

Those pieces either did not feed the held-out-session claim or introduced avoidable methodological risk:

- RF remains outside the main claim as an optional future ablation.
- Stacking was cut because it overfits at this sample size.
- Baseline-weighted ensembles were cut because CV-accuracy-derived weights leak model selection information into the evaluation.
- Threshold significance sweeps were replaced by one operating coverage plus a descriptive risk-coverage curve.
- The local CSP fallback was removed; reported runs require pinned MNE.

## Files

| File | Role |
|---|---|
| `config.py` | Paths, bands, seed, subject/session lists, operating point |
| `preprocess.py` | GDF to epochs for both train and evaluation sessions |
| `adaptation.py` | Label-free Euclidean alignment for T/E session shift |
| `fbcsp.py` | Filter-bank CSP feature extractor using MNE CSP |
| `decode.py` | Calibrated LDA/SVM and equal soft vote |
| `gating.py` | Confidence gate and debounced hysteresis helper |
| `eegnet.py` | Optional PyTorch EEGNet comparison baseline |
| `selective_eval.py` | Train-on-T, test-on-E risk-coverage evaluation |
| `reliability.py` | ECE, Brier score, reliability diagrams, risk-coverage helpers |
| `stats.py` | Wilcoxon and per-subject matched-coverage deltas |
| `moabb_eval.py` | External replication entry point |
| `run_all.sh` | One-command reproduction |
| `writeup.tex` | Paper-style writeup with findings, architecture, evaluation, and conclusion |

## Provenance

Raw data are not committed. Generated arrays, plots, and reports are reproducible from scripts and ignored by git.
