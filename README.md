# M.I.N.D v1

M.I.N.D. is a reproducible motor-imagery EEG study focused on reliability, not
just raw classifier accuracy. The project answers one main question:

> When a motor-imagery BCI is allowed to abstain on low-confidence trials, does a calibrated LDA+SVM ensemble stay more reliable than a single model when tested on a different recording session?

## Main Claim

The headline evaluation is subject-paired and session-held-out:

- Fit preprocessing outputs, FBCSP feature extractors, probability calibration, LDA, and SVM on `A0xT`.
- Apply label-free Euclidean alignment to reduce session covariance shift. `A0xE` labels are not used for alignment.
- Score exactly once on the paired `A0xE` session.
- Compare the equal-weight LDA+SVM soft vote against pre-specified LDA.
- Compare both decoders at matched coverage, not ensemble-confident trials against single-model all-trials.
- Report per-subject deltas and a Wilcoxon signed-rank test. With `n = 9`, inference is explicitly limited.

## Methodology

The pipeline is intentionally leakage-aware:

1. **Data split:** Train on BCI IV 2a `A0xT`; evaluate once on paired `A0xE`.
2. **Session adaptation:** Apply label-free Euclidean alignment separately to train and evaluation sessions. Evaluation labels are not used for alignment.
3. **Feature extraction:** Use FBCSP over `8-12`, `12-16`, `16-20`, and `20-30 Hz`, plus optional bandpower.
4. **Base decoders:** Train calibrated LDA and calibrated RBF-SVM.
5. **Main ensemble:** Average LDA and SVM probabilities with equal weights.
6. **Selective prediction:** Keep the highest-confidence trials at a fixed operating coverage and report risk/accuracy on those trials.
7. **Reliability evaluation:** Report ECE, Brier score, reliability diagrams, and risk-coverage curves.
8. **External validation:** Run the same decoder contract on MOABB datasets. EEGNet is optional and used only as a modern comparison baseline.

The main comparison is always the calibrated LDA+SVM ensemble vs calibrated LDA.
EEGNet is not part of the ensemble and should not be treated as the focus of the
project.

## Reproducibility

### 1. Environment

Use Python 3.11 if possible:

```bash
conda activate eegbci
python --version
```

Install the core dependencies:

```bash
python -m pip install -r requirements.txt
```

Optional EEGNet dependency:

```bash
python -m pip install -r requirements-eegnet.txt
```

Sanity check:

```bash
python -c "import mne, moabb, sklearn; print('core ok')"
python -c "import torch; print('eegnet ok')"
```

### 2. Data

Place BCI Competition IV 2a GDF files here:

```text
data/BCICIV_2a_gdf/
```

Expected examples:

```text
data/BCICIV_2a_gdf/A01T.gdf
data/BCICIV_2a_gdf/A01E.gdf
...
data/BCICIV_2a_gdf/A09T.gdf
data/BCICIV_2a_gdf/A09E.gdf
```

The `A0xE.gdf` files contain unknown cues (`783`) rather than class labels. The
true labels are published separately on the BCI Competition IV results page, and
`download_true_labels.py` normalizes them into
`data/BCICIV_2a_gdf/true_labels/A01E.csv` through `A09E.csv`.

### 3. Primary BCI IV 2a Run

From this directory, run:

```bash
python download_true_labels.py
./run_all.sh
```

`run_all.sh` performs:

1. preprocessing,
2. held-out selective evaluation,
3. Wilcoxon/per-subject statistics,
4. printed headline reliability summary.

Primary outputs are written to `outputs/heldout_session/`:

- `subject_results.csv`: one row per subject at the pre-registered coverage.
- `risk_coverage.csv`: threshold sweep for ensemble and LDA, used descriptively.
- `reliability_metrics.csv`: accuracy, mean confidence, ECE, and Brier score.
- `reliability_diagram_bins.csv`: top-label confidence-bin data for reliability diagrams.
- `reliability_diagrams/*.png`: plotted reliability diagrams when matplotlib is installed.
- `headline_stats.csv`: Wilcoxon summary.
- `per_subject_deltas.csv`: paired deltas used by the test.

### 4. No-Adaptation Ablation

To run the older cold-transfer baseline for comparison:

```bash
python selective_eval.py --adaptation none
python stats.py \
  --subject-results outputs/heldout_session_no_adaptation/subject_results.csv \
  --out outputs/heldout_session_no_adaptation/headline_stats.csv
```

### 5. EEGNet Comparison

To add EEGNet as a modern deep-learning comparison:

```bash
python -m pip install -r requirements-eegnet.txt
python selective_eval.py --include-eegnet --eegnet-epochs 60
```

This adds `EEGNet` rows/columns to the same reliability artifacts. It does not
change the main ensemble.

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

## Paper Writeup

The paper-style writeup is `writeup.tex`. It includes:

- abstract,
- purpose and exigence,
- background knowledge,
- architecture and novel application,
- evaluation,
- conclusion.

Compile it with:

```bash
pdflatex writeup.tex
pdflatex writeup.tex
```

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
| `writeup_v2.tex` | v2 paper-style writeup covering deep reliability architecture and results |

## Provenance

Raw data are not committed. Generated arrays, plots, and reports are reproducible from scripts and ignored by git.

## v2 Deep Reliability Pipeline

v2 keeps the v1 leakage-aware philosophy but adds modern deep backbones. The
protocol remains strict: `A0xT` trains the model and supplies the internal
validation split for calibration, threshold selection, and ensemble weighting;
`A0xE` labels are used only for final evaluation metrics.

Repository structure:

```text
v2/
  config.py        # defaults, paths, coverage points, seeds
  datasets.py      # BCI IV 2a and MOABB subject loaders
  models.py        # v1 LDA wrapper, EEGNet, spatial-attention EEG, ensembles
  calibration.py   # temperature scaling and softmax helpers
  uncertainty.py   # max-prob, entropy, and margin scores
  evaluation.py    # metrics, risk-coverage, reliability bins, tables
  plotting.py      # publication-style risk, reliability, delta, and summary figures
  runner.py        # experiment orchestration and CLI
scripts/
  run_v2_headline.py          # BCI IV 2a headline table runs
  run_v2_moabb.py             # configurable MOABB external validation
  make_v2_paper_artifacts.py  # combined paper tables and figures
v2_experiment.py              # direct CLI entry point
```

Architecture choices:

- **v1 LDA** remains the calibrated FBCSP + LDA baseline.
- **EEGNet** is the full deep backbone.
- **Spatial-attention EEG** is the second modern backbone. It learns channel
  importance weights before temporal convolution, giving a lightweight channel
  relationship model without the fragility of hand-designed graph adjacency.
- **Temperature scaling** is fit only on a validation split from the training
  session.
- **Deep ensembles** can use equal weights or validation-accuracy weights from
  the training-session validation split.
- **Post-ensemble calibration** can temperature-scale the soft-voted ensemble
  using validation predictions only.

Run the direct CLI:

```bash
conda activate eegbci
python v2_experiment.py \
  --model all \
  --use-ea \
  --calibrate \
  --ensemble deep \
  --ensemble-weights validation \
  --post-ensemble-calibration \
  --uncertainty-score max-prob \
  --coverage-target 0.60 \
  --epochs 60 \
  --out-dir outputs/v2/full_deep_bci_iv_2a_validation_postcal
```

Important CLI flags:

- `--model`: `lda`, `eegnet`, `spatial`, or `all`
- `--use-ea`: enable label-free Euclidean alignment
- `--calibrate`: add temperature-scaled deep model outputs
- `--ensemble`: `none`, `deep`, or `hybrid`
- `--ensemble-weights`: `equal` or `validation`
- `--post-ensemble-calibration`: recalibrate soft-voted ensemble probabilities
- `--uncertainty-score`: `max-prob`, `entropy`, or `margin`
- `--coverage-target`: matched-coverage operating point
- `--include-moabb`: include MOABB external datasets
- `--moabb-datasets`: MOABB dataset names, such as `Cho2017 PhysionetMI`
- `--moabb-subject-limit`: subjects per MOABB dataset

Reproduce the v2 BCI IV 2a headline comparison runs:

```bash
python scripts/run_v2_headline.py
```

Run external MOABB validation:

```bash
python scripts/run_v2_moabb.py \
  --datasets Cho2017 PhysionetMI \
  --subject-limit 5 \
  --epochs 60 \
  --out-dir outputs/v2/moabb_external
```

Generate paper-ready combined figures and tables from saved v2 outputs:

```bash
python scripts/make_v2_paper_artifacts.py \
  --root outputs/v2 \
  --out-dir outputs/v2/paper_artifacts
```

v2 writes:

- `subject_metrics.csv`
- `aggregate_metrics.csv`
- `risk_coverage.csv`
- `reliability_diagram_bins.csv`
- `coverage_sweep.csv`
- `ablation_summary.csv`
- `figures/*.png` and `figures/*.pdf`

Key paper artifacts:

- `outputs/v2/paper_artifacts/headline_summary_table.csv`
- `outputs/v2/paper_artifacts/coverage_sweep_summary_table.csv`
- `outputs/v2/paper_artifacts/figures/figure_1_headline_performance_BCI_IV_2a.pdf`
- `outputs/v2/paper_artifacts/figures/figure_2_risk_coverage_BCI_IV_2a.pdf`
- `outputs/v2/paper_artifacts/figures/figure_3_reliability_support_BCI_IV_2a.pdf`
- `outputs/v2/paper_artifacts/figures/figure_4_subject_deltas_BCI_IV_2a.pdf`
- `outputs/v2/paper_artifacts/figures/figure_5_coverage_sweep_BCI_IV_2a.pdf`

The corrected reliability figure is trial-weighted and includes confidence-bin
support, so sparse high-confidence bins from v1 LDA cannot visually dominate
the story.

Compile the v2 writeup:

```bash
pdflatex writeup_v2.tex
pdflatex writeup_v2.tex
```
