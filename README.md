# M.I.N.D. v2

M.I.N.D. v2 is a reliability-aware motor-imagery EEG decoding study. The
branch evaluates BCI decoders not only by all-trial accuracy, but also by
calibration, selective prediction, risk-coverage behavior, and controller-level
FIRE/HOLD replay.

The central question is:

> When a motor-imagery BCI decoder is uncertain, can calibrated probabilities
> and abstention produce a more reliable command stream than forced-choice
> classification?

## Branch Layout

- `m.i.n.d-v2`: active paper branch with deep models, external validation,
  calibration ablations, controller replay, and public paper artifacts.
- `m.i.n.d-v1`: classical baseline branch centered on calibrated FBCSP + LDA/SVM.
- `archive/main-confusion-matrices-t-test`: local archive branch for legacy
  confusion-matrix and t-test work.
- `archive/stats-tests`: local archive branch for older statistics scripts.

The remote default branch is `origin/m.i.n.d-v2`.

## Current Paper Claim

The v2 paper uses BCI Competition IV Dataset 2a as the primary held-out-session
benchmark. Each subject is trained on `A0xT` and evaluated once on paired
`A0xE`. Evaluation labels are not used for alignment, feature extraction,
model fitting, calibration, threshold choice, or ensemble weighting.

The active v2 comparison includes:

- calibrated v1 LDA baseline,
- EEGNet,
- lightweight spatial-attention EEG model,
- validation-weighted deep ensemble,
- optional post-ensemble temperature scaling,
- max-probability, entropy, and margin uncertainty scores,
- controller replay with confidence hysteresis and three-of-five debounce.

The paper should be read as a reliability evaluation study, not a deployment
claim. The current evidence supports calibrated abstention and auditability;
safe firing still requires tighter threshold calibration and external domain
review.

## Key Results

Primary BCI IV 2a deep run:

- v1 LDA: all-trial accuracy `0.435`, 60%-coverage selective accuracy `0.524`.
- Spatial-attention model: all-trial accuracy about `0.699`, selective accuracy about `0.780`.
- Validation-weighted deep ensemble: selective accuracy about `0.783`, ECE about `0.058`, Brier about `0.403`.

Additional validation and stress tests:

- Local BCI Competition IIIa LDA: accuracy `0.659`, selective accuracy `0.769`, coverage `0.614`.
- MOABB five-subject validation:
  - Cho2017: accuracy `0.640`, selective accuracy `0.724`, ECE `0.058`.
  - PhysionetMI: accuracy `0.560`, selective accuracy `0.600`, ECE `0.203`.
  - BNCI2014_001: accuracy `0.650`, selective accuracy `0.692`, ECE `0.102`.
- Focused calibration ablation:
  - no temperature scaling: ECE `0.137`,
  - per-model temperature scaling: ECE `0.126`,
  - post-ensemble temperature scaling: ECE `0.072`.

## Repository Structure

```text
v2/
  config.py        # v2 defaults, paths, subjects, coverage points, seeds
  datasets.py      # BCI IV 2a, local BCI IIIa, and MOABB loaders
  models.py        # v1 LDA wrapper, EEGNet, spatial model, ensembles
  calibration.py   # temperature scaling and softmax helpers
  uncertainty.py   # max-probability, entropy, and margin scores
  evaluation.py    # metrics, risk-coverage, reliability bins, controller tables
  plotting.py      # paper-style figures with LaTeX-like figure typography
  runner.py        # experiment orchestration and CLI

scripts/
  run_v2_headline.py          # primary BCI IV 2a headline runs
  run_v2_moabb.py             # MOABB validation helper
  make_v2_paper_artifacts.py  # combined paper tables and figures

public_artifacts/
  v2_paper_artifacts/         # 78-file public paper artifact package

AUDIT_PACKAGE.md              # commands for reproducing the larger audit runs
v2_experiment.py              # direct v2 CLI entry point
writeup_v2.tex                # v2 paper manuscript
writeup.tex                   # v1 baseline manuscript retained for reference
```

## Environment

Use Python 3.11 if possible.

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-eegnet.txt
```

Sanity checks:

```bash
python -c "import mne, moabb, sklearn; print('core ok')"
python -c "import torch; print('torch ok')"
```

## Data

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

The `A0xE.gdf` files use unknown cues, so evaluation labels must be supplied
from the official BCI Competition IV label files:

```bash
python download_true_labels.py
```

Local BCI Competition IIIa files are expected under:

```text
data/BCICIV_3a_gdf/
```

MOABB datasets are downloaded through MOABB and cached outside the repository.
Raw EEG files and caches are not committed.

## Running v2 Experiments

Direct CLI example:

```bash
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

Useful flags:

- `--model`: `lda`, `eegnet`, `spatial`, or `all`
- `--use-ea`: enable label-free Euclidean alignment
- `--calibrate`: add temperature-scaled deep model outputs
- `--ensemble`: `none`, `deep`, or `hybrid`
- `--ensemble-weights`: `equal` or `validation`
- `--post-ensemble-calibration`: recalibrate soft-voted ensemble probabilities
- `--uncertainty-score`: `max-prob`, `entropy`, or `margin`
- `--coverage-target`: matched-coverage operating point
- `--include-bci-iiia`: include local BCI Competition IIIa subjects
- `--include-moabb`: include MOABB datasets
- `--moabb-datasets`: dataset names such as `Cho2017 PhysionetMI BNCI2014_001`
- `--moabb-subject-limit`: subjects per MOABB dataset

Run the primary BCI IV 2a headline script:

```bash
python scripts/run_v2_headline.py
```

Run MOABB validation:

```bash
python scripts/run_v2_moabb.py \
  --datasets Cho2017 PhysionetMI BNCI2014_001 \
  --subject-limit 5 \
  --out-dir outputs/v2/moabb_5subj_3datasets_lda
```

Rebuild combined paper artifacts from saved runs:

```bash
python scripts/make_v2_paper_artifacts.py \
  --root outputs/v2 \
  --out-dir outputs/v2/paper_artifacts
```

## Outputs

Each v2 run writes:

- `config.json`
- `subject_metrics.csv`
- `aggregate_metrics.csv`
- `risk_coverage.csv`
- `reliability_diagram_bins.csv`
- `coverage_sweep.csv`
- `ablation_summary.csv`
- `controller_replay.csv`
- `figures/*.png`
- `figures/*.pdf`

The public paper artifact package is:

```text
public_artifacts/v2_paper_artifacts/
```

It contains 78 files: combined tables plus paper-ready PNG/PDF figures. The
larger run-level audit directories remain under ignored `outputs/v2/`. See
`AUDIT_PACKAGE.md` for the exact reproduction commands.

## Paper

Compile the v2 manuscript:

```bash
pdflatex writeup_v2.tex
pdflatex writeup_v2.tex
```

The manuscript references figures from:

```text
public_artifacts/v2_paper_artifacts/figures/
```

The v1 manuscript remains in `writeup.tex` as a baseline reference.

## Notes on Scope

The active paper path excludes older within-session CV runs, large committed
arrays, dashboard artifacts, Pygame demos, stacking experiments, and historical
plot outputs. Those items either do not support the held-out-session
reliability claim or are preserved only on archive branches.

Confusion matrices from the archive branch should not be copied into v2 as
static images. If confusion matrices are needed for the paper, regenerate them
from current v2 predictions so they match the reported runs.
