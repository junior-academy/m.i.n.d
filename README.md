# M.I.N.D Held-Out Session Evaluation

This repo now answers one question:

> When a motor-imagery BCI is allowed to abstain on low-confidence trials, does a calibrated LDA+SVM ensemble stay more reliable than a single model when tested on a different recording session?

## Main Claim

The headline evaluation is subject-paired and session-held-out:

- Fit preprocessing outputs, FBCSP feature extractors, probability calibration, LDA, and SVM on `A0xT`.
- Score exactly once on the paired `A0xE` session.
- Compare the equal-weight LDA+SVM soft vote against pre-specified LDA.
- Compare both decoders at matched coverage, not ensemble-confident trials against single-model all-trials.
- Report per-subject deltas and a Wilcoxon signed-rank test. With `n = 9`, inference is explicitly limited.

## Reproduce

Place BCI Competition IV 2a GDF files in `data/BCICIV_2a_gdf/`, then run:

```bash
./run_all.sh
```

Outputs are written to `outputs/heldout_session/`:

- `subject_results.csv`: one row per subject at the pre-registered coverage.
- `risk_coverage.csv`: threshold sweep for ensemble and LDA, used descriptively.
- `headline_stats.csv`: Wilcoxon summary.
- `per_subject_deltas.csv`: paired deltas used by the test.

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
| `fbcsp.py` | Filter-bank CSP feature extractor using MNE CSP |
| `decode.py` | Calibrated LDA/SVM and equal soft vote |
| `gating.py` | Confidence gate and debounced hysteresis helper |
| `selective_eval.py` | Train-on-T, test-on-E risk-coverage evaluation |
| `stats.py` | Wilcoxon and per-subject matched-coverage deltas |
| `moabb_eval.py` | External replication entry point |
| `run_all.sh` | One-command reproduction |

## Provenance

Raw data are not committed. Generated arrays, plots, and reports are reproducible from scripts and ignored by git.
