# Private Audit Package

The full audit package is intentionally kept private under ignored `outputs/v2/` directories. It contains raw per-run artifacts used to verify the paper claims, but it does not need to be published with the public paper package.

## Private Audit Directories

- `outputs/v2/moabb_5subj_3datasets_lda`
- `outputs/v2/local_iiia_lda_full`
- `outputs/v2/local_iiia_lda_entropy`
- `outputs/v2/local_iiia_lda_margin`
- `outputs/v2/calibration_ablation_no_temp_s2_e10`
- `outputs/v2/calibration_ablation_per_model_temp_s2_e10`
- `outputs/v2/calibration_ablation_post_ensemble_temp_s2_e10`

Together with the public paper artifact package, these directories preserve the full audit trail for the local IIIa, MOABB, uncertainty-score, calibration, and controller-replay claims.

## Reproduction Commands

Local IIIa and primary LDA:

```bash
python3 v2_experiment.py --model lda --use-ea --coverage-target 0.60 --include-bci-iiia --out-dir outputs/v2/local_iiia_lda_full
python3 v2_experiment.py --model lda --use-ea --coverage-target 0.60 --include-bci-iiia --uncertainty-score entropy --out-dir outputs/v2/local_iiia_lda_entropy
python3 v2_experiment.py --model lda --use-ea --coverage-target 0.60 --include-bci-iiia --uncertainty-score margin --out-dir outputs/v2/local_iiia_lda_margin
```

MOABB external validation:

```bash
python3 v2_experiment.py --model lda --use-ea --coverage-target 0.60 --include-moabb --moabb-datasets Cho2017 PhysionetMI BNCI2014_001 --moabb-subject-limit 5 --out-dir outputs/v2/moabb_5subj_3datasets_lda
```

Calibration ablations:

```bash
python3 v2_experiment.py --model all --use-ea --ensemble deep --ensemble-weights validation --coverage-target 0.60 --subject-limit 2 --epochs 10 --out-dir outputs/v2/calibration_ablation_no_temp_s2_e10
python3 v2_experiment.py --model all --use-ea --calibrate --ensemble deep --ensemble-weights validation --coverage-target 0.60 --subject-limit 2 --epochs 10 --out-dir outputs/v2/calibration_ablation_per_model_temp_s2_e10
python3 v2_experiment.py --model all --use-ea --calibrate --ensemble deep --ensemble-weights validation --post-ensemble-calibration --coverage-target 0.60 --subject-limit 2 --epochs 10 --out-dir outputs/v2/calibration_ablation_post_ensemble_temp_s2_e10
```

Public artifact regeneration:

```bash
python3 scripts/make_v2_paper_artifacts.py --root outputs/v2 --out-dir outputs/v2/paper_artifacts
```
