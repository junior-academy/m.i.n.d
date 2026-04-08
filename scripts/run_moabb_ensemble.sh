#!/usr/bin/env bash
set -euo pipefail

# Runs MOABB ensemble evaluation and writes outputs under:
#   m.i.n.d/outputs/moabb_ensemble/<dataset>/
#
# Examples:
#   bash m.i.n.d/scripts/run_moabb_ensemble.sh --dataset bnci2014_001 --n-classes 4 --subjects 1,2,3
#   bash m.i.n.d/scripts/run_moabb_ensemble.sh --dataset physionetmi --n-classes 2 --subjects 1,2,3 --events left_hand,right_hand --mne-data m.i.n.d/data/mne_data

python3 -m pip install --quiet moabb mne scikit-learn numpy pandas scipy

python3 m.i.n.d/moabb_ensemble_eval.py "$@"

