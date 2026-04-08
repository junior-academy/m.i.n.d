#!/usr/bin/env bash
set -euo pipefail

# Runs a MOABB MI evaluation (default dataset: PhysionetMI) and writes CSVs to:
#   m.i.n.d/outputs/moabb/physionetmi/
#
# Usage examples:
#   bash m.i.n.d/scripts/run_moabb_physionet.sh
#   bash m.i.n.d/scripts/run_moabb_physionet.sh --n-classes 2 --subjects 1,2,3 --n-jobs 2
#   bash m.i.n.d/scripts/run_moabb_physionet.sh --dataset bnci2014_001 --subjects 1,2,3
#   bash m.i.n.d/scripts/run_moabb_physionet.sh --download-timeout 300

python3 -m pip install --quiet moabb pyriemann mne scikit-learn pooch requests urllib3

python3 m.i.n.d/moabb_eval_physionet.py "$@"
