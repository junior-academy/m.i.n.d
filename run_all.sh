#!/usr/bin/env bash
set -euo pipefail

python preprocess.py
python selective_eval.py
python stats.py

python - <<'PY'
from pathlib import Path
import pandas as pd

base = Path("outputs/heldout_session")
subject = pd.read_csv(base / "subject_results.csv")
stats = pd.read_csv(base / "headline_stats.csv").iloc[0]

print("\nHeadline held-out A0xT -> A0xE result")
print(f"coverage: {stats['coverage']:.2f}")
print(f"ensemble accuracy: {stats['mean_ensemble_acc']:.4f}")
print(f"LDA accuracy: {stats['mean_lda_acc']:.4f}")
print(f"mean delta: {stats['mean_delta_ensemble_minus_lda']:.4f}")
print(f"Wilcoxon p (ensemble > LDA): {stats['wilcoxon_p_greater']:.4g}")
print("\nPer-subject deltas:")
print(subject[['subject', 'delta_ensemble_minus_lda']].to_string(index=False))
PY
