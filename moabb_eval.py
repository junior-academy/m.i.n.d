"""External replication placeholder for MOABB selective evaluation.

The held-out BCI IV 2a result is the main claim. MOABB replication should reuse
the same decoder contract from ``decode.py`` and report matched-coverage
risk-coverage curves only. RF, stacking, and baseline weighting stay out of the
main path.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "MOABB replication is Tier 2/3 and intentionally separate from the "
        "headline held-out-session run. Port the dataset-specific loader here "
        "while preserving decode.py's LDA/SVM/equal-vote contract."
    )


if __name__ == "__main__":
    main()
