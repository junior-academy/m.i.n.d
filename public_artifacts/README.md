# Public Artifacts

This directory contains the paper-facing artifact package.

- `v2_paper_artifacts/`: curated v2 paper tables and figures.
- File count: 78 files.
- Excluded: `.DS_Store`, Matplotlib font caches, raw smoke runs, and full audit runs.

The larger audit package remains under ignored `outputs/v2/` directories so the paper can be reproduced without including every intermediate run.

## Reproducing the Public Package

Build the paper artifacts:

```bash
python3 scripts/make_v2_paper_artifacts.py --root outputs/v2 --out-dir outputs/v2/paper_artifacts
```

Then copy the clean files into `public_artifacts/v2_paper_artifacts/`, excluding `.DS_Store` and `.mplconfig/`.
