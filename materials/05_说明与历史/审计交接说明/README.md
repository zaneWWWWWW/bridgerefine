# BridgeRefine paper handoff v2

This package follows the 2026-09-09 audit. It contains:

- `efficiency/` — independent 10×2 benchmark results;
- `qualitative_cases/` — real arrays and previews for three test patients;
- `metrics/` — patient, paired, seed, and significance exports;
- `checkpoints/` — checkpoint hashes and provenance;
- `configs/` and `scripts/` — reproducing configurations and scripts;
- `metadata/` — cohort/acquisition and ethics templates, with missing fields explicitly marked;
- `figures_source/` — source figure files, including the real qualitative composite;
- `paper/` — latest Chinese manuscript PDF/LaTeX.

Primary 10×2 efficiency results (separate processes, no model co-residency):

- CT-only: 2.06 ± 0.18 ms/slice; additional inference memory 39.3 MB;
- BridgeRefine: 233.6 ± 12.6 ms/slice; additional inference memory 134.0 MB.

The old 20×5 efficiency record is retained only as an obsolete reference and must
not be used for the final 10×2 configuration.

All paths are relative to this directory. See `manifest.csv` and `MISSING.md`.
