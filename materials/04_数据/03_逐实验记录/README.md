# Supplement Handoff Package

This directory contains one subdirectory per experiment, following 交接说明.md:

```text
input_ct_only_seed42
input_coarse_only_seed42
input_ct_coarse_seed42
input_ct_coarse_seed123
input_ct_coarse_seed2026
efficiency_rtx5060_batch1
```

Each experiment directory contains:

- `manifest.json`
- `config.json`
- `command.txt`
- `environment.txt`
- `run.log`
- `status.txt`
- `metrics_slice.csv`
- `metrics_patient.csv`
- `summary.json`
- `checkpoints/best.pt`
- `predictions/index.csv`
- `predictions/preds.pt`
- `sha256sums.txt`

All coarse MRI used for training, validation, and test are generated with the
frozen SelfRDB bridge using 10 sampling steps and 2 recursions.

Verification:

```bash
cd experiments/supplement/<experiment_id>
sha256sum -c sha256sums.txt
cat manifest.json
cat status.txt
```

The frozen paper bundle is in `experiments/final_frozen/`.
