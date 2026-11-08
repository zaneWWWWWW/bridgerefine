# BridgeRefine supplemental experiment results

Export date: 2026-11-08

This directory contains the final numerical results, training logs, audit data,
and reproduction scripts for experiments E01-E17. Large binary predictions,
checkpoints, image caches, and medical images are intentionally excluded.

## Main findings

- Official SelfRDB prior + refiner, three-seed mean SSIM: **0.8368**.
- Conditional GAN prior + refiner, three-seed mean SSIM: **0.8348**.
- GAN versus official SelfRDB: +0.00204, 95% CI
  [-0.00122, 0.00531], Wilcoxon p=0.304. The difference is not significant.
- Official SelfRDB versus strong CT-only, seed 42: +0.01056,
  16/18 patients improved, 95% CI [0.00638, 0.01495], p=0.000145.
- Local bridge 2x1 + refiner: 0.8242.
- Strict five-fold over 180 patients:
  - local bridge prior + refiner: 0.8059;
  - deterministic regressor prior + refiner: 0.8125;
  - CT-only: 0.7978.
- Official SelfRDB full inference cost: 1128.7 ms/slice before refinement and
  1129.1 ms/slice including the refiner.

The results support a controlled evaluation framing rather than a claim that
the diffusion bridge has a unique performance advantage. Official SelfRDB is
much stronger than the earlier local simplified bridge, but it does not
significantly outperform a conditional GAN prior and is computationally much
more expensive.

## Directory map

- `STATUS.md`: experiment-by-experiment completion record and interpretation.
- `results/stage0_*`: data audit, prior quality, unified metrics, and efficiency.
- `results/stage1_*`: official SelfRDB, fair baselines, and refiner seeds.
- `results/stage2_*`: input-dependence diagnostics, degradation retraining, and
  sampling retraining.
- `results/stage3_*`: 256x256, ROI/3D, final statistics, and official efficiency.
- `cv_strict/`: leakage-free five-fold summaries and each fold result.
- `reproduction/official_selfrdb/`: compatibility files and native PyTorch
  operator fallback used to run official SelfRDB on Windows.
- `*.py`, `*.ps1`: experiment and orchestration scripts.

## Excluded artifacts

The following remain on the experiment host:

- per-slice `.npy` predictions;
- `.pt` and `.ckpt` checkpoints;
- raw CT/MRI caches and source images.

Checkpoint paths and source provenance are recorded in
`results/stage0_audit/file_hashes.csv` and `EXPORT_MANIFEST.json`.

## Statistical unit

All image-quality summaries are computed by averaging slices within a patient
and then treating the patient as the unit across patients. Slice count is not
used as an independent sample size.
