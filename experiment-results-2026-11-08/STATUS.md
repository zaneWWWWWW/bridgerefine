# Ordered supplement experiments

Last updated: 2026-11-08 13:16 Asia/Taipei

## Completed

### Stage 0: audit and existing-data analysis

- `stage0_audit`: 71/71 checks passed.
- Patient split: 144 train / 18 validation / 18 test.
- Slice counts: 27,309 / 3,473 / 3,415.
- Patient and cached-slice intersections: none.
- Key cache, split, and checkpoint SHA-256 hashes recorded.
- Prior quality measured on the full 3,415-slice test set.
- GAN-prior and bridge-2x1 refiners evaluated across three seeds with
  SSIM, PSNR, MAE, LPIPS, patient-paired statistics, and center summaries.
- Four complete inference pipelines benchmarked in isolated processes on the
  RTX 3090.

### Main Stage 0 results

| Prior | Prior SSIM | Prior PSNR | Prior MAE | Prior LPIPS |
|---|---:|---:|---:|---:|
| Conditional GAN (pix2pix) | 0.8190 | 19.40 | 0.1002 | 0.0714 |
| Local bridge 2x1 | 0.5938 | 17.58 | 0.1394 | 0.2473 |
| Local bridge 10x2 | 0.5855 | 16.44 | 0.1568 | 0.2287 |
| Deterministic regressor | 0.8095 | 19.77 | 0.0980 | 0.1287 |
| Null prior (CT copy) | 0.4523 | 11.96 | 0.2603 | 0.2220 |

| Complete method (3-seed mean) | SSIM | PSNR | MAE | LPIPS |
|---|---:|---:|---:|---:|
| Conditional GAN prior + refiner | 0.8348 | 20.03 | 0.0922 | 0.1065 |
| Local bridge 2x1 + refiner | 0.8242 | 19.96 | 0.0942 | 0.1223 |

GAN prior + refiner versus bridge 2x1 + refiner:

- Patient-wise three-seed mean difference in SSIM: +0.01057.
- Better in 16/18 patients.
- Bootstrap 95% CI: [+0.00551, +0.01706].
- Wilcoxon p = 0.000252.

| Pipeline | Mean latency (ms/slice) | Peak allocated memory (MB) |
|---|---:|---:|
| CT-only U-Net | 3.28 | 4,213 |
| Conditional GAN prior + refiner | 10.03 | 4,205 |
| Local bridge 2x1 + refiner | 51.13 | 6,339 |
| Local bridge 10x2 + refiner | 341.34 | 6,339 |

## Official SelfRDB check

The first official training process completed 19 epochs and stopped during the
last epoch with `CUDNN_STATUS_EXECUTION_FAILED_CUDART` followed by a CUDA illegal
memory access. All 19 epoch checkpoints and the final complete validation result
were preserved.

Best completed validation point:

- checkpoint: `epoch=18-step=129694.ckpt`;
- validation SSIM: 79.91%;
- validation PSNR: 18.79 dB.

Full-test inference from this checkpoint completed without error under the
official metric implementation:

- PSNR: 19.05 dB;
- SSIM: 81.30%.

On the paper's common metric scale after converting the official output from
[0,1] to [-1,1], the official SelfRDB prior achieved:

| Metric | Official SelfRDB | Local bridge 2x1 |
|---|---:|---:|
| SSIM | 0.8243 | 0.5938 |
| PSNR | 19.19 | 17.58 |
| MAE | 0.1000 | 0.1394 |
| LPIPS | 0.0846 | 0.2473 |

This invalidates the previous claim that diffusion bridges are intrinsically
weak priors. The earlier weak result was caused by the local simplified bridge
implementation. The paper must distinguish local adaptation from official
SelfRDB.

## Official SelfRDB completion

`BridgeRefine-Official-Resume` completed the final epoch with exit code 0:

- checkpoint: `epoch=19-step=143348.ckpt`;
- final-epoch validation SSIM: 78.46%;
- final-epoch validation PSNR: 18.70 dB.

The final epoch was worse than epoch 18 (79.91% SSIM), so
`epoch=18-step=129694.ckpt` remains the selected official checkpoint.

The first retry failed while spawning the validation DataLoader because of a
Windows multiprocessing pickle error. The retry used `data.num_workers=0` and
completed successfully.

## Fair-training queue

The first `ct_long` process ran 18 epochs and was externally terminated without
a Python traceback after about 2 hours 9 minutes. Its best checkpoint was
preserved. The resumed run then stopped after epoch 39 because validation loss
had not improved for 12 epochs.

- best epoch: 27;
- best validation L1: 0.101314;
- test SSIM: 0.8094;
- test PSNR: 19.85 dB.

Checkpointing now stores the optimizer and a resumable `last.pt` every epoch.
`BridgeRefine-Fair-Resume` resumed `ct_long` from the preserved epoch-18 best
checkpoint at epoch 19:

- 128x128 input;
- official noise schedule, bridge posterior, recursion, and adversarial loss;
- batch 64, up to 100 epochs, early stopping for `ct_long`.

Completed follow-up experiments:

| Experiment | Parameters | Best epoch | Test SSIM | Test PSNR |
|---|---:|---:|---:|---:|
| `ct_long` | 1.33M | 27 | 0.8094 | 19.85 |
| `ct_strong` | 16.28M | 19 | 0.8256 | 20.22 |
| `joint_gan` | 12.70M | 2 | 0.8342 | 19.96 |

`joint_gan` did not beat the frozen GAN-prior result (0.8348 mean across three
seeds). The stronger and longer-trained CT-only baseline is much stronger than
the original 20-epoch baseline (0.8009 mean), reducing the two-stage advantage.

The first attempt's logs and checkpoint are archived as `pre_resume_*` and
`best_pre_resume.pt`.

The official repository's custom CUDA operators were replaced by its own native
PyTorch implementations because the Windows host has no MSVC toolchain. The
SelfRDB equations and network structure are unchanged.

## Official-prior refiner

The official-prior refiner completed all three seeds on the audited split:

| Seed | Best epoch | SSIM | PSNR | MAE | LPIPS |
|---|---:|---:|---:|---:|---:|
| 42 | 19 | 0.8362 | 20.12 | 0.0910 | 0.1145 |
| 123 | 20 | 0.8366 | 20.14 | 0.0908 | 0.1123 |
| 2026 | 18 | 0.8376 | 20.09 | 0.0909 | 0.1137 |

The three-seed mean SSIM is approximately 0.8368, above the frozen GAN-prior
result (0.8348) and substantially above the local bridge 2x1 result (0.8242).

The remaining queue initially failed on an excess dimension in the synthetic
spatial-shift transform. The queue restarted from E09/E10 after correcting the
transform and is currently running fixed-prior degradation diagnostics.

Fixed-prior diagnostics have completed. Selected results from the GAN-prior
refiner:

| Intervention | SSIM | PSNR | LPIPS |
|---|---:|---:|---:|
| Clean prior | 0.84 | 20.03 | 0.11 |
| Blur sigma 1.0 | 0.82 | 20.03 | 0.16 |
| Noise sigma 0.10 | 0.81 | 19.90 | 0.11 |
| Shift 4 pixels | 0.61 | 14.63 | 0.14 |
| Cross-patient prior | 0.55 | 10.66 | 0.58 |
| Zero prior | 0.55 | 10.57 | 0.65 |
| Zero CT channel | 0.82 | 19.68 | 0.12 |
| Local 32x32 prior erase | 0.79 | 15.98 | 0.17 |

The refiner depends strongly on a patient-matched prior; replacing it with a
cross-patient prior is almost as damaging as zeroing it. The CT channel is much
less important under this trained configuration.

The queue is now running matched refiner retraining for blur 1.0, noise 0.10,
and shift 4.

## Additional completed experiments

Matched retraining after controlled prior degradation:

| Degradation | SSIM | PSNR | LPIPS |
|---|---:|---:|---:|
| Blur sigma 1.0 | 0.8342 | 20.04 | 0.1065 |
| Noise sigma 0.10 | 0.8357 | 20.01 | 0.1084 |
| Shift 4 pixels | 0.8332 | 19.94 | 0.1093 |

Matched sampling retraining:

| Configuration | SSIM | PSNR |
|---|---:|---:|
| 2x2 | 0.8266 | 19.95 |
| 4x1 | 0.8233 | 19.87 |

256x256:

| Method | SSIM | PSNR |
|---|---:|---:|
| CT-only | 0.7657 | 19.29 |
| GAN prior + refiner | 0.7980 | 19.81 |

Final three-seed comparison:

- official SelfRDB prior + refiner mean SSIM: 0.8368;
- GAN prior + refiner mean SSIM: 0.8348;
- official versus GAN paired difference: +0.00204,
  95% CI [-0.00122, +0.00531], Wilcoxon p = 0.304;
- improved patients: 9/18;
- official versus local bridge 2x1: +0.01261, 16/18 improved, p < 0.0001.
- official seed 42 versus strong CT-only: +0.01056,
  16/18 improved, 95% CI [+0.00638, +0.01495], p = 0.000145.

ROI and 3D evaluation:

- strong CT-only: ROI SSIM 0.8721, 3D SSIM 0.8291;
- frozen GAN prior + refiner: ROI SSIM 0.8714, 3D SSIM 0.8346;
- official SelfRDB prior + refiner seed 42: ROI SSIM 0.8714,
  3D SSIM 0.8349.

Official inference cost:

- official SelfRDB prior: 1128.7 ms/slice;
- official prior + refiner: 1129.1 ms/slice.

## Strict five-fold cross-validation

All five folds completed with exit code 0. Each fold retrained the bridge,
deterministic regressor, and all three refiners using only that fold's training
patients.

| Method | Patients | SSIM |
|---|---:|---:|
| Local bridge prior + refiner | 180 | 0.8059 |
| Deterministic regressor prior + refiner | 180 | 0.8125 |
| CT-only | 180 | 0.7978 |

Patient-level paired comparisons:

| Comparison | Mean delta | Improved | 95% CI | Wilcoxon p |
|---|---:|---:|---:|---:|
| Bridge vs CT-only | +0.00816 | 142/180 | [+0.00657, +0.00976] | 6.8e-17 |
| Deterministic prior vs bridge | +0.00656 | 138/180 | [+0.00509, +0.00802] | 4.9e-14 |
| Deterministic prior vs CT-only | +0.01471 | 177/180 | [+0.01362, +0.01580] | 3.8e-31 |

The strict CV result supports a small but consistent benefit from a matched
prior over CT-only. It also shows that the deterministic prior, not the local
bridge, is the strongest first-stage prior in the leakage-free five-fold
evaluation.

## Final status

All E01-E17 experiments listed in the supplement plan are complete. The
remaining work is manuscript revision and final reporting, not additional
training.

## Output locations

- `results/stage0_audit/`
- `results/stage0_evidence/`
- `results/stage0_final_metrics/`
- `results/stage0_efficiency/`
- `results/stage1_official_prior/`
- `results/stage1_fair_training/`
- `results/stage1_official_refiner/`
- `official_selfrdb/logs/official_cached_seed42_b8_e20/`
