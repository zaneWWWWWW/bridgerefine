# Validation performed during archival

Date: 2026-09-17. Analysis runtime: Python 3.12.3, NumPy 1.26.4, SciPy 1.17.1.

- All 71 original source/config files match `SOURCE_MANIFEST.json` SHA-256 values.
- All 69 Python files parse successfully. One historical figure script emits two
  invalid-escape warnings (`\hat`); the archived source was preserved unchanged.
- Results summarization ran against the actual local exports: 18 test patients,
  3,415 slices, 17 patients with positive paired SSIM change, mean change
  0.0088944444, final-protocol latency ratio 113.4959.
- Recomputed run-level sample SD from the rounded archived means:
  CT-only 0.01116975; BridgeRefine-L1 0.00210792.
- The primary CT-only Wilcoxon result and adjusted p-value match the archive.
  The original six-comparison Holm table omits the cumulative-maximum operation
  for some entries. The new tool reports the corrected values alongside the
  unchanged archive; this is not a silent revision of the paper.
- All 10 primary checkpoint files were recovered locally and matched the original
  checkpoint manifest. They are not stored in this repository.
- The recovered L1 refiner checkpoint loaded strictly into `RefinerGenerator`;
  a CPU forward pass on two zero tensors produced a finite `[1, 1, 128, 128]`
  output. Parameter count: 1,337,793.
- The recovered bridge checkpoint loaded strictly into `X0UNet`. Parameter count:
  14,405,377. No diffusion sampling benchmark was performed.

These checks establish archive integrity and basic model compatibility. They do
not establish that the preserved scripts can reproduce training end to end.
Training, full image-metric recomputation, and GPU timing were not rerun.
The exact final multi-seed L1 training script remains absent from the supplied
source material. See README for the other recorded reproduction gaps.
