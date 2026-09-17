# Qualitative cases

Three test patients selected by BridgeRefine-L1 seed-42 patient SSIM: highest (1BA222), median (1BC007), lowest (1BC014).

For each patient, `arrays.npz` contains `ct`, `mri_real`, `coarse_mri`, `bridgerefine_l1`, `ct_only`, and `brain_mask` with shape [S,128,128]. The preview slice is the axial slice with the largest brain-mask area. All MRI outputs use the same [-1,1] to [0,255] display window; no per-model enhancement was applied. The error map is computed from raw arrays with a fixed 0-max scale.
