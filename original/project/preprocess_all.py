# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path
from main import mha_to_highres

brain_dir = Path("brain")
output_root = Path("data_slices")
patient_dirs = sorted(d for d in brain_dir.iterdir() if d.is_dir())

print(f"Found {len(patient_dirs)} patients")

for i, patient_dir in enumerate(patient_dirs):
    pid = patient_dir.name
    ct_path = patient_dir / "ct.nii.gz"
    mr_path = patient_dir / "mr.nii.gz"

    ct_out = output_root / pid / "ct"
    mr_out = output_root / pid / "mr"

    if ct_out.exists() and mr_out.exists() and        len(list(ct_out.glob("*.jpg"))) > 0 and len(list(mr_out.glob("*.png"))) > 0:
        print(f"[{i+1}/{len(patient_dirs)}] {pid} - SKIP (already processed)")
        continue

    try:
        print(f"[{i+1}/{len(patient_dirs)}] {pid} - processing CT...")
        n_ct = mha_to_highres(
            save_dir=str(ct_out),
            mha_path=str(ct_path),
            modality="CT",
            output_format="jpg",
            remove_black_images=True,
        )
        print(f"  CT: {n_ct} slices")

        print(f"[{i+1}/{len(patient_dirs)}] {pid} - processing MR...")
        n_mr = mha_to_highres(
            save_dir=str(mr_out),
            mha_path=str(mr_path),
            modality="MR",
            correct_bias_field=True,
            output_format="png",
        )
        print(f"  MR: {n_mr} slices")
    except Exception as e:
        print(f"[{i+1}/{len(patient_dirs)}] {pid} - ERROR: {e}", file=sys.stderr)

print("\nBatch processing complete.")
