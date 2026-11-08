"""E15: brain-mask ROI metrics and patient-level 3D consistency."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
from skimage.metrics import structural_similarity


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def ssim_gray(real: np.ndarray, generated: np.ndarray) -> float:
    real01 = (real.astype(np.float32) + 1.0) / 2.0
    generated01 = (generated.astype(np.float32) + 1.0) / 2.0
    c1, c2 = 0.0001, 0.0009
    mu_x = cv2.GaussianBlur(real01, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(generated01, (11, 11), 1.5)
    sx = cv2.GaussianBlur(real01**2, (11, 11), 1.5) - mu_x**2
    sy = cv2.GaussianBlur(generated01**2, (11, 11), 1.5) - mu_y**2
    sxy = cv2.GaussianBlur(real01 * generated01, (11, 11), 1.5) - mu_x * mu_y
    return float(
        np.mean(
            (2 * mu_x * mu_y + c1)
            * (2 * sxy + c2)
            / (mu_x**2 + mu_y**2 + c1)
            / (sx + sy + c2)
        )
    )


def psnr_gray(real: np.ndarray, generated: np.ndarray) -> float:
    real01 = (real.astype(np.float32) + 1.0) / 2.0
    generated01 = (generated.astype(np.float32) + 1.0) / 2.0
    mse = float(np.mean((real01 - generated01) ** 2))
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def find_predictions(root: Path) -> dict[str, Path]:
    candidates = {
        "CT-only-long": root
        / "experiments/ordered_supplement_20261030/results/stage1_fair_training/ct_long/test_predictions.npy",
        "CT-only-strong": root
        / "experiments/ordered_supplement_20261030/results/stage1_fair_training/ct_strong/test_predictions.npy",
        "GAN-joint": root
        / "experiments/ordered_supplement_20261030/results/stage1_fair_training/joint_gan/test_predictions.npy",
        "GAN-frozen-seed42": root
        / "experiments/ordered_supplement_20261030/results/stage0_final_metrics/predictions/Conditional_GAN_prior_+_refiner_seed42.npy",
        "Local-bridge-2x1-seed42": root
        / "experiments/ordered_supplement_20261030/results/stage0_final_metrics/predictions/Local_bridge_2x1_+_refiner_seed42.npy",
        "Official-SelfRDB-prior": root
        / "experiments/ordered_supplement_20261030/results/stage1_official_prior/official_test.npy",
        "Official-SelfRDB-refiner-seed42": root
        / "experiments/ordered_supplement_20261030/results/stage1_official_refiner/seed42/test_predictions.npy",
        "Official-SelfRDB-refiner-seed123": root
        / "experiments/ordered_supplement_20261030/results/stage1_official_refiner/seed123/test_predictions.npy",
        "Official-SelfRDB-refiner-seed2026": root
        / "experiments/ordered_supplement_20261030/results/stage1_official_refiner/seed2026/test_predictions.npy",
    }
    return {name: path for name, path in candidates.items() if path.exists()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = root / "experiments/controls/cache"
    names = [str(v) for v in load_json(cache / "names_test.json")]
    real = np.load(cache / "mr_test.npy", mmap_mode="r")[:, 0].astype(np.float32)

    masks = {}
    for patient in sorted({name.replace("\\", "/").split("/", 1)[0] for name in names}):
        path = root / "brain" / patient / "mask.nii.gz"
        if path.exists():
            masks[patient] = nib.load(path).get_fdata().astype(np.float32)

    predictions = find_predictions(root)
    rows = []
    for method, path in predictions.items():
        prediction = np.load(path)
        if prediction.ndim == 4:
            prediction = prediction[:, 0]
        if prediction.shape != real.shape:
            raise ValueError(f"{method}: {prediction.shape} vs {real.shape}")
        if prediction.min() >= 0 and prediction.max() <= 1.01:
            prediction = prediction * 2.0 - 1.0
        patient_indices: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, name in enumerate(names):
            patient, filename = name.replace("\\", "/").split("/", 1)
            slice_index = int(Path(filename).stem)
            patient_indices[patient].append((slice_index, index))
        for patient, indexed in patient_indices.items():
            indexed.sort()
            volume_real = np.stack([real[index] for _, index in indexed])
            volume_pred = np.stack([prediction[index] for _, index in indexed])
            roi_ssim = []
            roi_psnr = []
            if patient in masks:
                for slice_index, index in indexed:
                    if slice_index >= masks[patient].shape[2]:
                        continue
                    mask = cv2.resize(
                        masks[patient][:, :, slice_index],
                        (128, 128),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    mask = (mask > 0).astype(np.float32)
                    masked_real = real[index] * mask
                    masked_pred = prediction[index] * mask
                    roi_ssim.append(ssim_gray(masked_real, masked_pred))
                    roi_psnr.append(psnr_gray(masked_real, masked_pred))
            real01 = (volume_real + 1.0) / 2.0
            pred01 = (volume_pred + 1.0) / 2.0
            rows.append(
                {
                    "method": method,
                    "patient": patient,
                    "slices": len(indexed),
                    "roi_ssim": float(np.mean(roi_ssim)) if roi_ssim else float("nan"),
                    "roi_psnr": float(np.mean(roi_psnr)) if roi_psnr else float("nan"),
                    "vol_mae": float(np.mean(np.abs(real01 - pred01))),
                    "vol_rmse": float(np.sqrt(np.mean((real01 - pred01) ** 2))),
                    "vol_ssim_3d": float(
                        structural_similarity(
                            real01,
                            pred01,
                            data_range=1.0,
                        )
                    ),
                    "real_z_mad": float(
                        np.mean(np.abs(np.diff(volume_real, axis=0)))
                    ),
                    "pred_z_mad": float(
                        np.mean(np.abs(np.diff(volume_pred, axis=0)))
                    ),
                }
            )
    with (out_dir / "roi3d_patient.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        summary[method] = {
            metric: {
                "mean": float(
                    np.nanmean([row[metric] for row in method_rows])
                ),
                "sd": float(
                    np.nanstd([row[metric] for row in method_rows], ddof=1)
                ),
            }
            for metric in ("roi_ssim", "roi_psnr", "vol_mae", "vol_rmse", "vol_ssim_3d")
        }
        summary[method]["z_mad_ratio"] = {
            "mean": float(
                np.mean(
                    [row["pred_z_mad"] / row["real_z_mad"] for row in method_rows]
                )
            ),
            "sd": float(
                np.std(
                    [row["pred_z_mad"] / row["real_z_mad"] for row in method_rows],
                    ddof=1,
                )
            ),
        }
    (out_dir / "roi3d_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
