"""Evaluate an official SelfRDB prediction with the paper's metric pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch


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


def patient_means(values: list[float], names: list[str]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for name, value in zip(names, values):
        grouped[str(name).replace("\\", "/").split("/", 1)[0]].append(float(value))
    return {patient: float(np.mean(items)) for patient, items in sorted(grouped.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = root / "experiments/controls/cache"
    with (cache / "names_test.json").open("r", encoding="utf-8") as handle:
        names = [str(name) for name in json.load(handle)]
    real = np.load(cache / "mr_test.npy", mmap_mode="r")[:, 0].astype(np.float32)
    prediction = np.load(args.pred, mmap_mode="r").astype(np.float32)
    if prediction.ndim == 4:
        prediction = prediction[:, 0]
    if prediction.shape != real.shape:
        raise ValueError(
            f"prediction/real shape mismatch: {prediction.shape} vs {real.shape}"
        )
    if float(prediction.min()) >= 0 and float(prediction.max()) <= 1.01:
        prediction = prediction * 2.0 - 1.0
    np.save(out_dir / "official_test.npy", prediction[:, None])
    (out_dir / "official_test_names.json").write_text(
        json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    device = torch.device("cuda")
    import lpips

    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    ssim_values = []
    psnr_values = []
    mae_values = []
    lpips_values = []
    for index in range(len(names)):
        ssim_values.append(ssim_gray(real[index], prediction[index]))
        psnr_values.append(psnr_gray(real[index], prediction[index]))
        mae_values.append(
            float(
                np.mean(
                    np.abs(
                        real[index].astype(np.float32)
                        - prediction[index].astype(np.float32)
                    )
                )
            )
        )
    with torch.inference_mode():
        for start in range(0, len(names), args.batch_size):
            stop = min(start + args.batch_size, len(names))
            real_batch = (
                torch.from_numpy(real[start:stop])
                .float()
                .unsqueeze(1)
                .repeat(1, 3, 1, 1)
                .to(device)
            )
            pred_batch = (
                torch.from_numpy(prediction[start:stop])
                .float()
                .unsqueeze(1)
                .repeat(1, 3, 1, 1)
                .to(device)
            )
            lpips_values.extend(
                lpips_model(real_batch, pred_batch, normalize=False)
                .flatten()
                .cpu()
                .numpy()
                .tolist()
            )

    patient_values = {
        "ssim": patient_means(ssim_values, names),
        "psnr": patient_means(psnr_values, names),
        "mae": patient_means(mae_values, names),
        "lpips": patient_means(lpips_values, names),
    }
    rows = []
    for patient in sorted(patient_values["ssim"]):
        rows.append(
            {
                "prior": "Official SelfRDB",
                "patient": patient,
                **{
                    metric: values[patient]
                    for metric, values in patient_values.items()
                },
            }
        )
    with (out_dir / "official_prior_quality_patient.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["prior", "patient", "ssim", "psnr", "mae", "lpips"]
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "prior": "Official SelfRDB",
        "n_patients": len(patient_values["ssim"]),
        "n_slices": len(names),
        **{
            f"{metric}_mean": float(np.mean(list(values.values())))
            for metric, values in patient_values.items()
        },
        **{
            f"{metric}_sd": float(np.std(list(values.values()), ddof=1))
            for metric, values in patient_values.items()
        },
    }
    (out_dir / "official_prior_quality.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
