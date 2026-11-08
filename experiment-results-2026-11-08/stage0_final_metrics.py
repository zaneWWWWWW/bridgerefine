"""Recompute complete metrics for the GAN-prior and bridge-2x1 refiners."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch


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
    numerator = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sx + sy + c2)
    return float(np.mean(numerator / (denominator + 1e-12)))


def psnr_gray(real: np.ndarray, generated: np.ndarray) -> float:
    real01 = (real.astype(np.float32) + 1.0) / 2.0
    generated01 = (generated.astype(np.float32) + 1.0) / 2.0
    mse = float(np.mean((real01 - generated01) ** 2))
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def mae_gray(real: np.ndarray, generated: np.ndarray) -> float:
    return float(np.mean(np.abs(real.astype(np.float32) - generated.astype(np.float32))))


def patient_values(values: list[float], names: list[str]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for name, value in zip(names, values):
        grouped[name.replace("\\", "/").split("/", 1)[0]].append(float(value))
    return {patient: float(np.mean(items)) for patient, items in sorted(grouped.items())}


def lpips_batch(
    real: np.ndarray,
    generated: np.ndarray,
    device: torch.device,
    lpips_model,
) -> np.ndarray:
    values = []
    with torch.inference_mode():
        for start in range(0, len(real), 64):
            stop = min(start + 64, len(real))
            real_tensor = (
                torch.from_numpy(real[start:stop]).float().unsqueeze(1).repeat(1, 3, 1, 1)
            ).to(device)
            generated_tensor = (
                torch.from_numpy(generated[start:stop])
                .float()
                .unsqueeze(1)
                .repeat(1, 3, 1, 1)
            ).to(device)
            distance = lpips_model(real_tensor, generated_tensor, normalize=False)
            values.extend(distance.flatten().cpu().numpy().tolist())
    return np.asarray(values, dtype=np.float64)


def infer_refiner(model, ct: np.ndarray, coarse: np.ndarray, device: torch.device) -> np.ndarray:
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(ct), 64):
            stop = min(start + 64, len(ct))
            ct_tensor = (
                torch.from_numpy(ct[start:stop]).float().unsqueeze(1).to(device)
            )
            coarse_tensor = (
                torch.from_numpy(coarse[start:stop])
                .float()
                .unsqueeze(1)
                .to(device)
            )
            predictions.append(model(coarse_tensor, ct_tensor).cpu().numpy()[:, 0])
    return np.concatenate(predictions, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = out_dir / "predictions"
    prediction_dir.mkdir(exist_ok=True)

    sys.path.insert(0, str(root))
    from other_model.BridgeGAN.model import RefinerGenerator

    cache = root / "experiments/controls/cache"
    names = [str(v) for v in load_json(cache / "names_test.json")]
    real = np.load(cache / "mr_test.npy", mmap_mode="r")[:, 0].astype(np.float32)
    ct = np.load(cache / "ct_test.npy", mmap_mode="r")[:, 0].astype(np.float32)
    gan_prior = np.load(
        cache / "pix2pix_prior_test.npy", mmap_mode="r"
    )[:, 0].astype(np.float32)
    bridge2_artifact = torch.load(
        root / "experiments/step_ablation/test_coarse_2x1.pt",
        map_location="cpu",
        weights_only=False,
    )
    bridge2_prior = np.asarray(bridge2_artifact["coarse"], dtype=np.float32)[:, 0]
    if [str(v) for v in bridge2_artifact["names"]] != names:
        raise ValueError("2x1 cache names do not match test names")

    device = torch.device(args.device)
    import lpips

    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    runs = [
        ("Conditional GAN prior + refiner", 42, root / "experiments/gan_prior/last.pt", gan_prior),
        (
            "Conditional GAN prior + refiner",
            123,
            root / "experiments/gan_prior/seed123/last.pt",
            gan_prior,
        ),
        (
            "Conditional GAN prior + refiner",
            2026,
            root / "experiments/gan_prior/seed2026/last.pt",
            gan_prior,
        ),
        ("Local bridge 2x1 + refiner", 42, root / "experiments/coarse_2x1/last.pt", bridge2_prior),
        (
            "Local bridge 2x1 + refiner",
            123,
            root / "experiments/coarse_2x1/seed123/last.pt",
            bridge2_prior,
        ),
        (
            "Local bridge 2x1 + refiner",
            2026,
            root / "experiments/coarse_2x1/seed2026/last.pt",
            bridge2_prior,
        ),
    ]

    rows = []
    summaries = []
    for method, seed, checkpoint, prior in runs:
        print(f"metrics: {method}, seed {seed}", flush=True)
        model = RefinerGenerator(in_ch=2, out_ch=1, base_ch=64).to(device)
        checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint_data["model"])
        model.eval()
        prediction = infer_refiner(model, ct, prior, device)
        np.save(prediction_dir / f"{method.replace(' ', '_')}_seed{seed}.npy", prediction)
        del model
        torch.cuda.empty_cache()

        ssim_values = []
        psnr_values = []
        mae_values = []
        for index in range(len(names)):
            ssim_values.append(ssim_gray(real[index], prediction[index]))
            psnr_values.append(psnr_gray(real[index], prediction[index]))
            mae_values.append(mae_gray(real[index], prediction[index]))
        lpips_values = lpips_batch(real, prediction, device, lpips_model)
        patient_metric_values = {
            "ssim": patient_values(ssim_values, names),
            "psnr": patient_values(psnr_values, names),
            "mae": patient_values(mae_values, names),
            "lpips": patient_values(lpips_values.tolist(), names),
        }
        for patient in sorted(patient_metric_values["ssim"]):
            row = {"method": method, "seed": seed, "patient": patient}
            for metric, values in patient_metric_values.items():
                row[metric] = values[patient]
            rows.append(row)

        summary = {"method": method, "seed": seed}
        for metric, values in patient_metric_values.items():
            array = np.asarray(list(values.values()), dtype=np.float64)
            summary[f"{metric}_mean"] = float(array.mean())
            summary[f"{metric}_sd"] = float(array.std(ddof=1))
        summaries.append(summary)

        reference = load_json(
            root
            / (
                "experiments/gan_prior/summary.json"
                if "GAN" in method and seed == 42
                else "experiments/gan_prior/seed123/summary.json"
                if "GAN" in method and seed == 123
                else "experiments/gan_prior/seed2026/summary.json"
                if "GAN" in method
                else "experiments/coarse_2x1/summary.json"
                if seed == 42
                else "experiments/coarse_2x1/seed123/summary.json"
                if seed == 123
                else "experiments/coarse_2x1/seed2026/summary.json"
            )
        )
        recomputed_patients = list(patient_metric_values["ssim"].values())
        reference_patients = list(reference["per_patient_ssim"].values())
        summary["max_abs_ssim_difference_vs_summary"] = float(
            np.max(np.abs(np.asarray(recomputed_patients) - np.asarray(reference_patients)))
        )

    with (out_dir / "final_metrics_patient.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "seed", "patient", "ssim", "psnr", "mae", "lpips"],
        )
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "final_metrics_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fieldnames = [
            "method",
            "seed",
            "ssim_mean",
            "ssim_sd",
            "psnr_mean",
            "psnr_sd",
            "mae_mean",
            "mae_sd",
            "lpips_mean",
            "lpips_sd",
            "max_abs_ssim_difference_vs_summary",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    (out_dir / "final_metrics_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
