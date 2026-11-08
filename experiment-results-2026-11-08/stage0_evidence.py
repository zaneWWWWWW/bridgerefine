"""Recompute prior quality, patient-level statistics, and center summaries.

This stage uses existing caches and checkpoints. It does not train models.
"""

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
from scipy import stats


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def patient_id(name: str) -> str:
    return str(name).replace("\\", "/").split("/", 1)[0]


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


def patient_means(values: list[float], names: list[str]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for name, value in zip(names, values):
        grouped[patient_id(name)].append(float(value))
    return {pid: float(np.mean(items)) for pid, items in sorted(grouped.items())}


def bootstrap_ci(
    values: np.ndarray, seed: int = 42, n_boot: int = 10000
) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def paired_summary(
    left: dict[str, float],
    right: dict[str, float],
    label: str,
) -> dict:
    patients = sorted(set(left) & set(right))
    differences = np.asarray([left[p] - right[p] for p in patients], dtype=np.float64)
    pvalue = float(stats.wilcoxon(differences, alternative="two-sided").pvalue)
    return {
        "comparison": label,
        "n_patients": len(patients),
        "mean_delta": float(differences.mean()),
        "median_delta": float(np.median(differences)),
        "sd_delta": float(differences.std(ddof=1)),
        "improved_patients": int((differences > 0).sum()),
        "bootstrap_ci95": bootstrap_ci(differences),
        "wilcoxon_raw_p": pvalue,
        "patients": patients,
        "paired_differences": differences.tolist(),
    }


def center_summary(
    method_patient_values: dict[str, dict[str, float]]
) -> list[dict]:
    rows = []
    for center in ("1BA", "1BB", "1BC"):
        for method, values in method_patient_values.items():
            subset = [v for pid, v in values.items() if pid.startswith(center)]
            rows.append(
                {
                    "center": center,
                    "method": method,
                    "n_patients": len(subset),
                    "ssim_mean": float(np.mean(subset)),
                    "ssim_sd": float(np.std(subset, ddof=1)),
                }
            )
    return rows


def weighted_lpips(
    real: np.ndarray,
    generated: np.ndarray,
    device: torch.device,
    model,
    batch_size: int = 64,
) -> np.ndarray:
    values = []
    with torch.inference_mode():
        for start in range(0, len(real), batch_size):
            stop = min(start + batch_size, len(real))
            real_batch = (
                torch.from_numpy(real[start:stop]).float().unsqueeze(1)
            )
            generated_batch = (
                torch.from_numpy(generated[start:stop]).float().unsqueeze(1)
            )
            real_batch = real_batch.repeat(1, 3, 1, 1).to(device)
            generated_batch = generated_batch.repeat(1, 3, 1, 1).to(device)
            distance = model(real_batch, generated_batch, normalize=False)
            values.extend(distance.flatten().cpu().numpy().tolist())
    return np.asarray(values, dtype=np.float64)


def analyze(root: Path, out_dir: Path, device_name: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = root / "experiments/controls/cache"
    names = [str(v) for v in load_json(cache / "names_test.json")]
    real = np.load(cache / "mr_test.npy", mmap_mode="r")
    ct = np.load(cache / "ct_test.npy", mmap_mode="r")
    pix2pix = np.load(cache / "pix2pix_prior_test.npy", mmap_mode="r")
    bridge10 = np.load(
        root / "experiments/controls/bridge_eval_coarse_test.npy",
        mmap_mode="r",
    )
    detreg = np.load(
        root / "experiments/controls/detreg_coarse_test.npy",
        mmap_mode="r",
    )
    bridge2_artifact = torch.load(
        root / "experiments/step_ablation/test_coarse_2x1.pt",
        map_location="cpu",
        weights_only=False,
    )
    bridge2 = np.asarray(bridge2_artifact["coarse"], dtype=np.float32)
    if [str(v) for v in bridge2_artifact["names"]] != names:
        raise ValueError("2x1 coarse names do not match the audit test cache")

    priors = {
        "Conditional GAN (pix2pix)": pix2pix[:, 0].astype(np.float32),
        "Local bridge 10x2": bridge10[:, 0].astype(np.float32),
        "Local bridge 2x1": bridge2[:, 0].astype(np.float32),
        "Deterministic regressor": detreg[:, 0].astype(np.float32),
        "Null prior (CT copy)": ct[:, 0].astype(np.float32),
    }
    real_2d = real[:, 0].astype(np.float32)

    device = torch.device(device_name)
    import lpips

    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    prior_rows = []
    prior_patient_values: dict[str, dict[str, list[float]]] = {}
    for label, prediction in priors.items():
        print(f"prior metrics: {label}", flush=True)
        ssim_values = []
        psnr_values = []
        mae_values = []
        for index in range(len(names)):
            generated = prediction[index]
            target = real_2d[index]
            ssim_values.append(ssim_gray(target, generated))
            psnr_values.append(psnr_gray(target, generated))
            mae_values.append(mae_gray(target, generated))
        lpips_values = weighted_lpips(
            real_2d, prediction, device, lpips_model
        )
        patient_metric_values = {
            "ssim": patient_means(ssim_values, names),
            "psnr": patient_means(psnr_values, names),
            "mae": patient_means(mae_values, names),
            "lpips": patient_means(lpips_values.tolist(), names),
        }
        prior_patient_values[label] = patient_metric_values
        for patient in sorted(patient_metric_values["ssim"]):
            prior_rows.append(
                {
                    "prior": label,
                    "patient": patient,
                    "ssim": patient_metric_values["ssim"][patient],
                    "psnr": patient_metric_values["psnr"][patient],
                    "mae": patient_metric_values["mae"][patient],
                    "lpips": patient_metric_values["lpips"][patient],
                }
            )

    with (out_dir / "prior_quality_patient.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["prior", "patient", "ssim", "psnr", "mae", "lpips"]
        )
        writer.writeheader()
        writer.writerows(prior_rows)

    prior_summary = []
    for label, metrics in prior_patient_values.items():
        prior_summary.append(
            {
                "prior": label,
                **{
                    f"{metric}_mean": float(np.mean(list(values.values())))
                    for metric, values in metrics.items()
                },
                **{
                    f"{metric}_sd": float(np.std(list(values.values()), ddof=1))
                    for metric, values in metrics.items()
                },
                **{
                    f"{metric}_bootstrap_ci95": bootstrap_ci(
                        np.asarray(list(values.values()))
                    )
                    for metric, values in metrics.items()
                },
            }
        )

    # Final results from the multi-seed experiments.
    gan_paths = {
        42: root / "experiments/gan_prior/summary.json",
        123: root / "experiments/gan_prior/seed123/summary.json",
        2026: root / "experiments/gan_prior/seed2026/summary.json",
    }
    bridge2_paths = {
        42: root / "experiments/coarse_2x1/summary.json",
        123: root / "experiments/coarse_2x1/seed123/summary.json",
        2026: root / "experiments/coarse_2x1/seed2026/summary.json",
    }
    gan_seeds = {
        seed: load_json(path)["per_patient_ssim"]
        for seed, path in gan_paths.items()
    }
    bridge2_seeds = {
        seed: load_json(path)["per_patient_ssim"]
        for seed, path in bridge2_paths.items()
    }
    patients = sorted(gan_seeds[42])
    gan_average = {
        patient: float(np.mean([gan_seeds[seed][patient] for seed in gan_seeds]))
        for patient in patients
    }
    bridge2_average = {
        patient: float(
            np.mean([bridge2_seeds[seed][patient] for seed in bridge2_seeds])
        )
        for patient in patients
    }

    paired = {
        "gan_vs_bridge2_seedwise": {
            str(seed): paired_summary(
                gan_seeds[seed],
                bridge2_seeds[seed],
                f"GAN prior + refiner vs bridge 2x1 + refiner, seed {seed}",
            )
            for seed in (42, 123, 2026)
        },
        "gan_vs_bridge2_seed_average": paired_summary(
            gan_average,
            bridge2_average,
            "GAN prior + refiner vs bridge 2x1 + refiner, patient-wise 3-seed average",
        ),
    }

    seed_summary = []
    for method, seed_values in (
        ("Conditional GAN prior + refiner", gan_seeds),
        ("Local bridge 2x1 + refiner", bridge2_seeds),
    ):
        for seed, values in seed_values.items():
            seed_summary.append(
                {
                    "method": method,
                    "seed": seed,
                    "patient_mean": float(np.mean(list(values.values()))),
                    "patient_sd": float(np.std(list(values.values()), ddof=1)),
                }
            )

    method_patient_values = {}
    for seed in (42, 123, 2026):
        method_patient_values[f"Conditional GAN + refiner seed {seed}"] = gan_seeds[seed]
        method_patient_values[f"Local bridge 2x1 + refiner seed {seed}"] = bridge2_seeds[seed]
    centers = center_summary(method_patient_values)

    # Same-source gain at seed 42 for the prior-source comparison.
    metrics_path = (
        root
        / "other_model/paper/source_data/unified10x2/"
        "metrics_patient_unified_10x2.csv"
    )
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as handle:
        metrics_rows = {row["patient"]: row for row in csv.DictReader(handle)}
    final_by_source = {
        "Conditional GAN (pix2pix)": gan_seeds[42],
        "Local bridge 10x2": {
            patient: float(row["BridgeRefine-L1_ssim"])
            for patient, row in metrics_rows.items()
        },
        "Local bridge 2x1": bridge2_seeds[42],
    }
    gain_rows = []
    for source, final_values in final_by_source.items():
        coarse_values = prior_patient_values[source]["ssim"]
        for patient in patients:
            gain_rows.append(
                {
                    "source": source,
                    "patient": patient,
                    "prior_ssim": coarse_values[patient],
                    "final_ssim": final_values[patient],
                    "delta_ssim": final_values[patient] - coarse_values[patient],
                }
            )
    with (out_dir / "prior_to_final_gain.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source",
                "patient",
                "prior_ssim",
                "final_ssim",
                "delta_ssim",
            ],
        )
        writer.writeheader()
        writer.writerows(gain_rows)

    correlations = {}
    for source in final_by_source:
        source_rows = [row for row in gain_rows if row["source"] == source]
        prior_values = np.asarray([row["prior_ssim"] for row in source_rows])
        final_values = np.asarray([row["final_ssim"] for row in source_rows])
        gain_values = np.asarray([row["delta_ssim"] for row in source_rows])
        correlations[source] = {
            "prior_vs_final_spearman": float(
                stats.spearmanr(prior_values, final_values).statistic
            ),
            "prior_vs_gain_spearman": float(
                stats.spearmanr(prior_values, gain_values).statistic
            ),
            "prior_ssim_mean": float(prior_values.mean()),
            "final_ssim_mean": float(final_values.mean()),
            "delta_ssim_mean": float(gain_values.mean()),
        }

    result = {
        "n_patients": len(patients),
        "n_slices": len(names),
        "prior_quality": prior_summary,
        "seed_summary": seed_summary,
        "paired_statistics": paired,
        "center_summary": centers,
        "prior_to_final_gain": correlations,
    }
    (out_dir / "evidence.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with (out_dir / "center_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["center", "method", "n_patients", "ssim_mean", "ssim_sd"],
        )
        writer.writeheader()
        writer.writerows(centers)

    with (out_dir / "seed_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "seed", "patient_mean", "patient_sd"],
        )
        writer.writeheader()
        writer.writerows(seed_summary)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = analyze(args.root.resolve(), args.out.resolve(), args.device)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
