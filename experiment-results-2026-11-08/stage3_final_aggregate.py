"""E06/E14: final multi-seed, paired, and center summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def bootstrap_ci(values: np.ndarray, seed: int = 42, n_boot: int = 10000):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[indices].mean(axis=1)
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def paired(left, right, label):
    patients = sorted(set(left) & set(right))
    differences = np.asarray([left[p] - right[p] for p in patients])
    return {
        "comparison": label,
        "n_patients": len(patients),
        "mean_delta": float(differences.mean()),
        "median_delta": float(np.median(differences)),
        "improved_patients": int((differences > 0).sum()),
        "bootstrap_ci95": bootstrap_ci(differences),
        "wilcoxon_raw_p": float(
            stats.wilcoxon(differences, alternative="two-sided").pvalue
        ),
        "per_patient_delta": {
            patient: float(delta)
            for patient, delta in zip(patients, differences)
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gan_runs = {
        seed: load_json(
            root
            / (
                "experiments/gan_prior/summary.json"
                if seed == 42
                else f"experiments/gan_prior/seed{seed}/summary.json"
            )
        )
        for seed in (42, 123, 2026)
    }
    bridge_runs = {
        seed: load_json(
            root
            / (
                "experiments/coarse_2x1/summary.json"
                if seed == 42
                else f"experiments/coarse_2x1/seed{seed}/summary.json"
            )
        )
        for seed in (42, 123, 2026)
    }
    official_dir = (
        root
        / "experiments/ordered_supplement_20261030/results/stage1_official_refiner"
    )
    official_runs = {
        seed: load_json(official_dir / f"seed{seed}/result.json")
        for seed in (42, 123, 2026)
        if (official_dir / f"seed{seed}/result.json").exists()
    }

    methods = {
        "GAN prior + refiner": {
            seed: run["per_patient_ssim"] for seed, run in gan_runs.items()
        },
        "Local bridge 2x1 + refiner": {
            seed: run["per_patient_ssim"] for seed, run in bridge_runs.items()
        },
    }
    if official_runs:
        methods["Official SelfRDB prior + refiner"] = {
            seed: run["per_patient_ssim"]
            for seed, run in official_runs.items()
        }

    averages = {}
    seed_rows = []
    for method, seed_values in methods.items():
        patients = sorted(next(iter(seed_values.values())))
        averages[method] = {
            patient: float(
                np.mean(
                    [
                        seed_values[seed][patient]
                        for seed in seed_values
                    ]
                )
            )
            for patient in patients
        }
        for seed, values in seed_values.items():
            seed_rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "ssim_mean": float(np.mean(list(values.values()))),
                    "ssim_sd": float(np.std(list(values.values()), ddof=1)),
                }
            )

    paired_results = {}
    if "Official SelfRDB prior + refiner" in methods:
        paired_results["official_vs_gan"] = paired(
            averages["Official SelfRDB prior + refiner"],
            averages["GAN prior + refiner"],
            "Official SelfRDB prior + refiner vs GAN prior + refiner",
        )
        paired_results["official_vs_bridge2"] = paired(
            averages["Official SelfRDB prior + refiner"],
            averages["Local bridge 2x1 + refiner"],
            "Official SelfRDB prior + refiner vs local bridge 2x1 + refiner",
        )

    center_rows = []
    for method, values in averages.items():
        for center in ("1BA", "1BB", "1BC"):
            subset = [
                value
                for patient, value in values.items()
                if patient.startswith(center)
            ]
            center_rows.append(
                {
                    "method": method,
                    "center": center,
                    "n_patients": len(subset),
                    "ssim_mean": float(np.mean(subset)),
                    "ssim_sd": float(np.std(subset, ddof=1)),
                }
            )

    with (out_dir / "seed_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["method", "seed", "ssim_mean", "ssim_sd"]
        )
        writer.writeheader()
        writer.writerows(seed_rows)
    with (out_dir / "center_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "center", "n_patients", "ssim_mean", "ssim_sd"],
        )
        writer.writeheader()
        writer.writerows(center_rows)
    result = {
        "seed_metrics": seed_rows,
        "patient_three_seed_average": averages,
        "paired": paired_results,
        "center_metrics": center_rows,
    }
    (out_dir / "final_statistics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
