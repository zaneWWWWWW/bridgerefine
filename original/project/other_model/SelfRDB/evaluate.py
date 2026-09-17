# -*- coding: utf-8 -*-
"""
Comprehensive model evaluation for SelfRDB CT-to-MRI translation.

Metrics: SSIM, PSNR, MSE, MAE
Visual: side-by-side comparison grids, difference maps, histograms
Analysis: per-patient statistics, distribution comparison

Usage:
    # Full test set evaluation
    python evaluate.py --real-dir ../data_slices --gen-dir ../result/selfrdb_test \\
                       --output-dir ../result/eval_report

    # Single patient
    python evaluate.py --real-dir ../data_slices/1BA001/mr \\
                       --gen-dir ../result/selfrdb_test/1BA001/mr_generated
"""
import argparse
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return {}
    return {
        path.stem: path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def read_gray(path, size=None):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read: {path}")
    if size is not None and (img.shape[1], img.shape[0]) != size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


# ---- Metrics ----
def compute_mse(a, b):
    return float(np.mean((a - b) ** 2))


def compute_mae(a, b):
    return float(np.mean(np.abs(a - b)))


def compute_psnr(a, b, max_val=1.0):
    error = compute_mse(a, b)
    if error == 0:
        return float("inf")
    return float(10.0 * math.log10((max_val ** 2) / error))


def compute_ssim(a, b, max_val=1.0):
    """SSIM with default window size."""
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    mu_x = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(b, (11, 11), 1.5)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = cv2.GaussianBlur(a ** 2, (11, 11), 1.5) - mu_x_sq
    sigma_y_sq = cv2.GaussianBlur(b ** 2, (11, 11), 1.5) - mu_y_sq
    sigma_xy = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_xy

    num = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    den = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    return float(np.mean(num / (den + 1e-12)))


def compute_all_metrics(real, generated):
    return {
        "ssim":  compute_ssim(real, generated),
        "psnr":  compute_psnr(real, generated),
        "mse":   compute_mse(real, generated),
        "mae":   compute_mae(real, generated),
    }


# ---- Visualization ----
def make_comparison_grid(ct, real_mr, gen_mr, slice_name, output_dir):
    """Create a 2x2 comparison figure: CT | Real MRI / Gen MRI | Diff map."""
    diff = np.abs(real_mr - gen_mr)
    # Normalize for display
    diff_viz = (diff / max(diff.max(), 0.01) * 255).astype(np.uint8)
    diff_viz = cv2.applyColorMap(diff_viz, cv2.COLORMAP_HOT)

    ct_u8 = (ct * 255).astype(np.uint8)
    real_u8 = (real_mr * 255).astype(np.uint8)
    gen_u8 = (gen_mr * 255).astype(np.uint8)

    # Convert grayscale to BGR for concatenation
    top = np.concatenate([
        cv2.cvtColor(ct_u8, cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(real_u8, cv2.COLOR_GRAY2BGR),
    ], axis=1)
    bottom = np.concatenate([
        cv2.cvtColor(gen_u8, cv2.COLOR_GRAY2BGR),
        diff_viz,
    ], axis=1)
    grid = np.concatenate([top, bottom], axis=0)

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = ct_u8.shape
    cv2.putText(grid, "CT", (10, 25), font, 0.6, (0, 255, 0), 1)
    cv2.putText(grid, "Real MRI", (w + 10, 25), font, 0.6, (0, 255, 0), 1)
    cv2.putText(grid, "Generated", (10, h + 25), font, 0.6, (0, 255, 0), 1)
    cv2.putText(grid, "|Diff|", (w + 10, h + 25), font, 0.6, (0, 255, 0), 1)

    out_path = os.path.join(output_dir, f"cmp_{slice_name}.png")
    cv2.imwrite(out_path, grid)
    return out_path


def make_histogram_comparison(real_images, gen_images, output_path):
    """Plot histogram comparing real vs generated intensity distributions."""
    real_flat = np.concatenate([img.ravel() for img in real_images])
    gen_flat = np.concatenate([img.ravel() for img in gen_images])

    # Compute histograms
    bins = 100
    real_hist, _ = np.histogram(real_flat, bins=bins, range=(0, 1))
    gen_hist, _ = np.histogram(gen_flat, bins=bins, range=(0, 1))

    # Create histogram image using OpenCV
    hist_h, hist_w = 300, 512
    hist_img = np.ones((hist_h, hist_w, 3), dtype=np.uint8) * 255

    # Normalize to fit
    max_count = max(real_hist.max(), gen_hist.max())
    if max_count > 0:
        real_hist_norm = (real_hist / max_count * (hist_h - 20)).astype(int)
        gen_hist_norm = (gen_hist / max_count * (hist_h - 20)).astype(int)

    bin_w = hist_w // bins
    for i in range(bins - 1):
        x1 = i * bin_w
        x2 = (i + 1) * bin_w
        cv2.rectangle(hist_img, (x1, hist_h - 20 - real_hist_norm[i]),
                      (x2, hist_h - 20), (255, 0, 0), -1)
        cv2.rectangle(hist_img, (x1, hist_h - 20 - gen_hist_norm[i]),
                      (x2, hist_h - 20), (0, 0, 255), 2)

    cv2.putText(hist_img, "Blue=Real  Red=Generated", (10, hist_h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imwrite(output_path, hist_img)


# ---- Main evaluation ----
def evaluate_patient(real_dir, gen_dir, patient_id, output_dir, save_grids=True):
    """Evaluate one patient: compute metrics, optionally save comparison grids."""
    real_imgs = list_images(real_dir)
    gen_imgs = list_images(gen_dir)
    common = sorted(set(real_imgs) & set(gen_imgs))

    if not common:
        print(f"  [{patient_id}] No paired images found")
        return None, []

    results = []
    real_samples = []
    gen_samples = []

    for name in common:
        real = read_gray(real_imgs[name])
        gen = read_gray(gen_imgs[name], (real.shape[1], real.shape[0]))

        metrics = compute_all_metrics(real, gen)
        metrics["name"] = f"{patient_id}/{name}"
        results.append(metrics)

        real_samples.append(real)
        gen_samples.append(gen)

    if save_grids and results:
        # Save best, median, worst SSIM examples
        sorted_by_ssim = sorted(results, key=lambda x: x["ssim"])
        examples = [
            ("best", sorted_by_ssim[-1]),
            ("median", sorted_by_ssim[len(sorted_by_ssim) // 2]),
            ("worst", sorted_by_ssim[0]),
        ]
        for label, entry in examples:
            name = entry["name"].split("/")[-1]
            real = read_gray(real_imgs[name])
            gen = read_gray(gen_imgs[name], (real.shape[1], real.shape[0]))
            make_comparison_grid(real, real, gen, f"{patient_id}_{label}", output_dir)

    return results, real_samples, gen_samples


def evaluate_full_test_set(data_root, gen_root, output_dir, patient_split_path=None):
    """Evaluate the full test set with per-patient and aggregate statistics."""
    data_root = Path(data_root)
    gen_root = Path(gen_root)
    os.makedirs(output_dir, exist_ok=True)

    # Determine patients to evaluate
    if patient_split_path and os.path.exists(patient_split_path):
        with open(patient_split_path) as f:
            split = json.load(f)
        test_pids = split.get("test", [])
    else:
        test_pids = sorted(
            d.name for d in gen_root.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        )

    print(f"Evaluating {len(test_pids)} test patients...")

    all_results = []
    patient_summaries = {}
    all_real = []
    all_gen = []

    for pid in sorted(test_pids):
        real_dir = data_root / pid / "mr"
        gen_dir = gen_root / pid / "mr_generated"
        pat_out = os.path.join(output_dir, pid)
        os.makedirs(pat_out, exist_ok=True)

        results, reals, gens = evaluate_patient(real_dir, gen_dir, pid, pat_out, save_grids=True)

        if results is None:
            continue

        all_results.extend(results)
        all_real.extend(reals)
        all_gen.extend(gens)

        # Per-patient summary
        for metric in ["ssim", "psnr", "mse", "mae"]:
            vals = [r[metric] for r in results if math.isfinite(r[metric])]
            if metric not in patient_summaries:
                patient_summaries[metric] = {}
            patient_summaries[metric][pid] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "slices": len(vals),
            }

    # Aggregate statistics
    aggregate = {}
    for metric in ["ssim", "psnr", "mse", "mae"]:
        vals = [r[metric] for r in all_results if math.isfinite(r[metric])]
        vals_arr = np.array(vals)
        aggregate[metric] = {
            "mean": float(np.mean(vals_arr)),
            "std": float(np.std(vals_arr)),
            "median": float(np.median(vals_arr)),
            "min": float(np.min(vals_arr)),
            "max": float(np.max(vals_arr)),
        }

    # Histogram
    if all_real and all_gen:
        make_histogram_comparison(all_real, all_gen, os.path.join(output_dir, "histogram.png"))

    # Write reports
    report = {
        "total_slices": len(all_results),
        "total_patients": len([p for p in patient_summaries.get("ssim", {})]),
        "aggregate": aggregate,
        "per_patient": patient_summaries,
        # Top-5 and bottom-5 patients by SSIM
        "top5_ssim": sorted(
            [(pid, v["mean"]) for pid, v in patient_summaries.get("ssim", {}).items()],
            key=lambda x: -x[1]
        )[:5],
        "bottom5_ssim": sorted(
            [(pid, v["mean"]) for pid, v in patient_summaries.get("ssim", {}).items()],
            key=lambda x: x[1]
        )[:5],
    }

    report_path = os.path.join(output_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Evaluation Summary: {len(all_results)} slices from {report['total_patients']} patients")
    print(f"{'='*60}")
    for metric in ["ssim", "psnr", "mse", "mae"]:
        ag = aggregate[metric]
        print(f"  {metric.upper():>6}: mean={ag['mean']:.4f} ± {ag['std']:.4f}  "
              f"(median={ag['median']:.4f}, range=[{ag['min']:.4f}, {ag['max']:.4f}])")

    print(f"\nReport saved: {report_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate SelfRDB generated MRI quality")
    # Input
    parser.add_argument("--real-dir", default=None, help="Real MRI directory (single patient)")
    parser.add_argument("--gen-dir", default=None, help="Generated MRI directory (single patient)")
    parser.add_argument("--data-root", default="../data_slices", help="Full data root")
    parser.add_argument("--gen-root", default="../result/selfrdb_test", help="Full generated root")
    parser.add_argument("--patient-split", default="../checkpoints/selfrdb/patient_split.json")
    # Output
    parser.add_argument("--output-dir", default="../result/eval_report")
    # Options
    parser.add_argument("--max-slices", type=int, default=0, help="Max slices (0=all)")
    parser.add_argument("--no-grids", action="store_true", help="Skip comparison grid generation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.real_dir and args.gen_dir:
        # Single directory mode
        results, reals, gens = evaluate_patient(
            args.real_dir, args.gen_dir, "single", args.output_dir, save_grids=not args.no_grids
        )
        if results:
            print(f"\nEvaluated {len(results)} slices:")
            for metric in ["ssim", "psnr", "mse", "mae"]:
                vals = [r[metric] for r in results if math.isfinite(r[metric])]
                print(f"  {metric.upper()}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    else:
        # Full test set mode
        evaluate_full_test_set(args.data_root, args.gen_root, args.output_dir, args.patient_split)


if __name__ == "__main__":
    main()
