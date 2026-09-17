# -*- coding: utf-8 -*-
import argparse
import csv
import math
import os
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def read_gray_image(path, size=None):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Can not read image: {path}")
    if size is not None and (image.shape[1], image.shape[0]) != size:
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return image.astype(np.float32) / 255.0


def list_images(folder):
    folder = Path(folder)
    return {
        path.stem: path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def filter_informative_slices(real_dir, brain_threshold=0.10):
    image_dict = list_images(real_dir)
    names = sorted(image_dict)
    valid_names = []
    excluded_names = []
    for name in names:
        img = read_gray_image(image_dict[name])
        brain_ratio = float(np.mean(img > 0.01))
        if brain_ratio >= brain_threshold:
            valid_names.append(name)
        else:
            excluded_names.append(name)
    return valid_names, excluded_names


def mse(real, generated):
    return float(np.mean((real - generated) ** 2))


def mae(real, generated):
    return float(np.mean(np.abs(real - generated)))


def psnr(real, generated, max_value=1.0):
    error = mse(real, generated)
    if error == 0:
        return float("inf")
    return float(10.0 * math.log10((max_value ** 2) / error))


def ssim(real, generated, max_value=1.0, window_size=11, sigma=1.5):
    c1 = (0.01 * max_value) ** 2
    c2 = (0.03 * max_value) ** 2

    mu_x = cv2.GaussianBlur(real, (window_size, window_size), sigma)
    mu_y = cv2.GaussianBlur(generated, (window_size, window_size), sigma)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = cv2.GaussianBlur(real ** 2, (window_size, window_size), sigma) - mu_x_sq
    sigma_y_sq = cv2.GaussianBlur(generated ** 2, (window_size, window_size), sigma) - mu_y_sq
    sigma_xy = cv2.GaussianBlur(real * generated, (window_size, window_size), sigma) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    return float(np.mean(numerator / (denominator + 1e-12)))


def estimate_noise(image, kernel_size=5):
    denoised = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return image - denoised


def kl_divergence(real_noise, generated_noise, bins=256, epsilon=1e-8):
    hist_range = (-1.0, 1.0)
    p, _ = np.histogram(real_noise.ravel(), bins=bins, range=hist_range, density=False)
    q, _ = np.histogram(generated_noise.ravel(), bins=bins, range=hist_range, density=False)

    p = p.astype(np.float64) + epsilon
    q = q.astype(np.float64) + epsilon
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def evaluate_pair(real_image, generated_image, ct_image=None):
    values = {
        "ssim": ssim(real_image, generated_image),
        "psnr": psnr(real_image, generated_image),
        "mse": mse(real_image, generated_image),
        "mae": mae(real_image, generated_image),
    }

    if ct_image is not None:
        real_ct_noise = estimate_noise(ct_image)
        generated_noise = estimate_noise(generated_image)
        values["kl_noise"] = kl_divergence(real_ct_noise, generated_noise)

    return values


def evaluate_folders(real_dir, generated_dir, ct_dir=None, csv_path=None, valid_names=None):
    real_images = list_images(real_dir)
    generated_images = list_images(generated_dir)
    ct_images = list_images(ct_dir) if ct_dir else {}

    if valid_names is not None:
        common_names = [n for n in valid_names if n in real_images and n in generated_images]
    else:
        common_names = sorted(set(real_images) & set(generated_images))

    if not common_names:
        raise ValueError("No paired images found")

    rows = []
    for name in common_names:
        real_path = real_images[name]
        generated_path = generated_images[name]

        real_image = read_gray_image(real_path)
        target_size = (real_image.shape[1], real_image.shape[0])
        generated_image = read_gray_image(generated_path, size=target_size)

        ct_image = None
        has_ct_images = bool(ct_images)
        if name in ct_images:
            ct_image = read_gray_image(ct_images[name], size=target_size)

        row = {"name": name}
        row.update(evaluate_pair(real_image, generated_image, ct_image))
        if has_ct_images and "kl_noise" not in row:
            row["kl_noise"] = np.nan
        rows.append(row)

    metric_names = sorted({key for row in rows for key in row if key != "name"})
    summary = {}
    for metric in metric_names:
        values = np.array([row.get(metric, np.nan) for row in rows], dtype=np.float64)
        summary[f"mean_{metric}"] = float(np.nanmean(values))
        summary[f"median_{metric}"] = float(np.nanmedian(values))
        summary[f"std_{metric}"] = float(np.nanstd(values))
        summary[f"min_{metric}"] = float(np.nanmin(values))
        summary[f"max_{metric}"] = float(np.nanmax(values))

    if csv_path:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=["name", *metric_names])
            writer.writeheader()
            writer.writerows(rows)

            stat_names = ["mean", "median", "std", "min", "max"]
            for stat in stat_names:
                summary_row = {"name": stat}
                for metric in metric_names:
                    summary_row[metric] = summary.get(f"{stat}_{metric}", "")
                writer.writerow(summary_row)

    return rows, summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate CT-to-MRI generated images.")
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--ct-dir", default=None)
    parser.add_argument("--csv", default="result/evaluation_metrics.csv")
    parser.add_argument("--brain-threshold", type=float, default=0.10,
                        help="Min brain pixel ratio (0-1). Default 0.10 = 10%%")
    parser.add_argument("--test-patients", default=None, help="JSON file with test patient list")
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable auto-filtering, use all slices")
    args = parser.parse_args()

    if args.no_filter:
        valid_names = None
        excluded = []
    else:
        valid_names, excluded = filter_informative_slices(args.real_dir, args.brain_threshold)

    rows, summary = evaluate_folders(
        real_dir=args.real_dir,
        generated_dir=args.generated_dir,
        ct_dir=args.ct_dir,
        csv_path=args.csv,
        valid_names=valid_names,
    )

    print(f"Total paired slices: {len(rows) + len(excluded)}")
    if excluded:
        print(f"Excluded (brain content < {args.brain_threshold:.0%}): {len(excluded)} slices")
        print(f"  Range: {excluded[0]} -- {excluded[-1]}")
    print(f"Evaluated: {len(rows)} slices")
    print(f"{'Metric':<12} {'Mean':>10} {'Median':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 62)
    metric_names = sorted({k.split("_", 1)[1] for k in summary if k.startswith("mean_")})
    for m in metric_names:
        mean_v = summary.get(f"mean_{m}", float("nan"))
        med_v  = summary.get(f"median_{m}", float("nan"))
        std_v  = summary.get(f"std_{m}", float("nan"))
        min_v  = summary.get(f"min_{m}", float("nan"))
        max_v  = summary.get(f"max_{m}", float("nan"))
        print(f"{m:<12} {mean_v:10.4f} {med_v:10.4f} {std_v:10.4f} {min_v:10.4f} {max_v:10.4f}")
    print(f"\nCSV saved: {os.path.abspath(args.csv)}")


if __name__ == "__main__":
    main()
