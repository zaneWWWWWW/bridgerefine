# -*- coding: utf-8 -*-
"""
Batch preprocessing: NIfTI volumes → 2D paired CT/MRI slices.
Processes all 180 patients from ../brain/ into ../data_slices/
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import SimpleITK as sitk
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def n4_bias_correction(image):
    image = sitk.Cast(image, sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 30])
    return corrector.Execute(image, mask)


def auto_detect_ct_window(img_data):
    brain_tissue_range = (20, 100)
    hist, bins = np.histogram(img_data, bins=256, range=brain_tissue_range)
    peak = bins[np.argmax(hist)]
    if peak < 40:
        return 35, 110
    elif 40 <= peak < 60:
        return 40, 80
    else:
        return 80, 200


def auto_detect_mr_window(img_data):
    p1 = np.percentile(img_data, 2)
    p99 = np.percentile(img_data, 98)
    return (p1 + p99) / 2, p99 - p1


def apply_clahe(img, modality):
    params = {'CT': {'clipLimit': 1.5, 'tileGridSize': (4, 4)},
              'MR': {'clipLimit': 2.0, 'tileGridSize': (8, 8)}}
    clahe = cv2.createCLAHE(**params[modality])
    return clahe.apply(np.uint8(img))


def is_blank_image(img, threshold=0.01):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    valid_ratio = (img.size - hist[0][0]) / img.size
    return valid_ratio < threshold


def process_slice(slice_data, slice_num, wc, ws, modality, enable_clahe):
    low, high = wc - ws / 2, wc + ws / 2

    if np.max(slice_data) < 1:
        return None, slice_num

    clipped = np.clip(slice_data, low, high)
    if np.all(clipped == 0):
        return None, slice_num

    normalized = np.interp(clipped, [low, high], [0.0, 255.0]).astype(np.float32)

    if enable_clahe:
        normalized = apply_clahe(normalized, modality)

    gamma = 0.9 if modality == 'CT' else 1.1
    normalized = np.power(normalized / 255.0, gamma) * 255
    normalized = np.uint8(np.clip(normalized, 0, 255))

    if is_blank_image(normalized, threshold=0.005):
        return None, slice_num

    return normalized, slice_num


def process_patient(patient_dir, output_root, image_size):
    patient_id = patient_dir.name
    ct_path = patient_dir / "ct.nii.gz"
    mr_path = patient_dir / "mr.nii.gz"

    if not ct_path.exists() or not mr_path.exists():
        logger.warning(f"Skipping {patient_id}: missing CT or MR")
        return 0

    try:
        ct_img = sitk.ReadImage(str(ct_path))
        mr_img = sitk.ReadImage(str(mr_path))
        mr_img = n4_bias_correction(mr_img)

        ct_data = sitk.GetArrayFromImage(ct_img)
        mr_data = sitk.GetArrayFromImage(mr_img)

        if ct_data.shape != mr_data.shape:
            logger.warning(f"Skipping {patient_id}: shape mismatch CT{ct_data.shape} vs MR{mr_data.shape}")
            return 0

        ct_wc, ct_ws = auto_detect_ct_window(ct_data)
        mr_wc, mr_ws = auto_detect_mr_window(mr_data)

        ct_out = output_root / patient_id / "ct"
        mr_out = output_root / patient_id / "mr"
        ct_out.mkdir(parents=True, exist_ok=True)
        mr_out.mkdir(parents=True, exist_ok=True)

        saved = 0
        for i in range(ct_data.shape[0]):
            ct_slice, _ = process_slice(ct_data[i], i, ct_wc, ct_ws, 'CT', True)
            mr_slice, _ = process_slice(mr_data[i], i, mr_wc, mr_ws, 'MR', True)

            if ct_slice is None or mr_slice is None:
                continue

            ct_slice = cv2.resize(ct_slice, (image_size, image_size), interpolation=cv2.INTER_AREA)
            mr_slice = cv2.resize(mr_slice, (image_size, image_size), interpolation=cv2.INTER_AREA)

            cv2.imwrite(str(ct_out / f"{i:04d}.png"), ct_slice)
            cv2.imwrite(str(mr_out / f"{i:04d}.png"), mr_slice)
            saved += 1

        return saved

    except Exception as e:
        logger.error(f"Error processing {patient_id}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Preprocess NIfTI volumes to 2D slices")
    parser.add_argument("--data-root", default="../brain")
    parser.add_argument("--output-root", default="../data_slices")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    patients = sorted(d for d in data_root.iterdir() if d.is_dir() and not d.name.startswith("_") and d.name != "overview")
    logger.info(f"Found {len(patients)} patients")

    total_slices = 0
    for i, patient_dir in enumerate(patients):
        n = process_patient(patient_dir, output_root, args.image_size)
        total_slices += n
        if (i + 1) % 20 == 0:
            logger.info(f"Progress: {i+1}/{len(patients)} patients, {total_slices} slices so far")

    logger.info(f"Done. Total paired slices: {total_slices} from {len(patients)} patients")


if __name__ == "__main__":
    main()
