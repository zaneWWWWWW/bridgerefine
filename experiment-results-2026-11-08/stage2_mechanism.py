"""E09/E10: input dependence and controlled prior degradation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def transform_prior(array: np.ndarray, condition: str, seed: int = 42) -> np.ndarray:
    if condition == "clean":
        return array.copy()
    if condition.startswith("blur"):
        sigma = float(condition.removeprefix("blur"))
        return np.stack(
            [cv2.GaussianBlur(image, (0, 0), sigmaX=sigma) for image in array]
        )
    if condition.startswith("noise"):
        sigma = float(condition.removeprefix("noise"))
        rng = np.random.default_rng(seed)
        return np.clip(array + rng.normal(0.0, sigma, array.shape), -1.0, 1.0)
    if condition.startswith("shift"):
        pixels = int(condition.removeprefix("shift"))
        shifted = np.full_like(array, -1.0)
        shifted[:, :, pixels:] = array[:, :, :-pixels]
        return shifted
    if condition == "local_patch_zero":
        corrupted = array.copy()
        center = corrupted.shape[-1] // 2
        half = 16
        corrupted[
            :,
            center - half : center + half,
            center - half : center + half,
        ] = -1.0
        return corrupted
    raise ValueError(f"Unknown condition: {condition}")


def cross_patient_prior(
    prior: np.ndarray, names: list[str]
) -> np.ndarray:
    patients = sorted({name.replace("\\", "/").split("/", 1)[0] for name in names})
    next_patient = {
        patient: patients[(index + 1) % len(patients)]
        for index, patient in enumerate(patients)
    }
    indices: dict[str, int] = {}
    for index, name in enumerate(names):
        patient = name.replace("\\", "/").split("/", 1)[0]
        indices.setdefault(patient, index)
    output = prior.copy()
    for index, name in enumerate(names):
        patient = name.replace("\\", "/").split("/", 1)[0]
        output[index] = prior[indices[next_patient[patient]]]
    return output


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


def mae_gray(real: np.ndarray, generated: np.ndarray) -> float:
    return float(np.mean(np.abs(real.astype(np.float32) - generated.astype(np.float32))))


def infer(
    model,
    ct: np.ndarray,
    prior: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(ct), batch_size):
            stop = min(start + batch_size, len(ct))
            output = model(
                torch.from_numpy(prior[start:stop]).float().unsqueeze(1).to(device),
                torch.from_numpy(ct[start:stop]).float().unsqueeze(1).to(device),
            )
            chunks.append(output.cpu().numpy()[:, 0])
    return np.concatenate(chunks)


def evaluate(
    real: np.ndarray,
    prediction: np.ndarray,
    names: list[str],
    device: torch.device,
    lpips_model,
) -> tuple[dict, list[dict]]:
    ssim_values = []
    psnr_values = []
    mae_values = []
    for index in range(len(names)):
        ssim_values.append(ssim_gray(real[index], prediction[index]))
        psnr_values.append(psnr_gray(real[index], prediction[index]))
        mae_values.append(mae_gray(real[index], prediction[index]))
    lpips_values = []
    with torch.inference_mode():
        for start in range(0, len(names), 64):
            stop = min(start + 64, len(names))
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
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, name in enumerate(names):
        patient = name.replace("\\", "/").split("/", 1)[0]
        grouped[patient]["ssim"].append(ssim_values[index])
        grouped[patient]["psnr"].append(psnr_values[index])
        grouped[patient]["mae"].append(mae_values[index])
        grouped[patient]["lpips"].append(lpips_values[index])
    patient_values = {
        metric: {
            patient: float(np.mean(grouped[patient][metric]))
            for patient in sorted(grouped)
        }
        for metric in ("ssim", "psnr", "mae", "lpips")
    }
    summary = {
        metric: {
            "mean": float(np.mean(list(values.values()))),
            "sd": float(np.std(list(values.values()), ddof=1)),
        }
        for metric, values in patient_values.items()
    }
    rows = [
        {
            "patient": patient,
            **{
                metric: values[patient]
                for metric, values in patient_values.items()
            },
        }
        for patient in sorted(grouped)
    ]
    return summary, rows


def train_one(
    model,
    ct: np.ndarray,
    mr: np.ndarray,
    prior: np.ndarray,
    run_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    with (run_dir / "train_log.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["epoch", "train_l1", "seconds"]
        )
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            start = time.time()
            model.train()
            order = np.random.permutation(len(ct))
            values = []
            for offset in range(0, len(order) - batch_size + 1, batch_size):
                batch = order[offset : offset + batch_size]
                ct_b = (
                    torch.from_numpy(ct[batch]).float().unsqueeze(1).to(device)
                )
                mr_b = (
                    torch.from_numpy(mr[batch]).float().unsqueeze(1).to(device)
                )
                prior_b = (
                    torch.from_numpy(prior[batch])
                    .float()
                    .unsqueeze(1)
                    .to(device)
                )
                if random.random() < 0.5:
                    ct_b, mr_b, prior_b = (
                        torch.flip(x, dims=[3]) for x in (ct_b, mr_b, prior_b)
                    )
                prediction = model(prior_b, ct_b)
                loss = F.l1_loss(prediction, mr_b)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                values.append(float(loss))
            elapsed = time.time() - start
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_l1": float(np.mean(values)),
                    "seconds": elapsed,
                }
            )
            handle.flush()
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                run_dir / "last.pt",
            )
            print(
                f"[{run_dir.name}] epoch {epoch}/{epochs} "
                f"loss={np.mean(values):.6f} {elapsed:.0f}s",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mode", choices=["fixed", "retrain"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = root / "experiments/controls/cache"
    names = [str(v) for v in load_json(cache / "names_test.json")]
    ct_test = np.load(cache / "ct_test.npy", mmap_mode="r")[:, 0].astype(np.float32)
    mr_test = np.load(cache / "mr_test.npy", mmap_mode="r")[:, 0].astype(np.float32)
    gan_prior_test = np.load(
        cache / "pix2pix_prior_test.npy", mmap_mode="r"
    )[:, 0].astype(np.float32)

    sys.path.insert(0, str(root))
    from other_model.BridgeGAN.model import RefinerGenerator

    device = torch.device("cuda")
    import lpips

    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    if args.mode == "fixed":
        checkpoint = torch.load(
            root / "experiments/gan_prior/last.pt",
            map_location=device,
            weights_only=False,
        )
        model = RefinerGenerator(in_ch=2, out_ch=1, base_ch=64).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        conditions = [
            ("clean", gan_prior_test, ct_test),
            ("blur0.5", transform_prior(gan_prior_test, "blur0.5"), ct_test),
            ("blur1.0", transform_prior(gan_prior_test, "blur1.0"), ct_test),
            ("blur2.0", transform_prior(gan_prior_test, "blur2.0"), ct_test),
            ("noise0.05", transform_prior(gan_prior_test, "noise0.05"), ct_test),
            ("noise0.10", transform_prior(gan_prior_test, "noise0.10"), ct_test),
            ("noise0.20", transform_prior(gan_prior_test, "noise0.20"), ct_test),
            ("shift2", transform_prior(gan_prior_test, "shift2"), ct_test),
            ("shift4", transform_prior(gan_prior_test, "shift4"), ct_test),
            ("shift8", transform_prior(gan_prior_test, "shift8"), ct_test),
            (
                "cross_patient_prior",
                cross_patient_prior(gan_prior_test, names),
                ct_test,
            ),
            (
                "zero_prior",
                np.full_like(gan_prior_test, -1.0),
                ct_test,
            ),
            (
                "zero_ct",
                gan_prior_test,
                np.full_like(ct_test, -1.0),
            ),
            (
                "local_patch_zero",
                transform_prior(gan_prior_test, "local_patch_zero"),
                ct_test,
            ),
        ]
        summaries = {}
        all_rows = []
        for condition, prior, ct in conditions:
            print(f"fixed mechanism: {condition}", flush=True)
            prediction = infer(model, ct, prior, device)
            summary, rows = evaluate(
                mr_test, prediction, names, device, lpips_model
            )
            summaries[condition] = summary
            for row in rows:
                all_rows.append({"condition": condition, **row})
        (out_dir / "fixed_mechanism.json").write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with (out_dir / "fixed_mechanism_patient.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["condition", "patient", "ssim", "psnr", "mae", "lpips"],
            )
            writer.writeheader()
            writer.writerows(all_rows)
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
        return

    ct_train = np.load(cache / "ct_train.npy", mmap_mode="r")[:, 0].astype(np.float32)
    mr_train = np.load(cache / "mr_train.npy", mmap_mode="r")[:, 0].astype(np.float32)
    gan_prior_train = np.load(
        cache / "pix2pix_prior_train.npy", mmap_mode="r"
    )[:, 0].astype(np.float32)
    conditions = ["blur1.0", "noise0.10", "shift4"]
    summaries = {}
    for condition in conditions:
        print(f"matched retraining: {condition}", flush=True)
        prior_train = transform_prior(gan_prior_train, condition, args.seed)
        prior_test = transform_prior(gan_prior_test, condition, args.seed)
        run_dir = out_dir / condition.replace(".", "_")
        run_dir.mkdir(parents=True, exist_ok=True)
        model = RefinerGenerator(in_ch=2, out_ch=1, base_ch=64).to(device)
        train_one(
            model,
            ct_train,
            mr_train,
            prior_train,
            run_dir,
            args.seed,
            args.epochs,
            args.batch_size,
            device,
        )
        model.eval()
        prediction = infer(model, ct_test, prior_test, device)
        summary, rows = evaluate(
            mr_test, prediction, names, device, lpips_model
        )
        summaries[condition] = summary
        with (run_dir / "metrics_patient.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["patient", "ssim", "psnr", "mae", "lpips"]
            )
            writer.writeheader()
            writer.writerows(rows)
        torch.cuda.empty_cache()
    (out_dir / "matched_retraining.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
