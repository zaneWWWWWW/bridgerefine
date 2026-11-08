"""Train the common supervised refiner on official SelfRDB priors."""

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
from torch.utils.data import DataLoader, Dataset


_ARRAYS = {}


def open_array(path: str):
    array = _ARRAYS.get(path)
    if array is None:
        array = np.load(path, mmap_mode="r")
        _ARRAYS[path] = array
    return array


class PriorPairs(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        prior_dir: Path,
        augment: bool = False,
    ):
        cache = root / "experiments/controls/cache"
        self.ct_path = str(cache / f"ct_{split}.npy")
        self.mr_path = str(cache / f"mr_{split}.npy")
        self.prior_path = str(prior_dir / f"official_{split}.npy")
        with (prior_dir / f"official_{split}_names.json").open(
            "r", encoding="utf-8"
        ) as handle:
            self.names = [str(name) for name in json.load(handle)]
        self.augment = augment
        lengths = {
            len(open_array(self.ct_path)),
            len(open_array(self.mr_path)),
            len(open_array(self.prior_path)),
            len(self.names),
        }
        if len(lengths) != 1:
            raise ValueError(f"length mismatch for {split}: {lengths}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        ct = np.array(open_array(self.ct_path)[index], copy=True)
        mr = np.array(open_array(self.mr_path)[index], copy=True)
        prior = np.array(open_array(self.prior_path)[index], copy=True)
        if float(np.nanmin(prior)) >= 0 and float(np.nanmax(prior)) <= 1.01:
            prior = prior * 2.0 - 1.0
        if self.augment and random.random() < 0.5:
            ct = np.flip(ct, axis=2).copy()
            mr = np.flip(mr, axis=2).copy()
            prior = np.flip(prior, axis=2).copy()
        return (
            torch.from_numpy(ct).float(),
            torch.from_numpy(mr).float(),
            torch.from_numpy(prior).float(),
            index,
        )


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--prior-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args()
    root = args.root.resolve()
    prior_dir = args.prior_dir.resolve()
    run_dir = args.out.resolve() / f"seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(root))
    from other_model.BridgeGAN.model import RefinerGenerator

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    train_dataset = PriorPairs(root, "train", prior_dir, augment=True)
    val_dataset = PriorPairs(root, "val", prior_dir, augment=False)
    test_dataset = PriorPairs(root, "test", prior_dir, augment=False)
    loader_kwargs = {
        "num_workers": 4,
        "persistent_workers": True,
        "prefetch_factor": 4,
        "pin_memory": True,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    model = RefinerGenerator(in_ch=2, out_ch=1, base_ch=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
    best_val = float("inf")
    best_epoch = 0
    since_improvement = 0
    curve = []
    with (run_dir / "train_log.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_l1", "val_l1", "seconds", "best"],
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            model.train()
            train_sum = 0.0
            train_count = 0
            for ct, mr, prior, _ in train_loader:
                ct = ct.to(device, non_blocking=True)
                mr = mr.to(device, non_blocking=True)
                prior = prior.to(device, non_blocking=True)
                prediction = model(prior, ct)
                loss = F.l1_loss(prediction, mr)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                train_sum += loss.item() * len(mr)
                train_count += len(mr)

            model.eval()
            val_sum = 0.0
            val_count = 0
            with torch.inference_mode():
                for ct, mr, prior, _ in val_loader:
                    ct = ct.to(device, non_blocking=True)
                    mr = mr.to(device, non_blocking=True)
                    prior = prior.to(device, non_blocking=True)
                    prediction = model(prior, ct)
                    val_sum += F.l1_loss(
                        prediction, mr, reduction="sum"
                    ).item()
                    val_count += mr.numel()
            train_l1 = train_sum / max(train_count, 1)
            val_l1 = val_sum / max(val_count, 1)
            elapsed = time.time() - start
            improved = val_l1 < best_val
            if improved:
                best_val = val_l1
                best_epoch = epoch
                since_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "val_l1": val_l1,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    run_dir / "best.pt",
                )
            else:
                since_improvement += 1
            torch.save(
                {
                    "epoch": epoch,
                    "val_l1": best_val,
                    "best_epoch": best_epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                run_dir / "last.pt",
            )
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_l1": train_l1,
                    "val_l1": val_l1,
                    "seconds": elapsed,
                    "best": improved,
                }
            )
            handle.flush()
            curve.append(
                {
                    "epoch": epoch,
                    "train_l1": train_l1,
                    "val_l1": val_l1,
                    "seconds": elapsed,
                }
            )
            print(
                f"[official-refiner seed{args.seed}] epoch {epoch}/{args.epochs} "
                f"train={train_l1:.6f} val={val_l1:.6f} {elapsed:.0f}s",
                flush=True,
            )
            if since_improvement >= args.patience:
                break

    checkpoint = torch.load(
        run_dir / "best.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    predictions = []
    reals = []
    with torch.inference_mode():
        for ct, mr, prior, _ in test_loader:
            predictions.append(
                model(
                    prior.to(device, non_blocking=True),
                    ct.to(device, non_blocking=True),
                )
                .cpu()
                .numpy()[:, 0]
            )
            reals.append(mr.numpy()[:, 0])
    prediction = np.concatenate(predictions)
    real = np.concatenate(reals)
    np.save(run_dir / "test_predictions.npy", prediction)

    ssim_values = []
    psnr_values = []
    mae_values = []
    for index in range(len(prediction)):
        ssim_values.append(ssim_gray(real[index], prediction[index]))
        psnr_values.append(psnr_gray(real[index], prediction[index]))
        mae_values.append(mae_gray(real[index], prediction[index]))

    import lpips

    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    lpips_values = []
    with torch.inference_mode():
        for start in range(0, len(prediction), 64):
            stop = min(start + 64, len(prediction))
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
    for index, name in enumerate(test_dataset.names):
        patient = name.replace("\\", "/").split("/", 1)[0]
        grouped[patient]["ssim"].append(ssim_values[index])
        grouped[patient]["psnr"].append(psnr_values[index])
        grouped[patient]["mae"].append(mae_values[index])
        grouped[patient]["lpips"].append(lpips_values[index])
    patients = sorted(grouped)
    patient_values = {
        metric: {
            patient: float(np.mean(grouped[patient][metric]))
            for patient in patients
        }
        for metric in ("ssim", "psnr", "mae", "lpips")
    }
    patient_rows = []
    for patient in patients:
        patient_rows.append(
            {
                "seed": args.seed,
                "patient": patient,
                **{
                    metric: values[patient]
                    for metric, values in patient_values.items()
                },
            }
        )
    with (run_dir / "metrics_patient.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["seed", "patient", "ssim", "psnr", "mae", "lpips"],
        )
        writer.writeheader()
        writer.writerows(patient_rows)
    result = {
        "method": "Official SelfRDB prior + refiner",
        "seed": args.seed,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "best_epoch": int(checkpoint["epoch"]),
        "best_val_l1": float(checkpoint["val_l1"]),
        "curve": curve,
        **{
            f"{metric}_mean": float(np.mean(list(values.values())))
            for metric, values in patient_values.items()
        },
        **{
            f"{metric}_sd": float(np.std(list(values.values()), ddof=1))
            for metric, values in patient_values.items()
        },
        "per_patient_ssim": patient_values["ssim"],
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
