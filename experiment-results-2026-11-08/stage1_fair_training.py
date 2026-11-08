"""Fair CT-only convergence controls and joint GAN/refiner fine-tuning."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


_ARRAYS = {}


def open_array(path: str):
    array = _ARRAYS.get(path)
    if array is None:
        array = np.load(path, mmap_mode="r")
        _ARRAYS[path] = array
    return array


class SingleUNet(nn.Module):
    def __init__(self, base_ch: int):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, base_ch, 7, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(base_ch),
            nn.ReLU(True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(base_ch * 2),
            nn.ReLU(True),
        )
        self.res = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(
                        base_ch * 2,
                        base_ch * 2,
                        3,
                        padding=1,
                        padding_mode="reflect",
                    ),
                    nn.InstanceNorm2d(base_ch * 2),
                    nn.ReLU(True),
                    nn.Conv2d(
                        base_ch * 2,
                        base_ch * 2,
                        3,
                        padding=1,
                        padding_mode="reflect",
                    ),
                    nn.InstanceNorm2d(base_ch * 2),
                )
                for _ in range(4)
            ]
        )
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
            nn.InstanceNorm2d(base_ch),
            nn.ReLU(True),
        )
        self.out = nn.Sequential(
            nn.Conv2d(base_ch, 1, 7, padding=3, padding_mode="reflect"),
            nn.Tanh(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        residual = e2
        for block in self.res:
            residual = residual + block(residual)
        return self.out(self.dec1(residual))


class CachedPairs(Dataset):
    def __init__(self, root: Path, split: str, augment: bool = False):
        cache = root / "experiments/controls/cache"
        self.ct_path = str(cache / f"ct_{split}.npy")
        self.mr_path = str(cache / f"mr_{split}.npy")
        with (cache / f"names_{split}.json").open(
            "r", encoding="utf-8"
        ) as handle:
            self.names = [str(name) for name in json.load(handle)]
        self.augment = augment
        if len(open_array(self.ct_path)) != len(self.names):
            raise ValueError(f"CT/name mismatch for split {split}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        ct = np.array(open_array(self.ct_path)[index], copy=True)
        mr = np.array(open_array(self.mr_path)[index], copy=True)
        if self.augment and random.random() < 0.5:
            ct = np.flip(ct, axis=2).copy()
            mr = np.flip(mr, axis=2).copy()
        return (
            torch.from_numpy(ct).float(),
            torch.from_numpy(mr).float(),
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
    numerator = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sx + sy + c2)
    return float(np.mean(numerator / (denominator + 1e-12)))


def psnr_gray(real: np.ndarray, generated: np.ndarray) -> float:
    real01 = (real.astype(np.float32) + 1.0) / 2.0
    generated01 = (generated.astype(np.float32) + 1.0) / 2.0
    mse = float(np.mean((real01 - generated01) ** 2))
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def forward_batch(mode: str, models, batch):
    ct, mr, _ = batch
    ct = ct.to("cuda", non_blocking=True)
    mr = mr.to("cuda", non_blocking=True)
    if mode == "joint_gan":
        generator, refiner = models
        prediction = refiner(generator(ct), ct)
    else:
        prediction = models[0](ct)
    return prediction, mr


def evaluate(models, mode: str, loader, device: torch.device):
    predictions, reals = [], []
    with torch.inference_mode():
        for batch in loader:
            prediction, mr = forward_batch(mode, models, batch)
            predictions.append(prediction.cpu().numpy()[:, 0])
            reals.append(mr.cpu().numpy()[:, 0])
    predictions = np.concatenate(predictions)
    reals = np.concatenate(reals)
    ssim_values = []
    psnr_values = []
    for index in range(len(predictions)):
        ssim_values.append(ssim_gray(reals[index], predictions[index]))
        psnr_values.append(psnr_gray(reals[index], predictions[index]))
    return predictions, reals, np.asarray(ssim_values), np.asarray(psnr_values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=["ct_long", "ct_strong", "joint_gan"],
        required=True,
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    run_dir = args.out.resolve() / args.experiment
    run_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(root))
    from baselines.baselines_gan import ResnetGenerator
    from other_model.BridgeGAN.model import RefinerGenerator

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    if args.experiment == "ct_long":
        models = [SingleUNet(base_ch=64).to(device)]
        epochs = args.epochs or 100
        patience = args.patience or 12
        batch_size = args.batch_size or 64
    elif args.experiment == "ct_strong":
        models = [SingleUNet(base_ch=224).to(device)]
        epochs = args.epochs or 50
        patience = args.patience or 10
        batch_size = args.batch_size or 16
    else:
        generator = ResnetGenerator(1, 1, 64, 9).to(device)
        generator_checkpoint = torch.load(
            root / "experiments/baselines/pix2pix/last.pt",
            map_location=device,
            weights_only=False,
        )
        generator.load_state_dict(generator_checkpoint["model"])
        refiner = RefinerGenerator(in_ch=2, out_ch=1, base_ch=64).to(device)
        refiner_checkpoint = torch.load(
            root / "experiments/gan_prior/last.pt",
            map_location=device,
            weights_only=False,
        )
        refiner.load_state_dict(refiner_checkpoint["model"])
        models = [generator, refiner]
        epochs = args.epochs or 20
        patience = args.patience or 6
        batch_size = args.batch_size or 16

    train_dataset = CachedPairs(root, "train", augment=True)
    val_dataset = CachedPairs(root, "val", augment=False)
    test_dataset = CachedPairs(root, "test", augment=False)
    loader_options = {
        "num_workers": 4,
        "persistent_workers": True,
        "prefetch_factor": 4,
        "pin_memory": True,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_options,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_options,
    )

    parameters = [parameter for model in models for parameter in model.parameters()]
    optimizer = torch.optim.Adam(parameters, lr=2e-4, betas=(0.5, 0.999))
    best_val = float("inf")
    best_epoch = 0
    since_improvement = 0
    start_epoch = 1
    if args.resume:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        for model, state in zip(models, resume["models"]):
            model.load_state_dict(state)
        if "optimizer" in resume:
            optimizer.load_state_dict(resume["optimizer"])
        best_val = float(resume.get("val_l1", float("inf")))
        best_epoch = int(resume.get("epoch", 0))
        start_epoch = best_epoch + 1
        print(
            f"[{args.experiment}] resumed from {args.resume} "
            f"at epoch {start_epoch}, best_val={best_val:.6f}",
            flush=True,
        )
    curve = []
    log_path = run_dir / "train_log.csv"

    with log_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_l1", "val_l1", "seconds", "best"],
        )
        writer.writeheader()
        for epoch in range(start_epoch, epochs + 1):
            start = time.time()
            for model in models:
                model.train()
            train_sum = 0.0
            train_count = 0
            for batch in train_loader:
                prediction, target = forward_batch(args.experiment, models, batch)
                loss = F.l1_loss(prediction, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                train_sum += loss.item() * len(target)
                train_count += len(target)

            for model in models:
                model.eval()
            val_sum = 0.0
            val_count = 0
            with torch.inference_mode():
                for batch in val_loader:
                    prediction, target = forward_batch(
                        args.experiment, models, batch
                    )
                    val_sum += F.l1_loss(
                        prediction, target, reduction="sum"
                    ).item()
                    val_count += target.numel()
            train_l1 = train_sum / max(train_count, 1)
            val_l1 = val_sum / max(val_count, 1)
            elapsed = time.time() - start
            improved = val_l1 < best_val
            if improved:
                best_val = val_l1
                best_epoch = epoch
                since_improvement = 0
                state = {
                    "epoch": epoch,
                    "val_l1": val_l1,
                    "models": [model.state_dict() for model in models],
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(state, run_dir / "best.pt")
            else:
                since_improvement += 1
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
            torch.save(
                {
                    "epoch": epoch,
                    "val_l1": best_val,
                    "best_epoch": best_epoch,
                    "since_improvement": since_improvement,
                    "models": [model.state_dict() for model in models],
                    "optimizer": optimizer.state_dict(),
                },
                run_dir / "last.pt",
            )
            print(
                f"[{args.experiment}] epoch {epoch}/{epochs} "
                f"train={train_l1:.6f} val={val_l1:.6f} "
                f"{elapsed:.0f}s",
                flush=True,
            )
            if since_improvement >= patience:
                print(
                    f"[{args.experiment}] early stop at epoch {epoch}; "
                    f"best epoch {best_epoch}",
                    flush=True,
                )
                break

    best = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    for model, state in zip(models, best["models"]):
        model.load_state_dict(state)
        model.eval()
    predictions, reals, ssim_values, psnr_values = evaluate(
        models, args.experiment, test_loader, device
    )
    np.save(run_dir / "test_predictions.npy", predictions)
    names = test_dataset.names
    by_patient = {}
    for index, name in enumerate(names):
        patient = name.replace("\\", "/").split("/", 1)[0]
        by_patient.setdefault(patient, []).append(
            (float(ssim_values[index]), float(psnr_values[index]))
        )
    patients = sorted(by_patient)
    result = {
        "experiment": args.experiment,
        "seed": args.seed,
        "parameters": int(sum(parameter.numel() for parameter in parameters)),
        "best_epoch": int(best["epoch"]),
        "best_val_l1": float(best["val_l1"]),
        "curve": curve,
        "test_ssim_mean": float(
            np.mean([np.mean([v[0] for v in by_patient[p]]) for p in patients])
        ),
        "test_psnr_mean": float(
            np.mean([np.mean([v[1] for v in by_patient[p]]) for p in patients])
        ),
        "per_patient_ssim": {
            patient: float(np.mean([v[0] for v in by_patient[patient]]))
            for patient in patients
        },
    }
    (run_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
