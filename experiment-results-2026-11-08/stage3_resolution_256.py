"""E13: 256x256 CT-only and conditional-GAN-plus-refiner comparison."""

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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class SimpleUNet(nn.Module):
    def __init__(self, base_ch=64):
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


def evaluate(prediction, real, names):
    grouped = defaultdict(list)
    for index, name in enumerate(names):
        grouped[name.replace("\\", "/").split("/", 1)[0]].append(
            (
                ssim_gray(real[index], prediction[index]),
                psnr_gray(real[index], prediction[index]),
            )
        )
    patients = sorted(grouped)
    ssim = np.asarray([np.mean([v[0] for v in grouped[p]]) for p in patients])
    psnr = np.asarray([np.mean([v[1] for v in grouped[p]]) for p in patients])
    return {
        "ssim_mean": float(ssim.mean()),
        "ssim_sd": float(ssim.std(ddof=1)),
        "psnr_mean": float(psnr.mean()),
        "psnr_sd": float(psnr.std(ddof=1)),
        "per_patient_ssim": {
            patient: float(value) for patient, value in zip(patients, ssim)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", choices=["ct_only", "gan"], required=True
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out.resolve() / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(root))
    from baselines.baselines_gan import NLayerDiscriminator, ResnetGenerator, lsgan
    from other_model.BridgeGAN.model import RefinerGenerator
    from selfrdb.dataset import PairedCTMRIDataset

    with (root / "checkpoints/selfrdb/patient_split.json").open(
        "r", encoding="utf-8-sig"
    ) as handle:
        split = json.load(handle)
    dataset_root = root / "data_slices_256"
    datasets = {
        name: PairedCTMRIDataset(
            str(dataset_root),
            image_size=256,
            augment=name == "train",
            patient_ids=split[name],
        )
        for name in ("train", "val", "test")
    }
    cache = root / "experiments/controls/cache"
    for split_name in ("train", "val", "test"):
        with (cache / f"names_{split_name}.json").open(
            "r", encoding="utf-8"
        ) as handle:
            allowed_names = {str(name) for name in json.load(handle)}
        datasets[split_name].pairs = [
            pair for pair in datasets[split_name].pairs
            if str(pair[2]) in allowed_names
        ]
    loader_kwargs = {
        "num_workers": 4,
        "persistent_workers": True,
        "prefetch_factor": 4,
        "pin_memory": True,
    }
    batch_size = args.batch_size or (24 if args.experiment == "ct_only" else 4)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            **loader_kwargs,
        ),
    }
    device = torch.device("cuda")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    if args.experiment == "ct_only":
        model = SimpleUNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
        best_val = float("inf")
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            model.train()
            train_loss = 0.0
            train_count = 0
            for ct, mr, _ in loaders["train"]:
                ct = ct.to(device)
                mr = mr.to(device)
                prediction = model(ct)
                loss = F.l1_loss(prediction, mr)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(mr)
                train_count += len(mr)
            model.eval()
            val_sum = 0.0
            val_count = 0
            with torch.inference_mode():
                for ct, mr, _ in loaders["val"]:
                    ct = ct.to(device)
                    mr = mr.to(device)
                    val_sum += F.l1_loss(
                        model(ct), mr, reduction="sum"
                    ).item()
                    val_count += mr.numel()
            val_l1 = val_sum / max(val_count, 1)
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_l1": val_l1,
                },
                out_dir / "last.pt",
            )
            if val_l1 < best_val:
                best_val = val_l1
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "val_l1": val_l1,
                    },
                    out_dir / "best.pt",
                )
            print(
                f"[ct256] epoch {epoch}/{args.epochs} "
                f"train={train_loss / max(train_count, 1):.6f} "
                f"val={val_l1:.6f} {time.time() - start:.0f}s",
                flush=True,
            )
        checkpoint = torch.load(out_dir / "best.pt", map_location=device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        predictions, reals, names = [], [], []
        with torch.inference_mode():
            for ct, mr, name in loaders["test"]:
                predictions.append(model(ct.to(device)).cpu().numpy()[:, 0])
                reals.append(mr.numpy()[:, 0])
                names.extend(list(name))
        prediction = np.concatenate(predictions)
        real = np.concatenate(reals)
        result = {
            "experiment": "ct_only_256",
            "parameters": sum(p.numel() for p in model.parameters()),
            "best_epoch": checkpoint["epoch"],
            "best_val_l1": checkpoint["val_l1"],
            **evaluate(prediction, real, names),
        }
    else:
        generator = ResnetGenerator(1, 1, 64, 9).to(device)
        discriminator = NLayerDiscriminator(2, 64, 3).to(device)
        optimizer_g = torch.optim.Adam(
            generator.parameters(), lr=2e-4, betas=(0.5, 0.999)
        )
        optimizer_d = torch.optim.Adam(
            discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999)
        )
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            generator.train()
            discriminator.train()
            values_g, values_d = [], []
            for ct, mr, _ in loaders["train"]:
                ct = ct.to(device)
                mr = mr.to(device)
                fake = generator(ct)
                optimizer_d.zero_grad(set_to_none=True)
                loss_d = 0.5 * (
                    lsgan(discriminator(torch.cat([ct, mr], 1)), True)
                    + lsgan(
                        discriminator(torch.cat([ct, fake.detach()], 1)), False
                    )
                )
                loss_d.backward()
                optimizer_d.step()
                optimizer_g.zero_grad(set_to_none=True)
                loss_g = (
                    lsgan(discriminator(torch.cat([ct, fake], 1)), True)
                    + 100.0 * F.l1_loss(fake, mr)
                )
                loss_g.backward()
                optimizer_g.step()
                values_g.append(float(loss_g))
                values_d.append(float(loss_d))
            torch.save(
                {
                    "epoch": epoch,
                    "model": generator.state_dict(),
                    "optimizer": optimizer_g.state_dict(),
                },
                out_dir / "gan_last.pt",
            )
            print(
                f"[gan256] epoch {epoch}/{args.epochs} "
                f"G={np.mean(values_g):.4f} D={np.mean(values_d):.4f} "
                f"{time.time() - start:.0f}s",
                flush=True,
            )

        generator.eval()
        prior_loaders = {
            split_name: DataLoader(
                datasets[split_name],
                batch_size=batch_size,
                shuffle=False,
                **loader_kwargs,
            )
            for split_name in ("train", "val", "test")
        }
        train_priors = {}
        val_priors = {}
        test_priors = {}
        with torch.inference_mode():
            for split_name, loader, store in (
                ("train", prior_loaders["train"], train_priors),
                ("val", prior_loaders["val"], val_priors),
                ("test", prior_loaders["test"], test_priors),
            ):
                for ct, _, names_batch in loader:
                    outputs = generator(ct.to(device)).cpu().numpy()
                    for name, output in zip(names_batch, outputs):
                        store[str(name)] = output
                np.save(
                    out_dir / f"gan_prior_{split_name}.npy",
                    np.stack([store[name] for name in store]),
                )

        refiner = RefinerGenerator(in_ch=2, out_ch=1, base_ch=64).to(device)
        optimizer = torch.optim.Adam(refiner.parameters(), lr=2e-4, betas=(0.5, 0.999))
        best_val = float("inf")
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            refiner.train()
            values = []
            for ct, mr, names_batch in loaders["train"]:
                prior = torch.from_numpy(
                    np.stack([train_priors[str(name)] for name in names_batch])
                ).to(device)
                ct = ct.to(device)
                mr = mr.to(device)
                prediction = refiner(prior, ct)
                loss = F.l1_loss(prediction, mr)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                values.append(float(loss))
            refiner.eval()
            val_sum = 0.0
            val_count = 0
            with torch.inference_mode():
                for ct, mr, names_batch in loaders["val"]:
                    prior = torch.from_numpy(
                        np.stack([val_priors[str(name)] for name in names_batch])
                    ).to(device)
                    ct = ct.to(device)
                    mr = mr.to(device)
                    val_sum += F.l1_loss(
                        refiner(prior, ct), mr, reduction="sum"
                    ).item()
                    val_count += mr.numel()
            val_l1 = val_sum / max(val_count, 1)
            torch.save(
                {
                    "epoch": epoch,
                    "model": refiner.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_l1": val_l1,
                },
                out_dir / "refiner_last.pt",
            )
            if val_l1 < best_val:
                best_val = val_l1
                torch.save(
                    {
                        "epoch": epoch,
                        "model": refiner.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "val_l1": val_l1,
                    },
                    out_dir / "refiner_best.pt",
                )
            print(
                f"[gan-refiner-256] epoch {epoch}/{args.epochs} "
                f"train={np.mean(values):.6f} val={val_l1:.6f} "
                f"{time.time() - start:.0f}s",
                flush=True,
            )
        checkpoint = torch.load(out_dir / "refiner_best.pt", map_location=device)
        refiner.load_state_dict(checkpoint["model"])
        refiner.eval()
        predictions, reals, names = [], [], []
        with torch.inference_mode():
            for ct, mr, names_batch in loaders["test"]:
                prior = torch.from_numpy(
                    np.stack([test_priors[str(name)] for name in names_batch])
                ).to(device)
                predictions.append(
                    refiner(prior, ct.to(device)).cpu().numpy()[:, 0]
                )
                reals.append(mr.numpy()[:, 0])
                names.extend([str(name) for name in names_batch])
        prediction = np.concatenate(predictions)
        real = np.concatenate(reals)
        result = {
            "experiment": "gan_prior_refiner_256",
            "parameters": sum(p.numel() for p in generator.parameters())
            + sum(p.numel() for p in refiner.parameters()),
            "pretrain_epochs": args.epochs,
            "refiner_best_epoch": checkpoint["epoch"],
            "refiner_best_val_l1": checkpoint["val_l1"],
            **evaluate(prediction, real, names),
        }
    np.save(out_dir / "test_predictions.npy", prediction)
    with (out_dir / "metrics_patient.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient", "ssim"])
        writer.writeheader()
        writer.writerows(
            [
                {"patient": patient, "ssim": value}
                for patient, value in result["per_patient_ssim"].items()
            ]
        )
    (out_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
