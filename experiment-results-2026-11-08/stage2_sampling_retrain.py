"""E12: match-retrain refiners for bridge sampling configurations."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = root / "experiments/controls/cache"
    step_dir = root / "experiments/step_ablation"

    sys.path.insert(0, str(root))
    from other_model.BridgeGAN.model import RefinerGenerator
    from selfrdb.bridge_model import DiffusionBridge
    from selfrdb.network import X0UNet

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    ct_train = np.load(cache / "ct_train.npy", mmap_mode="r")
    mr_train = np.load(cache / "mr_train.npy", mmap_mode="r")
    ct_test = np.load(cache / "ct_test.npy", mmap_mode="r")
    mr_test = np.load(cache / "mr_test.npy", mmap_mode="r")
    names = [str(v) for v in load_json(cache / "names_test.json")]

    generator = X0UNet(
        in_ch=3, out_ch=1, base_ch=64, ch_mult=(1, 2, 4)
    ).to(device)
    checkpoint = torch.load(
        root / "other_model/SelfRDB/selfrdb_best.pt",
        map_location=device,
        weights_only=False,
    )
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()
    bridge = DiffusionBridge(generator, timesteps=300, gamma=0.1, device=device)

    configs = {"2x2": (2, 2), "4x1": (4, 1)}
    summaries = {}
    for tag, (steps, recursions) in configs.items():
        run_dir = out_dir / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        train_path = step_dir / f"coarse_{tag}_train.npy"
        if train_path.exists():
            coarse_train = np.load(train_path, mmap_mode="r")
        else:
            chunks = []
            start = time.time()
            for offset in range(0, len(ct_train), args.batch_size):
                batch = torch.from_numpy(
                    np.asarray(ct_train[offset : offset + args.batch_size])
                ).to(device)
                with torch.inference_mode():
                    chunks.append(
                        bridge.sample_ddib(
                            batch,
                            num_steps=steps,
                            num_recursions=recursions,
                        )
                        .cpu()
                        .numpy()
                    )
            coarse_train = np.concatenate(chunks).astype(np.float32)
            np.save(train_path, coarse_train)
            print(
                f"{tag} train coarse saved shape={coarse_train.shape} "
                f"elapsed={time.time() - start:.0f}s",
                flush=True,
            )

        test_blob = torch.load(
            step_dir / f"test_coarse_{tag}.pt",
            map_location="cpu",
            weights_only=False,
        )
        if [str(v) for v in test_blob["names"]] != names:
            raise ValueError(f"{tag} test names mismatch")
        coarse_test = np.asarray(test_blob["coarse"], dtype=np.float32)

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        model = RefinerGenerator(in_ch=2, out_ch=1, base_ch=64).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
        with (run_dir / "train_log.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["epoch", "train_l1", "seconds"]
            )
            writer.writeheader()
            for epoch in range(1, args.epochs + 1):
                start = time.time()
                model.train()
                order = np.random.permutation(len(ct_train))
                losses = []
                for offset in range(
                    0, len(order) - args.batch_size + 1, args.batch_size
                ):
                    batch = order[offset : offset + args.batch_size]
                    ct = torch.from_numpy(np.asarray(ct_train[batch])).to(device)
                    mr = torch.from_numpy(np.asarray(mr_train[batch])).to(device)
                    coarse = torch.from_numpy(
                        np.asarray(coarse_train[batch])
                    ).to(device)
                    if random.random() < 0.5:
                        ct, mr, coarse = (
                            torch.flip(x, dims=[3]) for x in (ct, mr, coarse)
                        )
                    prediction = model(coarse, ct)
                    loss = F.l1_loss(prediction, mr)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss))
                elapsed = time.time() - start
                writer.writerow(
                    {
                        "epoch": epoch,
                        "train_l1": float(np.mean(losses)),
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
                    f"[{tag}] epoch {epoch}/{args.epochs} "
                    f"loss={np.mean(losses):.6f} {elapsed:.0f}s",
                    flush=True,
                )

        model.eval()
        predictions = []
        with torch.inference_mode():
            for offset in range(0, len(ct_test), args.batch_size):
                stop = offset + args.batch_size
                predictions.append(
                    model(
                        torch.from_numpy(coarse_test[offset:stop]).to(device),
                        torch.from_numpy(np.asarray(ct_test[offset:stop])).to(device),
                    )
                    .cpu()
                    .numpy()[:, 0]
                )
        prediction = np.concatenate(predictions)
        grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for index, name in enumerate(names):
            grouped[name.replace("\\", "/").split("/", 1)[0]].append(
                (
                    ssim_gray(mr_test[index, 0], prediction[index]),
                    psnr_gray(mr_test[index, 0], prediction[index]),
                )
            )
        patients = sorted(grouped)
        ssim = np.asarray([np.mean([v[0] for v in grouped[p]]) for p in patients])
        psnr = np.asarray([np.mean([v[1] for v in grouped[p]]) for p in patients])
        result = {
            "configuration": tag,
            "steps": steps,
            "recursions": recursions,
            "ssim_mean": float(ssim.mean()),
            "ssim_sd": float(ssim.std(ddof=1)),
            "psnr_mean": float(psnr.mean()),
            "psnr_sd": float(psnr.std(ddof=1)),
            "per_patient_ssim": {
                patient: float(value) for patient, value in zip(patients, ssim)
            },
        }
        (run_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        np.save(run_dir / "test_predictions.npy", prediction)
        summaries[tag] = result
        del model
        torch.cuda.empty_cache()

    (out_dir / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
