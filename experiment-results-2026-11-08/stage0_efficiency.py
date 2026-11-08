"""Benchmark complete inference pipelines on one GPU with aligned settings."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


class SimpleUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 7, padding=3, padding_mode="reflect"),
            torch.nn.InstanceNorm2d(64),
            torch.nn.ReLU(True),
        )
        self.enc2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
            torch.nn.InstanceNorm2d(128),
            torch.nn.ReLU(True),
        )
        self.res = torch.nn.Sequential(
            *[
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        128, 128, 3, padding=1, padding_mode="reflect"
                    ),
                    torch.nn.InstanceNorm2d(128),
                    torch.nn.ReLU(True),
                    torch.nn.Conv2d(
                        128, 128, 3, padding=1, padding_mode="reflect"
                    ),
                    torch.nn.InstanceNorm2d(128),
                )
                for _ in range(4)
            ]
        )
        self.dec1 = torch.nn.Sequential(
            torch.nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=False
            ),
            torch.nn.Conv2d(128, 64, 3, padding=1),
            torch.nn.InstanceNorm2d(64),
            torch.nn.ReLU(True),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Conv2d(64, 1, 7, padding=3, padding_mode="reflect"),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        residual = e2
        for block in self.res:
            residual = residual + block(residual)
        return self.out(self.dec1(residual))


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_benchmark(
    fn,
    ct_inputs: list[torch.Tensor],
    warmup: int,
    timed: int,
    repeats: int,
    device: torch.device,
) -> dict:
    for _ in range(warmup):
        fn(ct_inputs.pop(0))
        ct_inputs.append(ct_inputs[0])
    synchronize(device)

    times_by_repeat = []
    for _ in range(repeats):
        samples_ms = []
        for _ in range(timed):
            ct = ct_inputs.pop(0)
            ct_inputs.append(ct)
            start = time.perf_counter()
            fn(ct)
            synchronize(device)
            samples_ms.append((time.perf_counter() - start) * 1000.0)
        times_by_repeat.append(samples_ms)

    means = [statistics.mean(values) for values in times_by_repeat]
    all_values = [value for values in times_by_repeat for value in values]
    return {
        "repeat_mean_ms": means,
        "mean_ms": float(statistics.mean(means)),
        "sd_repeat_ms": float(statistics.stdev(means)),
        "p50_single_ms": float(np.percentile(all_values, 50)),
        "p95_single_ms": float(np.percentile(all_values, 95)),
        "samples": len(all_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--timed", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--method",
        choices=["all", "ct", "gan", "bridge2", "bridge10"],
        default="all",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.method == "all":
        all_rows = []
        for method in ("ct", "gan", "bridge2", "bridge10"):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--root",
                str(root),
                "--out",
                str(out_dir),
                "--warmup",
                str(args.warmup),
                "--timed",
                str(args.timed),
                "--repeats",
                str(args.repeats),
                "--method",
                method,
            ]
            print(f"launching isolated benchmark process: {method}", flush=True)
            subprocess.run(command, check=True)
            with (out_dir / f"efficiency_{method}.json").open(
                "r", encoding="utf-8"
            ) as handle:
                all_rows.extend(json.load(handle)["results"])
        with (out_dir / "efficiency.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        result = {
            "device": torch.cuda.get_device_name(torch.device("cuda")),
            "torch_version": torch.__version__,
            "input_size": [1, 1, 128, 128],
            "batch_size": 1,
            "precision": "float32",
            "warmup": args.warmup,
            "timed_per_repeat": args.timed,
            "repeats": args.repeats,
            "scope": (
                "GPU model inference only; each method ran in an isolated "
                "process. Preprocessing, input transfer, and prior transfer "
                "from host storage are excluded."
            ),
            "results": all_rows,
        }
        (out_dir / "efficiency.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    sys.path.insert(0, str(root))
    from baselines.baselines_gan import ResnetGenerator
    from other_model.BridgeGAN.model import RefinerGenerator
    from other_model.SelfRDB.bridge_model import DiffusionBridge
    from other_model.SelfRDB.network import X0UNet

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    ct_array = np.load(
        root / "experiments/controls/cache/ct_test.npy", mmap_mode="r"
    )
    ct_inputs = [
        torch.from_numpy(np.array(ct_array[index], copy=True))
        .unsqueeze(0)
        .to(device)
        for index in range(min(140, len(ct_array)))
    ]

    def load_refiner(path: Path, key: str = "model"):
        model = RefinerGenerator(in_ch=2, out_ch=1, base_ch=64).to(device)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint[key])
        return model.eval()

    def load_ct():
        model = SimpleUNet().to(device)
        checkpoint = torch.load(
            root / "other_model/phase2_CT_only_L1/best.pt",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        return model.eval()

    def load_gan():
        model = ResnetGenerator(1, 1, 64, 9).to(device)
        checkpoint = torch.load(
            root / "experiments/baselines/pix2pix/last.pt",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model"])
        return model.eval()

    def load_bridge():
        model = X0UNet(
            in_ch=3, out_ch=1, base_ch=64, ch_mult=(1, 2, 4)
        ).to(device)
        checkpoint = torch.load(
            root / "other_model/SelfRDB/selfrdb_best.pt",
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["generator"])
        model.eval()
        return DiffusionBridge(model, timesteps=300, gamma=0.1, device=device)

    rows = []

    def benchmark_method(
        label: str,
        fn,
        models: list[torch.nn.Module],
    ) -> None:
        print(f"benchmark: {label}", flush=True)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        base_allocated = torch.cuda.memory_allocated(device)
        timing = run_benchmark(
            fn,
            ct_inputs,
            args.warmup,
            args.timed,
            args.repeats,
            device,
        )
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        rows.append(
            {
                "method": label,
                "parameters": int(
                    sum(parameter.numel() for model in models for parameter in model.parameters())
                ),
                "mean_ms": timing["mean_ms"],
                "repeat_sd_ms": timing["sd_repeat_ms"],
                "p50_single_ms": timing["p50_single_ms"],
                "p95_single_ms": timing["p95_single_ms"],
                "additional_peak_allocated_mb": (
                    peak_allocated - base_allocated
                )
                / (1024**2),
                "peak_reserved_mb": peak_reserved / (1024**2),
            }
        )
        for model in models:
            del model
        torch.cuda.empty_cache()

    with torch.inference_mode():
        if args.method == "ct":
            ct_model = load_ct()
            benchmark_method(
                "CT-only U-Net",
                lambda ct: ct_model(ct),
                [ct_model],
            )
        elif args.method == "gan":
            gan_prior = load_gan()
            gan_refiner = load_refiner(root / "experiments/gan_prior/last.pt")
            benchmark_method(
                "Conditional GAN prior + refiner",
                lambda ct: gan_refiner(gan_prior(ct), ct),
                [gan_prior, gan_refiner],
            )
        elif args.method == "bridge2":
            bridge2 = load_bridge()
            bridge2_refiner = load_refiner(root / "experiments/coarse_2x1/last.pt")
            benchmark_method(
                "Local bridge 2x1 + refiner",
                lambda ct: bridge2_refiner(
                    bridge2.sample_ddib(ct, num_steps=2, num_recursions=1), ct
                ),
                [bridge2.model, bridge2_refiner],
            )
        elif args.method == "bridge10":
            bridge10 = load_bridge()
            bridge10_refiner = load_refiner(
                root / "other_model/BridgeGAN_L1/best.pt", key="refiner"
            )
            benchmark_method(
                "Local bridge 10x2 + refiner",
                lambda ct: bridge10_refiner(
                    bridge10.sample_ddib(ct, num_steps=10, num_recursions=2),
                    ct,
                ),
                [bridge10.model, bridge10_refiner],
            )

    per_method_json = out_dir / f"efficiency_{args.method}.json"
    per_method_csv = out_dir / f"efficiency_{args.method}.csv"
    with per_method_csv.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "input_size": [1, 1, 128, 128],
        "batch_size": 1,
        "precision": "float32",
        "warmup": args.warmup,
        "timed_per_repeat": args.timed,
        "repeats": args.repeats,
        "scope": (
            "GPU model inference only; preprocessing, input transfer, and "
            "cached prior transfer are excluded"
        ),
        "results": rows,
    }
    per_method_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
