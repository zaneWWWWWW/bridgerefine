"""E16: isolated inference cost for official SelfRDB and its refiner."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch


def synchronize():
    torch.cuda.synchronize()


def benchmark(fn, inputs, warmup, timed, repeats):
    for index in range(warmup):
        fn(inputs[index % len(inputs)])
    synchronize()
    values = []
    for _ in range(repeats):
        samples = []
        for index in range(timed):
            input_tensor = inputs[index % len(inputs)]
            start = time.perf_counter()
            fn(input_tensor)
            synchronize()
            samples.append((time.perf_counter() - start) * 1000.0)
        values.append(samples)
    means = [statistics.mean(group) for group in values]
    all_values = [value for group in values for value in group]
    return {
        "mean_ms": float(statistics.mean(means)),
        "repeat_sd_ms": float(statistics.stdev(means)),
        "p50_single_ms": float(np.percentile(all_values, 50)),
        "p95_single_ms": float(np.percentile(all_values, 95)),
        "samples": len(all_values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--refiner", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--timed", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "official_selfrdb"))

    from main import BridgeRunner
    from other_model.BridgeGAN.model import RefinerGenerator

    device = torch.device("cuda")
    runner = BridgeRunner.load_from_checkpoint(
        args.checkpoint, map_location=device
    ).to(device).eval()
    refiner = RefinerGenerator(in_ch=2, out_ch=1, base_ch=64).to(device)
    refiner_checkpoint = torch.load(
        args.refiner, map_location=device, weights_only=False
    )
    refiner.load_state_dict(refiner_checkpoint["model"])
    refiner.eval()
    ct = np.load(
        root / "experiments/controls/cache/ct_test.npy", mmap_mode="r"
    )
    inputs = [
        torch.from_numpy(np.array(ct[index], copy=True)).unsqueeze(0).to(device)
        for index in range(140)
    ]
    results = []
    with torch.inference_mode():
        def prior_only(input_tensor):
            runner.diffusion.sample_x0(input_tensor, runner.generator)

        def full_pipeline(input_tensor):
            prior = runner.diffusion.sample_x0(input_tensor, runner.generator)
            refiner(prior, input_tensor)

        for name, fn, model in (
            ("Official SelfRDB prior", prior_only, runner),
            ("Official SelfRDB prior + refiner", full_pipeline, None),
        ):
            torch.cuda.reset_peak_memory_stats(device)
            base = torch.cuda.memory_allocated(device)
            timing = benchmark(
                fn, inputs, args.warmup, args.timed, args.repeats
            )
            parameters = sum(p.numel() for p in runner.parameters())
            if "refiner" in name:
                parameters += sum(p.numel() for p in refiner.parameters())
            results.append(
                {
                    "method": name,
                    "parameters": int(parameters),
                    **timing,
                    "additional_peak_allocated_mb": (
                        torch.cuda.max_memory_allocated(device) - base
                    )
                    / (1024**2),
                    "peak_reserved_mb": torch.cuda.max_memory_reserved(device)
                    / (1024**2),
                }
            )
    with (out_dir / "official_efficiency.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    payload = {
        "device": torch.cuda.get_device_name(device),
        "input_size": [1, 1, 128, 128],
        "batch_size": 1,
        "precision": "float32",
        "warmup": args.warmup,
        "timed_per_repeat": args.timed,
        "repeats": args.repeats,
        "scope": (
            "GPU model inference only; preprocessing, host transfer, and "
            "refiner input transfer are excluded"
        ),
        "results": results,
    }
    (out_dir / "official_efficiency.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
