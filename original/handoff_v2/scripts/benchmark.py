# -*- coding: utf-8 -*-
"""Independent 10-step x 2-recursion inference benchmark.

Run one method per process so CT-only and BridgeRefine are never co-resident:
    python benchmark_efficiency_10x2.py --method ct_only
    python benchmark_efficiency_10x2.py --method bridgerefine

Protocol:
    batch_size = 1
    warmup = 20
    timed = 100
    repeats = 3
    torch.cuda.synchronize() before/after each repeat
    model load, input transfer, and preprocessing excluded from timing
    peak memory measured after model+input allocation, so it is additional
    inference allocation rather than total device occupancy
"""

import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, "D:/Cross-modal conversion")
from other_model.SelfRDB.bridge_model import DiffusionBridge
from other_model.SelfRDB.network import X0UNet
from other_model.BridgeGAN.model import RefinerGenerator

DEVICE = torch.device("cuda")
OUT = "D:/Cross-modal conversion/experiments/handoff_v2/efficiency"


class SimpleUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 7, padding=3, padding_mode="reflect"),
            torch.nn.InstanceNorm2d(64), torch.nn.ReLU(True))
        self.enc2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
            torch.nn.InstanceNorm2d(128), torch.nn.ReLU(True))
        self.res = torch.nn.Sequential(*[
            torch.nn.Sequential(
                torch.nn.Conv2d(128, 128, 3, padding=1, padding_mode="reflect"),
                torch.nn.InstanceNorm2d(128), torch.nn.ReLU(True),
                torch.nn.Conv2d(128, 128, 3, padding=1, padding_mode="reflect"),
                torch.nn.InstanceNorm2d(128))
            for _ in range(4)])
        self.dec1 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            torch.nn.Conv2d(128, 64, 3, padding=1),
            torch.nn.InstanceNorm2d(64), torch.nn.ReLU(True))
        self.out = torch.nn.Sequential(
            torch.nn.Conv2d(64, 1, 7, padding=3, padding_mode="reflect"),
            torch.nn.Tanh())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        r = e2
        for block in self.res:
            r = r + block(r)
        d1 = self.dec1(r)
        return self.out(d1)


def load_input():
    img = cv2.imread("D:/Cross-modal conversion/data_slices/1BA012/ct/0002.jpg",
                     cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("test CT slice not found")
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(DEVICE)


def measure(fn, x, warmup, timed, repeats, raw_path, raw_count=30):
    with torch.no_grad():
        for _ in range(warmup):
            fn(x)
    torch.cuda.synchronize()

    raw_rows = []
    block_means = []
    for rep in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(timed):
                fn(x)
        torch.cuda.synchronize()
        block_means.append((time.perf_counter() - t0) * 1000.0 / timed)

        for i in range(raw_count):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with torch.no_grad():
                fn(x)
            end.record()
            torch.cuda.synchronize()
            raw_rows.append({"repeat": rep + 1, "slice_index": i,
                             "event_time_ms": f"{start.elapsed_time(end):.6f}"})

    with open(raw_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["repeat", "slice_index", "event_time_ms"])
        w.writeheader()
        w.writerows(raw_rows)

    return block_means, raw_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["ct_only", "bridgerefine"], required=True)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--timed", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    x = load_input()

    if args.method == "ct_only":
        model = SimpleUNet().to(DEVICE)
        model.load_state_dict(torch.load(
            "D:/Cross-modal conversion/other_model/phase2_CT_only_L1/best.pt",
            map_location=DEVICE, weights_only=False)["model"])
        model.eval()

        def fn(inp):
            return model(inp)

    else:
        gen = X0UNet(in_ch=3, out_ch=1, base_ch=64, ch_mult=(1, 2, 4)).to(DEVICE)
        gen.load_state_dict(torch.load(
            "D:/Cross-modal conversion/other_model/SelfRDB/selfrdb_best.pt",
            map_location=DEVICE, weights_only=False)["generator"])
        gen.eval()
        bridge = DiffusionBridge(gen, timesteps=300, gamma=0.1, device=DEVICE)
        refiner = RefinerGenerator(in_ch=2, out_ch=1, base_ch=64).to(DEVICE)
        refiner.load_state_dict(torch.load(
            "D:/Cross-modal conversion/other_model/BridgeGAN_L1/best.pt",
            map_location=DEVICE, weights_only=False)["refiner"])
        refiner.eval()

        def fn(inp):
            coarse = bridge.sample_ddib(inp, num_steps=10, num_recursions=2)
            return refiner(coarse, inp)

    # warm inference outside the memory window
    with torch.no_grad():
        fn(x)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        fn(x)
    torch.cuda.synchronize()
    alloc = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()

    raw_path = os.path.join(OUT, f"timing_raw_{args.method}.csv")
    block_means, raw_rows = measure(fn, x, args.warmup, args.timed, args.repeats, raw_path)

    raw_ms = [float(r["event_time_ms"]) for r in raw_rows]

    summary = {
        "method": args.method,
        "sampling_steps": 10 if args.method == "bridgerefine" else None,
        "recursive_estimates": 2 if args.method == "bridgerefine" else None,
        "protocol": {
            "batch_size": 1, "warmup": args.warmup, "timed": args.timed,
            "repeats": args.repeats, "cuda_synchronize": True,
            "includes_model_load": False, "includes_input_transfer": False,
            "includes_preprocessing": False,
            "co_resident_other_model": False,
            "memory_definition": "max_memory_allocated after model+input allocation; additional inference allocation",
        },
        "time_ms": {
            "primary_block_mean": float(np.mean(block_means)),
            "primary_block_std": float(np.std(block_means, ddof=1)),
            "block_repeat_means": block_means,
            "raw_event_mean": float(np.mean(raw_ms)),
            "raw_event_std": float(np.std(raw_ms, ddof=1)),
            "raw_event_median": float(np.median(raw_ms)),
        },
        "memory_mb": {
            "max_memory_allocated": alloc / 1e6,
            "max_memory_reserved": reserved / 1e6,
        },
    }
    with open(os.path.join(OUT, f"summary_{args.method}.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
