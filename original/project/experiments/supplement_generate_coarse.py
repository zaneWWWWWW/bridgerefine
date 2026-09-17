"""Generate aligned SelfRDB coarse-MRI caches for the held-out splits.

The original coarse cache contains the 27,309 training slices only. This script
uses the frozen bridge checkpoint to create validation/test caches keyed by the
dataset's deterministic patient/slice order. It is deliberately separate from
training so the bridge is never updated during the input ablation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from other_model.SelfRDB.bridge_model import DiffusionBridge
from other_model.SelfRDB.network import X0UNet
from selfrdb.dataset import PairedCTMRIDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--num-recursions", type=int, default=5)
    args = parser.parse_args()

    with open(ROOT / "checkpoints/selfrdb/patient_split.json") as f:
        split = json.load(f)
    ds = PairedCTMRIDataset(
        str(ROOT / "data_slices"), image_size=128, augment=False,
        patient_ids=split[args.split]
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = X0UNet(in_ch=3, out_ch=1, base_ch=64, ch_mult=(1, 2, 4)).to(device)
    ckpt_path = ROOT / "checkpoints/selfrdb/selfrdb_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["generator"])
    model.eval()
    bridge = DiffusionBridge(model, timesteps=300, gamma=0.1, device=device)

    # Keep the stochastic bridge cache reproducible on this machine.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = ROOT / "experiments/supplement_results/coarse"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}_coarse_seed{args.seed}.npz"
    names_path = out_dir / f"{args.split}_names.json"

    all_coarse = []
    all_names = []
    with torch.inference_mode():
        for i, (ct, _mr, names) in enumerate(loader):
            ct = ct.to(device, non_blocking=True)
            coarse = bridge.sample_ddib(
                ct, num_steps=args.num_steps, num_recursions=args.num_recursions
            )
            all_coarse.append(coarse.detach().float().cpu().numpy())
            all_names.extend(list(names))
            if (i + 1) % 100 == 0 or i + 1 == len(loader):
                print(f"{args.split}: {min((i + 1) * args.batch_size, len(ds))}/{len(ds)}", flush=True)

    coarse_np = np.concatenate(all_coarse, axis=0).astype(np.float32)
    np.savez_compressed(out_path, coarse=coarse_np)
    with open(names_path, "w") as f:
        json.dump(all_names, f)
    print(f"saved {out_path} shape={coarse_np.shape}")
    print(f"saved {names_path}")


if __name__ == "__main__":
    main()
