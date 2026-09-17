# -*- coding: utf-8 -*-
"""
SelfRDB inference: Generate MRI from CT using trained diffusion bridge.

Usage:
    python inference.py --checkpoint ../checkpoints/selfrdb/selfrdb_best.pt \\
                        --ct-dir ../data_slices/1BA001/ct \\
                        --output-dir ../result/selfrdb_generated

For full test-set evaluation:
    python inference.py --checkpoint ../checkpoints/selfrdb/selfrdb_best.pt \\
                        --data-root ../data_slices \\
                        --patient-split ../checkpoints/selfrdb/patient_split.json \\
                        --output-dir ../result/selfrdb_test
"""
import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from bridge_model import DiffusionBridge
from network import X0UNet
from dataset import list_images


def generate_from_ct(generator, bridge, ct_tensor, device, num_steps=20, num_recursions=5):
    """Generate MRI from a single CT slice."""
    y = ct_tensor.to(device)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            x0_hat = bridge.sample_ddib(y, num_steps=num_steps, num_recursions=num_recursions)
    return x0_hat.squeeze().cpu().numpy()


def process_ct_directory(generator, bridge, ct_dir, output_dir, image_size,
                         num_steps, num_recursions, max_slices, device):
    """Process all CT slices in a directory."""
    ct_images = list_images(ct_dir)
    names = sorted(ct_images)
    if max_slices > 0:
        names = names[:max_slices]

    os.makedirs(output_dir, exist_ok=True)
    total = len(names)
    print(f"Processing {total} CT slices from {ct_dir}")

    for i, name in enumerate(names, 1):
        img = cv2.imread(str(ct_images[name]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[{i}/{total}] Skip: {name}")
            continue

        orig_h, orig_w = img.shape
        img_rs = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
        ct_np = img_rs.astype(np.float32) / 127.5 - 1.0
        ct_tensor = torch.from_numpy(ct_np).unsqueeze(0).unsqueeze(0)

        gen_np = generate_from_ct(generator, bridge, ct_tensor, device, num_steps, num_recursions)
        gen_np = ((gen_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

        if (orig_h, orig_w) != (image_size, image_size):
            gen_np = cv2.resize(gen_np, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(os.path.join(output_dir, f"{name}.png"), gen_np)
        if i % 10 == 0 or i == total:
            print(f"[{i}/{total}] Generated: {name}.png")


def process_test_set(generator, bridge, data_root, patient_ids, output_root,
                     image_size, num_steps, num_recursions, device):
    """Generate MRI for all patients in the test set."""
    data_root = Path(data_root)
    output_root = Path(output_root)
    total_patients = len(patient_ids)
    total_slices = 0

    for pi, pid in enumerate(sorted(patient_ids), 1):
        ct_dir = data_root / pid / "ct"
        mr_out = output_root / pid / "mr_generated"
        mr_out.mkdir(parents=True, exist_ok=True)

        ct_images = list_images(ct_dir)
        for i, name in enumerate(sorted(ct_images)):
            img = cv2.imread(str(ct_images[name]), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            orig_h, orig_w = img.shape
            img_rs = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
            ct_np = img_rs.astype(np.float32) / 127.5 - 1.0
            ct_tensor = torch.from_numpy(ct_np).unsqueeze(0).unsqueeze(0)

            gen_np = generate_from_ct(generator, bridge, ct_tensor, device, num_steps, num_recursions)
            gen_np = ((gen_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            if (orig_h, orig_w) != (image_size, image_size):
                gen_np = cv2.resize(gen_np, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

            cv2.imwrite(str(mr_out / f"{name}.png"), gen_np)
            total_slices += 1

        if pi % 5 == 0:
            print(f"Patient {pi}/{total_patients}: {pid} ({total_slices} slices total)")

    print(f"Done. Generated {total_slices} MRI slices for {total_patients} patients")


def main():
    parser = argparse.ArgumentParser(description="SelfRDB CT-to-MRI inference")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model (.pt)")
    # Input options
    parser.add_argument("--ct-dir", default=None, help="Single CT slices directory")
    parser.add_argument("--data-root", default="../data_slices", help="Full data root for test set")
    parser.add_argument("--patient-split", default=None, help="patient_split.json for test patients")
    # Output
    parser.add_argument("--output-dir", default="../result/selfrdb_test")
    # Model
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-ch", type=int, default=64)
    parser.add_argument("--ch-mult", type=str, default="1,2,4")
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.1)
    # Sampling
    parser.add_argument("--num-steps", type=int, default=20, help="DDIB accelerated steps")
    parser.add_argument("--num-recursions", type=int, default=5, help="Max self-consistent recursions")
    parser.add_argument("--max-slices", type=int, default=0, help="Max slices (0=all)")
    # Device
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Sampling: {args.num_steps} steps, {args.num_recursions} recursions")

    # Load model
    ch_mult = tuple(int(x) for x in args.ch_mult.split(","))
    generator = X0UNet(in_ch=3, out_ch=1, base_ch=args.base_ch, ch_mult=ch_mult).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()
    print(f"Loaded checkpoint epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?')}")

    bridge = DiffusionBridge(generator, timesteps=args.timesteps, gamma=args.gamma, device=device)

    # Determine mode
    if args.ct_dir:
        process_ct_directory(
            generator, bridge, args.ct_dir, args.output_dir,
            args.image_size, args.num_steps, args.num_recursions, args.max_slices, device
        )
    elif args.patient_split:
        with open(args.patient_split) as f:
            split = json.load(f)
        test_pids = set(split.get("test", []))
        print(f"Test patients: {len(test_pids)}")
        process_test_set(
            generator, bridge, args.data_root, test_pids, args.output_dir,
            args.image_size, args.num_steps, args.num_recursions, device
        )
    else:
        print("Error: Provide --ct-dir for single directory or --patient-split for test set.")
        return


if __name__ == "__main__":
    main()
