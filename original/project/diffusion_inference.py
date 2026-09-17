# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path
import cv2
import numpy as np
import torch
from diffusion_model import ControlUNet, GaussianDiffusion
from diffusion_dataset import list_images, extract_canny_edge

def main():
    parser = argparse.ArgumentParser(description="Generate MRI from CT using ControlNet DDPM")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ct-dir", default="result/ct_brain")
    parser.add_argument("--output-dir", default="result/generated_mri")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-ch", type=int, default=64)
    parser.add_argument("--ch-mult", type=str, default="1,2,4")
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--max-slices", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"Device: {device}")

    ch_mult = tuple(int(x) for x in args.ch_mult.split(","))
    model = ControlUNet(in_ch=1, out_ch=1, cond_ch=2, base_ch=args.base_ch, ch_mult=ch_mult).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint epoch {ckpt.get('epoch', '?')}")

    diffusion = GaussianDiffusion(model, timesteps=args.timesteps, device=device)

    ct_images = list_images(args.ct_dir)
    names = sorted(ct_images)
    if args.max_slices > 0:
        names = names[:args.max_slices]
    total = len(names)
    print(f"Processing {total} CT slices")

    for i, name in enumerate(names, 1):
        ct_path = ct_images[name]
        img = cv2.imread(str(ct_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[{i}/{total}] Skip: {ct_path}")
            continue
        orig_h, orig_w = img.shape
        img_rs = cv2.resize(img, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)
        ct_np = img_rs.astype(np.float32) / 127.5 - 1.0
        ct_tensor = torch.from_numpy(ct_np).unsqueeze(0).unsqueeze(0).to(device)
        edge_np = extract_canny_edge(ct_np)
        edge_tensor = torch.from_numpy(edge_np).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            generated = diffusion.ddim_sample(ct_tensor, edge_tensor, ddim_steps=args.ddim_steps)

        gen_np = generated.squeeze().cpu().numpy()
        gen_np = ((gen_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        if (orig_h, orig_w) != (args.image_size, args.image_size):
            gen_np = cv2.resize(gen_np, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        out_path = os.path.join(args.output_dir, f"{name}.png")
        cv2.imwrite(out_path, gen_np)
        if i % 10 == 0 or i == total:
            print(f"[{i}/{total}] Generated: {name}.png")

    print(f"Done. {total} MRI slices saved to {args.output_dir}")

if __name__ == "__main__":
    main()
