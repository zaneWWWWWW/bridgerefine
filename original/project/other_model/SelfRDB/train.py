# -*- coding: utf-8 -*-
"""
SelfRDB training script.

Usage:
    python train.py --data-root ../data_slices --save-dir ../checkpoints/selfrdb
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from bridge_model import DiffusionBridge
from network import X0UNet, Discriminator
from dataset import create_dataloaders


def compute_ssim(a, b):
    """Quick SSIM for validation monitoring."""
    a_np = a.squeeze().cpu().numpy()
    b_np = b.squeeze().cpu().numpy()
    c1, c2 = (0.01) ** 2, (0.03) ** 2
    mu_x = cv2.GaussianBlur(a_np, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(b_np, (11, 11), 1.5)
    sigma_x_sq = cv2.GaussianBlur(a_np ** 2, (11, 11), 1.5) - mu_x ** 2
    sigma_y_sq = cv2.GaussianBlur(b_np ** 2, (11, 11), 1.5) - mu_y ** 2
    sigma_xy = cv2.GaussianBlur(a_np * b_np, (11, 11), 1.5) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x_sq + sigma_y_sq + c2)
    return float(np.mean(num / (den + 1e-12)))


def validation_sample(generator, bridge, val_loader, device, output_dir, epoch,
                      num_steps=10, num_recursions=3, max_samples=4):
    """Generate validation samples and save comparison grids."""
    generator.eval()
    os.makedirs(output_dir, exist_ok=True)

    samples_done = 0
    all_ssim = []
    all_psnr = []

    for batch in val_loader:
        ct, mr, names = batch
        ct, mr = ct.to(device), mr.to(device)

        for j in range(min(ct.shape[0], max_samples - samples_done)):
            y = ct[j:j+1]
            real = mr[j:j+1]
            with torch.no_grad():
                gen = bridge.sample_ddib(y, num_steps=num_steps, num_recursions=num_recursions)

            # Compute metrics
            gen_np = ((gen.squeeze().cpu().numpy() + 1.0) / 2.0).clip(0, 1)
            real_np = ((real.squeeze().cpu().numpy() + 1.0) / 2.0).clip(0, 1)
            ct_np = ((y.squeeze().cpu().numpy() + 1.0) / 2.0).clip(0, 1)

            ssim_val = compute_ssim(
                torch.from_numpy(real_np).unsqueeze(0).unsqueeze(0),
                torch.from_numpy(gen_np).unsqueeze(0).unsqueeze(0)
            )
            mse_val = np.mean((real_np - gen_np) ** 2)
            psnr_val = 10 * math.log10(1.0 / max(mse_val, 1e-12))

            all_ssim.append(ssim_val)
            all_psnr.append(psnr_val)

            # Build comparison grid
            diff = np.abs(real_np - gen_np)
            diff_viz = (diff / max(diff.max(), 0.01) * 255).astype(np.uint8)
            diff_viz = cv2.applyColorMap(diff_viz, cv2.COLORMAP_HOT)

            ct_viz = (ct_np * 255).astype(np.uint8)
            real_viz = (real_np * 255).astype(np.uint8)
            gen_viz = (gen_np * 255).astype(np.uint8)

            top = np.concatenate([
                cv2.cvtColor(ct_viz, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(real_viz, cv2.COLOR_GRAY2BGR),
            ], axis=1)
            bottom = np.concatenate([
                cv2.cvtColor(gen_viz, cv2.COLOR_GRAY2BGR),
                diff_viz,
            ], axis=1)
            grid = np.concatenate([top, bottom], axis=0)

            h = ct_viz.shape[0]
            cv2.putText(grid, "CT", (5, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
            cv2.putText(grid, f"SSIM={ssim_val:.3f} PSNR={psnr_val:.1f}",
                       (5, grid.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

            name_str = names[j].replace("/", "_") if isinstance(names, (list, tuple)) else f"sample{j}"
            cv2.imwrite(os.path.join(output_dir, f"epoch{epoch:03d}_{name_str}.png"), grid)
            samples_done += 1

        if samples_done >= max_samples:
            break

    return {"ssim": np.mean(all_ssim) if all_ssim else 0,
            "psnr": np.mean(all_psnr) if all_psnr else 0}


def main():
    parser = argparse.ArgumentParser(description="Train SelfRDB for CT-to-MRI translation")
    # Data
    parser.add_argument("--data-root", default="../data_slices")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    # Model
    parser.add_argument("--base-ch", type=int, default=64)
    parser.add_argument("--ch-mult", type=str, default="1,2,4")
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.1, help="Soft-prior strength")
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--use-amp", action="store_true", default=True, help="Use AMP mixed precision")
    parser.add_argument("--use-gan", action="store_true", default=False, help="Use adversarial loss")
    parser.add_argument("--gan-weight", type=float, default=0.1, help="Adversarial loss weight")
    # Checkpointing
    parser.add_argument("--save-dir", default="../checkpoints/selfrdb")
    parser.add_argument("--save-every", type=int, default=10)
    # Device
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, val_loader, test_loader, test_pids, val_pids, train_pids = create_dataloaders(
        args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    with open(os.path.join(args.save_dir, "patient_split.json"), "w") as f:
        json.dump({
            "train": sorted(train_pids), "val": sorted(val_pids), "test": sorted(test_pids)
        }, f, indent=2)

    # Model
    ch_mult = tuple(int(x) for x in args.ch_mult.split(","))
    generator = X0UNet(in_ch=3, out_ch=1, base_ch=args.base_ch, ch_mult=ch_mult, dropout=0.1).to(device)
    bridge = DiffusionBridge(generator, timesteps=args.timesteps, gamma=args.gamma, device=device)

    n_params = sum(p.numel() for p in generator.parameters())
    print(f"Generator params: {n_params:,}")

    opt_g = torch.optim.AdamW(generator.parameters(), lr=args.lr, weight_decay=1e-5)
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=args.epochs)

    if args.use_gan:
        discriminator = Discriminator(in_ch=2, base_ch=64, ch_mult=(1, 2, 4)).to(device)
        opt_d = torch.optim.AdamW(discriminator.parameters(), lr=args.lr * 0.5, weight_decay=1e-5)
        print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

    scaler = torch.amp.GradScaler(enabled=args.use_amp)
    best_val_loss = float("inf")

    print(f"Training {args.epochs} epochs, T={args.timesteps}, gamma={args.gamma}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ---- Train ----
        generator.train()
        train_loss = 0.0
        if args.use_gan:
            discriminator.train()

        for batch in train_loader:
            ct, mr, _ = batch
            ct, mr = ct.to(device), mr.to(device)

            # Generator loss
            with torch.amp.autocast("cuda", enabled=args.use_amp):
                loss_g = bridge.training_loss(mr, ct)

                if args.use_gan:
                    # Sample x_t for GAN training
                    t = torch.randint(1, args.timesteps + 1, (ct.shape[0],), device=device, dtype=torch.long)
                    xt, _ = bridge.q_sample(mr, ct, t)
                    x0_pred = generator(xt, t, ct, torch.zeros_like(xt))
                    xt_prev_fake = bridge.q_posterior(xt, t, x0_pred, ct)
                    d_fake = discriminator(xt_prev_fake, t - 1, xt)
                    loss_gan = F.binary_cross_entropy_with_logits(
                        d_fake, torch.ones_like(d_fake)
                    )
                    loss_g = loss_g + args.gan_weight * loss_gan

            opt_g.zero_grad()
            scaler.scale(loss_g).backward()
            scaler.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            scaler.step(opt_g)
            scaler.update()
            train_loss += loss_g.item()

            # Discriminator step
            if args.use_gan:
                with torch.amp.autocast("cuda", enabled=args.use_amp):
                    t = torch.randint(1, args.timesteps + 1, (ct.shape[0],), device=device, dtype=torch.long)
                    xt, _ = bridge.q_sample(mr, ct, t)
                    with torch.no_grad():
                        x0_pred = generator(xt, t, ct, torch.zeros_like(xt))
                        xt_prev_fake = bridge.q_posterior(xt, t, x0_pred, ct)
                    xt_prev_real = bridge.q_posterior(xt, t, mr, ct)
                    d_real = discriminator(xt_prev_real, t - 1, xt)
                    d_fake = discriminator(xt_prev_fake.detach(), t - 1, xt)
                    loss_d = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real)) + \
                             F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))

                opt_d.zero_grad()
                scaler.scale(loss_d).backward()
                scaler.step(opt_d)
                scaler.update()

        train_loss /= len(train_loader)
        sched_g.step()

        # ---- Validation ----
        generator.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                ct, mr, _ = batch
                ct, mr = ct.to(device), mr.to(device)
                with torch.amp.autocast("cuda", enabled=args.use_amp):
                    val_loss += bridge.training_loss(mr, ct).item()
        val_loss /= len(val_loader)

        elapsed = time.time() - t0
        lr_now = sched_g.get_last_lr()[0]
        print(f"Epoch {epoch:3d}/{args.epochs} | train: {train_loss:.6f} | val: {val_loss:.6f} | "
              f"time: {elapsed:.0f}s | lr: {lr_now:.2e}")

        # Checkpointing
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"selfrdb_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "generator": generator.state_dict(),
                "opt_g": opt_g.state_dict(),
                "val_loss": val_loss,
            }, ckpt_path)
            print(f"  -> saved {ckpt_path}")

            # Validation visualization
            val_viz_dir = os.path.join(args.save_dir, "val_samples")
            metrics = validation_sample(generator, bridge, val_loader, device,
                                        val_viz_dir, epoch)
            print(f"  -> val samples: SSIM={metrics['ssim']:.4f}, PSNR={metrics['psnr']:.1f}dB")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.save_dir, "selfrdb_best.pt")
            torch.save({
                "epoch": epoch,
                "generator": generator.state_dict(),
                "val_loss": val_loss,
            }, best_path)

    # Final save
    final_path = os.path.join(args.save_dir, "selfrdb_final.pt")
    torch.save({
        "epoch": args.epochs,
        "generator": generator.state_dict(),
        "val_loss": val_loss,
    }, final_path)
    print(f"Done. Best val_loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
