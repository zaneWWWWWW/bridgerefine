# -*- coding: utf-8 -*-
import argparse
import json
import os
import time
import torch
from diffusion_model import ControlUNet, GaussianDiffusion
from diffusion_dataset import create_dataloaders

def main():
    parser = argparse.ArgumentParser(description="Train ControlNet DDPM for CT-to-MRI")
    parser.add_argument("--data-root", default="data_slices")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--base-ch", type=int, default=64)
    parser.add_argument("--ch-mult", type=str, default="1,2,4")
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, test_pids, val_pids, train_pids = create_dataloaders(
        args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
    )

    with open(os.path.join(args.save_dir, "patient_split.json"), "w") as f:
        json.dump({"train": sorted(train_pids), "val": sorted(val_pids), "test": sorted(test_pids)}, f)
    print(f"Patient split saved to {args.save_dir}/patient_split.json")

    ch_mult = tuple(int(x) for x in args.ch_mult.split(","))
    model = ControlUNet(in_ch=1, out_ch=1, cond_ch=2, base_ch=args.base_ch, ch_mult=ch_mult, dropout=0.1).to(device)
    diffusion = GaussianDiffusion(model, timesteps=args.timesteps, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")
    print(f"Training for {args.epochs} epochs...")

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for ct, mr, edge, _ in train_loader:
            ct, mr, edge = ct.to(device), mr.to(device), edge.to(device)
            loss = diffusion.training_loss(mr, ct, edge)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for ct, mr, edge, _ in val_loader:
                ct, mr, edge = ct.to(device), mr.to(device), edge.to(device)
                val_loss += diffusion.training_loss(mr, ct, edge).item()
        val_loss /= len(val_loader)

        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f} | time: {elapsed:.1f}s | lr: {lr_now:.2e}")

        if epoch % args.save_every == 0:
            ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "val_loss": val_loss}
            torch.save(ckpt, os.path.join(args.save_dir, f"controlnet_epoch_{epoch}.pt"))
            print(f"  -> saved checkpoint")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_loss": val_loss}, os.path.join(args.save_dir, "controlnet_best.pt"))

    torch.save({"epoch": args.epochs, "model": model.state_dict(), "val_loss": val_loss}, os.path.join(args.save_dir, "controlnet_final.pt"))
    print(f"Done. Best val_loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    main()
