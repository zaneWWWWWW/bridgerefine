# -*- coding: utf-8 -*-
"""SelfRDB training runner - stable configuration."""
import os, sys, time, math, logging
import torch
import numpy as np
from bridge_model import DiffusionBridge
from network import X0UNet
from dataset import create_dataloaders

# FileHandler for guaranteed log flush
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
for h in [logging.FileHandler('D:/Cross-modal conversion/training_log.txt', mode='w'),
          logging.StreamHandler(sys.stdout)]:
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
    log.addHandler(h)

device = torch.device('cuda')
log.info(f'Device: {device}')

train_loader, val_loader, test_loader, test_pids, _, _ = create_dataloaders(
    'D:/Cross-modal conversion/data_slices', image_size=128, batch_size=8, num_workers=0
)
log.info(f'Train: {len(train_loader)} batches, Val: {len(val_loader)}, Test: {len(test_loader)}')

gen = X0UNet(in_ch=3, out_ch=1, base_ch=64, ch_mult=(1,2,4), dropout=0.1).to(device)
bridge = DiffusionBridge(gen, timesteps=300, gamma=0.1, device=device)
log.info(f'Params: {sum(p.numel() for p in gen.parameters()):,}')

# Conservative settings for stability
opt = torch.optim.AdamW(gen.parameters(), lr=1e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
save_dir = 'D:/Cross-modal conversion/checkpoints/selfrdb'
os.makedirs(save_dir, exist_ok=True)

best_val_loss = float('inf')
VAL_EVERY = 5
nan_count = 0

for epoch in range(1, 51):
    t0 = time.time()

    # ---- Train ----
    gen.train()
    train_loss = 0.0
    n_batches = 0
    for batch_idx, (ct, mr, _) in enumerate(train_loader):
        ct, mr = ct.to(device), mr.to(device)

        # NO AMP - it caused NaN with two-forward loss
        loss = bridge.training_loss(mr, ct)

        # Skip if NaN
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            if nan_count <= 3:
                log.warning(f'  NaN at epoch {epoch} batch {batch_idx}, skipping')
            continue

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        opt.step()
        train_loss += loss.item()
        n_batches += 1

        if (batch_idx + 1) % 500 == 0:
            log.info(f'  Epoch {epoch}: {batch_idx+1}/{len(train_loader)} batches, loss={loss.item():.4f}')

    train_loss /= max(n_batches, 1)
    sched.step()

    if nan_count > 0:
        log.warning(f'  Total NaN skips: {nan_count}')
        nan_count = 0

    # ---- Validation ----
    val_loss_str = 'N/A'
    if epoch % VAL_EVERY == 0 or epoch == 1:
        gen.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for ct, mr, _ in val_loader:
                ct, mr = ct.to(device), mr.to(device)
                loss = bridge.training_loss(mr, ct)
                if not torch.isnan(loss):
                    val_loss += loss.item()
                    n_val += 1
        val_loss /= max(n_val, 1)
        val_loss_str = f'{val_loss:.6f}'

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'epoch': epoch, 'generator': gen.state_dict(), 'val_loss': val_loss},
                       os.path.join(save_dir, 'selfrdb_best.pt'))
            log.info(f'  -> best model saved (val_loss={val_loss:.6f})')

    elapsed = time.time() - t0
    lr = sched.get_last_lr()[0]
    log.info(f'Epoch {epoch:3d}/50 | train: {train_loss:.6f} | val: {val_loss_str} | {elapsed:.0f}s | lr: {lr:.2e}')

    if epoch % 10 == 0:
        torch.save({'epoch': epoch, 'generator': gen.state_dict()},
                   os.path.join(save_dir, f'selfrdb_epoch_{epoch}.pt'))

torch.save({'epoch': 50, 'generator': gen.state_dict()}, os.path.join(save_dir, 'selfrdb_final.pt'))
log.info(f'Done! Best val_loss: {best_val_loss:.6f}')
