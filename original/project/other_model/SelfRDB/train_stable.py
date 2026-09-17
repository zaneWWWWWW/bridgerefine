# -*- coding: utf-8 -*-
"""SelfRDB stable training - verified on 5-patient test."""
import os, sys, time, torch
import numpy as np
from bridge_model import DiffusionBridge
from network import X0UNet
from dataset import create_dataloaders

device = torch.device('cuda')
SAVE_DIR = 'D:/Cross-modal conversion/checkpoints/selfrdb'
os.makedirs(SAVE_DIR, exist_ok=True)

# Write to file with explicit flush
log_f = open('D:/Cross-modal conversion/training_log.txt', 'w', buffering=1)

def log(msg):
    t = time.strftime('%H:%M:%S')
    line = f'{t} - {msg}'
    print(line, flush=True)
    log_f.write(line + '\n')
    log_f.flush()

log(f'Device: {device}')
log(f'GPU: {torch.cuda.get_device_name(0)}')

train_loader, val_loader, test_loader, test_pids, _, _ = create_dataloaders(
    'D:/Cross-modal conversion/data_slices', image_size=128, batch_size=8, num_workers=0
)
log(f'Train: {len(train_loader)} batches, Val: {len(val_loader)}')

gen = X0UNet(in_ch=3, out_ch=1, base_ch=64, ch_mult=(1,2,4), dropout=0.1).to(device)
bridge = DiffusionBridge(gen, timesteps=300, gamma=0.1, device=device)
log(f'Params: {sum(p.numel() for p in gen.parameters()):,}')

opt = torch.optim.AdamW(gen.parameters(), lr=1e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)

best_val_loss = float('inf')
VAL_EVERY = 5

for epoch in range(1, 51):
    t0 = time.time()

    # Train
    gen.train()
    train_loss = 0.0
    n_batches = 0
    for batch_idx, (ct, mr, _) in enumerate(train_loader):
        ct, mr = ct.to(device), mr.to(device)
        loss = bridge.training_loss(mr, ct)

        if torch.isnan(loss):
            log(f'  WARN: NaN at epoch {epoch} batch {batch_idx}')
            continue

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        opt.step()
        train_loss += loss.item()
        n_batches += 1

    train_loss /= max(n_batches, 1)
    sched.step()

    # Validate
    val_str = 'N/A'
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
        val_str = f'{val_loss:.6f}'

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'epoch': epoch, 'generator': gen.state_dict(), 'val_loss': val_loss},
                       os.path.join(SAVE_DIR, 'selfrdb_best.pt'))

    elapsed = time.time() - t0
    lr = sched.get_last_lr()[0]
    log(f'Epoch {epoch:3d}/50 | train: {train_loss:.6f} | val: {val_str} | {elapsed:.0f}s | lr: {lr:.2e}')

    if epoch % 10 == 0:
        torch.save({'epoch': epoch, 'generator': gen.state_dict()},
                   os.path.join(SAVE_DIR, f'selfrdb_epoch_{epoch}.pt'))

torch.save({'epoch': 50, 'generator': gen.state_dict()}, os.path.join(SAVE_DIR, 'selfrdb_final.pt'))
log(f'Done! Best val_loss: {best_val_loss:.6f}')
log_f.close()
