# -*- coding: utf-8 -*-
"""Train Mask-Guided Diffusion Bridge (pure mask, no adversarial)."""
import sys, os, time, json, torch
sys.path.insert(0, 'D:/Cross-modal conversion')
from other_model.Masked_Bridge.bridge_model import DiffusionBridge
from other_model.Masked_Bridge.network import X0UNet
from selfrdb.dataset import PairedCTMRIDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = torch.device('cuda')
print(f'Device: {device}', flush=True)

with open('D:/Cross-modal conversion/checkpoints/selfrdb/patient_split.json') as f:
    split = json.load(f)

train_ds = PairedCTMRIDataset('D:/Cross-modal conversion/data_slices', image_size=128, augment=True, patient_ids=split['train'])
val_ds = PairedCTMRIDataset('D:/Cross-modal conversion/data_slices', image_size=128, augment=False, patient_ids=split['val'])
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
print(f'Train: {len(train_ds)} ({len(train_loader)} batches), Val: {len(val_ds)}', flush=True)

gen = X0UNet(in_ch=4, out_ch=1, base_ch=64, ch_mult=(1,2,4), dropout=0.1).to(device)
bridge = DiffusionBridge(gen, timesteps=300, gamma=0.1, device=device)
n = sum(p.numel() for p in gen.parameters())
print(f'Params: {n:,}', flush=True)

opt = torch.optim.AdamW(gen.parameters(), lr=1e-4, weight_decay=1e-5)

SAVE = 'D:/Cross-modal conversion/other_model/Masked_Bridge'
os.makedirs(SAVE, exist_ok=True)

def extract_mask(ct):
    return F.max_pool2d((ct > -0.5).float(), 3, 1, 1)

best_val = float('inf')
N_EPOCHS = 20

for epoch in range(1, N_EPOCHS + 1):
    t0 = time.time()
    gen.train()
    train_loss, n_b = 0.0, 0

    for ct, mr, _ in train_loader:
        ct, mr = ct.to(device), mr.to(device)
        mask = extract_mask(ct)
        loss = bridge.training_loss(mr, ct, mask)
        if torch.isnan(loss): continue
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        opt.step()
        train_loss += loss.item(); n_b += 1
    train_loss /= max(n_b, 1)

    gen.eval()
    val_loss, n_v = 0.0, 0
    with torch.no_grad():
        for ct, mr, _ in val_loader:
            ct, mr = ct.to(device), mr.to(device)
            mask = extract_mask(ct)
            loss = bridge.training_loss(mr, ct, mask)
            if not torch.isnan(loss):
                val_loss += loss.item(); n_v += 1
    val_loss /= max(n_v, 1)

    elapsed = time.time() - t0
    print(f'Epoch {epoch:3d}/{N_EPOCHS} | train: {train_loss:.4f} | val: {val_loss:.4f} | {elapsed:.0f}s', flush=True)

    if val_loss < best_val:
        best_val = val_loss
        torch.save({'epoch': epoch, 'generator': gen.state_dict(), 'val_loss': val_loss}, os.path.join(SAVE, 'best.pt'))

    if epoch % 10 == 0:
        torch.save({'epoch': epoch, 'generator': gen.state_dict()}, os.path.join(SAVE, f'epoch_{epoch}.pt'))

torch.save({'epoch': N_EPOCHS, 'generator': gen.state_dict()}, os.path.join(SAVE, 'final.pt'))
print(f'Done! Best val_loss: {best_val:.6f}', flush=True)
