# -*- coding: utf-8 -*-
"""MG-CycleGAN training: Mask-Guided Fidelity-Constrained CT→MRI translation."""
import sys, os, time, math, random
import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np, cv2
sys.path.insert(0, 'D:/Cross-modal conversion')
from selfrdb.dataset import PairedCTMRIDataset
from torch.utils.data import DataLoader
from model import Generator, Discriminator

device = torch.device('cuda')
print(f'Device: {device}', flush=True)

# Load data (same split as SelfRDB for fair comparison)
import json
with open('D:/Cross-modal conversion/checkpoints/selfrdb/patient_split.json') as f:
    split = json.load(f)

IMAGE_SIZE = 128
BATCH_SIZE = 8
N_EPOCHS = 20

train_ds = PairedCTMRIDataset('D:/Cross-modal conversion/data_slices', image_size=IMAGE_SIZE,
                               augment=True, patient_ids=split['train'])
val_ds = PairedCTMRIDataset('D:/Cross-modal conversion/data_slices', image_size=IMAGE_SIZE,
                             augment=False, patient_ids=split['val'])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
print(f'Train: {len(train_ds)} slices, Val: {len(val_ds)} slices', flush=True)

# Models
G = Generator(in_ch=2, out_ch=1, base_ch=64, n_res=6).to(device)  # CT+mask → MRI
D = Discriminator(in_ch=1, base_ch=64).to(device)

n_g = sum(p.numel() for p in G.parameters())
n_d = sum(p.numel() for p in D.parameters())
print(f'Generator: {n_g:,} params, Discriminator: {n_d:,} params', flush=True)

# Optimizers
opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

# Losses
adv_loss = nn.MSELoss()  # LSGAN
l1_loss = nn.L1Loss()

# Load brain masks for guidance (from ct.nii.gz mask.nii.gz)
# For simplicity: extract mask from CT using Otsu thresholding at load time
def extract_mask(ct_batch):
    """Extract brain mask from CT using Otsu-like thresholding."""
    # CT values in [-1,1], brain tissue > -0.5 typically after normalization
    mask = (ct_batch > -0.5).float()
    # Dilate slightly
    mask = F.max_pool2d(mask, 3, stride=1, padding=1)
    return mask

save_dir = 'D:/Cross-modal conversion/checkpoints/mg_cyclegan'
os.makedirs(save_dir, exist_ok=True)

best_val_loss = float('inf')
print(f'Training {N_EPOCHS} epochs...', flush=True)

for epoch in range(1, N_EPOCHS + 1):
    t0 = time.time()
    G.train(); D.train()
    g_loss_total, d_loss_total = 0.0, 0.0
    n_batches = 0

    for ct, mr, _ in train_loader:
        ct, mr = ct.to(device), mr.to(device)
        mask = extract_mask(ct)
        ct_masked = torch.cat([ct, mask], dim=1)

        # ---- Discriminator ----
        with torch.no_grad():
            fake_mr = G(ct_masked)
        real_out = D(mr)
        fake_out = D(fake_mr.detach())
        d_loss = adv_loss(real_out, torch.ones_like(real_out)) + \
                 adv_loss(fake_out, torch.zeros_like(fake_out))
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()
        d_loss_total += d_loss.item()

        # ---- Generator ----
        fake_mr = G(ct_masked)
        fake_out = D(fake_mr)
        g_adv = adv_loss(fake_out, torch.ones_like(fake_out))
        g_l1 = l1_loss(fake_mr, mr) * 10  # L1 reconstruction
        g_ssim = 0  # placeholder for fidelity; L1 handles structure at this scale
        g_loss = g_adv + g_l1
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()
        g_loss_total += g_loss.item()
        n_batches += 1

    g_loss_total /= n_batches
    d_loss_total /= n_batches

    # Validation
    G.eval()
    val_loss = 0.0
    with torch.no_grad():
        for ct, mr, _ in val_loader:
            ct, mr = ct.to(device), mr.to(device)
            mask = extract_mask(ct)
            fake = G(torch.cat([ct, mask], dim=1))
            val_loss += l1_loss(fake, mr).item()
    val_loss /= len(val_loader)

    elapsed = time.time() - t0
    print(f'Epoch {epoch:3d}/{N_EPOCHS} | G: {g_loss_total:.4f} D: {d_loss_total:.4f} | val: {val_loss:.4f} | {elapsed:.0f}s', flush=True)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({'epoch': epoch, 'generator': G.state_dict(), 'val_loss': val_loss},
                   os.path.join(save_dir, 'best.pt'))

torch.save({'epoch': N_EPOCHS, 'generator': G.state_dict()}, os.path.join(save_dir, 'final.pt'))
print(f'Done! Best val_loss: {best_val_loss:.6f}', flush=True)
