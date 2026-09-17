# -*- coding: utf-8 -*-
"""Fix SynDiff: pre-train nd_generator, then joint training."""
import sys, os, time, json, torch
import torch.nn.functional as F
sys.path.insert(0, 'D:/Cross-modal conversion')
from other_model.SynDiff.model import DiffusionUNet, PatchDiscriminator, NonDiffusiveGenerator, SynDiff
from selfrdb.dataset import PairedCTMRIDataset
from torch.utils.data import DataLoader

device = torch.device('cuda')
print(f'Device: {device}', flush=True)

with open('D:/Cross-modal conversion/checkpoints/selfrdb/patient_split.json') as f:
    split = json.load(f)

train_ds = PairedCTMRIDataset('D:/Cross-modal conversion/data_slices', image_size=128, augment=True, patient_ids=split['train'])
val_ds = PairedCTMRIDataset('D:/Cross-modal conversion/data_slices', image_size=128, augment=False, patient_ids=split['val'])
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
print(f'Train: {len(train_ds)}, Val: {len(val_ds)}', flush=True)

save_dir = 'D:/Cross-modal conversion/other_model/SynDiff'

# ====== PHASE 1: Pre-train nd_generator alone ======
print('\n=== PHASE 1: Pre-train Non-Diffusive Generator ===', flush=True)

nd_gen = NonDiffusiveGenerator(in_ch=1, out_ch=1, base_ch=64).to(device)
opt_nd = torch.optim.Adam(nd_gen.parameters(), lr=2e-4)

for epoch in range(1, 6):
    nd_gen.train()
    train_loss, n_b = 0.0, 0
    for ct, mr, _ in train_loader:
        ct, mr = ct.to(device), mr.to(device)
        pred = nd_gen(ct)
        loss = F.l1_loss(pred, mr)
        opt_nd.zero_grad(); loss.backward(); opt_nd.step()
        train_loss += loss.item(); n_b += 1
    train_loss /= max(n_b, 1)

    nd_gen.eval()
    val_loss, n_v = 0.0, 0
    with torch.no_grad():
        for ct, mr, _ in val_loader:
            ct, mr = ct.to(device), mr.to(device)
            val_loss += F.l1_loss(nd_gen(ct), mr).item(); n_v += 1
    val_loss /= max(n_v, 1)
    print(f'  Pre-train Epoch {epoch}/5 | train: {train_loss:.4f} | val: {val_loss:.4f}', flush=True)

torch.save({'nd_gen': nd_gen.state_dict()}, os.path.join(save_dir, 'nd_pretrained.pt'))
print('Pre-training done!', flush=True)

# ====== PHASE 2: Joint training with diffusion ======
print('\n=== PHASE 2: Joint Adversarial Diffusion Training ===', flush=True)

diff_unet = DiffusionUNet(in_ch=2, out_ch=1, base_ch=64).to(device)
disc = PatchDiscriminator(in_ch=2).to(device)
syndiff = SynDiff(diff_unet, disc, nd_gen, n_steps=4, device=device)

opt_g = torch.optim.Adam(list(diff_unet.parameters()) + list(nd_gen.parameters()), lr=1e-4)
opt_d = torch.optim.Adam(disc.parameters(), lr=1e-4)

best_val = float('inf')

for epoch in range(1, 16):
    t0 = time.time()
    diff_unet.train(); nd_gen.train(); disc.train()
    g_loss_t, d_loss_t = 0.0, 0.0

    for ct, mr, _ in train_loader:
        ct, mr = ct.to(device), mr.to(device)
        # Discriminator
        d_loss = syndiff.training_loss_d(mr, ct)
        opt_d.zero_grad(); d_loss.backward(); opt_d.step()
        d_loss_t += d_loss.item()
        # Generator
        g_loss = syndiff.training_loss_g(mr, ct)
        opt_g.zero_grad(); g_loss.backward(); opt_g.step()
        g_loss_t += g_loss.item()

    g_loss_t /= len(train_loader); d_loss_t /= len(train_loader)

    # Val
    nd_gen.eval()
    val_loss, n_v = 0.0, 0
    with torch.no_grad():
        for ct, mr, _ in val_loader:
            ct, mr = ct.to(device), mr.to(device)
            # Use full SynDiff sampling for val
            pred = syndiff.sample(ct)
            val_loss += F.l1_loss(pred, mr).item(); n_v += 1
    val_loss /= max(n_v, 1)

    elapsed = time.time() - t0
    print(f'Epoch {epoch:3d}/15 | G: {g_loss_t:.4f} D: {d_loss_t:.4f} | val: {val_loss:.4f} | {elapsed:.0f}s', flush=True)

    if val_loss < best_val:
        best_val = val_loss
        torch.save({'epoch': epoch+5, 'diff_unet': diff_unet.state_dict(), 'nd_gen': nd_gen.state_dict(), 'val_loss': val_loss}, os.path.join(save_dir, 'best.pt'))

torch.save({'epoch': 20, 'diff_unet': diff_unet.state_dict(), 'nd_gen': nd_gen.state_dict()}, os.path.join(save_dir, 'final.pt'))
print(f'Done! Best val: {best_val:.6f}', flush=True)
