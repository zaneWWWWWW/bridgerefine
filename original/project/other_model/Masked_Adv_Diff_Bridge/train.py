# -*- coding: utf-8 -*-
"""Train Mask-Guided Adversarial Diffusion Bridge with resume support."""
import sys, os, time, json, torch
sys.path.insert(0, 'D:/Cross-modal conversion')
from other_model.Masked_Adv_Diff_Bridge.model import MaskedX0UNet, PatchGANDiscriminator, MaskedAdvDiffusionBridge
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
print(f'Train: {len(train_ds)} slices ({len(train_loader)} batches), Val: {len(val_ds)}', flush=True)

gen = MaskedX0UNet(in_ch=4, out_ch=1, base_ch=64, ch_mult=(1, 2, 4), dropout=0.1).to(device)
disc = PatchGANDiscriminator(in_ch=2).to(device)

SAVE = 'D:/Cross-modal conversion/other_model/Masked_Adv_Diff_Bridge'
os.makedirs(SAVE, exist_ok=True)

ckpt_path = os.path.join(SAVE, 'best.pt')
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    gen.load_state_dict(ckpt['generator'])
    disc.load_state_dict(ckpt['discriminator'])
    start_epoch = ckpt['epoch']
    best_val = ckpt['val_loss']
    print(f'Resumed from epoch {start_epoch}, val_loss={best_val:.6f}', flush=True)
else:
    start_epoch = 0
    best_val = float('inf')

bridge = MaskedAdvDiffusionBridge(gen, disc, timesteps=300, gamma=0.1, device=device)
n_g = sum(p.numel() for p in gen.parameters()); n_d = sum(p.numel() for p in disc.parameters())
print(f'Generator: {n_g:,}  Disc: {n_d:,}', flush=True)

opt_g = torch.optim.AdamW(gen.parameters(), lr=1e-4, weight_decay=1e-5)
opt_d = torch.optim.AdamW(disc.parameters(), lr=2e-5, weight_decay=1e-5)

def extract_mask(ct):
    m = (ct > -0.5).float()
    return F.max_pool2d(m, 3, 1, 1)

N_EPOCHS = 20
print(f'Training epochs {start_epoch+1} to {N_EPOCHS}...', flush=True)

for epoch in range(start_epoch + 1, N_EPOCHS + 1):
    t0 = time.time()
    gen.train()
    train_loss, n_b = 0.0, 0

    for ct, mr, _ in train_loader:
        ct, mr = ct.to(device), mr.to(device)
        mask = extract_mask(ct)

        # Pure reconstruction loss (no adversarial)
        t = torch.randint(1, 301, (ct.shape[0],), device=device)
        xt, _ = bridge.q_sample(mr, ct, t)
        x0_pred1 = gen(xt, t, ct, torch.zeros_like(xt), mask)
        x0_pred2 = gen(xt, t, ct, x0_pred1.detach(), mask)
        loss = F.l1_loss(x0_pred1, mr) + 0.5 * F.l1_loss(x0_pred2, mr)

        opt_g.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        opt_g.step()
        train_loss += loss.item(); n_b += 1

    train_loss /= max(n_b, 1)

    gen.eval()
    val_loss, n_v = 0.0, 0
    with torch.no_grad():
        for ct, mr, _ in val_loader:
            ct, mr = ct.to(device), mr.to(device)
            mask = extract_mask(ct)
            t = torch.randint(1, 301, (ct.shape[0],), device=device)
            xt, _ = bridge.q_sample(mr, ct, t)
            x0_pred = gen(xt, t, ct, torch.zeros_like(xt), mask)
            val_loss += F.l1_loss(x0_pred, mr).item(); n_v += 1
    val_loss /= max(n_v, 1)

    elapsed = time.time() - t0
    print(f'Epoch {epoch:3d}/{N_EPOCHS} | train: {train_loss:.4f} | val: {val_loss:.4f} | {elapsed:.0f}s', flush=True)

    if val_loss < best_val:
        best_val = val_loss
        torch.save({'epoch': epoch, 'generator': gen.state_dict(), 'discriminator': disc.state_dict(), 'val_loss': val_loss}, os.path.join(SAVE, 'best.pt'))
        print(f'  -> best!', flush=True)

    if epoch % 10 == 0:
        torch.save({'epoch': epoch, 'generator': gen.state_dict(), 'discriminator': disc.state_dict()}, os.path.join(SAVE, f'epoch_{epoch}.pt'))

torch.save({'epoch': N_EPOCHS, 'generator': gen.state_dict(), 'discriminator': disc.state_dict()}, os.path.join(SAVE, 'final.pt'))
print(f'Done! Best val_loss: {best_val:.6f}', flush=True)
