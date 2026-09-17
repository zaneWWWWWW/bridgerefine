# -*- coding: utf-8 -*-
"""
Train BridgeGAN Refiner using pre-computed coarse MRIs.
"""
import sys, os, time, json, torch, numpy as np
sys.path.insert(0, 'D:/Cross-modal conversion')
from other_model.BridgeGAN.model import RefinerGenerator, PatchGANDisc
from selfrdb.dataset import PairedCTMRIDataset
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

device = torch.device('cuda')
print(f'Device: {device}', flush=True)

with open('D:/Cross-modal conversion/checkpoints/selfrdb/patient_split.json') as f:
    split = json.load(f)

# Load pre-computed coarse MRIs
coarse_all = np.load('D:/Cross-modal conversion/other_model/BridgeGAN/coarse_mri.npy')
print(f'Coarse MRI: {coarse_all.shape}', flush=True)

# Create paired dataset: coarse + ct -> real mr
class CoarsePairedDataset(Dataset):
    def __init__(self, ct_dir, coarse_data, image_size=128, augment=False, patient_ids=None):
        base_ds = PairedCTMRIDataset(ct_dir, image_size=image_size, augment=False, patient_ids=patient_ids)
        self.pairs = base_ds.pairs  # list of (ct_path, mr_path, name)
        self.image_size = image_size
        self.augment = augment
        self.coarse = coarse_data

    def __len__(self):
        return len(self.pairs)

    def _load(self, path):
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None: raise ValueError(f'Cannot read: {path}')
        if (img.shape[0], img.shape[1]) != (self.image_size, self.image_size):
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return img.astype(np.float32) / 127.5 - 1.0

    def __getitem__(self, idx):
        import random
        ct_path, mr_path, name = self.pairs[idx]
        ct = self._load(ct_path)
        mr = self._load(mr_path)
        coarse = self.coarse[idx % len(self.coarse)].squeeze()
        if self.augment and random.random() < 0.5:
            ct = np.fliplr(ct).copy(); mr = np.fliplr(mr).copy(); coarse = np.fliplr(coarse).copy()
        ct=np.clip(ct,-1,1); mr=np.clip(mr,-1,1); coarse=np.clip(coarse,-1,1)
        return (torch.from_numpy(ct).unsqueeze(0).float(),
                torch.from_numpy(mr).unsqueeze(0).float(),
                torch.from_numpy(coarse).unsqueeze(0).float())

# Load data
train_ds = CoarsePairedDataset('D:/Cross-modal conversion/data_slices', coarse_all, image_size=128, augment=True, patient_ids=split['train'])
val_ds = CoarsePairedDataset('D:/Cross-modal conversion/data_slices', coarse_all, image_size=128, augment=False, patient_ids=split['val'])
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
print(f'Train: {len(train_ds)}, Val: {len(val_ds)}', flush=True)

# GAN refiner
refiner = RefinerGenerator(in_ch=2, out_ch=1, base_ch=64).to(device)
disc = PatchGANDisc(in_ch=1).to(device)
n_r = sum(p.numel() for p in refiner.parameters())
n_d = sum(p.numel() for p in disc.parameters())
print(f'Refiner: {n_r:,}  Disc: {n_d:,}', flush=True)

opt_g = torch.optim.Adam(refiner.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))

SAVE = 'D:/Cross-modal conversion/other_model/BridgeGAN'
os.makedirs(SAVE, exist_ok=True)
best_val = float('inf')
N_EPOCHS = 20

for epoch in range(1, N_EPOCHS + 1):
    t0 = time.time()
    refiner.train(); disc.train()
    g_loss_t, d_loss_t = 0.0, 0.0

    for ct, mr, coarse in train_loader:
        ct, mr, coarse = ct.to(device), mr.to(device), coarse.to(device)

        refined = refiner(coarse, ct)
        d_real = disc(mr)
        d_fake = disc(refined.detach())
        d_loss = F.mse_loss(d_real, torch.ones_like(d_real)) + \
                 F.mse_loss(d_fake, torch.zeros_like(d_fake))
        opt_d.zero_grad(); d_loss.backward(); opt_d.step()
        d_loss_t += d_loss.item()

        refined = refiner(coarse, ct)
        d_out = disc(refined)
        g_adv = F.mse_loss(d_out, torch.ones_like(d_out))
        g_l1 = F.l1_loss(refined, mr) * 20
        g_loss = g_adv + g_l1
        opt_g.zero_grad(); g_loss.backward(); opt_g.step()
        g_loss_t += g_loss.item()

    g_loss_t /= len(train_loader); d_loss_t /= len(train_loader)

    refiner.eval()
    val_loss, n_v = 0.0, 0
    with torch.no_grad():
        for ct, mr, coarse in val_loader:
            ct, mr, coarse = ct.to(device), mr.to(device), coarse.to(device)
            refined = refiner(coarse, ct)
            val_loss += F.l1_loss(refined, mr).item(); n_v += 1
    val_loss /= max(n_v, 1)

    elapsed = time.time() - t0
    print(f'Epoch {epoch:3d}/{N_EPOCHS} | G: {g_loss_t:.4f} D: {d_loss_t:.4f} | val: {val_loss:.4f} | {elapsed:.0f}s', flush=True)

    if val_loss < best_val:
        best_val = val_loss
        torch.save({'epoch': epoch, 'refiner': refiner.state_dict(), 'disc': disc.state_dict(), 'val_loss': val_loss}, os.path.join(SAVE, 'best.pt'))

torch.save({'epoch': N_EPOCHS, 'refiner': refiner.state_dict()}, os.path.join(SAVE, 'final.pt'))
print(f'Done! Best val: {best_val:.6f}', flush=True)
