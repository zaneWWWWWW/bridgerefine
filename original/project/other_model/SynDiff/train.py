# -*- coding: utf-8 -*-
"""SynDiff training: Adversarial Diffusion for CT→MRI translation."""
import sys, os, time, json
import torch
sys.path.insert(0, 'D:/Cross-modal conversion')
from selfrdb.dataset import PairedCTMRIDataset
from torch.utils.data import DataLoader
from model import DiffusionUNet, PatchDiscriminator, NonDiffusiveGenerator, SynDiff

device = torch.device('cuda')
print(f'Device: {device}', flush=True)

with open('D:/Cross-modal conversion/checkpoints/selfrdb/patient_split.json') as f:
    split = json.load(f)

IMAGE_SIZE, BATCH_SIZE, N_EPOCHS, N_STEPS = 128, 8, 20, 4

train_ds = PairedCTMRIDataset('D:/Cross-modal conversion/data_slices', image_size=IMAGE_SIZE, augment=True, patient_ids=split['train'])
val_ds = PairedCTMRIDataset('D:/Cross-modal conversion/data_slices', image_size=IMAGE_SIZE, augment=False, patient_ids=split['val'])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
print(f'Train: {len(train_ds)}, Val: {len(val_ds)}', flush=True)

# Non-diffusive module (fast initial estimate)
nd_gen = NonDiffusiveGenerator(in_ch=1, out_ch=1, base_ch=64).to(device)
# Diffusive module (adversarial diffusion refinement)
diff_unet = DiffusionUNet(in_ch=2, out_ch=1, base_ch=64).to(device)
disc = PatchDiscriminator(in_ch=2).to(device)

syndiff = SynDiff(diff_unet, disc, nd_gen, n_steps=N_STEPS, device=device)

n_params = sum(p.numel() for p in diff_unet.parameters()) + sum(p.numel() for p in nd_gen.parameters())
print(f'Params: {n_params:,}', flush=True)

opt_g = torch.optim.Adam(list(diff_unet.parameters()) + list(nd_gen.parameters()), lr=1e-4)
opt_d = torch.optim.Adam(disc.parameters(), lr=1e-4)

save_dir = 'D:/Cross-modal conversion/checkpoints/syndiff'
os.makedirs(save_dir, exist_ok=True)
best_val_loss = float('inf')

for epoch in range(1, N_EPOCHS + 1):
    t0 = time.time()
    diff_unet.train(); nd_gen.train(); disc.train()
    g_loss_t, d_loss_t = 0.0, 0.0

    for ct, mr, _ in train_loader:
        ct, mr = ct.to(device), mr.to(device)
        # Train discriminator
        d_loss = syndiff.training_loss_d(mr, ct)
        opt_d.zero_grad(); d_loss.backward(); opt_d.step()
        d_loss_t += d_loss.item()
        # Train generator
        g_loss = syndiff.training_loss_g(mr, ct)
        opt_g.zero_grad(); g_loss.backward(); opt_g.step()
        g_loss_t += g_loss.item()

    g_loss_t /= len(train_loader); d_loss_t /= len(train_loader)

    # Val
    diff_unet.eval(); nd_gen.eval()
    val_loss = 0.0
    with torch.no_grad():
        for ct, mr, _ in val_loader:
            ct, mr = ct.to(device), mr.to(device)
            x0_init = nd_gen(ct)
            val_loss += torch.nn.functional.l1_loss(x0_init, mr).item()
    val_loss /= len(val_loader)

    elapsed = time.time() - t0
    print(f'Epoch {epoch:3d}/{N_EPOCHS} | G: {g_loss_t:.4f} D: {d_loss_t:.4f} | val: {val_loss:.4f} | {elapsed:.0f}s', flush=True)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({'epoch': epoch, 'diff_unet': diff_unet.state_dict(), 'nd_gen': nd_gen.state_dict(), 'val_loss': val_loss}, os.path.join(save_dir, 'best.pt'))

torch.save({'epoch': N_EPOCHS, 'diff_unet': diff_unet.state_dict(), 'nd_gen': nd_gen.state_dict()}, os.path.join(save_dir, 'final.pt'))
print(f'Done! Best val_loss: {best_val_loss:.6f}', flush=True)
