# -*- coding: utf-8 -*-
"""Train SelfRDB at 256x256, then BridgeGAN."""
import sys, os, time, json, torch
sys.path.insert(0, 'D:/Cross-modal conversion')
from other_model.SelfRDB.bridge_model import DiffusionBridge
from other_model.SelfRDB.network import X0UNet
from selfrdb.dataset import PairedCTMRIDataset
from torch.utils.data import DataLoader

device = torch.device('cuda')
print(f'Device: {device}', flush=True)

with open('D:/Cross-modal conversion/checkpoints/selfrdb/patient_split.json') as f:
    split = json.load(f)

train_ds = PairedCTMRIDataset('D:/Cross-modal conversion/data_slices_256', image_size=256, augment=True, patient_ids=split['train'])
val_ds = PairedCTMRIDataset('D:/Cross-modal conversion/data_slices_256', image_size=256, augment=False, patient_ids=split['val'])
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
print(f'Train: {len(train_ds)}, Val: {len(val_ds)}', flush=True)

gen = X0UNet(in_ch=3, out_ch=1, base_ch=64,ch_mult=(1,2,4),dropout=0.1).to(device)
bridge = DiffusionBridge(gen, timesteps=300, gamma=0.1, device=device)
opt = torch.optim.AdamW(gen.parameters(), lr=1e-4, weight_decay=1e-5)
n = sum(p.numel() for p in gen.parameters())
print(f'Params: {n:,}', flush=True)

SAVE = 'D:/Cross-modal conversion/other_model/SelfRDB_256'
os.makedirs(SAVE, exist_ok=True)
best_val = float('inf')
N_EPOCHS = 20

for epoch in range(1, N_EPOCHS + 1):
    t0 = time.time()
    gen.train()
    train_loss, n_b = 0.0, 0
    for ct, mr, _ in train_loader:
        ct, mr = ct.to(device), mr.to(device)
        loss = bridge.training_loss(mr, ct)
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
            loss = bridge.training_loss(mr, ct)
            if not torch.isnan(loss):
                val_loss += loss.item(); n_v += 1
    val_loss /= max(n_v, 1)

    elapsed = time.time() - t0
    print(f'Epoch {epoch:3d}/20 | train: {train_loss:.4f} | val: {val_loss:.4f} | {elapsed:.0f}s', flush=True)

    if val_loss < best_val:
        best_val = val_loss
        torch.save({'epoch': epoch, 'generator': gen.state_dict(), 'val_loss': val_loss}, os.path.join(SAVE, 'best.pt'))

torch.save({'epoch': N_EPOCHS, 'generator': gen.state_dict()}, os.path.join(SAVE, 'final.pt'))
print(f'Done! Best val: {best_val:.4f}', flush=True)
