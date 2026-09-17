# -*- coding: utf-8 -*-
"""Continue SelfRDB training from checkpoint."""
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

train_ds = PairedCTMRIDataset('D:/Cross-modal conversion/data_slices', image_size=128, augment=True, patient_ids=split['train'])
val_ds = PairedCTMRIDataset('D:/Cross-modal conversion/data_slices', image_size=128, augment=False, patient_ids=split['val'])
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)
print(f'Train: {len(train_ds)}, Val: {len(val_ds)}', flush=True)

gen = X0UNet(in_ch=3, out_ch=1, base_ch=64, ch_mult=(1,2,4), dropout=0.1).to(device)
ckpt = torch.load('D:/Cross-modal conversion/other_model/SelfRDB/selfrdb_best.pt', map_location=device, weights_only=False)
gen.load_state_dict(ckpt['generator'])
start_epoch = ckpt.get('epoch', 5)
print(f'Resume from epoch {start_epoch}', flush=True)

bridge = DiffusionBridge(gen, timesteps=300, gamma=0.1, device=device)
opt = torch.optim.AdamW(gen.parameters(), lr=3e-5, weight_decay=1e-5)

save_dir = 'D:/Cross-modal conversion/other_model/SelfRDB'
best_val_loss = ckpt.get('val_loss', float('inf'))

log = open(os.path.join(save_dir, 'training_log.txt'), 'a')
def log_msg(msg):
    t = time.strftime('%H:%M:%S')
    line = f'{t} - {msg}'
    print(line, flush=True)
    log.write(line + '\n'); log.flush()

for epoch in range(start_epoch + 1, 51):
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

    val_str = 'N/A'
    if epoch % 5 == 0:
        gen.eval()
        val_loss, n_v = 0.0, 0
        with torch.no_grad():
            for ct, mr, _ in val_loader:
                ct, mr = ct.to(device), mr.to(device)
                loss = bridge.training_loss(mr, ct)
                if not torch.isnan(loss):
                    val_loss += loss.item(); n_v += 1
        val_loss /= max(n_v, 1)
        val_str = f'{val_loss:.6f}'
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'epoch': epoch, 'generator': gen.state_dict(), 'val_loss': val_loss}, os.path.join(save_dir, 'selfrdb_best.pt'))

    elapsed = time.time() - t0
    log_msg(f'Epoch {epoch:3d}/50 | train: {train_loss:.6f} | val: {val_str} | {elapsed:.0f}s')

    if epoch % 10 == 0:
        torch.save({'epoch': epoch, 'generator': gen.state_dict()}, os.path.join(save_dir, f'selfrdb_epoch_{epoch}.pt'))

log.close()
print(f'Done! Best: {best_val_loss:.6f}', flush=True)
