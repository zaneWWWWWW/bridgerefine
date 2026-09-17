# -*- coding: utf-8 -*-
"""Pre-compute coarse MRI from frozen SelfRDB bridge."""
import sys, os, torch, numpy as np
sys.path.insert(0, 'D:/Cross-modal conversion')
from other_model.SelfRDB.bridge_model import DiffusionBridge
from other_model.SelfRDB.network import X0UNet
from selfrdb.dataset import PairedCTMRIDataset
from torch.utils.data import DataLoader
import json

device = torch.device('cuda')
print(f'Device: {device}', flush=True)

with open('D:/Cross-modal conversion/checkpoints/selfrdb/patient_split.json') as f:
    split = json.load(f)

# Load frozen bridge
gen = X0UNet(in_ch=3, out_ch=1, base_ch=64, ch_mult=(1,2,4)).to(device)
ckpt = torch.load('D:/Cross-modal conversion/other_model/SelfRDB/selfrdb_best.pt', map_location=device, weights_only=False)
gen.load_state_dict(ckpt['generator']); gen.eval()
bridge = DiffusionBridge(gen, timesteps=300, gamma=0.1, device=device)
print(f'Bridge loaded', flush=True)

# Load training data (no augment for pre-computation)
ds = PairedCTMRIDataset('D:/Cross-modal conversion/data_slices', image_size=128, augment=False, patient_ids=split['train'])
loader = DataLoader(ds, batch_size=16, shuffle=False)
print(f'{len(ds)} training slices', flush=True)

OUT = 'D:/Cross-modal conversion/other_model/BridgeGAN/coarse_mri.npy'
all_coarse = []

for i, (ct, mr, name) in enumerate(loader):
    ct_dev = ct.to(device)
    with torch.no_grad():
        coarse = bridge.sample_ddib(ct_dev, num_steps=10, num_recursions=2)
    all_coarse.append(coarse.cpu().numpy())
    if (i+1) % 100 == 0:
        print(f'  {i+1}/{len(loader)} batches', flush=True)

data = np.concatenate(all_coarse, axis=0)
np.save(OUT, data)
print(f'Saved: {OUT} ({data.shape}, {data.nbytes/1024/1024:.0f}MB)', flush=True)
