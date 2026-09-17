# -*- coding: utf-8 -*-
"""Phase 1a: Generate L1 U-Net predictions, save to file. Resumable."""
import sys, torch, numpy as np, json, os
sys.path.insert(0,'D:/Cross-modal conversion')
device = torch.device('cuda')

with open('D:/Cross-modal conversion/checkpoints/selfrdb/patient_split.json') as f:
    split = json.load(f)

from other_model.SelfRDB.bridge_model import DiffusionBridge
from other_model.SelfRDB.network import X0UNet
from other_model.BridgeGAN.model import RefinerGenerator
from selfrdb.dataset import PairedCTMRIDataset
from torch.utils.data import DataLoader

OUT = 'D:/Cross-modal conversion/other_model/paper/phase1_l1_preds.pt'

# Resume checkpoint
if os.path.exists(OUT):
    print('Loading existing predictions...')
    saved = torch.load(OUT, map_location='cpu', weights_only=False)
    l1_preds = saved['l1_preds']
    start_idx = len(l1_preds)
    print(f'Resuming from {start_idx}/{3415}')
else:
    l1_preds = []
    start_idx = 0

if start_idx >= 3415:
    print('Already complete!')
    sys.exit(0)

# Load models
gen_s = X0UNet(in_ch=3,out_ch=1,base_ch=64,ch_mult=(1,2,4)).to(device)
gen_s.load_state_dict(torch.load('D:/Cross-modal conversion/other_model/SelfRDB/selfrdb_best.pt',map_location=device,weights_only=False)['generator']);gen_s.eval()
bridge_s = DiffusionBridge(gen_s,timesteps=300,gamma=0.1,device=device)

l1_ref = RefinerGenerator(in_ch=2,out_ch=1,base_ch=64).to(device)
l1_ref.load_state_dict(torch.load('D:/Cross-modal conversion/other_model/BridgeGAN_L1/best.pt',map_location=device,weights_only=False)['refiner']);l1_ref.eval()

ds = PairedCTMRIDataset('D:/Cross-modal conversion/data_slices',image_size=128,augment=False,patient_ids=split['test'])
loader = DataLoader(ds,batch_size=1,shuffle=False)

for i,(ct,mr,name) in enumerate(loader):
    if i < start_idx: continue
    ct_d = ct.to(device)
    with torch.no_grad():
        coarse = bridge_s.sample_ddib(ct_d,num_steps=20,num_recursions=5)
        ref_out = l1_ref(coarse,ct_d)
    l1_preds.append(ref_out.squeeze().cpu().numpy())
    if (i+1) % 500 == 0:
        torch.save({'l1_preds': l1_preds}, OUT)
        print(f'Saved {i+1}/{len(loader)}', flush=True)

torch.save({'l1_preds': l1_preds}, OUT)
print(f'Done! {len(l1_preds)} predictions saved', flush=True)
