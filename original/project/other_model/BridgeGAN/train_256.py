# -*- coding: utf-8 -*-
"""256x256 BridgeGAN: precompute + train + evaluate."""
import sys, os, time, json, torch, numpy as np
sys.path.insert(0, 'D:/Cross-modal conversion')
from other_model.BridgeGAN.model import RefinerGenerator, PatchGANDisc
from selfrdb.dataset import PairedCTMRIDataset
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

device = torch.device('cuda')
SIZE = 256
STEP = 20  # precompute resolution tier

with open('D:/Cross-modal conversion/checkpoints/selfrdb/patient_split.json') as f:
    split = json.load(f)

print(f'Step 1: Check 256x256 data...', flush=True)
# The dataset needs to be regenerated at 256x256 first
ds_test = PairedCTMRIDataset('D:/Cross-modal conversion/data_slices_256', image_size=SIZE, augment=False, patient_ids=split['test'][:1])
print(f'  256x256 data exists: {len(ds_test)} slices', flush=True)
print(f'  If 0, run: python selfrdb/preprocess.py --data-root ../brain --output-root ../data_slices_256 --image-size 256', flush=True)
print(f'  Then run this script again', flush=True)
