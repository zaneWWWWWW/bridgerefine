# -*- coding: utf-8 -*-
"""Phase 2b: L1+SSIM and L1+Gradient loss ablation (each ~25 min)."""
import sys, os, time, torch, numpy as np
sys.path.insert(0,'D:/Cross-modal conversion')
from other_model.BridgeGAN.model import RefinerGenerator
from selfrdb.dataset import PairedCTMRIDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = torch.device('cuda')

def gradient_loss(pred, target):
    dx_p = pred[:,:,:,1:]-pred[:,:,:,:-1]; dx_t = target[:,:,:,1:]-target[:,:,:,:-1]
    dy_p = pred[:,:,1:,:]-pred[:,:,:-1,:]; dy_t = target[:,:,1:,:]-target[:,:,:-1,:]
    return F.l1_loss(dx_p,dx_t)+F.l1_loss(dy_p,dy_t)

import json, cv2
with open('D:/Cross-modal conversion/checkpoints/selfrdb/patient_split.json') as f:
    split = json.load(f)

# Coarse paired dataset
coarse_all = np.load('D:/Cross-modal conversion/other_model/BridgeGAN/coarse_mri.npy')

class CDS:
    def __init__(self,ct_dir,coarse_data,image_size=128,augment=False,patient_ids=None):
        import cv2;base_ds=PairedCTMRIDataset(ct_dir,image_size=image_size,augment=False,patient_ids=patient_ids)
        self.pairs=base_ds.pairs;self.image_size=image_size;self.augment=augment;self.coarse=coarse_data
    def __len__(self):return len(self.pairs)
    def _load(self,p):
        import cv2;img=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE);img=cv2.resize(img,(self.image_size,self.image_size),interpolation=cv2.INTER_AREA)
        return img.astype(np.float32)/127.5-1.0
    def __getitem__(self,idx):
        import random;ctp,mrp,name=self.pairs[idx];ct=self._load(ctp);mr=self._load(mrp)
        coarse=self.coarse[idx%len(self.coarse)].squeeze()
        if self.augment and random.random()<0.5:
            ct=np.fliplr(ct).copy();mr=np.fliplr(mr).copy();coarse=np.fliplr(coarse).copy()
        return(torch.from_numpy(ct).unsqueeze(0).float(),torch.from_numpy(mr).unsqueeze(0).float(),torch.from_numpy(coarse).unsqueeze(0).float())

train_cd=CDS('D:/Cross-modal conversion/data_slices',coarse_all,128,True,split['train'])
val_cd=CDS('D:/Cross-modal conversion/data_slices',coarse_all,128,False,split['val'])
train_loader=DataLoader(train_cd,batch_size=16,shuffle=True,drop_last=True)
val_loader=DataLoader(val_cd,batch_size=16,shuffle=False)
print(f'Train: {len(train_cd)}, Val: {len(val_cd)}',flush=True)

# --- L1+SSIM ---
print('\n=== L1+SSIM ===',flush=True)
model=RefinerGenerator(in_ch=2,out_ch=1,base_ch=64).to(device)
opt=torch.optim.Adam(model.parameters(),lr=2e-4,betas=(0.5,0.999))
best=float('inf');SAVE='D:/Cross-modal conversion/other_model/phase2_L1_SSIM';os.makedirs(SAVE,exist_ok=True)
for epoch in range(1,16):
    t0=time.time();model.train();tr,nb=0.0,0
    for ct,mr,coarse in train_loader:
        ct,mr,coarse=ct.to(device),mr.to(device),coarse.to(device)
        pred=model(coarse,ct)
        # SSIM approximated via 1-SSIM
        c1,c2=0.0001,0.0009
        mu_x=F.avg_pool2d(pred,11,1);mu_y=F.avg_pool2d(mr,11,1)
        sx=F.avg_pool2d(pred**2,11,1)-mu_x**2;sy=F.avg_pool2d(mr**2,11,1)-mu_y**2
        sxy=F.avg_pool2d(pred*mr,11,1)-mu_x*mu_y
        ssim_m=(2*mu_x*mu_y+c1)*(2*sxy+c2)/(mu_x**2+mu_y**2+c1)/(sx+sy+c2)
        loss=F.l1_loss(pred,mr)+0.5*(1-ssim_m.mean())
        opt.zero_grad();loss.backward();opt.step();tr+=loss.item();nb+=1
    tr/=max(nb,1);model.eval();vl,nv=0.0,0
    with torch.no_grad():
        for ct,mr,coarse in val_loader:
            ct,mr,coarse=ct.to(device),mr.to(device),coarse.to(device)
            vl+=F.l1_loss(model(coarse,ct),mr).item();nv+=1
    vl/=max(nv,1)
    print(f'  Epoch {epoch:3d}/15 | train: {tr:.4f} | val: {vl:.4f} | {time.time()-t0:.0f}s',flush=True)
    if vl<best:best=vl;torch.save({'epoch':epoch,'model':model.state_dict(),'val_loss':vl},os.path.join(SAVE,'best.pt'))
print(f'L1+SSIM best: {best:.4f}',flush=True)

# --- L1+Gradient ---
print('\n=== L1+Gradient ===',flush=True)
model2=RefinerGenerator(in_ch=2,out_ch=1,base_ch=64).to(device)
opt2=torch.optim.Adam(model2.parameters(),lr=2e-4,betas=(0.5,0.999))
best2=float('inf');SAVE2='D:/Cross-modal conversion/other_model/phase2_L1_Grad'
os.makedirs(SAVE2,exist_ok=True)
for epoch in range(1,16):
    t0=time.time();model2.train();tr,nb=0.0,0
    for ct,mr,coarse in train_loader:
        ct,mr,coarse=ct.to(device),mr.to(device),coarse.to(device)
        pred=model2(coarse,ct)
        loss=F.l1_loss(pred,mr)+0.1*gradient_loss(pred,mr)
        opt2.zero_grad();loss.backward();opt2.step();tr+=loss.item();nb+=1
    tr/=max(nb,1);model2.eval();vl,nv=0.0,0
    with torch.no_grad():
        for ct,mr,coarse in val_loader:
            ct,mr,coarse=ct.to(device),mr.to(device),coarse.to(device)
            vl+=F.l1_loss(model2(coarse,ct),mr).item();nv+=1
    vl/=max(nv,1)
    print(f'  Epoch {epoch:3d}/15 | train: {tr:.4f} | val: {vl:.4f} | {time.time()-t0:.0f}s',flush=True)
    if vl<best2:best2=vl;torch.save({'epoch':epoch,'model':model2.state_dict(),'val_loss':vl},os.path.join(SAVE2,'best.pt'))
print(f'L1+Gradient best: {best2:.4f}',flush=True)
print('\nPhase 2b done!')
