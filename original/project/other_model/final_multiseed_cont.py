# -*- coding: utf-8 -*-
"""Multi-seed training continuation: CT seed 2026, BR seeds 123 & 2026."""
import sys, torch, numpy as np, json, os, time, random
sys.path.insert(0,'D:/Cross-modal conversion')
from other_model.BridgeGAN.model import RefinerGenerator
from selfrdb.dataset import PairedCTMRIDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = torch.device('cuda')
with open('D:/Cross-modal conversion/checkpoints/selfrdb/patient_split.json') as f:
    split = json.load(f)

# Simple U-Net
class SimpleUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1=torch.nn.Sequential(torch.nn.Conv2d(1,64,7,padding=3,padding_mode='reflect'),torch.nn.InstanceNorm2d(64),torch.nn.ReLU(True))
        self.enc2=torch.nn.Sequential(torch.nn.Conv2d(64,128,3,stride=2,padding=1),torch.nn.InstanceNorm2d(128),torch.nn.ReLU(True))
        self.res=torch.nn.Sequential(*[torch.nn.Sequential(torch.nn.Conv2d(128,128,3,padding=1,padding_mode='reflect'),torch.nn.InstanceNorm2d(128),torch.nn.ReLU(True),torch.nn.Conv2d(128,128,3,padding=1,padding_mode='reflect'),torch.nn.InstanceNorm2d(128)) for _ in range(4)])
        self.dec1=torch.nn.Sequential(torch.nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),torch.nn.Conv2d(128,64,3,padding=1),torch.nn.InstanceNorm2d(64),torch.nn.ReLU(True))
        self.out=torch.nn.Sequential(torch.nn.Conv2d(64,1,7,padding=3,padding_mode='reflect'),torch.nn.Tanh())
    def forward(self,x):e1=self.enc1(x);e2=self.enc2(e1);r=e2;[r:=r+block(r) for block in self.res];d1=self.dec1(r);return self.out(d1)

def grad_loss(pred,target):
    dx_p=pred[:,:,:,1:]-pred[:,:,:,:-1];dx_t=target[:,:,:,1:]-target[:,:,:,:-1]
    dy_p=pred[:,:,1:,:]-pred[:,:,:-1,:];dy_t=target[:,:,1:,:]-target[:,:,:-1,:]
    return F.l1_loss(dx_p,dx_t)+F.l1_loss(dy_p,dy_t)

coarse_all=np.load('D:/Cross-modal conversion/other_model/BridgeGAN/coarse_mri.npy')

class CDS:
    def __init__(self,ct_dir,coarse_data,image_size=128,augment=False,patient_ids=None):
        import cv2;base_ds=PairedCTMRIDataset(ct_dir,image_size=image_size,augment=False,patient_ids=patient_ids)
        self.pairs=base_ds.pairs;self.image_size=image_size;self.augment=augment;self.coarse=coarse_data
    def __len__(self):return len(self.pairs)
    def _load(self,p):
        import cv2;img=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
        if (img.shape[0],img.shape[1])!=(self.image_size,self.image_size):img=cv2.resize(img,(self.image_size,self.image_size),interpolation=cv2.INTER_AREA)
        return img.astype(np.float32)/127.5-1.0
    def __getitem__(self,idx):
        ct_path,mr_path,name=self.pairs[idx];ct=self._load(ct_path);mr=self._load(mr_path)
        coarse=self.coarse[idx%len(self.coarse)].squeeze()
        if self.augment and random.random()<0.5:ct=np.fliplr(ct).copy();mr=np.fliplr(mr).copy();coarse=np.fliplr(coarse).copy()
        return(torch.from_numpy(ct).unsqueeze(0).float(),torch.from_numpy(mr).unsqueeze(0).float(),torch.from_numpy(coarse).unsqueeze(0).float())

# ===== CT-only seed=2026 =====
print('=== CT-only seed=2026 ===',flush=True)
seed=2026;random.seed(seed);torch.manual_seed(seed)
train_ds=PairedCTMRIDataset('D:/Cross-modal conversion/data_slices',128,True,split['train'])
val_ds=PairedCTMRIDataset('D:/Cross-modal conversion/data_slices',128,False,split['val'])
train_loader=DataLoader(train_ds,batch_size=16,shuffle=True,drop_last=True)
val_loader=DataLoader(val_ds,batch_size=16,shuffle=False)
model=SimpleUNet().to(device);opt=torch.optim.Adam(model.parameters(),lr=2e-4,betas=(0.5,0.999))
best=float('inf');SAVE='D:/Cross-modal conversion/other_model/final_ct_seed2026';os.makedirs(SAVE,exist_ok=True)
for epoch in range(1,21):
    t0=time.time();model.train();tr,nb=0.0,0
    for ct,mr,_ in train_loader:ct,mr=ct.to(device),mr.to(device);loss=F.l1_loss(model(ct),mr);opt.zero_grad();loss.backward();opt.step();tr+=loss.item();nb+=1
    tr/=max(nb,1);model.eval();vl,nv=0.0,0
    with torch.no_grad():
        for ct,mr,_ in val_loader:ct,mr=ct.to(device),mr.to(device);vl+=F.l1_loss(model(ct),mr).item();nv+=1
    vl/=max(nv,1)
    print(f'  Epoch {epoch:3d}/20 | train: {tr:.4f} | val: {vl:.4f} | {time.time()-t0:.0f}s',flush=True)
    if vl<best:best=vl;torch.save({'epoch':epoch,'model':model.state_dict(),'val':vl},os.path.join(SAVE,'best.pt'))
torch.save({'epoch':20,'model':model.state_dict(),'val':best},os.path.join(SAVE,'final.pt'))
print(f'CT seed=2026 done: best={best:.4f}',flush=True)

# ===== BridgeRefine seed=123 =====
print('\n=== BridgeRefine seed=123 ===',flush=True)
seed=123;random.seed(seed);torch.manual_seed(seed)
train_cd=CDS('D:/Cross-modal conversion/data_slices',coarse_all,128,True,split['train'])
val_cd=CDS('D:/Cross-modal conversion/data_slices',coarse_all,128,False,split['val'])
train_cl=DataLoader(train_cd,batch_size=16,shuffle=True,drop_last=True)
val_cl=DataLoader(val_cd,batch_size=16,shuffle=False)
model_br=RefinerGenerator(in_ch=2,out_ch=1,base_ch=64).to(device)
opt_br=torch.optim.Adam(model_br.parameters(),lr=2e-4,betas=(0.5,0.999))
best_br=float('inf');SAVE_BR='D:/Cross-modal conversion/other_model/final_br_seed123';os.makedirs(SAVE_BR,exist_ok=True)
for epoch in range(1,21):
    t0=time.time();model_br.train();tr,nb=0.0,0
    for ct,mr,coarse in train_cl:ct,mr,coarse=ct.to(device),mr.to(device),coarse.to(device);pred=model_br(coarse,ct);loss=F.l1_loss(pred,mr)+0.1*grad_loss(pred,mr);opt_br.zero_grad();loss.backward();opt_br.step();tr+=loss.item();nb+=1
    tr/=max(nb,1);model_br.eval();vl,nv=0.0,0
    with torch.no_grad():
        for ct,mr,coarse in val_cl:ct,mr,coarse=ct.to(device),mr.to(device),coarse.to(device);vl+=F.l1_loss(model_br(coarse,ct),mr).item();nv+=1
    vl/=max(nv,1)
    print(f'  Epoch {epoch:3d}/20 | train: {tr:.4f} | val: {vl:.4f} | {time.time()-t0:.0f}s',flush=True)
    if vl<best_br:best_br=vl;torch.save({'epoch':epoch,'model':model_br.state_dict(),'val':vl},os.path.join(SAVE_BR,'best.pt'))
torch.save({'epoch':20,'model':model_br.state_dict(),'val':best_br},os.path.join(SAVE_BR,'final.pt'))
print(f'BR seed=123 done: best={best_br:.4f}',flush=True)

# ===== BridgeRefine seed=2026 =====
print('\n=== BridgeRefine seed=2026 ===',flush=True)
seed=2026;random.seed(seed);torch.manual_seed(seed)
train_cd2=CDS('D:/Cross-modal conversion/data_slices',coarse_all,128,True,split['train'])
val_cd2=CDS('D:/Cross-modal conversion/data_slices',coarse_all,128,False,split['val'])
train_cl2=DataLoader(train_cd2,batch_size=16,shuffle=True,drop_last=True)
val_cl2=DataLoader(val_cd2,batch_size=16,shuffle=False)
model_br2=RefinerGenerator(in_ch=2,out_ch=1,base_ch=64).to(device)
opt_br2=torch.optim.Adam(model_br2.parameters(),lr=2e-4,betas=(0.5,0.999))
best_br2=float('inf');SAVE_BR2='D:/Cross-modal conversion/other_model/final_br_seed2026';os.makedirs(SAVE_BR2,exist_ok=True)
for epoch in range(1,21):
    t0=time.time();model_br2.train();tr,nb=0.0,0
    for ct,mr,coarse in train_cl2:ct,mr,coarse=ct.to(device),mr.to(device),coarse.to(device);pred=model_br2(coarse,ct);loss=F.l1_loss(pred,mr)+0.1*grad_loss(pred,mr);opt_br2.zero_grad();loss.backward();opt_br2.step();tr+=loss.item();nb+=1
    tr/=max(nb,1);model_br2.eval();vl,nv=0.0,0
    with torch.no_grad():
        for ct,mr,coarse in val_cl2:ct,mr,coarse=ct.to(device),mr.to(device),coarse.to(device);vl+=F.l1_loss(model_br2(coarse,ct),mr).item();nv+=1
    vl/=max(nv,1)
    print(f'  Epoch {epoch:3d}/20 | train: {tr:.4f} | val: {vl:.4f} | {time.time()-t0:.0f}s',flush=True)
    if vl<best_br2:best_br2=vl;torch.save({'epoch':epoch,'model':model_br2.state_dict(),'val':vl},os.path.join(SAVE_BR2,'best.pt'))
torch.save({'epoch':20,'model':model_br2.state_dict(),'val':best_br2},os.path.join(SAVE_BR2,'final.pt'))
print(f'BR seed=2026 done: best={best_br2:.4f}',flush=True)
print('\nAll 3 remaining trainings complete!')
