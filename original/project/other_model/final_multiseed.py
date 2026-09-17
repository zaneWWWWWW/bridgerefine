# -*- coding: utf-8 -*-
"""Final: Multi-seed training for CT-only UNet + BridgeRefine refiner."""
import sys, torch, numpy as np, json, os, time, random
sys.path.insert(0,'D:/Cross-modal conversion')
from other_model.BridgeGAN.model import RefinerGenerator
from selfrdb.dataset import PairedCTMRIDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = torch.device('cuda')
SEEDS = [42, 123, 2026]

with open('D:/Cross-modal conversion/checkpoints/selfrdb/patient_split.json') as f:
    split = json.load(f)

# Dataset
train_ds = PairedCTMRIDataset('D:/Cross-modal conversion/data_slices', image_size=128, augment=True, patient_ids=split['train'])
val_ds = PairedCTMRIDataset('D:/Cross-modal conversion/data_slices', image_size=128, augment=False, patient_ids=split['val'])
test_ds = PairedCTMRIDataset('D:/Cross-modal conversion/data_slices', image_size=128, augment=False, patient_ids=split['test'])

# CT-only U-Net class
class SimpleUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1=torch.nn.Sequential(torch.nn.Conv2d(1,64,7,padding=3,padding_mode='reflect'),torch.nn.InstanceNorm2d(64),torch.nn.ReLU(True))
        self.enc2=torch.nn.Sequential(torch.nn.Conv2d(64,128,3,stride=2,padding=1),torch.nn.InstanceNorm2d(128),torch.nn.ReLU(True))
        self.res=torch.nn.Sequential(*[torch.nn.Sequential(torch.nn.Conv2d(128,128,3,padding=1,padding_mode='reflect'),torch.nn.InstanceNorm2d(128),torch.nn.ReLU(True),torch.nn.Conv2d(128,128,3,padding=1,padding_mode='reflect'),torch.nn.InstanceNorm2d(128)) for _ in range(4)])
        self.dec1=torch.nn.Sequential(torch.nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),torch.nn.Conv2d(128,64,3,padding=1),torch.nn.InstanceNorm2d(64),torch.nn.ReLU(True))
        self.out=torch.nn.Sequential(torch.nn.Conv2d(64,1,7,padding=3,padding_mode='reflect'),torch.nn.Tanh())
    def forward(self,x):e1=self.enc1(x);e2=self.enc2(e1);r=e2;[r:=r+block(r) for block in self.res];d1=self.dec1(r);return self.out(d1)

# ===== CT-only U-Net: seeds 123, 2026 (seed 42 already done) =====
print('=== CT-only U-Net multi-seed ===', flush=True)
for seed in [123, 2026]:
    random.seed(seed); torch.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    model = SimpleUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5,0.999))
    best_val = float('inf'); SAVE = f'D:/Cross-modal conversion/other_model/final_ct_seed{seed}'
    os.makedirs(SAVE, exist_ok=True)

    for epoch in range(1, 21):
        model.train(); tr, nb = 0.0, 0
        for ct, mr, _ in train_loader:
            ct,mr=ct.to(device),mr.to(device)
            loss=F.l1_loss(model(ct),mr)
            opt.zero_grad(); loss.backward(); opt.step(); tr+=loss.item(); nb+=1
        tr/=max(nb,1)
        model.eval(); vl,nv=0.0,0
        with torch.no_grad():
            for ct,mr,_ in val_loader:
                ct,mr=ct.to(device),mr.to(device);vl+=F.l1_loss(model(ct),mr).item();nv+=1
        vl/=max(nv,1)
        if epoch%5==0: print(f'  [CT seed={seed}] Epoch {epoch:3d}/20 | train: {tr:.4f} | val: {vl:.4f} | {time.time():.0f}s', flush=True)
        if vl<best_val: best_val=vl; torch.save({'epoch':epoch,'model':model.state_dict(),'val':vl},os.path.join(SAVE,'best.pt'))
    torch.save({'epoch':20,'model':model.state_dict(),'val':best_val},os.path.join(SAVE,'final.pt'))
    print(f'  CT seed={seed} done: best_val={best_val:.4f}', flush=True)

# ===== BridgeRefine refiner: seeds 123, 2026 (seed 42 already done) =====
print('\n=== BridgeRefine refiner multi-seed ===', flush=True)
coarse_all = np.load('D:/Cross-modal conversion/other_model/BridgeGAN/coarse_mri.npy')

class CDS:
    def __init__(self,ct_dir,coarse_data,image_size=128,augment=False,patient_ids=None):
        import cv2; base_ds=PairedCTMRIDataset(ct_dir,image_size=image_size,augment=False,patient_ids=patient_ids)
        self.pairs=base_ds.pairs;self.image_size=image_size;self.augment=augment;self.coarse=coarse_data
    def __len__(self): return len(self.pairs)
    def _load(self,p):
        import cv2;img=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
        if (img.shape[0],img.shape[1])!=(self.image_size,self.image_size): img=cv2.resize(img,(self.image_size,self.image_size),interpolation=cv2.INTER_AREA)
        return img.astype(np.float32)/127.5-1.0
    def __getitem__(self,idx):
        ct_path,mr_path,name=self.pairs[idx];ct=self._load(ct_path);mr=self._load(mr_path);coarse=self.coarse[idx%len(self.coarse)].squeeze()
        if self.augment and random.random()<0.5: ct=np.fliplr(ct).copy();mr=np.fliplr(mr).copy();coarse=np.fliplr(coarse).copy()
        return(torch.from_numpy(ct).unsqueeze(0).float(),torch.from_numpy(mr).unsqueeze(0).float(),torch.from_numpy(coarse).unsqueeze(0).float())

def gradient_loss(pred,target):
    dx_p=pred[:,:,:,1:]-pred[:,:,:,:-1];dx_t=target[:,:,:,1:]-target[:,:,:,:-1]
    dy_p=pred[:,:,1:,:]-pred[:,:,:-1,:];dy_t=target[:,:,1:,:]-target[:,:,:-1,:]
    return F.l1_loss(dx_p,dx_t)+F.l1_loss(dy_p,dy_t)

for seed in [123, 2026]:
    random.seed(seed); torch.manual_seed(seed)
    train_cd = CDS('D:/Cross-modal conversion/data_slices',coarse_all,128,True,split['train'])
    val_cd = CDS('D:/Cross-modal conversion/data_slices',coarse_all,128,False,split['val'])
    train_cl = DataLoader(train_cd,batch_size=16,shuffle=True,drop_last=True)
    val_cl = DataLoader(val_cd,batch_size=16,shuffle=False)

    model = RefinerGenerator(in_ch=2,out_ch=1,base_ch=64).to(device)
    opt = torch.optim.Adam(model.parameters(),lr=2e-4,betas=(0.5,0.999))
    best_val = float('inf'); SAVE = f'D:/Cross-modal conversion/other_model/final_br_seed{seed}'
    os.makedirs(SAVE,exist_ok=True)

    for epoch in range(1, 21):
        model.train(); tr,nb=0.0,0
        for ct,mr,coarse in train_cl:
            ct,mr,coarse=ct.to(device),mr.to(device),coarse.to(device)
            pred=model(coarse,ct)
            loss=F.l1_loss(pred,mr)+0.1*gradient_loss(pred,mr)
            opt.zero_grad();loss.backward();opt.step();tr+=loss.item();nb+=1
        tr/=max(nb,1)
        model.eval();vl,nv=0.0,0
        with torch.no_grad():
            for ct,mr,coarse in val_cl:
                ct,mr,coarse=ct.to(device),mr.to(device),coarse.to(device)
                vl+=F.l1_loss(model(coarse,ct),mr).item();nv+=1
        vl/=max(nv,1)
        if epoch%5==0: print(f'  [BR seed={seed}] Epoch {epoch:3d}/20 | train: {tr:.4f} | val: {vl:.4f}', flush=True)
        if vl<best_val: best_val=vl; torch.save({'epoch':epoch,'model':model.state_dict(),'val':vl},os.path.join(SAVE,'best.pt'))
    torch.save({'epoch':20,'model':model.state_dict(),'val':best_val},os.path.join(SAVE,'final.pt'))
    print(f'  BR seed={seed} done: best_val={best_val:.4f}', flush=True)

print('\nMulti-seed training complete!')
