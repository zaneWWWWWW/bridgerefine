# -*- coding: utf-8 -*-
"""Phase 2: CT-only L1 U-Net + loss function ablation."""
import sys, os, time, json, torch, numpy as np
sys.path.insert(0,'D:/Cross-modal conversion')
from other_model.BridgeGAN.model import RefinerGenerator
from selfrdb.dataset import PairedCTMRIDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = torch.device('cuda')
with open('D:/Cross-modal conversion/checkpoints/selfrdb/patient_split.json') as f:
    split = json.load(f)

train_ds = PairedCTMRIDataset('D:/Cross-modal conversion/data_slices', image_size=128, augment=True, patient_ids=split['train'])
val_ds = PairedCTMRIDataset('D:/Cross-modal conversion/data_slices', image_size=128, augment=False, patient_ids=split['val'])
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
print(f'Train: {len(train_ds)}, Val: {len(val_ds)}', flush=True)

def gradient_loss(pred, target):
    dx_pred = pred[:,:,:,1:] - pred[:,:,:,:-1]
    dx_target = target[:,:,:,1:] - target[:,:,:,:-1]
    dy_pred = pred[:,:,1:,:] - pred[:,:,:-1,:]
    dy_target = target[:,:,1:,:] - target[:,:,:-1,:]
    return F.l1_loss(dx_pred, dx_target) + F.l1_loss(dy_pred, dy_target)

def ssim_loss(pred, target):
    # 1-SSIM approximations via local statistics
    c1, c2 = 0.0001, 0.0009
    mu_x = F.avg_pool2d(pred, 11, 1)
    mu_y = F.avg_pool2d(target, 11, 1)
    sx = F.avg_pool2d(pred**2, 11, 1) - mu_x**2
    sy = F.avg_pool2d(target**2, 11, 1) - mu_y**2
    sxy = F.avg_pool2d(pred*target, 11, 1) - mu_x*mu_y
    ssim_map = (2*mu_x*mu_y+c1)*(2*sxy+c2)/(mu_x**2+mu_y**2+c1)/(sx+sy+c2)
    return 1 - ssim_map.mean()

def train_experiment(name, loss_fn, epochs=15):
    model = RefinerGenerator(in_ch=2, out_ch=1, base_ch=64).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
    best_val = float('inf')
    SAVE = f'D:/Cross-modal conversion/other_model/phase2_{name}'
    os.makedirs(SAVE, exist_ok=True)

    for epoch in range(1, epochs+1):
        t0 = time.time()
        model.train()
        train_loss, n_b = 0.0, 0
        for ct, mr, _ in train_loader:
            ct, mr = ct.to(device), mr.to(device)
            refined = model(mr, ct)  # for CT-only: input is MR placeholder
            loss = loss_fn(refined, mr, ct) if 'grad' in name else loss_fn(refined, mr)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item(); n_b += 1
        train_loss /= max(n_b, 1)

        model.eval()
        val_loss, n_v = 0.0, 0
        with torch.no_grad():
            for ct, mr, _ in val_loader:
                ct, mr = ct.to(device), mr.to(device)
                refined = model(mr, ct)
                vl = F.l1_loss(refined, mr)
                val_loss += vl.item(); n_v += 1
        val_loss /= max(n_v, 1)

        elapsed = time.time() - t0
        print(f'  [{name}] Epoch {epoch:3d}/{epochs} | train: {train_loss:.4f} | val: {val_loss:.4f} | {elapsed:.0f}s', flush=True)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'val_loss': val_loss}, os.path.join(SAVE, 'best.pt'))
    return model

# Experiment A: CT-only U-Net (critical baseline)
print('=== A: CT-only L1 U-Net ===', flush=True)

# Simple U-Net: CT(1ch) -> MRI(1ch), no bridge
class SimpleUNet(torch.nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=64):
        super().__init__()
        self.enc1 = torch.nn.Sequential(torch.nn.Conv2d(in_ch,base_ch,7,padding=3,padding_mode='reflect'),torch.nn.InstanceNorm2d(base_ch),torch.nn.ReLU(True))
        self.enc2 = torch.nn.Sequential(torch.nn.Conv2d(base_ch,base_ch*2,3,stride=2,padding=1),torch.nn.InstanceNorm2d(base_ch*2),torch.nn.ReLU(True))
        self.res = torch.nn.Sequential(*[torch.nn.Sequential(
            torch.nn.Conv2d(base_ch*2,base_ch*2,3,padding=1,padding_mode='reflect'),torch.nn.InstanceNorm2d(base_ch*2),torch.nn.ReLU(True),
            torch.nn.Conv2d(base_ch*2,base_ch*2,3,padding=1,padding_mode='reflect'),torch.nn.InstanceNorm2d(base_ch*2)) for _ in range(4)])
        self.dec1 = torch.nn.Sequential(torch.nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),torch.nn.Conv2d(base_ch*2,base_ch,3,padding=1),torch.nn.InstanceNorm2d(base_ch),torch.nn.ReLU(True))
        self.out = torch.nn.Sequential(torch.nn.Conv2d(base_ch,out_ch,7,padding=3,padding_mode='reflect'),torch.nn.Tanh())
    def forward(self, x):
        e1=self.enc1(x);e2=self.enc2(e1);r=e2
        for block in self.res: r=r+block(r)
        d1=self.dec1(r);return self.out(d1)

ct_model = SimpleUNet(in_ch=1, out_ch=1, base_ch=64).to(device)
opt_ct = torch.optim.Adam(ct_model.parameters(), lr=2e-4, betas=(0.5,0.999))
best_ct = float('inf')
SAVE_CT = 'D:/Cross-modal conversion/other_model/phase2_CT_only_L1'
os.makedirs(SAVE_CT, exist_ok=True)

for epoch in range(1, 21):
    t0 = time.time()
    ct_model.train()
    train_loss, n_b = 0.0, 0
    for ct, mr, _ in train_loader:
        ct, mr = ct.to(device), mr.to(device)
        loss = F.l1_loss(ct_model(ct), mr)
        opt_ct.zero_grad(); loss.backward(); opt_ct.step()
        train_loss += loss.item(); n_b += 1
    train_loss /= max(n_b, 1)

    ct_model.eval()
    val_loss, n_v = 0.0, 0
    with torch.no_grad():
        for ct, mr, _ in val_loader:
            ct, mr = ct.to(device), mr.to(device)
            val_loss += F.l1_loss(ct_model(ct), mr).item(); n_v += 1
    val_loss /= max(n_v, 1)

    elapsed = time.time() - t0
    print(f'  [CT-only] Epoch {epoch:3d}/20 | train: {train_loss:.4f} | val: {val_loss:.4f} | {elapsed:.0f}s', flush=True)

    if val_loss < best_ct:
        best_ct = val_loss
        torch.save({'epoch': epoch, 'model': ct_model.state_dict(), 'val_loss': val_loss}, os.path.join(SAVE_CT, 'best.pt'))

print(f'CT-only best val: {best_ct:.4f}', flush=True)

# Experiment B: Bridge + L1 + SSIM
print('\n=== B: Bridge + L1+SSIM ===', flush=True)
# Load coarse MRIs
coarse_all = np.load('D:/Cross-modal conversion/other_model/BridgeGAN/coarse_mri.npy')

class CoarsePairedDataset:
    def __init__(self, ct_dir, coarse_data, image_size=128, augment=False, patient_ids=None):
        import cv2
        base_ds = PairedCTMRIDataset(ct_dir, image_size=image_size, augment=False, patient_ids=patient_ids)
        self.pairs = base_ds.pairs; self.image_size = image_size
        self.augment = augment; self.coarse = coarse_data
    def __len__(self): return len(self.pairs)
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
        ct = self._load(ct_path); mr = self._load(mr_path)
        coarse = self.coarse[idx % len(self.coarse)].squeeze()
        if self.augment and random.random() < 0.5:
            ct=np.fliplr(ct).copy();mr=np.fliplr(mr).copy();coarse=np.fliplr(coarse).copy()
        return (torch.from_numpy(ct).unsqueeze(0).float(),
                torch.from_numpy(mr).unsqueeze(0).float(),
                torch.from_numpy(coarse).unsqueeze(0).float())

train_cd = CoarsePairedDataset('D:/Cross-modal conversion/data_slices', coarse_all, 128, True, split['train'])
val_cd = CoarsePairedDataset('D:/Cross-modal conversion/data_slices', coarse_all, 128, False, split['val'])
train_cl = DataLoader(train_cd, batch_size=16, shuffle=True, drop_last=True)
val_cl = DataLoader(val_cd, batch_size=16, shuffle=False)

# L1+SSIM
model_ssim = RefinerGenerator(in_ch=2, out_ch=1, base_ch=64).to(device)
opt_ssim = torch.optim.Adam(model_ssim.parameters(), lr=2e-4, betas=(0.5,0.999))
best_ssim = float('inf')
SAVE_SSIM = 'D:/Cross-modal conversion/other_model/phase2_L1_SSIM'
os.makedirs(SAVE_SSIM, exist_ok=True)
for epoch in range(1, 16):
    t0=time.time(); model_ssim.train()
    tr, nb = 0.0, 0
    for ct, mr, coarse in train_cl:
        ct,mr,coarse=ct.to(device),mr.to(device),coarse.to(device)
        pred=model_ssim(coarse,ct)
        loss=F.l1_loss(pred,mr)+0.5*ssim_loss(pred,mr)
        opt_ssim.zero_grad();loss.backward();opt_ssim.step()
        tr+=loss.item();nb+=1
    tr/=max(nb,1)
    model_ssim.eval();vl,nv=0.0,0
    with torch.no_grad():
        for ct,mr,coarse in val_cl:
            ct,mr,coarse=ct.to(device),mr.to(device),coarse.to(device)
            vl+=F.l1_loss(model_ssim(coarse,ct),mr).item();nv+=1
    vl/=max(nv,1)
    print(f'  [L1+SSIM] Epoch {epoch:3d}/15 | train: {tr:.4f} | val: {vl:.4f} | {time.time()-t0:.0f}s', flush=True)
    if vl<best_ssim: best_ssim=vl; torch.save({'epoch':epoch,'model':model_ssim.state_dict(),'val_loss':vl},os.path.join(SAVE_SSIM,'best.pt'))

print(f'L1+SSIM best val: {best_ssim:.4f}', flush=True)

# L1+Gradient
print('\n=== C: Bridge + L1+Gradient ===', flush=True)
model_grad = RefinerGenerator(in_ch=2, out_ch=1, base_ch=64).to(device)
opt_grad = torch.optim.Adam(model_grad.parameters(), lr=2e-4, betas=(0.5,0.999))
best_grad = float('inf')
SAVE_GRAD = 'D:/Cross-modal conversion/other_model/phase2_L1_Grad'
os.makedirs(SAVE_GRAD, exist_ok=True)
for epoch in range(1, 16):
    t0=time.time(); model_grad.train()
    tr, nb = 0.0, 0
    for ct,mr,coarse in train_cl:
        ct,mr,coarse=ct.to(device),mr.to(device),coarse.to(device)
        pred=model_grad(coarse,ct)
        loss=F.l1_loss(pred,mr)+0.1*gradient_loss(pred,mr)
        opt_grad.zero_grad();loss.backward();opt_grad.step()
        tr+=loss.item();nb+=1
    tr/=max(nb,1)
    model_grad.eval();vl,nv=0.0,0
    with torch.no_grad():
        for ct,mr,coarse in val_cl:
            ct,mr,coarse=ct.to(device),mr.to(device),coarse.to(device)
            vl+=F.l1_loss(model_grad(coarse,ct),mr).item();nv+=1
    vl/=max(nv,1)
    print(f'  [L1+Grad] Epoch {epoch:3d}/15 | train: {tr:.4f} | val: {vl:.4f} | {time.time()-t0:.0f}s', flush=True)
    if vl<best_grad: best_grad=vl; torch.save({'epoch':epoch,'model':model_grad.state_dict(),'val_loss':vl},os.path.join(SAVE_GRAD,'best.pt'))

print(f'L1+Grad best val: {best_grad:.4f}', flush=True)
print('\nPhase 2 complete!')
