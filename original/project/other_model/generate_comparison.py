# -*- coding: utf-8 -*-
"""Generate final 4-model comparison figures."""
import sys, os, torch, math, json
import cv2, numpy as np
sys.path.insert(0, 'D:/Cross-modal conversion')
device = torch.device('cuda')

with open('D:/Cross-modal conversion/checkpoints/selfrdb/patient_split.json') as f:
    split = json.load(f)
test_pids = split['test'][:3]

from selfrdb.dataset import PairedCTMRIDataset
from torch.utils.data import DataLoader
ds = PairedCTMRIDataset('D:/Cross-modal conversion/data_slices', image_size=128, augment=False, patient_ids=test_pids)
loader = DataLoader(ds, batch_size=1, shuffle=False)
print(f'Test: {len(ds)} slices', flush=True)

# Load all 4 models
from diffusion_model import ControlUNet, GaussianDiffusion
ctrl = ControlUNet(in_ch=1, out_ch=1, cond_ch=2, base_ch=64, ch_mult=(1,2,4)).to(device)
ctrl.load_state_dict(torch.load('D:/Cross-modal conversion/checkpoints/controlnet_best.pt', map_location=device, weights_only=False)['model'])
ctrl.eval(); diff_ctrl = GaussianDiffusion(ctrl, timesteps=500, device=device)
print('[0] ControlNet', flush=True)

from other_model.SelfRDB.bridge_model import DiffusionBridge
from other_model.SelfRDB.network import X0UNet
gen1 = X0UNet(in_ch=3, out_ch=1, base_ch=64, ch_mult=(1,2,4)).to(device)
gen1.load_state_dict(torch.load('D:/Cross-modal conversion/other_model/SelfRDB/selfrdb_best.pt', map_location=device, weights_only=False)['generator'])
gen1.eval(); bridge1 = DiffusionBridge(gen1, timesteps=300, gamma=0.1, device=device)
print('[1] SelfRDB', flush=True)

from other_model.MG_CycleGAN.model import Generator as GAN_Gen
gen2 = GAN_Gen(in_ch=2, out_ch=1, base_ch=64, n_res=6).to(device)
gen2.load_state_dict(torch.load('D:/Cross-modal conversion/other_model/MG_CycleGAN/best.pt', map_location=device, weights_only=False)['generator'])
gen2.eval()
print('[2] MG-CycleGAN', flush=True)

from other_model.SynDiff.model import DiffusionUNet, NonDiffusiveGenerator, SynDiff, PatchDiscriminator
nd = NonDiffusiveGenerator(in_ch=1, out_ch=1, base_ch=64).to(device)
du = DiffusionUNet(in_ch=2, out_ch=1, base_ch=64).to(device)
ckpt3 = torch.load('D:/Cross-modal conversion/other_model/SynDiff/best.pt', map_location=device, weights_only=False)
du.load_state_dict(ckpt3['diff_unet']); nd.load_state_dict(ckpt3['nd_gen'])
du.eval(); nd.eval()
syndiff = SynDiff(du, PatchDiscriminator(in_ch=2).to(device), nd, n_steps=4, device=device)
print('[3] SynDiff', flush=True)

def extract_mask(ct):
    m = (ct > -0.5).float()
    return torch.nn.functional.max_pool2d(m, 3, 1, 1)

def compute_ssim(a, b):
    a01 = (a+1)/2; b01 = (b+1)/2
    c1, c2 = 0.0001, 0.0009
    mu_x = cv2.GaussianBlur(a01, (11,11), 1.5); mu_y = cv2.GaussianBlur(b01, (11,11), 1.5)
    sx = cv2.GaussianBlur(a01**2, (11,11), 1.5) - mu_x**2
    sy = cv2.GaussianBlur(b01**2, (11,11), 1.5) - mu_y**2
    sxy = cv2.GaussianBlur(a01*b01, (11,11), 1.5) - mu_x*mu_y
    return float(np.mean((2*mu_x*mu_y+c1)*(2*sxy+c2)/(mu_x**2+mu_y**2+c1)/(sx+sy+c2)))

results = {m: {'ssim':[], 'psnr':[]} for m in ['ControlNet','SelfRDB','MG-CycleGAN','SynDiff']}
slices_data = []

for i, (ct, mr, name) in enumerate(loader):
    if i >= 40: break
    ct_d, mr_d = ct.to(device), mr.to(device)
    mr_np = mr_d.squeeze().cpu().numpy()
    mask = extract_mask(ct_d)
    with torch.no_grad():
        gc = diff_ctrl.ddim_sample(ct_d, torch.zeros(1,1,128,128,device=device), ddim_steps=20)
        g1 = bridge1.sample_ddib(ct_d, num_steps=20, num_recursions=5)
        g2 = gen2(torch.cat([ct_d, mask], dim=1))
        g3 = syndiff.sample(ct_d)
    ct_np = ct.squeeze().cpu().numpy()
    gens = [gc, g1, g2, g3]
    if i < 3:
        slices_data.append((ct_np, mr_np, [g.squeeze().cpu().numpy() for g in gens], name[0]))
    for label, gen_out in [('ControlNet', gc), ('SelfRDB', g1), ('MG-CycleGAN', g2), ('SynDiff', g3)]:
        gen_np = gen_out.squeeze().cpu().numpy()
        ssim = compute_ssim(mr_np, gen_np)
        mse = np.mean(((mr_np+1)/2 - (gen_np+1)/2)**2)
        psnr = 10 * math.log10(1.0 / max(mse, 1e-12))
        results[label]['ssim'].append(ssim); results[label]['psnr'].append(psnr)

print(f'Metrics computed for 40 slices', flush=True)

# ===== FIGURES =====
OUT = 'D:/Cross-modal conversion/other_model'
font = cv2.FONT_HERSHEY_SIMPLEX
def to_u8(t):
    return ((t + 1) * 127.5).clip(0, 255).astype(np.uint8)
def to_heat(diff):
    d = (diff / max(diff.max(), 0.01) * 255).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_HOT)

# FIG 1: 3-slice comparison grid
row_h, col_w = 128, 128
fig1_rows = []
for ct_np, mr_np, gens, name in slices_data:
    ct_u = to_u8(ct_np); mr_u = to_u8(mr_np)
    gen_imgs = [to_u8(g) for g in gens]
    diffs = [to_heat(np.abs(mr_u.astype(float) - g.astype(float))) for g in gen_imgs]
    cols = [cv2.cvtColor(ct_u, cv2.COLOR_GRAY2BGR), cv2.cvtColor(mr_u, cv2.COLOR_GRAY2BGR)]
    cols += [cv2.cvtColor(g, cv2.COLOR_GRAY2BGR) for g in gen_imgs]
    cols += diffs
    fig1_rows.append(np.concatenate(cols, axis=1))

labels = ['CT','True MRI','ControlNet','SelfRDB','MG-CycGAN','SynDiff','C-Err','S-Err','M-Err','D-Err']
header = np.ones((28, col_w*len(labels), 3), dtype=np.uint8)*240
for i, lab in enumerate(labels):
    ts = cv2.getTextSize(lab, font, 0.4, 1)[0]
    cv2.putText(header, lab, (i*col_w+(col_w-ts[0])//2, 18), font, 0.4, (0,0,0), 1)
cv2.imwrite(os.path.join(OUT, 'final_grid.png'), np.concatenate([header]+fig1_rows, axis=0))
print('Fig1: grid', flush=True)

# FIG 2: PSNR + SSIM bar chart
fig2 = np.ones((450, 750, 3), dtype=np.uint8)*255
models = ['ControlNet','SelfRDB','SynDiff','MG-CycleGAN']
colors = [(200,60,60),(60,100,200),(200,150,60),(40,160,40)]
psnrs = [np.mean(results[m]['psnr']) for m in models]
ssims = [np.mean(results[m]['ssim']) for m in models]

for i, (m, p, s, c) in enumerate(zip(models, psnrs, ssims, colors)):
    x = 60 + i*165
    ph = int(p/30*280); sh = int(s*200)
    # PSNR bar
    cv2.rectangle(fig2, (x, 340-ph), (x+60, 340), c, -1)
    cv2.putText(fig2, f'{p:.1f}dB', (x-5, 330-ph), font, 0.6, c, 2)
    # SSIM bar (smaller, beside)
    cv2.rectangle(fig2, (x+70, 340-sh), (x+100, 340), c, 2)
    cv2.putText(fig2, f'{s:.3f}', (x+70, 330-sh), font, 0.5, c, 2)
    # Labels
    for j, nl in enumerate(m.split('-')):
        cv2.putText(fig2, nl if j==0 else ('-'+nl), (x, 370+j*18), font, 0.5, (0,0,0), 1)

cv2.putText(fig2, 'Brain CT-to-MRI Translation: 4-Model Comparison', (120, 35), font, 0.9, (0,0,0), 2)
cv2.putText(fig2, 'PSNR (solid bars, dB)     SSIM (outline bars)', (120, 65), font, 0.5, (100,100,100), 1)
cv2.putText(fig2, '20-epoch lightweight reproduction on 180-patient dataset', (120, 85), font, 0.5, (100,100,100), 1)
cv2.imwrite(os.path.join(OUT, 'final_metrics.png'), fig2)
print('Fig2: metrics', flush=True)

# FIG 3: Detail panel - best slice
ct_np, mr_np, gens, name = slices_data[1]
ct_u, mr_u = to_u8(ct_np), to_u8(mr_np)
gen_imgs = [to_u8(g) for g in gens]
diffs = [to_heat(np.abs(mr_u.astype(float)-g.astype(float))) for g in gen_imgs]

panel = np.ones((128*3+15, 128*2+5, 3), dtype=np.uint8)*255
# Row 1: CT | True MRI
panel[0:128, 0:128] = cv2.cvtColor(ct_u, cv2.COLOR_GRAY2BGR)
panel[0:128, 133:261] = cv2.cvtColor(mr_u, cv2.COLOR_GRAY2BGR)
# Row 2: ControlNet | SelfRDB
panel[133:261, 0:128] = cv2.cvtColor(gen_imgs[0], cv2.COLOR_GRAY2BGR)
panel[133:261, 133:261] = cv2.cvtColor(gen_imgs[1], cv2.COLOR_GRAY2BGR)
# Row 3: MG-CycleGAN | SynDiff
panel[266:394, 0:128] = cv2.cvtColor(gen_imgs[2], cv2.COLOR_GRAY2BGR)
panel[266:394, 133:261] = cv2.cvtColor(gen_imgs[3], cv2.COLOR_GRAY2BGR)

labels_p = [(0,0,'CT'),(0,133,'True MRI'),(133,0,'ControlNet DDPM'),(133,133,'SelfRDB'),
            (266,0,'MG-CycleGAN'),(266,133,'SynDiff')]
for y,x,t in labels_p:
    cv2.putText(panel, t, (x+3, y+14), font, 0.4, (0,255,0), 1)
cv2.imwrite(os.path.join(OUT, 'final_detail.png'), panel)
print('Fig3: detail', flush=True)

# Print summary
print()
print('='*60)
print('  FINAL 4-MODEL COMPARISON (40 slices)')
print('='*60)
for m in models:
    print(f'  {m:<15s}  SSIM={np.mean(results[m]["ssim"]):.4f}  PSNR={np.mean(results[m]["psnr"]):.1f}dB')
print(f'  {"-"*45}')
print(f'  SelfRDB still training (epoch 21/50), MG-CycleGAN & SynDiff done')
print(f'  Figures saved to: other_model/')
