# -*- coding: utf-8 -*-
"""Phase 1b: CPU metrics from saved predictions + real mask."""
import sys, torch, numpy as np, json, cv2, math, os, csv, nibabel
from scipy import stats as scipy_stats
from pathlib import Path

OUT = 'D:/Cross-modal conversion/other_model/paper/phase1_results'
os.makedirs(OUT, exist_ok=True)

# Load saved data
data = torch.load('D:/Cross-modal conversion/other_model/paper/improved/predictions.pt', map_location='cpu', weights_only=False)
l1_data = torch.load('D:/Cross-modal conversion/other_model/paper/phase1_l1_preds.pt', map_location='cpu', weights_only=False)

all_real = data['real']; all_preds = data['preds']; all_names = data['names']
all_preds['Bridge+L1UNet'] = l1_data['l1_preds']
methods = ['SelfRDB','SynDiff','MG-CycleGAN','BridgeGAN','Bridge+L1UNet']
N = len(all_real)

with open('D:/Cross-modal conversion/checkpoints/selfrdb/patient_split.json') as f:
    test_pids = sorted(json.load(f)['test'])

def cssim(r,g):
    r01=(r+1)/2;g01=(g+1)/2;c1,c2=0.0001,0.0009
    mu_x=cv2.GaussianBlur(r01,(11,11),1.5);mu_y=cv2.GaussianBlur(g01,(11,11),1.5)
    sx=cv2.GaussianBlur(r01**2,(11,11),1.5)-mu_x**2;sy=cv2.GaussianBlur(g01**2,(11,11),1.5)-mu_y**2
    sxy=cv2.GaussianBlur(r01*g01,(11,11),1.5)-mu_x*mu_y
    return float(np.mean((2*mu_x*mu_y+c1)*(2*sxy+c2)/(mu_x**2+mu_y**2+c1)/(sx+sy+c2)))

# Load real masks
print('Loading masks...', flush=True)
patient_masks = {}
for pid in test_pids:
    p = Path(f'D:/Cross-modal conversion/brain/{pid}/mask.nii.gz')
    if p.exists():
        m = nibabel.load(str(p)).get_fdata()
        patient_masks[pid] = m

# Per-slice + per-patient metrics
pat_data = {m:{} for m in methods}
slice_records = []
for i, name in enumerate(all_names):
    pid, sid = name.split('/'); sn = int(sid.split('.')[0])
    r = all_real[i]
    if pid not in pat_data[methods[0]]:
        for m in methods: pat_data[m][pid] = {'ssim':[],'psnr':[],'mssim':[],'mpsnr':[]}

    # Get real mask
    rmask = None
    if pid in patient_masks and sn < patient_masks[pid].shape[2]:
        ms = patient_masks[pid][:,:,sn]
        ms = cv2.resize(ms.astype(np.float32),(128,128),interpolation=cv2.INTER_NEAREST)
        rmask = (ms>0).astype(np.float32)

    for m in methods:
        g = all_preds[m][i]
        s_full = cssim(r,g)
        mse = np.mean(((r+1)/2-(g+1)/2)**2)
        p_full = 10*math.log10(1/max(mse,1e-12))

        s_mask = s_full; p_mask = p_full
        if rmask is not None and rmask.sum()>100:
            rm=((r+1)/2)*rmask; gm=((g+1)/2)*rmask
            mse_m=np.mean(((rm-gm)[rmask>0])**2)
            p_mask=10*math.log10(1/max(mse_m,1e-12))
            mu_x=cv2.GaussianBlur(rm,(11,11),1.5);mu_y=cv2.GaussianBlur(gm,(11,11),1.5)
            sx=cv2.GaussianBlur(rm**2,(11,11),1.5)-mu_x**2;sy=cv2.GaussianBlur(gm**2,(11,11),1.5)-mu_y**2
            sxy=cv2.GaussianBlur(rm*gm,(11,11),1.5)-mu_x*mu_y
            c1m,c2m=0.0001,0.0009
            s_mask=float(np.mean((2*mu_x*mu_y+c1m)*(2*sxy+c2m)/(mu_x**2+mu_y**2+c1m)/(sx+sy+c2m)))

        pat_data[m][pid]['ssim'].append(s_full)
        pat_data[m][pid]['psnr'].append(p_full)
        pat_data[m][pid]['mssim'].append(s_mask)
        pat_data[m][pid]['mpsnr'].append(p_mask)

    if i%500==0: print(f'  {i}/{N}', flush=True)

# Patient-level aggregation
pat_means = {m:{k:[] for k in ['ssim','psnr','mssim','mpsnr']} for m in methods}
for pid in test_pids:
    for m in methods:
        for k in ['ssim','psnr','mssim','mpsnr']:
            pat_means[m][k].append(np.mean(pat_data[m][pid][k]))

# Inter-slice consistency
print('Inter-slice...', flush=True)
for pid in test_pids:
    pid_recs = sorted([r for r in slice_records if r.get('pid','')==pid], key=lambda x: x.get('sn',0))
    # Use the pat_data structure instead
    slices_v = pat_data[methods[0]][pid]['ssim']  # number of slices for this patient
    n_sl = len(slices_v)
    # Match slices from all_names
    pid_names = [(j,n) for j,n in enumerate(all_names) if n.startswith(pid+'/')]
    pid_names.sort(key=lambda x: int(x[1].split('/')[1].split('.')[0]))
    for k in range(len(pid_names)-1):
        i1,i2 = pid_names[k][0], pid_names[k+1][0]
        for m in methods:
            g1,g2 = all_preds[m][i1], all_preds[m][i2]
            mad = np.mean(np.abs(g1-g2))
            if 'imad' not in pat_means[m]: pat_means[m]['imad']=[]
            pat_means[m]['imad'].append(mad)

# ====== REPORT ======
print();print('='*90)
print(f'  UNIFIED PATIENT-LEVEL RESULTS (n={len(test_pids)}, {N} slices)')
print('='*90)
print(f'  {"Method":<18s} {"SSIM":>10s} {"PSNR":>8s} {"MaskSSIM":>10s} {"MaskPSNR":>10s} {"InterMAD":>10s}')
for m in methods:
    s=np.mean(pat_means[m]['ssim']);ss=np.std(pat_means[m]['ssim'])
    p=np.mean(pat_means[m]['psnr']);ms=np.mean(pat_means[m]['mssim'])
    mp=np.mean(pat_means[m]['mpsnr']);im=np.mean(pat_means[m]['imad'])
    print(f'  {m:<18s} {s:>6.4f}s{ss:.3f} {p:>7.2f} {ms:>9.4f} {mp:>9.2f} {im:>10.6f}')

# Holm-Bonferroni
print();print('--- Wilcoxon + Holm (Bridge+L1UNet vs others) ---')
pvals=[];comps=[]
for other in ['SelfRDB','SynDiff','MG-CycleGAN','BridgeGAN']:
    _,p=scipy_stats.wilcoxon(pat_means['Bridge+L1UNet']['ssim'],pat_means[other]['ssim'])
    pvals.append(p);comps.append(other)
si=np.argsort(pvals);nt=len(pvals)
for r,idx in enumerate(si):
    pr=pvals[idx];pc=min(pr*(nt-r),1.0)
    sg='***' if pc<0.001 else ('**' if pc<0.01 else ('*' if pc<0.05 else 'n.s.'))
    d=np.mean(pat_means['Bridge+L1UNet']['ssim'])-np.mean(pat_means[comps[idx]]['ssim'])
    print(f'  vs {comps[idx]:<15s}: raw_p={pr:.5f} corr_p={pc:.5f} diff={d:+.4f} {sg}')

# 95% CI
best=np.array(pat_means['Bridge+L1UNet']['ssim'])
ci=1.96*best.std()/np.sqrt(len(best))
print(f'\n  Bridge+L1UNet: {best.mean():.4f}s{best.std():.4f} [95%CI {best.mean()-ci:.3f}-{best.mean()+ci:.3f}]')

# Save CSVs
with open(os.path.join(OUT,'patient_summary.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['patient']+[f'{m}_{k}' for m in methods for k in ['ssim','psnr','mssim','mpsnr']])
    w.writeheader()
    for ip,pid in enumerate(test_pids):
        row={'patient':pid}
        for m in methods:
            for k in ['ssim','psnr','mssim','mpsnr']: row[f'{m}_{k}']=pat_means[m][k][ip]
        w.writerow(row)
print(f'\nSaved to {OUT}', flush=True)
