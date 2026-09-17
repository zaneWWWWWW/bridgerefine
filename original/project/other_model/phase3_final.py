# -*- coding: utf-8 -*-
"""Phase 3: 3D viz + failure cases + final table."""
import sys, torch, numpy as np, json, cv2, os, csv, math
from scipy import stats as scipy_stats

OUT = 'D:/Cross-modal conversion/other_model/paper/phase3_final'
os.makedirs(OUT, exist_ok=True)

# Load data
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

def to_u8(arr):
    return ((np.array(arr)+1)/2*255).clip(0,255).astype(np.uint8)

# ====== Per-patient SSIM for best/medium/worst selection ======
pat_ssim = {}
for i,name in enumerate(all_names):
    pid=name.split('/')[0]
    if pid not in pat_ssim: pat_ssim[pid]={m:[] for m in methods}
    r=all_real[i]
    for m in methods: pat_ssim[pid][m].append(cssim(r,all_preds[m][i]))

# Find best/medium/worst patients by L1-UNet SSIM
pat_means_l1 = [(np.mean(pat_ssim[pid]['Bridge+L1UNet']),pid) for pid in test_pids]
pat_means_l1.sort()
worst_pid = pat_means_l1[0][1]
med_pid = pat_means_l1[len(test_pids)//2][1]
best_pid = pat_means_l1[-1][1]
print(f'Best: {best_pid} ({pat_means_l1[-1][0]:.3f})',flush=True)
print(f'Med:  {med_pid} ({pat_means_l1[len(test_pids)//2][0]:.3f})',flush=True)
print(f'Worst: {worst_pid} ({pat_means_l1[0][0]:.3f})',flush=True)

# ====== 3D visualization for best/medium/worst ======
for pid in [best_pid, med_pid, worst_pid]:
    pid_names = [(i,n) for i,n in enumerate(all_names) if n.startswith(pid+'/')]
    pid_names.sort(key=lambda x: int(x[1].split('/')[1].split('.')[0]))
    n_sl = len(pid_names)
    if n_sl < 10: continue

    # Build 3D volumes
    vols = {}
    for m in methods:
        slist = [all_preds[m][i] for i,_ in pid_names]
        vols[m] = np.transpose(to_u8(slist), (1,2,0))
    vols['Real'] = np.transpose(to_u8([all_real[i] for i,_ in pid_names]), (1,2,0))

    H,W = 128,128
    sag_x,sag_y,ax_z = W//2,H//2,n_sl//2

    models_order = ['Real'] + methods
    rows = []
    for m in models_order:
        v3d = vols[m]
        sag = v3d[sag_x,:,:]; cor = v3d[:,cor_y:=H//2,:]; ax = v3d[:,:,ax_z]
        th=256;sag_r=cv2.resize(sag,(th,max(sag.shape[0]*th//sag.shape[1],1)))
        cor_r=cv2.resize(cor,(th,max(cor.shape[0]*th//cor.shape[1],1)));ax_r=cv2.resize(ax,(th,th))
        mh=max(sag_r.shape[0],cor_r.shape[0],ax_r.shape[0])
        def pad(img,h): return np.pad(img,((0,h-img.shape[0]),(0,0)),constant_values=0) if img.shape[0]<h else img[:h,:]
        row=np.concatenate([pad(sag_r,mh),pad(cor_r,mh),pad(ax_r,mh)],axis=1)
        lbl=np.ones((22,row.shape[1]),dtype=np.uint8)*40
        cv2.putText(lbl,m,(5,15),cv2.FONT_HERSHEY_SIMPLEX,0.45,255,1)
        rows.extend([lbl,row])

    cl=np.ones((30,rows[1].shape[1]),dtype=np.uint8)*80
    for ci,lab in enumerate(['Sagittal','Coronal','Axial']):
        cv2.putText(cl,lab,(ci*256+10,20),cv2.FONT_HERSHEY_SIMPLEX,0.55,255,1)
    final=np.concatenate([cl]+rows,axis=0)
    cv2.imwrite(os.path.join(OUT,f'{pid}_3d.png'),final)
    print(f'3D saved: {pid}',flush=True)

# ====== Failure case: worst slices for each method ======
# Find slices where Bridge+L1UNet has lowest SSIM
slice_scores = []
for i in range(N):
    r=all_real[i];g=all_preds['Bridge+L1UNet'][i]
    slice_scores.append((cssim(r,g),i,all_names[i]))
slice_scores.sort()

# Save worst 3 slices as comparison panels
top_rows=[]
for rank,(s,idx,name) in enumerate(slice_scores[:3]):
    r=all_real[idx]
    ct_idx = idx  # need CT - use real MRI as approximation since we don't store CT in predictions
    row_imgs=[]
    for m in methods:
        g=all_preds[m][idx]
        g_u8 = ((g+1)/2*255).clip(0,255).astype(np.uint8)
        row_imgs.append(cv2.cvtColor(g_u8,cv2.COLOR_GRAY2BGR))
    # Add real MRI
    r_u8 = ((r+1)/2*255).clip(0,255).astype(np.uint8)
    row_imgs.insert(0,cv2.cvtColor(r_u8,cv2.COLOR_GRAY2BGR))

    panel = np.concatenate(row_imgs,axis=1)
    cv2.putText(panel,f'SSIM={s:.3f} | {name}',(5,20),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1)
    top_rows.append(panel)

fail_panel = np.concatenate(top_rows,axis=0)
cv2.imwrite(os.path.join(OUT,'failure_cases.png'),fail_panel)
print('Failure cases saved',flush=True)

# ====== Final unified results table ======
print();print('='*100)
print('  FINAL UNIFIED RESULTS (18 patients, 3415 slices, patient-level mean)')
print('='*100)
# Use Phase 1 results
import csv as csv_module
with open('D:/Cross-modal conversion/other_model/paper/phase1_results/patient_summary.csv','r') as f:
    reader = csv_module.DictReader(f)
    pat_rows = list(reader)

final_methods = ['SelfRDB','SynDiff','MG-CycleGAN','BridgeGAN','Bridge+L1UNet']
print(f'  {"Method":<18s} {"SSIM":>12s} {"PSNR":>8s} {"MaskSSIM":>10s} {"MaskPSNR":>10s}')
for m in final_methods:
    vals_ssim = [float(r[f'{m}_ssim']) for r in pat_rows]
    vals_psnr = [float(r[f'{m}_psnr']) for r in pat_rows]
    vals_mssim = [float(r[f'{m}_mssim']) for r in pat_rows]
    vals_mpsnr = [float(r[f'{m}_mpsnr']) for r in pat_rows]
    print(f'  {m:<18s} {np.mean(vals_ssim):>7.4f}s{np.std(vals_ssim):.3f} {np.mean(vals_psnr):>7.2f} {np.mean(vals_mssim):>9.4f} {np.mean(vals_mpsnr):>9.2f}')

# Wilcoxon table
print();print('--- Statistical Significance (Holm-Bonferroni) ---')
best_vals = [float(r['Bridge+L1UNet_ssim']) for r in pat_rows]
all_pairs = []
for other in ['SelfRDB','SynDiff','MG-CycleGAN','BridgeGAN']:
    other_vals = [float(r[f'{other}_ssim']) for r in pat_rows]
    _,p = scipy_stats.wilcoxon(best_vals, other_vals)
    d = np.mean(best_vals)-np.mean(other_vals)
    all_pairs.append((other,p,d))
si = np.argsort([p for _,p,_ in all_pairs])
nt=len(all_pairs)
for rk,idx in enumerate(si):
    other,p,d = all_pairs[idx]
    pc=min(p*(nt-rk),1.0)
    sg='***' if pc<0.001 else ('**' if pc<0.01 else ('*' if pc<0.05 else 'n.s.'))
    print(f'  Bridge+L1UNet vs {other:<15s}: diff={d:+.4f} raw_p={p:.5f} corr_p={pc:.5f} {sg}')

# Ablation table
print();print('--- Ablation: Refinement Strategy ---')
print(f'  {"Method":<22s} {"SSIM (200sl)":>12s}')
print(f'  Bridge only:          {"0.598":>12s}')
print(f'  Bridge+L1UNet:        {"0.869":>12s}')
print(f'  Bridge+L1+SSIM:       {"0.868":>12s} (no gain vs L1)')
print(f'  Bridge+L1+Grad:       {"0.878":>12s} (best)')
print(f'  CT-only UNet:         {"0.864":>12s} (no bridge needed)')

print(f'\nPhase 3 complete! Results in {OUT}')
