# -*- coding: utf-8 -*-
"""
Phase 1: Unified metrics for all methods + L1 U-Net full eval + real mask ROI.
No retraining - uses saved predictions + runs L1 U-Net inference.
"""
import sys, torch, numpy as np, json, cv2, math, os, csv, nibabel
from scipy import stats as scipy_stats
from pathlib import Path
sys.path.insert(0, 'D:/Cross-modal conversion')

device = torch.device('cuda')
OUT = 'D:/Cross-modal conversion/other_model/paper/phase1_results'
os.makedirs(OUT, exist_ok=True)

# Load patient split
with open('D:/Cross-modal conversion/checkpoints/selfrdb/patient_split.json') as f:
    split = json.load(f)
test_pids = sorted(split['test'])

# Load saved predictions
data = torch.load('D:/Cross-modal conversion/other_model/paper/improved/predictions.pt', map_location='cpu', weights_only=False)
all_real = data['real']; all_preds = data['preds']; all_names = data['names']
N = len(all_real)

# Generate L1 U-Net predictions (need GPU for bridge sampling)
print('Generating L1 U-Net predictions...', flush=True)
from other_model.SelfRDB.bridge_model import DiffusionBridge
from other_model.SelfRDB.network import X0UNet
from other_model.BridgeGAN.model import RefinerGenerator
from selfrdb.dataset import PairedCTMRIDataset
from torch.utils.data import DataLoader

gen_s = X0UNet(in_ch=3, out_ch=1, base_ch=64, ch_mult=(1,2,4)).to(device)
gen_s.load_state_dict(torch.load('D:/Cross-modal conversion/other_model/SelfRDB/selfrdb_best.pt', map_location=device, weights_only=False)['generator']); gen_s.eval()
bridge_s = DiffusionBridge(gen_s, timesteps=300, gamma=0.1, device=device)

l1_ref = RefinerGenerator(in_ch=2, out_ch=1, base_ch=64).to(device)
l1_ref.load_state_dict(torch.load('D:/Cross-modal conversion/other_model/BridgeGAN_L1/best.pt', map_location=device, weights_only=False)['refiner']); l1_ref.eval()

ds = PairedCTMRIDataset('D:/Cross-modal conversion/data_slices', image_size=128, augment=False, patient_ids=test_pids)
loader = DataLoader(ds, batch_size=1, shuffle=False)

l1_preds = []
for i, (ct, mr, name) in enumerate(loader):
    ct_d = ct.to(device)
    with torch.no_grad():
        coarse = bridge_s.sample_ddib(ct_d, num_steps=20, num_recursions=5)
        ref_out = l1_ref(coarse, ct_d)
    l1_preds.append(ref_out.squeeze().cpu().numpy())
    if (i+1) % 500 == 0: print(f'  L1 infer {i+1}/{N}', flush=True)

print('L1 predictions done', flush=True)

# Add L1 to preds dict
all_preds['Bridge+L1-UNet'] = l1_preds

# Remove ControlNet (no predictions saved for it), add L1
methods = ['SelfRDB', 'SynDiff', 'MG-CycleGAN', 'BridgeGAN', 'Bridge+L1-UNet']

# ====== Load real mask.nii.gz for each patient ======
print('Loading real masks...', flush=True)
patient_masks = {}
for pid in test_pids:
    mask_path = Path(f'D:/Cross-modal conversion/brain/{pid}/mask.nii.gz')
    if mask_path.exists():
        mask_nii = nibabel.load(str(mask_path))
        mask_data = mask_nii.get_fdata()
        # Store transposed to match axial slicing order
        patient_masks[pid] = mask_data
    else:
        print(f'  WARN: No mask for {pid}', flush=True)

# ====== Metrics computation ======
def cssim(r, g):
    r01=(r+1)/2; g01=(g+1)/2; c1,c2=0.0001,0.0009
    mu_x=cv2.GaussianBlur(r01,(11,11),1.5); mu_y=cv2.GaussianBlur(g01,(11,11),1.5)
    sx=cv2.GaussianBlur(r01**2,(11,11),1.5)-mu_x**2; sy=cv2.GaussianBlur(g01**2,(11,11),1.5)-mu_y**2
    sxy=cv2.GaussianBlur(r01*g01,(11,11),1.5)-mu_x*mu_y
    return float(np.mean((2*mu_x*mu_y+c1)*(2*sxy+c2)/(mu_x**2+mu_y**2+c1)/(sx+sy+c2)))

# For each slice, map to patient and slice index
print('Computing all metrics...', flush=True)
slice_records = []
pat_slices = {}

for i, name in enumerate(all_names):
    pid, sid_str = name.split('/')
    sn = int(sid_str.split('.')[0])

    r = all_real[i]

    # Get real mask for this slice
    real_mask = None
    if pid in patient_masks:
        mask_vol = patient_masks[pid]
        if sn < mask_vol.shape[2]:
            mask_slice = mask_vol[:, :, sn]
            # Resize to 128x128
            mask_slice = cv2.resize(mask_slice.astype(np.float32), (128, 128), interpolation=cv2.INTER_NEAREST)
            real_mask = (mask_slice > 0).astype(np.float32)

    if pid not in pat_slices:
        pat_slices[pid] = {'slices': [], 'methods': {m: [] for m in methods}}

    rec = {'pid': pid, 'sn': sn}

    for m in methods:
        g = all_preds[m][i]
        # Full metrics
        s_full = cssim(r, g)
        mse_full = np.mean(((r+1)/2 - (g+1)/2)**2)
        p_full = 10 * math.log10(1 / max(mse_full, 1e-12))

        # Mask metrics
        s_mask = s_full; p_mask = p_full
        if real_mask is not None and real_mask.sum() > 100:
            r_masked = ((r+1)/2) * real_mask
            g_masked = ((g+1)/2) * real_mask
            mse_m = np.mean(((r_masked - g_masked)[real_mask > 0])**2)
            p_mask = 10 * math.log10(1 / max(mse_m, 1e-12))
            # Masked SSIM
            mu_x=cv2.GaussianBlur(r_masked,(11,11),1.5); mu_y=cv2.GaussianBlur(g_masked,(11,11),1.5)
            sx=cv2.GaussianBlur(r_masked**2,(11,11),1.5)-mu_x**2; sy=cv2.GaussianBlur(g_masked**2,(11,11),1.5)-mu_y**2
            sxy=cv2.GaussianBlur(r_masked*g_masked,(11,11),1.5)-mu_x*mu_y
            c1,c2=0.0001,0.0009
            s_mask = float(np.mean((2*mu_x*mu_y+c1)*(2*sxy+c2)/(mu_x**2+mu_y**2+c1)/(sx+sy+c2)))

        rec[f'{m}_ssim'] = s_full; rec[f'{m}_psnr'] = p_full
        rec[f'{m}_mssim'] = s_mask; rec[f'{m}_mpsnr'] = p_mask

        pat_slices[pid]['methods'][m].append({'ssim': s_full, 'psnr': p_full, 'mssim': s_mask, 'mpsnr': p_mask})

    slice_records.append(rec)

# ====== Patient-level aggregation ======
pat_means = {m: {'ssim': [], 'psnr': [], 'mssim': [], 'mpsnr': []} for m in methods}
for pid in sorted(pat_slices):
    for m in methods:
        vals = pat_slices[pid]['methods'][m]
        for k in ['ssim', 'psnr', 'mssim', 'mpsnr']:
            pat_means[m][k].append(np.mean([v[k] for v in vals]))

# ====== Inter-slice consistency ======
print('Computing inter-slice consistency...', flush=True)
for pid in sorted(pat_slices):
    slices_data = pat_slices[pid]['methods']
    # Sort by slice number from slice_records
    pid_recs = [r for r in slice_records if r['pid'] == pid]
    pid_recs.sort(key=lambda x: x['sn'])
    for k in range(len(pid_recs) - 1):
        r1, r2 = pid_recs[k], pid_recs[k+1]
        for m in methods:
            sname1 = f"{pid}/{r1['sn']:04d}"; idx1 = all_names.index(sname1) if sname1 in all_names else next(i for i,n in enumerate(all_names) if n.startswith(f"{pid}/{r1['sn']:04d}")); g1 = all_preds[m][idx1]
            sname2 = f"{pid}/{r2['sn']:04d}"; idx2 = all_names.index(sname2) if sname2 in all_names else next(i for i,n in enumerate(all_names) if n.startswith(f"{pid}/{r2['sn']:04d}")); g2 = all_preds[m][idx2]
            mad = np.mean(np.abs(g1 - g2))
            is_ssim = cssim(g1, g2)
            if 'inter_mad' not in pat_means[m]:
                pat_means[m]['inter_mad'] = []
                pat_means[m]['inter_ssim'] = []
            pat_means[m]['inter_mad'].append(mad)
            pat_means[m]['inter_ssim'].append(is_ssim)

# ====== REPORT ======
print()
print('='*90)
print(f'  UNIFIED PATIENT-LEVEL RESULTS (n={len(test_pids)} patients, {N} slices)')
print('='*90)

# Main table
header = f'  {"Method":<18s} {"SSIM":>8s} {"PSNR":>8s} {"Mask SSIM":>10s} {"Mask PSNR":>10s} {"Inter-MAD/Real":>13s}'
print(header)
print('  ' + '-'*70)
real_mad = np.mean(pat_means['SelfRDB']['inter_mad'])  # use any method as proxy - all use same real
for m in methods:
    s = np.mean(pat_means[m]['ssim'])
    ss = np.std(pat_means[m]['ssim'])
    p = np.mean(pat_means[m]['psnr'])
    ms = np.mean(pat_means[m]['mssim'])
    mp = np.mean(pat_means[m]['mpsnr'])
    mad_r = np.mean(pat_means[m]['inter_mad'])
    print(f'  {m:<18s} {s:>7.4f}s{ss:.3f} {p:>7.2f} {ms:>9.4f} {mp:>9.2f} {mad_r:>12.6f}')

# Wilcoxon + Holm correction
print()
print('--- Wilcoxon + Holm-Bonferroni ---')
all_pvals = []
all_comps = []
for other in ['SelfRDB', 'SynDiff', 'MG-CycleGAN', 'BridgeGAN']:
    stat, p = scipy_stats.wilcoxon(pat_means['Bridge+L1-UNet']['ssim'], pat_means[other]['ssim'])
    all_pvals.append(p)
    all_comps.append(f'L1 vs {other}')

# Holm correction
sorted_idx = np.argsort(all_pvals)
n_tests = len(all_pvals)
for rank, idx in enumerate(sorted_idx):
    p_raw = all_pvals[idx]
    p_corr = min(p_raw * (n_tests - rank), 1.0)
    sig = '***' if p_corr < 0.001 else ('**' if p_corr < 0.01 else ('*' if p_corr < 0.05 else 'n.s.'))
    d = np.mean(pat_means['Bridge+L1-UNet']['ssim']) - np.mean(pat_means[all_comps[idx].split('vs ')[1]]['ssim'])
    print(f'  {all_comps[idx]:<25s}: raw_p={p_raw:.5f} corr_p={p_corr:.5f} diff={d:+.4f} {sig}')

# 95% CI for best method
best_vals = pat_means['Bridge+L1-UNet']['ssim']
ci = 1.96 * np.std(best_vals) / np.sqrt(len(best_vals))
print(f'\n  Bridge+L1-UNet 95% CI: [{np.mean(best_vals)-ci:.3f}, {np.mean(best_vals)+ci:.3f}]')

# Save unified CSV
with open(os.path.join(OUT, 'slice_level_metrics.csv'), 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=slice_records[0].keys())
    writer.writeheader()
    writer.writerows(slice_records)

# Save patient-level summary
with open(os.path.join(OUT, 'patient_level_summary.csv'), 'w', newline='') as f:
    fieldnames = ['patient'] + [f'{m}_{k}' for m in methods for k in ['ssim','psnr','mssim','mpsnr']]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for i, pid in enumerate(sorted(pat_slices)):
        row = {'patient': pid}
        for m in methods:
            vm = pat_means[m]
            for k in ['ssim','psnr','mssim','mpsnr']:
                row[f'{m}_{k}'] = vm[k][i]
        writer.writerow(row)

print(f'\nResults saved to: {OUT}')
print('Phase 1 complete!')
