"""Evaluate all 4 models: SSIM, PSNR, LPIPS, t-tests. Single run, no background."""
import sys, torch, math, json, numpy as np, cv2, lpips
from scipy import stats as scipy_stats
import torch.nn.functional as F

device = torch.device('cuda')
print('Loading data...', flush=True)
with open('D:/Cross-modal conversion/checkpoints/selfrdb/patient_split.json') as f:
    split = json.load(f)

sys.path.insert(0, 'D:/Cross-modal conversion')
from selfrdb.dataset import PairedCTMRIDataset
from torch.utils.data import DataLoader
ds = PairedCTMRIDataset('D:/Cross-modal conversion/data_slices',image_size=128,augment=False,patient_ids=split['test'])
loader = DataLoader(ds,batch_size=1,shuffle=False)
N = len(ds); print(f'{len(split["test"])} patients, {N} slices', flush=True)

lpips_fn = lpips.LPIPS(net='alex').to(device)

def cssim(r,g):
    r01=(r+1)/2;g01=(g+1)/2;c1,c2=0.0001,0.0009
    mu_x=cv2.GaussianBlur(r01,(11,11),1.5);mu_y=cv2.GaussianBlur(g01,(11,11),1.5)
    sx=cv2.GaussianBlur(r01**2,(11,11),1.5)-mu_x**2;sy=cv2.GaussianBlur(g01**2,(11,11),1.5)-mu_y**2
    sxy=cv2.GaussianBlur(r01*g01,(11,11),1.5)-mu_x*mu_y
    return float(np.mean((2*mu_x*mu_y+c1)*(2*sxy+c2)/(mu_x**2+mu_y**2+c1)/(sx+sy+c2)))

results={m:{'ssim':[],'psnr':[],'lpips':[]} for m in ['MG-CycleGAN','SynDiff','SelfRDB','BridgeGAN']}

# All models
from other_model.MG_CycleGAN.model import Generator as GAN_Gen
g_mg=GAN_Gen(in_ch=2,out_ch=1,base_ch=64,n_res=6).to(device)
g_mg.load_state_dict(torch.load('D:/Cross-modal conversion/other_model/MG_CycleGAN/best.pt',map_location=device,weights_only=False)['generator']);g_mg.eval()

from other_model.SynDiff.model import DiffusionUNet,NonDiffusiveGenerator,SynDiff,PatchDiscriminator
nd=NonDiffusiveGenerator(in_ch=1,out_ch=1,base_ch=64).to(device);du=DiffusionUNet(in_ch=2,out_ch=1,base_ch=64).to(device)
cs=torch.load('D:/Cross-modal conversion/other_model/SynDiff/best.pt',map_location=device,weights_only=False)
du.load_state_dict(cs['diff_unet']);nd.load_state_dict(cs['nd_gen']);du.eval();nd.eval()
syndiff=SynDiff(du,PatchDiscriminator(in_ch=2).to(device),nd,n_steps=4,device=device)

from other_model.SelfRDB.bridge_model import DiffusionBridge
from other_model.SelfRDB.network import X0UNet
gen_s=X0UNet(in_ch=3,out_ch=1,base_ch=64,ch_mult=(1,2,4)).to(device)
gen_s.load_state_dict(torch.load('D:/Cross-modal conversion/other_model/SelfRDB/selfrdb_best.pt',map_location=device,weights_only=False)['generator']);gen_s.eval()
bridge_s=DiffusionBridge(gen_s,timesteps=300,gamma=0.1,device=device)

from other_model.BridgeGAN.model import RefinerGenerator
ref=RefinerGenerator(in_ch=2,out_ch=1,base_ch=64).to(device)
ref.load_state_dict(torch.load('D:/Cross-modal conversion/other_model/BridgeGAN/best.pt',map_location=device,weights_only=False)['refiner']);ref.eval()

print('Evaluating...', flush=True)
for i,(ct,mr,_) in enumerate(loader):
    ct_d=ct.to(device);mr_d=mr.to(device)
    mask=F.max_pool2d((ct_d>-0.5).float(),3,1,1)
    real=mr.squeeze().cpu().numpy()

    with torch.no_grad():
        mg=g_mg(torch.cat([ct_d,mask],dim=1))
        sd=syndiff.sample(ct_d)
        sr=bridge_s.sample_ddib(ct_d,num_steps=20,num_recursions=5)
        bg=ref(sr,ct_d)

    for label,gm in [('MG-CycleGAN',mg),('SynDiff',sd),('SelfRDB',sr),('BridgeGAN',bg)]:
        gn=gm.squeeze().cpu().numpy()
        s=cssim(real,gn);m=np.mean(((real+1)/2-(gn+1)/2)**2);p=10*math.log10(1/max(m,1e-12))
        lp=lpips_fn(gm,mr_d,normalize=False).item()
        results[label]['ssim'].append(s);results[label]['psnr'].append(p);results[label]['lpips'].append(lp)
    if (i+1)%500==0: print(f'  {i+1}/{N}', flush=True)

# REPORT
print()
print('='*70)
print('  {:<15s} {:>8s} {:>8s} {:>8s}'.format('Method', 'SSIM', 'PSNR', 'LPIPS'))
for m in ['MG-CycleGAN','SynDiff','SelfRDB','BridgeGAN']:
    r=results[m]
    print('  {:<15s} {:7.4f}  {:6.1f}  {:7.4f}'.format(m, np.mean(r['ssim']), np.mean(r['psnr']), np.mean(r['lpips'])))
print();print('T-tests (SSIM):')
for other in ['MG-CycleGAN','SynDiff','SelfRDB']:
    t,p=scipy_stats.ttest_rel(results['BridgeGAN']['ssim'],results[other]['ssim'])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
    d=np.mean(results['BridgeGAN']['ssim'])-np.mean(results[other]['ssim'])
    print('  vs {:<15s}: diff={:+.4f} p={:.6f} {}'.format(other, d, p, sig))

# Save final results
torch.save(results,'D:/Cross-modal conversion/other_model/full_eval_results.pt')
print();print('Saved: full_eval_results.pt')
