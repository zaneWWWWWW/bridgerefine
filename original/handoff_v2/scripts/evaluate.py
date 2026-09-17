# -*- coding: utf-8 -*-
"""Unified patient-level evaluation with 10-step x 2-recursion coarse MRI for
training, validation, and test.

Methods that depend on coarse MRI (SelfRDB, BridgeGAN, coarse-only,
BridgeRefine-L1 seeds 42/123/2026) are regenerated with the unified coarse.
CT-only, SynDiff, and MG-CycleGAN do not use coarse and are reused/regenerated.
"""

import csv
import json
import math
import os
import sys

import cv2
import nibabel
import numpy as np
import torch
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader

sys.path.insert(0, "D:/Cross-modal conversion")
from other_model.BridgeGAN.model import RefinerGenerator
from selfrdb.dataset import PairedCTMRIDataset

DEVICE = torch.device("cuda")
OUT = "D:/Cross-modal conversion/other_model/paper/source_data/unified10x2"
os.makedirs(OUT, exist_ok=True)


class SimpleUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 7, padding=3, padding_mode="reflect"),
            torch.nn.InstanceNorm2d(64), torch.nn.ReLU(True))
        self.enc2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
            torch.nn.InstanceNorm2d(128), torch.nn.ReLU(True))
        self.res = torch.nn.Sequential(*[
            torch.nn.Sequential(
                torch.nn.Conv2d(128, 128, 3, padding=1, padding_mode="reflect"),
                torch.nn.InstanceNorm2d(128), torch.nn.ReLU(True),
                torch.nn.Conv2d(128, 128, 3, padding=1, padding_mode="reflect"),
                torch.nn.InstanceNorm2d(128))
            for _ in range(4)])
        self.dec1 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            torch.nn.Conv2d(128, 64, 3, padding=1),
            torch.nn.InstanceNorm2d(64), torch.nn.ReLU(True))
        self.out = torch.nn.Sequential(
            torch.nn.Conv2d(64, 1, 7, padding=3, padding_mode="reflect"),
            torch.nn.Tanh())

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        r = e2
        for block in self.res:
            r = r + block(r)
        d1 = self.dec1(r)
        return self.out(d1)


class CoarseOnlyRefiner(torch.nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=64):
        super().__init__()
        self.enc1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, base_ch, 7, padding=3, padding_mode="reflect"),
            torch.nn.InstanceNorm2d(base_ch), torch.nn.ReLU(True))
        self.enc2 = torch.nn.Sequential(
            torch.nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            torch.nn.InstanceNorm2d(base_ch * 2), torch.nn.ReLU(True))
        self.res = torch.nn.Sequential(*[
            torch.nn.Sequential(
                torch.nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1, padding_mode="reflect"),
                torch.nn.InstanceNorm2d(base_ch * 2), torch.nn.ReLU(True),
                torch.nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1, padding_mode="reflect"),
                torch.nn.InstanceNorm2d(base_ch * 2))
            for _ in range(4)])
        self.dec1 = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            torch.nn.Conv2d(base_ch * 2, base_ch, 3, padding=1),
            torch.nn.InstanceNorm2d(base_ch), torch.nn.ReLU(True))
        self.out = torch.nn.Sequential(
            torch.nn.Conv2d(base_ch, out_ch, 7, padding=3, padding_mode="reflect"),
            torch.nn.Tanh())

    def forward(self, coarse):
        e1 = self.enc1(coarse)
        e2 = self.enc2(e1)
        r = e2
        for block in self.res:
            r = r + block(r)
        d1 = self.dec1(r)
        return self.out(d1)


def cssim(r, g):
    r01 = (r + 1) / 2
    g01 = (g + 1) / 2
    c1, c2 = 0.0001, 0.0009
    mu_x = cv2.GaussianBlur(r01, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(g01, (11, 11), 1.5)
    sx = cv2.GaussianBlur(r01 ** 2, (11, 11), 1.5) - mu_x ** 2
    sy = cv2.GaussianBlur(g01 ** 2, (11, 11), 1.5) - mu_y ** 2
    sxy = cv2.GaussianBlur(r01 * g01, (11, 11), 1.5) - mu_x * mu_y
    return float(np.mean((2 * mu_x * mu_y + c1) * (2 * sxy + c2)
                         / (mu_x ** 2 + mu_y ** 2 + c1) / (sx + sy + c2)))


def main():
    with open("D:/Cross-modal conversion/checkpoints/selfrdb/patient_split.json") as f:
        test_pids = sorted(json.load(f)["test"])

    old = torch.load("D:/Cross-modal conversion/other_model/paper/improved/predictions.pt",
                     map_location="cpu", weights_only=False)
    all_real = old["real"]
    old_names = old["names"]
    old_preds = old["preds"]
    n = len(all_real)

    unified = torch.load(os.path.join(OUT, "test_coarse_10x2.pt"), map_location="cpu",
                         weights_only=False)
    coarse = unified["coarse"]
    names = unified["names"]
    assert names == old_names, "name order mismatch between coarse and saved predictions"

    ds = PairedCTMRIDataset("D:/Cross-modal conversion/data_slices", image_size=128,
                            augment=False, patient_ids=test_pids)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    # models
    ct_model = SimpleUNet().to(DEVICE)
    ct_model.load_state_dict(torch.load(
        "D:/Cross-modal conversion/other_model/phase2_CT_only_L1/best.pt",
        map_location=DEVICE, weights_only=False)["model"])
    ct_model.eval()

    co_model = CoarseOnlyRefiner(in_ch=1, out_ch=1, base_ch=64).to(DEVICE)
    co_model.load_state_dict(torch.load(
        "D:/Cross-modal conversion/other_model/input_ablation/coarse_only_seed42/best.pt",
        map_location=DEVICE, weights_only=False)["refiner"])
    co_model.eval()

    ref42 = RefinerGenerator(in_ch=2, out_ch=1, base_ch=64).to(DEVICE)
    ref42.load_state_dict(torch.load(
        "D:/Cross-modal conversion/other_model/BridgeGAN_L1/best.pt",
        map_location=DEVICE, weights_only=False)["refiner"])
    ref42.eval()

    ref123 = RefinerGenerator(in_ch=2, out_ch=1, base_ch=64).to(DEVICE)
    ref123.load_state_dict(torch.load(
        "D:/Cross-modal conversion/other_model/final_br_l1_seed123/best.pt",
        map_location=DEVICE, weights_only=False)["refiner"])
    ref123.eval()

    ref2026 = RefinerGenerator(in_ch=2, out_ch=1, base_ch=64).to(DEVICE)
    ref2026.load_state_dict(torch.load(
        "D:/Cross-modal conversion/other_model/final_br_l1_seed2026/best.pt",
        map_location=DEVICE, weights_only=False)["refiner"])
    ref2026.eval()

    ref_gan = RefinerGenerator(in_ch=2, out_ch=1, base_ch=64).to(DEVICE)
    ref_gan.load_state_dict(torch.load(
        "D:/Cross-modal conversion/other_model/BridgeGAN/best.pt",
        map_location=DEVICE, weights_only=False)["refiner"])
    ref_gan.eval()

    preds = {m: [] for m in ["CT-only", "coarse-only", "BridgeRefine-L1",
                             "BridgeRefine-L1-seed123", "BridgeRefine-L1-seed2026",
                             "SelfRDB", "BridgeGAN"]}
    for i, (ct, mr, name) in enumerate(loader):
        ct_d = ct.to(DEVICE)
        coarse_d = torch.from_numpy(coarse[i][0]).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            p_ct = ct_model(ct_d)
            p_co = co_model(coarse_d)
            p42 = ref42(coarse_d, ct_d)
            p123 = ref123(coarse_d, ct_d)
            p2026 = ref2026(coarse_d, ct_d)
            p_gan = ref_gan(coarse_d, ct_d)
        preds["CT-only"].append(p_ct.squeeze().cpu().numpy())
        preds["coarse-only"].append(p_co.squeeze().cpu().numpy())
        preds["BridgeRefine-L1"].append(p42.squeeze().cpu().numpy())
        preds["BridgeRefine-L1-seed123"].append(p123.squeeze().cpu().numpy())
        preds["BridgeRefine-L1-seed2026"].append(p2026.squeeze().cpu().numpy())
        preds["SelfRDB"].append(coarse[i][0])
        preds["BridgeGAN"].append(p_gan.squeeze().cpu().numpy())
        if (i + 1) % 500 == 0:
            print(f"  infer {i + 1}/{n}", flush=True)

    # add coarse-independent methods
    preds["SynDiff"] = old_preds["SynDiff"]
    preds["MG-CycleGAN"] = old_preds["MG-CycleGAN"]

    torch.save({"names": names, "real": all_real, "coarse_10x2": coarse,
                "preds": preds}, os.path.join(OUT, "predictions_10x2.pt"))

    # masks
    masks = {}
    for pid in test_pids:
        p = f"D:/Cross-modal conversion/brain/{pid}/mask.nii.gz"
        if os.path.exists(p):
            masks[pid] = nibabel.load(p).get_fdata()

    import lpips
    lp_model = lpips.LPIPS(net="alex").to(DEVICE).eval()

    methods = ["CT-only", "coarse-only", "BridgeRefine-L1", "SelfRDB",
               "BridgeGAN", "SynDiff", "MG-CycleGAN"]
    pat = {pid: {m: {k: [] for k in
                     ["ssim", "psnr", "mssim", "mpsnr", "lpips", "imad"]}
                 for m in methods} for pid in test_pids}
    real_mad_by_pid = {pid: [] for pid in test_pids}

    for i, name in enumerate(names):
        pid, sid = name.split("/")
        sn = int(sid.split(".")[0])
        r = all_real[i]
        rmask = None
        if pid in masks and sn < masks[pid].shape[2]:
            ms = masks[pid][:, :, sn]
            ms = cv2.resize(ms.astype(np.float32), (128, 128),
                            interpolation=cv2.INTER_NEAREST)
            rmask = (ms > 0).astype(np.float32)
        r_t = torch.from_numpy(r).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(DEVICE)
        for m in methods:
            g = preds[m][i]
            s = cssim(r, g)
            mse = np.mean(((r + 1) / 2 - (g + 1) / 2) ** 2)
            p = 10 * math.log10(1 / max(mse, 1e-12))
            s_mask, p_mask = s, p
            if rmask is not None and rmask.sum() > 100:
                rm = ((r + 1) / 2) * rmask
                gm = ((g + 1) / 2) * rmask
                mse_m = np.mean(((rm - gm)[rmask > 0]) ** 2)
                p_mask = 10 * math.log10(1 / max(mse_m, 1e-12))
                mu_x = cv2.GaussianBlur(rm, (11, 11), 1.5)
                mu_y = cv2.GaussianBlur(gm, (11, 11), 1.5)
                sx = cv2.GaussianBlur(rm ** 2, (11, 11), 1.5) - mu_x ** 2
                sy = cv2.GaussianBlur(gm ** 2, (11, 11), 1.5) - mu_y ** 2
                sxy = cv2.GaussianBlur(rm * gm, (11, 11), 1.5) - mu_x * mu_y
                c1m, c2m = 0.0001, 0.0009
                s_mask = float(np.mean((2 * mu_x * mu_y + c1m) * (2 * sxy + c2m)
                                       / (mu_x ** 2 + mu_y ** 2 + c1m)
                                       / (sx + sy + c2m)))
            g_t = torch.from_numpy(g).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                lp = float(lp_model(r_t, g_t, normalize=False).item())
            pat[pid][m]["ssim"].append(s)
            pat[pid][m]["psnr"].append(p)
            pat[pid][m]["mssim"].append(s_mask)
            pat[pid][m]["mpsnr"].append(p_mask)
            pat[pid][m]["lpips"].append(lp)
        if i % 500 == 0:
            print(f"  metrics {i}/{n}", flush=True)

    # inter-slice MAD
    for pid in test_pids:
        idx = [i for i, name in enumerate(names) if name.startswith(pid + "/")]
        idx.sort(key=lambda i: int(names[i].split("/")[1].split(".")[0]))
        for k in range(len(idx) - 1):
            real_mad_by_pid[pid].append(np.mean(np.abs(all_real[idx[k + 1]] - all_real[idx[k]])))
            for m in methods:
                pat[pid][m]["imad"].append(
                    np.mean(np.abs(preds[m][idx[k + 1]] - preds[m][idx[k]])))

    summary = {m: {k: [] for k in
                   ["ssim", "psnr", "mssim", "mpsnr", "lpips", "imad_ratio"]}
               for m in methods}
    rows = []
    for pid in test_pids:
        row = {"patient": pid}
        for m in methods:
            for k in ["ssim", "psnr", "mssim", "mpsnr", "lpips"]:
                v = float(np.mean(pat[pid][m][k]))
                summary[m][k].append(v)
                row[f"{m}_{k}"] = f"{v:.4f}"
            rm = float(np.mean(real_mad_by_pid[pid])) if real_mad_by_pid[pid] else float("nan")
            pm = float(np.mean(pat[pid][m]["imad"])) if pat[pid][m]["imad"] else float("nan")
            ratio = pm / rm if rm else float("nan")
            summary[m]["imad_ratio"].append(ratio)
            row[f"{m}_imad_ratio"] = f"{ratio:.4f}"
        rows.append(row)

    with open(os.path.join(OUT, "metrics_patient_unified_10x2.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\nPatient-level cohort means (mean +/- SD across 18 patients):")
    for m in methods:
        s = np.mean(summary[m]["ssim"]); ss = np.std(summary[m]["ssim"], ddof=1)
        p = np.mean(summary[m]["psnr"]); lp = np.mean(summary[m]["lpips"])
        ms = np.mean(summary[m]["mssim"]); ir = np.mean(summary[m]["imad_ratio"])
        print(f"  {m:<20s} SSIM={s:.4f}+-{ss:.4f} PSNR={p:.2f} LPIPS={lp:.4f} "
              f"mSSIM={ms:.4f} MAD_ratio={ir:.4f}")

    # L1 seed cohort means for multi-seed reporting
    l1_seeds = {}
    for m, seed in [("BridgeRefine-L1", 42), ("BridgeRefine-L1-seed123", 123),
                    ("BridgeRefine-L1-seed2026", 2026)]:
        vals = []
        for pid in test_pids:
            g = preds[m][[i for i, name in enumerate(names) if name.startswith(pid + "/")][0]]
            vals.append(np.mean([cssim(all_real[j], preds[m][j])
                                 for j in [i for i, name in enumerate(names)
                                           if name.startswith(pid + "/")]]))
        l1_seeds[seed] = float(np.mean(vals))
    print("\nL1 seed cohort means:", l1_seeds)

    # paired stats CT-only vs BridgeRefine-L1 seed42
    diff = np.array(summary["BridgeRefine-L1"]["ssim"]) - np.array(summary["CT-only"]["ssim"])
    w_stat, raw_p = scipy_stats.wilcoxon(diff)
    rng = np.random.default_rng(42)
    boots = [np.mean(rng.choice(diff, size=len(diff), replace=True)) for _ in range(10000)]
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    paired = {
        "mean_paired_diff": float(np.mean(diff)),
        "median_paired_diff": float(np.median(diff)),
        "improved_patients": f"{int(np.sum(diff > 0))}/18",
        "wilcoxon_raw_p": float(raw_p),
        "bootstrap_ci95": ci,
        "ci_method": "percentile bootstrap, 10,000 resamples, seed 42",
    }
    print("\nPaired CT-only vs BridgeRefine-L1 (seed 42):", paired)

    with open(os.path.join(OUT, "summary_unified_10x2.json"), "w",
              encoding="utf-8") as f:
        json.dump({"patient_summary": summary, "l1_seed_cohort_means": l1_seeds,
                   "paired": paired}, f, ensure_ascii=False, indent=2)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
