# Is a Frozen Diffusion Bridge Useful? BridgeRefine for Paired Brain CT-to-MRI Translation

**Authors and affiliations:** [TO BE COMPLETED]

## Abstract

We investigate whether a diffusion bridge contributes information beyond a directly supervised CT-to-MRI mapping, and whether this contribution justifies its computational cost. The study uses 180 paired brain CT–MRI volumes split by patient into 144 training, 18 validation, and 18 test patients, with 3,415 retained axial test slices. A frozen SelfRDB bridge (300 training timesteps) produces coarse MRI using 10 sampling steps and two recursive estimates. A 1.34M-parameter refiner receives CT and coarse MRI and is trained with L1 loss. BridgeRefine-L1 achieves SSIM 0.8184±0.0530, PSNR 19.95 dB, LPIPS 0.125, mask SSIM 0.8562, and inter-slice MAD ratio 0.967. CT-only achieves SSIM 0.8095±0.0536. BridgeRefine improves 17/18 patients, with mean paired difference 0.0089, 95% bootstrap interval [0.0058, 0.0122], and Holm-adjusted p=0.0002. Input ablation gives 0.8095±0.0536 (CT-only), 0.7823±0.0538 (coarse-only), and 0.8184±0.0530 (combined). Three-run means are 0.8009±0.0112 for CT-only and 0.8171±0.0020 for BridgeRefine. Inference takes 233.6±12.6 ms per slice versus 2.06±0.18 ms. Coarse MRI therefore contributes a small measurable signal, while CT and supervised refinement account for most performance. The gain costs approximately 113-fold higher latency and does not establish diagnostic equivalence to acquired MRI.

**Keywords:** brain CT; MRI synthesis; diffusion bridge; supervised refinement; medical image translation

## 1. Introduction

CT is rapid and widely available, whereas MRI offers stronger soft-tissue contrast. Translating CT into MRI-like images could provide supplementary information when MRI is impractical. The relevant question is whether an intermediate diffusion representation adds useful information beyond direct supervision, not whether the output merely looks plausible.

We test whether frozen-bridge output contains information unavailable to a CT-only model and whether any improvement justifies its cost. We use controlled input ablation, patient-level comparisons, multi-seed runs, and an independent efficiency benchmark.

## 2. Methods

The materials contain 180 paired, co-registered brain CT–MRI volumes and aligned binary masks. Patients are split 144/18/18, with 3,415 retained axial slices in the test set. Study center, eligibility criteria, CT–MRI interval, scanner and sequence details, acquisition parameters, registration procedure, and quality control remain unavailable: [TO BE COMPLETED].

MRI undergoes N4 correction; CT windowing uses histogram-peak rules and MRI uses the 2nd and 98th percentiles. CLAHE settings are 1.5/4×4 for CT and 2.0/8×8 for MRI, with gamma values 0.9 and 1.1. Low-information slices (<0.5% non-zero pixels) are removed, images resized to 128×128, and intensities mapped to [-1,1].

The SelfRDB bridge is $x_t=(1-t/T)x_0+(t/T)y+\sqrt{\gamma t/T}\epsilon$, with $T=300$ and $\gamma=0.1$. Its checkpoint is frozen. Bridge-dependent models use 10 sampling steps and two recursive estimates. The refiner predicts $\hat{x}=R_\theta([y,\hat{x}_{bridge}])$ using an encoder, four residual blocks, a decoder, Instance Normalization, 64 base channels, and approximately 1.34M trainable parameters. The objective is $\mathcal{L}_{L1}=\lVert\hat{x}-x_0\rVert_1$.

CT-only U-Net receives CT alone; coarse-only receives bridge output alone. Other fixed-checkpoint comparators are SelfRDB, SynDiff, MG-CycleGAN, and BridgeGAN. Training uses Adam, learning rate 2×10⁻⁴, betas (0.5,0.999), batch size 16, and up to 20 epochs. Metrics are SSIM, PSNR, LPIPS, mask SSIM, mask PSNR, and inter-slice MAD ratio. Slice scores are averaged within patients. Patient-level SSIM uses two-sided Wilcoxon testing with Holm correction across six comparisons; confidence intervals use 10,000 percentile-bootstrap resamples (seed 42). Exploratory 200-slice loss tests and a three-patient adversarial pilot are not confirmatory.

## 3. Results

| Method | SSIM | PSNR (dB) | LPIPS | Mask SSIM | Mask PSNR | MAD ratio |
|---|---:|---:|---:|---:|---:|---:|
| SelfRDB | 0.5855±0.0395 | 16.44 | 0.229 | 0.7545 | 13.64 | 3.608 |
| SynDiff | 0.7473±0.0687 | 18.36 | 0.122 | 0.8207 | 16.12 | 1.578 |
| MG-CycleGAN | 0.7882±0.0533 | 18.86 | 0.082 | 0.8403 | 16.31 | 1.197 |
| BridgeGAN | 0.8046±0.0538 | 18.92 | 0.078 | 0.8462 | 16.50 | 1.409 |
| CT-only U-Net | 0.8095±0.0536 | 19.77 | 0.129 | 0.8528 | 17.34 | 0.853 |
| **BridgeRefine-L1** | **0.8184±0.0530** | **19.95** | **0.125** | **0.8562** | **17.49** | **0.967** |

BridgeRefine ranks highest on SSIM, PSNR, and mask metrics, while LPIPS favors MG-CycleGAN and BridgeGAN. Input ablation shows CT-only 0.8095±0.0536, coarse-only 0.7823±0.0538, and combined input 0.8184±0.0530. BridgeRefine improves 17/18 patients; mean and median paired differences are 0.0089 and 0.0088, with interval [0.0058,0.0122], raw p=0.0001, and adjusted p=0.0002.

Across seeds 42, 123, and 2026, CT-only SSIM is 0.8095, 0.8050, and 0.7883 (mean 0.8009±0.0112); BridgeRefine is 0.8184, 0.8183, and 0.8147 (mean 0.8171±0.0020). The gradient ablation has no consistent full-cohort direction. The tested adversarial setting is below L1 refinement.

| Efficiency | CT-only | BridgeRefine |
|---|---:|---:|
| Trainable parameters | 1.33M | 1.34M |
| Total parameters | 1.33M | 15.74M |
| Additional inference memory | 39.3 MB | 134.0 MB |
| Time per slice | 2.06±0.18 ms | 233.6±12.6 ms |

## 4. Discussion and Limitations

The combined input exceeds CT-only, whereas coarse-only is weaker. CT is therefore dominant and the bridge is auxiliary. The 0.0089 SSIM gain comes with approximately 113-fold higher latency. LPIPS, SSIM, and the MAD ratio quantify different image properties and do not demonstrate anatomical or clinical validity. The independent test cohort has 18 patients, the model processes 128×128 two-dimensional slices, ablation run counts are unequal, exploratory subsets may be selected, and the adversarial pilot contains three patients. No blinded reading, lesion endpoint, downstream task, external validation, or diagnostic agreement study is available. Ethics, consent, acquisition and registration metadata, data governance, code revision, full logs, funding, and author contributions remain [TO BE COMPLETED].

## 5. Conclusion

Frozen SelfRDB output followed by L1 supervised refinement yields a small patient-level improvement over direct CT-only translation. CT remains the dominant source, while coarse MRI contributes limited auxiliary information. The gain has substantial latency cost and represents an accuracy–efficiency trade-off rather than a validated clinical substitute for acquired MRI.

## Statements and References

**Ethics, consent, data availability, code availability, funding, conflicts, author contributions:** [TO BE COMPLETED]

[CITATION NEEDED] Verify authoritative records for SelfRDB, SynDiff, MG-CycleGAN, DDPM, DDIM, and pix2pix before submission.
