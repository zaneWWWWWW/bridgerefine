# BridgeRefine 实验总览

## 交接状态

交接包：`/home/zanewang/projects/paper/BridgeRefine_交接包.zip`

解压目录：`/home/zanewang/projects/paper/BridgeRefine_交接包/`

最终冻结材料：`handoff_send/final_frozen_light/`

验收日期：2026-08-27

### 验收结论

- 六个实验包的 checkpoint、指标、预测索引、日志、配置和哈希文件均已解压。
- 六个 supplement 包中的文件哈希全部通过。
- final frozen 包中的论文、图、表和 source data 哈希通过；其中 3 个中文文档因 ZIP 文件名编码失真导致 `sha256sum -c` 无法按原文件名读取，但文档内容实际存在且可读，不影响实验数值文件。
- 当前最终结果应以 `final_frozen_light` 的统一 `10 sampling steps x 2 recursions` 版本为准。
- 早先的 `0.579 / 0.8169 / 0.075 / 1264 ms / 632x` 旧版本结果不再作为最终主结果。旧值中包括旧的 bridge 输出、旧 LPIPS 和旧效率测量。

## 实验设置

| 项目 | 最终设置 |
|---|---|
| 任务 | 配对脑 CT 到 MRI 图像转换 |
| 数据 | 180 名患者，患者级划分 |
| 训练/验证/测试 | 144 / 18 / 18 名患者 |
| 测试数据 | 3,415 张轴向切片 |
| 图像尺寸 | 128 x 128 |
| 原实验 GPU | NVIDIA RTX 5060 Laptop GPU，8 GB |
| bridge | 冻结 SelfRDB checkpoint |
| bridge 采样 | 10 steps，2 recursions |
| refiner loss | L1 |
| batch size | 16 |
| 训练轮数 | 20 |
| 优化器 | Adam，lr=2e-4，betas=(0.5, 0.999) |
| 患者级统计 | 患者为独立统计单位 |

## 最终主结果

以下结果来自 `handoff_send/final_frozen_light/source_data/figure2_method_summary.csv` 和 `metrics_patient_final.csv`。均值和标准差为 18 名患者的患者级统计。

| 方法 | SSIM | PSNR (dB) | LPIPS | Mask SSIM | Inter-slice MAD ratio |
|---|---:|---:|---:|---:|---:|
| SelfRDB | 0.5855 +/- 0.0395 | 16.44 | 0.229 | 0.7545 | 3.608 |
| SynDiff | 0.7473 +/- 0.0687 | 18.36 | 0.121 | 0.8207 | 1.575 |
| MG-CycleGAN | 0.7882 +/- 0.0533 | 18.86 | 0.082 | 0.8403 | 1.202 |
| BridgeGAN | 0.8046 +/- 0.0538 | 18.92 | 0.078 | 0.8462 | 1.409 |
| CT-only U-Net | 0.8095 +/- 0.0536 | 19.77 | 0.129 | 0.8528 | 0.853 |
| **BridgeRefine-L1** | **0.8184 +/- 0.0530** | **19.95** | **0.125** | **0.8562** | **0.967** |

BridgeRefine-L1 相对 CT-only 的 seed-42 结果为：

- SSIM：`+0.0089`；
- PSNR：`+0.18 dB`；
- LPIPS：降低约 `0.004`；
- Mask SSIM：`+0.0034`；
- Inter-slice MAD ratio 更接近 1。

BridgeRefine 的 SSIM 最高，PSNR 和 Mask SSIM 也最高；但 LPIPS 不是所有方法中最低，不能再写“lowest LPIPS”。

## 输入来源消融

统一 `10 x 2` coarse 协议、相同网络结构和 seed 42 下：

| 输入配置 | SSIM | PSNR (dB) | LPIPS | 解释 |
|---|---:|---:|---:|---|
| CT-only | 0.8095 +/- 0.0536 | 19.77 | 0.129 | 直接监督基线 |
| Coarse-only | 0.7823 +/- 0.0538 | 18.88 | 0.174 | coarse 单独信息不足 |
| CT + Coarse | 0.8184 +/- 0.0530 | 19.95 | 0.125 | 完整 BridgeRefine |

排序为：

```text
CT + Coarse > CT-only > Coarse-only
```

这支持两个同时成立、但强度不同的结论：

1. CT 是主要信息来源；
2. coarse MRI 在 CT 之外提供了小幅额外结构先验。

因此论文主线应写成：监督式后桥精炼是主要性能来源，扩散桥在当前配置下带来有限但可测的额外收益。

## 多随机种子

### CT-only U-Net

| Seed | SSIM |
|---|---:|
| 42 | 0.8095 |
| 123 | 0.8050 |
| 2026 | 0.7883 |
| 均值 | 0.8009 |

### BridgeRefine-L1

| Seed | SSIM |
|---|---:|
| 42 | 0.8184 |
| 123 | 0.8183 |
| 2026 | 0.8147 |
| 均值 | 0.8171 |

BridgeRefine-L1 的三次运行标准差约为 `0.0021`，CT-only 的观察波动约为 `0.009` 至 `0.011`，具体取决于采用总体标准差还是样本标准差。论文中必须统一标准差定义；建议明确写为“standard deviation across training runs”，并使用同一种计算约定。

多 seed 结果支持 BridgeRefine-L1 的运行间排序和较低观察波动，但由于每个模型只有 3 次运行，不能把它写成稳健的方差定理或“稳定性提高若干倍”。

### Gradient loss 消融

最终主方法为 L1。L1+Gradient 仅作为消融：

- L1 三 seed 均值约 `0.8171`；
- L1+Gradient 两 seed 均值约 `0.8180`；
- 两个 seed 的方向不一致。

结论：当前实验不支持 Gradient loss 是核心贡献，也不支持“Gradient loss 稳定提升性能”。200-slice 结果应保留为探索性分析。

## 患者级配对统计

最终冻结汇总 `summary_final.json` 记录 BridgeRefine-L1 seed 42 相对 CT-only seed 42：

- 平均配对 SSIM 差：`+0.0089`；
- 中位配对差：`+0.0088`；
- 改善患者：`17/18`；
- Wilcoxon 原始 `p=0.0001`；
- Holm 校正后 `p=0.0002`；
- 95% CI：`[0.0058, 0.0122]`；
- CI：10,000 次百分位 bootstrap，随机种子 42。

该统计支持 BridgeRefine-L1 相对 CT-only 的小幅、患者级一致性改善。它不支持临床诊断等价性。

## 效率和显存

效率实验包：`supplement/efficiency_rtx5060_batch1/`

协议：batch size 1，warm-up 10，正式计时 100 张，重复 3 次，CUDA 同步，不包含模型加载时间。

| 指标 | CT-only | BridgeRefine |
|---|---:|---:|
| 推理时间 | 2.422 +/- 0.118 ms/slice | 1016.992 +/- 13.968 ms/slice |
| 额外峰值推理显存 | 44.6 MB | 144.7 MB |

时间比约为 `420x`。该结果支持明确的 accuracy-efficiency trade-off：BridgeRefine 的重建指标略高，但 CT-only 更适合低延迟场景。

显存定义是模型加载后新增的 `torch.cuda.max_memory_allocated()`，不是完整 GPU 占用。

## 当前实验支持的最终科学结论

可以在论文中保留以下主张：

1. 监督式后桥精炼是当前实验中最大的性能增益来源。
2. CT 是主要信息源，coarse MRI 作为额外结构先验带来小幅增益。
3. BridgeRefine-L1 在 3 个 seed 下维持约 0.815--0.818 的 SSIM。
4. Gradient loss 在完整队列上没有显示一致的独立收益。
5. 所测试的对抗监督配置没有优于 L1 精炼。
6. BridgeRefine 带来约 420 倍的推理时间代价。
7. 合成 MRI 与真实 MRI 的诊断等价性未被本研究建立。

推荐总括表述：

> A frozen diffusion bridge followed by supervised post-bridge refinement achieved a modest patient-level reconstruction improvement over direct CT-only refinement. CT remained the dominant information source, while the coarse bridge output provided a smaller additional benefit. The gain was accompanied by a substantial inference-time penalty, and no diagnostic equivalence to acquired MRI is claimed.

## 交接包中的已知问题

### 1. supplement 单包 summary 的 `n_slices` 填写错误

五个输入实验的 `summary.json` 中 `n_slices` 写成了 `54`，但对应 `metrics_slice.csv` 和预测索引均为 `3,415` 行。最终冻结包的论文和统一切片数据使用的是正确的 3,415；写作时不要读取这些单包 summary 的 `n_slices` 字段作为测试样本数。

### 2. final frozen 哈希的中文文档文件名编码问题

final frozen 中 3 个中文文档实际存在，但 ZIP 解压后文件名为乱码形式，导致哈希清单中的原始中文路径无法直接匹配。数值文件、论文源文件、PDF、图表和表格哈希均通过。后续可在写作机器上将 3 个文档重命名为 UTF-8 文件名并重新生成 final frozen 的哈希清单。

### 3. 训练日志不完整

输入消融的 `run.log` 只保留部分 epoch 的 train/validation loss；两个已有 checkpoint 没有重新捕获完整 per-epoch log。这不影响当前测试指标，但应在论文中避免声称完整训练轨迹已归档。

### 4. 代码版本不是 Git revision

交接包记录当前代码目录不是 Git 仓库。投稿前应对完整代码源文件建立压缩包并记录 SHA-256，或在正式代码仓库中固定 commit。

### 5. 医学论文信息仍待补齐

投稿前还需补充：伦理审批或豁免、知情同意、去标识化与数据治理、扫描设备和采集参数、MRI 序列、配准方法、基金和作者贡献。

## 写作时应使用的最终文件

优先使用：

```text
BridgeRefine_交接包/handoff_send/final_frozen_light/paper_twocolumn_zh_revised.tex
BridgeRefine_交接包/handoff_send/final_frozen_light/paper_twocolumn_zh_revised.pdf
BridgeRefine_交接包/handoff_send/final_frozen_light/source_data/
BridgeRefine_交接包/handoff_send/final_frozen_light/figures/
BridgeRefine_交接包/handoff_send/final_frozen_light/tables/
```

不要继续使用工作区中旧的 `other_model/paper/nature_bridge_refine/paper*.tex` 作为最终稿，除非先将 final frozen 的 source data 和数字同步进去。

## 最终状态判断

### 科学实验

已达到可以正式写作和准备 Q4 方法学投稿的程度。不需要再增加新的 GAN、扩散模型、loss 或 3D 网络。

### 投稿前必须完成

- [ ] 把 final frozen 的新数字同步到写作目录的 LaTeX 和 source data；
- [ ] 修正或在写作脚本中忽略 supplement summary 的 `n_slices=54` 元数据错误；
- [ ] 统一全文标准差定义；
- [ ] 处理中文文档文件名并重新生成冻结包哈希；
- [ ] 记录完整代码源包 SHA-256；
- [ ] 补伦理、采集参数、配准和数据治理信息；
- [ ] 用当前 final frozen 源数据重新编译并检查 PDF。

最终判断：**实验内容已经基本收口，当前工作重点应从补模型转为结果同步、可复现性修正、医学信息补齐和论文写作。**
