# 论文缺失材料审计与跨机器交接说明：执行结果

更新日期：2026-09-10

本文档是对《论文缺失材料审计与跨机器交接说明》的执行结果。审计要求中可以在当前机器完成的实验修改、图像导出、provenance 和交接材料已经补齐；必须由数据负责人确认的伦理、采集和配准信息仍保留为缺失项，没有用推测内容替代。

---

## 一、已完成的实验修改

### 1. 统一 coarse MRI 采样协议

训练、验证和测试的 coarse MRI 已统一为：

- 10 sampling steps；
- 2 recursive estimates；
- 冻结 SelfRDB bridge checkpoint。

测试 coarse 已重新生成，所有依赖 coarse 的模型输出和患者级指标已重算。新结果位于：

```text
other_model/paper/source_data/unified10x2/
```

### 2. 10×2 独立效率测量

旧的 20×5 效率记录已标记为过时，不再作为最终结论。新的效率测量满足：

- CT-only 与 BridgeRefine 分进程独立测量；
- 两个模型不混驻显存；
- warm-up 20 张切片；
- 正式计时 100 张切片；
- 重复 3 次；
- 计时前后调用 CUDA synchronize；
- 模型加载、输入传输和预处理不计入时间；
- 显存为模型与输入分配后重置统计所得的额外推理峰值分配。

最终结果：

| 方法 | 单切片时间 | 额外推理显存 allocated | 额外推理显存 reserved |
|---|---:|---:|---:|
| CT-only | 2.06 ± 0.18 ms | 39.3 MB | 54.5 MB |
| BridgeRefine | 233.6 ± 12.6 ms | 134.0 MB | 167.8 MB |

推理时间比约为 113 倍。

对应文件：

```text
experiments/handoff_v2/efficiency/efficiency_final_10x2.csv
experiments/handoff_v2/efficiency/benchmark_config.json
experiments/handoff_v2/efficiency/benchmark.log
experiments/handoff_v2/efficiency/benchmark.py
experiments/handoff_v2/efficiency/summary_ct_only.json
experiments/handoff_v2/efficiency/summary_bridgerefine.json
```

### 3. 真实定性病例

已按 BridgeRefine-L1 seed-42 的患者级 SSIM 选择三个测试患者：

- 高 SSIM：1BA222；
- 中位 SSIM：1BC007；
- 低 SSIM：1BC014。

每个患者均导出：

```text
arrays.npz
slices.csv
metadata.json
slice_XXXX/
  ct.png
  mri_real.png
  coarse_mri.png
  bridgerefine_l1.png
  ct_only.png
  error_bridgerefine_l1.png
```

`arrays.npz` 的键为：

```text
ct, mri_real, coarse_mri, bridgerefine_l1, ct_only, brain_mask
```

数组形状为 `[S, 128, 128]`，保留原始 float32/bool/uint8 数值。预览图统一将 MRI 模型从 `[-1,1]` 映射到 `[0,255]`，没有对 BridgeRefine 单独锐化或增强；误差图由原始预测和参考数组计算。

对应目录：

```text
experiments/handoff_v2/qualitative_cases/
```

### 4. 指标和统计导出

已导出：

- 长表患者级指标：`metrics/patient_metrics_final.csv`；
- CT-only 与 BridgeRefine-L1 逐患者配对差：`metrics/paired_differences.csv`；
- 三个 seed 的指标汇总：`metrics/seed_metrics.csv`；
- Wilcoxon、Holm 家族和 bootstrap 结果：`metrics/significance.json`。

统计设置：

- 双侧 Wilcoxon signed-rank；
- 检验单位为患者；
- Holm 校正家族包含 BridgeRefine-L1 对 SelfRDB、SynDiff、MG-CycleGAN、BridgeGAN、coarse-only 和 CT-only 的比较；
- 95% CI 使用 10,000 次百分位 bootstrap，随机种子 42；
- CT-only vs BridgeRefine-L1：平均配对差 +0.0089，17/18 患者改善，bootstrap CI [0.0058, 0.0122]，Holm p=0.0002。

### 5. Provenance

已生成 checkpoint 清单和哈希：

```text
experiments/handoff_v2/checkpoints/checkpoints_manifest.csv
experiments/handoff_v2/checkpoints/hashes_sha256.txt
```

清单包含 SelfRDB、CT-only 三个 seed、BridgeRefine-L1 三个 seed、BridgeGAN、SynDiff 和 MG-CycleGAN 的 checkpoint 路径、SHA-256、seed、checkpoint 选择方式、采样步数和递归次数。

### 6. 配置和脚本

已放入：

```text
experiments/handoff_v2/configs/bridge.yaml
experiments/handoff_v2/configs/refiner_l1.yaml
experiments/handoff_v2/configs/ct_only.yaml
experiments/handoff_v2/configs/evaluation.yaml
experiments/handoff_v2/scripts/evaluate.py
experiments/handoff_v2/scripts/statistics.py
experiments/handoff_v2/scripts/benchmark.py
```

### 7. 论文同步

中文稿已同步更新：

- 统一 10×2 的患者级指标；
- 输入消融结果；
- BridgeRefine-L1 三个 seed；
- 新的 10×2 效率和显存；
- 不宣称诊断等价、不主张扩散桥部署不可替代。

最新论文位于：

```text
experiments/handoff_v2/paper/论文修改稿.pdf
experiments/handoff_v2/paper/论文修改稿.tex
```

---

## 二、仍然缺失、必须由真实记录补齐的内容

以下内容没有在本次工作中伪造或猜测：

1. 伦理审批编号或豁免说明；
2. 知情同意或回顾性豁免说明；
3. 去标识化和数据治理说明；
4. 研究中心或匿名化中心描述；
5. 纳入和排除标准；
6. CT 与 MRI 检查时间间隔；
7. MRI 序列、场强、扫描仪型号、TR/TE、层厚；
8. CT kVp/mAs、重建方式和层厚；
9. 配准软件、配准方向、插值方式和质量控制；
10. 基金、利益冲突、作者贡献和数据/代码可用性；
11. 真实 SelfRDB 中间采样状态 `states.npy`/`states.csv`；
12. 若中文稿使用的 `BridgeRefine_pipeline_final.svg` 不在本工作区，需要在写作机器补齐；当前 `figures_source/pipeline_source.svg` 只是明确标注的示意占位，不是真实定性证据；
13. 当前目录不是 Git 仓库，需要补源代码压缩包 SHA-256；
14. CT-only seed 42 和 BridgeRefine-L1 seed 42 的完整逐 epoch 训练日志需要从原训练记录补回。

这些缺项已列在：

```text
experiments/handoff_v2/MISSING.md
```

---

## 三、交接包验证

交接根目录：

```text
experiments/handoff_v2/
```

文件清单和哈希：

```text
experiments/handoff_v2/manifest.csv
```

同门接收后建议依次检查：

1. `manifest.csv` 中 available 文件全部存在；
2. `qualitative_cases/*/arrays.npz` 的键和形状正确；
3. `metrics/patient_metrics_final.csv` 的 18 名患者 ID 与测试清单一致；
4. `checkpoints/checkpoints_manifest.csv` 的哈希与本地 checkpoint 一致；
5. `efficiency/efficiency_final_10x2.csv` 只包含 10×2 结果；
6. `MISSING.md` 中的缺失项没有被静默填充；
7. 论文 PDF/LaTeX 中的效率、输入消融和多 seed 数字与 `metrics/`、`efficiency/` 一致。

---

## 四、结论

本次已经按审计说明完成可以在当前机器完成的实验修改和交接材料生成：

- 10×2 独立效率测量；
- 三个测试患者的真实定性数组和误差图；
- 完整患者级、配对、seed 和显著性导出；
- checkpoint provenance 和哈希；
- 配置、脚本和最新论文；
- 明确的 MISSING 清单。

不能在当前机器确认的伦理、采集、配准和治理信息仍保持缺失状态，等待数据负责人提供真实记录后再补入论文和交接包。
