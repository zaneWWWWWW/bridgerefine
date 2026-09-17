# BridgeRefine research materials and code

配对脑 CT 到 MRI 转换研究的代码整理仓库。保存现有模型、训练/评估脚本、配置和画图代码，并提供可运行的结果汇总工具。仓库于 2026-09-17 从论文工作区整理；这是代码归档版本，尚不是能够从零一键复现全部论文实验的发行版。

代码、手稿和研究材料现统一保存在本私有仓库。可浏览的手稿、常用图表、指标和说明见 [`materials/`](materials/00_使用说明.md)；完整影像、预测、权重和高分辨率图均收录在 [完整材料 Release](https://github.com/zaneWWWWWW/bridgerefine/releases/tag/materials-2026-09-17) 的独立 ZIP 分包中。论文尚未正式发表；作者、引用格式与代码授权信息待确认。

## 手稿与完整材料

| 内容 | 入口 |
|---|---|
| 最新中文稿 PDF / LaTeX | [`materials/01_手稿/中文主稿/`](materials/01_手稿/中文主稿/) |
| 英文工作稿、投稿长稿、匿名稿、补充材料 | [`materials/01_手稿/`](materials/01_手稿/) |
| 常用图表和指标 | [`materials/03_图表/`](materials/03_图表/) / [`materials/04_数据/`](materials/04_数据/) |
| 全部影像、模型、预测、高分辨率图和其余文件 | [下载全部 ZIP 分包](https://github.com/zaneWWWWWW/bridgerefine/releases/tag/materials-2026-09-17) |
| 版本差异及复现缺项 | [`materials/06_整理记录/版本核对与缺项.md`](materials/06_整理记录/版本核对与缺项.md) |

每个 ZIP 都能独立解压，不是需要拼接的二进制分卷。下载所有分包后，自动校验 SHA-256、解压并恢复相对链接：

```bash
gh auth login  # 已登录且可访问本私有仓库时跳过
python3 tools/download_materials.py
```

该命令需要约 20 GB 可用空间用于 ZIP 缓存和解压，默认恢复到仓库内 `materials/`。已有且相同的文件会复用；内容冲突时停止，避免覆盖本地修改。当前 `git clone` 只含供浏览的小文件，完整材料通过上述 Release 下载。

## 目录

| 路径 | 内容 |
|---|---|
| `original/project/selfrdb/` | 数据集、预处理和 SelfRDB 模型/训练/推理实现 |
| `original/project/other_model/SelfRDB/` | 实验所引用的桥模型及网络 |
| `original/project/other_model/BridgeGAN/model.py` | BridgeRefine 使用的 `RefinerGenerator` 网络 |
| `original/project/other_model/improve_l1_unet.py` | 原始 L1 精炼训练脚本，保留历史实现 |
| `original/project/other_model/phase2_ct_unet.py` | CT-only 与历史损失实验 |
| `original/project/other_model/final_multiseed*.py` | 历史多 seed 实验；包含梯度损失，不能视为最终纯 L1 多 seed 脚本 |
| `original/project/other_model/{SynDiff,MG_CycleGAN,Masked_Bridge,Masked_Adv_Diff_Bridge}/` | 对照及探索实验 |
| `original/handoff_v2/` | 10×2 评估、效率测量与统计脚本原件 |
| `original/figure_scripts/` | 原始图表绘制脚本，部分包含旧结果 |
| `configs/` | 交接材料中记录的模型/评估参数说明 |
| `tools/` | 本次新增、可独立使用的来源校验与结果汇总工具 |
| `SOURCE_MANIFEST.json` | 每份原始代码/配置的来源与 SHA-256 |

## 可直接使用的工具

建议 Python 3.11+。结果汇总仅需 NumPy、SciPy，不需要 GPU 或影像数据。材料下载工具仅依赖 Python 标准库和已登录的 GitHub CLI。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python tools/verify_sources.py
```

在本地整理包的 `02_代码/bridgerefine` 中运行：

```bash
python tools/summarize_results.py \
  --metrics ../../04_数据/01_最终指标/metrics \
  --seeds ../../04_数据/02_原始结果表/投稿版/seed_results.csv \
  --efficiency ../../04_数据/01_最终指标/efficiency/efficiency_final_10x2.csv \
  --output results/recomputed
```

从 GitHub 克隆时，也可以把上述三个输入路径的 `../../` 改为 `materials/`，直接使用已随 Git 提交的结果表。输出包含患者均值/样本标准差、seed 波动、效率和显著性核对；原始证据不会被覆盖。Bootstrap 使用 `default_rng(42)`；Wilcoxon 基于已导出的四位小数差值，Holm 实现包含累积最大值，输出与旧记录的差异单独列出。

## 原始实验代码的状态

`original/` 内容逐字节保留，不在此次归档中改写算法。其 Windows 路径、默认采样参数以及历史实现问题也保留，因此不要将“语法检查通过”等同于“已复现训练”。

- 最终桥采样协议为 10 steps × 2 recursions；部分旧脚本仍默认 20×5。
- 交接记录引用的 `train_input_ablation_and_l1_seeds.py` 在已解压工程和原 ZIP 中均未找到；最终多 seed L1 模型的完整训练入口仍需从原训练机器补回。
- `original/handoff_v2/scripts/statistics.py` 原件只有两行注释；可使用新增的 `tools/summarize_results.py` 进行结果核对。
- 部分旧精炼脚本通过 `idx % len(coarse)` 取缓存，不能据此保证患者与切片对齐；重跑前必须检查缓存对应关系。
- 评估和 benchmark 原件依赖旧目录布局、缓存、GPU 和影像路径。当前配置 YAML 是参数记录，未实现统一配置加载入口。
- `checkpoints/selfrdb/selfrdb_best.pt` 和 `other_model/SelfRDB/selfrdb_best.pt` 在原材料中不是同一文件。应按论文 checkpoint 清单核对 SHA-256；整理包已恢复并核对全部 10 个主要权重，另附 coarse-only 权重。

原训练环境的可恢复依赖见 `requirements-training-recorded.txt`，它是归档记录，不是已验证的跨平台环境锁。此次未重新训练网络、重算全套影像指标或复测 GPU 时延。

## 验证与授权

本次验证项目与结论见 `VALIDATION.md`。当前 Git 提交记录的是整理时的源代码快照，不能追溯证明既有模型由该提交训练得到。

仓库暂为私有。没有替未知权利人授予开源许可；已有来源的权利归属保留，见 `NOTICE.md`。
