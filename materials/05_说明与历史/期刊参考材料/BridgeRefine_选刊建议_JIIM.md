# BridgeRefine 选刊建议：Journal of Imaging Informatics in Medicine

核验日期：2026-09-04

## 推荐结论

首投建议：**Journal of Imaging Informatics in Medicine（JIIM）**，原刊名为 *Journal of Digital Imaging*，Springer Nature 出版，Society for Imaging Informatics in Medicine 官方期刊。

推荐稿件类型：以 **data-driven original research / research article** 的方式组织，不建议把稿件写成单纯算法说明或临床替代研究。

匹配度判断：**中等偏高，但必须补齐伦理、扫描参数、配准和数据治理信息。** 该判断只依据实验完整性、结果幅度和贡献类型，不依据当前中文稿文风。

## 为什么适合

### 1. 期刊范围直接覆盖研究主题

JIIM 官方 aims and scope 明确包含：

- 医学影像环境中的临床、工程和信息技术研究；
- artificial intelligence、machine learning、deep learning 和 generative AI；
- image processing、computer-aided diagnosis、quantitative imaging；
- AI safety、fairness、bias、quality assurance 和 data curation。

BridgeRefine 属于医学影像中的跨模态生成、定量评价和效率分析，主题在该刊范围内。

官方范围页：<https://www.springer.com/journal/10278/aims-and-scope>

### 2. 你的实验包达到该刊可考虑的完整度

现有实验包括：

- 180 名急性脑卒中患者，按患者划分为 144/18/18；
- 3,415 张测试切片；
- SelfRDB、SynDiff、MG-CycleGAN、BridgeGAN 和 CT-only 强基线；
- CT-only/coarse-only/CT+coarse 输入消融；
- BridgeRefine-L1 三随机种子；
- 患者级 Wilcoxon、Holm 校正和 bootstrap CI；
- SSIM、PSNR、LPIPS、脑掩膜指标和层间 MAD；
- 推理时间、参数量和额外显存；
- Gradient、GAN 和早期 DDPM 的负结果；
- 可追溯 checkpoint、患者级/切片级 CSV 和冻结结果包。

这套证据比“只报告一个模型和一张结果表”的纯算法稿完整，适合写成验证导向的影像信息学研究。

### 3. 贡献类型与同刊已发表论文相符

BridgeRefine 的贡献不是全新扩散理论，而是：

1. 将冻结扩散桥和监督精炼器进行职责分解；
2. 通过输入消融识别 CT 与 coarse MRI 的相对信息贡献；
3. 通过多随机种子和患者级统计确认有限增益；
4. 揭示约 420 倍推理成本的 accuracy-efficiency trade-off；
5. 报告 Gradient 和对抗监督的负结果。

JIIM 已发表多篇以系统验证、数据规模、评价可靠性和负结果为贡献的生成式医学影像论文，因此不要求每篇文章都提出新的基础模型。

## 主要风险

### 1. 期刊更偏好验证或实际应用，而非单纯算法描述

JIIM 官方投稿指南明确指出：相较于只描述图像处理算法、独立分析工具或技术本身的论文，期刊优先考虑经过验证或实施的影像信息学技术。

因此新稿不能以“我们提出一个新扩散网络”为主线，应写成：

> A patient-level validation study of a decoupled diffusion-bridge and supervised-refinement workflow for paired brain CT-to-synthetic-MRI translation.

正文应强调公平比较、输入来源消融、患者级统计、重复性、失败配置和部署代价。

### 2. 临床证据仍然有限

- 单中心数据；
- 只有 18 名测试患者；
- 没有外部验证；
- 没有盲法读片；
- 没有病灶检出、分割或预后等下游终点；
- 输入为 2D、128x128 切片。

因此不能把论文写成“CT 可替代 MRI”或“临床诊断等价”。适合的结论是“在当前队列上提高重建保真度，并揭示准确率与效率的权衡”。

### 3. 投稿前的硬性缺口

- 伦理审批/豁免和知情同意；
- 去标识化及数据治理；
- CT/MRI 扫描设备和参数；
- MRI 序列；
- CT-MRI 采集间隔；
- 配准方法和质量控制；
- 完整代码版本哈希；
- 作者、基金、利益冲突和贡献声明。

这些项目不补齐，会比模型性能更容易导致编辑部退回。

## 官方期刊信息

| 项目 | 核验结果 |
|---|---|
| 期刊 | Journal of Imaging Informatics in Medicine |
| 出版商 | Springer Nature |
| 学会 | Society for Imaging Informatics in Medicine |
| 电子 ISSN | 2948-2933 |
| 出版模式 | Hybrid |
| 官方 2025 JIF | 3.1 |
| 官方 2025 5-year JIF | 3.1 |
| 官方页面列出的索引 | SCIE、Scopus、PubMed Central、EI Compendex 等 |
| 官方页面给出的首次决定中位数 | 5 天；该数字可能包含编辑初筛，不等同于完整同行评审周期 |
| 摘要 | 150--250 英文词 |
| 关键词 | 4--6 个 |
| 评审 | Double blind，需要完整稿和匿名稿 |
| 投稿文件 | Word 推荐；含数学内容可提交 LaTeX |

期刊主页：<https://www.springer.com/journal/10278>

投稿指南：<https://www.springer.com/journal/10278/submission-guidelines>

### 分区说明

Springer 官方公开页没有显示可核验的 JCR quartile。JCR 分区、中科院分区和学校认定并不是同一体系，也会随年份变化。投稿前必须通过你所在单位可访问的 **2025 JCR**、当年中科院分区表及学校预警/认可名单确认；不要仅使用第三方期刊查询网站。

## 推荐对标论文

### 主对标论文

Matteo Lai, Mario Mascalchi, Carlo Tessa, Stefano Diciotti. **Generating Brain MRI with StyleGAN2-ADA: The Effect of the Training Set Size on the Quality of Synthetic Images.** *Journal of Imaging Informatics in Medicine*, 39:2319--2329. DOI: 10.1007/s10278-025-01536-0.

推荐理由：

- 同一期刊；
- 同为脑 MRI 合成；
- 同为二维生成实验；
- 贡献重点是实验问题和评价框架，不是全新基础架构；
- 系统报告正结果、负结果、计算成本、数据/代码、伦理和局限性；
- 使用多维指标，并讨论指标失真和 mode collapse；
- 正式版本为 CC BY 4.0，可合法下载和制作中文译读版。

原文 DOI：<https://doi.org/10.1007/s10278-025-01536-0>

## 与对标论文的比较

| 维度 | Lai et al. | BridgeRefine |
|---|---|---|
| 任务 | 无条件健康脑 MRI 合成 | 配对脑 CT 到合成 MRI |
| 数据 | 3,227 训练，757 独立测试，多来源公开数据 | 144/18/18 患者，单中心临床配对数据 |
| 核心问题 | 训练集规模如何影响保真、多样性和泛化 | coarse MRI 是否提供 CT 之外的信息，收益是否值得计算代价 |
| 方法 | StyleGAN2-ADA | 冻结 SelfRDB + L1 refiner |
| 评价 | FID/KID、precision/recall、density/coverage、authenticity、Turing test | SSIM、PSNR、LPIPS、mask 指标、MAD、患者级统计 |
| 重复性 | 3 个训练集规模 | 3 个随机种子 |
| 人工评价 | 两位资深专家视觉 Turing 测试 | 无读者实验 |
| 下游任务 | 无，作为局限性 | 无，作为局限性 |
| 主要负结果 | 增加数据不能解决 mode collapse；多样性指标受样本数影响 | Gradient 无一致收益；对抗配置未改善；420 倍延迟 |

## 按该刊重写 BridgeRefine 的建议结构

### Title

避免“clinical replacement”“diagnostic equivalence”“novel diffusion model”。建议：

> Patient-Level Evaluation of Diffusion-Bridge-Assisted Supervised Refinement for Brain CT-to-Synthetic-MRI Translation

或：

> Does a Diffusion Bridge Add Value Beyond Direct CT-to-MRI Regression? A Patient-Level Ablation and Efficiency Study

第二个标题更贴合实验发现，也更像 JIIM 的验证型论文。

### Abstract

150--250 英文词，按以下顺序：

1. 背景：跨模态合成需要结构保真，扩散桥是否比直接监督回归更有价值尚不清楚；
2. 方法：180 名患者，144/18/18 划分，三输入消融、多 seed、患者级统计；
3. 结果：0.8184 vs 0.8095，delta 0.0089，17/18，p=0.0002，420 倍延迟；
4. 结论：coarse MRI 提供有限增量，但 CT 是主要信息源，部署需权衡成本。

### Introduction

1. 定义 CT-to-synthetic-MRI 重建任务及临床解释边界；
2. 对比直接回归、GAN 和扩散方法；
3. 提出尚未回答的问题：扩散桥是否提供 CT 之外的信息；
4. 给出两阶段框架和可检验假设；
5. 列出患者级消融、重复性和效率贡献。

### Materials and Methods

- 数据来源、伦理、采集和配准；
- 患者级划分；
- 统一 10x2 coarse 协议；
- CT-only、coarse-only、CT+coarse；
- 对比方法；
- 指标和患者级聚合；
- Wilcoxon、Holm 和 bootstrap CI；
- 硬件和效率协议。

### Results

1. 主比较；
2. 输入来源消融；
3. 患者级 paired analysis；
4. 三 seed；
5. Gradient/GAN 负结果；
6. accuracy-efficiency trade-off；
7. 定性病例。

### Discussion

重点不是重复 SSIM，而是回答：

- CT 是主要信息源；
- coarse MRI 有小幅但一致的增量；
- 监督精炼是主要增益来源；
- 0.0089 SSIM 是否值得约 420 倍推理时间，取决于离线/低延迟场景；
- 指标提升不代表病灶或诊断等价；
- 单中心、2D、小测试队列和无读者实验限制临床解释。

### Declarations

JIIM 要求在参考文献前提供 Funding、Competing interests、Ethics approval、Consent、Data/Code availability、Author contributions。期刊采用双盲评审，需准备完整版本和匿名版本。

## 投稿策略

1. 先补齐医学元数据和伦理信息，再开始英文稿；
2. 用 JIIM 的验证型叙事重写，而不是直接翻译现有中文稿；
3. 主标题和摘要突出“does the bridge add value”与患者级验证；
4. 正文保留主结果、输入消融、三 seed 和效率，pilot 放补充材料；
5. 在 cover letter 中说明文章提供了组件贡献、重复性和成本的完整证据，而非宣称新的扩散理论；
6. 投稿前再次核验 2025 JCR/中科院分区和学校认可名单。

## 不建议优先冲击的期刊

- **Medical Image Analysis**：通常要求更强的方法创新、多数据集/外部验证和更广泛实验，当前贡献偏工程分解和单中心验证。
- **Computerized Medical Imaging and Graphics**：主题匹配，但近期同类论文往往有语义保持设计、外部验证或更强任务终点；可以作为升级目标，而不是当前最稳妥首投。
- **BMC Medical Imaging**：更偏临床影像和临床问题验证；当前缺少读者实验、病灶终点和外部队列，需先加强临床验证。

