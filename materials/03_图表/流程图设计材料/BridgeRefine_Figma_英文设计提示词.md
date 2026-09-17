# BridgeRefine Figma 英文流程图设计提示词

请参考 `reference_images/` 中的四张风格图，绘制一张可编辑的、适合医学影像论文投稿的 BridgeRefine 方法架构图。参考图只用于借鉴版式、字体层级、浅色面板、医学影像缩略图、冻结模块标识和上下层总览布局，不得复制其中的 ControlNet、超图、师生网络或下游分割等科学模块。

## Prompt

```text
Create an editable, publication-quality vector architecture figure for a medical imaging paper entitled “BridgeRefine: Two-Stage Framework for Paired Brain CT-to-MRI Synthesis”. Use the supplied reference images as visual style references only. Match their overview-plus-detail composition, pale tinted panels, rounded containers, medical image thumbnails, compact feature-map stacks, and clear frozen/trainable indicators. Use English labels only.

Use a landscape canvas with an aspect ratio close to 1.8:1. Place a complete overview in the top row, an enlarged Frozen bridge panel in the lower-left, and an enlarged Supervised refinement panel in the lower-right.

Top overview (a) must show the exact inference path: Input CT → Frozen SelfRDB bridge → Coarse MRI estimate → Channel concatenation → Supervised refiner → Synthetic MRI. Add an orthogonal bypass from Input CT to Channel concatenation labelled “Direct CT conditioning”. The coarse MRI is generated from CT and is not an independent patient observation.

Bottom-left panel (b), titled “Frozen bridge”, must show CT conditioning → SelfRDB sampling → Coarse MRI estimate. Add a lock icon and the labels “Frozen during refiner training” and “10 sampling steps · 2 recursive estimates”. Show a compact sequence of iterative image tiles and an ellipsis, but do not invent diffusion states, channel counts, time schedules, or internal layers.

Bottom-right panel (c), titled “Supervised refinement”, must show separate CT and Coarse MRI inputs entering a block labelled “Channel concat”, followed by Encoder → 4 residual blocks → Decoder → Synthetic MRI. Add “InstanceNorm · Bilinear upsampling · Tanh output · ≈1.34 M trainable parameters”. Use schematic feature-map stacks without invented resolutions, channels, attention, transformers, or skip connections.

Add a separate pale amber dashed training-only subpanel: Synthetic MRI + Paired reference MRI → L1 reconstruction loss. Draw a dashed arrow labelled “Update refiner only”. Do not connect reference MRI to inference, and do not update the frozen bridge. Add “Reference MRI is used only for training supervision”.

Use a white background, restrained scientific colors, thin borders, modest rounded corners, aligned orthogonal connectors, and readable sans-serif typography. Suggested colors: navy #19324A, frozen blue #5B83A8, pale blue #EEF4F9, trainable teal #178B82, pale teal #EDF8F5, supervision amber #D99A52, pale amber #FFF6E9. Use solid arrows for inference, dashed amber arrows for training supervision, and dashed gray lines for overview-to-detail correspondence.

Use authentic experimental CT/MRI thumbnails from experimental_images/ when available. Do not generate or retouch anatomy, lesions, or model outputs. Do not add ControlNet, text prompts, edge conditioning, hypergraphs, EMA, teacher–student branches, discriminators, segmentation heads, clinical diagnosis, MRI replacement claims, cost savings, or waiting-time claims.

Export an editable SVG, a vector PDF, and a high-resolution PNG preview. Check arrow directions, panel alignment, text legibility at 180 mm print width, font embedding, and absence of clipped or overlapping labels.
```

## 图像使用规则

- `reference_images/`：只用于风格参考，不作为实验结果。
- `experimental_images/qualitative_results.png`：用于选择真实定性案例和统一影像显示风格。
- `experimental_images/main_results.png`：仅作为结果图视觉参考，不要把柱状图内容塞入流程图。
- `experimental_images/current_pipeline_reference.png` 和 `redrawn_pipeline_draft.png`：作为现有流程和重绘方向参考，不应直接作为最终稿。

## 科学核对清单

- 推理路径只从 CT 开始。
- CT 和 coarse MRI 在 channel concatenation 处合并。
- 配对真实 MRI 只进入 L1 supervision。
- 训练时只更新 refiner，SelfRDB 保持冻结。
- coarse MRI 是 CT 派生表示，不是额外实测信息。
- 10 steps × 2 recursive estimates 对应主实验配置；不要写成 20 × 5。
- 不要在图中加入未经实验验证的临床或效率结论。
