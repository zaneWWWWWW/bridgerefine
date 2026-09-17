# BridgeRefine Architecture Figure v2

## Design Prompt

```text
Create an editable, publication-quality vector architecture figure for the
paired brain CT-to-MRI reconstruction method “BridgeRefine”. Use the supplied
reference SVG and PNG files only as visual style references. The figure must
represent the actual implemented method, not a generic end-to-end diffusion
model.

Use a clean landscape canvas, approximately 2:1 aspect ratio, white background,
rounded pale panels, dark navy text, muted blue for the frozen bridge, teal for
the trainable refiner, and amber for training-only supervision. Use English
labels and export editable SVG, vector PDF, and high-resolution PNG.

TOP PANEL: OVERALL FRAMEWORK

Title: “(a) Overall CT-to-MRI reconstruction framework”. Show the inference
path from left to right:

Input CT y
  → Frozen SelfRDB diffusion bridge
  → Coarse MRI representation x_bridge
  → Channel concatenation [CT, coarse MRI]
  → Supervised refiner R_theta
  → Synthetic MRI x_hat_MRI

Use a solid navy arrow for inference. Add a lock icon and the labels:
“Frozen during refiner training” and “10 sampling steps · 2 recursive
estimates” inside or below the SelfRDB block.

Show a direct CT bypass from Input CT to the channel-concatenation block,
labelled “Direct CT conditioning”. Make it clear that CT and coarse MRI are
concatenated along the channel dimension, not added or averaged.

Show a separate amber dashed training-only path:

Synthetic MRI + Paired reference MRI
  → L1 reconstruction loss
  → Update refiner only

Do not connect the reference MRI to the inference path. Do not draw a gradient
update to SelfRDB. Add a note: “Reference MRI is used only for training and
evaluation”.

LOWER PANEL: MODULE DETAILS

Panel (b), “Frozen diffusion bridge”: show CT conditioning entering SelfRDB,
then producing coarse MRI. Show a small sequence of abstract sampling-state
tiles and an ellipsis. Label the sequence “10 steps · 2 recursive estimates”.
Do not invent a standard DDPM Gaussian-noise start, cross-attention, latent
space, transformer, channel numbers, or internal SelfRDB layers. The bridge
is a pre-trained frozen module in this study.

Panel (c), “Supervised refiner”: show two separate input tiles, CT and Coarse
MRI, entering “Channel concat”. Then show:

Encoder → Residual block 1 → Residual block 2 → Residual block 3
→ Residual block 4 → Decoder → Synthetic MRI

Annotate the actual components only:
“InstanceNorm · Bilinear upsampling · Tanh output · ≈1.34 M trainable
parameters”. Use schematic feature-map stacks, but do not invent channel
counts, spatial resolutions, attention, perceptual branches, SSIM loss,
anatomical loss, or discriminator branches.

LOWER-LEFT ABLATION STRIP

Add a compact three-way input ablation strip:
“CT-only” | “coarse-only” | “CT + coarse”. State that all use the same
patient split and optimization protocol, with seed 42 for the main ablation.
Do not show ablation results inside the architecture diagram.

SCIENTIFIC BOUNDARIES

- This is a frozen diffusion bridge followed by a supervised refiner.
- The bridge output is a learned intermediate representation derived from CT,
  not an independently acquired MRI measurement.
- The main refiner loss is paired-image L1 loss.
- A plane-wise x/y gradient term exists only as an exploratory ablation; it is
  not a z-direction continuity or 3D anatomical loss.
- Inference uses CT only; paired MRI is unavailable during inference.
- Do not claim MRI replacement, diagnostic equivalence, lesion preservation,
  cost reduction, waiting-time reduction, or clinical validation.

VISUAL STYLE

Use dark navy #19324A for text and inference arrows, muted blue #5B83A8 for
the frozen bridge, pale blue #EEF4F9 for its panel, teal #178B82 for the
trainable refiner, pale teal #EDF8F5 for its panel, amber #D99A52 for
training-only supervision, and pale amber #FFF6E9 for the loss panel.
Use modest rounded corners, thin borders, orthogonal arrows, generous spacing,
and a small legend. Avoid gradients, glow, 3D effects, excessive icons, or
decorative anatomy.

IMAGE RULES

Use authentic experimental thumbnails only. The supplied qualitative image is
for choosing display style and representative CT/MRI/output tiles. Do not use
the schematic placeholders as quantitative evidence. Do not fabricate CT,
MRI, lesion, or diffusion intermediate images. If a genuine intermediate
state is unavailable, use a clearly labelled abstract tensor/sampling tile.
```

## Required Deliverables

- `BridgeRefine_architecture_v2.svg` (editable master)
- `BridgeRefine_architecture_v2.pdf` (vector export)
- `BridgeRefine_architecture_v2.png` (400–600 dpi preview)

## Final QA

- No reference MRI enters the inference pathway.
- No update arrow reaches frozen SelfRDB.
- CT and coarse MRI enter channel concatenation separately.
- The main loss is L1; no unsupported hybrid losses are shown.
- The bridge is labelled 10 steps × 2 recursive estimates.
- All English labels are readable at journal print width.
