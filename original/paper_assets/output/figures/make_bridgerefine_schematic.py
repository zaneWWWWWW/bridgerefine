"""BridgeRefine method-overview figure (nature-figure skill, Python backend)."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# MANDATORY editable-text settings for SVG / PDF
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Arial", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7

BLUE = "#0F4D92"
BLUE_SOFT = "#DDE9F6"
GREEN = "#2E9E44"
GREEN_SOFT = "#E6F4E8"
TEAL = "#42949E"
TEAL_SOFT = "#E3F1F2"
GRAY = "#4D4D4D"
GRAY_SOFT = "#F1F1F1"
RED = "#B64342"
RED_SOFT = "#F7E0DE"
PANEL_A = "#EDF2F9"
PANEL_B = "#EDF7F0"
PANEL_C = "#F6F6F6"


def draw_bg(ax, x, y, w, h, fc, ec="#B8BEC6", lw=1.0):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.15,rounding_size=1.0",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=0,
    )
    ax.add_patch(patch)


def draw_box(ax, x, y, w, h, text, fc="white", ec="#333333", lw=1.1,
             ls="-", fs=7.0, tc="black", zorder=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.10,rounding_size=0.6",
        linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls, zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, linespacing=1.35, zorder=3)


def draw_metric_box(ax, x, y, w, h, head, body, fs_head=6.2, fs_body=5.9):
    draw_box(ax, x, y, w, h, "", fc="white", ec=GRAY, fs=6.0)
    ax.text(x + w / 2, y + h - 1.1, head, ha="center", va="top",
            fontsize=fs_head, fontweight="bold")
    ax.text(x + w / 2, y + h / 2 + 0.2, body, ha="center", va="center",
            fontsize=fs_body, linespacing=1.35)


def draw_arrow(ax, x1, y1, x2, y2, color="#333333", lw=1.1, ls="-",
               style="-|>", zorder=1):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                        linestyle=ls, shrinkA=0, shrinkB=0),
        zorder=zorder,
    )


def draw_panel_label(ax, x, y, label, title, fs=8.5):
    ax.text(x, y, label, ha="left", va="top", fontsize=fs,
            fontweight="bold", color="#111111")
    ax.text(x + 4.5, y, title, ha="left", va="top", fontsize=fs,
            fontweight="bold", color="#111111")


def main():
    fig = plt.figure(figsize=(7.2, 4.7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 62)
    ax.axis("off")

    # ---- figure title -------------------------------------------------
    ax.text(50, 59.8,
            "BridgeRefine：基于冻结扩散桥与轻量监督精炼的脑 CT 到 MRI 转换",
            ha="center", va="top", fontsize=9.5, fontweight="bold")

    # ===================================================================
    # Panel a: frozen SelfRDB diffusion bridge
    # ===================================================================
    draw_bg(ax, 2, 31, 48, 25, PANEL_A)
    draw_panel_label(ax, 3.8, 53.5, "a", "冻结扩散桥（SelfRDB）")

    # four bridge states
    draw_box(ax, 5.5, 42.5, 5.6, 5.2, "目标 MRI\n$x_0$",
             fc="white", ec=BLUE, fs=6.6)
    draw_box(ax, 15.0, 42.5, 5.6, 5.2, "$x_{t-1}$",
             fc="white", ec=GRAY, fs=6.6)
    draw_box(ax, 24.5, 42.5, 5.6, 5.2, "$x_t$",
             fc="white", ec=GRAY, fs=6.6)
    draw_box(ax, 34.0, 42.5, 5.6, 5.2, "带噪 CT\n$x_T=y_\\varepsilon$",
             fc=GRAY_SOFT, ec=GRAY, fs=6.2)

    # forward / reverse arrows
    for x1, x2 in [(10.3, 13.9), (19.8, 23.4), (29.3, 32.9)]:
        draw_arrow(ax, x1, 48.4, x2, 48.4, color=BLUE)
    for x1, x2 in [(32.9, 29.3), (23.4, 19.8)]:
        draw_arrow(ax, x1, 40.9, x2, 40.9, color=TEAL)
    draw_arrow(ax, 17.8, 42.4, 11.75, 36.8, color=TEAL)

    ax.text(28.0, 50.3, "前向过程 $q(x_t \\mid x_{t-1}, y)$",
            ha="center", va="center", fontsize=6.8, color=BLUE)
    ax.text(28.0, 38.8, "反向过程 $p_\\theta(x_{t-1} \\mid x_t, y)$",
            ha="center", va="center", fontsize=6.8, color=TEAL)

    # coarse output and bridge configuration
    draw_box(ax, 5.5, 32.0, 12.5, 4.6, "粗略 MRI\n$\\hat{x}_{\\mathrm{bridge}}$",
             fc=TEAL_SOFT, ec=TEAL, fs=6.8)
    draw_box(ax, 21.0, 32.0, 27.0, 4.6,
             "SelfRDB：约 1,440 万参数，保持冻结\n推理：20 个采样步 × 5 次递归估计",
             fc="white", ec=GRAY, fs=6.0)
    # ===================================================================
    # Panel b: supervised refiner and BridgeGAN control
    # ===================================================================
    draw_bg(ax, 52, 31, 46, 25, PANEL_B)
    draw_panel_label(ax, 53.8, 53.5, "b", "监督精炼器与消融对照")

    draw_box(ax, 53.5, 46.0, 8.5, 3.8, "CT\n$y$",
             fc="white", ec=GRAY, fs=6.8)
    draw_box(ax, 53.5, 38.5, 10.5, 3.8, "粗略 MRI\n$\\hat{x}_{\\mathrm{bridge}}$",
             fc=TEAL_SOFT, ec=TEAL, fs=6.6)

    draw_arrow(ax, 62.2, 47.9, 75.3, 47.9, color=GRAY)
    draw_arrow(ax, 64.2, 40.4, 75.3, 40.4, color=GRAY)

    draw_box(ax, 75.5, 39.0, 13.0, 10.0,
             "监督精炼器 $R_\\theta$\n编码器–残差块–解码器\n约 130 万可训练参数",
             fc=GREEN_SOFT, ec=GREEN, fs=6.3)
    draw_arrow(ax, 88.7, 44.0, 90.3, 44.0, color=GREEN)

    draw_box(ax, 90.5, 41.5, 7.4, 5.0, "合成 MRI\n$\\hat{x}_{\\mathrm{MRI}}$",
             fc="white", ec=GREEN, fs=6.6)

    draw_box(ax, 90.5, 31.5, 7.4, 4.2, "配对 MRI\n$x_0$",
             fc=BLUE_SOFT, ec=BLUE, fs=6.6)
    draw_arrow(ax, 94.2, 41.3, 94.2, 35.8, color=RED, lw=1.2)
    ax.plot([94.2, 92.0], [38.8, 37.8], color=RED, lw=0.8, zorder=1)
    ax.text(88.0, 37.4, "L1 损失\n$\\mathcal{L}=\\|\\hat{x}_{\\mathrm{MRI}}-x_0\\|_1$",
            ha="center", va="center", fontsize=6.0, color=RED,
            bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none"))

    draw_box(ax, 53.5, 31.5, 28.0, 4.2,
             "BridgeGAN 对照：PatchGAN 判别器 $D$\n（仅消融，最终模型不使用）",
             fc=RED_SOFT, ec=RED, ls="--", fs=5.8)
    draw_arrow(ax, 79.0, 38.9, 79.0, 35.8, color=RED, ls="--", lw=0.9)

    # ===================================================================
    # Panel c: patient-level fixed-checkpoint evidence
    # ===================================================================
    draw_bg(ax, 2, 4, 96, 23, PANEL_C)
    draw_panel_label(ax, 3.8, 24.8, "c", "患者级固定检查点评估")

    rows = [
        (4.5, "SelfRDB 桥输出", "SSIM\n0.5790 ± 0.0405"),
        (22.5, "BridgeRefine-L1", "SSIM 0.8169 ± 0.0529\nPSNR 19.88 dB"),
        (41.5, "感知与掩膜指标", "LPIPS 0.075\n掩膜 SSIM 0.8551\n掩膜 PSNR 17.41 dB"),
        (60.0, "层间一致性与效率", "层间 MAD 比值 1.005\n推理 1,264 ms/切片"),
        (77.5, "CT-only 基线", "SSIM\n0.8009 ± 0.0093"),
    ]
    for x, head, body in rows:
        draw_metric_box(ax, x, 7.0, 16.5, 12.5, head, body)

    ax.text(50, 5.2,
            "数值均来自本文表 2 与效率实验：SSIM 为患者间均值 ± 样本标准差；"
            "CT-only 仅保留 SSIM；评估队列为 18 例患者、3,415 张轴位切片。",
            ha="center", va="bottom", fontsize=5.8, color="#444444")

    out = r"D:\Cross-modal conversion\output\figures\BridgeRefine_schematic"
    for ext, kw in [("svg", {}), ("pdf", {}),
                    ("tiff", {"dpi": 600}), ("png", {"dpi": 600})]:
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", **kw)
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()
