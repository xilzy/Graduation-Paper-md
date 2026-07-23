#!/usr/bin/env python3
"""Generate the U-MoE-Fusion framework figure (SVG/PDF/PNG).

The figure is intentionally self-contained: raster examples are embedded into
SVG/PDF, while all boxes, arrows, labels, and equations remain vector objects.
It mirrors the released W96L configuration and the five method innovations
validated in the thesis ablation study.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path


def _bootstrap_mpl_cache() -> None:
    """Use a fixed bundled-font cache; avoid scanning shared storage."""
    cache = Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/u_moe_framework_mpl"))
    os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")
    cache.mkdir(parents=True, exist_ok=True)
    version = importlib.metadata.version("matplotlib")
    cache_file = cache / f"fontlist-v{version}.json"
    if cache_file.exists():
        return

    def entry(filename, weight=400, style="normal"):
        return {
            "fname": f"fonts/ttf/{filename}",
            "index": 0,
            "name": "DejaVu Sans",
            "style": style,
            "variant": "normal",
            "weight": weight,
            "stretch": "normal",
            "size": "scalable",
            "__class__": "FontEntry",
        }

    payload = {
        "_version": version,
        "_FontManager__default_weight": "normal",
        "default_size": None,
        "defaultFamily": {"ttf": "DejaVu Sans", "afm": "Helvetica"},
        "afmlist": [],
        "ttflist": [
            entry("DejaVuSans.ttf"),
            entry("DejaVuSans-Bold.ttf", 700),
            entry("DejaVuSans-Oblique.ttf", 400, "oblique"),
            entry("DejaVuSans-BoldOblique.ttf", 700, "oblique"),
        ],
        "__class__": "FontManager",
    }
    cache_file.write_text(json.dumps(payload), encoding="utf-8")


_bootstrap_mpl_cache()

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image, ImageOps


COLORS = {
    "ink": "#17324D",
    "muted": "#557086",
    "line": "#A9BBC9",
    "panel": "#F6F9FC",
    "blue": "#4C9ED9",
    "blue_light": "#DCEEF9",
    "teal": "#42B7AA",
    "teal_light": "#DDF4F1",
    "orange": "#F2A65A",
    "orange_light": "#FCEBD8",
    "coral": "#E76F51",
    "coral_light": "#FBE4DE",
    "purple": "#8B6BBE",
    "purple_light": "#EEE8F7",
    "green": "#62AE72",
    "green_light": "#E5F3E8",
    "yellow": "#F4C95D",
    "yellow_light": "#FFF6D8",
    "white": "#FFFFFF",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "text.color": COLORS["ink"],
            "axes.edgecolor": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def rounded(ax, x, y, w, h, fc="white", ec=None, lw=1.0, radius=0.012,
            z=1, ls="-"):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec if ec is not None else COLORS["line"],
        linewidth=lw,
        linestyle=ls,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def text(ax, x, y, s, size=9, weight="normal", color=None, ha="center",
         va="center", z=10, linespacing=1.15, style="normal"):
    return ax.text(
        x, y, s,
        fontsize=size,
        fontweight=weight,
        color=color or COLORS["ink"],
        ha=ha,
        va=va,
        zorder=z,
        linespacing=linespacing,
        fontstyle=style,
    )


def arrow(ax, p0, p1, color=None, lw=1.5, style="-|>", ms=10, z=4,
          connectionstyle="arc3", ls="-"):
    patch = FancyArrowPatch(
        p0, p1,
        arrowstyle=style,
        mutation_scale=ms,
        linewidth=lw,
        color=color or COLORS["ink"],
        connectionstyle=connectionstyle,
        linestyle=ls,
        shrinkA=0,
        shrinkB=0,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def load_thumb(path: Path, size=(240, 180), grayscale=False):
    if not path.is_file():
        arr = np.ones((size[1], size[0], 3), dtype=np.uint8) * 241
        arr[::12, :, :] = 225
        arr[:, ::12, :] = 225
        return arr
    with Image.open(path) as im:
        im = im.convert("L" if grayscale else "RGB")
        im = ImageOps.fit(im, size, method=Image.Resampling.LANCZOS)
        return np.asarray(im)


def image_box(ax, path, x, y, w, h, border, grayscale=False, z=3):
    arr = load_thumb(Path(path), grayscale=grayscale)
    ax.imshow(
        arr, extent=(x, x + w, y, y + h), aspect="auto", zorder=z,
        cmap="gray" if grayscale else None,
        vmin=0 if grayscale else None, vmax=255 if grayscale else None,
    )
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=border,
                           linewidth=1.2, zorder=z + 1))


def task_card(ax, y, title, source_a, source_b, label_a, label_b, accent,
              gray_b=True):
    x, w, h = 0.018, 0.174, 0.174
    rounded(ax, x, y, w, h, fc=COLORS["white"], ec=accent, lw=1.5, radius=0.011, z=2)
    text(ax, x + 0.010, y + h - 0.018, title, size=9.2, weight="bold",
         color=accent, ha="left")
    iw, ih, gap = 0.071, 0.097, 0.009
    ix1 = x + 0.012
    ix2 = ix1 + iw + gap
    iy = y + 0.040
    image_box(ax, source_a, ix1, iy, iw, ih, accent, grayscale=False)
    image_box(ax, source_b, ix2, iy, iw, ih, accent, grayscale=gray_b)
    text(ax, ix1 + iw / 2, y + 0.022, label_a, size=7.3, weight="bold")
    text(ax, ix2 + iw / 2, y + 0.022, label_b, size=7.3, weight="bold")
    return (x + w, y + h / 2)


def output_card(ax, y, title, image, accent, grayscale=False):
    x, w, h = 0.920, 0.067, 0.174
    rounded(ax, x, y, w, h, fc=COLORS["white"], ec=accent, lw=1.5,
            radius=0.010, z=2)
    text(ax, x + w / 2, y + h - 0.019, title, size=7.8, weight="bold", color=accent)
    image_box(ax, image, x + 0.008, y + 0.026, w - 0.016, h - 0.061,
              accent, grayscale=grayscale)


def chip(ax, x, y, w, h, label, fc, ec, size=8, weight="bold"):
    rounded(ax, x, y, w, h, fc=fc, ec=ec, lw=1.0, radius=0.010, z=4)
    text(ax, x + w / 2, y + h / 2, label, size=size, weight=weight, color=ec)


def build_figure(code_root: Path, data_root: Path, output_dir: Path):
    setup_style()
    fig = plt.figure(figsize=(18.0, 9.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Paths for representative examples used elsewhere in the thesis figures.
    examples = {
        "ir_a": data_root / "MSRS/test/vi/00778N.png",
        "ir_b": data_root / "MSRS/test/ir/00778N.png",
        "ir_f": data_root.parent / "fusion_bench/fused/W96L/irvis/00778N.png",
        "med_a": data_root / "Harvard-Medical/test/func/pet_25027.png",
        "med_b": data_root / "Harvard-Medical/test/mri/pet_25027.png",
        "med_f": data_root.parent / "fusion_bench/fused_final/W96L/medical/pet_25027.png",
        "gfp_a": code_root / "source images/GFP-PC/GFP/05-A02-g.jpg",
        "gfp_b": code_root / "source images/GFP-PC/PCI/05-A02-t.jpg",
        "gfp_f": data_root.parent / "fusion_bench/fused_final/W96L/gfp_pc/05-A02.png",
    }

    # Title and panel headings.
    text(ax, 0.5, 0.975, "U-MoE-Fusion: One Model for Unified Multi-Modal Image Fusion",
         size=18, weight="bold")
    text(ax, 0.104, 0.931, "A  Unified multi-task input", size=10.5, weight="bold",
         color=COLORS["blue"])
    text(ax, 0.535, 0.931, "B  Multi-scale U-MoE backbone", size=10.5, weight="bold",
         color=COLORS["purple"])
    text(ax, 0.870, 0.931, "C  Decision-map fusion", size=10.5, weight="bold",
         color=COLORS["orange"])

    # Main panel backgrounds.
    rounded(ax, 0.010, 0.151, 0.290, 0.757, fc="#F8FBFD", ec="#CCDAE4",
            lw=1.1, radius=0.014, z=0)
    rounded(ax, 0.310, 0.151, 0.448, 0.757, fc="#FAF8FD", ec="#D8CCE9",
            lw=1.1, radius=0.014, z=0)
    rounded(ax, 0.768, 0.151, 0.222, 0.757, fc="#FFFBF6", ec="#E8D4BD",
            lw=1.1, radius=0.014, z=0)

    # Three task-specific source pairs.
    source_ports = []
    source_ports.append(task_card(ax, 0.699, "IR–VIS", examples["ir_a"], examples["ir_b"],
                                  "Visible", "Infrared", COLORS["blue"], gray_b=True))
    source_ports.append(task_card(ax, 0.489, "Medical", examples["med_a"], examples["med_b"],
                                  "PET/SPECT", "MRI", COLORS["coral"], gray_b=True))
    source_ports.append(task_card(ax, 0.279, "Microscopy", examples["gfp_a"], examples["gfp_b"],
                                  "GFP", "Phase contrast", COLORS["green"], gray_b=True))

    # Unified luminance contract and balanced task stream.
    px, py, pw, ph = 0.207, 0.421, 0.079, 0.382
    rounded(ax, px, py, pw, ph, fc=COLORS["blue_light"], ec=COLORS["blue"],
            lw=1.4, radius=0.012, z=3)
    text(ax, px + pw / 2, py + ph - 0.034, "Unified\npreprocessing", size=9.2,
         weight="bold", color=COLORS["blue"])
    text(ax, px + pw / 2, py + 0.268, "RGB → YCbCr", size=8.0, weight="bold")
    text(ax, px + pw / 2, py + 0.222, "Fuse luminance Y\nkeep source CbCr", size=7.6)
    ax.plot([px + 0.010, px + pw - 0.010], [py + 0.184, py + 0.184],
            color="#A8CBE2", lw=1.0, zorder=4)
    text(ax, px + pw / 2, py + 0.146, "$X=[Y_A,Y_B]$\n2-channel input", size=8.0,
         weight="bold")
    text(ax, px + pw / 2, py + 0.084, "170×170 crops", size=7.6)
    text(ax, px + pw / 2, py + 0.041, "balanced task quota", size=7.6)
    for port in source_ports:
        arrow(ax, port, (px, port[1]), color=COLORS["muted"], lw=1.25, ms=8)

    # Task ID / embedding is a separate conditioning stream.
    rounded(ax, 0.208, 0.309, 0.077, 0.078, fc=COLORS["purple_light"],
            ec=COLORS["purple"], lw=1.2, radius=0.012, z=4)
    text(ax, 0.2465, 0.353, "Task ID $t$", size=8.3, weight="bold", color=COLORS["purple"])
    text(ax, 0.2465, 0.327, "embedding $e_t$", size=7.5)
    for _, cy in source_ports:
        arrow(ax, (0.194, cy - 0.045), (0.207, 0.350), color=COLORS["purple"],
              lw=0.8, ms=6, ls="--", connectionstyle="arc3,rad=-0.12")

    # Input enters task-aware stem.
    arrow(ax, (px + pw, py + 0.125), (0.326, 0.612), color=COLORS["ink"], lw=1.8, ms=11)
    rounded(ax, 0.326, 0.565, 0.071, 0.094, fc=COLORS["teal_light"],
            ec=COLORS["teal"], lw=1.5, radius=0.010, z=4)
    text(ax, 0.3615, 0.625, "3×3 Conv", size=8.5, weight="bold", color="#247E75")
    text(ax, 0.3615, 0.590, "+ task bias", size=7.8, weight="bold", color=COLORS["purple"])
    arrow(ax, (0.285, 0.348), (0.361, 0.565), color=COLORS["purple"],
          lw=1.2, ms=8, ls="--", connectionstyle="arc3,rad=0.12")

    # Split to three receptive-field branches.
    split_x = 0.412
    ax.plot([0.397, split_x], [0.612, 0.612], color=COLORS["ink"], lw=1.4, zorder=3)
    ax.plot([split_x, split_x], [0.438, 0.786], color=COLORS["ink"], lw=1.4, zorder=3)
    branch_y = [0.767, 0.594, 0.421]
    branch_labels = ["shallow context", "mid-level context", "deep context"]
    conv_counts = ["ACM ×1", "ACM ×2", "ACM ×3"]

    for i, cy in enumerate(branch_y):
        arrow(ax, (split_x, cy), (0.430, cy), color=COLORS["ink"], lw=1.3, ms=8)
        rounded(ax, 0.430, cy - 0.036, 0.060, 0.072, fc=COLORS["teal_light"],
                ec=COLORS["teal"], lw=1.25, radius=0.008, z=4)
        text(ax, 0.460, cy + 0.008, conv_counts[i], size=7.9, weight="bold", color="#247E75")
        text(ax, 0.460, cy - 0.018, branch_labels[i], size=6.4, color=COLORS["muted"])
        arrow(ax, (0.490, cy), (0.506, cy), lw=1.25, ms=8)

        # Transformer / MoE block with two explicit sublayers.
        rounded(ax, 0.506, cy - 0.050, 0.154, 0.100, fc=COLORS["white"],
                ec=COLORS["purple"], lw=1.35, radius=0.010, z=4)
        rounded(ax, 0.514, cy - 0.038, 0.061, 0.076, fc=COLORS["orange_light"],
                ec=COLORS["orange"], lw=1.0, radius=0.007, z=5)
        text(ax, 0.5445, cy + 0.010, "8×8 window", size=7.3, weight="bold", color="#A96220")
        text(ax, 0.5445, cy - 0.014, "attention", size=7.3, weight="bold", color="#A96220")
        arrow(ax, (0.576, cy), (0.590, cy), color=COLORS["purple"], lw=1.0, ms=7)
        rounded(ax, 0.590, cy - 0.038, 0.061, 0.076, fc=COLORS["purple_light"],
                ec=COLORS["purple"], lw=1.0, radius=0.007, z=5)
        text(ax, 0.6205, cy + 0.010, "task-cond.", size=7.2, weight="bold", color=COLORS["purple"])
        text(ax, 0.6205, cy - 0.014, "MoE-FFN", size=7.2, weight="bold", color=COLORS["purple"])
        text(ax, 0.651, cy + 0.039, "×4", size=7.0, weight="bold", color=COLORS["purple"], ha="right")

        arrow(ax, (0.660, cy), (0.677, cy), lw=1.25, ms=8)
        rounded(ax, 0.677, cy - 0.036, 0.047, 0.072, fc=COLORS["teal_light"],
                ec=COLORS["teal"], lw=1.2, radius=0.008, z=4)
        text(ax, 0.7005, cy, "ACM", size=8.0, weight="bold", color="#247E75")
        arrow(ax, (0.724, cy), (0.739, cy), lw=1.2, ms=8)

    # Shared parameters and task conditioning indicators.
    text(ax, 0.583, 0.847, "shared weights across all tasks", size=7.8,
         color=COLORS["muted"], style="italic")
    arrow(ax, (0.285, 0.348), (0.620, 0.389), color=COLORS["purple"], lw=1.2,
          ms=8, ls="--", connectionstyle="arc3,rad=-0.10")
    ax.plot([0.620, 0.620], [0.389, 0.731], color=COLORS["purple"], lw=1.0,
            ls="--", zorder=3)
    for cy in branch_y:
        arrow(ax, (0.620, cy - 0.052), (0.620, cy - 0.039), color=COLORS["purple"],
              lw=1.0, ms=6, ls="--")

    # Multi-branch feature summation.
    sum_x, sum_y = 0.744, 0.594
    for cy in branch_y:
        ax.plot([0.739, sum_x], [cy, sum_y], color=COLORS["ink"], lw=1.1, zorder=3)
    rounded(ax, 0.729, 0.565, 0.030, 0.058, fc=COLORS["green_light"],
            ec=COLORS["green"], lw=1.4, radius=0.016, z=5)
    text(ax, 0.744, 0.594, "Σ", size=14, weight="bold", color=COLORS["green"])

    # MoE zoom-in inset.
    rounded(ax, 0.326, 0.171, 0.416, 0.176, fc="#F5F0FA", ec=COLORS["purple"],
            lw=1.25, radius=0.012, z=2)
    text(ax, 0.338, 0.326, "Task-conditioned MoE-FFN (inside each Transformer block)",
         size=8.5, weight="bold", color=COLORS["purple"], ha="left")
    # Input token and router.
    chip(ax, 0.340, 0.224, 0.050, 0.055, "token $h$", COLORS["white"], COLORS["ink"], size=7.5)
    chip(ax, 0.340, 0.181, 0.050, 0.031, "$e_t$", COLORS["purple_light"], COLORS["purple"], size=7.5)
    rounded(ax, 0.412, 0.214, 0.074, 0.076, fc=COLORS["purple_light"],
            ec=COLORS["purple"], lw=1.2, radius=0.009, z=4)
    text(ax, 0.449, 0.264, "softmax router", size=7.6, weight="bold", color=COLORS["purple"])
    text(ax, 0.449, 0.235, "top-2 selection", size=7.0)
    arrow(ax, (0.390, 0.252), (0.412, 0.252), lw=1.0, ms=7)
    arrow(ax, (0.390, 0.196), (0.427, 0.214), color=COLORS["purple"], lw=1.0,
          ms=7, ls="--")

    # Shared expert path and routed experts.
    rounded(ax, 0.505, 0.265, 0.084, 0.044, fc=COLORS["teal_light"],
            ec=COLORS["teal"], lw=1.1, radius=0.008, z=4)
    text(ax, 0.547, 0.287, "shared expert", size=7.2, weight="bold", color="#247E75")
    text(ax, 0.547, 0.270, "always on", size=6.3, color=COLORS["muted"])
    arrow(ax, (0.390, 0.267), (0.505, 0.287), color=COLORS["teal"], lw=1.0,
          ms=7, connectionstyle="arc3,rad=-0.08")

    expert_x = [0.505, 0.558, 0.611, 0.664]
    expert_labels = ["E1", "E2", "…", "E12"]
    for ex, lab in zip(expert_x, expert_labels):
        rounded(ax, ex, 0.200, 0.040, 0.047, fc=COLORS["orange_light"],
                ec=COLORS["orange"], lw=1.0, radius=0.007, z=4)
        text(ax, ex + 0.020, 0.2235, lab, size=7.2, weight="bold", color="#A96220")
    for ex in (0.505, 0.558):
        arrow(ax, (0.486, 0.245), (ex, 0.224), color=COLORS["orange"], lw=1.1,
              ms=7, connectionstyle="arc3,rad=0.06")
    text(ax, 0.607, 0.184, "12 routed experts · sparse activation", size=6.7,
         color=COLORS["muted"])

    rounded(ax, 0.708, 0.225, 0.025, 0.055, fc=COLORS["green_light"],
            ec=COLORS["green"], lw=1.1, radius=0.012, z=4)
    text(ax, 0.7205, 0.252, "Σ", size=10, weight="bold", color=COLORS["green"])
    for ex in (0.589, 0.704):
        arrow(ax, (ex, 0.287 if ex == 0.589 else 0.224), (0.708, 0.252),
              color=COLORS["green"], lw=1.0, ms=7)
    text(ax, 0.697, 0.304, "$L_{balance}$", size=7.0, weight="bold",
         color=COLORS["purple"])
    arrow(ax, (0.678, 0.296), (0.486, 0.276), color=COLORS["purple"], lw=0.9,
          ms=6, ls="--", connectionstyle="arc3,rad=0.10")

    # Decision-map head.
    arrow(ax, (0.759, 0.594), (0.783, 0.594), color=COLORS["ink"], lw=1.8, ms=11)
    rounded(ax, 0.783, 0.510, 0.120, 0.168, fc=COLORS["orange_light"],
            ec=COLORS["orange"], lw=1.5, radius=0.012, z=4)
    text(ax, 0.843, 0.650, "Decision-map head", size=9.1, weight="bold", color="#A96220")
    text(ax, 0.843, 0.615, "$w=\\sigma(\\mathrm{Conv}_{1\\times1}(\\sum_s F_s))$",
         size=8.3, weight="bold")
    # Small visual decision map.
    grad = np.tile(np.linspace(0.08, 0.95, 160), (34, 1))
    ax.imshow(grad, extent=(0.801, 0.885, 0.570, 0.595), cmap="viridis",
              aspect="auto", zorder=5)
    ax.add_patch(Rectangle((0.801, 0.570), 0.084, 0.025, facecolor="none",
                           edgecolor="#A96220", linewidth=0.8, zorder=6))
    text(ax, 0.843, 0.542, "$F_Y=w\\odot Y_A+(1-w)\\odot Y_B$",
         size=8.0, weight="bold")
    text(ax, 0.843, 0.519, "contrast-preserving convex blend", size=6.6,
         color=COLORS["muted"])

    # Recombine chroma and output cards.
    rounded(ax, 0.801, 0.411, 0.084, 0.061, fc=COLORS["blue_light"],
            ec=COLORS["blue"], lw=1.15, radius=0.009, z=4)
    text(ax, 0.843, 0.449, "color tasks", size=6.6, weight="bold", color=COLORS["blue"])
    text(ax, 0.843, 0.428, "$F_Y+$ source CbCr", size=7.4, weight="bold")
    arrow(ax, (0.843, 0.510), (0.843, 0.472), color=COLORS["blue"], lw=1.1, ms=8)

    output_card(ax, 0.699, "IR–VIS", examples["ir_f"], COLORS["blue"], grayscale=True)
    output_card(ax, 0.489, "Medical", examples["med_f"], COLORS["coral"], grayscale=False)
    output_card(ax, 0.279, "GFP–PC", examples["gfp_f"], COLORS["green"], grayscale=False)
    for oy in (0.786, 0.576, 0.366):
        arrow(ax, (0.903, 0.594), (0.916, oy), color=COLORS["muted"], lw=1.1,
              ms=7, connectionstyle="arc3,rad=0.08")

    # Self-supervised maxfuse objective, connected back to the model.
    rounded(ax, 0.785, 0.176, 0.188, 0.178, fc=COLORS["coral_light"],
            ec=COLORS["coral"], lw=1.25, radius=0.012, z=3)
    text(ax, 0.879, 0.329, "Unsupervised maxfuse objective", size=8.7,
         weight="bold", color=COLORS["coral"])
    chip(ax, 0.797, 0.277, 0.078, 0.035, "SSIM to max", COLORS["white"], COLORS["coral"], size=6.9)
    chip(ax, 0.882, 0.277, 0.078, 0.035, "max intensity", COLORS["white"], COLORS["coral"], size=6.9)
    chip(ax, 0.797, 0.229, 0.078, 0.035, "joint gradient", COLORS["white"], COLORS["coral"], size=6.9)
    chip(ax, 0.882, 0.229, 0.078, 0.035, "RMI content", COLORS["white"], COLORS["coral"], size=6.9)
    text(ax, 0.879, 0.196, "$\\mathcal{L}=\\mathcal{L}_{str}+\\mathcal{L}_{content}+0.01\\,\\mathcal{L}_{balance}$",
         size=7.6, weight="bold")
    arrow(ax, (0.903, 0.510), (0.934, 0.354), color=COLORS["coral"], lw=1.0,
          ms=7, ls="--", connectionstyle="arc3,rad=-0.12")
    arrow(ax, (0.785, 0.265), (0.742, 0.265), color=COLORS["coral"], lw=1.0,
          ms=7, ls="--")
    text(ax, 0.765, 0.281, "training", size=6.4, color=COLORS["coral"])

    # Compact infrastructure strip (implementation optimisation, not a new inference path).
    rounded(ax, 0.010, 0.028, 0.980, 0.091, fc=COLORS["yellow_light"],
            ec=COLORS["yellow"], lw=1.15, radius=0.014, z=1)
    text(ax, 0.027, 0.084, "Efficient training", size=9.0, weight="bold",
         color="#9A7420", ha="left")
    text(ax, 0.027, 0.055, "same model semantics", size=6.7, color=COLORS["muted"], ha="left")
    chips = [
        (0.150, "grouped-capacity\nMoE dispatch"),
        (0.345, "fused SDPA\nwindow attention"),
        (0.540, "torch.compile\ngraph fusion"),
        (0.735, "DDP overlap +\nrank balancing"),
    ]
    for x, lab in chips:
        chip(ax, x, 0.047, 0.165, 0.052, lab, COLORS["white"], "#A47B20", size=7.4)
        if x != chips[-1][0]:
            arrow(ax, (x + 0.168, 0.073), (x + 0.188, 0.073), color="#B58D34", lw=1.0, ms=7)

    # Small innovation markers make correspondence with the ablation section explicit.
    innovations = [
        (0.646, 0.820, "I1", COLORS["purple"]),
        (0.891, 0.664, "I2", COLORS["orange"]),
        (0.566, 0.820, "I3", COLORS["orange"]),
        (0.960, 0.339, "I4", COLORS["coral"]),
        (0.274, 0.377, "I5", COLORS["purple"]),
    ]
    for x, y, lab, c in innovations:
        rounded(ax, x, y, 0.026, 0.026, fc=c, ec=c, lw=0.8, radius=0.010, z=8)
        text(ax, x + 0.013, y + 0.013, lab, size=6.6, weight="bold", color="white", z=9)

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "fig_u_moe_fusion_framework"
    svg_path = base.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.04)
    # Matplotlib emits trailing spaces in multiline SVG path data. Normalise the
    # generated source so repository whitespace checks remain clean.
    svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text("\n".join(line.rstrip() for line in svg_lines) + "\n", encoding="utf-8")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return base


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--code-root",
        type=Path,
        default=Path("/ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper"),
        help="Graduation-Paper repository (for GFP-PC source examples)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/ytech_m2v4_hdd/lizhongyin/data"),
        help="Dataset root; sibling fusion_bench is used for result examples",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "Materials/figs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base = build_figure(args.code_root, args.data_root, args.output_dir)
    print(f"wrote {base}.svg/.pdf/.png")
