#!/usr/bin/env python3
"""Generate the U-MoE-Fusion framework figure in PDF and PNG."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path


def _bootstrap_mpl_cache() -> None:
    """Use a fixed font cache without recursively scanning shared storage."""
    cache = Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/u_moe_framework_mpl"))
    os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")
    cache.mkdir(parents=True, exist_ok=True)
    version = importlib.metadata.version("matplotlib")
    cache_file = cache / f"fontlist-v{version}.json"
    if cache_file.exists():
        return

    def entry(filename: str, weight: int = 400, style: str = "normal") -> dict:
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
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image


C = {
    "ink": "#243746",
    "muted": "#667987",
    "line": "#9AAAB5",
    "white": "#FFFFFF",
    "panel_a": "#F8FAFC",
    "panel_b": "#FAF9FC",
    "panel_c": "#FFFBF7",
    "pre": "#3D73B9",
    "pre_light": "#E2ECF8",
    "conv": "#2E8B57",
    "conv_light": "#E2F1E8",
    "acm": "#159A96",
    "acm_light": "#DDF3F2",
    "attn": "#5262C6",
    "attn_light": "#E6E9FA",
    "moe": "#8552B3",
    "moe_light": "#EEE4F6",
    "router": "#C04B91",
    "router_light": "#F7E2F0",
    "shared": "#6C9E3D",
    "shared_light": "#E9F2DF",
    "routed": "#C58F1D",
    "routed_light": "#FBF0D2",
    "sum": "#667C8D",
    "sum_light": "#E7EDF1",
    "decision": "#E47B25",
    "decision_light": "#FBE7D6",
    "chroma": "#318FA8",
    "chroma_light": "#E0F0F4",
    "loss": "#D94C4C",
    "loss_light": "#F9E1E1",
    "infra": "#9A7420",
    "infra_light": "#FFF4D1",
    "task": "#536775",
    "task_light": "#EDF1F4",
}


def register_times_new_roman(font_dir: Path) -> str:
    """Register the genuine Microsoft Times New Roman font family."""
    # Font discovery is already complete from the fixed cache. Remove the
    # discovery guard so findfont can select explicitly registered external fonts.
    os.environ.pop("MPL_IGNORE_SYSTEM_FONTS", None)
    names = ["Times.TTF", "Timesbd.TTF", "Timesi.TTF", "Timesbi.TTF"]
    paths = [font_dir / name for name in names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Times New Roman files are required. Missing: " + ", ".join(missing)
        )
    for path in paths:
        font_manager.fontManager.addfont(path)
    family = font_manager.FontProperties(fname=paths[0]).get_name()
    if family != "Times New Roman":
        raise RuntimeError(f"Expected Times New Roman, found {family}")
    return family


def setup_style(font_dir: Path) -> None:
    family = register_times_new_roman(font_dir)
    mpl.rcParams.update(
        {
            "font.family": family,
            "font.serif": [family],
            "font.size": 9,
            "text.color": C["ink"],
            "axes.edgecolor": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def rounded(ax, x, y, w, h, fc="white", ec=None, lw=1.0, radius=0.010, z=4):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.003,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec or C["line"],
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def publication_text_size(size):
    if size >= 15:
        return size
    if size >= 10:
        return size + 1.5
    if size >= 8:
        return size + 2.0
    return max(8.5, size + 2.5)


def txt(ax, x, y, value, size=9, weight="normal", color=None, ha="center",
        va="center", z=7, linespacing=1.10, style="normal"):
    return ax.text(
        x,
        y,
        value,
        fontsize=publication_text_size(size),
        fontweight=weight,
        color=color or C["ink"],
        ha=ha,
        va=va,
        zorder=z,
        linespacing=linespacing,
        fontstyle=style,
    )


def line(ax, points, color=None, lw=1.4, z=2):
    xs, ys = zip(*points)
    ax.plot(
        xs,
        ys,
        color=color or C["ink"],
        linewidth=lw,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=z,
    )


def arrow(ax, start, end, color=None, lw=1.4, ms=9, z=3,
          shrink_a=1.5, shrink_b=1.5):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=max(ms, 9),
        linewidth=lw,
        color=color or C["ink"],
        shrinkA=shrink_a,
        shrinkB=shrink_b,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def routed_arrow(ax, points, color=None, lw=1.4, ms=9, z=3):
    if len(points) < 2:
        raise ValueError("routed_arrow needs at least two points")
    if len(points) > 2:
        line(ax, points[:-1], color=color, lw=lw, z=max(2, z - 3))
        return arrow(ax, points[-2], points[-1], color=color, lw=lw, ms=ms,
                     z=z, shrink_a=0, shrink_b=1.5)
    return arrow(ax, points[-2], points[-1], color=color, lw=lw, ms=ms, z=z)


def load_thumb(path: Path, size=(320, 320), grayscale=False):
    """Downsample while preserving the original aspect ratio without padding."""
    mode = "L" if grayscale else "RGB"
    if not path.is_file():
        return np.full((size[1], size[0]), 245, dtype=np.uint8) if grayscale else np.full(
            (size[1], size[0], 3), 245, dtype=np.uint8
        )
    with Image.open(path) as image:
        image = image.convert(mode)
        image.thumbnail(size, Image.Resampling.LANCZOS)
        return np.asarray(image)


def image_box(ax, path, x, y, w, h, border, grayscale=False, z=5):
    array = load_thumb(Path(path), grayscale=grayscale)
    ax.imshow(
        array,
        extent=(x, x + w, y, y + h),
        aspect="auto",
        zorder=z,
        cmap="gray" if grayscale else None,
        vmin=0 if grayscale else None,
        vmax=255 if grayscale else None,
    )
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor="none",
            edgecolor=border,
            linewidth=1.0,
            zorder=z + 1,
        )
    )


FIGURE_XY_ASPECT = 18.0 / 9.4
THUMBNAIL_WIDTH = 0.055
THUMBNAIL_AREA_Y = 0.020
THUMBNAIL_AREA_HEIGHT = 0.110


def task_card(ax, y, title, source_a, source_b, label_a, label_b,
              frame_aspect, gray_b=True):
    x, w, h = 0.018, 0.171, 0.155
    rounded(ax, x, y, w, h, fc=C["white"], ec=C["task"], lw=1.25, radius=0.009)
    txt(ax, x + 0.010, y + h - 0.015, title, size=8.5, weight="bold",
        color=C["task"], ha="left")
    image_w = THUMBNAIL_WIDTH
    image_h = image_w * FIGURE_XY_ASPECT / frame_aspect
    gap = 0.012
    total_w = image_w * 2 + gap
    image_x1 = x + (w - total_w) / 2
    image_x2 = image_x1 + image_w + gap
    image_y = y + THUMBNAIL_AREA_Y + (THUMBNAIL_AREA_HEIGHT - image_h) / 2
    image_box(ax, source_a, image_x1, image_y, image_w, image_h, C["task"])
    image_box(ax, source_b, image_x2, image_y, image_w, image_h, C["task"], grayscale=gray_b)
    txt(ax, image_x1 + image_w / 2, y + 0.012, label_a, size=6.4, weight="bold")
    txt(ax, image_x2 + image_w / 2, y + 0.012, label_b, size=6.4, weight="bold")
    return x + w, y + h / 2


def output_card(ax, y, title, image, frame_aspect, grayscale=False):
    x, w, h = 0.908, 0.073, 0.150
    rounded(ax, x, y, w, h, fc=C["white"], ec=C["task"], lw=1.25, radius=0.009)
    txt(ax, x + w / 2, y + h - 0.015, title, size=7.1, weight="bold", color=C["task"])
    image_w = THUMBNAIL_WIDTH
    image_h = image_w * FIGURE_XY_ASPECT / frame_aspect
    image_y = y + THUMBNAIL_AREA_Y + (THUMBNAIL_AREA_HEIGHT - image_h) / 2
    image_box(ax, image, x + (w - image_w) / 2, image_y,
              image_w, image_h, C["task"], grayscale=grayscale)
    return x, y + h / 2


def pill(ax, x, y, w, h, label, fc, ec, size=7.2, weight="bold", radius=0.007):
    rounded(ax, x, y, w, h, fc=fc, ec=ec, lw=1.0, radius=radius, z=5)
    txt(ax, x + w / 2, y + h / 2, label, size=size, weight=weight, color=ec, z=7)


def build_figure(code_root: Path, data_root: Path, output_dir: Path, font_dir: Path):
    setup_style(font_dir)
    fig = plt.figure(figsize=(18.0, 9.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

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

    txt(ax, 0.5, 0.974, "U-MoE-Fusion: One Model for Unified Multi-Modal Image Fusion",
        size=18, weight="bold")
    txt(ax, 0.105, 0.932, "A  Unified Multi-Task Input", size=10.5, weight="bold",
        color=C["pre"])
    txt(ax, 0.523, 0.932, "B  Multi-Scale U-MoE Backbone", size=10.5, weight="bold",
        color=C["moe"])
    txt(ax, 0.867, 0.932, "C  Decision-Map Fusion and Outputs", size=10.5,
        weight="bold", color=C["decision"])

    rounded(ax, 0.010, 0.170, 0.285, 0.730, fc=C["panel_a"], ec="#CCD6DE",
            lw=1.0, radius=0.014, z=0)
    rounded(ax, 0.305, 0.170, 0.437, 0.730, fc=C["panel_b"], ec="#D7CDE2",
            lw=1.0, radius=0.014, z=0)
    rounded(ax, 0.752, 0.170, 0.238, 0.730, fc=C["panel_c"], ec="#E6D4C3",
            lw=1.0, radius=0.014, z=0)

    source_ports = [
        task_card(ax, 0.720, "IR-VIS", examples["ir_a"], examples["ir_b"],
                  "Visible", "Infrared", frame_aspect=4 / 3, gray_b=True),
        task_card(ax, 0.525, "Medical", examples["med_a"], examples["med_b"],
                  "PET/SPECT", "MRI", frame_aspect=1.0, gray_b=True),
        task_card(ax, 0.330, "Microscopy", examples["gfp_a"], examples["gfp_b"],
                  "GFP", "Phase Contrast", frame_aspect=1.0, gray_b=True),
    ]

    pre_x, pre_y, pre_w, pre_h = 0.207, 0.370, 0.076, 0.450
    rounded(ax, pre_x, pre_y, pre_w, pre_h, fc=C["pre_light"], ec=C["pre"],
            lw=1.35, radius=0.011)
    txt(ax, pre_x + pre_w / 2, pre_y + pre_h - 0.036, "Unified\nPreprocessing",
        size=8.6, weight="bold", color=C["pre"])
    txt(ax, pre_x + pre_w / 2, pre_y + 0.324, "RGB to YCbCr", size=7.4, weight="bold")
    txt(ax, pre_x + pre_w / 2, pre_y + 0.265, "Fuse Luminance Y\nKeep Source CbCr", size=6.9)
    line(ax, [(pre_x + 0.010, pre_y + 0.220), (pre_x + pre_w - 0.010, pre_y + 0.220)],
         color="#9AB6D7", lw=0.9, z=5)
    txt(ax, pre_x + pre_w / 2, pre_y + 0.178, "X = [Y_A, Y_B]\nTwo-Channel Input",
        size=7.2, weight="bold")
    txt(ax, pre_x + pre_w / 2, pre_y + 0.100, "170 x 170 Crops", size=7.0)
    txt(ax, pre_x + pre_w / 2, pre_y + 0.048, "Balanced Task Quota", size=7.0)

    for source_port in source_ports:
        arrow(ax, source_port, (pre_x, source_port[1]), color=C["task"], lw=1.05, ms=6)

    rounded(ax, 0.207, 0.180, 0.075, 0.050, fc=C["router_light"], ec=C["router"],
            lw=1.15, radius=0.009)
    txt(ax, 0.2445, 0.213, "Task ID t", size=7.6, weight="bold", color=C["router"])
    txt(ax, 0.2445, 0.194, "Embedding e_t", size=6.6)

    stem_x, stem_y, stem_w, stem_h = 0.322, 0.575, 0.072, 0.090
    arrow(ax, (pre_x + pre_w, 0.620), (stem_x, 0.620), lw=1.6, ms=9)
    rounded(ax, stem_x, stem_y, stem_w, stem_h, fc=C["conv_light"], ec=C["conv"],
            lw=1.35, radius=0.010)
    txt(ax, stem_x + stem_w / 2, stem_y + 0.055, "3 x 3 Conv Stem", size=8.0,
        weight="bold", color=C["conv"])
    txt(ax, stem_x + stem_w / 2, stem_y + 0.027, "+ Task Bias", size=7.1,
        weight="bold", color=C["router"])

    pill(ax, 0.420, 0.850, 0.205, 0.030,
         "One Shared Parameter Set Across All Three Tasks",
         C["task_light"], C["task"], size=7.0)

    split_x = 0.410
    branch_y = [0.780, 0.620, 0.460]
    line(ax, [(stem_x + stem_w, 0.620), (split_x, 0.620)], lw=1.3)
    line(ax, [(split_x, branch_y[-1]), (split_x, branch_y[0])], lw=1.3)

    for index, center_y in enumerate(branch_y, start=1):
        arrow(ax, (split_x, center_y), (0.430, center_y), lw=1.2, ms=7,
              shrink_a=0)
        rounded(ax, 0.430, center_y - 0.033, 0.052, 0.066,
                fc=C["acm_light"], ec=C["acm"], lw=1.15, radius=0.007)
        txt(ax, 0.456, center_y + 0.008, f"ACM x{index}", size=7.3,
            weight="bold", color=C["acm"])
        txt(ax, 0.456, center_y - 0.015,
            ("Shallow", "Mid-Level", "Deep")[index - 1], size=6.1, color=C["muted"])
        arrow(ax, (0.482, center_y), (0.492, center_y), lw=1.1, ms=6)

        rounded(ax, 0.492, center_y - 0.048, 0.164, 0.096,
                fc=C["white"], ec=C["task"], lw=1.2, radius=0.009)
        txt(ax, 0.646, center_y + 0.038, "x4 Blocks", size=6.3, weight="bold",
            color=C["task"], ha="right")
        rounded(ax, 0.499, center_y - 0.026, 0.069, 0.052,
                fc=C["attn_light"], ec=C["attn"], lw=1.0, radius=0.006, z=5)
        txt(ax, 0.5335, center_y + 0.009, "I3 · 8 x 8 Window", size=6.3,
            weight="bold", color=C["attn"])
        txt(ax, 0.5335, center_y - 0.010, "Attention", size=6.5,
            weight="bold", color=C["attn"])
        arrow(ax, (0.568, center_y), (0.577, center_y), color=C["task"], lw=0.9, ms=5)
        rounded(ax, 0.577, center_y - 0.026, 0.071, 0.052,
                fc=C["moe_light"], ec=C["moe"], lw=1.0, radius=0.006, z=5)
        txt(ax, 0.6125, center_y + 0.009, "I1 · Task-Cond.", size=6.2,
            weight="bold", color=C["moe"])
        txt(ax, 0.6125, center_y - 0.010, "MoE-FFN", size=6.6,
            weight="bold", color=C["moe"])

        arrow(ax, (0.656, center_y), (0.672, center_y), lw=1.1, ms=7)
        rounded(ax, 0.672, center_y - 0.033, 0.035, 0.066,
                fc=C["acm_light"], ec=C["acm"], lw=1.1, radius=0.007)
        txt(ax, 0.6895, center_y, "ACM", size=6.8, weight="bold", color=C["acm"])
        if index == 2:
            arrow(ax, (0.707, center_y), (0.718, center_y), lw=1.05, ms=7)
        else:
            arrow(ax, (0.707, center_y), (0.729, center_y), lw=1.05, ms=7,
                  shrink_b=0)

    line(ax, [(0.729, branch_y[-1]), (0.729, branch_y[0])], lw=1.15)
    sum_circle = Circle((0.729, branch_y[1]), 0.011, facecolor=C["sum_light"],
                        edgecolor=C["sum"], linewidth=1.2, zorder=5)
    ax.add_patch(sum_circle)
    txt(ax, 0.729, branch_y[1], "SUM", size=5.7, weight="bold", color=C["sum"])

    inset_x, inset_y, inset_w, inset_h = 0.321, 0.180, 0.408, 0.170
    rounded(ax, inset_x, inset_y, inset_w, inset_h, fc="#F8F5FA", ec=C["task"],
            lw=1.1, radius=0.011, z=2)
    txt(ax, inset_x + 0.012, inset_y + inset_h - 0.021,
        "Zoom-In: I1 + I5 Task-Conditioned MoE-FFN", size=8.0, weight="bold",
        color=C["task"], ha="left")

    rounded(ax, 0.335, 0.235, 0.060, 0.075, fc=C["task_light"], ec=C["task"],
            lw=1.0, radius=0.007)
    txt(ax, 0.365, 0.273, "Token h", size=7.0, weight="bold", color=C["task"])

    rounded(ax, 0.420, 0.235, 0.075, 0.055, fc=C["router_light"], ec=C["router"],
            lw=1.05, radius=0.007)
    txt(ax, 0.4575, 0.274, "I5 · Softmax Router", size=6.5, weight="bold", color=C["router"])
    txt(ax, 0.4575, 0.254, "Top-2 + L_balance", size=6.2)
    arrow(ax, (0.395, 0.260), (0.420, 0.260), color=C["router"], lw=0.95, ms=5)

    rounded(ax, 0.420, 0.190, 0.052, 0.030, fc=C["router_light"], ec=C["router"],
            lw=0.95, radius=0.006)
    txt(ax, 0.446, 0.205, "e_t", size=7.0, weight="bold", color=C["router"])
    arrow(ax, (0.446, 0.220), (0.446, 0.235), color=C["router"], lw=0.95, ms=5)
    arrow(ax, (0.282, 0.205), (0.420, 0.205), color=C["router"], lw=1.05, ms=6)

    rounded(ax, 0.525, 0.288, 0.115, 0.034, fc=C["shared_light"], ec=C["shared"],
            lw=1.0, radius=0.006)
    txt(ax, 0.5825, 0.305, "Shared Expert · Always On", size=6.3,
        weight="bold", color=C["shared"])
    arrow(ax, (0.395, 0.305), (0.525, 0.305), color=C["shared"], lw=0.95, ms=5)

    rounded(ax, 0.525, 0.233, 0.115, 0.047, fc=C["routed_light"], ec=C["routed"],
            lw=1.0, radius=0.006)
    txt(ax, 0.5825, 0.269, "12 Routed Experts", size=6.1,
        weight="bold", color=C["routed"])
    for x, label in [(0.534, "E1"), (0.558, "E2"), (0.582, "..."), (0.606, "E12")]:
        pill(ax, x, 0.239, 0.020, 0.022, label, C["white"], C["routed"], size=5.2)
    arrow(ax, (0.495, 0.258), (0.525, 0.258), color=C["routed"], lw=0.95, ms=5)

    zoom_sum = Circle((0.695, 0.278), 0.011, facecolor=C["sum_light"],
                      edgecolor=C["sum"], linewidth=1.2, zorder=5)
    ax.add_patch(zoom_sum)
    txt(ax, 0.695, 0.278, "SUM", size=5.7, weight="bold", color=C["sum"])
    arrow(ax, (0.640, 0.305), (0.686, 0.284), color=C["shared"], lw=0.9, ms=7)
    arrow(ax, (0.640, 0.258), (0.686, 0.272), color=C["routed"], lw=0.9, ms=7)
    arrow(ax, (0.706, 0.278), (0.722, 0.278), color=C["sum"], lw=0.9, ms=7)
    txt(ax, 0.718, 0.297, "Output", size=5.5, color=C["muted"])

    decision_x, decision_y, decision_w, decision_h = 0.765, 0.545, 0.111, 0.150
    arrow(ax, (0.740, 0.620), (decision_x, 0.620), lw=1.5, ms=8)
    rounded(ax, decision_x, decision_y, decision_w, decision_h,
            fc=C["decision_light"], ec=C["decision"], lw=1.35, radius=0.010)
    txt(ax, decision_x + decision_w / 2, decision_y + 0.126,
        "I2 · Decision-Map Head", size=7.7, weight="bold", color=C["decision"])
    txt(ax, decision_x + decision_w / 2, decision_y + 0.091,
        "w = sigmoid(Conv1x1(SUM F_s))", size=6.9, weight="bold")
    gradient = np.tile(np.linspace(0.08, 0.95, 160), (30, 1))
    ax.imshow(gradient, extent=(0.783, 0.858, 0.600, 0.620), cmap="viridis",
              aspect="auto", zorder=5)
    ax.add_patch(Rectangle((0.783, 0.600), 0.075, 0.020, facecolor="none",
                           edgecolor=C["decision"], linewidth=0.7, zorder=6))
    txt(ax, decision_x + decision_w / 2, decision_y + 0.033,
        "F_Y = w * Y_A + (1-w) * Y_B", size=6.8, weight="bold")

    rounded(ax, 0.775, 0.435, 0.091, 0.060, fc=C["chroma_light"], ec=C["chroma"],
            lw=1.05, radius=0.008)
    txt(ax, 0.8205, 0.472, "Color Tasks", size=6.3, weight="bold", color=C["chroma"])
    txt(ax, 0.8205, 0.451, "F_Y + Source CbCr", size=6.6, weight="bold")
    arrow(ax, (0.8205, decision_y), (0.8205, 0.495), color=C["chroma"], lw=0.95, ms=5)

    output_ports = [
        output_card(ax, 0.715, "IR-VIS", examples["ir_f"], frame_aspect=4 / 3, grayscale=True),
        output_card(ax, 0.515, "Medical", examples["med_f"], frame_aspect=1.0),
        output_card(ax, 0.315, "GFP-PC", examples["gfp_f"], frame_aspect=1.0),
    ]
    bus_x = 0.894
    line(ax, [(bus_x, output_ports[-1][1]), (bus_x, output_ports[0][1])], lw=1.1)
    arrow(ax, (decision_x + decision_w, 0.620), (bus_x, 0.620), lw=1.1,
          ms=6, shrink_b=0)
    for port_x, port_y in output_ports:
        arrow(ax, (bus_x, port_y), (port_x, port_y), color=C["task"],
              lw=1.0, ms=5, shrink_a=0)

    loss_x, loss_y, loss_w, loss_h = 0.765, 0.195, 0.111, 0.190
    rounded(ax, loss_x, loss_y, loss_w, loss_h, fc=C["loss_light"], ec=C["loss"],
            lw=1.15, radius=0.010)
    txt(ax, loss_x + loss_w / 2, loss_y + loss_h - 0.025,
        "I4 · Maxfuse Objective", size=7.3, weight="bold", color=C["loss"])
    pill(ax, 0.773, 0.290, 0.046, 0.031, "SSIM to Max", C["white"], C["loss"], size=5.6)
    pill(ax, 0.824, 0.290, 0.044, 0.031, "Max Intensity", C["white"], C["loss"], size=5.3)
    pill(ax, 0.773, 0.248, 0.046, 0.031, "Joint Gradient", C["white"], C["loss"], size=5.3)
    pill(ax, 0.824, 0.248, 0.044, 0.031, "RMI Content", C["white"], C["loss"], size=5.4)
    txt(ax, loss_x + loss_w / 2, loss_y + 0.032,
        "L = L_str + L_content\n+ 0.01 L_balance", size=6.5, weight="bold")
    routed_arrow(ax, [(decision_x, 0.575), (0.758, 0.575), (0.758, 0.350),
                      (loss_x, 0.350)], color=C["loss"], lw=0.9, ms=5)
    txt(ax, 0.760, 0.402, "Training Only", size=5.7, color=C["loss"],
        ha="left", style="italic")

    rounded(ax, 0.010, 0.015, 0.720, 0.140, fc="#F8F9FA", ec="#CBD3D9",
            lw=1.0, radius=0.012, z=1)
    txt(ax, 0.025, 0.137, "Complete Module Legend", size=8.4, weight="bold",
        color=C["task"], ha="left")
    legend_items = [
        (0.025, 0.088, "Preprocess", C["pre_light"], C["pre"]),
        (0.124, 0.088, "Conv Stem", C["conv_light"], C["conv"]),
        (0.223, 0.088, "ACM", C["acm_light"], C["acm"]),
        (0.322, 0.088, "I3 Window Attn", C["attn_light"], C["attn"]),
        (0.421, 0.088, "I1 MoE-FFN", C["moe_light"], C["moe"]),
        (0.520, 0.088, "I5 Router", C["router_light"], C["router"]),
        (0.619, 0.088, "Shared Expert", C["shared_light"], C["shared"]),
        (0.025, 0.040, "Routed Experts", C["routed_light"], C["routed"]),
        (0.124, 0.040, "Feature Sum", C["sum_light"], C["sum"]),
        (0.223, 0.040, "I2 Decision", C["decision_light"], C["decision"]),
        (0.322, 0.040, "Chroma Rebuild", C["chroma_light"], C["chroma"]),
        (0.421, 0.040, "I4 Maxfuse", C["loss_light"], C["loss"]),
        (0.520, 0.040, "Task I/O", C["task_light"], C["task"]),
        (0.619, 0.040, "Infra Optim.", C["infra_light"], C["infra"]),
    ]
    for x, y, label, fill, edge in legend_items:
        pill(ax, x, y, 0.093, 0.034, label, fill, edge, size=5.9, radius=0.006)

    rounded(ax, 0.740, 0.015, 0.250, 0.140, fc=C["infra_light"], ec=C["infra"],
            lw=1.0, radius=0.012, z=1)
    txt(ax, 0.755, 0.137, "Efficient Training", size=8.4, weight="bold",
        color=C["infra"], ha="left")
    efficiency = [
        (0.755, 0.088, "Grouped-Capacity\nMoE Dispatch"),
        (0.870, 0.088, "Fused SDPA\nWindow Attention"),
        (0.755, 0.040, "torch.compile\nGraph Fusion"),
        (0.870, 0.040, "DDP Overlap +\nRank Balance"),
    ]
    for x, y, label in efficiency:
        pill(ax, x, y, 0.105, 0.034, label, C["white"], C["infra"], size=5.7)

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "fig_u_moe_fusion_framework"
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
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/ytech_m2v4_hdd/lizhongyin/data"),
    )
    parser.add_argument(
        "--font-dir",
        type=Path,
        default=Path("/ytech_m2v4_hdd/lizhongyin/.cache/fonts/times-new-roman"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "Materials/figs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base = build_figure(args.code_root, args.data_root, args.output_dir, args.font_dir)
    print(f"wrote {base}.pdf and {base}.png with Times New Roman")
