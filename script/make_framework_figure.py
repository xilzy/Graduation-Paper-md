#!/usr/bin/env python3
"""Generate the U-MoE-Fusion framework figure in PDF and PNG."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path


def _bootstrap_mpl_cache() -> None:
    """Use bundled fonts without recursively scanning shared storage."""
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
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image, ImageOps


C = {
    "ink": "#17324D",
    "muted": "#5D7487",
    "line": "#9BB0C0",
    "white": "#FFFFFF",
    "panel_a": "#F7FBFD",
    "panel_b": "#FAF8FD",
    "panel_c": "#FFFBF6",
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
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
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


def txt(ax, x, y, value, size=9, weight="normal", color=None, ha="center",
        va="center", z=7, linespacing=1.12, style="normal"):
    return ax.text(
        x,
        y,
        value,
        fontsize=size,
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


def arrow(ax, start, end, color=None, lw=1.4, ms=9, z=3):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color or C["ink"],
        shrinkA=0,
        shrinkB=0,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def routed_arrow(ax, points, color=None, lw=1.4, ms=9, z=3):
    if len(points) < 2:
        raise ValueError("routed_arrow needs at least two points")
    if len(points) > 2:
        line(ax, points[:-1], color=color, lw=lw, z=z)
    return arrow(ax, points[-2], points[-1], color=color, lw=lw, ms=ms, z=z)


def load_thumb(path: Path, size=(240, 180), grayscale=False):
    if not path.is_file():
        arr = np.full((size[1], size[0], 3), 241, dtype=np.uint8)
        arr[::12, :, :] = 225
        arr[:, ::12, :] = 225
        return arr
    with Image.open(path) as image:
        image = image.convert("L" if grayscale else "RGB")
        image = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)
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
            linewidth=1.1,
            zorder=z + 1,
        )
    )


def task_card(ax, y, title, source_a, source_b, label_a, label_b, accent, gray_b=True):
    x, w, h = 0.018, 0.171, 0.160
    rounded(ax, x, y, w, h, fc=C["white"], ec=accent, lw=1.4, radius=0.010)
    txt(ax, x + 0.010, y + h - 0.018, title, size=8.9, weight="bold",
        color=accent, ha="left")
    image_y = y + 0.037
    image_w, image_h, gap = 0.069, 0.088, 0.009
    image_x1 = x + 0.012
    image_x2 = image_x1 + image_w + gap
    image_box(ax, source_a, image_x1, image_y, image_w, image_h, accent)
    image_box(ax, source_b, image_x2, image_y, image_w, image_h, accent, grayscale=gray_b)
    txt(ax, image_x1 + image_w / 2, y + 0.020, label_a, size=6.8, weight="bold")
    txt(ax, image_x2 + image_w / 2, y + 0.020, label_b, size=6.8, weight="bold")
    return x + w, y + h / 2


def output_card(ax, y, title, image, accent, grayscale=False):
    x, w, h = 0.908, 0.073, 0.150
    rounded(ax, x, y, w, h, fc=C["white"], ec=accent, lw=1.35, radius=0.009)
    txt(ax, x + w / 2, y + h - 0.017, title, size=7.2, weight="bold", color=accent)
    image_box(ax, image, x + 0.008, y + 0.023, w - 0.016, h - 0.052,
              accent, grayscale=grayscale)
    return x, y + h / 2


def pill(ax, x, y, w, h, label, fc, ec, size=7.4, weight="bold"):
    rounded(ax, x, y, w, h, fc=fc, ec=ec, lw=1.0, radius=0.008, z=5)
    txt(ax, x + w / 2, y + h / 2, label, size=size, weight=weight, color=ec, z=7)


def build_figure(code_root: Path, data_root: Path, output_dir: Path):
    setup_style()
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
    txt(ax, 0.105, 0.932, "A  Unified multi-task input", size=10.5, weight="bold",
        color=C["blue"])
    txt(ax, 0.523, 0.932, "B  Multi-scale U-MoE backbone", size=10.5, weight="bold",
        color=C["purple"])
    txt(ax, 0.867, 0.932, "C  Decision-map fusion and outputs", size=10.5,
        weight="bold", color=C["orange"])

    rounded(ax, 0.010, 0.150, 0.285, 0.750, fc=C["panel_a"], ec="#CCDAE4",
            lw=1.1, radius=0.014, z=0)
    rounded(ax, 0.305, 0.150, 0.437, 0.750, fc=C["panel_b"], ec="#D8CCE9",
            lw=1.1, radius=0.014, z=0)
    rounded(ax, 0.752, 0.150, 0.238, 0.750, fc=C["panel_c"], ec="#E8D4BD",
            lw=1.1, radius=0.014, z=0)

    source_ports = [
        task_card(ax, 0.710, "IR–VIS", examples["ir_a"], examples["ir_b"],
                  "Visible", "Infrared", C["blue"], gray_b=True),
        task_card(ax, 0.510, "Medical", examples["med_a"], examples["med_b"],
                  "PET/SPECT", "MRI", C["coral"], gray_b=True),
        task_card(ax, 0.310, "Microscopy", examples["gfp_a"], examples["gfp_b"],
                  "GFP", "Phase contrast", C["green"], gray_b=True),
    ]

    pre_x, pre_y, pre_w, pre_h = 0.207, 0.430, 0.076, 0.370
    rounded(ax, pre_x, pre_y, pre_w, pre_h, fc=C["blue_light"], ec=C["blue"],
            lw=1.4, radius=0.011)
    txt(ax, pre_x + pre_w / 2, pre_y + pre_h - 0.036, "Unified\npreprocessing",
        size=8.8, weight="bold", color=C["blue"])
    txt(ax, pre_x + pre_w / 2, pre_y + 0.270, "RGB → YCbCr", size=7.7, weight="bold")
    txt(ax, pre_x + pre_w / 2, pre_y + 0.218, "Fuse luminance Y\nkeep source CbCr", size=7.1)
    line(ax, [(pre_x + 0.010, pre_y + 0.183), (pre_x + pre_w - 0.010, pre_y + 0.183)],
         color="#A8CBE2", lw=1.0, z=5)
    txt(ax, pre_x + pre_w / 2, pre_y + 0.143, "X = [Y_A, Y_B]\n2-channel input",
        size=7.5, weight="bold")
    txt(ax, pre_x + pre_w / 2, pre_y + 0.082, "170×170 crops", size=7.2)
    txt(ax, pre_x + pre_w / 2, pre_y + 0.040, "balanced task quota", size=7.2)

    routed_arrow(ax, [source_ports[0], (0.198, source_ports[0][1]), (0.198, 0.742),
                      (pre_x, 0.742)], color=C["muted"], lw=1.15, ms=7)
    routed_arrow(ax, [source_ports[1], (0.198, source_ports[1][1]), (0.198, 0.615),
                      (pre_x, 0.615)], color=C["muted"], lw=1.15, ms=7)
    routed_arrow(ax, [source_ports[2], (0.198, source_ports[2][1]), (0.198, 0.480),
                      (pre_x, 0.480)], color=C["muted"], lw=1.15, ms=7)

    rounded(ax, 0.209, 0.220, 0.072, 0.068, fc=C["purple_light"], ec=C["purple"],
            lw=1.2, radius=0.010)
    txt(ax, 0.245, 0.262, "Task ID t", size=8.0, weight="bold", color=C["purple"])
    txt(ax, 0.245, 0.238, "embedding e_t", size=7.0)

    stem_x, stem_y, stem_w, stem_h = 0.322, 0.570, 0.072, 0.095
    routed_arrow(ax, [(pre_x + pre_w, 0.615), (0.307, 0.615), (0.307, 0.618),
                      (stem_x, 0.618)], lw=1.7, ms=10)
    rounded(ax, stem_x, stem_y, stem_w, stem_h, fc=C["teal_light"], ec=C["teal"],
            lw=1.45, radius=0.010)
    txt(ax, stem_x + stem_w / 2, stem_y + 0.059, "3×3 Conv stem", size=8.2,
        weight="bold", color="#247E75")
    txt(ax, stem_x + stem_w / 2, stem_y + 0.029, "+ task bias", size=7.4,
        weight="bold", color=C["purple"])

    pill(ax, 0.420, 0.850, 0.205, 0.030,
         "single shared parameter set across all three tasks",
         C["purple_light"], C["purple"], size=7.2)

    split_x = 0.410
    branch_y = [0.770, 0.610, 0.450]
    line(ax, [(stem_x + stem_w, 0.618), (split_x, 0.618)], lw=1.35)
    line(ax, [(split_x, branch_y[-1]), (split_x, branch_y[0])], lw=1.35)

    for index, center_y in enumerate(branch_y, start=1):
        arrow(ax, (split_x, center_y), (0.430, center_y), lw=1.25, ms=7)
        rounded(ax, 0.430, center_y - 0.034, 0.052, 0.068,
                fc=C["teal_light"], ec=C["teal"], lw=1.2, radius=0.007)
        txt(ax, 0.456, center_y + 0.008, f"ACM ×{index}", size=7.5,
            weight="bold", color="#247E75")
        txt(ax, 0.456, center_y - 0.016,
            ("shallow", "mid-level", "deep")[index - 1], size=6.2, color=C["muted"])
        arrow(ax, (0.482, center_y), (0.492, center_y), lw=1.2, ms=7)

        rounded(ax, 0.492, center_y - 0.050, 0.164, 0.100,
                fc=C["white"], ec=C["purple"], lw=1.3, radius=0.009)
        txt(ax, 0.646, center_y + 0.040, "×4 blocks", size=6.5, weight="bold",
            color=C["purple"], ha="right")
        rounded(ax, 0.499, center_y - 0.027, 0.069, 0.054,
                fc=C["orange_light"], ec=C["orange"], lw=1.0, radius=0.006, z=5)
        txt(ax, 0.5335, center_y + 0.010, "I3 · 8×8 window", size=6.5,
            weight="bold", color="#A96220")
        txt(ax, 0.5335, center_y - 0.011, "attention", size=6.6,
            weight="bold", color="#A96220")
        arrow(ax, (0.568, center_y), (0.577, center_y), color=C["purple"], lw=1.0, ms=6)
        rounded(ax, 0.577, center_y - 0.027, 0.071, 0.054,
                fc=C["purple_light"], ec=C["purple"], lw=1.0, radius=0.006, z=5)
        txt(ax, 0.6125, center_y + 0.010, "I1 · task-cond.", size=6.4,
            weight="bold", color=C["purple"])
        txt(ax, 0.6125, center_y - 0.011, "MoE-FFN", size=6.7,
            weight="bold", color=C["purple"])

        arrow(ax, (0.656, center_y), (0.670, center_y), lw=1.2, ms=7)
        rounded(ax, 0.670, center_y - 0.034, 0.043, 0.068,
                fc=C["teal_light"], ec=C["teal"], lw=1.15, radius=0.007)
        txt(ax, 0.6915, center_y, "ACM", size=7.3, weight="bold", color="#247E75")
        arrow(ax, (0.713, center_y), (0.727, center_y), lw=1.15, ms=7)

    line(ax, [(0.727, branch_y[-1]), (0.727, branch_y[0])], lw=1.25)
    sum_circle = Circle((0.727, branch_y[1]), 0.016, facecolor=C["green_light"],
                        edgecolor=C["green"], linewidth=1.3, zorder=5)
    ax.add_patch(sum_circle)
    txt(ax, 0.727, branch_y[1], "Σ", size=10.5, weight="bold", color=C["green"])

    inset_x, inset_y, inset_w, inset_h = 0.321, 0.170, 0.408, 0.180
    rounded(ax, inset_x, inset_y, inset_w, inset_h, fc="#F5F0FA", ec=C["purple"],
            lw=1.2, radius=0.011, z=2)
    txt(ax, inset_x + 0.012, inset_y + inset_h - 0.022,
        "Zoom-in: I1 + I5 task-conditioned MoE-FFN", size=8.2, weight="bold",
        color=C["purple"], ha="left")

    pill(ax, 0.337, 0.245, 0.052, 0.044, "token h", C["white"], C["ink"], size=7.0)
    pill(ax, 0.337, 0.188, 0.052, 0.040, "e_t", C["purple_light"], C["purple"], size=7.4)
    rounded(ax, 0.418, 0.217, 0.075, 0.075, fc=C["purple_light"], ec=C["purple"],
            lw=1.15, radius=0.008)
    txt(ax, 0.4555, 0.267, "softmax router", size=7.0, weight="bold", color=C["purple"])
    txt(ax, 0.4555, 0.244, "top-2 selection", size=6.6)
    txt(ax, 0.4555, 0.225, "+ L_balance", size=6.3, color=C["muted"])

    arrow(ax, (0.389, 0.267), (0.418, 0.267), lw=1.0, ms=6)
    routed_arrow(ax, [(0.389, 0.208), (0.403, 0.208), (0.403, 0.235),
                      (0.418, 0.235)], color=C["purple"], lw=1.0, ms=6)

    rounded(ax, 0.528, 0.276, 0.108, 0.040, fc=C["teal_light"], ec=C["teal"],
            lw=1.05, radius=0.007)
    txt(ax, 0.582, 0.296, "shared expert · always on", size=6.7, weight="bold",
        color="#247E75")
    routed_arrow(ax, [(0.363, 0.289), (0.363, 0.307), (0.508, 0.307),
                      (0.508, 0.296), (0.528, 0.296)], color=C["teal"], lw=1.0, ms=6)

    rounded(ax, 0.528, 0.197, 0.108, 0.066, fc=C["orange_light"], ec=C["orange"],
            lw=1.05, radius=0.007)
    txt(ax, 0.582, 0.250, "12 routed experts", size=6.7, weight="bold", color="#A96220")
    for x, label in [(0.537, "E1"), (0.561, "E2"), (0.585, "…"), (0.609, "E12")]:
        pill(ax, x, 0.207, 0.020, 0.026, label, C["white"], C["orange"], size=5.7)
    routed_arrow(ax, [(0.493, 0.244), (0.510, 0.244), (0.510, 0.230),
                      (0.528, 0.230)], color=C["orange"], lw=1.0, ms=6)

    sum_detail = Circle((0.691, 0.255), 0.017, facecolor=C["green_light"],
                        edgecolor=C["green"], linewidth=1.2, zorder=5)
    ax.add_patch(sum_detail)
    txt(ax, 0.691, 0.255, "Σ", size=9.5, weight="bold", color=C["green"])
    routed_arrow(ax, [(0.636, 0.296), (0.664, 0.296), (0.664, 0.267),
                      (0.674, 0.267)], color=C["green"], lw=1.0, ms=6)
    routed_arrow(ax, [(0.636, 0.230), (0.664, 0.230), (0.664, 0.243),
                      (0.674, 0.243)], color=C["green"], lw=1.0, ms=6)
    arrow(ax, (0.708, 0.255), (0.718, 0.255), color=C["green"], lw=1.0, ms=6)
    txt(ax, 0.715, 0.276, "output", size=5.8, color=C["muted"])

    routed_arrow(ax, [(0.281, 0.254), (0.299, 0.254), (0.299, 0.208),
                      (0.337, 0.208)], color=C["purple"], lw=1.15, ms=7)

    decision_x, decision_y, decision_w, decision_h = 0.765, 0.560, 0.111, 0.150
    routed_arrow(ax, [(0.743, 0.610), (0.753, 0.610), (0.753, 0.635),
                      (decision_x, 0.635)], lw=1.65, ms=9)
    rounded(ax, decision_x, decision_y, decision_w, decision_h,
            fc=C["orange_light"], ec=C["orange"], lw=1.45, radius=0.010)
    txt(ax, decision_x + decision_w / 2, decision_y + 0.126,
        "I2 · Decision-map head", size=8.0, weight="bold", color="#A96220")
    txt(ax, decision_x + decision_w / 2, decision_y + 0.091,
        "w = sigmoid(Conv1×1(ΣF_s))", size=7.1, weight="bold")
    gradient = np.tile(np.linspace(0.08, 0.95, 160), (30, 1))
    ax.imshow(gradient, extent=(0.783, 0.858, 0.615, 0.635), cmap="viridis",
              aspect="auto", zorder=5)
    ax.add_patch(Rectangle((0.783, 0.615), 0.075, 0.020, facecolor="none",
                           edgecolor="#A96220", linewidth=0.7, zorder=6))
    txt(ax, decision_x + decision_w / 2, decision_y + 0.033,
        "F_Y = w⊙Y_A + (1−w)⊙Y_B", size=7.0, weight="bold")

    rounded(ax, 0.775, 0.445, 0.091, 0.060, fc=C["blue_light"], ec=C["blue"],
            lw=1.1, radius=0.008)
    txt(ax, 0.8205, 0.482, "color tasks", size=6.4, weight="bold", color=C["blue"])
    txt(ax, 0.8205, 0.461, "F_Y + source CbCr", size=6.8, weight="bold")
    arrow(ax, (0.8205, decision_y), (0.8205, 0.505), color=C["blue"], lw=1.0, ms=6)

    output_ports = [
        output_card(ax, 0.715, "IR–VIS", examples["ir_f"], C["blue"], grayscale=True),
        output_card(ax, 0.515, "Medical", examples["med_f"], C["coral"]),
        output_card(ax, 0.315, "GFP–PC", examples["gfp_f"], C["green"]),
    ]
    bus_x = 0.894
    line(ax, [(bus_x, output_ports[-1][1]), (bus_x, output_ports[0][1])], lw=1.2)
    arrow(ax, (decision_x + decision_w, 0.635), (bus_x, 0.635), lw=1.2, ms=7)
    for port_x, port_y in output_ports:
        arrow(ax, (bus_x, port_y), (port_x, port_y), color=C["muted"], lw=1.05, ms=6)

    loss_x, loss_y, loss_w, loss_h = 0.765, 0.185, 0.111, 0.200
    rounded(ax, loss_x, loss_y, loss_w, loss_h, fc=C["coral_light"], ec=C["coral"],
            lw=1.2, radius=0.010)
    txt(ax, loss_x + loss_w / 2, loss_y + loss_h - 0.025,
        "I4 · maxfuse objective", size=7.5, weight="bold", color=C["coral"])
    pill(ax, 0.773, 0.292, 0.046, 0.032, "SSIM→max", C["white"], C["coral"], size=5.9)
    pill(ax, 0.824, 0.292, 0.044, 0.032, "max intensity", C["white"], C["coral"], size=5.5)
    pill(ax, 0.773, 0.248, 0.046, 0.032, "joint gradient", C["white"], C["coral"], size=5.5)
    pill(ax, 0.824, 0.248, 0.044, 0.032, "RMI content", C["white"], C["coral"], size=5.6)
    txt(ax, loss_x + loss_w / 2, loss_y + 0.035,
        "L = L_str + L_content\n+ 0.01 L_balance", size=6.7, weight="bold")
    routed_arrow(ax, [(decision_x, 0.590), (0.758, 0.590), (0.758, 0.350),
                      (loss_x, 0.350)], color=C["coral"], lw=1.0, ms=6)
    txt(ax, 0.760, 0.405, "training only", size=5.8, color=C["coral"],
        ha="left", style="italic")

    rounded(ax, 0.010, 0.020, 0.475, 0.105, fc="#F7F3FB", ec="#D8CCE9",
            lw=1.05, radius=0.012, z=1)
    txt(ax, 0.025, 0.103, "Legend · method innovations", size=8.4, weight="bold",
        color=C["purple"], ha="left")
    legend_items = [
        (0.025, "I1\nMoE-FFN", C["purple_light"], C["purple"]),
        (0.113, "I2\nDecision map", C["orange_light"], C["orange"]),
        (0.201, "I3\nWindow attn", C["orange_light"], C["orange"]),
        (0.289, "I4\nmaxfuse", C["coral_light"], C["coral"]),
        (0.377, "I5\nTask routing", C["purple_light"], C["purple"]),
    ]
    for x, label, fill, edge in legend_items:
        pill(ax, x, 0.040, 0.079, 0.046, label, fill, edge, size=6.6)

    rounded(ax, 0.495, 0.020, 0.495, 0.105, fc=C["yellow_light"], ec=C["yellow"],
            lw=1.05, radius=0.012, z=1)
    txt(ax, 0.510, 0.103, "Efficient training · same inference semantics", size=8.4,
        weight="bold", color="#9A7420", ha="left")
    efficiency = [
        (0.510, "grouped-capacity\nMoE dispatch"),
        (0.628, "fused SDPA\nwindow attention"),
        (0.746, "torch.compile\ngraph fusion"),
        (0.864, "DDP overlap +\nrank balance"),
    ]
    for index, (x, label) in enumerate(efficiency):
        pill(ax, x, 0.040, 0.104, 0.046, label, C["white"], "#A47B20", size=6.5)
        if index < len(efficiency) - 1:
            arrow(ax, (x + 0.106, 0.063), (x + 0.116, 0.063), color="#A47B20",
                  lw=0.9, ms=5)

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
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "Materials/figs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base = build_figure(args.code_root, args.data_root, args.output_dir)
    print(f"wrote {base}.pdf and {base}.png")
