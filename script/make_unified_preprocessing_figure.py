#!/usr/bin/env python3
"""Generate the detailed U-MoE-Fusion unified preprocessing diagram."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path


def _bootstrap_mpl_cache() -> None:
    """Use a fixed font cache without recursively scanning shared storage."""
    cache = Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/u_moe_preprocess_detail_mpl"))
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
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


C = {
    "ink": "#263746",
    "muted": "#657786",
    "line": "#93A2AD",
    "white": "#FFFFFF",
    "main_bg": "#F8FAFC",
    "main_edge": "#4D718F",
    "task": "#536775",
    "task_light": "#EDF1F4",
    "pair": "#6B6FA6",
    "pair_light": "#ECECF7",
    "luma": "#C08A18",
    "luma_light": "#FBF0D1",
    "chroma": "#258DA3",
    "chroma_light": "#DFF1F4",
    "align": "#188D84",
    "align_light": "#DDF3EF",
    "train": "#5A9144",
    "train_light": "#E7F1E2",
    "infer": "#3E70B5",
    "infer_light": "#E2ECF8",
    "contract": "#8552B3",
    "contract_light": "#EEE4F6",
    "network": "#E17A27",
    "network_light": "#FBE7D5",
    "red": "#CF4C55",
    "green": "#4D9B63",
    "blue": "#467AC7",
    "gray": "#7D8790",
}


def register_times_new_roman(font_dir: Path) -> str:
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


def rounded(
    ax,
    x,
    y,
    w,
    h,
    *,
    fc=C["white"],
    ec=C["line"],
    lw=1.0,
    radius=0.010,
    linestyle="solid",
    z=4,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.003,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=linestyle,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def group_box(ax, x, y, w, h, *, fc, ec, lw=1.35, radius=0.016, z=1):
    return rounded(
        ax,
        x,
        y,
        w,
        h,
        fc=fc,
        ec=ec,
        lw=lw,
        radius=radius,
        linestyle=(0, (1.5, 1.5)),
        z=z,
    )


def publication_text_size(size):
    if size >= 15:
        return size
    if size >= 10:
        return size + 1.5
    if size >= 8:
        return size + 2.0
    return max(8.5, size + 2.5)


def text(
    ax,
    x,
    y,
    value,
    *,
    size=9,
    weight="normal",
    color=None,
    ha="center",
    va="center",
    style="normal",
    linespacing=1.08,
    bbox=None,
    z=7,
):
    return ax.text(
        x,
        y,
        value,
        fontsize=publication_text_size(size),
        fontweight=weight,
        color=color or C["ink"],
        ha=ha,
        va=va,
        fontstyle=style,
        linespacing=linespacing,
        bbox=bbox,
        zorder=z,
    )


def line(ax, points, *, color=None, lw=1.2, linestyle="solid", z=2):
    xs, ys = zip(*points)
    ax.plot(
        xs,
        ys,
        color=color or C["ink"],
        linewidth=lw,
        linestyle=linestyle,
        solid_capstyle="round",
        solid_joinstyle="round",
        dash_capstyle="round",
        zorder=z,
    )


def arrow(
    ax,
    start,
    end,
    *,
    color=None,
    lw=1.25,
    ms=10,
    linestyle="solid",
    shrink_a=2.0,
    shrink_b=2.0,
    z=3,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        linestyle=linestyle,
        color=color or C["ink"],
        shrinkA=shrink_a,
        shrinkB=shrink_b,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def routed_arrow(
    ax,
    points,
    *,
    color=None,
    lw=1.25,
    ms=10,
    linestyle="solid",
    shrink_b=2.0,
    z=3,
):
    if len(points) < 2:
        raise ValueError("routed_arrow needs at least two points")
    if len(points) > 2:
        line(
            ax,
            points[:-1],
            color=color,
            lw=lw,
            linestyle=linestyle,
            z=max(2, z - 1),
        )
        return arrow(
            ax,
            points[-2],
            points[-1],
            color=color,
            lw=lw,
            ms=ms,
            linestyle=linestyle,
            shrink_a=0,
            shrink_b=shrink_b,
            z=z,
        )
    return arrow(
        ax,
        points[0],
        points[1],
        color=color,
        lw=lw,
        ms=ms,
        linestyle=linestyle,
        shrink_b=shrink_b,
        z=z,
    )


def module_box(
    ax,
    x,
    y,
    w,
    h,
    label,
    *,
    fc,
    ec,
    size=8.0,
    weight="bold",
    linestyle="solid",
    radius=0.008,
):
    rounded(
        ax,
        x,
        y,
        w,
        h,
        fc=fc,
        ec=ec,
        lw=1.1,
        radius=radius,
        linestyle=linestyle,
        z=4,
    )
    text(ax, x + w / 2, y + h / 2, label, size=size, weight=weight, color=ec)


def rgb_icon(ax, x, y, w=0.035, h=0.045):
    offsets = [(0.000, 0.000, C["red"], "R"), (0.008, 0.006, C["green"], "G"),
               (0.016, 0.012, C["blue"], "B")]
    for dx, dy, color, label in offsets:
        ax.add_patch(
            Rectangle(
                (x + dx, y + dy),
                w,
                h,
                facecolor=color,
                edgecolor=C["white"],
                linewidth=0.8,
                alpha=0.78,
                zorder=5,
            )
        )
        text(
            ax,
            x + dx + w / 2,
            y + dy + h / 2,
            label,
            size=5.8,
            weight="bold",
            color=C["white"],
            z=6,
        )


def gray_icon(ax, x, y, w=0.050, h=0.055):
    shades = ["#D9DEE2", "#AEB7BE", "#7D8790"]
    stripe = w / 3
    for index, shade in enumerate(shades):
        ax.add_patch(
            Rectangle(
                (x + index * stripe, y),
                stripe,
                h,
                facecolor=shade,
                edgecolor=C["white"],
                linewidth=0.6,
                zorder=5,
            )
        )
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            facecolor="none",
            edgecolor=C["gray"],
            linewidth=0.8,
            zorder=6,
        )
    )


def channel_planes(ax, x, y, w=0.044, h=0.054):
    for index, (fc, ec, label) in enumerate(
        [(C["luma_light"], C["luma"], "Y_A"), ("#F4E5BB", "#A86E12", "Y_B")]
    ):
        dx, dy = index * 0.012, index * 0.010
        rounded(ax, x + dx, y + dy, w, h, fc=fc, ec=ec, lw=0.9, radius=0.004, z=5)
        text(
            ax,
            x + dx + w / 2,
            y + dy + h / 2,
            label,
            size=5.8,
            weight="bold",
            color=ec,
            z=7,
        )


def task_card(ax, x, y, title, color_label, gray_label, task_id):
    w, h = 0.165, 0.128
    rounded(ax, x, y, w, h, fc=C["white"], ec=C["task"], lw=1.05, radius=0.009, z=4)
    text(
        ax,
        x + 0.009,
        y + h - 0.017,
        title,
        size=7.7,
        weight="bold",
        color=C["task"],
        ha="left",
    )
    rounded(
        ax,
        x + w - 0.040,
        y + h - 0.030,
        0.030,
        0.020,
        fc=C["pair_light"],
        ec=C["pair"],
        lw=0.8,
        radius=0.004,
        z=5,
    )
    text(
        ax,
        x + w - 0.025,
        y + h - 0.020,
        f"t={task_id}",
        size=5.7,
        weight="bold",
        color=C["pair"],
    )
    rgb_icon(ax, x + 0.020, y + 0.035)
    gray_icon(ax, x + 0.096, y + 0.041)
    text(ax, x + 0.046, y + 0.018, color_label, size=5.8, weight="bold")
    text(ax, x + 0.121, y + 0.018, gray_label, size=5.8, weight="bold")
    return x + w, y + h / 2


def legend_item(ax, x, y, w, label, fc, ec):
    rounded(ax, x, y, w, 0.034, fc=fc, ec=ec, lw=0.95, radius=0.006, z=4)
    text(ax, x + w / 2, y + 0.017, label, size=6.6, weight="bold", color=ec)


def build_figure(output_dir: Path, font_dir: Path) -> Path:
    setup_style(font_dir)
    fig = plt.figure(figsize=(18.0, 8.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    text(
        ax,
        0.5,
        0.968,
        "Unified Multi-Task Luminance Preprocessing in U-MoE-Fusion",
        size=18,
        weight="bold",
    )

    group_box(
        ax,
        0.012,
        0.130,
        0.976,
        0.800,
        fc=C["main_bg"],
        ec=C["main_edge"],
        lw=1.7,
        radius=0.025,
        z=0,
    )

    panels = [
        (0.027, 0.178, 0.200, 0.690, "#FCFDFE", C["task"], "Heterogeneous Paired Inputs"),
        (0.245, 0.178, 0.260, 0.690, "#FFFCF5", C["luma"], "Shared Luminance Contract"),
        (0.523, 0.178, 0.220, 0.690, "#F8FCFB", C["align"], "Alignment and Sampling"),
        (0.761, 0.178, 0.212, 0.690, "#FCFAFD", C["contract"], "Unified Model Contract"),
    ]
    for x, y, w, h, fc, ec, title in panels:
        group_box(ax, x, y, w, h, fc=fc, ec=ec, lw=1.25, radius=0.014, z=1)
        text(ax, x + w / 2, y + h - 0.026, title, size=8.7, weight="bold", color=ec)

    task_ports = [
        task_card(ax, 0.044, 0.650, "IR-VIS", "Visible RGB", "Infrared", 0),
        task_card(ax, 0.044, 0.470, "Medical", "PET / SPECT", "MRI", 1),
        task_card(ax, 0.044, 0.290, "Microscopy", "GFP", "Phase Contrast", 2),
    ]

    module_box(
        ax,
        0.050,
        0.208,
        0.155,
        0.052,
        "Folder or suffix pairing\nmatch by shared stem",
        fc=C["pair_light"],
        ec=C["pair"],
        size=6.6,
    )

    input_bus_x = 0.236
    line(
        ax,
        [(input_bus_x, task_ports[-1][1]), (input_bus_x, 0.7675)],
        color=C["task"],
        lw=1.1,
        z=2,
    )
    for port_x, port_y in task_ports:
        arrow(
            ax,
            (port_x, port_y),
            (input_bus_x, port_y),
            color=C["task"],
            lw=1.0,
            ms=9,
            shrink_b=0,
        )

    module_box(
        ax,
        0.263,
        0.735,
        0.225,
        0.065,
        "Pair by stem and preserve source order\nA = color source, B = grayscale source",
        fc=C["pair_light"],
        ec=C["pair"],
        size=7.1,
    )
    arrow(
        ax,
        (input_bus_x, 0.7675),
        (0.263, 0.7675),
        color=C["pair"],
        lw=1.15,
        shrink_a=0,
    )

    module_box(
        ax,
        0.260,
        0.595,
        0.065,
        0.060,
        "Source A\nColor",
        fc=C["task_light"],
        ec=C["task"],
        size=6.7,
    )
    module_box(
        ax,
        0.340,
        0.580,
        0.095,
        0.090,
        "BT.601 RGB to YCbCr\nY = 0.299R + 0.587G\n+ 0.114B",
        fc=C["luma_light"],
        ec=C["luma"],
        size=6.5,
    )
    module_box(
        ax,
        0.450,
        0.595,
        0.040,
        0.060,
        "Y_A\n/ 255",
        fc=C["luma_light"],
        ec=C["luma"],
        size=6.3,
    )
    arrow(ax, (0.325, 0.625), (0.340, 0.625), color=C["task"], ms=9)
    arrow(ax, (0.435, 0.625), (0.450, 0.625), color=C["luma"], ms=9)

    module_box(
        ax,
        0.260,
        0.430,
        0.065,
        0.060,
        "Source B\nGray",
        fc=C["task_light"],
        ec=C["task"],
        size=6.7,
    )
    module_box(
        ax,
        0.340,
        0.415,
        0.095,
        0.090,
        "Gray to Luminance\nY = gray\nCb = Cr = 128",
        fc=C["luma_light"],
        ec=C["luma"],
        size=6.6,
    )
    module_box(
        ax,
        0.450,
        0.430,
        0.040,
        0.060,
        "Y_B\n/ 255",
        fc=C["luma_light"],
        ec=C["luma"],
        size=6.3,
    )
    arrow(ax, (0.325, 0.460), (0.340, 0.460), color=C["task"], ms=9)
    arrow(ax, (0.435, 0.460), (0.450, 0.460), color=C["luma"], ms=9)

    text(
        ax,
        0.375,
        0.697,
        "Color and gray sources share the same Y-domain contract",
        size=6.4,
        color=C["muted"],
        style="italic",
    )

    module_box(
        ax,
        0.315,
        0.255,
        0.140,
        0.070,
        "Inference-only Cb / Cr Buffer\ncolor CbCr + neutral gray CbCr",
        fc=C["chroma_light"],
        ec=C["chroma"],
        size=6.8,
    )
    routed_arrow(
        ax,
        [(0.3875, 0.580), (0.3875, 0.550), (0.250, 0.550),
         (0.250, 0.290), (0.315, 0.290)],
        color=C["chroma"],
        lw=1.0,
        ms=9,
    )
    arrow(
        ax,
        (0.420, 0.415),
        (0.420, 0.325),
        color=C["chroma"],
        lw=1.0,
        ms=9,
    )
    text(
        ax,
        0.375,
        0.225,
        "Only normalized Y enters the backbone",
        size=6.6,
        weight="bold",
        color=C["luma"],
    )

    y_bus_x = 0.514
    line(ax, [(y_bus_x, 0.460), (y_bus_x, 0.7675)], color=C["luma"], lw=1.1, z=2)
    arrow(
        ax,
        (0.490, 0.625),
        (y_bus_x, 0.625),
        color=C["luma"],
        lw=1.0,
        ms=9,
        shrink_b=0,
    )
    arrow(
        ax,
        (0.490, 0.460),
        (y_bus_x, 0.460),
        color=C["luma"],
        lw=1.0,
        ms=9,
        shrink_b=0,
    )

    module_box(
        ax,
        0.540,
        0.735,
        0.185,
        0.065,
        "Shape Check and Alignment\nIf B differs: bilinear resize B to A",
        fc=C["align_light"],
        ec=C["align"],
        size=7.2,
    )
    arrow(
        ax,
        (y_bus_x, 0.7675),
        (0.540, 0.7675),
        color=C["luma"],
        lw=1.15,
        shrink_a=0,
    )

    group_box(
        ax,
        0.540,
        0.420,
        0.185,
        0.265,
        fc="#FAFDF8",
        ec=C["train"],
        lw=1.15,
        radius=0.011,
        z=1,
    )
    text(ax, 0.6325, 0.659, "Training Path", size=8.0, weight="bold", color=C["train"])
    module_box(
        ax,
        0.558,
        0.585,
        0.149,
        0.050,
        "Reflect-pad if H or W < 170",
        fc=C["train_light"],
        ec=C["train"],
        size=6.9,
    )
    module_box(
        ax,
        0.558,
        0.510,
        0.149,
        0.050,
        "Same (top, left) crop for A and B\n170 x 170; no resizing",
        fc=C["train_light"],
        ec=C["train"],
        size=6.5,
    )
    module_box(
        ax,
        0.558,
        0.435,
        0.149,
        0.050,
        "Balanced task quota\nabout 4000 crops / task; fixed seed",
        fc=C["train_light"],
        ec=C["train"],
        size=6.6,
    )
    arrow(ax, (0.6325, 0.585), (0.6325, 0.560), color=C["train"], ms=9)
    arrow(ax, (0.6325, 0.510), (0.6325, 0.485), color=C["train"], ms=9)

    group_box(
        ax,
        0.540,
        0.245,
        0.185,
        0.120,
        fc="#F8FAFD",
        ec=C["infer"],
        lw=1.15,
        radius=0.011,
        z=1,
    )
    text(ax, 0.6325, 0.340, "Inference Path", size=8.0, weight="bold", color=C["infer"])
    module_box(
        ax,
        0.558,
        0.270,
        0.149,
        0.050,
        "Full-resolution aligned Y pair\nno crop; batch size = 1",
        fc=C["infer_light"],
        ec=C["infer"],
        size=6.7,
    )

    routed_arrow(
        ax,
        [(0.595, 0.735), (0.595, 0.700), (0.595, 0.635)],
        color=C["train"],
        lw=1.05,
        ms=9,
    )
    routed_arrow(
        ax,
        [(0.540, 0.755), (0.530, 0.755), (0.530, 0.295), (0.558, 0.295)],
        color=C["infer"],
        lw=1.05,
        ms=9,
    )

    rounded(
        ax,
        0.780,
        0.525,
        0.175,
        0.205,
        fc=C["contract_light"],
        ec=C["contract"],
        lw=1.2,
        radius=0.010,
        z=4,
    )
    text(
        ax,
        0.8675,
        0.704,
        "Unified Network Input",
        size=8.6,
        weight="bold",
        color=C["contract"],
    )
    channel_planes(ax, 0.797, 0.615)
    text(
        ax,
        0.885,
        0.653,
        "X = concat(Y_A, Y_B)\n2 channels; range [0, 1]",
        size=7.1,
        weight="bold",
        color=C["contract"],
    )
    text(
        ax,
        0.8675,
        0.582,
        "Train: 2 x 170 x 170\nInfer: 2 x H x W\nplus task_id = t",
        size=7.0,
        weight="bold",
    )

    output_bus_x = 0.751
    line(
        ax,
        [(output_bus_x, 0.295), (output_bus_x, 0.595)],
        color=C["contract"],
        lw=1.1,
        z=2,
    )
    arrow(
        ax,
        (0.707, 0.460),
        (output_bus_x, 0.460),
        color=C["train"],
        lw=1.1,
        ms=10,
        shrink_b=0,
    )
    arrow(
        ax,
        (0.707, 0.295),
        (output_bus_x, 0.295),
        color=C["infer"],
        lw=1.1,
        ms=10,
        shrink_b=0,
    )
    arrow(
        ax,
        (output_bus_x, 0.595),
        (0.780, 0.595),
        color=C["contract"],
        lw=1.15,
        ms=10,
        shrink_a=0,
    )

    module_box(
        ax,
        0.800,
        0.405,
        0.135,
        0.065,
        "Shared U-MoE Backbone\nsame weights for all tasks",
        fc=C["network_light"],
        ec=C["network"],
        size=7.0,
    )
    arrow(
        ax,
        (0.8675, 0.525),
        (0.8675, 0.470),
        color=C["contract"],
        lw=1.2,
    )

    module_box(
        ax,
        0.785,
        0.195,
        0.165,
        0.130,
        "Post-Fusion Color Reconstruction\nCb_f / Cr_f weighted by |c - 128|\n"
        "gray contributes neutral chroma\nRGB tasks: fused Y + Cb_f / Cr_f to RGB\n"
        "gray tasks: save fused Y directly",
        fc=C["chroma_light"],
        ec=C["chroma"],
        size=6.5,
    )
    routed_arrow(
        ax,
        [(0.455, 0.290), (0.500, 0.290), (0.500, 0.145),
         (0.8675, 0.145), (0.8675, 0.195)],
        color=C["chroma"],
        lw=1.15,
        ms=10,
    )

    rounded(
        ax,
        0.015,
        0.018,
        0.973,
        0.090,
        fc="#F7F9FA",
        ec="#CBD3D9",
        lw=1.0,
        radius=0.012,
        z=1,
    )
    text(
        ax,
        0.027,
        0.091,
        "Complete Legend",
        size=8.0,
        weight="bold",
        color=C["main_edge"],
        ha="left",
    )
    legend = [
        ("Paired Tasks", C["task_light"], C["task"], 0.095),
        ("Pair Matching", C["pair_light"], C["pair"], 0.100),
        ("Luminance Y", C["luma_light"], C["luma"], 0.095),
        ("Alignment", C["align_light"], C["align"], 0.090),
        ("Training Only", C["train_light"], C["train"], 0.095),
        ("Inference", C["infer_light"], C["infer"], 0.085),
        ("Cb / Cr Bypass", C["chroma_light"], C["chroma"], 0.100),
        ("Model Contract", C["contract_light"], C["contract"], 0.100),
    ]
    x = 0.027
    for label, fc, ec, width in legend:
        legend_item(ax, x, 0.035, width, label, fc, ec)
        x += width + 0.008
    text(
        ax,
        0.966,
        0.052,
        "All model inputs are luminance-only",
        size=6.5,
        color=C["muted"],
        ha="right",
        style="italic",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "fig_u_moe_unified_preprocessing_detail"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return base


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
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
    figure_base = build_figure(args.output_dir, args.font_dir)
    print(f"wrote {figure_base}.pdf and {figure_base}.png with Times New Roman")
