#!/usr/bin/env python3
"""Generate the detailed U-MoE-Fusion 8x8 window-attention diagram."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path


def _bootstrap_mpl_cache() -> None:
    """Use a fixed font cache without recursively scanning shared storage."""
    cache = Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/u_moe_window_attn_mpl"))
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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


C = {
    "ink": "#263746",
    "muted": "#657786",
    "line": "#93A2AD",
    "white": "#FFFFFF",
    "main_bg": "#FFF4E3",
    "main_edge": "#E38A22",
    "detail_bg": "#E8EEF7",
    "detail_edge": "#7589AA",
    "input": "#DF7B28",
    "input_light": "#FBE4C9",
    "norm": "#168A87",
    "norm_light": "#DDF3F0",
    "pad": "#3D73B9",
    "pad_light": "#E2ECF8",
    "window": "#6C9E3D",
    "window_light": "#E9F2DF",
    "q": "#D65B61",
    "q_light": "#F7DCDD",
    "k": "#5577B8",
    "k_light": "#E1E8F5",
    "v": "#4E9A69",
    "v_light": "#E0F0E5",
    "bias": "#C18B1B",
    "bias_light": "#FBF0D1",
    "sdpa": "#9B55A9",
    "sdpa_light": "#F1E4F4",
    "merge": "#627C8E",
    "merge_light": "#E6EDF1",
    "residual": "#DE7B25",
    "residual_light": "#FBE7D4",
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


def group_box(ax, x, y, w, h, *, fc, ec, lw=1.35, radius=0.017, z=1):
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
        fontsize=size,
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


def draw_window_grid(ax, x, y, w, h, n=16):
    colors = ["#E6F1DD", "#E2ECF8", "#F7E2F0", "#FBF0D1"]
    half = n // 2
    ax.add_patch(Rectangle((x, y + h / 2), w / 2, h / 2, facecolor=colors[0],
                           edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((x + w / 2, y + h / 2), w / 2, h / 2, facecolor=colors[1],
                           edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((x, y), w / 2, h / 2, facecolor=colors[2],
                           edgecolor="none", zorder=3))
    ax.add_patch(Rectangle((x + w / 2, y), w / 2, h / 2, facecolor=colors[3],
                           edgecolor="none", zorder=3))
    for index in range(n + 1):
        lw = 1.4 if index in (0, half, n) else 0.35
        color = C["window"] if index in (0, half, n) else "#AFB9B2"
        ax.plot([x + index * w / n, x + index * w / n], [y, y + h],
                color=color, linewidth=lw, zorder=5)
        ax.plot([x, x + w], [y + index * h / n, y + index * h / n],
                color=color, linewidth=lw, zorder=5)
    text(ax, x + w * 0.25, y + h * 0.75, "W1", size=6.2, weight="bold",
         color=C["window"], bbox={"facecolor": C["white"], "edgecolor": "none", "pad": 0.8})
    text(ax, x + w * 0.75, y + h * 0.75, "W2", size=6.2, weight="bold",
         color=C["pad"], bbox={"facecolor": C["white"], "edgecolor": "none", "pad": 0.8})
    text(ax, x + w * 0.25, y + h * 0.25, "W3", size=6.2, weight="bold",
         color=C["sdpa"], bbox={"facecolor": C["white"], "edgecolor": "none", "pad": 0.8})
    text(ax, x + w * 0.75, y + h * 0.25, "W4", size=6.2, weight="bold",
         color=C["bias"], bbox={"facecolor": C["white"], "edgecolor": "none", "pad": 0.8})


def draw_token_strip(ax, x, y, w, h):
    colors = ["#D9EBCB", "#BFDDA7", "#9CC67C", "#7CAF5E"]
    count = 8
    gap = 0.002
    cell_w = (w - gap * (count - 1)) / count
    for index in range(count):
        ax.add_patch(
            Rectangle(
                (x + index * (cell_w + gap), y),
                cell_w,
                h,
                facecolor=colors[index % len(colors)],
                edgecolor=C["window"],
                linewidth=0.55,
                zorder=5,
            )
        )


def draw_small_window(ax, x, y, w, h):
    n = 8
    for row in range(n):
        for col in range(n):
            value = (row + col) / (2 * (n - 1))
            fc = (0.84 - 0.20 * value, 0.91 - 0.12 * value, 0.98 - 0.04 * value)
            ax.add_patch(
                Rectangle(
                    (x + col * w / n, y + (n - 1 - row) * h / n),
                    w / n,
                    h / n,
                    facecolor=fc,
                    edgecolor="#AAB7C6",
                    linewidth=0.35,
                    zorder=4,
                )
            )
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=C["detail_edge"],
                           linewidth=1.0, zorder=5))
    ax.add_patch(Rectangle((x + 3 * w / n, y + 4 * h / n), w / n, h / n,
                           facecolor=C["q_light"], edgecolor=C["q"], linewidth=1.0, zorder=6))


def draw_attention_matrix(ax, x, y, w, h):
    n = 8
    for row in range(n):
        for col in range(n):
            distance = abs(row - col) / (n - 1)
            intensity = 1.0 - 0.72 * distance
            fc = (0.74 + 0.20 * (1 - intensity), 0.55 + 0.30 * (1 - intensity),
                  0.78 + 0.15 * (1 - intensity))
            ax.add_patch(
                Rectangle(
                    (x + col * w / n, y + (n - 1 - row) * h / n),
                    w / n,
                    h / n,
                    facecolor=fc,
                    edgecolor=C["white"],
                    linewidth=0.35,
                    zorder=4,
                )
            )
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=C["sdpa"],
                           linewidth=1.0, zorder=5))


def legend_item(ax, x, y, w, label, fc, ec):
    rounded(ax, x, y, w, 0.034, fc=fc, ec=ec, lw=0.95, radius=0.006, z=4)
    text(ax, x + w / 2, y + 0.017, label, size=6.7, weight="bold", color=ec)


def build_figure(output_dir: Path, font_dir: Path) -> Path:
    setup_style(font_dir)
    fig = plt.figure(figsize=(18.0, 7.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    text(
        ax,
        0.5,
        0.968,
        "Detailed 8 x 8 Window Attention in U-MoE-Fusion",
        size=18,
        weight="bold",
    )

    group_box(
        ax,
        0.012,
        0.130,
        0.720,
        0.800,
        fc=C["main_bg"],
        ec=C["main_edge"],
        lw=1.7,
        radius=0.025,
        z=0,
    )
    group_box(
        ax,
        0.745,
        0.130,
        0.243,
        0.800,
        fc=C["detail_bg"],
        ec=C["detail_edge"],
        lw=1.7,
        radius=0.025,
        z=0,
    )

    text(
        ax,
        0.372,
        0.902,
        "Regular Non-Shifted Window Attention Path",
        size=11.0,
        weight="bold",
        color=C["main_edge"],
    )
    text(
        ax,
        0.8665,
        0.902,
        "Inside One 8 x 8 Window",
        size=11.0,
        weight="bold",
        color=C["detail_edge"],
    )

    subpanels = [
        (0.027, 0.178, 0.190, 0.680, "#FFFCF7", C["window"], "Spatial Windowing"),
        (0.235, 0.178, 0.290, 0.680, "#FFF9FC", C["sdpa"], "Q / K / V and 8 Heads"),
        (0.543, 0.178, 0.170, 0.680, "#FAFCFD", C["merge"], "Reverse and Residual"),
    ]
    for x, y, w, h, fc, ec, title in subpanels:
        group_box(ax, x, y, w, h, fc=fc, ec=ec, lw=1.2, radius=0.013, z=1)
        text(ax, x + w / 2, y + h - 0.025, title, size=8.4, weight="bold", color=ec)

    module_box(
        ax,
        0.047,
        0.752,
        0.150,
        0.055,
        "Input Feature F   B x H x W x 96",
        fc=C["input_light"],
        ec=C["input"],
        size=7.5,
    )
    module_box(
        ax,
        0.047,
        0.670,
        0.150,
        0.052,
        "Layer Normalization",
        fc=C["norm_light"],
        ec=C["norm"],
        size=7.5,
        linestyle=(0, (1.5, 1.5)),
    )
    module_box(
        ax,
        0.047,
        0.585,
        0.150,
        0.055,
        "Reflect Pad H and W to multiples of 8\ntraining: 170 x 170 to 176 x 176",
        fc=C["pad_light"],
        ec=C["pad"],
        size=6.5,
    )
    arrow(ax, (0.122, 0.752), (0.122, 0.722), color=C["norm"], ms=9)
    arrow(ax, (0.122, 0.670), (0.122, 0.640), color=C["pad"], ms=9)

    draw_window_grid(ax, 0.065, 0.335, 0.114, 0.220, n=16)
    arrow(ax, (0.122, 0.585), (0.122, 0.555), color=C["window"], ms=9)

    module_box(
        ax,
        0.052,
        0.205,
        0.140,
        0.070,
        "Flatten: (B x nW) x 64 x 96\ntrain nW = 22 x 22 = 484\nregular windows; shift size = 0",
        fc=C["window_light"],
        ec=C["window"],
        size=6.3,
    )
    arrow(ax, (0.122, 0.335), (0.122, 0.275), color=C["window"])

    module_box(
        ax,
        0.255,
        0.205,
        0.105,
        0.070,
        "Window Tokens\n64 x 96",
        fc=C["window_light"],
        ec=C["window"],
        size=7.2,
    )
    arrow(
        ax,
        (0.192, 0.240),
        (0.255, 0.240),
        color=C["window"],
        lw=1.15,
    )

    module_box(
        ax,
        0.255,
        0.320,
        0.105,
        0.065,
        "Linear QKV\n96 to 288",
        fc=C["sdpa_light"],
        ec=C["sdpa"],
        size=7.2,
    )
    arrow(ax, (0.3075, 0.275), (0.3075, 0.320), color=C["sdpa"])

    qkv_specs = [
        (0.255, "Q\n8 x 64 x 12", C["q_light"], C["q"]),
        (0.345, "K\n8 x 64 x 12", C["k_light"], C["k"]),
        (0.435, "V\n8 x 64 x 12", C["v_light"], C["v"]),
    ]
    qkv_centers = []
    for x, label, fc, ec in qkv_specs:
        module_box(ax, x, 0.440, 0.065, 0.060, label, fc=fc, ec=ec, size=6.7)
        qkv_centers.append(x + 0.0325)

    line(ax, [(0.3075, 0.385), (0.3075, 0.415)],
         color=C["sdpa"], lw=1.0, z=2)
    line(ax, [(0.2875, 0.415), (0.4675, 0.415)],
         color=C["sdpa"], lw=1.0, z=2)
    for center, color in zip(qkv_centers, [C["q"], C["k"], C["v"]]):
        arrow(
            ax,
            (center, 0.415),
            (center, 0.440),
            color=color,
            lw=1.0,
            ms=9,
            shrink_a=0,
        )

    module_box(
        ax,
        0.265,
        0.565,
        0.230,
        0.085,
        "Fused Scaled Dot-Product Attention\n"
        "softmax(Q K^T / sqrt(12) + B_rel) V",
        fc=C["sdpa_light"],
        ec=C["sdpa"],
        size=7.3,
    )
    target_ports = [0.305, 0.380, 0.455]
    for center, target, color in zip(qkv_centers, target_ports, [C["q"], C["k"], C["v"]]):
        arrow(
            ax,
            (center, 0.500),
            (target, 0.565),
            color=color,
            lw=1.0,
            ms=9,
        )

    module_box(
        ax,
        0.275,
        0.705,
        0.210,
        0.060,
        "8 Head Outputs   (B x nW) x 8 x 64 x 12",
        fc=C["merge_light"],
        ec=C["merge"],
        size=7.0,
    )
    arrow(ax, (0.380, 0.650), (0.380, 0.705), color=C["merge"], lw=1.1)

    module_box(
        ax,
        0.560,
        0.705,
        0.135,
        0.060,
        "Head Outputs\n8 x 64 x 12",
        fc=C["merge_light"],
        ec=C["merge"],
        size=6.9,
    )
    arrow(
        ax,
        (0.485, 0.735),
        (0.560, 0.735),
        color=C["merge"],
        lw=1.15,
    )

    module_box(
        ax,
        0.560,
        0.610,
        0.135,
        0.055,
        "Concat Heads   64 x 96",
        fc=C["merge_light"],
        ec=C["merge"],
        size=7.1,
    )
    module_box(
        ax,
        0.560,
        0.520,
        0.135,
        0.055,
        "Linear Projection   96 to 96",
        fc=C["merge_light"],
        ec=C["merge"],
        size=6.9,
    )
    module_box(
        ax,
        0.560,
        0.425,
        0.135,
        0.060,
        "Reshape Each Output\n64 tokens to 8 x 8",
        fc=C["window_light"],
        ec=C["window"],
        size=6.8,
    )
    module_box(
        ax,
        0.560,
        0.325,
        0.135,
        0.060,
        "Window Reverse\nand Crop to H x W",
        fc=C["pad_light"],
        ec=C["pad"],
        size=6.8,
    )
    module_box(
        ax,
        0.555,
        0.205,
        0.145,
        0.075,
        "Attention Residual\nF_attn = F + WindowAttn(LN(F))",
        fc=C["residual_light"],
        ec=C["residual"],
        size=6.5,
    )
    arrow(ax, (0.6275, 0.705), (0.6275, 0.665), color=C["merge"], ms=9)
    arrow(ax, (0.6275, 0.610), (0.6275, 0.575), color=C["merge"], ms=9)
    arrow(ax, (0.6275, 0.520), (0.6275, 0.485), color=C["window"], ms=9)
    arrow(ax, (0.6275, 0.425), (0.6275, 0.385), color=C["pad"], ms=9)
    arrow(ax, (0.6275, 0.325), (0.6275, 0.280), color=C["residual"], ms=9)

    module_box(
        ax,
        0.770,
        0.790,
        0.193,
        0.055,
        "Final W96L: window = 8, C = 96, heads = 8, d = 12",
        fc=C["white"],
        ec=C["detail_edge"],
        size=7.2,
    )

    draw_small_window(ax, 0.775, 0.550, 0.095, 0.215)
    text(
        ax,
        0.8225,
        0.530,
        "64 spatial tokens",
        size=7.0,
        weight="bold",
        color=C["detail_edge"],
    )

    module_box(
        ax,
        0.885,
        0.665,
        0.080,
        0.080,
        "Relative Position\nBias Table\n15 x 15 = 225 / head",
        fc=C["bias_light"],
        ec=C["bias"],
        size=6.3,
    )
    module_box(
        ax,
        0.882,
        0.525,
        0.086,
        0.095,
        "One Head\nQ_h, K_h, V_h\n64 x 12",
        fc=C["sdpa_light"],
        ec=C["sdpa"],
        size=6.6,
    )
    arrow(
        ax,
        (0.870, 0.6575),
        (0.882, 0.5725),
        color=C["detail_edge"],
        lw=1.0,
        ms=9,
    )
    routed_arrow(
        ax,
        [(0.965, 0.705), (0.976, 0.705), (0.976, 0.4475), (0.965, 0.4475)],
        color=C["bias"],
        lw=1.0,
        ms=9,
    )

    module_box(
        ax,
        0.770,
        0.410,
        0.195,
        0.075,
        "A_h = softmax(Q_h K_h^T / sqrt(12) + B_rel)\nO_h = A_h V_h",
        fc=C["sdpa_light"],
        ec=C["sdpa"],
        size=6.9,
    )
    arrow(
        ax,
        (0.925, 0.525),
        (0.925, 0.485),
        color=C["sdpa"],
        lw=1.0,
        ms=9,
    )

    draw_attention_matrix(ax, 0.785, 0.220, 0.082, 0.170)
    text(
        ax,
        0.826,
        0.203,
        "A_h   64 x 64",
        size=6.7,
        weight="bold",
        color=C["sdpa"],
    )
    module_box(
        ax,
        0.885,
        0.270,
        0.080,
        0.070,
        "Head Output\nO_h   64 x 12",
        fc=C["merge_light"],
        ec=C["merge"],
        size=6.6,
    )
    arrow(
        ax,
        (0.826, 0.410),
        (0.826, 0.390),
        color=C["sdpa"],
        lw=1.0,
        ms=9,
    )
    arrow(
        ax,
        (0.867, 0.305),
        (0.885, 0.305),
        color=C["merge"],
        lw=1.0,
        ms=9,
    )

    module_box(
        ax,
        0.775,
        0.145,
        0.185,
        0.050,
        "PyTorch fused SDPA   mask = None   no shifted windows",
        fc=C["white"],
        ec=C["detail_edge"],
        size=6.4,
        linestyle=(0, (1.5, 1.5)),
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
        color=C["detail_edge"],
        ha="left",
    )
    legend = [
        ("Input / Residual", C["input_light"], C["input"], 0.105),
        ("LayerNorm", C["norm_light"], C["norm"], 0.090),
        ("Reflect Pad", C["pad_light"], C["pad"], 0.095),
        ("8 x 8 Windows", C["window_light"], C["window"], 0.100),
        ("Q / K / V", C["q_light"], C["q"], 0.085),
        ("Relative Bias", C["bias_light"], C["bias"], 0.095),
        ("Fused SDPA", C["sdpa_light"], C["sdpa"], 0.095),
        ("Concat / Projection", C["merge_light"], C["merge"], 0.115),
    ]
    x = 0.027
    for label, fc, ec, width in legend:
        legend_item(ax, x, 0.035, width, label, fc, ec)
        x += width + 0.008
    text(
        ax,
        0.970,
        0.052,
        "Each window attends locally over 64 tokens",
        size=6.5,
        color=C["muted"],
        ha="right",
        style="italic",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "fig_u_moe_window_attention_detail"
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
