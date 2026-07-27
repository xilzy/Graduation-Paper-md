#!/usr/bin/env python3
"""Generate the detailed U-MoE-Fusion MoE-FFN diagram in PDF and PNG."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path


def _bootstrap_mpl_cache() -> None:
    """Use a fixed font cache without recursively scanning shared storage."""
    cache = Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/u_moe_ffn_detail_mpl"))
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
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


C = {
    "ink": "#263746",
    "muted": "#657786",
    "line": "#93A2AD",
    "white": "#FFFFFF",
    "main_bg": "#FFF5E6",
    "main_edge": "#E48A1D",
    "expert_bg": "#E8EEF7",
    "expert_edge": "#7689AA",
    "context": "#168A87",
    "context_light": "#DDF3F0",
    "input": "#E99633",
    "input_light": "#FBE2BC",
    "router": "#C04B91",
    "router_light": "#F7E2F0",
    "shared": "#6C9E3D",
    "shared_light": "#E9F2DF",
    "routed": "#C58F1D",
    "routed_light": "#FBF0D2",
    "active": "#D96624",
    "active_light": "#F8D5B8",
    "merge": "#627C8E",
    "merge_light": "#E6EDF1",
    "residual": "#DF7C26",
    "residual_light": "#FBE7D4",
    "loss": "#D44E55",
    "loss_light": "#F9E2E4",
    "expert_net": "#8BAA62",
    "expert_net_light": "#EDF3E4",
}


def register_times_new_roman(font_dir: Path) -> str:
    """Register the genuine Microsoft Times New Roman font family."""
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


def group_box(ax, x, y, w, h, *, fc, ec, lw=1.4, radius=0.018, z=1):
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
    size=8.5,
    weight="bold",
    linestyle="solid",
    radius=0.009,
):
    rounded(
        ax,
        x,
        y,
        w,
        h,
        fc=fc,
        ec=ec,
        lw=1.15,
        radius=radius,
        linestyle=linestyle,
        z=4,
    )
    text(ax, x + w / 2, y + h / 2, label, size=size, weight=weight, color=ec)


def draw_dense_connections(ax, upper, lower, color):
    for ux, uy in upper:
        for lx, ly in lower:
            line(ax, [(ux, uy), (lx, ly)], color=color, lw=0.65, z=2)


def draw_node_row(ax, xs, y, *, fc, ec, radius=0.008):
    nodes = []
    for x in xs:
        circle = Circle(
            (x, y),
            radius,
            facecolor=fc,
            edgecolor=ec,
            linewidth=0.9,
            zorder=5,
        )
        ax.add_patch(circle)
        nodes.append((x, y))
    return nodes


def legend_item(ax, x, y, w, label, fc, ec):
    rounded(ax, x, y, w, 0.034, fc=fc, ec=ec, lw=0.95, radius=0.006, z=4)
    text(ax, x + w / 2, y + 0.017, label, size=6.8, weight="bold", color=ec)


def build_figure(output_dir: Path, font_dir: Path) -> Path:
    setup_style(font_dir)
    fig = plt.figure(figsize=(18.0, 7.3))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    text(
        ax,
        0.5,
        0.966,
        "Task-Conditioned MoE-FFN in U-MoE-Fusion",
        size=18,
        weight="bold",
    )

    group_box(
        ax,
        0.012,
        0.130,
        0.735,
        0.795,
        fc=C["main_bg"],
        ec=C["main_edge"],
        lw=1.7,
        radius=0.025,
        z=0,
    )
    group_box(
        ax,
        0.758,
        0.130,
        0.230,
        0.795,
        fc=C["expert_bg"],
        ec=C["expert_edge"],
        lw=1.7,
        radius=0.025,
        z=0,
    )

    text(
        ax,
        0.379,
        0.901,
        "MoE-FFN Forward Path",
        size=11.5,
        weight="bold",
        color=C["main_edge"],
    )
    text(
        ax,
        0.873,
        0.901,
        "One FFN Expert",
        size=11.5,
        weight="bold",
        color=C["expert_edge"],
    )

    group_box(
        ax,
        0.027,
        0.178,
        0.145,
        0.675,
        fc="#FFFDF8",
        ec=C["context"],
        lw=1.25,
        radius=0.012,
        z=1,
    )
    text(
        ax,
        0.0995,
        0.829,
        "Transformer Context",
        size=8.4,
        weight="bold",
        color=C["context"],
    )

    module_box(
        ax,
        0.044,
        0.752,
        0.111,
        0.060,
        "Input Feature Map  F'",
        fc=C["input_light"],
        ec=C["input"],
        size=8.3,
    )
    module_box(
        ax,
        0.044,
        0.657,
        0.111,
        0.058,
        "Layer Normalization",
        fc=C["context_light"],
        ec=C["context"],
        size=8.1,
        linestyle=(0, (1.5, 1.5)),
    )
    module_box(
        ax,
        0.044,
        0.555,
        0.111,
        0.065,
        "Tokens h  (shape T x C)",
        fc=C["white"],
        ec=C["context"],
        size=8.2,
        linestyle=(0, (1.5, 1.5)),
    )
    text(
        ax,
        0.0995,
        0.532,
        "T = B x H x W after flatten",
        size=6.5,
        color=C["muted"],
        style="italic",
    )
    arrow(ax, (0.0995, 0.752), (0.0995, 0.715), color=C["context"])
    arrow(ax, (0.0995, 0.657), (0.0995, 0.620), color=C["context"])

    group_box(
        ax,
        0.194,
        0.178,
        0.204,
        0.675,
        fc="#FFF9FC",
        ec=C["router"],
        lw=1.25,
        radius=0.012,
        z=1,
    )
    text(
        ax,
        0.296,
        0.829,
        "Task-Conditioned Router",
        size=8.4,
        weight="bold",
        color=C["router"],
    )

    module_box(
        ax,
        0.211,
        0.750,
        0.071,
        0.055,
        "Task ID  t",
        fc=C["router_light"],
        ec=C["router"],
        size=7.7,
    )
    module_box(
        ax,
        0.304,
        0.750,
        0.077,
        0.055,
        "Embedding  e_t",
        fc=C["router_light"],
        ec=C["router"],
        size=7.4,
    )
    arrow(
        ax,
        (0.282, 0.7775),
        (0.304, 0.7775),
        color=C["router"],
        shrink_a=1.5,
        shrink_b=1.5,
    )

    module_box(
        ax,
        0.211,
        0.555,
        0.170,
        0.065,
        "Routing Condition  z = h + e_t",
        fc=C["router_light"],
        ec=C["router"],
        size=8.0,
    )
    arrow(
        ax,
        (0.155, 0.5875),
        (0.211, 0.5875),
        color=C["context"],
    )
    routed_arrow(
        ax,
        [(0.3425, 0.750), (0.3425, 0.660), (0.322, 0.620)],
        color=C["router"],
        lw=1.15,
    )

    module_box(
        ax,
        0.211,
        0.445,
        0.170,
        0.062,
        "Linear Gate  W_g z  +  Softmax",
        fc=C["router_light"],
        ec=C["router"],
        size=7.9,
    )
    arrow(ax, (0.296, 0.555), (0.296, 0.507), color=C["router"])

    module_box(
        ax,
        0.211,
        0.340,
        0.170,
        0.062,
        "Top-2 + Renormalize\n{(alpha_i, i), (alpha_j, j)}",
        fc=C["white"],
        ec=C["router"],
        size=7.5,
        linestyle=(0, (1.5, 1.5)),
    )
    arrow(ax, (0.296, 0.445), (0.296, 0.402), color=C["router"])

    module_box(
        ax,
        0.211,
        0.215,
        0.170,
        0.060,
        "Training Only\n0.01 L_balance = 0.01 x 12 sum_i(f_i p_i)",
        fc=C["loss_light"],
        ec=C["loss"],
        size=6.7,
    )
    arrow(
        ax,
        (0.250, 0.340),
        (0.250, 0.275),
        color=C["loss"],
        lw=1.0,
        ms=9,
        linestyle=(0, (2.0, 2.0)),
    )

    group_box(
        ax,
        0.414,
        0.178,
        0.216,
        0.675,
        fc="#FFFCF4",
        ec=C["routed"],
        lw=1.25,
        radius=0.012,
        z=1,
    )
    text(
        ax,
        0.522,
        0.829,
        "Shared + Routed Experts",
        size=8.4,
        weight="bold",
        color=C["routed"],
    )

    module_box(
        ax,
        0.434,
        0.716,
        0.176,
        0.070,
        "Shared Expert  E_s(h)\nAlways On",
        fc=C["shared_light"],
        ec=C["shared"],
        size=7.7,
    )
    routed_arrow(
        ax,
        [
            (0.155, 0.600),
            (0.182, 0.600),
            (0.182, 0.875),
            (0.405, 0.875),
            (0.405, 0.751),
            (0.434, 0.751),
        ],
        color=C["shared"],
        lw=1.15,
    )
    module_box(
        ax,
        0.434,
        0.340,
        0.176,
        0.062,
        "Grouped-Capacity Top-2 Dispatch\ncap=max(1, floor(1.25Tk/E)); k=2, E=12",
        fc=C["routed_light"],
        ec=C["routed"],
        size=6.5,
    )
    arrow(
        ax,
        (0.381, 0.371),
        (0.434, 0.371),
        color=C["router"],
        lw=1.2,
    )

    expert_labels = ["E1", "E2", "E3", "...", "E9", "...", "E12"]
    expert_x = [0.434 + i * 0.026 for i in range(len(expert_labels))]
    active_indices = {2, 4}
    for index, (x, label) in enumerate(zip(expert_x, expert_labels)):
        is_active = index in active_indices
        module_box(
            ax,
            x,
            0.257,
            0.022,
            0.038,
            label,
            fc=C["active_light"] if is_active else C["white"],
            ec=C["active"] if is_active else C["routed"],
            size=6.0,
            radius=0.005,
        )

    e3_center = expert_x[2] + 0.011
    e9_center = expert_x[4] + 0.011
    arrow(
        ax,
        (0.485, 0.340),
        (e3_center, 0.295),
        color=C["active"],
        lw=1.05,
        ms=9,
    )
    arrow(
        ax,
        (0.561, 0.340),
        (e9_center, 0.295),
        color=C["active"],
        lw=1.05,
        ms=9,
    )
    module_box(
        ax,
        0.445,
        0.185,
        0.160,
        0.043,
        "Routed Output  sum over Top-2 alpha_i E_i(h)",
        fc=C["routed_light"],
        ec=C["routed"],
        size=6.5,
    )
    arrow(
        ax,
        (e3_center, 0.257),
        (0.493, 0.228),
        color=C["active"],
        lw=0.95,
        ms=8,
    )
    arrow(
        ax,
        (e9_center, 0.257),
        (0.557, 0.228),
        color=C["active"],
        lw=0.95,
        ms=8,
    )

    group_box(
        ax,
        0.640,
        0.178,
        0.092,
        0.675,
        fc="#FAFCFD",
        ec=C["merge"],
        lw=1.25,
        radius=0.012,
        z=1,
    )
    text(
        ax,
        0.686,
        0.829,
        "Merge",
        size=8.4,
        weight="bold",
        color=C["merge"],
    )

    moe_sum = Circle(
        (0.686, 0.600),
        0.018,
        facecolor=C["merge_light"],
        edgecolor=C["merge"],
        linewidth=1.15,
        zorder=5,
    )
    ax.add_patch(moe_sum)
    text(ax, 0.686, 0.600, "SUM", size=6.4, weight="bold", color=C["merge"])
    arrow(
        ax,
        (0.610, 0.751),
        (0.672, 0.612),
        color=C["shared"],
        lw=1.05,
    )
    routed_arrow(
        ax,
        [(0.605, 0.2065), (0.621, 0.2065), (0.621, 0.588), (0.668, 0.588)],
        color=C["routed"],
        lw=1.05,
    )

    module_box(
        ax,
        0.648,
        0.492,
        0.076,
        0.064,
        "F_MoE = E_s(h)\n+ sum alpha_i E_i(h)",
        fc=C["merge_light"],
        ec=C["merge"],
        size=6.3,
    )
    arrow(ax, (0.686, 0.582), (0.686, 0.556), color=C["merge"], ms=9)

    residual_sum = Circle(
        (0.686, 0.398),
        0.018,
        facecolor=C["residual_light"],
        edgecolor=C["residual"],
        linewidth=1.15,
        zorder=5,
    )
    ax.add_patch(residual_sum)
    text(ax, 0.686, 0.398, "+", size=11.0, weight="bold", color=C["residual"])
    arrow(ax, (0.686, 0.492), (0.686, 0.416), color=C["merge"])

    module_box(
        ax,
        0.650,
        0.252,
        0.072,
        0.062,
        "Output  F''\nF'' = F' + F_MoE",
        fc=C["input_light"],
        ec=C["input"],
        size=6.7,
    )
    arrow(ax, (0.686, 0.380), (0.686, 0.314), color=C["residual"])

    routed_arrow(
        ax,
        [
            (0.044, 0.782),
            (0.021, 0.782),
            (0.021, 0.151),
            (0.636, 0.151),
            (0.636, 0.398),
            (0.668, 0.398),
        ],
        color=C["residual"],
        lw=1.15,
        ms=10,
    )
    text(
        ax,
        0.325,
        0.166,
        "Residual shortcut  F'",
        size=6.8,
        weight="bold",
        color=C["residual"],
        bbox={"facecolor": C["main_bg"], "edgecolor": "none", "pad": 1.0},
    )

    module_box(
        ax,
        0.797,
        0.792,
        0.152,
        0.058,
        "Expert Input  h_i  (C channels)",
        fc=C["white"],
        ec=C["expert_edge"],
        size=8.1,
    )

    network_group = group_box(
        ax,
        0.783,
        0.390,
        0.180,
        0.355,
        fc="#F8FAFD",
        ec=C["expert_net"],
        lw=1.2,
        radius=0.014,
        z=1,
    )
    del network_group

    input_nodes = draw_node_row(
        ax,
        [0.823, 0.856, 0.889, 0.922],
        0.700,
        fc=C["expert_bg"],
        ec=C["expert_edge"],
        radius=0.008,
    )
    hidden_nodes = draw_node_row(
        ax,
        [0.804, 0.827, 0.850, 0.873, 0.896, 0.919, 0.942],
        0.575,
        fc=C["expert_net_light"],
        ec=C["expert_net"],
        radius=0.008,
    )
    output_nodes = draw_node_row(
        ax,
        [0.823, 0.856, 0.889, 0.922],
        0.450,
        fc=C["expert_bg"],
        ec=C["expert_edge"],
        radius=0.008,
    )
    draw_dense_connections(ax, input_nodes, hidden_nodes, "#A6B5C6")
    draw_dense_connections(ax, hidden_nodes, output_nodes, "#A8BE8A")

    arrow(
        ax,
        (0.873, 0.792),
        (0.873, 0.708),
        color=C["expert_edge"],
        lw=1.15,
    )
    text(
        ax,
        0.873,
        0.663,
        "Linear  C to 4C",
        size=7.4,
        weight="bold",
        color=C["expert_edge"],
        bbox={"facecolor": "#F8FAFD", "edgecolor": "none", "pad": 1.2},
    )
    text(
        ax,
        0.873,
        0.518,
        "GELU + Dropout",
        size=7.4,
        weight="bold",
        color=C["expert_net"],
        bbox={"facecolor": "#F8FAFD", "edgecolor": "none", "pad": 1.2},
    )
    text(
        ax,
        0.873,
        0.408,
        "Linear  4C to C + Dropout",
        size=7.4,
        weight="bold",
        color=C["expert_edge"],
        bbox={"facecolor": "#F8FAFD", "edgecolor": "none", "pad": 1.2},
    )

    module_box(
        ax,
        0.797,
        0.302,
        0.152,
        0.058,
        "Expert Output  E_i(h)",
        fc=C["white"],
        ec=C["expert_edge"],
        size=8.1,
    )
    arrow(
        ax,
        (0.873, 0.442),
        (0.873, 0.360),
        color=C["expert_edge"],
        lw=1.15,
    )
    text(
        ax,
        0.873,
        0.243,
        "E_i(h) = Dropout(W2 x Dropout(GELU(W1 h)))",
        size=7.7,
        weight="bold",
        color=C["expert_edge"],
    )
    text(
        ax,
        0.873,
        0.185,
        "Same topology, independent parameters\nfor E_s, E1, ..., E12",
        size=7.2,
        color=C["muted"],
        style="italic",
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
        color=C["merge"],
        ha="left",
    )
    legend = [
        ("LayerNorm / Tokens", C["context_light"], C["context"], 0.105),
        ("Task Condition / Router", C["router_light"], C["router"], 0.120),
        ("Shared Expert", C["shared_light"], C["shared"], 0.095),
        ("Routed Expert", C["routed_light"], C["routed"], 0.095),
        ("Selected Top-2", C["active_light"], C["active"], 0.095),
        ("Merge / Residual", C["merge_light"], C["merge"], 0.100),
        ("Training Only", C["loss_light"], C["loss"], 0.095),
    ]
    x = 0.027
    for label, fc, ec, width in legend:
        legend_item(ax, x, 0.035, width, label, fc, ec)
        x += width + 0.010
    text(
        ax,
        0.962,
        0.052,
        "Solid: inference   Red dotted: auxiliary loss",
        size=6.4,
        color=C["muted"],
        ha="right",
        style="italic",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "fig_u_moe_ffn_detail"
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
