#!/usr/bin/env python3
"""Draw a ten-way distributed-training mechanism comparison."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path


def _bootstrap_mpl_cache() -> None:
    """Build a tiny deterministic cache without scanning shared filesystems."""
    cache = Path(os.environ.setdefault("MPLCONFIGDIR", "/tmp/u_moe_dist_parallel_mpl"))
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
    "ink": "#253746",
    "muted": "#647786",
    "line": "#AAB6BF",
    "card": "#FFFFFF",
    "canvas": "#F7F9FB",
    "data": "#2A9D8F",
    "data_light": "#E0F3EF",
    "overlap": "#3D7EDB",
    "overlap_light": "#E3EDFA",
    "state": "#517DA2",
    "state_light": "#E4EDF4",
    "balance": "#4D9B68",
    "balance_light": "#E2F0E6",
    "tp": "#8E6BBE",
    "tp_light": "#EEE7F6",
    "pp": "#E38A35",
    "pp_light": "#FBE9D8",
    "ep": "#D65F54",
    "ep_light": "#F8E2DF",
    "up": "#179A9A",
    "up_light": "#DFF2F2",
    "cp": "#5E708E",
    "cp_light": "#E5E9F0",
    "model": "#466B8A",
    "model_light": "#DCE8F1",
    "param": "#7061A8",
    "grad": "#D75B77",
    "optim": "#C28B2C",
    "activation": "#4DA17A",
    "ar": "#2675D8",
    "rsag": "#7A5CC7",
    "a2a": "#D44D70",
    "p2p": "#E37C25",
    "ring": "#526783",
    "adopt": "#287D4C",
    "adopt_light": "#DDF0E5",
    "builtin": "#2C6CB0",
    "builtin_light": "#E0EBF8",
    "skip": "#7A6654",
    "skip_light": "#F0EBE6",
    "sample_a": "#EF8C67",
    "sample_b": "#5EA9D6",
    "sample_c": "#86B965",
    "sample_d": "#B784C7",
}


def register_times_new_roman(font_dir: Path) -> str:
    os.environ.pop("MPL_IGNORE_SYSTEM_FONTS", None)
    paths = [font_dir / name for name in ("Times.TTF", "Timesbd.TTF", "Timesi.TTF", "Timesbi.TTF")]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Times New Roman files: " + ", ".join(missing))
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
            "font.size": 8,
            "text.color": C["ink"],
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
    fc=C["card"],
    ec=C["line"],
    lw=1.0,
    radius=0.008,
    linestyle="solid",
    z=3,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.0025,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=linestyle,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def publication_text_size(size):
    if size >= 15:
        return size
    if size >= 8:
        return size + 3.0
    if size >= 6:
        return size + 4.0
    return max(9.5, size + 4.5)


def text(
    ax,
    x,
    y,
    value,
    *,
    size=8,
    weight="normal",
    color=None,
    ha="center",
    va="center",
    style="normal",
    linespacing=1.05,
    bbox=None,
    z=8,
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


def line(ax, points, *, color=None, lw=1.1, linestyle="solid", z=2):
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
    lw=1.1,
    ms=8,
    linestyle="solid",
    shrink_a=1.5,
    shrink_b=1.5,
    z=4,
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
    lw=1.1,
    ms=8,
    linestyle="solid",
    z=4,
):
    if len(points) < 2:
        raise ValueError("routed_arrow requires at least two points")
    if len(points) > 2:
        line(ax, points[:-1], color=color, lw=lw, linestyle=linestyle, z=max(2, z - 1))
        return arrow(
            ax,
            points[-2],
            points[-1],
            color=color,
            lw=lw,
            ms=ms,
            linestyle=linestyle,
            shrink_a=0,
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
        z=z,
    )


def badge(ax, x, y, label, kind):
    if kind == "adopt":
        fc, ec = C["adopt_light"], C["adopt"]
    elif kind == "builtin":
        fc, ec = C["builtin_light"], C["builtin"]
    else:
        fc, ec = C["skip_light"], C["skip"]
    width = 0.009 + len(label) * 0.0040
    rounded(ax, x - width, y - 0.012, width, 0.024, fc=fc, ec=ec, lw=0.8, radius=0.006, z=5)
    text(ax, x - width / 2, y, label, size=5.5, weight="bold", color=ec)


def card_frame(ax, x, y, w, h, title, subtitle, *, color, light, verdict, verdict_kind):
    rounded(ax, x, y, w, h, fc=C["card"], ec=color, lw=1.2, radius=0.012, z=1)
    rounded(ax, x + 0.004, y + h - 0.070, w - 0.008, 0.064, fc=light, ec="none",
            lw=0, radius=0.009, z=2)
    text(ax, x + 0.010, y + h - 0.025, title, size=8.2, weight="bold", color=color, ha="left")
    text(ax, x + 0.010, y + h - 0.050, subtitle, size=6.2, color=C["muted"], ha="left")
    badge(ax, x + w - 0.010, y + h - 0.025, verdict, verdict_kind)
    line(ax, [(x + 0.008, y + 0.104), (x + w - 0.008, y + 0.104)],
         color="#D9E0E5", lw=0.7, z=2)


def notes(ax, x, y, w, shard, comm, fit, *, accent):
    rows = [("Shard", shard), ("Comm", comm), ("U-MoE", fit)]
    ys = [y + 0.084, y + 0.058, y + 0.032]
    for (label, value), yy in zip(rows, ys):
        text(ax, x + 0.012, yy, label, size=5.8, weight="bold", color=accent, ha="left")
        text(ax, x + 0.047, yy, value, size=5.7, color=C["ink"], ha="left")


def gpu(ax, x, y, w, h, label, rows, *, edge):
    rounded(ax, x, y, w, h, fc="#FBFCFD", ec=edge, lw=0.85, radius=0.004, z=4)
    ax.add_patch(Rectangle((x + 0.0015, y + h - 0.024), w - 0.003, 0.022,
                           facecolor=edge, edgecolor="none", alpha=0.92, zorder=5))
    text(ax, x + w / 2, y + h - 0.013, label, size=5.0, weight="bold", color=C["card"])
    if not rows:
        return
    inner_y = y + 0.006
    inner_h = h - 0.034
    row_h = inner_h / len(rows)
    for index, (value, fc, tc) in enumerate(rows):
        yy = inner_y + (len(rows) - index - 1) * row_h
        rounded(ax, x + 0.004, yy + 0.002, w - 0.008, row_h - 0.004,
                fc=fc, ec="none", lw=0, radius=0.0025, z=5)
        text(ax, x + w / 2, yy + row_h / 2, value, size=4.8, weight="bold", color=tc)


def four_gpu_centers(x, w):
    gpu_w = 0.034
    left = x + 0.014
    gap = (w - 0.028 - 4 * gpu_w) / 3
    xs = [left + index * (gpu_w + gap) for index in range(4)]
    return xs, gpu_w


def double_collective(ax, x0, x1, y, label, color):
    arrow(ax, (x0, y + 0.006), (x1, y + 0.006), color=color, lw=1.0, ms=7)
    arrow(ax, (x1, y - 0.006), (x0, y - 0.006), color=color, lw=1.0, ms=7)
    text(
        ax,
        (x0 + x1) / 2,
        y + 0.023,
        label,
        size=5.4,
        weight="bold",
        color=color,
        bbox={"facecolor": C["card"], "edgecolor": "none", "pad": 0.4},
    )


def draw_ddp(ax, x, y, w, h):
    color, light = C["data"], C["data_light"]
    card_frame(ax, x, y, w, h, "Data Parallel (DDP)", "replicate model; split mini-batch",
               color=color, light=light, verdict="ADOPT", verdict_kind="adopt")
    xs, gpu_w = four_gpu_centers(x, w)
    sample_colors = [C["sample_a"], C["sample_b"], C["sample_c"], C["sample_d"]]
    for index, gx in enumerate(xs):
        gpu(
            ax, gx, y + 0.130, gpu_w, 0.100, f"GPU {index}",
            [("Model M", C["model_light"], C["model"]),
             (f"Data D{index}", sample_colors[index] + "33", sample_colors[index])],
            edge=color,
        )
    centers = [gx + gpu_w / 2 for gx in xs]
    double_collective(ax, centers[0], centers[-1], y + 0.258, "Gradient All-Reduce", C["ar"])
    notes(
        ax, x, y, w,
        "batch only; complete model per GPU",
        "bucketed gradient All-Reduce",
        "adopt: bucket view + static graph",
        accent=color,
    )


def timeline_block(ax, x, y, w, h, label, fc, ec):
    rounded(ax, x, y, w, h, fc=fc, ec=ec, lw=0.7, radius=0.003, z=4)
    text(ax, x + w / 2, y + h / 2, label, size=5.0, weight="bold", color=ec)


def draw_overlap(ax, x, y, w, h):
    color, light = C["overlap"], C["overlap_light"]
    card_frame(ax, x, y, w, h, "Overlap Grad Reduce", "launch each bucket when gradients are ready",
               color=color, light=light, verdict="IN DDP", verdict_kind="builtin")
    left, right = x + 0.040, x + w - 0.014
    text(ax, x + 0.012, y + 0.235, "Backward", size=5.5, weight="bold", color=color, ha="left")
    text(ax, x + 0.012, y + 0.170, "Comm", size=5.5, weight="bold", color=C["ar"], ha="left")
    bw = (right - left - 0.009) / 4
    for index in range(4):
        xx = left + index * (bw + 0.003)
        timeline_block(ax, xx, y + 0.218, bw, 0.036, f"B{4-index}",
                       C["overlap_light"], color)
    comm_specs = [
        (left + bw * 0.72, bw * 1.18, "AR4"),
        (left + (bw + 0.003) * 1.65, bw * 1.18, "AR3"),
        (left + (bw + 0.003) * 2.58, bw * 1.18, "AR2"),
    ]
    for xx, ww, label in comm_specs:
        timeline_block(ax, xx, y + 0.153, ww, 0.034, label, "#E3EDFA", C["ar"])
    arrow(ax, (left, y + 0.205), (right, y + 0.205), color=C["muted"], lw=0.8, ms=7)
    text(ax, right, y + 0.267, "time", size=5.0, color=C["muted"], ha="right", style="italic")
    notes(
        ax, x, y, w,
        "same DDP model and batch layout",
        "async bucket All-Reduce vs backward",
        "DDP reducer already provides it",
        accent=color,
    )


def draw_distributed_optimizer(ax, x, y, w, h):
    color, light = C["state"], C["state_light"]
    card_frame(ax, x, y, w, h, "Distributed Optimizer", "Megatron optimizer-state sharding",
               color=color, light=light, verdict="NOT NOW", verdict_kind="skip")
    xs, gpu_w = four_gpu_centers(x, w)
    for index, gx in enumerate(xs):
        gpu(
            ax, gx, y + 0.118, gpu_w, 0.100, f"GPU {index}",
            [("P full", C["model_light"], C["model"]),
             (f"G{index}", "#F8E1E7", C["grad"]),
             (f"O{index}", "#F7ECD4", C["optim"])],
            edge=color,
        )
    centers = [gx + gpu_w / 2 for gx in xs]
    arrow(ax, (centers[0], y + 0.267), (centers[-1], y + 0.267),
          color=C["rsag"], lw=1.0, ms=7)
    text(ax, (centers[0] + centers[-1]) / 2, y + 0.281,
         "Reduce-Scatter G", size=5.2, weight="bold", color=C["rsag"])
    arrow(ax, (centers[-1], y + 0.240), (centers[0], y + 0.240),
          color=C["state"], lw=1.0, ms=7)
    text(ax, (centers[0] + centers[-1]) / 2, y + 0.252,
         "All-Gather P", size=5.2, weight="bold", color=C["state"])
    notes(
        ax, x, y, w,
        "optimizer and gradient shards",
        "Reduce-Scatter + parameter All-Gather",
        "4.11 M params: little memory saved",
        accent=color,
    )


def draw_fsdp(ax, x, y, w, h):
    color, light = C["param"], "#ECE8F5"
    card_frame(ax, x, y, w, h, "ZeRO / FSDP", "shard P, G and O by stage",
               color=color, light=light, verdict="NOT NOW", verdict_kind="skip")
    xs, gpu_w = four_gpu_centers(x, w)
    for index, gx in enumerate(xs):
        gpu(
            ax, gx, y + 0.128, gpu_w, 0.105, f"GPU {index}",
            [(f"P{index}", "#E9E3F5", C["param"]),
             (f"G{index}", "#F8E1E7", C["grad"]),
             (f"O{index}", "#F7ECD4", C["optim"])],
            edge=color,
        )
    centers = [gx + gpu_w / 2 for gx in xs]
    rounded(ax, x + 0.040, y + 0.260, w - 0.080, 0.030,
            fc="#F5F2FA", ec=color, lw=0.8, radius=0.004, z=4)
    text(ax, x + w / 2, y + 0.275, "Layer P: All-Gather before compute", size=5.3,
         weight="bold", color=color)
    for center in centers:
        arrow(ax, (center, y + 0.260), (center, y + 0.233),
              color=C["rsag"], lw=0.8, ms=6)
    notes(
        ax, x, y, w,
        "parameters / gradients / optimizer",
        "per-layer All-Gather + Reduce-Scatter",
        "activation peak dominates; extra collectives",
        accent=color,
    )


def draw_balance(ax, x, y, w, h):
    color, light = C["balance"], C["balance_light"]
    card_frame(ax, x, y, w, h, "Cost-Aware Data Shard", "equalize predicted work on every rank",
               color=color, light=light, verdict="PRINCIPLE", verdict_kind="adopt")
    xs, bin_w = four_gpu_centers(x, w)
    schemes = [
        [(0.038, C["sample_a"], "A"), (0.024, C["sample_b"], "B"), (0.018, C["sample_c"], "C")],
        [(0.030, C["sample_d"], "D"), (0.028, C["sample_b"], "B"), (0.022, C["sample_c"], "C")],
        [(0.042, C["sample_b"], "B"), (0.020, C["sample_a"], "A"), (0.018, C["sample_c"], "C")],
        [(0.034, C["sample_c"], "C"), (0.026, C["sample_d"], "D"), (0.020, C["sample_a"], "A")],
    ]
    base_y = y + 0.140
    for index, (gx, scheme) in enumerate(zip(xs, schemes)):
        rounded(ax, gx, base_y, bin_w, 0.105, fc="#FBFCFB", ec=color, lw=0.8,
                radius=0.004, z=4)
        yy = base_y + 0.008
        for block_h, fc, label in scheme:
            rounded(ax, gx + 0.005, yy, bin_w - 0.010, block_h,
                    fc=fc + "33", ec=fc, lw=0.55, radius=0.002, z=5)
            text(ax, gx + bin_w / 2, yy + block_h / 2, label, size=4.8,
                 weight="bold", color=fc)
            yy += block_h + 0.002
        text(ax, gx + bin_w / 2, base_y - 0.013, f"rank {index}", size=4.8,
             weight="bold", color=color)
    line(ax, [(xs[0], y + 0.262), (xs[-1] + bin_w, y + 0.262)],
         color=color, lw=1.0, linestyle=(0, (2, 1)), z=3)
    text(ax, x + w / 2, y + 0.278, "equal total predicted cost", size=5.4,
         weight="bold", color=color)
    notes(
        ax, x, y, w,
        "samples; fixed model on every rank",
        "ordinary DDP after balanced assignment",
        "use only with measured cost heterogeneity",
        accent=color,
    )


def draw_tp(ax, x, y, w, h):
    color, light = C["tp"], C["tp_light"]
    card_frame(ax, x, y, w, h, "Tensor Parallel (TP)", "split matrix dimensions within each layer",
               color=color, light=light, verdict="NOT NOW", verdict_kind="skip")
    matrix_x, matrix_y, matrix_w, matrix_h = x + 0.016, y + 0.128, 0.064, 0.125
    shard_colors = ["#DCCCEF", "#CBB4E4", "#B89BD8", "#A680CC"]
    stripe_h = matrix_h / 4
    for index in range(4):
        yy = matrix_y + (3 - index) * stripe_h
        ax.add_patch(Rectangle((matrix_x, yy), matrix_w, stripe_h,
                               facecolor=shard_colors[index], edgecolor=color,
                               linewidth=0.55, zorder=4))
        text(ax, matrix_x + matrix_w / 2, yy + stripe_h / 2, f"W{index}",
             size=5.0, weight="bold", color=color)
    text(ax, matrix_x + matrix_w / 2, matrix_y + matrix_h + 0.015,
         "Weight matrix W", size=5.3, weight="bold", color=color)
    gpu_x, gpu_w = x + 0.122, 0.050
    for index in range(4):
        yy = matrix_y + (3 - index) * stripe_h + 0.003
        rounded(ax, gpu_x, yy, gpu_w, stripe_h - 0.006,
                fc=C["tp_light"], ec=color, lw=0.75, radius=0.003, z=4)
        text(ax, gpu_x + gpu_w / 2, yy + (stripe_h - 0.006) / 2,
             f"GPU {index}", size=4.9, weight="bold", color=color)
        arrow(ax, (matrix_x + matrix_w, yy + (stripe_h - 0.006) / 2),
              (gpu_x, yy + (stripe_h - 0.006) / 2),
              color=color, lw=0.75, ms=6)
    text(ax, x + w / 2, y + 0.114, "collective after layer GEMM",
         size=5.2, weight="bold", color=C["ar"])
    notes(
        ax, x, y, w,
        "rows / columns of every large tensor",
        "All-Reduce or All-Gather each layer",
        "C=96 matrices too small for TP",
        accent=color,
    )


def draw_pp(ax, x, y, w, h):
    color, light = C["pp"], C["pp_light"]
    card_frame(ax, x, y, w, h, "Pipeline Parallel (PP)", "place consecutive layer stages on GPUs",
               color=color, light=light, verdict="NOT NOW", verdict_kind="skip")
    stage_w = 0.034
    left = x + 0.014
    gap = (w - 0.028 - 4 * stage_w) / 3
    stage_xs = [left + i * (stage_w + gap) for i in range(4)]
    for index, sx in enumerate(stage_xs):
        rounded(ax, sx, y + 0.205, stage_w, 0.052,
                fc=C["pp_light"], ec=color, lw=0.85, radius=0.004, z=4)
        text(ax, sx + stage_w / 2, y + 0.231, f"GPU {index}\nStage {index}",
             size=4.8, weight="bold", color=color)
        if index < 3:
            arrow(
                ax,
                (sx + stage_w, y + 0.231),
                (stage_xs[index + 1], y + 0.231),
                color=C["p2p"],
                lw=0.9,
                ms=7,
                shrink_a=4.0,
                shrink_b=4.0,
                z=3,
            )
    mb_colors = [C["sample_a"], C["sample_b"], C["sample_c"]]
    for row, mb_color in enumerate(mb_colors):
        yy = y + 0.130 + row * 0.022
        for stage in range(4):
            xx = stage_xs[stage] + row * 0.004
            rounded(ax, xx, yy, stage_w - 0.004, 0.016,
                    fc=mb_color + "33", ec=mb_color, lw=0.55, radius=0.002, z=4)
            text(ax, xx + (stage_w - 0.004) / 2, yy + 0.008, f"m{row}",
                 size=4.2, weight="bold", color=mb_color)
    text(ax, x + w / 2, y + 0.116, "P2P activations; micro-batch pipeline",
         size=5.2, weight="bold", color=C["p2p"])
    notes(
        ax, x, y, w,
        "contiguous layer stages",
        "P2P activations and gradients",
        "shallow backbone; bubble dominates",
        accent=color,
    )


def draw_ep(ax, x, y, w, h):
    color, light = C["ep"], C["ep_light"]
    card_frame(ax, x, y, w, h, "Expert Parallel (EP)", "place different experts on different GPUs",
               color=color, light=light, verdict="NOT NOW", verdict_kind="skip")
    xs, box_w = four_gpu_centers(x, w)
    token_colors = [C["sample_a"], C["sample_b"], C["sample_c"], C["sample_d"]]
    token_y = y + 0.245
    expert_y = y + 0.125
    for index, xx in enumerate(xs):
        rounded(ax, xx, token_y, box_w, 0.028, fc=token_colors[index] + "33",
                ec=token_colors[index], lw=0.65, radius=0.003, z=5)
        text(ax, xx + box_w / 2, token_y + 0.014, f"T{index}",
             size=4.8, weight="bold", color=token_colors[index])
        gpu(
            ax, xx, expert_y, box_w, 0.050, f"GPU {index}",
            [(f"E{index*3+1}-E{index*3+3}", C["ep_light"], color)],
            edge=color,
        )
    rounded(ax, x + 0.020, y + 0.195, w - 0.040, 0.034,
            fc="#FAE8EC", ec=C["a2a"], lw=0.8, radius=0.004, z=5)
    text(ax, x + w / 2, y + 0.212, "All-to-All token dispatch", size=5.3,
         weight="bold", color=C["a2a"])
    for index, xx in enumerate(xs):
        center = xx + box_w / 2
        arrow(
            ax,
            (center, token_y),
            (center, y + 0.229),
            color=token_colors[index],
            lw=0.7,
            ms=6,
            z=3,
        )
        arrow(
            ax,
            (center, y + 0.195),
            (center, expert_y + 0.050),
            color=token_colors[(index + 2) % 4],
            lw=0.7,
            ms=6,
            z=3,
        )
    notes(
        ax, x, y, w,
        "expert weights and routed tokens",
        "All-to-All dispatch + combine",
        "12 small experts already fit locally",
        accent=color,
    )


def draw_up(ax, x, y, w, h):
    color, light = C["up"], C["up_light"]
    card_frame(ax, x, y, w, h, "Ulysses Parallelism (UP)", "swap sequence shards for head shards",
               color=color, light=light, verdict="NOT NOW", verdict_kind="skip")
    xs, box_w = four_gpu_centers(x, w)
    top_y, bottom_y = y + 0.232, y + 0.128
    for index, xx in enumerate(xs):
        rounded(ax, xx, top_y, box_w, 0.035, fc="#E5F3EA",
                ec=C["activation"], lw=0.7, radius=0.003, z=5)
        text(ax, xx + box_w / 2, top_y + 0.0175, "S/4 x H",
             size=4.7, weight="bold", color=C["activation"])
        rounded(ax, xx, bottom_y, box_w, 0.039, fc=C["up_light"],
                ec=color, lw=0.7, radius=0.003, z=5)
        text(ax, xx + box_w / 2, bottom_y + 0.0195, "S x H/4",
             size=4.7, weight="bold", color=color)
    rounded(ax, x + 0.035, y + 0.180, w - 0.070, 0.034,
            fc="#FAE8EC", ec=C["a2a"], lw=0.8, radius=0.004, z=5)
    text(ax, x + w / 2, y + 0.197, "All-to-All: sequence -> heads",
         size=5.2, weight="bold", color=C["a2a"])
    for xx in xs:
        arrow(ax, (xx + box_w / 2, top_y),
              (xx + box_w / 2, y + 0.214),
              color=C["a2a"], lw=0.7, ms=6)
        arrow(ax, (xx + box_w / 2, y + 0.180),
              (xx + box_w / 2, bottom_y + 0.039),
              color=C["a2a"], lw=0.7, ms=6)
    notes(
        ax, x, y, w,
        "sequence first, then attention heads",
        "two All-to-All operations per layer",
        "window sequence is only 64 tokens",
        accent=color,
    )


def draw_cp(ax, x, y, w, h):
    color, light = C["cp"], C["cp_light"]
    card_frame(ax, x, y, w, h, "Context Parallel (CP)", "keep local Q; rotate K/V context blocks",
               color=color, light=light, verdict="NOT NOW", verdict_kind="skip")
    box_w, box_h = 0.049, 0.055
    positions = [
        (x + 0.026, y + 0.215),
        (x + w - 0.026 - box_w, y + 0.215),
        (x + w - 0.026 - box_w, y + 0.125),
        (x + 0.026, y + 0.125),
    ]
    for index, (xx, yy) in enumerate(positions):
        rounded(ax, xx, yy, box_w, box_h, fc=C["cp_light"], ec=color,
                lw=0.8, radius=0.004, z=5)
        text(ax, xx + box_w / 2, yy + box_h / 2,
             f"GPU {index}\nQ{index} + KV{index}", size=4.6, weight="bold", color=color)
    margin = 0.008
    arrow(ax, (positions[0][0] + box_w, positions[0][1] + box_h + margin),
          (positions[1][0], positions[1][1] + box_h + margin),
          color=C["ring"], lw=0.9, ms=7)
    arrow(ax, (positions[1][0] + box_w + margin, positions[1][1]),
          (positions[2][0] + box_w + margin, positions[2][1] + box_h),
          color=C["ring"], lw=0.9, ms=7)
    arrow(ax, (positions[2][0], positions[2][1] - margin),
          (positions[3][0] + box_w, positions[3][1] - margin),
          color=C["ring"], lw=0.9, ms=7)
    arrow(ax, (positions[3][0] - margin, positions[3][1] + box_h),
          (positions[0][0] - margin, positions[0][1]),
          color=C["ring"], lw=0.9, ms=7)
    text(
        ax,
        x + w / 2,
        y + 0.195,
        "Ring K/V\nonline softmax",
        size=5.2,
        weight="bold",
        color=color,
        bbox={"facecolor": C["card"], "edgecolor": "none", "pad": 0.8},
    )
    notes(
        ax, x, y, w,
        "sequence context and K/V blocks",
        "ring P2P K/V exchange",
        "local 8 x 8 windows need no CP",
        accent=color,
    )


def legend_line(ax, x, y, label, color, style="solid"):
    arrow(ax, (x, y), (x + 0.036, y), color=color, lw=1.4, ms=7, linestyle=style)
    text(ax, x + 0.041, y, label, size=5.8, color=C["ink"], ha="left")


def summary_strip(ax):
    rounded(ax, 0.012, 0.014, 0.976, 0.148, fc="#F6F8FA", ec="#BCC8D1",
            lw=1.0, radius=0.010, z=1)
    text(ax, 0.026, 0.144, "Communication Legend", size=6.5, weight="bold",
         color=C["muted"], ha="left")
    legend_specs = [
        (0.132, "All-Reduce", C["ar"]),
        (0.282, "Reduce-Scatter / All-Gather", C["rsag"]),
        (0.510, "All-to-All", C["a2a"]),
        (0.650, "P2P", C["p2p"]),
        (0.755, "Ring P2P", C["ring"]),
    ]
    for xx, label, color in legend_specs:
        legend_line(ax, xx, 0.143, label, color)
    box_y, box_h = 0.032, 0.076
    specs = [
        (
            0.024,
            0.280,
            "Measured Profile",
            "4.11 M parameters; < 0.1 GB model states\n61-76 GB peak is activation / loss / MoE capacity",
            C["state_light"],
            C["state"],
        ),
        (
            0.360,
            0.310,
            "Recommended for U-MoE-Fusion",
            "DDP + bucket view + static graph + 8 MiB buckets + fused Adam\nCost-aware data assignment only after measured heterogeneity",
            C["adopt_light"],
            C["adopt"],
        ),
        (
            0.726,
            0.250,
            "Why Other Axes Wait",
            "No parameter, expert-capacity, depth, or long-context bottleneck\nExtra collectives would target the wrong resource",
            C["skip_light"],
            C["skip"],
        ),
    ]
    for xx, ww, title, body, fc, ec in specs:
        rounded(ax, xx, box_y, ww, box_h, fc=fc, ec=ec, lw=0.9, radius=0.007, z=3)
        text(ax, xx + 0.010, box_y + box_h - 0.021, title, size=6.4,
             weight="bold", color=ec, ha="left")
        text(ax, xx + 0.010, box_y + 0.027, body, size=5.7,
             color=C["ink"], ha="left", linespacing=1.15)
    arrow(
        ax,
        (0.304, box_y + box_h / 2),
        (0.360, box_y + box_h / 2),
        color=C["adopt"],
        lw=1.1,
        ms=9,
        shrink_a=5.0,
        shrink_b=5.0,
        z=2,
    )
    arrow(
        ax,
        (0.670, box_y + box_h / 2),
        (0.726, box_y + box_h / 2),
        color=C["skip"],
        lw=1.1,
        ms=9,
        shrink_a=5.0,
        shrink_b=5.0,
        z=2,
    )


def build_figure(output_dir: Path, font_dir: Path) -> Path:
    setup_style(font_dir)
    fig = plt.figure(figsize=(16.0, 24.0))
    fig.patch.set_facecolor(C["canvas"])
    fig.text(
        0.5,
        0.982,
        "Distributed Training Mechanisms: What Is Sharded and How It Communicates",
        fontsize=18,
        fontweight="bold",
        color=C["ink"],
        ha="center",
        va="center",
    )
    fig.text(
        0.5,
        0.958,
        "Mechanisms from Table 4-21 plus TP, PP, Ulysses Parallelism and Context Parallelism",
        fontsize=11,
        color=C["muted"],
        ha="center",
        va="center",
    )

    card_w, card_h = 0.188, 0.365
    draw_cards = [
        draw_ddp,
        draw_overlap,
        draw_distributed_optimizer,
        draw_fsdp,
        draw_balance,
        draw_tp,
        draw_pp,
        draw_ep,
        draw_up,
        draw_cp,
    ]
    row_bottoms = [0.790, 0.635, 0.480, 0.325, 0.170]
    column_lefts = [0.020, 0.510]
    for index, draw_card in enumerate(draw_cards):
        row, column = divmod(index, 2)
        card_ax = fig.add_axes([column_lefts[column], row_bottoms[row], 0.470, 0.145])
        card_ax.set_xlim(-0.004, card_w + 0.004)
        card_ax.set_ylim(-0.004, card_h + 0.004)
        card_ax.axis("off")
        draw_card(card_ax, 0.0, 0.0, card_w, card_h)

    summary_ax = fig.add_axes([0.020, 0.015, 0.960, 0.140])
    summary_ax.set_xlim(0, 1)
    summary_ax.set_ylim(0, 0.17)
    summary_ax.axis("off")
    summary_strip(summary_ax)

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "distributed_parallelism_comparison"
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
