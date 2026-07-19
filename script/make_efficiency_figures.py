#!/usr/bin/env python3
"""Generate publication-ready principle diagrams and evidence plots for §4.5."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


COLORS = {
    "blue": "#3B82F6", "cyan": "#06B6D4", "green": "#10B981",
    "orange": "#F59E0B", "red": "#EF4444", "purple": "#8B5CF6",
    "gray": "#64748B", "light": "#F1F5F9", "dark": "#0F172A",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path,
                   default=Path("Materials/efficiency/data"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("Materials/efficiency/figures"))
    return p.parse_args()


def save(fig, output_dir: Path, stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def rounded_box(ax, xy, width, height, text, color, fontsize=9,
                face_alpha=0.12, linestyle="-"):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle="round,pad=0.04,rounding_size=0.10",
        edgecolor=color, facecolor=color, alpha=face_alpha,
        linewidth=1.7, linestyle=linestyle,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha="center", va="center", fontsize=fontsize, color=COLORS["dark"])
    return patch


def arrow(ax, start, end, color=None, style="-|>", connectionstyle="arc3"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=12,
        linewidth=1.5, color=color or COLORS["gray"],
        connectionstyle=connectionstyle,
    ))


def compile_fusion(output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.8))
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")

    ax = axes[0]
    ax.set_title("(a) Eager sparse path", fontsize=12, fontweight="bold")
    labels = ("top-k", "nonzero", "gather", "small GEMM × E", "index_add")
    colors = (COLORS["purple"], COLORS["red"], COLORS["orange"],
              COLORS["red"], COLORS["orange"])
    for i, (label, color) in enumerate(zip(labels, colors)):
        x = 0.25 + i * 1.93
        rounded_box(ax, (x, 3.0), 1.55, 0.9, label, color, fontsize=8)
        ax.text(x + 0.78, 2.55, "kernel launch", ha="center", va="center",
                fontsize=7, color=COLORS["red"])
        if i:
            arrow(ax, (x - 0.38, 3.45), (x, 3.45))
    ax.text(5, 5.6, "dynamic shapes split the graph", ha="center",
            fontsize=10, color=COLORS["red"])
    ax.text(5, 1.1, "many Python-dispatched kernels; launch and indexing bound",
            ha="center", fontsize=9, color=COLORS["gray"])

    ax = axes[1]
    ax.set_title("(b) Grouped capacity + torch.compile", fontsize=12,
                 fontweight="bold")
    rounded_box(ax, (0.5, 3.0), 2.1, 0.9, "sort + capacity", COLORS["cyan"])
    rounded_box(ax, (3.35, 3.0), 3.0, 0.9, "2 batched GEMMs", COLORS["green"])
    rounded_box(ax, (7.1, 3.0), 2.1, 0.9, "weighted scatter", COLORS["purple"])
    arrow(ax, (2.6, 3.45), (3.35, 3.45))
    arrow(ax, (6.35, 3.45), (7.1, 3.45))
    ax.add_patch(Rectangle((3.05, 2.55), 3.6, 1.8, fill=False,
                           edgecolor=COLORS["green"], linewidth=2,
                           linestyle="--"))
    ax.text(4.85, 4.75, "regular fixed-capacity region", ha="center",
            fontsize=9, color=COLORS["green"], fontweight="bold")
    ax.text(5, 1.1,
            "compile fuses surrounding pointwise work; dynamic boundaries remain",
            ha="center", fontsize=9, color=COLORS["gray"])
    fig.suptitle("torch.compile gains require a compiler-friendly execution shape",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save(fig, output_dir, "compile_fusion_principle")


def sdpa_attention(output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.0))
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")

    ax = axes[0]
    ax.set_title("(a) Hand-written window attention", fontsize=12,
                 fontweight="bold")
    labels = ("QKᵀ", "+ bias", "softmax", "P·V")
    colors = (COLORS["blue"], COLORS["orange"], COLORS["red"], COLORS["blue"])
    for i, (label, color) in enumerate(zip(labels, colors)):
        x = 0.45 + i * 2.35
        rounded_box(ax, (x, 4.5), 1.7, 0.95, label, color)
        if i:
            arrow(ax, (x - 0.65, 4.98), (x, 4.98))
    rounded_box(ax, (2.25, 1.75), 5.5, 1.05,
                "materialized attention matrix [B, heads, N, N]",
                COLORS["red"], fontsize=9)
    arrow(ax, (5.0, 4.5), (5.0, 2.8), color=COLORS["red"])
    ax.text(5, 6.45, "multiple kernels + global-memory round trips",
            ha="center", fontsize=10, color=COLORS["red"])

    ax = axes[1]
    ax.set_title("(b) SDPA / Flash-compatible path", fontsize=12,
                 fontweight="bold")
    rounded_box(ax, (0.55, 4.5), 2.0, 0.95, "Q, K, V + bias", COLORS["blue"])
    rounded_box(ax, (3.25, 3.65), 3.5, 2.5,
                "scaled_dot_product_attention\n\n tiled QKᵀ → online softmax → PV",
                COLORS["green"], fontsize=10)
    rounded_box(ax, (7.45, 4.5), 1.9, 0.95, "output", COLORS["blue"])
    arrow(ax, (2.55, 4.98), (3.25, 4.98))
    arrow(ax, (6.75, 4.98), (7.45, 4.98))
    ax.text(5, 2.25, "N×N intermediates stay tiled/on-chip",
            ha="center", fontsize=10, color=COLORS["green"],
            fontweight="bold")
    ax.text(5, 1.55, "same weights and attention equation; lower activation memory",
            ha="center", fontsize=9, color=COLORS["gray"])
    fig.suptitle("SDPA removes attention intermediates without changing the model",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save(fig, output_dir, "sdpa_principle")


def grouped_pipeline(output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.0))
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    ax = axes[0]
    ax.set_title("(a) Sparse per-expert dispatch: dynamic and launch-bound",
                 fontsize=12, fontweight="bold")
    rounded_box(ax, (0.3, 7.8), 2.0, 1.0, "T tokens\n[T, C]", COLORS["blue"])
    rounded_box(ax, (3.0, 7.8), 2.2, 1.0, "Top-k router\n(T × k pairs)", COLORS["purple"])
    arrow(ax, (2.3, 8.3), (3.0, 8.3))
    for i, n in enumerate(("n1", "n2", "...", "nE")):
        y = 6.5 - i * 1.25
        rounded_box(ax, (3.1, y), 1.9, 0.78, f"expert {i+1}\n[{n}, C]",
                    COLORS["orange"], fontsize=8)
        rounded_box(ax, (6.1, y), 2.6, 0.78, "FC1 → GELU → FC2",
                    COLORS["red"], fontsize=8)
        arrow(ax, (5.0, y + 0.39), (6.1, y + 0.39))
        arrow(ax, (4.1, 7.8), (4.1, y + 0.78), connectionstyle="arc3,rad=0.08")
    rounded_box(ax, (3.8, 0.55), 3.8, 0.9, "E × nonzero / index / index_add",
                COLORS["red"], fontsize=9)
    for i in range(4):
        y = 6.5 - i * 1.25
        arrow(ax, (8.7, y + 0.39), (6.8, 1.45), connectionstyle="arc3,rad=-0.08")
    ax.text(5.0, 9.25, "Sparse FLOPs, but variable shapes + many small GEMMs",
            ha="center", color=COLORS["red"], fontsize=10)

    ax = axes[1]
    ax.set_title("(b) Grouped-capacity dispatch + compile: regular batched GEMM",
                 fontsize=12, fontweight="bold")
    rounded_box(ax, (0.2, 7.8), 1.6, 1.0, "T tokens\n[T, C]", COLORS["blue"])
    rounded_box(ax, (2.2, 7.8), 1.8, 1.0, "Top-k\npairs", COLORS["purple"])
    rounded_box(ax, (4.4, 7.8), 2.0, 1.0, "sort by expert\n+ local offset", COLORS["cyan"])
    rounded_box(ax, (7.0, 7.8), 2.5, 1.0, "capacity mask\ncap = αTk/E", COLORS["orange"])
    for x0, x1 in ((1.8, 2.2), (4.0, 4.4), (6.4, 7.0)):
        arrow(ax, (x0, 8.3), (x1, 8.3))
    rounded_box(ax, (1.0, 5.6), 3.6, 1.25,
                "right-padded expert buffer\n[E, cap, C]", COLORS["cyan"], fontsize=10)
    ax.text(2.8, 5.25, "overflow dispatches dropped", ha="center",
            color=COLORS["red"], fontsize=8)
    rounded_box(ax, (5.4, 5.6), 3.6, 1.25,
                "2 batched GEMMs\n[E, cap, C] → [E, cap, C]",
                COLORS["green"], fontsize=10)
    arrow(ax, (8.25, 7.8), (4.6, 6.25), connectionstyle="arc3,rad=0.08")
    arrow(ax, (4.6, 6.25), (5.4, 6.25))
    rounded_box(ax, (2.8, 2.9), 4.4, 1.05,
                "gate-weighted gather → index_add scatter",
                COLORS["purple"], fontsize=10)
    arrow(ax, (7.2, 5.6), (5.0, 3.95))
    rounded_box(ax, (3.8, 0.9), 2.4, 0.95, "routed output\n[T, C]", COLORS["blue"])
    arrow(ax, (5.0, 2.9), (5.0, 1.85))
    ax.add_patch(Rectangle((0.65, 4.9), 8.8, 2.55, fill=False,
                           edgecolor=COLORS["green"], linewidth=2.0,
                           linestyle="--"))
    ax.text(5.05, 7.1, "regular region accelerated by torch.compile",
            ha="center", color=COLORS["green"], fontsize=9, fontweight="bold")
    ax.text(5.0, 9.25,
            "Work ≈ E·cap = αTk (padding buys fixed shapes)",
            ha="center", color=COLORS["green"], fontsize=10)
    fig.suptitle("Grouped-capacity MoE changes the execution shape, not the top-k routing rule",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, output_dir, "grouped_moe_principle")


def capacity_balance(output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.7), sharey=True)
    loads = np.array([1.62, 1.43, 1.23, 1.08, 0.98, 0.88, 0.79, 0.72])
    specs = [
        (loads, 1.25, "Unbalanced router, α=1.25\nfast, but overflow"),
        (np.array([1.12, 1.08, 1.04, 1.01, 0.99, 0.96, 0.92, 0.88]),
         1.25, "Aux-balanced router, α=1.25\nless drop, same buffer"),
        (loads, 2.0, "Unbalanced router, α=2.0\nno drop, more padding/memory"),
    ]
    for ax, (values, cap, title) in zip(axes, specs):
        x = np.arange(len(values))
        kept = np.minimum(values, cap)
        dropped = np.maximum(values - cap, 0)
        ax.bar(x, kept, color=COLORS["blue"], alpha=0.82, label="processed")
        ax.bar(x, dropped, bottom=kept, color=COLORS["red"], alpha=0.88,
               label="dropped")
        ax.axhline(cap, color=COLORS["orange"], linestyle="--", linewidth=2,
                   label="capacity")
        ax.fill_between([-0.5, 7.5], values.max(), cap,
                        where=np.array([cap, cap]) > values.max(), alpha=0)
        for i, value in enumerate(values):
            if value < cap:
                ax.add_patch(Rectangle((i - 0.4, value), 0.8, cap - value,
                                       facecolor=COLORS["light"], edgecolor=COLORS["gray"],
                                       hatch="//", alpha=0.65))
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("expert id")
        ax.set_xticks(x); ax.set_xticklabels([str(i + 1) for i in x], fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("load / mean expert load")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Capacity factor and load balancing are a coupled supply–demand mechanism",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.01,
             "lower α: less padded compute but more token drops   |   higher α: safer quality but larger buffers",
             ha="center", fontsize=10, color=COLORS["dark"])
    fig.tight_layout(rect=(0, 0.09, 1, 0.92))
    save(fig, output_dir, "capacity_balance_principle")


def ddp_overlap(output_dir: Path):
    fig, axes = plt.subplots(3, 1, figsize=(13.6, 7.2), sharex=True)
    rows = [
        ("25 MB cap > 16 MB model: one bucket", [(0, 7.2, "backward", COLORS["blue"]),
          (7.2, 1.7, "all-reduce", COLORS["red"])], "no useful overlap"),
        ("very small buckets", [(0, 1.5, "BWD", COLORS["blue"]),
          (1.5, .55, "AR", COLORS["red"]), (2.05, 1.4, "BWD", COLORS["blue"]),
          (3.45, .55, "AR", COLORS["red"]), (4.0, 1.4, "BWD", COLORS["blue"]),
          (5.4, .55, "AR", COLORS["red"]), (5.95, 1.3, "BWD", COLORS["blue"]),
          (7.25, .55, "AR", COLORS["red"])], "latency paid many times"),
        ("profile-guided 8 MiB cap", [(0, 3.5, "backward bucket 1", COLORS["blue"]),
          (3.5, 3.4, "backward bucket 2", COLORS["blue"]),
          (3.55, 1.25, "AR 1", COLORS["green"]),
          (6.95, 1.25, "AR 2", COLORS["green"])],
         "two bandwidth-efficient buckets"),
    ]
    for ax, (label, segments, note) in zip(axes, rows):
        ax.set_ylim(0, 1); ax.set_yticks([]); ax.grid(axis="x", alpha=0.18)
        for j, (start, width, text, color) in enumerate(segments):
            y = 0.52 if "AR" in text or "all" in text else 0.12
            height = 0.30
            ax.broken_barh([(start, width)], (y, height), facecolors=color,
                           edgecolors="white", linewidth=1.2, alpha=0.9)
            ax.text(start + width / 2, y + height / 2, text,
                    ha="center", va="center", fontsize=8, color="white")
        ax.text(-0.12, 0.5, label, transform=ax.transAxes, ha="right", va="center",
                fontsize=10, fontweight="bold")
        ax.text(1.01, 0.5, note, transform=ax.transAxes, ha="left", va="center",
                fontsize=9, color=COLORS["gray"])
    axes[-1].set_xlabel("step timeline (normalized)")
    axes[-1].set_xlim(0, 9.2)
    fig.suptitle("DDP bucket co-design: balance collective latency against overlap opportunity",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0.12, 0.02, 0.88, 0.93))
    save(fig, output_dir, "ddp_overlap_principle")


def rank_balance(output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    workloads = ([10, 8, 6, 4], [7, 7, 7, 7])
    titles = ("(a) Count-only/random split", "(b) Cost-aware balanced split")
    notes = ("fast ranks wait at collective", "fixed crop + task/cost stratification")
    colors = (COLORS["red"], COLORS["green"])
    for ax, values, title, note, color in zip(
            axes, workloads, titles, notes, colors):
        ranks = np.arange(4)
        ax.barh(ranks, values, color=color, alpha=0.82)
        maximum = max(values)
        for rank, value in zip(ranks, values):
            if value < maximum:
                ax.barh(rank, maximum - value, left=value, color=COLORS["light"],
                        edgecolor=COLORS["gray"], hatch="//", alpha=0.8)
        ax.axvline(maximum, color=COLORS["dark"], linestyle="--", linewidth=1.2)
        ax.set_yticks(ranks); ax.set_yticklabels([f"rank {i}" for i in ranks])
        ax.set_xlim(0, 11); ax.set_xlabel("per-step work / time (normalized)")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.text(0.5, -0.19, note, transform=ax.transAxes, ha="center",
                fontsize=9, color=COLORS["gray"])
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("The slowest rank defines DDP step time",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    save(fig, output_dir, "ddp_rank_balance_principle")


def load_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def data_plots(data_dir: Path, output_dir: Path):
    # Grouped dispatch and compile interaction.
    core_names = {
        ("sparse", "eager"): "core_sparse_eager.json",
        ("sparse", "compile"): "core_sparse_compile.json",
        ("grouped", "eager"): "core_grouped_eager.json",
        ("grouped", "compile"): "core_grouped_compile.json",
    }
    core = {key: load_json(data_dir / name) for key, name in core_names.items()}
    if all(core.values()):
        labels = ["sparse\neager", "grouped\neager",
                  "sparse\ncompile", "grouped\ncompile"]
        keys = [("sparse", "eager"), ("grouped", "eager"),
                ("sparse", "compile"), ("grouped", "compile")]
        values = [core[key]["measurement"]["mean_ms"] for key in keys]
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        bars = ax.bar(labels, values, color=[COLORS["red"], COLORS["orange"],
                                             COLORS["purple"], COLORS["green"]])
        ax.bar_label(bars, fmt="%.1f ms", padding=3, fontsize=9)
        ax.set_ylabel("training step time (ms, lower is better)")
        ax.set_ylim(0, max(values) * 1.18); ax.grid(axis="y", alpha=0.22)
        ax.set_title("Grouped execution unlocks the compiler gain")
        fig.tight_layout(); save(fig, output_dir, "grouped_compile_synergy")

    # Expert-count crossover.
    experts, sparse, grouped = [], [], []
    for e in (4, 8, 12, 16, 24, 32):
        s = load_json(data_dir / f"experts_sparse_{e}.json")
        g = load_json(data_dir / f"experts_grouped_{e}.json")
        if e == 12 and g is None:
            g = load_json(data_dir / "cap_1.25.json")
        if s and g:
            experts.append(e)
            sparse.append(s["measurement"]["mean_ms"])
            grouped.append(g["measurement"]["mean_ms"])
    if experts:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.plot(experts, sparse, "o-", label="sparse + SDPA + compile",
                color=COLORS["red"], linewidth=2)
        ax.plot(experts, grouped, "o-", label="grouped + SDPA + compile",
                color=COLORS["green"], linewidth=2)
        for e, s, g in zip(experts, sparse, grouped):
            ax.annotate(f"{s/g:.2f}×", (e, g), xytext=(0, -17),
                        textcoords="offset points", ha="center", fontsize=8)
        ax.set_xlabel("number of routed experts E")
        ax.set_ylabel("training step time (ms, lower is better)")
        ax.set_xticks(experts); ax.grid(alpha=0.25); ax.legend(frameon=False)
        ax.set_title("Grouped dispatch crosses over as expert count grows")
        fig.tight_layout(); save(fig, output_dir, "expert_count_scaling")

    # Capacity/drop/quality and feasible throughput.
    quality = load_json(data_dir / "capacity_quality.json")
    perf_files = {
        1.0: "batch_cap1.0_bs12.json", 1.25: "batch_cap1.25_bs11.json",
        1.5: "cap_1.5.json", 2.0: "batch_cap2.0_bs8.json",
        4.0: "batch_cap4.0_bs4.json",
    }
    perf = {cap: load_json(data_dir / name) for cap, name in perf_files.items()}
    if quality and all(perf.values()):
        caps = [1.0, 1.25, 1.5, 2.0, 4.0]
        drops = [quality["overall"][f"cap{c:g}"]["routing"]["drop_pct"] for c in caps]
        mae = [quality["overall"][f"cap{c:g}"]["output_mae_vs_sparse"] * 1e4 for c in caps]
        throughput = [perf[c]["measurement"]["samples_per_second"] for c in caps]
        memory = [perf[c]["measurement"]["peak_memory_gb"] for c in caps]
        batches = [perf[c]["config"]["bs"] for c in caps]
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
        ax = axes[0]
        ax.plot(caps, drops, "o-", color=COLORS["red"], label="dispatch drop (%)")
        ax.set_xlabel("capacity factor α"); ax.set_ylabel("drop rate (%)", color=COLORS["red"])
        ax2 = ax.twinx(); ax2.plot(caps, mae, "s--", color=COLORS["blue"], label="output MAE ×10⁴")
        ax2.set_ylabel("output MAE vs sparse (×10⁴)", color=COLORS["blue"])
        ax.grid(alpha=0.22); ax.set_title("Quality risk falls with capacity")
        ax = axes[1]
        ax.plot(caps, throughput, "o-", color=COLORS["green"], label="samples/s")
        for c, t, b in zip(caps, throughput, batches):
            ax.annotate(f"bs={b}", (c, t), xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=8)
        ax.set_xlabel("capacity factor α"); ax.set_ylabel("max-feasible throughput (samples/s)")
        ax2 = ax.twinx(); ax2.plot(caps, memory, "s--", color=COLORS["orange"])
        ax2.set_ylabel("peak memory (GB)", color=COLORS["orange"])
        ax.grid(alpha=0.22); ax.set_title("Safety costs padding and per-GPU batch")
        fig.tight_layout(); save(fig, output_dir, "capacity_quality_pareto")

    nccl = load_json(data_dir / "nccl_world4.json")
    if nccl:
        x = [r["size_mb"] for r in nccl["rows"]]
        lat = [r["latency_p50_ms"] * 1000 for r in nccl["rows"]]
        bw = [r["bus_bandwidth_gbps"] for r in nccl["rows"]]
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.semilogx(x, lat, "o-", color=COLORS["red"])
        ax.set_xlabel("all-reduce payload / bucket (MiB)")
        ax.set_ylabel("p50 latency (μs)", color=COLORS["red"])
        ax2 = ax.twinx(); ax2.semilogx(x, bw, "s--", color=COLORS["blue"])
        ax2.set_ylabel("bus bandwidth (GB/s)", color=COLORS["blue"])
        ax.grid(alpha=0.25, which="both")
        ax.set_title("4-GPU NVLink all-reduce: bandwidth rises at 4–8 MiB")
        fig.tight_layout(); save(fig, output_dir, "nccl_bucket_curve")

    bucket_paths = sorted(data_dir.glob("ddp_bucket_*.json"))
    rows = []
    for path in bucket_paths:
        data = load_json(path)
        if data and data.get("schema") == "ddp-training-v1":
            rows.append((data["config"]["bucket_cap_mb"], data["measurement"]["mean_step_ms"],
                         data["measurement"]["rank_gap_ms_mean"]))
    if rows:
        rows.sort()
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.plot([r[0] for r in rows], [r[1] for r in rows], "o-", color=COLORS["green"])
        ax.set_xscale("log"); ax.set_xlabel("DDP bucket cap (MiB)")
        ax.set_ylabel("critical-rank step time (ms)")
        ax.grid(alpha=0.25, which="both")
        ax.set_title("Measured bucket-size trade-off on 4×H800")
        fig.tight_layout(); save(fig, output_dir, "ddp_bucket_sweep")

    scale_files = {
        1: "ddp_scale_1gpu.json", 2: "ddp_scale_2gpu.json",
        4: "ddp_comm_default.json", 8: "ddp_scale_8gpu.json",
    }
    scale = {n: load_json(data_dir / name) for n, name in scale_files.items()}
    delays = {
        0: load_json(data_dir / "ddp_final_bucket8.json"),
        2: load_json(data_dir / "ddp_straggler_2ms.json"),
        5: load_json(data_dir / "ddp_straggler_5ms.json"),
        10: load_json(data_dir / "ddp_straggler_10ms.json"),
    }
    if all(scale.values()) and all(delays.values()):
        cards = sorted(scale)
        throughput = [scale[n]["measurement"]["global_samples_per_second"]
                      for n in cards]
        base = throughput[0]
        efficiency = [100 * value / (base * n)
                      for n, value in zip(cards, throughput)]
        injected = sorted(delays)
        step = [delays[d]["measurement"]["mean_step_ms"] for d in injected]
        base_step = step[0]
        penalty = [value - base_step for value in step]
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
        ax = axes[0]
        ax.plot(cards, throughput, "o-", color=COLORS["green"], linewidth=2)
        for n, value, eff in zip(cards, throughput, efficiency):
            ax.annotate(f"{value:.1f}/s\n{eff:.1f}%", (n, value),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8)
        ax.set_xticks(cards); ax.set_xlabel("GPU count")
        ax.set_ylabel("global samples/s"); ax.grid(alpha=0.22)
        ax.set_title("Near-linear DDP scaling")
        ax = axes[1]
        ax.plot(injected, penalty, "o-", color=COLORS["red"], linewidth=2,
                label="measured step penalty")
        ax.plot(injected, injected, "--", color=COLORS["gray"],
                label="1:1 propagation")
        ax.set_xlabel("injected delay on one rank (ms)")
        ax.set_ylabel("critical-step penalty (ms)")
        ax.grid(alpha=0.22); ax.legend(frameon=False)
        ax.set_title("One straggler stalls every rank")
        fig.tight_layout(); save(fig, output_dir, "ddp_scaling_straggler")


def bottleneck_plots(data_dir: Path, output_dir: Path):
    summary = load_json(data_dir / "bottleneck_summary.json")
    bottleneck_dir = data_dir / "bottleneck"
    if not summary:
        return

    # Direct evidence that current DDP is compute/straggler bound.
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    ax = axes[0]
    worlds = [4, 8]
    deltas = [
        summary["paired_communication_deltas"][
            f"w{world}_default_minus_noop_ms"]
        for world in worlds
    ]
    means = [row["mean"] for row in deltas]
    lower = [row["mean"] - row["ci95_low"] for row in deltas]
    upper = [row["ci95_high"] - row["mean"] for row in deltas]
    ax.errorbar(worlds, means, yerr=[lower, upper], fmt="o",
                color=COLORS["blue"], capsize=6, linewidth=2)
    ax.axhline(0, color=COLORS["gray"], linestyle="--", linewidth=1)
    ax.set_xticks(worlds); ax.set_xlabel("GPU count")
    ax.set_ylabel("default − noop step time (ms)")
    ax.set_title("(a) Exposed DDP cost: 95% CI crosses zero")
    ax.grid(alpha=0.22)

    ax = axes[1]
    hook = summary["timed_hook"]["w8"]
    points = [
        ("first bucket ready", hook["first_bucket_ready_ms_mean"]["mean"],
         COLORS["blue"], (-50, 22)),
        ("last bucket ready", hook["last_bucket_ready_ms_mean"]["mean"],
         COLORS["purple"], (-45, 68)),
        ("all comm complete", hook["all_comm_complete_ms_mean"]["mean"],
         COLORS["green"], (42, 22)),
        ("step end", hook["step_end_ms_mean"]["mean"],
         COLORS["orange"], (38, 68)),
    ]
    for label, value, color, offset in points:
        ax.scatter(value, 0, s=90, color=color, zorder=3)
        ax.annotate(f"{label}\n{value:.2f} ms", (value, 0),
                    xytext=offset,
                    textcoords="offset points", ha="center", fontsize=8,
                    arrowprops={"arrowstyle": "-", "color": color})
    tail = hook["comm_tail_after_last_ready_ms_mean"]["mean"]
    ax.plot([points[0][1], points[-1][1]], [0, 0],
            color=COLORS["gray"], linewidth=2)
    ax.text((points[1][1] + points[2][1]) / 2, -0.09,
            f"exposed tail = {tail:.3f} ms", ha="center",
            color=COLORS["green"], fontsize=9, fontweight="bold")
    ax.set_xlim(245, 445); ax.set_ylim(-0.16, 0.28)
    ax.set_yticks([]); ax.set_xlabel("8-GPU step timeline (ms)")
    ax.set_title("(b) 16.43 MB gradients in two buckets")
    ax.grid(axis="x", alpha=0.22)

    ax = axes[2]
    rounds = [0, 1, 4, 16, 64]
    for world, color in ((4, COLORS["green"]), (8, COLORS["red"])):
        values = [
            summary["communication_round_pressure"][f"w{world}_r{repeat}"][
                "mean"]
            for repeat in rounds
        ]
        baseline = values[0]
        ax.plot(rounds, [value - baseline for value in values], "o-",
                color=color, linewidth=2, label=f"{world} GPUs")
    ax.axhline(0, color=COLORS["gray"], linestyle="--", linewidth=1)
    ax.set_xlabel("extra serialized 16 MiB all-reduces / step")
    ax.set_ylabel("step penalty vs 0 rounds (ms)")
    ax.set_title("(c) NCCL becomes visible only under stress")
    ax.grid(alpha=0.22); ax.legend(frameon=False)
    fig.suptitle("Current DDP bottleneck is not NCCL communication",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, output_dir, "ddp_bottleneck_evidence")

    # Physical-card, rank-mapping, and load-partition evidence.
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.0))
    ax = axes[0, 0]
    cards = list(range(8))
    gpu_times = [
        summary["per_gpu"][f"gpu{card}"]["mean"] for card in cards
    ]
    bars = ax.bar(cards, gpu_times,
                  color=[COLORS["red"] if card == 7 else COLORS["blue"]
                         for card in cards], alpha=0.85)
    ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_ylim(min(gpu_times) - 1, max(gpu_times) + 1.5)
    ax.set_xticks(cards); ax.set_xlabel("physical GPU")
    ax.set_ylabel("single-GPU step time (ms)")
    ax.set_title("(a) GPU 7 is consistently the slowest card")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[0, 1]
    mapping_specs = [
        ("normal+affinity", sorted(bottleneck_dir.glob(
            "rankdiag_normal_rank_t*.json")), "o-", COLORS["blue"]),
        ("reversed", sorted(bottleneck_dir.glob(
            "rankdiag_reverse_none_t*.json")), "s--", COLORS["orange"]),
    ]
    for label, paths, style, color in mapping_specs:
        samples = {card: [] for card in cards}
        for path in paths:
            data = load_json(path)
            visible = [
                int(value) for value in
                data["config"]["cuda_visible_devices"].split(",")
            ]
            ready = data["timed_communication_by_rank"]
            median = float(np.median([
                row["first_bucket_ready_ms_mean"] for row in ready
            ]))
            for rank, row in enumerate(ready):
                samples[visible[rank]].append(
                    row["first_bucket_ready_ms_mean"] - median)
        values = [float(np.mean(samples[card])) for card in cards]
        ax.plot(cards, values, style, color=color, linewidth=2, label=label)
    ax.axhline(0, color=COLORS["gray"], linestyle=":", linewidth=1)
    ax.set_xticks(cards); ax.set_xlabel("physical GPU after remapping")
    ax.set_ylabel("first-gradient delay vs run median (ms)")
    ax.set_title("(b) Delay follows physical GPU 7, not rank id")
    ax.grid(alpha=0.22); ax.legend(frameon=False)

    ax = axes[1, 0]
    cost_keys = ["none_20ms", "balanced_20ms", "skewed_20ms"]
    cost_labels = ["no extra cost", "cost-balanced", "all cost on rank 0"]
    cost_values = [
        summary["cost_partition"][key]["mean"] for key in cost_keys
    ]
    bars = ax.bar(cost_labels, cost_values,
                  color=[COLORS["gray"], COLORS["green"], COLORS["red"]],
                  alpha=0.85)
    ax.bar_label(bars, fmt="%.1f ms", fontsize=8, padding=2)
    ax.set_ylabel("DDP-4 critical step time (ms)")
    ax.set_title("(c) Equal-total-cost partition recovers idle time")
    ax.tick_params(axis="x", rotation=10); ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 1]
    worker_rows = [
        ("1 worker", summary["real_data_workers"]["workers1_none"],
         COLORS["blue"]),
        ("4 default", summary["real_data_sampler"]["distributed_workers4"],
         COLORS["red"]),
        ("4 + core\nisolation",
         summary["real_data_workers"]["workers4_isolated"], COLORS["orange"]),
        ("4 + controlled\ntask balance",
         summary["real_data_sampler"]["taskbalanced_workers4"],
         COLORS["green"]),
        ("8 workers", summary["real_data_workers"]["workers8_none"],
         COLORS["cyan"]),
    ]
    worker_labels = [row[0] for row in worker_rows]
    worker_values = [row[1]["mean"] for row in worker_rows]
    worker_errors = [row[1]["std"] for row in worker_rows]
    bars = ax.bar(worker_labels, worker_values, yerr=worker_errors, capsize=4,
                  color=[row[2] for row in worker_rows], alpha=0.85)
    ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
    ax.set_ylabel("real-data critical step time (ms)")
    ax.set_title("(d) Task balance alone is neutral on fixed-shape data")
    ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Straggler diagnosis and controlled mitigation",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, output_dir, "ddp_straggler_diagnosis")


if __name__ == "__main__":
    args = parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})
    compile_fusion(args.output_dir)
    sdpa_attention(args.output_dir)
    grouped_pipeline(args.output_dir)
    capacity_balance(args.output_dir)
    ddp_overlap(args.output_dir)
    rank_balance(args.output_dir)
    data_plots(args.data_dir, args.output_dir)
    bottleneck_plots(args.data_dir, args.output_dir)
