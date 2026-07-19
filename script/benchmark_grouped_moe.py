#!/usr/bin/env python3
"""Reproducible single-GPU benchmark for sparse vs grouped-capacity MoE.

The script imports the thesis implementation without modifying it and records
repeat statistics, peak memory, Dynamo graph counters, and optional operator
profiles as JSON.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--code-root", type=Path,
                   default=Path("/ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper"))
    p.add_argument("--output", type=Path)
    p.add_argument("--checkpoint", type=Path,
                   help="optional trained state_dict, relative to --code-root")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20260719)
    p.add_argument("--bs", type=int, default=10)
    p.add_argument("--patch", type=int, default=170)
    p.add_argument("--oc", type=int, default=96)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--nr", type=int, default=12)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--combine", choices=("sparse", "grouped"), default="sparse")
    p.add_argument("--cap-factor", type=float, default=1.25)
    p.add_argument("--attn", choices=("vanilla", "sdpa"), default="vanilla")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"),
                   default="default")
    p.add_argument("--fused-adam", action="store_true")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--profile-steps", type=int, default=0)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    sys.path.insert(0, str(a.code_root))

    import torch
    import torch.optim as optim
    from Networks.net_moe import MODEL_MoE
    from losses import RMI_ir, RMI_vi, joint_grad, ssim_loss

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)

    try:
        from torch._dynamo.utils import counters
        counters.clear()
    except Exception:
        counters = None

    model = MODEL_MoE(
        in_channel=2, n_tasks=3, out_channel=a.oc, depth=a.depth,
        window_size=8, n_routed=a.nr, k=a.k, n_shared=1, out_scale=True,
        fusion_head="blend", res_scale=0.0, attn_impl=a.attn,
    ).to(a.device)
    model.set_combine(a.combine, cap_factor=a.cap_factor)
    checkpoint = None
    if a.checkpoint:
        checkpoint = a.checkpoint if a.checkpoint.is_absolute() else a.code_root / a.checkpoint
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    if a.compile:
        model = torch.compile(model, mode=a.compile_mode)
    opt = optim.Adam(model.parameters(), lr=1e-3, fused=a.fused_adam)

    x = torch.rand(a.bs, 2, a.patch, a.patch, device=a.device)
    tid = (torch.arange(a.bs, device=a.device) % 3).long()
    src_a, src_b = x[:, :1], x[:, 1:]

    def step() -> float:
        opt.zero_grad(set_to_none=True)
        out, aux = model(x, tid)
        loss = (
            ssim_loss(out, src_a, src_b)
            + RMI_ir(out, src_a)
            + RMI_vi(out, src_b)
            + torch.mean((torch.maximum(src_a, src_b) - out) ** 2)
            + joint_grad(src_a, src_b, out).mean()
            + 0.01 * aux
        )
        loss.backward()
        opt.step()
        return float(loss.detach())

    torch.cuda.reset_peak_memory_stats(a.device)
    compile_t0 = time.perf_counter()
    for _ in range(a.warmup):
        step()
    torch.cuda.synchronize(a.device)
    warmup_seconds = time.perf_counter() - compile_t0

    repeat_ms = []
    last_loss = None
    for _ in range(a.repeats):
        torch.cuda.synchronize(a.device)
        t0 = time.perf_counter()
        for _ in range(a.steps):
            last_loss = step()
        torch.cuda.synchronize(a.device)
        repeat_ms.append((time.perf_counter() - t0) * 1000.0 / a.steps)

    profile_rows = []
    if a.profile_steps:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(a.profile_steps):
                step()
            torch.cuda.synchronize(a.device)
        for event in prof.key_averages():
            device_us = float(getattr(event, "device_time_total", 0.0) or
                              getattr(event, "cuda_time_total", 0.0) or 0.0)
            profile_rows.append({
                "op": event.key,
                "count": int(event.count),
                "device_time_ms": device_us / 1000.0,
                "cpu_time_ms": float(event.cpu_time_total) / 1000.0,
            })
        profile_rows.sort(key=lambda row: row["device_time_ms"], reverse=True)
        profile_rows = profile_rows[:25]

    dynamo = {}
    if counters is not None:
        for group in ("stats", "graph_break", "inductor"):
            values = counters.get(group, {})
            dynamo[group] = {str(k): int(v) for k, v in values.items()}

    mean_ms = statistics.fmean(repeat_ms)
    std_ms = statistics.stdev(repeat_ms) if len(repeat_ms) > 1 else 0.0
    result = {
        "schema": "grouped-moe-single-v1",
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(a.device),
        "seed": a.seed,
        "config": {
            "bs": a.bs, "patch": a.patch, "out_channel": a.oc,
            "depth": a.depth, "n_routed": a.nr, "top_k": a.k,
            "combine": a.combine, "cap_factor": a.cap_factor,
            "attn": a.attn, "compile": a.compile,
            "compile_mode": a.compile_mode, "fused_adam": a.fused_adam,
            "checkpoint": str(checkpoint) if checkpoint else None,
        },
        "measurement": {
            "warmup_steps": a.warmup, "steps_per_repeat": a.steps,
            "repeats": a.repeats, "repeat_ms": repeat_ms,
            "mean_ms": mean_ms, "std_ms": std_ms,
            "samples_per_second": a.bs * 1000.0 / mean_ms,
            "peak_memory_gb": torch.cuda.max_memory_allocated(a.device) / 1e9,
            "warmup_seconds_including_compile": warmup_seconds,
            "last_loss": last_loss,
        },
        "dynamo_counters": dynamo,
        "profile_top_ops": profile_rows,
    }

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
