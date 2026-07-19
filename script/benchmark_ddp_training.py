#!/usr/bin/env python3
"""DDP benchmark for bucket tuning, overlap, rank skew, and static-graph ablation.

Run with torchrun. Synthetic inputs isolate model/optimizer/communication costs from
storage and DataLoader noise while preserving the thesis model and full loss.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--code-root", type=Path,
                   default=Path("/ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper"))
    p.add_argument("--output", type=Path)
    p.add_argument("--checkpoint", type=Path,
                   help="optional trained state_dict, relative to --code-root")
    p.add_argument("--seed", type=int, default=20260719)
    p.add_argument("--bs", type=int, default=10)
    p.add_argument("--patch", type=int, default=170)
    p.add_argument("--oc", type=int, default=96)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--nr", type=int, default=12)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--combine", choices=("sparse", "grouped"), default="grouped")
    p.add_argument("--cap-factor", type=float, default=1.5)
    p.add_argument("--attn", choices=("vanilla", "sdpa"), default="sdpa")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"),
                   default="default")
    p.add_argument("--bucket-cap", type=float, default=25.0)
    p.add_argument("--grad-bucket-view", action="store_true")
    p.add_argument("--find-unused", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--static-graph", action="store_true")
    p.add_argument("--fused-adam", action="store_true")
    p.add_argument("--comm-mode", choices=("default", "sync", "noop"), default="default",
                   help="sync serializes all-reduce; noop estimates compute-only time")
    p.add_argument("--delay-rank", type=int, default=0,
                   help="rank receiving an artificial pre-forward input stall")
    p.add_argument("--rank-delay-ms", type=float, default=0.0,
                   help="artificial input stall used for straggler sensitivity")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--steps", type=int, default=30)
    return p.parse_args()


def completed_future(tensor):
    import torch
    future = torch.futures.Future()
    future.set_result(tensor)
    return future


def main() -> None:
    a = parse_args()
    if a.static_graph and a.find_unused:
        raise ValueError("--static-graph requires --no-find-unused for this benchmark")
    sys.path.insert(0, str(a.code_root.resolve()))

    import torch
    import torch.distributed as dist
    import torch.optim as optim
    from torch.nn.parallel import DistributedDataParallel as DDP
    from Networks.net_moe import MODEL_MoE, MoEFFN
    from losses import RMI_ir, RMI_vi, joint_grad, ssim_loss

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)

    raw_model = MODEL_MoE(
        in_channel=2, n_tasks=3, out_channel=a.oc, depth=a.depth,
        window_size=8, n_routed=a.nr, k=a.k, n_shared=1, out_scale=True,
        fusion_head="blend", res_scale=0.0, attn_impl=a.attn,
    ).to(device)
    raw_model.set_combine(a.combine, cap_factor=a.cap_factor)
    checkpoint = None
    if a.checkpoint:
        checkpoint = a.checkpoint if a.checkpoint.is_absolute() else a.code_root / a.checkpoint
        raw_model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model = torch.compile(raw_model, mode=a.compile_mode) if a.compile else raw_model
    ddp = DDP(
        model, device_ids=[local_rank],
        find_unused_parameters=a.find_unused,
        bucket_cap_mb=a.bucket_cap,
        gradient_as_bucket_view=a.grad_bucket_view,
        static_graph=a.static_graph,
    )

    if a.comm_mode == "sync":
        def sync_hook(_state, bucket):
            tensor = bucket.buffer()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=False)
            tensor.div_(world)
            return completed_future(tensor)
        ddp.register_comm_hook(None, sync_hook)
    elif a.comm_mode == "noop":
        def noop_hook(_state, bucket):
            return completed_future(bucket.buffer())
        ddp.register_comm_hook(None, noop_hook)

    optimizer = optim.Adam(ddp.parameters(), lr=a.lr, fused=a.fused_adam)
    generator = torch.Generator(device=device).manual_seed(a.seed + 1009 * rank)
    x = torch.rand(a.bs, 2, a.patch, a.patch, generator=generator, device=device)
    task_id = ((torch.arange(a.bs, device=device) + rank) % 3).long()
    src_a, src_b = x[:, :1], x[:, 1:]

    def train_step() -> float:
        if rank == a.delay_rank and a.rank_delay_ms:
            time.sleep(a.rank_delay_ms / 1000.0)
        optimizer.zero_grad(set_to_none=True)
        out, aux = ddp(x, task_id)
        loss = (
            ssim_loss(out, src_a, src_b)
            + RMI_ir(out, src_a)
            + RMI_vi(out, src_b)
            + torch.mean((torch.maximum(src_a, src_b) - out) ** 2)
            + joint_grad(src_a, src_b, out).mean()
            + 0.01 * aux
        )
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    torch.cuda.reset_peak_memory_stats(device)
    warmup_start = time.perf_counter()
    for _ in range(a.warmup):
        train_step()
    torch.cuda.synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_start

    critical_ms = []
    rank_mean_ms = []
    rank_cv = []
    rank_gap_ms = []
    last_loss = None
    for _ in range(a.steps):
        dist.barrier()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        last_loss = train_step()
        torch.cuda.synchronize(device)
        local_ms = (time.perf_counter() - start) * 1000.0
        local_tensor = torch.tensor([local_ms], dtype=torch.float64, device=device)
        gathered = [torch.empty_like(local_tensor) for _ in range(world)]
        dist.all_gather(gathered, local_tensor)
        if rank == 0:
            values = np.array([float(item.item()) for item in gathered])
            critical_ms.append(float(values.max()))
            rank_mean_ms.append(float(values.mean()))
            rank_cv.append(float(values.std() / max(values.mean(), 1e-12)))
            rank_gap_ms.append(float(values.max() - values.mean()))

    # Routing telemetry is collected after timing on the uncompiled raw model so
    # it cannot perturb the measured graph or communication schedule.
    route_rows = []
    def route_hook(module: MoEFFN, args: tuple) -> None:
        tokens, task_emb = args
        batch, count, channels = tokens.shape
        flat = tokens.reshape(-1, channels)
        gate_in = flat
        if module.task_cond and task_emb is not None:
            gate_in = flat + task_emb[:, None, :].expand(
                batch, count, channels).reshape(-1, channels)
        logits = module.gate(gate_in)
        if module.routing == "deepseek":
            topi = (torch.sigmoid(logits) + module.ebias).topk(module.k, -1).indices
        else:
            topi = torch.softmax(logits, -1).topk(module.k, -1).indices
        loads = torch.bincount(topi.reshape(-1), minlength=module.n_routed)
        loads = loads.detach().cpu().numpy().astype(np.int64)
        dispatches = int(topi.numel())
        capacity = max(1, int(a.cap_factor * dispatches / module.n_routed))
        dropped = int(np.maximum(loads - capacity, 0).sum()) if a.combine == "grouped" else 0
        route_rows.append({
            "dispatches": dispatches,
            "dropped": dropped,
            "load_cv": float(loads.std() / max(loads.mean(), 1e-12)),
            "max_over_mean": float(loads.max() / max(loads.mean(), 1e-12)),
        })

    hooks = [m.register_forward_pre_hook(route_hook) for m in raw_model.modules()
             if isinstance(m, MoEFFN)]
    raw_model.eval()
    with torch.no_grad():
        raw_model(x, task_id)
    for hook in hooks:
        hook.remove()

    route_summary = {
        "dispatches": sum(row["dispatches"] for row in route_rows),
        "dropped": sum(row["dropped"] for row in route_rows),
        "load_cv_mean": statistics.fmean(row["load_cv"] for row in route_rows),
        "max_over_mean_mean": statistics.fmean(row["max_over_mean"] for row in route_rows),
    }
    route_summary["drop_pct"] = (
        100.0 * route_summary["dropped"] / route_summary["dispatches"])
    gathered_routes = [None for _ in range(world)]
    dist.all_gather_object(gathered_routes, route_summary)

    peak = torch.tensor([torch.cuda.max_memory_allocated(device)],
                        dtype=torch.float64, device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    logging_data = ddp._get_ddp_logging_data()

    if rank == 0:
        mean_ms = statistics.fmean(critical_ms)
        result = {
            "schema": "ddp-training-v1",
            "torch_version": torch.__version__,
            "nccl_version": list(torch.cuda.nccl.version()),
            "gpu": torch.cuda.get_device_name(device),
            "world_size": world,
            "config": {
                "bs_per_gpu": a.bs, "patch": a.patch, "out_channel": a.oc,
                "depth": a.depth, "n_routed": a.nr, "top_k": a.k,
                "combine": a.combine, "cap_factor": a.cap_factor,
                "attn": a.attn, "compile": a.compile,
                "compile_mode": a.compile_mode,
                "bucket_cap_mb": a.bucket_cap,
                "gradient_as_bucket_view": a.grad_bucket_view,
                "find_unused_parameters": a.find_unused,
                "static_graph": a.static_graph,
                "fused_adam": a.fused_adam,
                "comm_mode": a.comm_mode,
                "delay_rank": a.delay_rank,
                "rank_delay_ms": a.rank_delay_ms,
                "checkpoint": str(checkpoint) if checkpoint else None,
            },
            "measurement": {
                "warmup_steps": a.warmup, "steps": a.steps,
                "critical_step_ms": critical_ms,
                "mean_step_ms": mean_ms,
                "std_step_ms": statistics.stdev(critical_ms) if len(critical_ms) > 1 else 0.0,
                "p50_step_ms": float(np.percentile(critical_ms, 50)),
                "p95_step_ms": float(np.percentile(critical_ms, 95)),
                "global_samples_per_second": world * a.bs * 1000.0 / mean_ms,
                "rank_time_cv_mean": statistics.fmean(rank_cv),
                "rank_gap_ms_mean": statistics.fmean(rank_gap_ms),
                "rank_mean_step_ms": statistics.fmean(rank_mean_ms),
                "peak_memory_gb": float(peak.item()) / 1e9,
                "warmup_seconds_including_compile": warmup_seconds,
                "last_loss": last_loss,
            },
            "ddp_logging": {str(k): v for k, v in logging_data.items()
                            if isinstance(v, (str, int, float, bool))},
            "routing_by_rank": gathered_routes,
        }
        text = json.dumps(result, ensure_ascii=False, indent=2)
        if a.output:
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(text + "\n")
        print(text)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
