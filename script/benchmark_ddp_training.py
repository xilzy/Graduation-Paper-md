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
import subprocess
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
    p.add_argument("--compile-order", choices=("before-ddp", "after-ddp"),
                   default="before-ddp",
                   help="after-ddp enables TorchDynamo's bucket-aware DDPOptimizer")
    p.add_argument("--bucket-cap", type=float, default=25.0)
    p.add_argument("--grad-bucket-view", action="store_true")
    p.add_argument("--find-unused", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--static-graph", action="store_true")
    p.add_argument("--fused-adam", action="store_true")
    p.add_argument("--comm-mode", choices=("default", "timed", "sync", "noop"),
                   default="default",
                   help="timed records async all-reduce; sync serializes it; "
                        "noop estimates compute-only time")
    p.add_argument("--task-layout", choices=("balanced", "homogeneous"),
                   default="balanced",
                   help="homogeneous assigns one task per rank to stress routing skew")
    p.add_argument("--delay-rank", type=int, default=0,
                   help="rank receiving an artificial pre-forward input stall")
    p.add_argument("--rank-delay-ms", type=float, default=0.0,
                   help="artificial input stall used for straggler sensitivity")
    p.add_argument("--rank-delay-kind", choices=("input", "gpu"), default="input",
                   help="inject a host/input stall or calibrated GPU work")
    p.add_argument("--cost-layout", choices=("none", "skewed", "balanced"),
                   default="none",
                   help="controlled equal-total-cost partition experiment")
    p.add_argument("--cost-total-ms", type=float, default=0.0,
                   help="global synthetic input cost distributed by --cost-layout")
    p.add_argument("--extra-allreduce-mb", type=float, default=0.0,
                   help="extra serialized all-reduce payload for sensitivity analysis")
    p.add_argument("--extra-allreduce-repeats", type=int, default=1,
                   help="number of serialized collectives for the extra payload")
    p.add_argument("--cpu-affinity", choices=("none", "rank"), default="none",
                   help="pin each rank to disjoint physical cores local to its GPU")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--steps", type=int, default=30)
    return p.parse_args()


def completed_future(tensor):
    import torch
    future = torch.futures.Future()
    future.set_result(tensor)
    return future


def parse_cpu_list(value: str) -> list[int]:
    cpus = []
    for item in value.strip().split(","):
        if "-" in item:
            start, end = (int(part) for part in item.split("-", 1))
            cpus.extend(range(start, end + 1))
        elif item:
            cpus.append(int(item))
    return cpus


def apply_rank_affinity(local_rank: int) -> dict:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    local_world_size = int(os.environ.get(
        "LOCAL_WORLD_SIZE", len(tokens) if tokens else 1))
    rows = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,pci.bus_id",
        "--format=csv,noheader,nounits",
    ], text=True).splitlines()
    pci_by_gpu = {}
    gpu_by_token = {}
    for row in rows:
        index, uuid, bus_id = (part.strip() for part in row.split(",", 2))
        fields = bus_id.lower().split(":")
        physical_index = int(index)
        pci_by_gpu[physical_index] = f"{fields[-2]}:{fields[-1]}"
        gpu_by_token[str(physical_index)] = physical_index
        gpu_by_token[uuid] = physical_index
    if not tokens:
        tokens = [
            str(index) for index in sorted(pci_by_gpu)[:local_world_size]]
    elif len(tokens) >= local_world_size:
        tokens = tokens[:local_world_size]
    else:
        raise RuntimeError(
            f"LOCAL_WORLD_SIZE={local_world_size} exceeds "
            f"CUDA_VISIBLE_DEVICES={visible!r}")
    if local_rank >= len(tokens):
        raise RuntimeError(
            f"local rank {local_rank} is missing from CUDA_VISIBLE_DEVICES={visible!r}")
    try:
        physical_gpu = gpu_by_token[tokens[local_rank]]
    except KeyError as error:
        raise RuntimeError(
            "CPU affinity requires numeric GPU ids or full GPU UUIDs; "
            f"cannot resolve {tokens[local_rank]!r}") from error
    matches = list(Path("/sys/bus/pci/devices").glob(
        f"*:{pci_by_gpu[physical_gpu]}"))
    if len(matches) != 1:
        raise RuntimeError(f"cannot resolve NUMA node for GPU {physical_gpu}")
    numa_node = int((matches[0] / "numa_node").read_text().strip())
    node_cpus = set(parse_cpu_list(
        Path(f"/sys/devices/system/node/node{numa_node}/cpulist").read_text()))
    allowed = set(os.sched_getaffinity(0))
    node_cpus &= allowed

    sibling_groups = {}
    for cpu in sorted(node_cpus):
        siblings = parse_cpu_list(Path(
            f"/sys/devices/system/cpu/cpu{cpu}/topology/"
            "thread_siblings_list").read_text())
        group = tuple(sorted(set(siblings) & node_cpus))
        sibling_groups[group] = group
    groups = sorted(sibling_groups, key=lambda group: min(group))

    peer_gpus = []
    for peer_local, token in enumerate(tokens):
        if token not in gpu_by_token:
            raise RuntimeError(f"cannot resolve visible GPU token {token!r}")
        peer_physical = gpu_by_token[token]
        peer_matches = list(Path("/sys/bus/pci/devices").glob(
            f"*:{pci_by_gpu[peer_physical]}"))
        if peer_matches and int(
                (peer_matches[0] / "numa_node").read_text().strip()) == numa_node:
            peer_gpus.append((peer_local, peer_physical))
    peer_gpus.sort(key=lambda item: item[1])
    peer_position = [item[0] for item in peer_gpus].index(local_rank)
    start = len(groups) * peer_position // len(peer_gpus)
    end = len(groups) * (peer_position + 1) // len(peer_gpus)
    assigned = sorted(cpu for group in groups[start:end] for cpu in group)
    if not assigned:
        raise RuntimeError(f"empty CPU affinity for rank {local_rank}")
    os.sched_setaffinity(0, assigned)
    return {
        "local_rank": local_rank,
        "physical_gpu": physical_gpu,
        "numa_node": numa_node,
        "cpu_count": len(assigned),
        "cpus": assigned,
    }


def main() -> None:
    a = parse_args()
    if a.static_graph and a.find_unused:
        raise ValueError("--static-graph requires --no-find-unused for this benchmark")
    if a.extra_allreduce_repeats < 0:
        raise ValueError("--extra-allreduce-repeats must be non-negative")
    local_rank = int(os.environ["LOCAL_RANK"])
    affinity_info = (
        apply_rank_affinity(local_rank)
        if a.cpu_affinity == "rank"
        else {
            "local_rank": local_rank,
            "physical_gpu": None,
            "numa_node": None,
            "cpu_count": len(os.sched_getaffinity(0)),
            "cpus": sorted(os.sched_getaffinity(0)),
        }
    )
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
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    if not 0 <= a.delay_rank < world:
        raise ValueError(f"--delay-rank must be in [0, {world}), got {a.delay_rank}")

    gpu_cycles_per_ms = None
    if a.rank_delay_kind == "gpu" and a.rank_delay_ms:
        probe_cycles = 10_000_000
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        torch.cuda._sleep(probe_cycles)
        end_event.record()
        torch.cuda.synchronize(device)
        gpu_cycles_per_ms = probe_cycles / start_event.elapsed_time(end_event)

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
    model = (torch.compile(raw_model, mode=a.compile_mode)
             if a.compile and a.compile_order == "before-ddp" else raw_model)
    ddp = DDP(
        model, device_ids=[local_rank],
        find_unused_parameters=a.find_unused,
        bucket_cap_mb=a.bucket_cap,
        gradient_as_bucket_view=a.grad_bucket_view,
        static_graph=a.static_graph,
    )

    timed_records = []
    timed_active = False
    timed_step = -1
    timed_step_start_event = None
    if a.comm_mode == "timed":
        def timed_hook(_state, bucket):
            ready_event = torch.cuda.Event(enable_timing=True)
            ready_event.record(torch.cuda.current_stream(device))
            record_active = timed_active
            step_id = timed_step
            step_start_event = timed_step_start_event
            tensor = bucket.buffer()
            nbytes = tensor.numel() * tensor.element_size()
            bucket_index = int(bucket.index())
            is_last = bool(bucket.is_last())
            work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)

            def finish(future):
                result = future.value()[0]
                callback_stream = torch.cuda.current_stream(device)
                allreduce_complete_event = torch.cuda.Event(enable_timing=True)
                allreduce_complete_event.record(callback_stream)
                result.div_(world)
                hook_complete_event = torch.cuda.Event(enable_timing=True)
                hook_complete_event.record(callback_stream)
                if record_active:
                    timed_records.append({
                        "step": step_id,
                        "bucket_index": bucket_index,
                        "is_last": is_last,
                        "bytes": int(nbytes),
                        "step_start_event": step_start_event,
                        "ready_event": ready_event,
                        "allreduce_complete_event": allreduce_complete_event,
                        "hook_complete_event": hook_complete_event,
                    })
                return result

            return work.get_future().then(finish)

        ddp.register_comm_hook(None, timed_hook)
    elif a.comm_mode == "sync":
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

    train_model = (torch.compile(ddp, mode=a.compile_mode)
                   if a.compile and a.compile_order == "after-ddp" else ddp)
    optimizer = optim.Adam(ddp.parameters(), lr=a.lr, fused=a.fused_adam)
    generator = torch.Generator(device=device).manual_seed(a.seed + 1009 * rank)
    x = torch.rand(a.bs, 2, a.patch, a.patch, generator=generator, device=device)
    if a.task_layout == "balanced":
        task_id = ((torch.arange(a.bs, device=device) + rank) % 3).long()
    else:
        task_id = torch.full((a.bs,), rank % 3, dtype=torch.long, device=device)
    src_a, src_b = x[:, :1], x[:, 1:]
    if a.cost_layout == "skewed":
        rank_cost_ms = a.cost_total_ms if rank == 0 else 0.0
    elif a.cost_layout == "balanced":
        rank_cost_ms = a.cost_total_ms / world
    else:
        rank_cost_ms = 0.0
    extra_buffer = None
    if a.extra_allreduce_mb and a.extra_allreduce_repeats:
        extra_elements = max(1, int(a.extra_allreduce_mb * 1024 * 1024 / 4))
        extra_buffer = torch.zeros(extra_elements, dtype=torch.float32, device=device)

    def train_step() -> float:
        if rank_cost_ms:
            time.sleep(rank_cost_ms / 1000.0)
        if (rank == a.delay_rank and a.rank_delay_ms
                and a.rank_delay_kind == "input"):
            time.sleep(a.rank_delay_ms / 1000.0)
        if (rank == a.delay_rank and a.rank_delay_ms
                and a.rank_delay_kind == "gpu"):
            torch.cuda._sleep(max(1, int(gpu_cycles_per_ms * a.rank_delay_ms)))
        optimizer.zero_grad(set_to_none=True)
        out, aux = train_model(x, task_id)
        loss = (
            ssim_loss(out, src_a, src_b)
            + RMI_ir(out, src_a)
            + RMI_vi(out, src_b)
            + torch.mean((torch.maximum(src_a, src_b) - out) ** 2)
            + joint_grad(src_a, src_b, out).mean()
            + 0.01 * aux
        )
        loss.backward()
        if extra_buffer is not None:
            for _ in range(a.extra_allreduce_repeats):
                dist.all_reduce(extra_buffer, op=dist.ReduceOp.SUM, async_op=False)
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
    step_end_offsets_ms = []
    for step_idx in range(a.steps):
        dist.barrier()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        if a.comm_mode == "timed":
            timed_step = step_idx
            timed_step_start_event = torch.cuda.Event(enable_timing=True)
            timed_step_start_event.record(torch.cuda.current_stream(device))
            timed_active = True
        last_loss = train_step()
        if a.comm_mode == "timed":
            step_end_event = torch.cuda.Event(enable_timing=True)
            step_end_event.record(torch.cuda.current_stream(device))
        torch.cuda.synchronize(device)
        if a.comm_mode == "timed":
            timed_active = False
            step_end_offsets_ms.append(
                timed_step_start_event.elapsed_time(step_end_event))
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

    measured_peak_bytes = torch.cuda.max_memory_allocated(device)

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
    gathered_affinity = [None for _ in range(world)]
    dist.all_gather_object(gathered_affinity, affinity_info)
    comm_summary = None
    if a.comm_mode == "timed":
        by_step = {}
        for row in timed_records:
            start_event = row["step_start_event"]
            ready_ms = start_event.elapsed_time(row["ready_event"])
            allreduce_complete_ms = start_event.elapsed_time(
                row["allreduce_complete_event"])
            hook_complete_ms = start_event.elapsed_time(
                row["hook_complete_event"])
            by_step.setdefault(row["step"], []).append({
                "step": row["step"],
                "bucket_index": row["bucket_index"],
                "is_last": row["is_last"],
                "bytes": row["bytes"],
                "ready_offset_ms": ready_ms,
                "complete_offset_ms": allreduce_complete_ms,
                "hook_complete_offset_ms": hook_complete_ms,
                "hook_to_complete_ms": allreduce_complete_ms - ready_ms,
                "hook_to_return_ms": hook_complete_ms - ready_ms,
            })
        rows = []
        for step_id, bucket_rows in sorted(by_step.items()):
            rows.append({
                "step": step_id,
                "bucket_count": len(bucket_rows),
                "bytes": sum(row["bytes"] for row in bucket_rows),
                "first_ready_ms": min(row["ready_offset_ms"] for row in bucket_rows),
                "last_ready_ms": max(row["ready_offset_ms"] for row in bucket_rows),
                "all_complete_ms": max(row["complete_offset_ms"] for row in bucket_rows),
                "max_hook_to_complete_ms": max(
                    row["hook_to_complete_ms"] for row in bucket_rows),
                "max_hook_to_return_ms": max(
                    row["hook_to_return_ms"] for row in bucket_rows),
                "step_end_ms": step_end_offsets_ms[step_id],
            })
        comm_summary = {
            "timing_source": "cuda_event",
            "steps_observed": len(rows),
            "bucket_count_mean": statistics.fmean(
                row["bucket_count"] for row in rows),
            "bytes_per_step_mean": statistics.fmean(row["bytes"] for row in rows),
            "first_bucket_ready_ms_mean": statistics.fmean(
                row["first_ready_ms"] for row in rows),
            "last_bucket_ready_ms_mean": statistics.fmean(
                row["last_ready_ms"] for row in rows),
            "all_comm_complete_ms_mean": statistics.fmean(
                row["all_complete_ms"] for row in rows),
            "comm_tail_after_last_ready_ms_mean": statistics.fmean(
                row["all_complete_ms"] - row["last_ready_ms"] for row in rows),
            "max_hook_to_complete_ms_mean": statistics.fmean(
                row["max_hook_to_complete_ms"] for row in rows),
            "max_hook_to_return_ms_mean": statistics.fmean(
                row["max_hook_to_return_ms"] for row in rows),
            "step_end_ms_mean": statistics.fmean(row["step_end_ms"] for row in rows),
            "rows": rows,
        }
    gathered_comm = [None for _ in range(world)]
    dist.all_gather_object(gathered_comm, comm_summary)

    peak = torch.tensor([measured_peak_bytes],
                        dtype=torch.float64, device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    logging_data = ddp._get_ddp_logging_data()

    if rank == 0:
        mean_ms = statistics.fmean(critical_ms)
        result = {
            "schema": "ddp-training-v1",
            "protocol_revision": 2,
            "torch_version": torch.__version__,
            "nccl_version": list(torch.cuda.nccl.version()),
            "gpu": torch.cuda.get_device_name(device),
            "world_size": world,
            "config": {
                "seed": a.seed,
                "bs_per_gpu": a.bs, "patch": a.patch, "out_channel": a.oc,
                "depth": a.depth, "n_routed": a.nr, "top_k": a.k,
                "combine": a.combine, "cap_factor": a.cap_factor,
                "attn": a.attn, "compile": a.compile,
                "compile_mode": a.compile_mode,
                "compile_order": a.compile_order,
                "bucket_cap_mb": a.bucket_cap,
                "gradient_as_bucket_view": a.grad_bucket_view,
                "find_unused_parameters": a.find_unused,
                "static_graph": a.static_graph,
                "fused_adam": a.fused_adam,
                "comm_mode": a.comm_mode,
                "task_layout": a.task_layout,
                "delay_rank": a.delay_rank,
                "rank_delay_ms": a.rank_delay_ms,
                "rank_delay_kind": a.rank_delay_kind,
                "cost_layout": a.cost_layout,
                "cost_total_ms": a.cost_total_ms,
                "rank_cost_ms": rank_cost_ms,
                "extra_allreduce_mb": a.extra_allreduce_mb,
                "extra_allreduce_repeats": a.extra_allreduce_repeats,
                "extra_allreduce_implementation": (
                    "zero-buffer-no-scale"
                    if a.extra_allreduce_mb and a.extra_allreduce_repeats
                    else None
                ),
                "cpu_affinity": a.cpu_affinity,
                "learning_rate": a.lr,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
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
            "affinity_by_rank": gathered_affinity,
            "timed_communication_by_rank": gathered_comm,
        }
        text = json.dumps(result, ensure_ascii=False, indent=2)
        if a.output:
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(text + "\n")
        print(text)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
