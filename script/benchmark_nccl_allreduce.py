#!/usr/bin/env python3
"""Measure single-node NCCL all-reduce latency/bandwidth over DDP bucket sizes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sizes-mb", nargs="+", type=float,
                   default=(0.015625, 0.0625, 0.25, 0.5, 1, 2, 4, 8, 16))
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--output", type=Path)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    rows = []

    for size_mb in a.sizes_mb:
        nbytes = max(4, int(size_mb * 1024 * 1024))
        numel = (nbytes + 3) // 4
        tensor = torch.zeros(numel, dtype=torch.float32, device=device)
        for _ in range(a.warmup):
            dist.all_reduce(tensor)
        torch.cuda.synchronize(device)
        dist.barrier()

        local_ms = []
        for _ in range(a.iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            dist.all_reduce(tensor)
            end.record()
            end.synchronize()
            local_ms.append(float(start.elapsed_time(end)))

        gathered = [None for _ in range(world)]
        dist.all_gather_object(gathered, local_ms)
        if rank == 0:
            rank_times = np.asarray(gathered, dtype=np.float64)
            critical = rank_times.max(axis=0)
            mean_ms = float(critical.mean())
            alg_gbps = nbytes / (mean_ms / 1000.0) / 1e9
            bus_gbps = alg_gbps * (2.0 * (world - 1) / world) if world > 1 else 0.0
            rows.append({
                "size_mb": nbytes / (1024 * 1024),
                "bytes": nbytes,
                "latency_mean_ms": mean_ms,
                "latency_std_ms": float(critical.std()),
                "latency_p50_ms": float(np.percentile(critical, 50)),
                "latency_p95_ms": float(np.percentile(critical, 95)),
                "algorithm_bandwidth_gbps": alg_gbps,
                "bus_bandwidth_gbps": bus_gbps,
            })

    if rank == 0:
        result = {
            "schema": "nccl-allreduce-v1",
            "world_size": world,
            "torch_version": torch.__version__,
            "nccl_version": list(torch.cuda.nccl.version()),
            "gpu": torch.cuda.get_device_name(device),
            "warmup": a.warmup,
            "iters": a.iters,
            "rows": rows,
        }
        text = json.dumps(result, ensure_ascii=False, indent=2)
        if a.output:
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(text + "\n")
        print(text)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
