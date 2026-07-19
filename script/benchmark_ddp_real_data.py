#!/usr/bin/env python3
"""Real-data DDP bottleneck benchmark with rank-level stage timing.

The benchmark keeps the production dataset, model, losses, and DDP reducer but
sets lr=0 by default. It reports data wait, H2D, model-step critical path, and
optional timed all-reduce summaries without writing a profiler trace.
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


WORKER_CPU_SETS: list[list[int]] = []


def pin_loader_worker(worker_id: int) -> None:
    if WORKER_CPU_SETS:
        os.sched_setaffinity(0, WORKER_CPU_SETS[worker_id])


def isolate_trainer_and_workers(affinity_info: dict, workers: int) -> dict:
    """Reserve physical cores for the trainer and partition the rest by worker."""
    cpus = set(affinity_info["cpus"])
    sibling_groups = {}
    for cpu in sorted(cpus):
        value = Path(
            f"/sys/devices/system/cpu/cpu{cpu}/topology/"
            "thread_siblings_list").read_text().strip()
        siblings = []
        for item in value.split(","):
            if "-" in item:
                start, end = (int(part) for part in item.split("-", 1))
                siblings.extend(range(start, end + 1))
            else:
                siblings.append(int(item))
        group = tuple(sorted(set(siblings) & cpus))
        sibling_groups[group] = group
    groups = sorted(sibling_groups, key=lambda group: min(group))
    trainer_group_count = min(4, max(1, len(groups) // 4))
    trainer_cpus = sorted(
        cpu for group in groups[:trainer_group_count] for cpu in group)
    worker_groups = groups[trainer_group_count:]
    if workers > len(worker_groups):
        raise RuntimeError(
            f"cannot isolate {workers} workers with only "
            f"{len(worker_groups)} worker core groups")
    global WORKER_CPU_SETS
    WORKER_CPU_SETS = []
    for worker_id in range(workers):
        assigned_groups = worker_groups[worker_id::workers]
        assigned = sorted(
            cpu for group in assigned_groups for cpu in group)
        if not assigned:
            raise RuntimeError(f"worker {worker_id} received no isolated cores")
        WORKER_CPU_SETS.append(assigned)
    os.sched_setaffinity(0, trainer_cpus)
    return {
        **affinity_info,
        "cpu_count": len(trainer_cpus),
        "cpus": trainer_cpus,
        "trainer_cpus": trainer_cpus,
        "worker_cpu_sets": WORKER_CPU_SETS,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--code-root", type=Path,
                   default=Path("/ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path,
                   default=Path("models/Ours_v3_frozen/model_26.pth"))
    p.add_argument("--config", nargs="+", default=[
        "configs/gfp_pc.json",
        "configs/irvis_msrs.json",
        "configs/medical_harvard.json",
    ])
    p.add_argument("--seed", type=int, default=20260719)
    p.add_argument("--bs", type=int, default=10)
    p.add_argument("--patch", type=int, default=170)
    p.add_argument("--crops-per-task", type=int, default=4000)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--sampler", choices=("distributed", "task-balanced"),
                   default="distributed")
    p.add_argument("--task-costs", type=float, nargs="+",
                   help="optional positive per-task cost estimates")
    p.add_argument("--oc", type=int, default=96)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--nr", type=int, default=12)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--combine", choices=("sparse", "grouped"), default="grouped")
    p.add_argument("--cap-factor", type=float, default=1.25)
    p.add_argument("--attn", choices=("vanilla", "sdpa"), default="sdpa")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--bucket-cap", type=float, default=8.0)
    p.add_argument("--grad-bucket-view", action="store_true")
    p.add_argument("--static-graph", action="store_true")
    p.add_argument("--find-unused", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--fused-adam", action="store_true")
    p.add_argument("--comm-mode", choices=("default", "timed", "sync", "noop"),
                   default="default")
    p.add_argument("--cpu-affinity", choices=("none", "rank", "isolated"),
                   default="none")
    p.add_argument("--lr", type=float, default=0.0)
    p.add_argument("--warmup", type=int, default=15)
    p.add_argument("--steps", type=int, default=50)
    return p.parse_args()


def completed_future(tensor):
    import torch
    future = torch.futures.Future()
    future.set_result(tensor)
    return future


def mean(values):
    return statistics.fmean(values) if values else 0.0


def main() -> None:
    a = parse_args()
    if a.static_graph and a.find_unused:
        raise ValueError("--static-graph requires --no-find-unused")
    output = a.output.resolve()
    code_root = a.code_root.resolve()
    checkpoint = (a.checkpoint if a.checkpoint.is_absolute()
                  else code_root / a.checkpoint)
    configs = [str(path if Path(path).is_absolute() else code_root / path)
               for path in a.config]
    sys.path.insert(0, str(code_root))
    os.chdir(code_root)
    local_rank = int(os.environ["LOCAL_RANK"])
    if a.cpu_affinity in ("rank", "isolated"):
        from benchmark_ddp_training import apply_rank_affinity
        affinity_info = apply_rank_affinity(local_rank)
        if a.cpu_affinity == "isolated":
            affinity_info = isolate_trainer_and_workers(
                affinity_info, a.workers)
    else:
        affinity_info = {
            "local_rank": local_rank,
            "physical_gpu": None,
            "numa_node": None,
            "cpu_count": len(os.sched_getaffinity(0)),
            "cpus": sorted(os.sched_getaffinity(0)),
        }

    import torch
    import torch.distributed as dist
    import torch.optim as optim
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from Networks.net_moe import MODEL_MoE
    from losses import RMI_ir, RMI_vi, joint_grad, ssim_loss
    from mm_fusion_dataset import MMFusionDataset

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)

    class TaskBalancedDistributedSampler(torch.utils.data.Sampler):
        """Balance each global batch without changing its sample multiset."""

        def __init__(self, data, replicas, replica_rank, seed, batch_size,
                     task_costs=None):
            self.data = data
            self.replicas = replicas
            self.rank = replica_rank
            self.seed = seed
            self.batch_size = batch_size
            self.epoch = 0
            self.task_by_index = []
            self.task_sizes = {}
            for record in data.index:
                task = int(record[0])
                self.task_by_index.append(task)
                self.task_sizes[task] = self.task_sizes.get(task, 0) + 1
            tasks = sorted(self.task_sizes)
            if task_costs is not None and len(task_costs) != len(tasks):
                raise ValueError(
                    f"--task-costs needs {len(tasks)} values, got {len(task_costs)}")
            costs = task_costs or [1.0] * len(tasks)
            if any(cost <= 0 for cost in costs):
                raise ValueError("--task-costs values must be positive")
            self.task_costs = dict(zip(tasks, costs))
            self.global_batch_size = replicas * batch_size
            self.total_size = (
                len(data) // self.global_batch_size * self.global_batch_size)
            self.length = self.total_size // replicas

        def set_epoch(self, epoch):
            self.epoch = epoch

        def __len__(self):
            return self.length

        def __iter__(self):
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            global_indices = torch.randperm(
                len(self.data), generator=generator).tolist()[:self.total_size]
            rank_sequences = [[] for _ in range(self.replicas)]
            for block_id, start in enumerate(
                    range(0, self.total_size, self.global_batch_size)):
                block = global_indices[start:start + self.global_batch_size]
                by_task = {}
                for index in block:
                    by_task.setdefault(
                        self.task_by_index[index], []).append(index)
                assignments = [[] for _ in range(self.replicas)]
                task_counts = [dict() for _ in range(self.replicas)]
                predicted_costs = [0.0] * self.replicas
                groups = sorted(
                    by_task.items(),
                    key=lambda item: (
                        -self.task_costs[item[0]], -len(item[1]), item[0]))
                for task, indices in groups:
                    for index in indices:
                        candidates = [
                            replica for replica in range(self.replicas)
                            if len(assignments[replica]) < self.batch_size
                        ]
                        replica = min(
                            candidates,
                            key=lambda candidate: (
                                task_counts[candidate].get(task, 0),
                                predicted_costs[candidate],
                                (candidate - block_id) % self.replicas,
                            ))
                        assignments[replica].append(index)
                        task_counts[replica][task] = (
                            task_counts[replica].get(task, 0) + 1)
                        predicted_costs[replica] += self.task_costs[task]
                if any(len(indices) != self.batch_size
                       for indices in assignments):
                    raise RuntimeError(
                        "failed to build a full balanced global batch")
                for replica, indices in enumerate(assignments):
                    rank_sequences[replica].extend(indices)
            if len(rank_sequences[self.rank]) != self.length:
                raise RuntimeError("balanced sampler produced an invalid length")
            return iter(rank_sequences[self.rank])

    dataset = MMFusionDataset(
        configs, "train", a.patch, a.crops_per_task, random_crop=True)
    if a.sampler == "task-balanced":
        sampler = TaskBalancedDistributedSampler(
            dataset, replicas=world, replica_rank=rank, seed=a.seed,
            batch_size=a.bs, task_costs=a.task_costs)
    else:
        sampler = DistributedSampler(
            dataset, num_replicas=world, rank=rank, shuffle=True,
            seed=a.seed, drop_last=True)
    generator = torch.Generator().manual_seed(a.seed + rank)
    loader = DataLoader(
        dataset, batch_size=a.bs, sampler=sampler, num_workers=a.workers,
        pin_memory=True, drop_last=True, persistent_workers=a.workers > 0,
        generator=generator,
        worker_init_fn=pin_loader_worker
        if a.cpu_affinity == "isolated" and a.workers > 0 else None)

    raw_model = MODEL_MoE(
        in_channel=2, n_tasks=len(configs), out_channel=a.oc, depth=a.depth,
        window_size=8, n_routed=a.nr, k=a.k, n_shared=1, out_scale=True,
        fusion_head="blend", res_scale=0.0, attn_impl=a.attn,
    ).to(device)
    raw_model.set_combine(a.combine, cap_factor=a.cap_factor)
    raw_model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True),
        strict=True)
    model = torch.compile(raw_model) if a.compile else raw_model
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
    timed_step_end_offsets_ms = {}
    if a.comm_mode == "timed":
        def timed_hook(_state, bucket):
            ready_event = torch.cuda.Event(enable_timing=True)
            ready_event.record(torch.cuda.current_stream(device))
            record_active = timed_active
            step_id = timed_step
            step_start_event = timed_step_start_event
            tensor = bucket.buffer()
            nbytes = tensor.numel() * tensor.element_size()
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

    optimizer = optim.Adam(ddp.parameters(), lr=a.lr, fused=a.fused_adam)

    def train_batch(batch):
        src_a = batch["src_a"].to(device, non_blocking=True)
        src_b = batch["src_b"].to(device, non_blocking=True)
        task_id = batch["task_id"].to(device, non_blocking=True)
        torch.cuda.synchronize(device)
        optimizer.zero_grad(set_to_none=True)
        out, aux = ddp(torch.cat((src_a, src_b), 1), task_id)
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
        return loss

    sampler_epoch = 0
    sampler.set_epoch(sampler_epoch)
    iterator = iter(loader)

    def next_batch():
        nonlocal iterator, sampler_epoch
        try:
            return next(iterator)
        except StopIteration:
            sampler_epoch += 1
            sampler.set_epoch(sampler_epoch)
            iterator = iter(loader)
            return next(iterator)

    for _ in range(a.warmup):
        batch = next_batch()
        train_batch(batch)
    torch.cuda.synchronize(device)

    local_rows = []
    last_loss = None
    for step in range(a.steps):
        dist.barrier()
        iteration_start = time.perf_counter()
        data_start = iteration_start
        batch = next_batch()
        data_end = time.perf_counter()
        h2d_start = data_end
        src_a = batch["src_a"].to(device, non_blocking=True)
        src_b = batch["src_b"].to(device, non_blocking=True)
        task_id = batch["task_id"].to(device, non_blocking=True)
        torch.cuda.synchronize(device)
        h2d_end = time.perf_counter()

        if a.comm_mode == "timed":
            timed_step = step
            timed_step_start_event = torch.cuda.Event(enable_timing=True)
            timed_step_start_event.record(torch.cuda.current_stream(device))
            timed_active = True
        model_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        out, aux = ddp(torch.cat((src_a, src_b), 1), task_id)
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
        if a.comm_mode == "timed":
            step_end_event = torch.cuda.Event(enable_timing=True)
            step_end_event.record(torch.cuda.current_stream(device))
        torch.cuda.synchronize(device)
        model_end = time.perf_counter()
        if a.comm_mode == "timed":
            timed_active = False
            timed_step_end_offsets_ms[step] = (
                timed_step_start_event.elapsed_time(step_end_event))
        last_loss = float(loss.detach())
        task_counts = np.bincount(
            batch["task_id"].numpy(), minlength=len(configs)).tolist()
        local_rows.append({
            "step": step,
            "data_wait_ms": (data_end - data_start) * 1000,
            "h2d_ms": (h2d_end - h2d_start) * 1000,
            "model_step_ms": (model_end - model_start) * 1000,
            "iteration_ms": (model_end - iteration_start) * 1000,
            "task_counts": task_counts,
        })

    materialized_timed_records = []
    for row in timed_records:
        start_event = row["step_start_event"]
        ready_ms = start_event.elapsed_time(row["ready_event"])
        allreduce_complete_ms = start_event.elapsed_time(
            row["allreduce_complete_event"])
        hook_complete_ms = start_event.elapsed_time(row["hook_complete_event"])
        materialized_timed_records.append({
            "step": row["step"],
            "bytes": row["bytes"],
            "ready_offset_ms": ready_ms,
            "complete_offset_ms": allreduce_complete_ms,
            "hook_complete_offset_ms": hook_complete_ms,
            "hook_to_complete_ms": allreduce_complete_ms - ready_ms,
            "hook_to_return_ms": hook_complete_ms - ready_ms,
            "step_end_ms": timed_step_end_offsets_ms[row["step"]],
        })

    gathered_rows = [None for _ in range(world)]
    dist.all_gather_object(gathered_rows, local_rows)
    gathered_comm = [None for _ in range(world)]
    dist.all_gather_object(gathered_comm, materialized_timed_records)
    gathered_affinity = [None for _ in range(world)]
    dist.all_gather_object(gathered_affinity, affinity_info)
    logging_data = ddp._get_ddp_logging_data()

    if rank == 0:
        critical = []
        rank_gap = []
        for step in range(a.steps):
            values = [rows[step]["iteration_ms"] for rows in gathered_rows]
            critical.append(max(values))
            rank_gap.append(max(values) - statistics.fmean(values))
        stage_summary = []
        for rank_id, rows in enumerate(gathered_rows):
            stage_summary.append({
                "rank": rank_id,
                "data_wait_ms_mean": mean([row["data_wait_ms"] for row in rows]),
                "data_wait_ms_p95": float(np.percentile(
                    [row["data_wait_ms"] for row in rows], 95)),
                "h2d_ms_mean": mean([row["h2d_ms"] for row in rows]),
                "model_step_ms_mean": mean(
                    [row["model_step_ms"] for row in rows]),
                "iteration_ms_mean": mean([row["iteration_ms"] for row in rows]),
                "task_counts": [
                    sum(row["task_counts"][task] for row in rows)
                    for task in range(len(configs))
                ],
            })
        comm_summary = []
        for rank_id, records in enumerate(gathered_comm):
            by_step = {}
            for row in records:
                by_step.setdefault(row["step"], []).append(row)
            observed = []
            for step, rows in sorted(by_step.items()):
                observed.append({
                    "step": step,
                    "bucket_count": len(rows),
                    "bytes": sum(row["bytes"] for row in rows),
                    "first_ready_ms": min(row["ready_offset_ms"] for row in rows),
                    "last_ready_ms": max(row["ready_offset_ms"] for row in rows),
                    "all_complete_ms": max(
                        row["complete_offset_ms"] for row in rows),
                    "max_hook_to_complete_ms": max(
                        row["hook_to_complete_ms"] for row in rows),
                    "max_hook_to_return_ms": max(
                        row["hook_to_return_ms"] for row in rows),
                    "step_end_ms": max(row["step_end_ms"] for row in rows),
                })
            comm_summary.append({
                "rank": rank_id,
                "timing_source": "cuda_event" if records else None,
                "steps_observed": len(observed),
                "bytes_per_step_mean": mean(
                    [row["bytes"] for row in observed]),
                "bucket_count_mean": mean(
                    [row["bucket_count"] for row in observed]),
                "first_ready_ms_mean": mean(
                    [row["first_ready_ms"] for row in observed]),
                "last_ready_ms_mean": mean(
                    [row["last_ready_ms"] for row in observed]),
                "all_complete_ms_mean": mean(
                    [row["all_complete_ms"] for row in observed]),
                "comm_tail_after_last_ready_ms_mean": mean([
                    row["all_complete_ms"] - row["last_ready_ms"]
                    for row in observed
                ]),
                "max_hook_to_complete_ms_mean": mean([
                    row["max_hook_to_complete_ms"] for row in observed
                ]),
                "max_hook_to_return_ms_mean": mean([
                    row["max_hook_to_return_ms"] for row in observed
                ]),
                "step_end_ms_mean": mean([
                    row["step_end_ms"] for row in observed
                ]),
            })
        result = {
            "schema": "ddp-real-data-bottleneck-v1",
            "protocol_revision": 2,
            "torch_version": torch.__version__,
            "gpu": torch.cuda.get_device_name(device),
            "world_size": world,
            "config": {
                "seed": a.seed,
                "configs": configs,
                "checkpoint": str(checkpoint),
                "bs_per_gpu": a.bs,
                "patch": a.patch,
                "workers": a.workers,
                "sampler": a.sampler,
                "sampler_implementation": (
                    "global-batch-task-balanced-v2"
                    if a.sampler == "task-balanced"
                    else "torch-distributed-v1"
                ),
                "task_costs": a.task_costs,
                "sampler_epoch_last": sampler_epoch,
                "sampler_total_size": sampler.total_size,
                "loader_samples_per_epoch": len(loader) * a.bs * world,
                "sampler_preserves_global_batch_multiset": (
                    a.sampler == "task-balanced"),
                "dataset_task_sizes": {
                    str(task): sum(
                        int(record[0]) == task for record in dataset.index)
                    for task in range(len(configs))
                },
                "combine": a.combine,
                "cap_factor": a.cap_factor,
                "attn": a.attn,
                "compile": a.compile,
                "bucket_cap_mb": a.bucket_cap,
                "gradient_as_bucket_view": a.grad_bucket_view,
                "static_graph": a.static_graph,
                "find_unused_parameters": a.find_unused,
                "fused_adam": a.fused_adam,
                "comm_mode": a.comm_mode,
                "cpu_affinity": a.cpu_affinity,
                "learning_rate": a.lr,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "measurement": {
                "warmup_steps": a.warmup,
                "steps": a.steps,
                "critical_iteration_ms": critical,
                "critical_iteration_ms_mean": mean(critical),
                "critical_iteration_ms_p95": float(np.percentile(critical, 95)),
                "global_samples_per_second": world * a.bs * 1000 / mean(critical),
                "rank_gap_ms_mean": mean(rank_gap),
                "last_loss": last_loss,
                "task_counts_by_step_by_rank": [
                    [rows[step]["task_counts"] for rows in gathered_rows]
                    for step in range(a.steps)
                ],
            },
            "stage_by_rank": stage_summary,
            "affinity_by_rank": gathered_affinity,
            "timed_communication_by_rank": comm_summary,
            "ddp_logging": {str(k): v for k, v in logging_data.items()
                            if isinstance(v, (str, int, float, bool))},
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
