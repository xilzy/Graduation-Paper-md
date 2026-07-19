#!/usr/bin/env python3
"""Aggregate independent bottleneck benchmark trials into one JSON report."""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path


T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
    10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
    20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
    25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path,
                   default=Path("Materials/efficiency/data/bottleneck"))
    p.add_argument("--output", type=Path,
                   default=Path("Materials/efficiency/data/bottleneck_summary.json"))
    return p.parse_args()


def stats(values):
    values = [float(value) for value in values]
    n = len(values)
    if not values:
        raise ValueError("cannot summarize an empty sample")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n > 1 else 0.0
    t = T95.get(n - 1, 1.96)
    half = t * std / math.sqrt(n) if n > 1 else 0.0
    return {
        "n": n, "values": values, "mean": mean, "std": std,
        "ci95_low": mean - half if n > 1 else None,
        "ci95_high": mean + half if n > 1 else None,
    }


def load(path):
    return json.loads(path.read_text())


def measurement(schema, key):
    def getter(data):
        if data.get("schema") != schema:
            raise ValueError(
                f"expected schema {schema}, got {data.get('schema')}")
        return data["measurement"][key]
    return getter


def communication_key(match, data):
    world = int(match.group(1))
    mode = match.group(2)
    if data.get("world_size") != world:
        raise ValueError(
            f"communication filename says world={world}, "
            f"JSON says {data.get('world_size')}")
    if data.get("config", {}).get("comm_mode") != mode:
        raise ValueError(
            f"communication filename says mode={mode}, "
            f"JSON says {data.get('config', {}).get('comm_mode')}")
    if data.get("config", {}).get("learning_rate") != 0:
        raise ValueError("communication decomposition requires learning_rate=0")
    if data.get("protocol_revision") != 2:
        raise ValueError("communication decomposition requires protocol revision 2")
    return f"w{world}_{mode}"


def batch_key(match, data):
    world, batch, mode = (
        int(match.group(1)), int(match.group(2)), match.group(3))
    config = data.get("config", {})
    if (data.get("world_size") != world
            or data.get("protocol_revision") != 2
            or config.get("bs_per_gpu") != batch
            or config.get("comm_mode") != mode
            or config.get("learning_rate") != 0):
        raise ValueError("batch-boundary filename/config mismatch")
    return f"w{world}_bs{batch}_{mode}"


def pressure_key(match, data, rounds):
    world = int(match.group(1))
    amount_text = match.group(2)
    amount = int(amount_text) if rounds else float(amount_text)
    config = data.get("config", {})
    expected_mb = 16.0 if rounds else float(amount)
    expected_repeats = amount if rounds else 1
    expected_impl = (
        "zero-buffer-no-scale"
        if expected_mb > 0 and expected_repeats > 0 else None)
    if (data.get("schema") != "ddp-training-v1"
            or data.get("protocol_revision") != 2
            or data.get("world_size") != world
            or config.get("comm_mode") != "noop"
            or config.get("learning_rate") != 0
            or float(config.get("extra_allreduce_mb", -1)) != expected_mb
            or config.get("extra_allreduce_repeats") != expected_repeats
            or config.get("extra_allreduce_implementation") != expected_impl):
        raise ValueError("communication-pressure filename/config mismatch")
    label = f"r{amount}" if rounds else f"mb{amount_text}"
    return f"w{world}_{label}"


def real_communication_key(match, data):
    world = int(match.group(1))
    mode = match.group(2)
    config = data.get("config", {})
    if (data.get("world_size") != world
            or data.get("protocol_revision") != 2
            or config.get("comm_mode") != mode
            or config.get("learning_rate") != 0):
        raise ValueError("real-data communication filename/config mismatch")
    return f"w{world}_{mode}"


def sampler_key(match, data):
    label, workers = match.group(1), int(match.group(2))
    expected_sampler = (
        "task-balanced" if label == "taskbalanced" else "distributed")
    expected_impl = (
        "global-batch-task-balanced-v2"
        if label == "taskbalanced" else "torch-distributed-v1")
    config = data.get("config", {})
    if (data.get("schema") != "ddp-real-data-bottleneck-v1"
            or data.get("protocol_revision") != 2
            or data.get("world_size") != 4
            or config.get("workers") != workers
            or config.get("sampler") != expected_sampler
            or config.get("sampler_implementation") != expected_impl
            or config.get("seed") is None
            or config.get("learning_rate") != 0
            or config.get("loader_samples_per_epoch") is None
            or ("task_counts_by_step_by_rank"
                not in data.get("measurement", {}))):
        raise ValueError("sampler filename/config/protocol mismatch")
    if label == "taskbalanced" and not config.get(
            "sampler_preserves_global_batch_multiset"):
        raise ValueError("task-balanced sampler lacks multiset guarantee")
    return f"{label}_workers{workers}"


def assert_same_experiment(left, right, ignored_config_fields):
    for field in (
            "schema", "protocol_revision", "torch_version", "gpu",
            "world_size"):
        if left.get(field) != right.get(field):
            raise ValueError(f"paired experiments differ in {field}")
    left_config = {
        key: value for key, value in left["config"].items()
        if key not in ignored_config_fields
    }
    right_config = {
        key: value for key, value in right["config"].items()
        if key not in ignored_config_fields
    }
    if left_config != right_config:
        changed = sorted(
            key for key in set(left_config) | set(right_config)
            if left_config.get(key) != right_config.get(key))
        raise ValueError(f"paired experiment configs differ: {changed}")
    for field in ("warmup_steps", "steps"):
        if left["measurement"].get(field) != right["measurement"].get(field):
            raise ValueError(f"paired measurements differ in {field}")


def aggregate_pattern(paths, pattern, key_builder, value_getter):
    groups = {}
    for path in paths:
        match = re.fullmatch(pattern, path.name)
        if not match:
            continue
        data = load(path)
        key = key_builder(match, data)
        groups.setdefault(key, []).append(value_getter(data))
    return {key: stats(values) for key, values in sorted(groups.items())}


def main():
    a = parse_args()
    paths = sorted(a.data_dir.glob("*.json"))
    result = {
        "schema": "ddp-bottleneck-summary-v1",
        "source_files": len(paths),
    }

    comm = aggregate_pattern(
        paths, r"comm_w(\d+)_(default|noop|timed|sync)_t(\d+)\.json",
        communication_key,
        measurement("ddp-training-v1", "mean_step_ms"))
    result["communication_modes"] = comm
    paired = {}
    for world in (4, 8):
        by_mode = {}
        for mode in ("default", "noop", "timed", "sync"):
            by_mode[mode] = {}
            for path in paths:
                match = re.fullmatch(
                    rf"comm_w{world}_{mode}_t(\d+)\.json", path.name)
                if match:
                    data = load(path)
                    communication_key(re.fullmatch(
                        r"comm_w(\d+)_(default|noop|timed|sync)_t(\d+)\.json",
                        path.name), data)
                    by_mode[mode][int(match.group(1))] = data
        default_common = sorted(set(by_mode["noop"]) & set(by_mode["default"]))
        if default_common:
            deltas = []
            for trial in default_common:
                default = by_mode["default"][trial]
                noop = by_mode["noop"][trial]
                assert_same_experiment(default, noop, {"comm_mode"})
                deltas.append(
                    default["measurement"]["mean_step_ms"]
                    - noop["measurement"]["mean_step_ms"])
            paired[f"w{world}_default_minus_noop_ms"] = stats(deltas)
        common = sorted(set(by_mode["noop"]) & set(by_mode["timed"])
                        & set(by_mode["sync"]))
        if common:
            timed_deltas = []
            sync_deltas = []
            sync_timed_deltas = []
            for trial in common:
                noop = by_mode["noop"][trial]
                timed = by_mode["timed"][trial]
                sync = by_mode["sync"][trial]
                assert_same_experiment(timed, noop, {"comm_mode"})
                assert_same_experiment(sync, noop, {"comm_mode"})
                timed_deltas.append(
                    timed["measurement"]["mean_step_ms"]
                    - noop["measurement"]["mean_step_ms"])
                sync_deltas.append(
                    sync["measurement"]["mean_step_ms"]
                    - noop["measurement"]["mean_step_ms"])
                sync_timed_deltas.append(
                    sync["measurement"]["mean_step_ms"]
                    - timed["measurement"]["mean_step_ms"])
            paired[f"w{world}_timed_minus_noop_ms"] = stats(timed_deltas)
            paired[f"w{world}_sync_minus_noop_ms"] = stats(sync_deltas)
            paired[f"w{world}_sync_minus_timed_ms"] = stats(
                sync_timed_deltas)
    result["paired_communication_deltas"] = paired

    timed_hook = {}
    for world in (4, 8):
        records = []
        for path in paths:
            if re.fullmatch(rf"comm_w{world}_timed_t\d+\.json", path.name):
                data = load(path)
                rank_records = [
                    item for item in data["timed_communication_by_rank"] if item]
                if not rank_records:
                    raise ValueError(f"{path} has no timed communication records")
                if any(record.get("timing_source") != "cuda_event"
                       for record in rank_records):
                    raise ValueError(f"{path} does not use CUDA-event timing")
                fields = (
                    "bucket_count_mean", "bytes_per_step_mean",
                    "first_bucket_ready_ms_mean", "last_bucket_ready_ms_mean",
                    "all_comm_complete_ms_mean",
                    "comm_tail_after_last_ready_ms_mean", "step_end_ms_mean",
                    "max_hook_to_complete_ms_mean",
                    "max_hook_to_return_ms_mean",
                )
                missing = [
                    field for field in fields
                    if any(field not in record for record in rank_records)
                ]
                if missing:
                    raise ValueError(f"{path} is missing timed fields: {missing}")
                records.append({
                    field: statistics.fmean(
                        record[field] for record in rank_records)
                    for field in fields
                })
        if records:
            timed_hook[f"w{world}"] = {
                field: stats([record[field] for record in records])
                for field in fields
            }
    result["timed_hook"] = timed_hook

    result["batch_compute_boundary"] = aggregate_pattern(
        paths, r"batch_w(\d+)_bs(\d+)_(default|noop|timed)_t(\d+)\.json",
        batch_key,
        measurement("ddp-training-v1", "mean_step_ms"))
    result["communication_pressure"] = aggregate_pattern(
        paths, r"pressure_w(\d+)_mb([\d.]+)_t(\d+)\.json",
        lambda m, d: pressure_key(m, d, rounds=False),
        measurement("ddp-training-v1", "mean_step_ms"))
    result["communication_round_pressure"] = aggregate_pattern(
        paths, r"rounds_w(\d+)_r(\d+)_t(\d+)\.json",
        lambda m, d: pressure_key(m, d, rounds=True),
        measurement("ddp-training-v1", "mean_step_ms"))
    result["per_gpu"] = aggregate_pattern(
        paths, r"gpu_card_(\d+)_t(\d+)\.json",
        lambda m, _d: f"gpu{m.group(1)}",
        measurement("grouped-moe-single-v1", "mean_ms"))
    result["gpu_straggler"] = aggregate_pattern(
        paths, r"straggler_gpu_([\d.]+)ms_t(\d+)\.json",
        lambda m, _d: f"delay_{m.group(1)}ms",
        measurement("ddp-training-v1", "mean_step_ms"))
    result["cost_partition"] = aggregate_pattern(
        paths, r"cost_(none|skewed|balanced)_([\d.]+)ms_t(\d+)\.json",
        lambda m, _d: f"{m.group(1)}_{m.group(2)}ms",
        measurement("ddp-training-v1", "mean_step_ms"))
    result["task_layout"] = aggregate_pattern(
        paths, r"task_(sparse|grouped)_(balanced|homogeneous)_t(\d+)\.json",
        lambda m, _d: f"{m.group(1)}_{m.group(2)}",
        measurement("ddp-training-v1", "mean_step_ms"))
    result["real_data"] = aggregate_pattern(
        paths, r"real_w(\d+)_(default|noop|timed|sync)_t(\d+)\.json",
        real_communication_key,
        measurement(
            "ddp-real-data-bottleneck-v1", "critical_iteration_ms_mean"))
    result["real_data_workers"] = aggregate_pattern(
        paths, r"real_workers(\d+)_(none|rank|isolated)_t(\d+)\.json",
        lambda m, _d: f"workers{m.group(1)}_{m.group(2)}",
        measurement(
            "ddp-real-data-bottleneck-v1", "critical_iteration_ms_mean"))
    result["real_data_mapping"] = aggregate_pattern(
        paths, r"real_workers(\d+)_(normal|reverse)_t(\d+)\.json",
        lambda m, _d: f"workers{m.group(1)}_{m.group(2)}",
        measurement(
            "ddp-real-data-bottleneck-v1", "critical_iteration_ms_mean"))
    result["rank_diagnosis"] = aggregate_pattern(
        paths, r"rankdiag_(normal|reverse)_(none|rank)_t(\d+)\.json",
        lambda m, _d: f"{m.group(1)}_{m.group(2)}",
        measurement("ddp-training-v1", "mean_step_ms"))
    result["compile_after_ddp"] = aggregate_pattern(
        paths, r"compile_after_w(\d+)_t(\d+)\.json",
        lambda m, _d: f"w{m.group(1)}",
        measurement("ddp-training-v1", "mean_step_ms"))
    result["real_data_sampler"] = aggregate_pattern(
        paths, r"real_sampler_(distributed|taskbalanced)_workers(\d+)_t(\d+)\.json",
        sampler_key,
        measurement(
            "ddp-real-data-bottleneck-v1", "critical_iteration_ms_mean"))
    sampler_trials = {}
    for path in paths:
        match = re.fullmatch(
            r"real_sampler_(distributed|taskbalanced)_workers(\d+)_t(\d+)\.json",
            path.name)
        if match:
            mode, workers, trial = match.groups()
            data = load(path)
            sampler_key(match, data)
            sampler_trials.setdefault(
                (int(workers), int(trial)), {})[mode] = data
    paired_sampler = {}
    for workers in sorted({key[0] for key in sampler_trials}):
        deltas = []
        for (worker_count, _trial), modes in sorted(sampler_trials.items()):
            if (worker_count != workers
                    or not {"distributed", "taskbalanced"} <= modes.keys()):
                continue
            distributed = modes["distributed"]
            balanced = modes["taskbalanced"]
            assert_same_experiment(
                distributed,
                balanced,
                {
                    "sampler", "sampler_implementation",
                    "sampler_total_size",
                    "sampler_preserves_global_batch_multiset",
                },
            )
            left_steps = distributed["measurement"][
                "task_counts_by_step_by_rank"]
            right_steps = balanced["measurement"][
                "task_counts_by_step_by_rank"]
            if len(left_steps) != len(right_steps):
                raise ValueError("paired sampler step counts differ")
            for left_ranks, right_ranks in zip(left_steps, right_steps):
                left_global = [
                    sum(rank[task] for rank in left_ranks)
                    for task in range(len(left_ranks[0]))
                ]
                right_global = [
                    sum(rank[task] for rank in right_ranks)
                    for task in range(len(right_ranks[0]))
                ]
                if left_global != right_global:
                    raise ValueError(
                        "paired sampler changed a global-step task multiset")
            deltas.append(
                balanced["measurement"]["critical_iteration_ms_mean"]
                - distributed["measurement"]["critical_iteration_ms_mean"])
        if deltas:
            paired_sampler[f"workers{workers}_taskbalanced_minus_distributed_ms"] = (
                stats(deltas))
    result["paired_sampler_deltas"] = paired_sampler

    known_patterns = (
        r"comm_w\d+_(?:default|noop|timed|sync)_t\d+\.json",
        r"batch_w\d+_bs\d+_(?:default|noop|timed)_t\d+\.json",
        r"pressure_w\d+_mb[\d.]+_t\d+\.json",
        r"rounds_w\d+_r\d+_t\d+\.json",
        r"gpu_card_\d+_t\d+\.json",
        r"straggler_gpu_[\d.]+ms_t\d+\.json",
        r"cost_(?:none|skewed|balanced)_[\d.]+ms_t\d+\.json",
        r"task_(?:sparse|grouped)_(?:balanced|homogeneous)_t\d+\.json",
        r"real_w\d+_(?:default|noop|timed|sync)_t\d+\.json",
        r"real_workers\d+_(?:none|rank|isolated)_t\d+\.json",
        r"real_workers\d+_(?:normal|reverse)_t\d+\.json",
        r"rankdiag_(?:normal|reverse)_(?:none|rank)_t\d+\.json",
        r"compile_after_w\d+_t\d+\.json",
        r"real_sampler_(?:distributed|taskbalanced)_workers\d+_t\d+\.json",
    )
    unmatched = [
        path.name for path in paths
        if not any(re.fullmatch(pattern, path.name)
                   for pattern in known_patterns)
    ]
    if unmatched:
        raise ValueError(f"unmatched input files: {unmatched}")
    result["unmatched_files"] = []

    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
