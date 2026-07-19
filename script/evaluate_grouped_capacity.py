#!/usr/bin/env python3
"""Evaluate the grouped-MoE capacity/quality trade-off on a frozen checkpoint.

For every probe image the sparse path is the paired control. Grouped paths reuse
exactly the same weights and differ only in dispatch layout/capacity, allowing a
causal measurement of token drops and objective-quality changes.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("MI", "SSIM", "Qabf", "VIF", "Nabf")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--code-root", type=Path,
                   default=Path("/ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper"))
    p.add_argument("--checkpoint", type=Path,
                   default=Path("models/Ours_v3_frozen/model_26.pth"))
    p.add_argument("--configs", nargs="*", default=(
        "configs/gfp_pc.json", "configs/irvis_msrs.json", "configs/medical_harvard.json"))
    p.add_argument("--cap-factors", nargs="+", type=float,
                   default=(1.0, 1.25, 1.5, 2.0, 4.0))
    p.add_argument("--max-samples", type=int, default=0,
                   help="0 uses every frozen probe sample")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path)
    return p.parse_args()


def read_saved_args(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        try:
            out[key.strip()] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            out[key.strip()] = value
    return out


def y255(path: str) -> np.ndarray:
    from PIL import Image
    image = Image.open(path)
    if image.mode in ("RGB", "RGBA", "P", "CMYK"):
        return np.asarray(image.convert("YCbCr"), np.float32)[:, :, 0]
    return np.asarray(image.convert("L"), np.float32)


def safe_mean(values: list[float]) -> float:
    return float(np.nanmean(values)) if values else float("nan")


def safe_std(values: list[float]) -> float:
    return float(np.nanstd(values)) if values else float("nan")


def main() -> None:
    a = parse_args()
    code_root = a.code_root.resolve()
    sys.path.insert(0, str(code_root))

    import torch
    from PIL import Image
    import metrics as metric_lib
    import mm_fusion_data as mfd
    from Networks.net_moe import MODEL_MoE, MoEFFN

    output_path = a.output.resolve() if a.output else None
    checkpoint = a.checkpoint if a.checkpoint.is_absolute() else code_root / a.checkpoint
    configs = [Path(c) if Path(c).is_absolute() else code_root / c for c in a.configs]
    # GFP-PC paths are intentionally repository-relative in its frozen config.
    os.chdir(code_root)
    saved = read_saved_args(checkpoint.parent / "args.txt")

    def flag(name: str, default: bool = False) -> bool:
        value = saved.get(name, default)
        return value if isinstance(value, bool) else str(value).lower() == "true"

    model = MODEL_MoE(
        in_channel=int(saved.get("in_channel", 2)),
        n_tasks=len(configs),
        out_channel=int(saved.get("out_channel", 96)),
        depth=int(saved.get("depth", 4)),
        num_heads=int(saved.get("num_heads", 8)),
        window_size=int(saved.get("window_size", 8)),
        n_routed=int(saved.get("n_routed", 12)),
        k=int(saved.get("k", 2)),
        n_shared=int(saved.get("n_shared", 1)),
        task_cond=not flag("no_task_cond"),
        out_scale=flag("out_scale", True),
        use_task_bias=not flag("no_task_bias"),
        fusion_head=str(saved.get("fusion_head", "blend")),
        res_scale=float(saved.get("res_scale", 0.0)),
        routing=str(saved.get("routing", "softmax")),
        per_task_head=flag("per_task_head"),
        attn_impl="vanilla",
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().to(a.device)

    modes = [("sparse", None)] + [(f"cap{cap:g}", cap) for cap in a.cap_factors]
    current = {"mode": "sparse", "cap": None, "calls": []}

    def route_hook(module: MoEFFN, args: tuple) -> None:
        x, task_emb = args
        batch, tokens, channels = x.shape
        flat = x.reshape(-1, channels)
        gate_in = flat
        if module.task_cond and task_emb is not None:
            gate_in = flat + task_emb[:, None, :].expand(
                batch, tokens, channels).reshape(-1, channels)
        logits = module.gate(gate_in)
        if module.routing == "deepseek":
            affinity = torch.sigmoid(logits)
            topi = (affinity + module.ebias).topk(module.k, dim=-1).indices
        else:
            topi = torch.softmax(logits, dim=-1).topk(module.k, dim=-1).indices
        counts = torch.bincount(topi.reshape(-1), minlength=module.n_routed)
        dispatches = int(topi.numel())
        counts_np = counts.detach().cpu().numpy().astype(np.int64)
        row = {
            "dispatches": dispatches,
            "load_cv": float(counts_np.std() / max(counts_np.mean(), 1e-12)),
            "max_over_mean": float(counts_np.max() / max(counts_np.mean(), 1e-12)),
        }
        if current["cap"] is not None:
            capacity = max(1, int(float(current["cap"]) * dispatches / module.n_routed))
            kept = int(np.minimum(counts_np, capacity).sum())
            row.update({
                "capacity_slots": int(module.n_routed * capacity),
                "kept": kept,
                "dropped": dispatches - kept,
                "drop_rate": (dispatches - kept) / dispatches,
            })
        current["calls"].append(row)

    hooks = [m.register_forward_pre_hook(route_hook) for m in model.modules()
             if isinstance(m, MoEFFN)]

    accum = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    route_accum = defaultdict(lambda: defaultdict(list))
    sample_counts = defaultdict(int)

    with torch.no_grad():
        for task_id, config_path in enumerate(configs):
            cfg = mfd.load_config(str(config_path))
            task = cfg["task"]
            probe = set(mfd.probe_stems(cfg))
            pairs = [pair for pair in mfd.list_pairs(cfg, "test") if pair[0] in probe]
            if a.max_samples:
                pairs = pairs[:a.max_samples]
            sample_counts[task] = len(pairs)

            for stem, src_a_path, src_b_path in pairs:
                src_a = y255(src_a_path)
                src_b = y255(src_b_path)
                if src_b.shape != src_a.shape:
                    src_b = np.asarray(
                        Image.fromarray(src_b.astype("uint8")).resize(
                            (src_a.shape[1], src_a.shape[0]), Image.BILINEAR),
                        np.float32,
                    )
                tensor = torch.from_numpy(np.stack([src_a, src_b])[None] / 255.0)
                tensor = tensor.float().to(a.device)
                task_tensor = torch.tensor([task_id], device=a.device)
                sparse_output = None

                for mode, cap in modes:
                    current.update({"mode": mode, "cap": cap, "calls": []})
                    model.set_combine("sparse" if cap is None else "grouped",
                                      cap_factor=cap)
                    output, _ = model(tensor, task_tensor)
                    fused01 = output.squeeze().clamp(0, 1).cpu().numpy()
                    fused255 = fused01 * 255.0
                    values = metric_lib.compute_all(
                        src_a, src_b, fused255, include_diagnostic=True)
                    for metric in METRICS:
                        accum[task][mode][metric].append(float(values[metric]))

                    if sparse_output is None:
                        sparse_output = fused01
                        accum[task][mode]["mae_vs_sparse"].append(0.0)
                        accum[task][mode]["max_abs_vs_sparse"].append(0.0)
                    else:
                        delta = np.abs(fused01 - sparse_output)
                        accum[task][mode]["mae_vs_sparse"].append(float(delta.mean()))
                        accum[task][mode]["max_abs_vs_sparse"].append(float(delta.max()))
                    route_accum[task][mode].extend(current["calls"])

    for hook in hooks:
        hook.remove()

    tasks_out = {}
    for task, mode_map in accum.items():
        tasks_out[task] = {}
        baseline = {metric: safe_mean(mode_map["sparse"][metric]) for metric in METRICS}
        for mode, values in mode_map.items():
            calls = route_accum[task][mode]
            dispatches = sum(row["dispatches"] for row in calls)
            dropped = sum(row.get("dropped", 0) for row in calls)
            kept = sum(row.get("kept", row["dispatches"]) for row in calls)
            slots = sum(row.get("capacity_slots", 0) for row in calls)
            metric_means = {metric: safe_mean(values[metric]) for metric in METRICS}
            tasks_out[task][mode] = {
                "n": sample_counts[task],
                "metrics": metric_means,
                "metric_std": {metric: safe_std(values[metric]) for metric in METRICS},
                "delta_vs_sparse": {
                    metric: metric_means[metric] - baseline[metric] for metric in METRICS
                },
                "output_mae_vs_sparse": safe_mean(values["mae_vs_sparse"]),
                "output_max_abs_vs_sparse": max(values["max_abs_vs_sparse"]),
                "routing": {
                    "dispatches": dispatches,
                    "drop_pct": 100.0 * dropped / dispatches if dispatches else 0.0,
                    "max_layer_drop_pct": 100.0 * max(
                        (row.get("drop_rate", 0.0) for row in calls), default=0.0),
                    "occupancy_pct": 100.0 * kept / slots if slots else None,
                    "load_cv_mean": safe_mean([row["load_cv"] for row in calls]),
                    "max_over_mean_mean": safe_mean(
                        [row["max_over_mean"] for row in calls]),
                },
            }

    overall = {}
    for mode, _ in modes:
        total_n = sum(sample_counts.values())
        overall[mode] = {
            "metrics": {}, "delta_vs_sparse": {},
            "output_mae_vs_sparse": 0.0, "routing": {},
        }
        for metric in METRICS:
            overall[mode]["metrics"][metric] = sum(
                tasks_out[t][mode]["metrics"][metric] * sample_counts[t]
                for t in tasks_out) / total_n
            overall[mode]["delta_vs_sparse"][metric] = sum(
                tasks_out[t][mode]["delta_vs_sparse"][metric] * sample_counts[t]
                for t in tasks_out) / total_n
        overall[mode]["output_mae_vs_sparse"] = sum(
            tasks_out[t][mode]["output_mae_vs_sparse"] * sample_counts[t]
            for t in tasks_out) / total_n
        dispatches = sum(tasks_out[t][mode]["routing"]["dispatches"] for t in tasks_out)
        dropped = sum(
            tasks_out[t][mode]["routing"]["drop_pct"] / 100.0
            * tasks_out[t][mode]["routing"]["dispatches"] for t in tasks_out)
        overall[mode]["routing"] = {
            "dispatches": dispatches,
            "drop_pct": 100.0 * dropped / dispatches if dispatches else 0.0,
            "max_layer_drop_pct": max(
                tasks_out[t][mode]["routing"]["max_layer_drop_pct"] for t in tasks_out),
            "load_cv_mean": sum(
                tasks_out[t][mode]["routing"]["load_cv_mean"] * sample_counts[t]
                for t in tasks_out) / total_n,
        }

    result = {
        "schema": "grouped-moe-capacity-quality-v1",
        "checkpoint": str(checkpoint),
        "sample_counts": dict(sample_counts),
        "metric_directions": {
            "MI": "higher", "SSIM": "higher", "Qabf": "higher",
            "VIF": "higher", "Nabf": "lower",
        },
        "tasks": tasks_out,
        "overall": overall,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
