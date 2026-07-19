# §4.5 efficiency evidence

本目录保存 `EXP-INFRA-03-grouped-moe-ddp-evidence.md` 和 `content/section-efficiency.md` 使用的可复现实验证据。

## 目录

- `data/`：核心实验 JSON；`data/bottleneck/` 保存通信分解、快慢卡、真实 DataLoader 和任务均衡实验。
- `figures/`：`script/make_efficiency_figures.py` 从 JSON 生成的 SVG/PNG；SVG 用于论文排版，PNG 用于 Markdown 预览。

## 数据分组

| 文件前缀 | 内容 |
|---|---|
| `core_*` | sparse/grouped × eager/compile 交叉消融 |
| `trained_*` | 加载 `model_26.pth` 的单卡复测 |
| `experts_*` | E=4/8/12/16/24/32 专家数扫描 |
| `cap_*`、`batch_*` | 容量因子、显存可行性与最大已验证 batch |
| `capacity_quality.json` | 三任务各 15 样本冻结质量探针 |
| `profile_*` | eager 算子 profile 汇总 |
| `nccl_world*.json` | 4/8 卡 all-reduce 微基准 |
| `ddp_bucket_*` | DDP bucket cap 扫描 |
| `ddp_ablation_*` | bucket view、fused Adam、static graph 消融 |
| `ddp_final_bucket8*`、`ddp_comm_*` | default/sync/noop 通信对照 |
| `ddp_scale_*` | 1/2/4/8 卡扩展 |
| `ddp_straggler_*` | 单 rank 输入停顿敏感性 |
| `bottleneck/comm_*` | 4/8 卡 default/noop/sync/timed 三次独立重复 |
| `bottleneck/batch_*`、`pressure_*`、`rounds_*` | 计算强度边界与额外通信压力 |
| `bottleneck/rankdiag_*`、`gpu_card_*` | GPU 映射反转、NUMA 绑核与逐物理卡速度 |
| `bottleneck/real_*`、`real_workers*`、`real_sampler_*` | 真实数据阶段计时、worker 与同样本任务均衡对照 |
| `bottleneck_summary.json` | `summarize_bottleneck_evidence.py` 生成的独立试验汇总与 95% CI |

## 重新出图

```bash
cd /ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper-md
/opt/conda/envs/py310/bin/python3 script/summarize_bottleneck_evidence.py
/opt/conda/envs/py310/bin/python3 script/make_efficiency_figures.py
```

质量边界：`capacity_quality.json` 是冻结权重的即时 dispatch 切换探针，不是各容量独立训练后的最终质量。分布式边界：当前数据仅代表单机 H800 NVLink/NVSwitch，不外推跨节点网络。
