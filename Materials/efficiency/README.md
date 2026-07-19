# §4.5 efficiency evidence

本目录保存 `EXP-INFRA-03-grouped-moe-ddp-evidence.md` 和 `content/section-efficiency.md` 使用的可复现实验证据。

## 目录

- `data/`：每次实验的原始 JSON；成功记录包含环境、完整配置、逐次/逐步计时和统计量，失败记录使用 `experiment-failure-v1` schema。
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

## 重新出图

```bash
cd /ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper-md
/opt/conda/envs/py310/bin/python3 script/make_efficiency_figures.py
```

质量边界：`capacity_quality.json` 是冻结权重的即时 dispatch 切换探针，不是各容量独立训练后的最终质量。分布式边界：当前数据仅代表单机 H800 NVLink/NVSwitch，不外推跨节点网络。
