# EXP-INFRA-01：v3 训练提速（profile 驱动的 AI-infra 调优）

- 日期：2026-07-02　模型：v3 = oc96-depth4 MoE（blend 头, ws8, 12 路由专家 top-2 稀疏调度, ~4.1M）
- 目标：降低单网络训练时间。方法：先 profile 定位瓶颈 → 逐项优化并测量 → 记录"为什么做 + 效果"。
- 环境：跳板机 8×H800；Track A 用 GPU 0–3。测量口径：bs10/GPU、170 裁块、profile_step.py 20 步均值 step time；epoch = 1061 step。

## 1. Profile 定位瓶颈（`bench/profile_step.py --profile`）
单步 CUDA 时间 Top 项：
| op | CUDA% | 来源 |
|---|---|---|
| aten::mm / addmm / linear | **~45%** | MoE 专家 FFN 的大量**小 linear**（12 专家×3 伪尺度×depth4×block → 数千次小 mm）|
| aten::bmm | 11.6% | 窗口注意力的批量矩阵乘 |
| IndexBackward0 | 18%(CPU) | **稀疏 top-k 调度**的 index/nonzero |
| aten::copy_ | 7% | 调度的 gather/scatter 拷贝 |

**结论**：瓶颈是 **MoE 稀疏专家调度**（大量小算子 + Python 循环 + 索引/拷贝开销），其次是注意力 bmm。→ 优化方向：算子融合（减少小 kernel 数）+ 数据并行。

## 2. 逐项优化与实测（why + effect）
基线（1-GPU, sparse, fp32）：**560.7 ms/step**，70.8GB，吞吐 17.9 samples/s，epoch ≈ **594s**。

| 优化 | 为什么做 | 实测 | 结论 |
|---|---|---|---|
| **torch.compile** | 把数千小算子做图融合、降 kernel launch 开销（正是 profile 的瓶颈） | 560→**472 ms/step (1.19×)** | ✅ 采用（免费增益）|
| **bf16 autocast** | H800 张量核，常规提速 2× | 560→**1540 ms (2.7× 变慢！)** | ❌ **反例**：本负载是海量小算子（小专家/index），bf16 的 cast 开销 + 小 GEMM 用不满张量核 + RMI 的 cholesky 在 bf16 不稳，得不偿失。**不采用**。|
| **batched MoE（sonic-moe 式，专家权重堆叠+einsum 融合）** | 用 2 个大 GEMM 替 12 个小 linear+index_add，减 launch | **OOM**（E=12×T≈289k×H → 5GB+ 中间量） | ❌ 稠密 batched 在此 token 量下显存爆；稀疏路径反而省显存。**不采用**（保留 `combine="batched"` 开关）。|
| **DDP 4 卡数据并行** | 模型小(4M)、profile 表明 compute 受小算子拖累 → 直接靠数据并行摊薄 wall-clock | epoch 594→**~224s (2.6×)** | ✅ 采用（`train_moe_ddp.py`，`find_unused_parameters=True` 因稀疏路由有未用专家）|
| **DDP-4 + compile** | 叠加两者 | epoch → **~197s** | ✅ **最终：3.0× wall-clock 提速**（594→197s）|
| touch-all + find_unused=False | 试图去掉 DDP 未用参数遍历开销 | 无额外增益（~198s） | DDP 是**通信受限**（4M 模型的大量微小专家梯度→ all-reduce 延迟受限），非未用参数受限 |

## 3. 最终方案与效果
**v3 训练：单卡 594s/epoch → DDP-4 + torch.compile 197s/epoch，≈ 3.0× 提速。**
```
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_moe_ddp.py \
  --config ... --out-channel 96 --depth 4 --window-size 8 --n-routed 12 --compile
```
注：DDP 有效 batch = 4×per-gpu-bs，正式训练按线性缩放规则相应调 LR。

## 4. AI-infra 洞见（可写入论文/报告）
1. **profile 先行**：本模型瓶颈是"大量小算子/稀疏调度开销"而非算力，决定了优化取舍。
2. **bf16 不是万灵药**：小算子 + 索引 + cholesky 主导的负载，bf16 反而更慢——与"大 GEMM 主导的大模型"结论相反。
3. **稠密 batched-MoE ↔ 稀疏调度是显存/launch 的权衡**：本规模(12 专家、~29 万 token/前向)下稠密融合 OOM，稀疏 + torch.compile 更实际。
4. **小模型 DDP 通信受限**：4M 模型的大量微小梯度使 all-reduce 延迟受限，4 卡实得 ~2.6–3×（非线性 4×）；提高 per-GPU batch 或梯度分桶可进一步改善。
5. FA3/FlashAttention：窗口 ws8→每窗仅 64 token，注意力占比 11.6%，且已被 torch.compile 部分融合；收益有限，未单独接入（可用 `F.scaled_dot_product_attention` 走 Hopper flash 内核，列为后续）。

## 5. 产物
`bench/profile_step.py`（profile+计时，含 --amp/--compile/--combine）、`train_moe_ddp.py`（DDP+compile+touch-all）、`net_moe.py` 新增 `set_combine("batched")` 融合专家路径（可选）。
