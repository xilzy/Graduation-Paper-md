# 4.5　训练效率与分布式优化

本文方法只有约 4.11 M 参数，但 170×170 多尺度特征、12 路由专家 top-2 以及多分支融合损失使训练峰值显存达到 61–76 GB。因而，“参数少”不等于“训练开销小”。本节遵循“**profile 定位—执行形状改造—质量约束—分布式临界路径验证**”的顺序：先确认瓶颈，再解释 torch.compile、SDPA 和分组容量 MoE 三项单卡创新为何有效，最后用 NCCL、DDP 分桶、1/2/4/8 卡扩展和 straggler 对照重新判断分布式瓶颈。

除特别说明外，实验在单机 8×NVIDIA H800、PyTorch 2.9.0+cu128、NCCL 2.27.5 上完成；单卡采用 batch=10、170×170 输入，5 步预热后做 3×20 步；DDP 每卡 batch=10，10 步预热后测 20–30 步。性能实验严格加载冻结权重 `model_26.pth`，学习率置零以消除权重漂移，但保留完整前向、融合损失、反向和优化器。完整实验记录及原始 JSON 见 `../EXP-INFRA-03-grouped-moe-ddp-evidence.md` 和 `../Materials/efficiency/data/`。

## 4.5.1　profile 先行：瓶颈是执行碎片而非参数通信

原始 sparse MoE 对每个专家分别执行 `nonzero → gather → FC1 → GELU → FC2 → index_add`，造成大量动态 shape 小算子。表 4-15 给出同一训练步的关键 profile。

**表 4-15　sparse MoE 关键算子 profile**

| 算子 | 调用数 | CUDA 总时长 | 瓶颈来源 |
|---|---:|---:|---|
| `aten::linear` | 1504 | 236.30 ms | 多尺度、逐专家小 linear |
| `aten::mm` | 3168 | 449.07 ms | 小矩阵前向与反向 |
| `AddmmBackward0` | 1344 | 435.38 ms | 专家 FFN 反向 |
| `IndexBackward0` | 1200 | 100.06 ms | top-k gather/index 反向 |
| `aten::copy_` | 3372 | 138.38 ms | dispatch gather/scatter |
| `aten::bmm` | 488 | 231.81 ms | 注意力及其他批量矩阵乘 |

**结果分析。** 热点不是 4.11 M 参数本身，而是“数千次小 GEMM + 动态索引 + kernel launch”。这决定了优化方向：减少中间张量与 launch、将动态专家执行改造成规则批量 GEMM，而不是直接套用面向大参数模型的 ZeRO/张量并行。

## 4.5.2　三项单卡创新及其耦合关系

### 4.5.2.1　torch.compile：收益取决于图是否规则

![torch.compile 原理图](../Materials/efficiency/figures/compile_fusion_principle.png)

torch.compile 可以融合相邻 pointwise 算子并降低 Python/launch 开销，但无法凭空消除数据依赖的动态 shape。为区分“编译器收益”和“dispatch 形状收益”，本文做 2×2 交叉消融。

**表 4-16　dispatch×compile 交叉消融（vanilla attention）**

| dispatch | 模式 | ms/step | 相对 sparse eager |
|---|---:|---:|---:|
| sparse | eager | 552.63 | 1.000× |
| grouped, α=1.25 | eager | 549.21 | 1.006× |
| sparse | compile | 473.40 | 1.167× |
| grouped, α=1.25 | compile | **427.47** | **1.293×** |

**结果分析。** grouped 单独只快 0.6%，compile 单独快 16.7%，二者联合快 29.3%；grouped+compile 相对 sparse+compile 仍快 10.7%。因此创新点不是简单地“以 bmm 替换 linear”，而是先把执行形状改造成编译器友好的固定容量区域，再由 compile 将规则区域的收益兑现为墙钟时间。

![grouped 与 compile 交互证据](../Materials/efficiency/figures/grouped_compile_synergy.png)

### 4.5.2.2　SDPA：不改变权重，消除注意力中间矩阵

![SDPA 原理图](../Materials/efficiency/figures/sdpa_principle.png)

手写窗口注意力依次执行 `QKᵀ`、相对位置偏置、softmax 和 `P·V`，并显式保存 `[B, heads, N, N]` 中间矩阵。`scaled_dot_product_attention` 将该过程交给 PyTorch 后端，使用分块/融合内核时可避免将完整注意力矩阵往返全局显存；模型权重和注意力方程保持不变。

**表 4-17　SDPA 与 compile/grouped 的组合结果（bs=10）**

| 配置 | ms/step | 峰值显存 | 相对 sparse-eager |
|---|---:|---:|---:|
| sparse + vanilla + eager | 552.63 | 70.83 GB | 1.000× |
| sparse + vanilla + compile | 473.40 | 68.80 GB | 1.167× |
| sparse + SDPA + compile | 474.05 | **61.32 GB** | 1.166× |
| grouped + vanilla + eager | 549.21 | 77.04 GB | 1.006× |
| grouped + vanilla + compile | 427.47 | 77.03 GB | 1.293× |
| grouped + SDPA + compile | **427.11** | **69.51 GB** | **1.294×** |

**结果分析。** SDPA 在当前小窗口上主要贡献显存而非额外速度：sparse compile 显存减少约 7.5 GB，grouped compile 也减少约 7.5 GB，单步时间基本不变。该显存恰好抵消 grouped 固定容量缓冲的大部分成本，使“规则专家 GEMM”和“较低注意力激活”可以同时采用。同权重对拍的最大绝对误差为 1.7e-5，满足数值一致性要求。

### 4.5.2.3　分组容量 MoE：改变执行形状，不改变 top-k 规则

![分组容量 MoE 原理图](../Materials/efficiency/figures/grouped_moe_principle.png)

设 token 数为 T、top-k 为 k、专家数为 E。分组路径先将 `D=T·k` 个 dispatch 按专家排序，用每个专家的起始偏移得到桶内位置，再按

\[
\mathrm{cap}=\left\lfloor \alpha\frac{Tk}{E}\right\rfloor
\]

构造 `[E, cap, C]` 右填充缓冲。所有专家的 FC1/FC2 因而收敛为两次批量 GEMM，最后按 gate 权重散回 token。超过容量的 dispatch 被丢弃；α 控制速度、显存和质量之间的权衡。

**表 4-18　分组前后的算子结构**

| 算子 | sparse：调用数 / CUDA | grouped：调用数 / CUDA | 变化 |
|---|---:|---:|---:|
| `aten::linear` | 1504 / 236.30 ms | 352 / 98.52 ms | 调用数 -76.6% |
| `aten::mm` | 3168 / 449.07 ms | 864 / 197.55 ms | 调用数 -72.7% |
| `AddmmBackward0` | 1344 / 435.38 ms | 192 / 183.88 ms | 调用数 -85.7% |
| `IndexBackward0` | 1200 / 100.06 ms | Top-20 中消失 | 动态索引反向退出主路径 |
| `aten::bmm` | 488 / 231.81 ms | 776 / 746.32 ms | 规则批量 GEMM成为主算子 |

**结果分析。** grouped 确实将大量小 linear/mm 和索引反向收敛为 bmm；但 padding 使 bmm 总时长上升，所以 eager 总时间没有明显改善。该表与表 4-16 共同说明：结构改造和 compile 是不可拆分的联合创新。`copy_` 仍为 3324 次/136.79 ms，表明后续优化空间主要在 gather/scatter 融合。

**表 4-19　专家数 E 对 grouped 收益的影响（SDPA+compile）**

| E | sparse ms | grouped ms | grouped 单步降幅 |
|---:|---:|---:|---:|
| 4 | 405.76 | 511.47 | -26.1% |
| 8 | 437.64 | 436.78 | +0.2% |
| 12 | 470.78 | 427.35 | +9.2% |
| 16 | 505.50 | 428.63 | +15.2% |
| 24 | 595.99 | 426.86 | +28.4% |
| 32 | 683.39 | 435.80 | **+36.2%** |

**结果分析。** sparse 的逐专家循环随 E 增长，grouped 始终保持两次专家批量 GEMM；交叉点约为 E=8。当前 E=12 已有 9.2% 单步降幅，E=32 达 36.2%，证明该方案针对的是“多专家、小专家”的可扩展瓶颈，而非偶然优化一个固定配置。

![专家数扩展曲线](../Materials/efficiency/figures/expert_count_scaling.png)

### 4.5.2.4　容量—吞吐—质量的 Pareto 选择

![容量与负载均衡原理](../Materials/efficiency/figures/capacity_balance_principle.png)

**表 4-20　预训练权重下的容量性能与冻结质量探针**

| 配置 | bs=10 samples/s | 峰值显存 | dispatch 丢弃 | 输出 MAE vs sparse | ΔMI | ΔVIF |
|---|---:|---:|---:|---:|---:|---:|
| sparse | 21.095 | 61.32 GB | 0 | 0 | 0 | 0 |
| α=1.00 | **25.435** | 62.75 GB | 6.748% | 2.63e-4 | -2.92e-2 | -1.40e-3 |
| α=1.25 | 23.413 | 69.51 GB | 0.800% | 3.67e-5 | -5.71e-3 | -2.48e-4 |
| α=1.50 | 21.669 | 76.20 GB | 0.121% | 3.13e-6 | -8.67e-4 | -3.20e-5 |
| α=2.00 | bs=10 OOM | — | 0.0039% | 7.82e-8 | +1.07e-5 | +6.64e-8 |
| α=4.00 | bs=10 OOM | — | 0 | 0 | 0 | 0 |

**结果分析。** α=1.0 吞吐提高 20.6%，但 6.75% 丢弃已造成可见 MI/VIF 下降；α=1.25 吞吐提高 11.0%，总体丢弃降至 0.80%，五项融合指标变化很小，是速度档；α=1.5 基本贴近 sparse，但吞吐收益只剩 2.7%，是保守档；α≥2 数值等价却无法维持 bs=10。质量实验覆盖三任务各 15 个真实样本，只能证明冻结模型的即时扰动很小，不能代替不同 α 独立训练后的最终质量统计。

![容量 Pareto 曲线](../Materials/efficiency/figures/capacity_quality_pareto.png)

在最大可运行 batch 下，sparse 的已验证最大吞吐为 22.20 samples/s（bs=13），α=1.0 为 25.83（bs=12，+16.4%），α=1.25 为 23.71（bs=11，+6.8%）；α≥1.5 因容量缓冲挤占 batch 空间而不再具备系统吞吐优势。故本文将 α=1.25 作为性能配置、α=1.5 作为质量保守配置，并保留 sparse 零丢弃回退。

## 4.5.3　分布式优化：从“通信猜测”转向临界路径证据

### 4.5.3.1　为何当前选择 DDP 而不是 FSDP/EP

4.11 M FP32 参数、梯度及 Adam 两个状态合计不足 0.1 GB，而训练峰值为 61–76 GB，显存显然由激活、损失分支和容量缓冲主导。FSDP/ZeRO 切分不到 0.2% 的峰值，却需要额外 reduce-scatter/all-gather；Expert Parallel 则会将当前本地专家 bmm 改成 token all-to-all。因此当前最优并行维度是纯数据并行，分布式优化应集中在梯度缓冲、分桶重叠和 rank 负载均衡。

**表 4-21　开源框架机制与当前模型的匹配**

| 机制 | 解决对象 | 本模型结论 |
|---|---|---|
| PyTorch DDP 分桶、bucket view、static graph | 梯度归约与图搜索 | **采用** |
| Megatron overlap-grad-reduce | 通信与反向重叠 | DDP reducer 已提供同类能力 |
| Megatron distributed optimizer / param gather | 大参数状态切分 | 参数太小，暂不采用 |
| DeepSpeed ZeRO/FSDP | 参数/梯度/优化器显存 | 激活才是瓶颈，暂不采用 |
| Expert Parallel A2A | 单卡放不下的大量专家 | 12 个小专家可本地容纳，暂不采用 |
| 固定 shape + 任务/成本均衡分片 | rank straggler | **采用为数据侧原则** |

**结果分析。** 本文吸收大规模框架的“连续缓冲、异步分桶、静态图”思想，但不机械照搬模型状态切分和专家并行。优化并行维度必须由参数、激活和通信的实测比例决定。

### 4.5.3.2　NCCL 链路与模型大小感知分桶

4/8 卡 all-reduce 微基准显示，16 KiB–1 MiB 消息主要受约 40 μs 固定启动延迟控制；4/8/16 MiB 时 4 卡总线带宽分别达到 104.85/161.94/217.62 GB/s。当前全部梯度只有约 15.67 MiB，一次归约约百微秒，远小于约 422 ms 的计算步。

![NCCL 桶曲线](../Materials/efficiency/figures/nccl_bucket_curve.png)

![DDP 分桶与重叠原理](../Materials/efficiency/figures/ddp_overlap_principle.png)

**表 4-22　DDP-4 梯度分桶扫描**

| bucket cap | 重建桶数 | ms/step | global samples/s |
|---:|---:|---:|---:|
| 1 MiB | 15 | 427.75 | 93.512 |
| 2 MiB | 8 | 426.36 | 93.817 |
| 4 MiB | 4 | 426.23 | 93.847 |
| 8 MiB | 2 | **425.57** | **93.993** |
| 25 MiB | 1 | 426.78 | 93.726 |

**结果分析。** 1 MiB 重复支付 15 次 collective 启动延迟；25 MiB 大于全部梯度，只形成一个桶，不能在反向中提前启动；8 MiB 重建为约 8.09/7.58 MiB 两个高带宽桶，在启动次数和重叠机会之间最优。最终 static-graph 配置中，8 MiB 相比 25 MiB 也将 423.20 ms 降到 422.32 ms，方向一致。0.5 MiB 因小于 PyTorch 1 MiB 首桶约束而不支持。

![DDP 分桶实测](../Materials/efficiency/figures/ddp_bucket_sweep.png)

### 4.5.3.3　连续梯度、fused Adam 与静态图

**表 4-23　DDP-4 逐项消融**

| 配置 | ms/step | global samples/s | 相对上一项 |
|---|---:|---:|---:|
| baseline：25 MiB、find-unused | 439.63 | 90.986 | — |
| + `gradient_as_bucket_view` | 437.70 | 91.388 | +0.44% |
| + fused Adam | 424.99 | 94.120 | +2.99% |
| + static graph、关闭 unused 搜索 | 423.56 | 94.437 | +0.34% |
| + 8 MiB 两桶 | **422.32** | **94.716** | +0.30% |

**结果分析。** 累计吞吐提升 4.10%，最大贡献来自 fused Adam，说明大量小参数的 optimizer launch 比 NCCL 更值得优化。bucket view 避免梯度到通信桶的拷贝；DDP 日志只发现 768 B unused 参数，且使用集合固定，因此可以安全启用 static graph。最终 8 MiB 下 noop/default/sync 分别为 422.63/422.32/422.83 ms，最大差 0.51 ms，小于单步 0.62–0.68 ms 标准差，说明通信暴露已进入测量噪声，不能再把单卡—DDP 差值全部解释为 NCCL。

### 4.5.3.4　卡数扩展与最慢 rank

**表 4-24　1/2/4/8 卡扩展（每卡 batch=10）**

| GPU 数 | ms/step | global samples/s | 扩展效率 |
|---:|---:|---:|---:|
| 1 | 421.37 | 23.732 | 100.00% |
| 2 | 422.63 | 47.323 | 99.70% |
| 4 | 423.20 | 94.519 | 99.56% |
| 8 | 425.22 | 188.138 | **99.08%** |

**结果分析。** 8 卡达到单卡吞吐的 7.93×。这修正了旧版“轻量模型 DDP 严重通信受限”的推断：在稳定输入和单机高速互连下，当前模型近似线性扩展；分布式下一瓶颈不是 all-reduce，而是每卡计算与输入供给。

**表 4-25　单 rank straggler 敏感性**

| rank0 注入停顿 | DDP-4 ms/step | 全局额外耗时 |
|---:|---:|---:|
| 0 ms | 422.32 | 0 |
| 2 ms | 424.41 | +2.10 ms |
| 5 ms | 427.52 | +5.20 ms |
| 10 ms | 432.55 | +10.24 ms |

**结果分析。** 一个 rank 的停顿几乎 1:1 进入全局步时，其他卡在 collective 等待；同步后的 rank 完成时间差仍只有约 0.08 ms，因此仅看 step 尾部 CV 会掩盖上游 straggler。生产训练需保持固定 crop、每 rank 等 batch、`drop_last=True`，并按任务和样本成本分层打散；若引入变分辨率，应按像素/token 数或历史耗时均衡，而不是只均衡样本数。

![DDP 扩展与 straggler](../Materials/efficiency/figures/ddp_scaling_straggler.png)

![rank 负载均衡原理](../Materials/efficiency/figures/ddp_rank_balance_principle.png)

路由负载本身也存在 rank 差异：四个 rank 的 expert load CV 为 0.148–0.154，dispatch 丢弃率为 0.578%–0.733%。但 grouped 在每个 rank 都执行相同 `[E, cap, C]` 主 GEMM，最终 rank 时间 CV 仍约 0.03%。固定容量因此不仅利于 compile，也将路由不均与主计算形状解耦，是分布式负载稳定器。

## 4.5.4　本节小结

本节得到五点结论：

1. **profile 决定优化对象。** 当前热点是小 GEMM、动态索引和 launch，而非参数规模；
2. **三项单卡创新分工明确。** compile 消 launch，SDPA 省注意力中间激活，grouped 将专家执行改造成规则批量 GEMM；grouped 与 compile 联合达到 1.294×，而非各自收益的简单相加；
3. **grouped 的价值随专家数增长。** E=12 已降低 9.2% 单步，E=32 降低 36.2%，但 E=4 会变慢；
4. **容量必须与质量和显存共同选择。** α=1.25 是速度档，α=1.5 是保守档，sparse 是零丢弃回退；冻结探针不能替代重新训练；
5. **当前分布式不是通信受限。** 最终 DDP 1→8 卡效率为 99.08%，真正需要防范的是单 rank 输入/计算 straggler。推荐配置为 `grouped α=1.25 + SDPA + compile + gradient_as_bucket_view + fused Adam + static_graph + 8 MiB bucket`。

局限包括：质量仅做 45 样本冻结探针；多卡实验仅覆盖单机高速互连，不能外推跨节点；通信模式差异已进入噪声，本文不报告不可靠的隐藏比例；gather/scatter 的 `copy_` 尚未被融合，是下一步自定义 Triton kernel 的候选方向。
