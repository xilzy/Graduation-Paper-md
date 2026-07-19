# EXP-INFRA-03：分组容量 MoE 多维证据与分布式瓶颈再评估

- 日期：2026-07-19
- 模型：v3，oc96-depth4、12 路由专家、top-2，约 4.11 M 参数
- 设备：单机 8×NVIDIA H800
- 软件：PyTorch 2.9.0+cu128、NCCL 2.27.5
- 权重：`models/Ours_v3_frozen/model_26.pth`，`strict=True` 加载
- 结果级别：性能为 S1 infra 基准；质量为冻结权重的 45 样本探针，不等同于重新训练后的最终质量
- 原始数据：`Materials/efficiency/data/*.json`
- 复现实验：`script/benchmark_grouped_moe.py`、`script/evaluate_grouped_capacity.py`、`script/benchmark_nccl_allreduce.py`、`script/benchmark_ddp_training.py`
- 出图脚本：`script/make_efficiency_figures.py`

本实验回答两个问题：

1. 分组容量 MoE 的收益究竟来自哪里，是否只在某个孤立配置上有效，以及容量因子带来的性能、显存、丢弃和质量代价是什么；
2. 当前 4.11 M 参数模型的多卡瓶颈究竟是 NCCL 通信、负载不均，还是单卡计算/激活，并据此选择 DDP、分桶、FSDP/ZeRO 或专家并行。

---

## 1　实验设计与边界

**表 E3-1　测量口径**

| 实验 | 输入与权重 | 预热/测量 | 主要指标 |
|---|---|---|---|
| 单卡训练步 | 预训练权重；bs=10、170×170；完整前向、融合损失、反向、Adam | 5 步预热；3×20 步 | ms/step、samples/s、峰值显存 |
| 专家数扫描 | 与单卡相同；E∈{4,8,12,16,24,32} | 5 步预热；3×20 步 | sparse/grouped 交叉点 |
| 容量质量探针 | 冻结 `model_26.pth`；3 任务各 15 样本 | 单次确定性推理 | MI、SSIM、Qabf、VIF、Nabf、输出 MAE、路由丢弃率 |
| NCCL 微基准 | 4/8 卡；16 KiB–16 MiB all-reduce | 10 次预热；100 次 | p50/p95 延迟、算法/总线带宽 |
| DDP 训练步 | 预训练权重；每卡 bs=10；完整损失；lr=0 | 10 步预热；20–30 步 | 临界 rank 单步、全局吞吐、rank 偏斜 |
| straggler 敏感性 | DDP-4 最优配置；只给 rank0 注入 2/5/10 ms 输入停顿 | 5 步预热；20 步 | 临界步额外耗时 |

**结果分析。** 性能实验使用真实模型、损失和反向图，仅将输入替换为固定随机张量，以隔离存储与 DataLoader 抖动；质量实验则使用真实融合样本。两者不能混为一谈：性能数据回答执行效率，冻结权重探针回答“切换 dispatch 后立即产生多大输出扰动”，不能替代按不同容量重新训练后的最终精度比较。

---

## 2　分组容量 MoE：机制与多维证据

### 2.1　为什么 grouped 必须与 compile 联合观察

![分组容量 MoE 原理](Materials/efficiency/figures/grouped_moe_principle.png)

![compile 与规则执行形状](Materials/efficiency/figures/compile_fusion_principle.png)

稀疏路径逐专家执行 `nonzero → gather → 两层 linear → index_add`。它节省 FLOPs，却把一次 MoE 变成 E 组动态 shape 小算子。grouped 路径不改变 top-k 路由规则，而是将 `D=T·k` 个 dispatch 按专家排序，计算桶内位置，并写入 `[E, cap, C]` 固定容量缓冲；专家 FFN 因而变为两个批量 GEMM。固定形状本身会引入 padding，所以 grouped 的目标不是“无条件减少 eager 时间”，而是把图改造成编译器和 GPU 更容易处理的规则形状。

**表 E3-2　dispatch 形状与 torch.compile 的 2×2 交叉消融（vanilla attention，bs=10）**

| dispatch | 执行模式 | ms/step | 相对 sparse eager 加速 |
|---|---:|---:|---:|
| sparse | eager | 552.63 | 1.000× |
| grouped, α=1.25 | eager | 549.21 | 1.006× |
| sparse | compile | 473.40 | 1.167× |
| grouped, α=1.25 | compile | **427.47** | **1.293×** |

**结果分析。** grouped 单独只快 0.6%，说明“把 linear 换成 bmm”不是收益的充分条件；compile 单独快 16.7%；二者联合快 29.3%，且 grouped+compile 比 sparse+compile 再快 10.7%。这验证了创新点的核心不是改变 MoE 数学，而是改变执行形状，使编译器能融合外围点算子并减少 Python/launch 开销。用预训练权重复测时，sparse=474.05 ms、grouped α=1.25=427.11 ms，仍为 1.110×，排除了随机初始化造成偶然性能结论的可能。

![grouped 与 compile 交互](Materials/efficiency/figures/grouped_compile_synergy.png)

### 2.2　算子结构是否真的发生改变

**表 E3-3　eager profile 的关键算子变化**

| 算子 | sparse：调用数 / CUDA 总时长 | grouped：调用数 / CUDA 总时长 | 结构变化 |
|---|---:|---:|---:|
| `aten::linear` | 1504 / 236.30 ms | 352 / 98.52 ms | 调用数 -76.6% |
| `aten::mm` | 3168 / 449.07 ms | 864 / 197.55 ms | 调用数 -72.7% |
| `AddmmBackward0` | 1344 / 435.38 ms | 192 / 183.88 ms | 调用数 -85.7% |
| `IndexBackward0` | 1200 / 100.06 ms | Top-20 中消失 | 稀疏索引反向被移出主路径 |
| `aten::bmm` | 488 / 231.81 ms | 776 / 746.32 ms | 规则批量 GEMM 成为主算子 |
| `aten::copy_` | 3372 / 138.38 ms | 3324 / 136.79 ms | 基本不变 |

**结果分析。** profile 直接证实：小 `linear/mm` 和索引反向显著减少，代价是 `bmm` 变成主算子。grouped eager 总步时与 sparse eager 接近，正是因为减少的 launch/index 时间被 padding 后的大 bmm 抵消；compile 能优化规则 bmm 周边图，才把结构变化兑现为墙钟收益。`copy_` 几乎不变，也说明下一轮若继续优化，应针对 gather/scatter 融合，而不是继续改专家 GEMM。

### 2.3　收益是否随专家数增长

**表 E3-4　专家数扫描（SDPA+compile，bs=10，α=1.25）**

| 路由专家 E | sparse ms | grouped ms | grouped 单步降幅 |
|---:|---:|---:|---:|
| 4 | 405.76 | 511.47 | -26.1%（更慢） |
| 8 | 437.64 | 436.78 | +0.2% |
| 12 | 470.78 | 427.35 | +9.2% |
| 16 | 505.50 | 428.63 | +15.2% |
| 24 | 595.99 | 426.86 | +28.4% |
| 32 | 683.39 | 435.80 | **+36.2%** |

**结果分析。** sparse 路径的 Python 循环和小 GEMM 数量随 E 近似增长；grouped 的两次批量 GEMM 数量不随 E 增长，主要变化是 batch 维。交叉点约在 E=8：少专家时 padding/排序成本不划算，E≥12 后收益稳定扩大，E=32 时单步降低 36.2%。因此该创新不是所有 MoE 的通用替换，而是面向“专家数较多、单专家矩阵较小”的执行优化。

![专家数交叉曲线](Materials/efficiency/figures/expert_count_scaling.png)

### 2.4　容量因子的速度—显存代价

容量为

\[
\mathrm{cap}=\left\lfloor \alpha \frac{T k}{E}\right\rfloor ,
\]

其中 α 为容量因子。α 越小，规则缓冲越小、GEMM 越快，但热点专家更容易溢出；α 越大，输出越接近 sparse，但 padding、激活和显存同步增加。

![容量与负载均衡原理](Materials/efficiency/figures/capacity_balance_principle.png)

**表 E3-5　固定 bs=10 的容量性能（预训练权重，SDPA+compile）**

| 配置 | ms/step | samples/s | 相对 sparse 吞吐 | 峰值显存 | 相对 sparse 显存 |
|---|---:|---:|---:|---:|---:|
| sparse | 474.05 | 21.095 | 1.000× | 61.32 GB | — |
| grouped α=1.00 | **393.16** | **25.435** | **1.206×** | 62.75 GB | +2.3% |
| grouped α=1.25 | 427.11 | 23.413 | 1.110× | 69.51 GB | +13.4% |
| grouped α=1.50 | 461.49 | 21.669 | 1.027× | 76.20 GB | +24.3% |

**结果分析。** α 每增加 0.25，固定容量缓冲近似线性增长，性能收益快速收窄。α=1.0 的速度最好，但后续质量表显示其丢弃率过高；α=1.25 保留 11.0% 吞吐收益，是性能与质量风险之间的激进甜点；α=1.5 仅剩 2.7% 吞吐收益，但输出更接近 sparse。α=2/4 在 bs=10 分别因下一笔 1.65 GiB/424 MiB 分配失败而 OOM，说明高容量不能只看数值等价。

**表 E3-6　各容量的最大已验证可运行 batch 与吞吐**

| 配置 | 最大已验证 bs | ms/step | samples/s | 峰值显存 | 相对 sparse 最大吞吐 |
|---|---:|---:|---:|---:|---:|
| sparse | 13 | 585.59 | 22.200 | 79.56 GB | 1.000× |
| grouped α=1.00 | 12 | 464.50 | **25.834** | 75.33 GB | **1.164×** |
| grouped α=1.25 | 11 | 463.89 | 23.713 | 76.42 GB | 1.068× |
| grouped α=1.50 | 10 | 461.49 | 21.669 | 76.20 GB | 0.976× |
| grouped α=2.00 | 8 | 427.95 | 18.694 | 71.57 GB | 0.842× |
| grouped α=4.00 | 4 | 344.67 | 11.605 | 57.20 GB | 0.523× |

**结果分析。** 用“最大 batch”衡量时，α=1.0/1.25 仍分别比 sparse 高 16.4%/6.8%；α≥1.5 则因容量缓冲挤占 batch 空间而失去系统吞吐优势。由此可见，容量不是越大越安全：高 α 虽减少 token 丢弃，却降低每卡有效样本数，最终使卡利用率变差。

### 2.5　容量是否损害冻结模型输出质量

**表 E3-7　45 样本冻结权重探针（3 任务各 15；Δ 均相对 sparse）**

| 配置 | dispatch 丢弃率 | 输出 MAE | ΔMI ↑ | ΔSSIM ↑ | ΔQabf ↑ | ΔVIF ↑ | ΔNabf ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| α=1.00 | 6.748% | 2.63e-4 | -2.92e-2 | +1.15e-4 | -6.35e-5 | -1.40e-3 | -2.98e-4 |
| α=1.25 | 0.800% | 3.67e-5 | -5.71e-3 | +1.08e-5 | -7.02e-6 | -2.48e-4 | -4.73e-6 |
| α=1.50 | 0.121% | 3.13e-6 | -8.67e-4 | +3.11e-7 | -8.08e-7 | -3.20e-5 | +2.57e-5 |
| α=2.00 | 0.0039% | 7.82e-8 | +1.07e-5 | -9.17e-9 | +3.31e-8 | +6.64e-8 | +4.15e-7 |
| α=4.00 | 0% | 0 | 0 | 0 | 0 | 0 | 0 |

**结果分析。** 输出扰动随丢弃率单调下降。α=1.0 的 6.75% 丢弃已造成可见 MI/VIF 下降，不宜作为默认质量路径；α=1.25 的总体丢弃降至 0.80%，输出 MAE 仅 3.67e-5，五项指标变化很小，但个别层最大丢弃仍达 10.71%，仍需重新训练验证；α=1.5 基本贴近 sparse；α≥2 数值上等价，却不具备 bs=10 的显存可行性。建议将 α=1.25 作为速度档、α=1.5 作为保守档、sparse 作为零丢弃回退，而不是用一个容量覆盖所有场景。

![容量—质量—吞吐 Pareto](Materials/efficiency/figures/capacity_quality_pareto.png)

---

## 3　分布式瓶颈：先判断通信是否值得优化

### 3.1　模型状态很小，显存瓶颈是激活而非参数

模型参数约 4.11 M，FP32 参数、梯度及 Adam 两个状态合计不足 0.1 GB，而实测峰值显存为 61–76 GB。也就是说，参数/优化器状态不到峰值的 0.2%，主要显存来自 170×170 多尺度激活、损失分支及 grouped 容量缓冲。

**表 E3-8　并行方案与当前瓶颈的匹配度**

| 方案 | 主要解决对象 | 当前模型判断 | 结论 |
|---|---|---|---|
| 纯 DDP | 样本并行；每卡完整副本 | 参数副本极小，通信量仅约 15.67 MiB 梯度 | **采用** |
| FSDP / ZeRO-2/3 | 切分参数、梯度、优化器状态 | 最多节省不足 0.1 GB，却新增 reduce-scatter/all-gather | **当前不采用** |
| Expert Parallel | 切分专家并做 token all-to-all | 仅 12 个小专家；会把本地 bmm 变成 A2A 延迟 | **当前不采用** |
| Tensor/Pipeline Parallel | 切分大层或深流水 | 层小且 depth=4，通信/气泡大于收益 | **当前不采用** |
| DDP 分桶 + contiguous grad | 在反向中异步 all-reduce | 代价小，且可避免梯度拷贝 | **采用并实测** |
| 固定 shape + 成本均衡分片 | 减少 rank straggler | DDP 临界路径由最慢 rank 决定 | **采用为数据侧原则** |

**结果分析。** 大模型框架中的 ZeRO/FSDP/EP 并非越多越先进。当前模型是“参数很小、激活很大”，切分模型状态几乎不释放可用 batch 空间，反而引入额外 collective。最合适的路线是保持纯 DDP，把优化集中在单卡执行、输入均衡和少量梯度通信上。

### 3.2　NCCL 链路处于什么状态

**表 E3-9　单机 H800 NCCL all-reduce（100 次；延迟取 p50）**

| payload | 4 卡 p50 | 4 卡 bus BW | 8 卡 p50 | 8 卡 bus BW |
|---:|---:|---:|---:|---:|
| 16 KiB | 38.8 μs | 0.35 GB/s | 39.5 μs | 0.70 GB/s |
| 1 MiB | 38.8 μs | 39.61 GB/s | 43.6 μs | 21.69 GB/s |
| 4 MiB | 58.4 μs | 104.85 GB/s | 69.2 μs | 84.94 GB/s |
| 8 MiB | 77.1 μs | 161.94 GB/s | 93.0 μs | 112.09 GB/s |
| 16 MiB | 115.1 μs | 217.62 GB/s | 135.1 μs | 186.61 GB/s |

**结果分析。** 小于 1 MiB 时几乎全是约 40 μs 固定延迟，拆成许多小桶会反复支付启动成本；4–16 MiB 后带宽快速爬升。当前完整梯度约 15.67 MiB，一次 all-reduce 的量级仅约 0.1–0.2 ms，远小于约 422 ms 的训练步，因此 NCCL 不是主瓶颈。分桶目标应是“至少两个足够大的桶以获得重叠机会”，而不是无限减小桶。

![NCCL 桶大小曲线](Materials/efficiency/figures/nccl_bucket_curve.png)

### 3.3　模型大小感知的 DDP 分桶

![DDP 通信重叠原理](Materials/efficiency/figures/ddp_overlap_principle.png)

**表 E3-10　DDP-4 分桶扫描（其余配置固定）**

| `bucket_cap_mb` | 重建后桶数 | ms/step | samples/s | 相对 1 MiB |
|---:|---:|---:|---:|---:|
| 0.5 | — | 不支持 | — | PyTorch 首桶默认为 1 MiB，触发断言 |
| 1 | 15 | 427.75 | 93.512 | 1.000× |
| 2 | 8 | 426.36 | 93.817 | 1.003× |
| 4 | 4 | 426.23 | 93.847 | 1.004× |
| 8 | 2 | **425.57** | **93.993** | **1.005×** |
| 25 | 1 | 426.78 | 93.726 | 1.002× |

**结果分析。** 1 MiB 的 15 个 collective 受启动延迟拖累；25 MiB 大于模型全部梯度，只形成一个桶，失去反向重叠机会；8 MiB 恰好形成两个约 8.09/7.58 MiB 的高带宽桶，取得最优结果。在最终 `static_graph` 配置上，8 MiB 也从 423.20 ms 降到 422.32 ms（吞吐 +0.21%），两次独立实验方向一致。收益不大，是因为通信本来就很小，但该选择有明确机制而非经验猜值。

![DDP 桶扫描](Materials/efficiency/figures/ddp_bucket_sweep.png)

### 3.4　DDP 运行时消融

**表 E3-11　DDP-4 优化逐项消融（预训练权重）**

| 配置 | ms/step | global samples/s | 相对上一项 |
|---|---:|---:|---:|
| baseline：25 MiB、find-unused | 439.63 | 90.986 | — |
| + `gradient_as_bucket_view` | 437.70 | 91.388 | +0.44% |
| + fused Adam | 424.99 | 94.120 | +2.99% |
| + `static_graph`、关闭 unused 搜索 | 423.56 | 94.437 | +0.34% |
| + 8 MiB 两桶 | **422.32** | **94.716** | +0.30% |

**结果分析。** 总吞吐从 90.99 提升到 94.72 samples/s，累计 +4.10%。最大贡献来自 fused Adam，因为模型包含大量小参数张量；bucket view 的显存收益被 69 GB 级激活峰值淹没，但仍省去梯度到通信桶的拷贝；DDP 日志只发现 768 B unused 参数，且使用集合不随迭代改变，所以 `static_graph=True` 在当前图上可安全跳过每步图搜索。最终配置为 `gradient_as_bucket_view + fused Adam + static_graph + 8 MiB`。

**表 E3-12　最终 8 MiB 两桶下的计算/通信对照**

| 通信模式 | 含义 | ms/step | 步内标准差 |
|---|---|---:|---:|
| noop | 通信钩子直接返回，近似计算下界 | 422.63 | 0.67 ms |
| default | DDP 异步 all-reduce | **422.32** | 0.62 ms |
| sync | 每桶同步 all-reduce | 422.83 | 0.68 ms |

**结果分析。** 三者最大差仅 0.51 ms（0.12%），小于单步标准差，不能据此声称精确的“重叠比例”；能可靠得出的结论是：调优后通信暴露时间已落入测量噪声，当前步时几乎完全由计算决定。旧报告用“DDP 单卡差值”估算出 5.9% 通信开销，其中混入编译、输入及运行时差异，不能再解释为纯 NCCL 时间。

---

## 4　如何让每张卡有效工作：扩展效率与 straggler

### 4.1　1/2/4/8 卡扩展

**表 E3-13　单机 DDP 强度不变扩展（每卡 bs=10）**

| GPU 数 | ms/step | global samples/s | 相对 1 卡扩展效率 |
|---:|---:|---:|---:|
| 1 | 421.37 | 23.732 | 100.00% |
| 2 | 422.63 | 47.323 | 99.70% |
| 4 | 423.20 | 94.519 | 99.56% |
| 8 | 425.22 | 188.138 | **99.08%** |

**结果分析。** 8 卡吞吐为单卡的 7.93×，扩展效率 99.08%。这与“当前小模型 DDP 严重通信受限”的旧判断相反：在固定输入、稳定图和单机 NVLink/NVSwitch 条件下，通信只使 8 卡单步比 1 卡增加 3.85 ms。当前优化重点应继续放在每卡计算和输入供给，而不是引入更复杂的模型并行。

### 4.2　最慢 rank 的 1:1 放大效应

**表 E3-14　只给一个 rank 注入输入停顿**

| 注入停顿 | DDP-4 ms/step | 相对无停顿增加 | 增加/注入 |
|---:|---:|---:|---:|
| 0 ms | 422.32 | 0 | — |
| 2 ms | 424.41 | +2.10 ms | 1.05× |
| 5 ms | 427.52 | +5.20 ms | 1.04× |
| 10 ms | 432.55 | +10.24 ms | 1.02× |

**结果分析。** 单 rank 停顿几乎 1:1 进入全局临界步，其他卡在 collective 处等待。值得注意的是，同步之后观测到的 rank 完成时间差仍只有约 0.08 ms，说明只看 rank 末端 CV 会掩盖上游 straggler；应同时监控 DataLoader 等待、首个梯度 ready 时间和全局 step。生产数据侧应继续使用固定 170×170 crop、`drop_last=True` 和每 rank 等 batch，并将三任务按“任务×样本成本”分层打散；若未来引入变分辨率，应以像素/token 数或上一 epoch EMA 时间做成本均衡，而不是只保证样本数相同。

![DDP 扩展与 straggler](Materials/efficiency/figures/ddp_scaling_straggler.png)

![rank 成本均衡原理](Materials/efficiency/figures/ddp_rank_balance_principle.png)

### 4.3　路由不均是否会变成 rank 不均

**表 E3-15　DDP-4 各 rank 路由遥测（α=1.25）**

| rank | load CV | max/mean expert load | dispatch 丢弃率 |
|---:|---:|---:|---:|
| 0 | 0.1500 | 1.269 | 0.588% |
| 1 | 0.1526 | 1.274 | 0.637% |
| 2 | 0.1536 | 1.283 | 0.733% |
| 3 | 0.1483 | 1.268 | 0.578% |

**结果分析。** 各 rank 的路由分布并不完全相同，但 grouped 每个 rank 都执行相同 `[E, cap, C]` 形状的 bmm，因此路由差异主要改变有效槽位和丢弃，不改变主 GEMM 形状。固定容量除了利于 compile，也是一种“计算形状均衡器”；这解释了路由 load CV 约 15% 时，最终 rank 时间 CV 仍只有约 0.03%。

---

## 5　与开源训练框架的对照及取舍

PyTorch DDP、Megatron-Core 和 DeepSpeed 的共同思想都是：用连续梯度缓冲与分桶启动异步 collective，使通信和反向计算重叠；Megatron 进一步用 distributed optimizer、reduce-scatter/all-gather 以及 EP all-to-all 支撑大模型。将这些机制映射到当前模型后，结论如下。

**表 E3-16　开源框架机制迁移判断**

| 框架机制 | 可借鉴部分 | 本项目取舍 |
|---|---|---|
| PyTorch DDP bucket reducer | 按梯度 ready 顺序分桶；`gradient_as_bucket_view`；`static_graph` | 已落地并实测，8 MiB 两桶最优 |
| Megatron `overlap_grad_reduce` | 异步梯度归约与 backward 重叠 | DDP 默认 reducer 已提供同类能力 |
| Megatron distributed optimizer / `overlap_param_gather` | 切分状态并把参数 all-gather 与 forward 重叠 | 参数仅约 16 MB，不值得增加 param gather |
| Megatron EP A2A overlap | 将 token all-to-all 与相邻微批计算重叠 | 当前专家小且单卡可容纳，EP 不合算 |
| DeepSpeed ZeRO-1/2/3 | 依次切分 optimizer、gradient、parameter | 当前峰值由激活主导，暂不采用 |
| 通信压缩（fp16/bf16 hook） | 减少跨节点梯度字节 | 单机通信已低于噪声，且本负载 bf16 曾显著变慢，暂不采用 |

**结果分析。** 本项目吸收的是“分桶、连续缓冲、静态图、异步归约”的方法，而不是照搬框架的全部并行维度。只有当模型参数、专家数或节点数扩大到使通信暴露显著上升时，才应重新评估 ZeRO/FSDP/EP；当前直接引入会优化错误的瓶颈。

官方依据：

- [PyTorch DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)：`bucket_cap_mb`、`gradient_as_bucket_view`、`static_graph` 的语义；
- [Megatron-Core MoE](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)：DP 梯度重叠、参数 gather 重叠和 EP A2A 重叠；
- [DeepSpeed ZeRO](https://deepspeed.readthedocs.io/en/latest/zero3.html)：optimizer/gradient/parameter 三阶段切分；
- [GShard](https://arxiv.org/abs/2006.16668)：固定 expert capacity、overflow token 和负载均衡损失。

---

## 6　结论、推荐配置与限制

1. **分组容量 MoE 的已证收益不是孤立数字。** 它把动态的 E 路小算子改成固定容量批量 GEMM；收益需要 compile 才兑现，并随专家数增长：E=12/16/24/32 的单步降幅为 9.2%/15.2%/28.4%/36.2%。
2. **容量因子是系统—质量联合旋钮。** α=1.25 在预训练权重上带来 1.110× 单卡吞吐、0.80% dispatch 丢弃和 3.67e-5 输出 MAE；α=1.5 更接近 sparse，但只剩 1.027×，且显存升至 76.2 GB。
3. **当前多卡不是通信受限。** 15.67 MiB 梯度在单机 H800 上约百微秒量级；调优后 1→8 卡扩展效率为 99.08%，default/sync/noop 的差异低于测量噪声。
4. **分布式主要风险是最慢 rank。** 单 rank 2/5/10 ms 停顿分别造成 2.10/5.20/10.24 ms 全局损失；固定 shape、等 batch、任务/成本分层采样比增加复杂模型并行更重要。
5. **推荐训练配置。** 性能档采用 `grouped α=1.25 + SDPA + compile + DDP + gradient_as_bucket_view + fused Adam + static_graph + bucket_cap_mb=8`；质量保守档采用 α=1.5；零丢弃要求下回退 sparse。

限制必须明确：

- 质量结果是冻结权重的 45 样本探针，不是 α=1.25/1.5 独立训练后的统计显著性结论；
- 所有分布式数据来自单机 NVLink/NVSwitch，不能外推到跨节点 IB；
- noop/default/sync 差异已进入噪声区，本文不报告不可靠的“通信隐藏百分比”；
- 尚未实现自定义 Triton gather/scatter；profile 显示 copy 基本未下降，这是下一轮单卡优化空间。

