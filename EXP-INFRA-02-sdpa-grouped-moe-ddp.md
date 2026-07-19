# EXP-INFRA-02：SDPA + 分组容量 MoE + DDP 通信优化（训练加速创新点·续）

> **2026-07-19 复核说明：** 本文保留 2026-07-02 的首轮探索数据作为历史记录。预训练权重容量质量、专家数/容量扫描、NCCL、DDP 分桶、1/2/4/8 卡扩展和 straggler 对照已在 [EXP-INFRA-03](EXP-INFRA-03-grouped-moe-ddp-evidence.md) 中完成；关于“5.9% 纯通信开销”和“质量尚未量化”的旧结论，以 EXP-INFRA-03 的隔离测量为准。

- 日期：2026-07-02　模型：v3 = oc96-depth4 MoE（blend 头, ws8, 12 路由专家 top-2, ~4.1M）
- 所属阶段：阶段 4（分布式训练与效率，问题 (5)）
- 结果级别：S1（infra 基准，功能/数值正确性 + 单步/epoch 计时）
- 关联上一实验：[EXP-INFRA-01](EXP-INFRA-01-training-speedup.md)（profile 定位瓶颈 + torch.compile + DDP-4，3.0×）
- 关联代码：`Networks/net_moe.py`（新增 `SDPAWindowAttention`、`combine="grouped"` 容量调度）、`bench/profile_step.py`、`bench/verify_infra.py`、`train_moe_ddp.py`（DDP 通信优化开关）
- 测量口径：跳板机 8×H800，本轮用 GPU 4–7；bs10/GPU、170 裁块、profile_step.py 20 步均值单步时间；DDP 稳态 = warmup 40 步后 80/225 步均值；完整 epoch = 265 步/卡（12000 样本 4 分）。

---

## 1. 本次改造内容（动机 = 消解 EXP-INFRA-01 profile 暴露的三大开销）

EXP-INFRA-01 的 profile 结论：瓶颈是**大量小算子 + 稀疏 top-k 调度的 index/copy + kernel launch 开销**，而非算力（`aten::mm` 6336 次/22% + `IndexBackward` 2400 次/18% + `aten::bmm` 976 次/11.6% + `copy_` 4520 次/6.6%）。本轮针对这三处各下一刀，并把分布式扩展效率补齐：

1. **路线 A — SDPA/Flash 窗口注意力**（打 bmm 11.6%）：新增 `SDPAWindowAttention`（`WindowAttention` 的**权重完全兼容**子类），把手写的 `q@kᵀ → +相对位置偏置 → softmax → @v`（3 个 bmm/softmax kernel）换成单次 `F.scaled_dot_product_attention`，H800 上派发到 FlashAttention / mem-efficient 内核；相对位置偏置作为 SDPA 的加性 `attn_mask` 传入。**数值等价**（SDPA 内部做 1/√d 缩放，故 q 不再预缩放）。
2. **路线 C — 容量受限的分组 MoE 调度**（打 mm 45% + index 18% + copy 7%，**论文创新主线**）：稀疏路径是「E 个小 linear + `nonzero` + `index_add`」的 Python 循环（计算省、但小算子/索引多）；稠密 batched 对「全部 token × 全部专家」算 → OOM。分组调度保持**稀疏计算预算**（每 token 只落 top-k 专家），但用**固定容量的右填充布局**把所有专家收敛成 **2 次批量 GEMM（bmm）**：
   - 按专家对 dispatch 排序（`argsort`）+ 每专家小 `cumsum` 偏移，得到「桶内位置」——避免 O(D×E) 的 one-hot 大 cumsum（eager 下病态，见 §4）；
   - 每专家容量 `cap = cap_factor · T·k / E`，超出丢弃（GShard 风格）；
   - `gather → (E,cap,C)` 缓冲 → 2×`bmm` → 按 gate 加权 `index_add` 散回。
3. **路线 D — CUDA Graph / reduce-overhead**（试打 launch 开销）：`torch.compile(mode="reduce-overhead")`。
4. **路线 B — DDP 通信优化**（打小模型 all-reduce 延迟受限）：`gradient_as_bucket_view=True`、`bucket_cap_mb` 可调、`fused` Adam（减少大量小参数张量上的 optimizer launch）。

> 全部改造均为**可切换开关**（`attn_impl` / `combine` / `--fused-adam` 等），默认回退到 EXP-INFRA-01 的已验证路径，便于消融与对拍。

---

## 2. 实验设置

- 正确性对拍：`bench/verify_infra.py`，同权重下比对 SDPA vs 手写、grouped(高容量无丢弃) vs sparse 的输出 max|Δ|。
- 计时：`bench/profile_step.py`（单卡 20 步均值）、`train_moe_ddp.py --max-steps/--warmup`（DDP 稳态）。
- 对照锚点：EXP-INFRA-01 的 sparse-fp32 基线与 DDP-4+compile。

---

## 3. 结果

### 3.1 正确性（数值等价）
| 检查 | max\|Δ\| | 判据 |
|---|---|---|
| SDPA vs 手写注意力（同权重） | **1.7e-5** | ✅ 等价 |
| grouped(cap8=无丢弃) vs sparse | **1.25e-5** | ✅ 逻辑正确 |
| grouped(cap1.25) vs sparse（随机初始化） | 6.68e-2 | 未训练时负载极不均→少量 token 溢出丢弃；训练时 aux loss 均衡负载后收敛（见 §4 注） |

### 3.2 单卡计时（bs10 oc96 d4 nr12，20 步均值）
| 配置 | ms/step | 峰值显存 | vs 基线 |
|---|---|---|---|
| sparse fp32（基线, INFRA-01） | 555.8 | 70.8 GB | 1.00× |
| **sdpa** | 527.7 | **63.4 GB** | 1.05× + 省 7.4GB |
| sparse + compile | 471.3 | 68.8 GB | 1.18× |
| **sdpa + compile** | 470.1 | **61.3 GB** | 1.18× + 省 7.5GB |
| grouped(sort) eager | 560.9 | 77.0 GB | 0.99×（≈sparse，已消除 cumsum 病态）|
| grouped + compile | 432.2 | 77.0 GB | 1.29× |
| **grp + sdpa + compile（单卡最优）** | **430.8** | 69.5 GB | **1.29×** |
| reduce-overhead(grp+sdpa+compile) | 659.8 | — | ✗ 变慢 |
| reduce-overhead(sparse+sdpa+compile) | 1950.1 | — | ✗ 变慢 |

### 3.3 分组调度的算子分布变化（profile，证实创新点落地）
| 算子 | sparse | grouped(sort) |
|---|---|---|
| `aten::mm` 调用数 / CUDA | 6336 次 / 949ms (22%) | **1728 次 / 417ms (9%)** |
| `IndexBackward0` | 2400 次 / 322ms (18%) | **消失** |
| `aten::bmm`（专家批量 GEMM）| 976 次 / 498ms | 1552 次 / 1.58s（35.75%，成为主算子）|
| `aten::cumsum`（旧 one-hot 实现）| — | 排序法后**消失**（旧实现曾占 82%）|

即「上千个小 linear + 索引/散射」被收敛为「少数规则批量 GEMM」——这正是分组容量调度的目标：**匹配容量的稀疏计算量 + 大幅更少、更规则的 kernel**（利于 compile 融合与访存）。

### 3.4 DDP-4 扩展效率（GPU 4–7，bs10/GPU）
| 配置 | 稳态 ms/step | 完整 epoch(265步/卡) |
|---|---|---|
| sparse + compile + touch-all（INFRA-01 最优）| 503.8 | 143.7 s |
| **grp + sdpa + compile + fusedAdam + bucketView（新最优）** | **456.4** | **135.8 s** |

- 单卡最优 430.8 → DDP-4 最优 456.4，**通信开销 +25.6ms（5.9%）**；旧路径单卡 compile 471 → DDP 503.8，通信开销 +32.8ms（7.0%）。通信优化 + 分组的规则大 GEMM（更易与 all-reduce 重叠）把 4 卡扩展效率抬到 **~94%**。
- **端到端**（vs 原始单卡 fp32 基线：12000 样本、bs10、1200 步 ≈ 667s）：DDP-4 新最优 135.8s = **≈4.9× 端到端加速**；其中数据并行贡献 ~3.7×、kernel(A+C) 贡献 1.29×。

---

## 4. 分析与问题

**本次改造是否有效（结论）**：
- **A（SDPA）= 无条件采用**：数值等价、零风险，速度持平/略快，且**省 ~7.5GB 显存**（flash 不落 (B_,nH,N,N) 注意力矩阵）。省下的显存可换更大 bs 或更多专家。
- **C（分组容量 MoE）= 创新主线，有速度收益但带容量-质量权衡**：compile 下单卡 471→431（+8.5%），并把稀疏调度的小算子/索引开销结构性消除。代价：① cap1.25 会丢弃溢出 token（质量风险，需训练后用 S1 评测确认不掉点）；② 显存高于稀疏（用 SDPA 抵消）。**收益随专家数 E 增长**——阶段 2 的专家数 sweep {4,8,16,…} 正是它的用武之地。
- **B（DDP 通信优化）= 采用**：fused Adam + `gradient_as_bucket_view` + 分组的规则 GEMM，把 4 卡通信开销从 7% 降到 5.9%，扩展效率 ~94%。
- **D（reduce-overhead / CUDA Graph）= 弃用（诚实负结果）**：稀疏/分组 MoE 的 dispatch 是**数据依赖的动态 shape**（`nonzero`/容量），CUDA Graph 反复重捕获，反而 1.4–4× 变慢——与 EXP-INFRA-01 的 bf16 一样，是「与大模型直觉相反」的负结果。

**暴露的问题 / 经验**：
1. **eager 下 one-hot 大 cumsum 是陷阱**：分组位置最初用 `one_hot(D,E).cumsum(0)`（D≈58 万），profile 显示 `cumsum` 独占 82%、单步飙到 3098ms；compile 能优化掉它，但**不应让 compile 去掩盖坏实现**。改为**排序法**（`argsort` 分组 + 每专家小 cumsum 偏移）后 eager 从 3098→561ms，compile 从 442→432ms。→ 教训：MoE dispatch 的「桶内位置」用排序/散射，勿用 O(D×E) one-hot。
2. **grouped 的质量校验缺口**：cap1.25 丢弃 token 对小数据融合质量的影响尚未用 S1 评测量化；下一步需训练一版 grp 模型跑核心 9 指标平均排名，确认不掉点再决定是否设为默认。
3. **numpy 2.x 环境坑**：集群 numpy 被升到 2.2.6，`timm→wandb` 链撞 `np.float_` 移除。用非侵入的 `.npshim/sitecustomize.py`（`PYTHONPATH` 注入，恢复 `np.float_/complex_` 别名）绕过，不动全局环境、不改已发布的 `net.py`。

---

## 5. 下一步
- 训练一版 `combine=grouped, cap_factor∈{1.25,1.5,2.0}` 的模型，用 S1 平均排名确认容量丢弃不掉点 → 决定 grouped 是否设为默认训练路径。
- 阶段 2 专家数 sweep 时，对 E∈{4,8,16,32} 对比 sparse vs grouped 的单步时间-显存曲线，量化「grouped 优势随 E 增长」并作为论文图。
- SDPA + 更大 per-GPU bs（省下的显存）复测 DDP 扩展效率是否进一步逼近线性。
- （可选）grouped 的散回 `index_add` 与 gather 尝试用 `torch.compile` 的 `mode="max-autotune"` 或自定义 Triton kernel 进一步融合。

---

## 6. 产物
- `Networks/net_moe.py`：`SDPAWindowAttention`（权重兼容 SDPA 注意力，`attn_impl="sdpa"`）；`MoEFFN.combine="grouped"` 容量分组调度（排序法位置 + 2×bmm）；`set_combine(mode, cap_factor)`。
- `bench/profile_step.py`：新增 `--attn/--combine grouped/--cap-factor/--compile-mode`。
- `bench/verify_infra.py`：SDPA / grouped 的数值等价对拍。
- `train_moe_ddp.py`：`--attn/--cap-factor/--fused-adam/--bucket-cap/--grad-bucket-view/--warmup` DDP 通信优化开关 + 稳态计时。
- `.npshim/sitecustomize.py`：numpy 2.x 兼容垫片（非侵入，仅经 PYTHONPATH 生效）。
