# 参考：最新 MoE / 多模态融合创新点调研与落地（2026-06）

> 为 v2 方法改进所做的文献调研（web 核实，附出处），并说明每个创新点的**意义**与**如何接入我们的方法**。落地实验见 `EXP-CMP-OURS-vs-18methods.md` 的 v2 部分。

## 1. DeepSeek 路由（V3，无辅助损失负载均衡）—— 已接入（Innovation B）
来源：DeepSeek-V3 技术报告 [arXiv:2412.19437]、Loss-Free Balancing [arXiv:2408.15664]、HF `modeling_deepseek.py: MoEGate`（`topk_method="noaux_tc"`, `e_score_correction_bias`, `scoring_func="sigmoid"`）。

机制（我们 `net_moe.py: MoEFFN(routing="deepseek")` 的实现）：
- **亲和度用 sigmoid**（非 softmax）：`s_i = sigmoid(gate(u))`，逐专家独立。
- **top-k 选择用 `s_i + b_i`，但门控权重用原始 `s_i`**（选中后归一化）。偏置只influence"选谁"，不进前向信号 → 不破坏专家特化。
- **偏置更新（无梯度）**：`b_i += γ·sign(c̄ − c_i)`，c_i=专家 i 被分配的 token 数；过载降偏置、欠载升偏置；γ=1e-3。
- **小 batch 改进（调研建议，已采纳）**：对 c_i 做 **EMA** 再更新偏置（逐步 count 噪声大）。
- 仅保留**极小互补 aux**（α=1e-4），不与任务梯度对抗。

**意义**：DeepSeek-V3 报告指出 loss-free 比传统 aux-loss **特化更强**（负载均衡不再与任务梯度打架）→ 各模态专家更能保留模态独有信息 → 利好 MI/EN/VIF。这正契合"让每个模态在更多指标上强"的目标。
**注**：device/node-limited routing 是为多卡通信，单卡省略；DeepSeek-V4(2026-04)路由内部未公开技术报告，故用 V3（已核实的规范）。

## 2. 数据集配额课程（Innovation A，用户提出）—— 已接入
`train_moe_curri.py`：前 `warmup` epoch 严格 1:1:1；之后按**每任务训练 loss** 自适应配额——训练不够好（loss 高）的任务**适当多给**数据，按 `cap`（默认 1.5×）封顶、总量近守恒。
**意义**：早期防大数据集主导、保模态公平；后期把算力倾斜给"还没学好"的模态，平衡到合适比例。冒烟验证：medical（loss 最高）配额 120→148，gfp/irvis 相应减少。

## 3. 调研中可借鉴的其它创新（按与我们架构契合度）
| 方法 | 出处 | 创新 | 对我们 |
|---|---|---|---|
| TC-MoA | CVPR24 [arXiv:2403.12494] | 任务特异路由 + 主导强度 + **MI 正则**保跨任务兼容 | 任务条件路由已用；MI 正则可加 |
| MoE-Fusion | ICCV23 [arXiv:2302.01392] | **局部专家+全局专家**样本自适应 | 专家分工模板 |
| **W-DUALMINE** | 2026 [arXiv:2601.08920] | **空间专家+小波频率专家** + 可靠度图加权 + **residual-to-average**（保 MI 又增细节） | **高**：直插我们决策图头，治 MI-细节权衡 |
| CDDFuse/WIFE/DSSFusion | 23–25 | 低频/高频(base/detail)分解专家 | **治我们 SF/细节短板**（Addition B） |
| ST-MoE z-loss | [arXiv:2202.08906] | `mean(logsumexp(logits)²)` 稳路由 | 小 batch 可加 |
| 专家正交/对比损失 | NeurIPS25 [arXiv:2505.22323] | 推专家功能多样 | 无架构改动可加 |

## 4. 给我们方法的最高价值两点（调研结论）
- **A（已做）**：DeepSeek 偏置法负载均衡 + 模态条件路由 → 更强特化 → MI/VIF/EN 普涨。
- **B（治 SF/Qabf 短板，下一步）**：**频率分解(base/detail)专家**，detail 分支驱动决策图权重 w + **多尺度 max-梯度损失** `‖|∇F| − max(|∇A|,|∇B|)‖` → SF/Qabf/AG 升，同时 base 分支 + SSIM 保 MI/SSIM/VIF/SD。这是单模型同时拿下多指标的反复出现的机制（CDDFuse/EMMA）。

## 5. 校验旗标
- 已核实主源：V3 全部公式、偏置更新 `b+=u·sign(c̄−c)` u=0.001、α=1e-4、DeepSeekMoE 共享+细粒度、`MoEGate`/`e_score_correction_bias` 符号、V4 存在(API页)。
- 未核实：V4 路由内部（无技术报告）；部分 2025 融合论文指标表（付费墙）；融合-MoE 文献中无 GFP-PC 基准（我们是少见基准）。
- 纠正：不存在名为 "FusionMoE" 的论文（疑为 FuseMoE/Flex-MoE）。
