# EXP-CMP-OURS：我们的方法（MoE 决策图融合）vs 18 个对比方法

- 日期：2026-06-30　评测协议：`EVALUATION-metrics.md`（核心 9 + 诊断 + 平衡感知 *_hm；方向感知）
- 数据：`fusion_bench/`（irvis n=50 / medical n=48 / gfp_pc n=30），18 个已复现方法 + `_AvgBaseline`，统一流水线计分。
- **MoE 是核心创新，全程保留**；所有改进都在 MoE 框架内进行。

---

## 0. 结论
经多轮在 **MoE 框架内**的架构创新，我们的方法 **U-MoE-Fusion**：
- **8 个指标上做到全场第一**（vs 18 方法）：`MI`(三任务全胜)、`VIF`(三任务全胜)、`gfp_pc:Qabf`、`medical:MI_hm`；
- 综合**平均排名进入每类前 5**（irvis #5、medical #4、gfp_pc #4，仅次于 CDDFuse/SwinFusion/SeAFusion/PIAFusion）；从最初的中游（11–14 名）大幅跃升；
- 17 指标×3 任务中 **15 个进 Top-3**。

## 1. 最终配置（Ours = U-MoE-Fusion）
`models/TG_exp`：MoE（**12 路由专家 + 1 共享专家**，top-2 稀疏调度）+ 真实窗口注意力(ws=8) + **MoE 决策图融合头** + maxfuse 损失。out_channel=64，patch170，20 epoch。

## 2. 把 MoE 做对、做强的关键创新（全部在 MoE 内）
| 创新 | 作用 | 效果 |
|---|---|---|
| **MoE 决策图融合头**（核心）：MoE 预测逐像素权重 w，`F = w·A+(1−w)·B`（+可选残差） | F 被锚定为两源凸组合 → 继承源动态范围(EN/SD)、线性保留双源信息(MI/SCD/CC/VIF)、边界干净(Nabf) | **决定性提升**：把中游方法一举拉进前 5，并拿下 MI/VIF 全胜 |
| **真实窗口注意力**(ws=8，padding-safe) | 取代退化的逐像素线性(ws=1)，提供空间上下文 | SSIM/Qabf/MI 明显改善 |
| **稀疏 top-k 专家调度**（index_add，仅算路由到的 token） | 显存/算力降 ~3×，使 MoE 可放大 | 解除 OOM，支持 12 专家/oc64-96 |
| **细粒度专家数**：8→12 路由专家 | 更细任务/模态特异化 | 全场第一指标数 6→7→8（更多专家更优） |
| **maxfuse 损失**（强度/SSIM/梯度朝 per-pixel max，平衡 RMI） | 保对比度+边缘，修复早期"融合塌成低对比"的根因 | SD 由 9.7→40+（irvis），追平领先 |
| 共享专家 + 任务条件路由 + 负载均衡 aux + out_scale | 防专家坍塌/稳定残差幅度 | 训练稳定 |

## 3. 量化对比（Ours vs 18 方法；**粗体=全场第一**）

### irvis (n=50) — 综合 AvgRank **#5**
| 指标 | Ours | 全场最优 | Ours 名次 |
|---|---|---|---|
| **MI** | **5.582** | 5.582 (Ours) | **#1** |
| **VIF** | **0.146** | 0.146 (Ours) | **#1** |
| SD | 40.69 | 42.0 (CDDFuse) | #5 |
| EN | 6.49 | 6.60 (CDDFuse) | #4 |
| SSIM | 0.720 | 0.738 (DenseFuse) | #4 |
| Qabf | 0.648 | 0.681 (CDDFuse) | #5 |

### medical (n=48) — 综合 AvgRank **#4**
| 指标 | Ours | 全场最优 | Ours 名次 |
|---|---|---|---|
| **MI** | **4.496** | 4.496 (Ours) | **#1** |
| **VIF** | **0.116** | 0.116 (Ours) | **#1** |
| **MI_hm** | **1.889** | 1.889 (Ours) | **#1** |
| SSIM | 0.725 | 0.741 (DenseFuse) | #3 |
| Qabf | 0.697 | 0.739 (SwinFusion) | #3 |
| SD | 72.7 | 79.5 (CDDFuse) | #5 |

### gfp_pc (n=30) — 综合 AvgRank **#4**
| 指标 | Ours | 全场最优 | Ours 名次 |
|---|---|---|---|
| **MI** | **5.334** | 5.334 (Ours) | **#1** |
| **VIF** | **0.124** | 0.124 (Ours) | **#1** |
| **Qabf** | **0.678** | 0.678 (Ours) | **#1** |
| SSIM | 0.538 | 0.545 (DenseFuse) | #3 |
| SD | 25.0 | 39.8 (TarDAL) | #8 |

**8 个全场第一**：irvis(MI,VIF)、medical(MI,VIF,MI_hm)、gfp_pc(MI,VIF,Qabf)。

## 4. 诚实边界
- 8 个第一**集中在信息/保真族（MI/VIF + MI_hm/Qabf）**——这正是"决策图凸组合融合"的天然强项（线性保双源信息）。
- **细节/锐度族仍非第一**：`SF`（高频细节）仍偏低（融合较平滑），`SD/EN/SSIM/Nabf` 多为 #3–#5（接近但未夺冠，主要被 CDDFuse/DenseFuse/SwinFusion 压住）。
- 综合 AvgRank 仍以 CDDFuse 领跑；我们稳居每类前 4–5。
- 规模：probe/test 各 30–50 图、单 seed；终评需全测试集 + 多 seed + 显著性。

## 5. 迭代历程（从中游到前 5）
倒数第一(irvis rank17，融合塌成低对比) → maxfuse 修对比(rank14) → 真实窗口注意力(SSIM/Qabf↑) → **MoE 决策图融合头**(跃入前 5、MI/VIF 全胜) → 稀疏调度+12 专家(凑齐 8 个第一)。MoE 路由早期"不如 shared"的负结果，在**决策图头 + 真实注意力 + 细粒度专家**的组合下被扭转——MoE 现在是有效的核心。

## 6. 复现
`bench/run_ours.py --model TG_exp --name Ours` → `recombine_rescore` → `eval_method`(×3) → `rank_view.py`。训练命令见 `models/TG_exp/args.txt`。所有实验在跳板机(ge85-68) GPU 上跑。
