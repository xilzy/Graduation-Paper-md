# EXP-CMP-OURS：我们的方法 vs 18 个对比方法（三类融合，统一指标）

- 日期：2026-06-29　评测协议：`EVALUATION-metrics.md`（核心 9 指标 EN/MI/SD/SF/AG/SSIM/MS_SSIM/Qabf/VIF 的方向感知**平均排名**，越低越好）
- 数据：`fusion_bench/`（irvis n=50 / medical n=48 / gfp_pc n=30 标准化测试对），18 个已复现方法 + `_AvgBaseline`。
- 我们的方法跑同一流水线：`bench/run_ours.py`(融合Y) → `recombine_rescore.py`(彩色任务RGB重组) → `eval_method.py` → `consolidate.py`，与对比方法**计分口径完全一致**。

> 本文件随迭代更新。结论以诚实为准：**目前我们的方法处于中游，尚未做到每类最佳**；下面记录现状、根因诊断、已做的改进迭代、以及仍在进行的架构改进。

---

## 1. 现状排名（Ours = 当前最优配置：共享骨干 + 真实窗口注意力 + maxfuse，**未用 MoE**）

最优配置：`shared-only(n_routed=0) + out_channel=64 + window_size=8 + maxfuse(ssim→max) + 多任务`，12 epoch。

| 任务 | Ours 排名 | Ours AvgRank | 该任务 Top-3 | Ours 关键值 vs 领先 |
|---|---|---|---|---|
| irvis (n=50) | **13 / 19** | 12.00 | CDDFuse 3.67 / SeAFusion 5.00 / SwinFusion 5.11 | SD 27(vs42)、SF **6.6**(vs11)、SSIM 0.67(vs0.72)、Qabf 0.48(vs0.68)、MI 3.69 |
| medical (n=48) | **14 / 19** | 11.11 | CDDFuse 4.33 / IFCNN 5.56 / SwinFusion 5.89 | SD 60(vs80)、SF **18**(vs28)、Qabf 0.56、MI 3.25 |
| gfp_pc (n=30) | **11 / 20** | 10.67 | SeAFusion 5.89 / SwinFusion 6.33 / PIAFusion 6.44 | SF **6.8**(vs10-13)、Qabf 0.60、**MI 3.88（全场最高）** |

**诚实结论：当前方法三类仍处中游（11–14 名），未达到"每类最佳"。** 经三轮迭代已显著改善（irvis 由倒数第一升到 13），结构/边缘/信息族（SSIM/Qabf/MI，gfp_pc 的 MI 已全场第一）追近，但**细节/锐度族 SF（以及 SD）持续偏低**——融合偏平滑，是当前主要短板与下一步攻关点。

---

## 2. 根因诊断（为什么一开始很差）

第一版（h2h_moe_ta，MoE+任务自适应）跑 bench **irvis 倒数第一（rank 17）**。诊断：
- **融合图动态范围被压缩到比输入还低**：irvis 融合 SD=9.7 < 源 VIS(41)、IR(17)；连"平均"基线都不如。
- 根因：损失是 GFP-PC 标定的——SSIM 对源 B 权重 **5×**、RMI **2.5×**，而 irvis 的 B=低对比度红外，导致输出塌向暗 IR；且对称 mean-intensity 数学上把输出拉向**两源平均**（方差更低）。→ 所有对比度/细节指标垫底。

## 3. 已做的改进迭代

| 迭代 | 改动 | irvis SD | irvis 排名 | 结论 |
|---|---|---|---|---|
| v1 (h2h_moe_ta) | MoE + 任务自适应 + 原损失权重 | 9.7 | 17/17 (末) | 对比度塌陷 |
| **v2 (maxfuse)** | **强度/SSIM/梯度都朝 per-pixel max(a,b)**，保留源动态范围；放大模型 oc16→48 | **32.3** | **14/19** | **大幅改善**（核心修复） |
| 对照 (orig+大模型) | 仅放大模型、不改损失 | 7.6 | 末 | 证明**是损失而非容量**修好了对比度 |

**关键正向结论**：`maxfuse`（朝 max 融合）损失是对比度问题的解药——irvis SD 9.7→32、Qabf 0.10→0.42、排名 17→14。

**关键负向结论（按用户指示，效果不好的创新点不再硬推）**：
- **MoE 路由在本 benchmark 上没有帮助**：同条件下 shared-only（n_routed=0，≈普通骨干）与 full-MoE 互有胜负、shared 往往更稳更好（Round-2：shared48 三任务均优于 moe48）；且 MoE 在不同 batch/seed 下**不稳定**（oc32/oc64 曾塌成噪声）。当前 backbone 退化（见 §4）可能使路由无从发挥。→ 暂不把 MoE 作为提分手段，改以"修骨干+损失"为主。

## 4. 仍在进行的架构改进（本轮）

诊断出的下一个根因：**`window_size=1` 使"Transformer"退化为逐像素线性投影**（无空间注意力），而所有强方法（CDDFuse/SwinFusion）都有真实空间建模。本轮实现 **padding-safe 的真实窗口注意力**（ws>1，反射 pad 到窗口整数倍、常规非移位窗口、推理任意尺寸可用），训练对比 shared vs MoE、oc48/64、ws8/10。

结果（vs 上一版 Ours=R1mf48 oc48 ws1）：

| 配置 | irvis 排名 | medical | gfp_pc | 备注 |
|---|---|---|---|---|
| 上一版 (ws1, oc48, MoE) | 14 | 14 | 12 | — |
| **Wsh8_64 (ws8, oc64, shared)** | **13** | **14** | **11** | **本轮最佳→设为 Ours**；SSIM/Qabf/MI 全面追近 |
| Wsh8 (ws8, oc48, shared) | 差 | 差 | 差 | oc48 不够 |
| Wmoe8 (ws8, oc48, MoE) | 比 shared 差 | 差 | 差 | **再次验证 MoE 不如 shared** |
| Wsh10 (ws10) | 差 | 差 | 差 | ws8 优于 ws10 |

- **窗口注意力有效**：把"逐像素线性"换成真实 8×8 窗口注意力后，SSIM/Qabf/MI 明显提升（irvis SSIM 0.59→0.67、Qabf 0.42→0.48；gfp Qabf 0.52→0.60、MI 至 3.88 全场第一），整体排名小幅上移。
- **但 SF/SD 仍偏低**：窗口注意力提升了结构一致性却没解决"细节/锐度不足"；这是与 Top-3 的主要剩余差距。
- **MoE 再次未显价值**：即使在真实注意力骨干下，full-MoE 仍不如 shared-only。按用户指示，**MoE 暂不作为提分手段保留**（作为消融/负结果记录）。

---

## 5. 下一步（达到"每类领先"的路线）
- 若窗口注意力带来提升：再叠加 **独立多尺度参数**（现三分支共享权重=伪多尺度）、更长训练、更高分辨率裁块。
- 损失继续调 SD/SSIM 平衡（v2 朝-max 提对比度但 SSIM 偏低；需兼顾）。
- 重新评估 MoE 是否在"真实注意力 + 独立多尺度"骨干下才显价值（届时再决定保留/弃用）。
- 终评走全测试集 + 多 seed + 显著性。
