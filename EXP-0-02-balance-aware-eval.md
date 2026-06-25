# EXP-0-02：评测协议平衡化（balance-aware + 两轴评分 + 欺骗探测器）

- 日期：2026-06-22
- 所属阶段：阶段 0（评测协议改造，承接 EXP-0-01 §4(A) 暴露的问题）
- 结果级别：S1（probe 15 对）
- 关联代码：`metrics/fusion_metrics.py`（新增 `*_hm` 平衡感知指标、`balance`、`FIDELITY_AXIS`/`QUALITY_AXIS`），`eval_fusion.py`（两轴排名 + 多进程 + 欺骗探测器自检）
- 证据：`reports/gfp_pc_probe_balanced/`（`means.csv`、`two_axis_rank.csv`、`per_image.csv`）

## 1. 本次改造内容（对应 EXP-0-01 §5.1）
1. **平衡感知指标**：对可分源的保真类指标加"源间调和平均(soft-min)"变体 `SSIM_hm / MI_hm / Qabf_hm / VIF_hm / MS_SSIM_hm`，复制单源会被较弱那一侧拉低；加 `Balance = min/max(SSIM_A,SSIM_B)`。
2. **两轴评分**：把排名指标分成两条轴，分别排名再等权合成，避免"指标数量多的轴"主导单一标量：
   - **FIDELITY 轴**（对两个源的平衡保真）：SSIM_hm/MS_SSIM_hm/Qabf_hm/VIF_hm/MI_hm
   - **QUALITY 轴**（无参考图像丰富度）：EN/SD/SF/AG
3. **欺骗探测器**：avg/max 移出真方法排名池，改为"试金石"——做轴向自检（真方法是否在 QUALITY 轴上超过平凡 avg/max）。
4. **提速**：指标计算多进程化，**probe 15 对从 ~13min → ~18s**。

## 2. 关键证据（pixel 统计，样本 03-C09）
| 图 | mean | std | p95 |
|---|---|---|---|
| GFP | 4.2 | 10.7 | 12 |
| PCI | 117.7 | 37.8 | 171 |
| **Max** | **117.8** | **37.8** | **171** ← 与 PCI 完全相同 |
| Avg | 61.0 | 18.9 | 86 |
| MDFNet | 65.2 | 23.6 | 100 |

→ **GFP 几乎全黑(95% 像素 ≤12)**，所以 `max(GFP,PCI)=PCI`，"Max" 实际就是把 PC 原样复制；Avg 就是字面混合；MDFNet 是真实的对比度重分布变换（非退化，推理无 bug）。

## 3. 结果（probe 15，两轴排名，越低越好）
真方法：
| 方法 | FidRank | QualRank | Composite |
|---|---|---|---|
| MDFNet-a2 | 1.6 | 2.25 | 1.93 |
| Retrained | 2.6 | 1.25 | 1.93 |
| MDFNet | 1.8 | 2.50 | 2.15 |

欺骗探测器自检（真方法+探测器一起排）：
| | QualRank | FidRank | 角色 |
|---|---|---|---|
| Retrained | 1.75 | 4.6 | method |
| MDFNet-a2 | 2.50 | 3.6 | method |
| Max | 2.75 | 2.2 | detector |
| MDFNet | 3.00 | 3.6 | method |
| Avg | 5.00 | 1.0 | detector |
- **QUALITY 自检：FAIL**（worst real=3.00 ≥ best detector Max=2.75）。

## 4. 分析与问题（重要）
1. **平衡化部分见效**：Max 在 SSIM 上的"复制作弊"被压住（SSIM_hm 0.193，不再是 0.99 的假高分）。
2. **两轴更诚实**：FIDELITY 轴 Avg 必然领先（字面混合最像两个源），QUALITY 轴学习型方法领先——这本就是"保真↔增强"的取舍，单标量本不该掩盖它。
3. **最深问题：通用指标无法判定 GFP-PC 融合是否成功。**
   - FIDELITY 轴：GFP 近全黑→该轴几乎只由 PC 决定→复制/混合 PC 天然占优；
   - QUALITY 轴：PC 本身就是信息丰富图，**Max(=复制 PC) 直接继承 PC 的 EN/SD**，于是连"无参考质量"也压住 MDFNet(MDFNet 仅在 SF/AG 锐度上赢)。
   - 本质：**GFP-PC 融合的真正价值=把 GFP 的功能/荧光信息注入到 PC 结构上，而 GFP 稀疏近黑，这个"价值增量"对所有通用指标都几乎不可见。** 这解释了为何 EXP-0-01 里官方 MDFNet 反被 Max/Avg 超过——不是方法差，是指标测不到要点。

## 5. 下一步（新方向，需确认）
**加"功能/显著区保留"指标**：只在 GFP 的信号区域（非背景的高亮像素，例如 GFP>阈值 的掩码）上度量融合图是否保住了这些功能信息——例如掩码内的 MI/相关性/对比度保留、或显著区 IoU。这是唯一能捕捉 GFP-PC 融合"要点"的指标，也对应多任务里 IR-VIS 的"热目标显著性保留"。
- 完成后两轴升级为三量：FIDELITY / QUALITY / **FUNCTION(显著区保留)**，FUNCTION 作为该任务的主判据。
- 其余通用指标仍报，但定位为"辅助/防退化"。

> 备注：本记录对应 MASTER_PLAN §6。此为评测协议第 2 次迭代；协议可信前不进入阶段 1。
