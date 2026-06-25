# EXP-0-06：统一多模态数据集（IR-VIS + 医学 + GFP-PC）与训练耦合

- 日期：2026-06-24
- 所属阶段：阶段 0（多任务底座，deliverable 0.9）→ 衔接阶段 1（稠密多任务）与阶段 2（MoE）
- 结果级别：S0（冒烟通过）+ 训练收敛性验证
- 关联代码：`mm_fusion_data.py`、`mm_fusion_dataset.py`、`train_mm.py`、`Networks/net_moe.py`、`train_moe.py`、`configs/{gfp_pc,irvis_msrs,medical_harvard}.json`
- 关联上一实验：EXP-0-05（FUNCTION 轴）；数据/预处理依据见 `DATASETS_AND_PREPROCESSING.md`，MoE 设计见 `SURVEY_AND_MOE_PLAN.md`

## 1. 本次改造内容

**动机**：阶段 1（稠密多任务、任务干扰证据）与阶段 2（MoE）都以"一个能同时吃三类模态的数据底座"为硬前提。本次把 GFP-PC、IR-VIS(MSRS)、医学(Harvard AANLIB) 三类源图像统一成同一网络契约的训练流，并把"数据处理→网络输入→训练"端到端打通、不报错。

**数据落地**
- 医学：本次下载 `xianming-gu/Havard-Medical-Image-Fusion-Datasets`（810 对 256×256，PET/SPECT 伪彩 RGB + MRI 灰度，仓库内直接含图 + 官方 train/test 划分）到 `code/ref/Harvard-AANLIB`，整理 MRI-PET∪MRI-SPECT 功能对到 `data/Harvard-Medical`（**train 578 / test 48**，A=功能伪彩，B=MRI 灰度）。
- IR-VIS：沿用 `data/MSRS`（train 1083 / test 361）。
- GFP-PC：沿用仓库内 148 对（train 118 / test 30）。
- 参考源码已拉到 `code/ref/`：PIAFusion、SwinFusion、U2Fusion、MMIF-CDDFuse、MMIF-DDFM、Image-Fusion。

**统一数据集方案（核心设计）**——见 §"为什么这样合理"。
- 统一网络契约：三类源都化为单通道亮度 Y∈[0,1]（彩色→BT.601 YCbCr 取 Y，灰度即 Y），concat 成 (2,H,W)，沿用 MDFNet 既有 `in_channel=2` 不改通道。
- 配置 schema 升级（`mm_fusion_data.py`，向后兼容）：支持 **suffix 配对**（GFP-PC 同目录、不同后缀）与 **folder 配对**（MSRS/医学：vi/ir 或 func/mri 分目录、同名配对、显式 train/test 文件夹）。
- 裁块统一 170×170（因 `joint_grad` 的归一化写死 `/170²`），尺寸不一**裁不缩放**，小图反射 padding；B 仅在尺寸不一致时双线性对齐 A。
- **按任务配额平衡**（`crops_per_task`，固定 seed）：避免 MSRS（1083）淹没 GFP（118）。

**网络/损失改造（阶段 2 首版，新建文件不动原网络）**
- `Networks/net_moe.py: MODEL_MoE`：Transformer 的 FFN(Mlp) → **MoE-FFN**（1 共享专家 always-on + N 路由专家 top-k 门控，DeepSeek-MoE 式）；**任务嵌入条件路由**（TC-MoA 式，可 `--no-task-cond` 切隐式路由做 TITA 消融）；**负载均衡 aux loss**（Switch/GShard）；stem 后加**每任务通道偏置**使稠密路径也任务感知。I/O 契约与原 `MODEL` 一致，仅 `forward(x, task_id)` 返回 `(out, aux)`。
- `train_moe.py`：传 `task_id`、加 aux loss；可选 **任务自适应强度**（`--task-adaptive`）——IR-VIS/医学用 `max(a,b)` 目标（保热目标/功能亮区），GFP-PC 保留均值强度，化解 MASTER_PLAN 指出的跨任务强度冲突。

## 2. 实验设置

- 任务与数据：三任务联合，train 配额 ≈ GFP 3894 / IR-VIS 3249 / 医学 3468 ≈ 1.06 万裁块/epoch。
- 网络：原始 MDFNet（稠密，`train_mm.py`）作多任务底座对照；`MODEL_MoE`（n_routed=4, top-2, shared=1, task_cond, task_adaptive）作阶段 2 首版。
- 训练：8 epoch，Adam lr 1e-3，每 epoch ×0.8 衰减，bf16 否（fp32），单卡 H800；稠密 bs64@GPU0、MoE bs48@GPU1（损失核绑 cuda:0，故用 `CUDA_VISIBLE_DEVICES` 隔离）。
- 损失：稠密沿用 MDFNet 原式（SSIM+RMI+intensity+grad）做干净对照；MoE 额外 +aux(0.01) +任务自适应强度。

## 3. 结果

### 3.1 耦合冒烟（S0）—— 全部通过
- 2 任务（GFP+IR-VIS）与 3 任务（+医学）`MMFusionDataset` 构建成功，per-task 裁块均衡，item 均为 (1,170,170)、Y∈[0,1]。
- `train_mm.py`（稠密）与 `train_moe.py`（MoE）端到端前向/反向/存档**无报错**；MoE forward 返回 (out,aux)，反向通过。
- 网络参数：MoE 版（out_channel=16）约 0.04M（专家维度小，扩展性曲线留阶段 2 加 out_channel/专家数）。

### 3.2 训练收敛性（8 epoch）

**稠密多任务（mm_dense_v1，原始 MDFNet + 原始对称损失）** —— 8 epoch，165s/epoch，单调收敛：

| epoch | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| loss | 7.15 | 4.49 | 4.21 | 3.99 | **3.92** |
| content | 2.24 | 1.77 | 1.77 | 1.76 | 1.76 |
| structure | 2.46 | 1.36 | 1.22 | 1.12 | 1.08 |

**MoE 多任务（mm_moe_v1，task_cond + task_adaptive + aux）** —— 8 epoch，209s/epoch，0.04M 参数（routed=4/top-2/shared=1）：

| epoch | 1 | 3 | 5 | 8 |
|---|---|---|---|---|
| loss | 11.38 | 9.60 | 9.52 | **9.46** |
| content | 2.93 | 2.90 | 2.91 | 2.91 |
| structure | 4.18 | 3.30 | 3.26 | 3.23 |
| aux(负载均衡) | 9.29 | 9.13 | 9.18 | 9.12 |

> 注：MoE 与稠密的 loss 绝对值**不可直接比**（MoE 用 max-intensity 任务自适应强度 + aux，损失构成不同）。两者各自单调收敛即说明训练健康；质量高下以 §3.3 退化自检 + 后续 S1 三轴平均排名为准。

### 3.3 退化自检（S0 判据：非全黑、非复制单源、无 NaN；`mm_infer_check.py`，各任务 probe 5 图）

| 模型 | gfp_pc | irvis | medical | 总判 |
|---|---|---|---|---|
| **稠密 ep8** | PASS (mean .240, corrB .96) | **FAIL — BLACK (mean .008)** | PASS (mean .214, corrA .60/corrB .96) | **SOME FAIL** |
| **MoE ep8** | PASS (mean .490, corrB −.06) | **PASS (mean .095, corrB −.25)** | PASS (mean .337, corrB .19) | **ALL PASS** |

关键对比（IR-VIS 亮度）：稠密对**白天明亮可见光**（VIS-Y 均值 0.34–0.45）也输出近黑（融合均值 0.003–0.007）；MoE+任务自适应把 IR-VIS 融合均值从 0.008 拉回 0.095，退化解除。
**但**：MoE ep8 各任务逐源 Y-corr 偏低（gfp corrB −.06、irvis corrB −.25），即融合亮度未紧跟结构源——这不是 S0 能判的质量问题，须由 S1 三轴平均排名定论；当前只能说"MoE 解除了塌黑退化，质量优劣待评测"。

## 4. 分析与问题

**1）数据集方案有效，耦合稳固。** 三类异构源（彩色 358 显微 / 彩色 640×480 红外可见 / 伪彩 256 医学）经"统一 Y 域 + 170 方裁 + 任务配额平衡"后进同一网络、同一损失，冒烟与 8-epoch 训练全程无报错，稠密与 MoE 均单调收敛。GFP-PC、医学两任务 S0 全过、且融合图同时保留结构（corrB 高）与功能色（色度通道回填）。

**2）暴露了清晰的跨任务干扰证据（阶段 1 的核心论据，提前显形）。** 稠密单模型 + GFP-PC 标定的对称损失（SSIM 对源 B 权重 5×、RMI 2.5×、强度往两源均值拉）施加到 IR-VIS 时，把输出强烈往**暗的红外**（IR 均值仅 ~0.06）拉，导致 **IR-VIS 融合整体塌成近黑**——即便可见光很亮。这正是 MASTER_PLAN §2「intensity 把融合图往两源均值拉，跨任务易冲突」与关键问题(4)预言的现象，是论证 MoE/任务自适应损失必要性的**直接证据**。

**3）任务自适应 + MoE 解除该退化。** MoE 版对 IR-VIS/医学改用 `max(a,b)` 强度目标（保热目标/功能亮区）、GFP-PC 保留均值强度，并加共享专家+路由专家与负载均衡 aux；S0 三任务全过，IR-VIS 亮度恢复。**诚实边界**：本轮 MoE 同时改了**架构(MoE)**与**损失(任务自适应)**，因此"解除塌黑"主要归功于任务自适应强度；要拆开二者贡献，需补"稠密+任务自适应"与"MoE+对称损失"的消融。MoE ep5 的逐源 corr 偏低（结构保真仍在收敛 / 任务自适应改变了优化动态），不能仅凭 S0 判优劣——**质量结论须走 S1 三轴平均排名**（eval_fusion.py），列为下一步。

**4）发现的真实限制**：`Networks/layers.py` 的 ACM 自适应卷积用 `int(np.sqrt(l))` 重排，**假设方形输入**；非方形（MSRS 640×480）整图推理会崩。训练用 170 方裁块不受影响；退化自检脚本已加中心方裁兜底。整图彩色推理流水线需在 0.8（真窗口注意力）一并修 ACM 或固定方形输入。

## 5. 下一步
- 阶段 1 正式产出：稠密下"各任务单训 vs 三任务联合训"的**任务干扰矩阵**（用本数据底座，逐任务跑 S1 平均排名）。
- 阶段 2 深化：专家数 {4,8,16}×top-k 扩展性曲线、专家-任务激活热力图、共享专家 MI 正则、显式 vs 隐式路由消融（`--no-task-cond`）。
- 评测打通：把 `eval_fusion.py` 的三轴+平均排名跑到 IR-VIS/医学 probe 子集（func_source 已在 config 标好），三任务统一报表。
- 数据扩展：CT-MRI（结构-结构）纳入医学任务的消融；MFI-WHU 多聚焦待集群补解压器后接入第四任务。
