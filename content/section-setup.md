# 4.1　实验设置

为系统评价本文所提出的统一多模态融合方法 **Ours（U-MoE-Fusion，v3）**，本节交代实验所依赖的实现细节与运行环境、三个模态的数据集、客观评测指标以及对比方法。本文的核心设定是"**一套权重、三类任务**"：同一模型在红外–可见光、医学、显微三个差异极大的融合任务上共享参数并统一训练，因此实验设置在数据组织、预处理契约与评测协议上均以"跨任务可比、跨方法公平"为原则。为便于阅读，本节末尾以编号 `[n]` 统一列出所引文献。

## 4.1.1　实现细节与实验环境

**网络与实现。** 本文方法基于 PyTorch 实现，训练与测试使用同一套推理代码以保证一致性。骨干为三分支多尺度结构，由自适应卷积模块（ACM）与窗口注意力 Transformer 模块（TM）级联组成；其中 TM 的前馈网络（FFN）被替换为**混合专家前馈网络（MoE-FFN）**，即 1 个常开共享专家与 12 个 top-2 稀疏路由专家，并将任务嵌入注入路由以实现任务特异特化。网络统一以两路亮度通道拼接为输入（`in_channel=2`），特征通道数 `out_channel=96`，骨干深度 `depth=4`，注意力窗口大小 `window_size=8`；融合输出端采用**决策图融合头**，以凸组合 `F = w·A + (1−w)·B` 生成融合结果。完整模型约 411 万参数（4.11 M），是一个轻量级模型。

**训练配置。** 训练采用统一的 170×170 方形裁块（不缩放，理由见 §4.1.2），批大小 `bs=10`，共训练 20 个 epoch，优化器为 Adam、学习率 `1e-3`。损失函数由强度项、结构相似（SSIM）项与梯度项组成，并采用**任务自适应的 maxfuse 模式**（朝逐像素最大响应对齐，以保留更强的结构与对比）；MoE 的负载均衡辅助损失权重取 0.01，用于防止路由塌缩。为避免大数据集淹没小数据集，训练时按任务配额平衡采样，每个任务的裁块数约 4000（固定随机种子以可复现）。上述超参数的取值依据见 §4.4 超参数分析。

**运行环境。** 所有训练与评测均在 GPU 集群上完成，硬件为 8×NVIDIA H800，软件环境为 Python 3.10 + PyTorch（CUDA）。面向该轻量模型的分布式训练加速（torch.compile 图融合、DDP 数据并行、分组 MoE 调度等）单列于 §4.5 训练效率与分布式优化。实现与训练的关键配置汇总于表 4-1。

**表 4-1　实现与训练配置汇总**

| 类别 | 配置项 | 取值 |
|---|---|---|
| 网络 | 输入通道 in_channel | 2（两路亮度 Y 拼接）|
| 网络 | 特征通道 out_channel | 96 |
| 网络 | 骨干深度 depth | 4 |
| 网络 | 窗口大小 window_size | 8 |
| MoE | 共享专家 / 路由专家 | 1 / 12 |
| MoE | 激活专家数 top-k | 2 |
| MoE | 路由方式 | softmax + 负载均衡辅助损失 |
| 模型规模 | 参数量 | 约 4.11 M |
| 训练 | 裁块尺寸 / 批大小 | 170×170 / 10 |
| 训练 | epoch / 优化器 / 学习率 | 20 / Adam / 1e-3 |
| 训练 | 损失模式 | maxfuse（强度 + SSIM + 梯度）|
| 训练 | 辅助损失权重 aux_weight | 0.01 |
| 训练 | 每任务裁块配额 | ≈ 4000（固定 seed）|
| 环境 | 框架 / 硬件 | PyTorch / 8×NVIDIA H800 |

## 4.1.2　数据集

本文在三个代表性的多模态融合任务上进行实验，各任务对应一个公开数据集，如表 4-2 所示。

- **红外–可见光（IR-VIS）**：采用 **MSRS** 数据集<sup>[1]</sup>，该数据集由 PIAFusion 在 MFNet 基础上剔除错位对、重新配准得到，是当前红外–可见光融合的主流基准，含 1083 对训练、361 对测试图像，分辨率 640×480；其中源 A 为彩色可见光（RGB），源 B 为红外（单通道灰度）。为控制评测规模，测试时抽样 50 对。
- **医学（Medical）**：采用 **哈佛全脑图谱（Harvard AANLIB）**<sup>[2]</sup> 派生的功能–结构配对子集，取其中 **PET–MRI 与 SPECT–MRI 的功能伪彩对**，含 578 对训练、48 对测试图像，分辨率 256×256；源 A 为 PET/SPECT 功能伪彩（RGB，反映代谢/血流），源 B 为 MRI（灰度，反映结构）。
- **显微（Microscopy）**：采用 **GFP–PC** 绿色荧光蛋白–相位衬度显微数据集<sup>[3]</sup>（源自 John Innes Centre 荧光图像库），含 118 对训练、30 对测试图像，分辨率 358×358；源 A 为 GFP 荧光（RGB，反映功能表达），源 B 为相位衬度 PC（灰度，反映细胞结构）。

**表 4-2　三个模态数据集规格**

| 任务 | 数据集 | 训练 / 测试对数 | 评测张数 | 分辨率 | 源 A（彩色 · 功能）| 源 B（灰度 · 结构）|
|---|---|---|---|---|---|---|
| 红外–可见光 | MSRS<sup>[1]</sup> | 1083 / 361 | 50 | 640×480 | 可见光 VIS（RGB）| 红外 IR（L）|
| 医学 | Harvard AANLIB<sup>[2]</sup> | 578 / 48 | 48 | 256×256 | PET / SPECT 伪彩（RGB）| MRI（L）|
| 显微 | GFP–PC<sup>[3]</sup> | 118 / 30 | 30 | 358×358 | GFP 荧光（RGB）| 相位衬度 PC（L）|

**统一预处理契约。** 三个任务的源图像尺寸、模态与色彩空间各异，为使"一套权重"能够统一训练，本文遵循社区标准的 **"融合亮度、重组色度"（fuse-Y / recombine-CbCr）** 契约<sup>[1,4]</sup>：彩色源先由 RGB 转到 YCbCr 色彩空间，只取亮度分量 Y 参与融合，色度分量 CbCr 原样保留；灰度源本身即为 Y。两路 Y 归一化到 [0,1] 后拼接为 (2, H, W) 送入网络；色度不进入训练，在**推理阶段**与融合得到的 Y 重组、经逆变换还原为 RGB。三个数据集均已配准，源 B 仅在与源 A 尺寸不一致时做双线性对齐。由于本文的梯度损失将归一化尺度固定为 170²，训练统一采用 **170×170 方形裁块且不缩放**（小于该尺寸者反射填充），以避免缩放扭曲亮度统计。此外，考虑到 MSRS（1083 对）与 GFP–PC（118 对）体量悬殊，训练时**按任务配额平衡裁块**，防止大数据集主导优化。

## 4.1.3　评测指标

图像融合缺乏参考真值，故采用一组互补的无参考客观指标从不同维度刻画融合质量。与 §4.2 的对比实验一致，本文选取 5 项被广泛使用、且能够正交地覆盖"信息量、结构、边缘、视觉感知、伪影"五个维度的指标，如表 4-3 所示：

- **MI（互信息，↑）**<sup>[5]</sup>：度量从两路源图像转移到融合图的信息总量，`MI = MI(A,F) + MI(B,F)`，越大表示保留的源信息越多；
- **SSIM（结构相似度，↑）**<sup>[6]</sup>：度量融合图对两源结构的保真程度，取对 A、B 的 SSIM 均值，越大表示结构保持越好；
- **Qabf（梯度转移质量，↑）**<sup>[7]</sup>：基于 Xydeas–Petrovic 框架度量源图像边缘被融合图保留的比例，反映边缘/细节的转移质量；
- **VIF（视觉信息保真度，↑）**<sup>[8]</sup>：基于人眼视觉系统的信息保真度度量，越大表示感知质量越高；
- **Nabf（伪影率，↓）**<sup>[9]</sup>：度量融合过程引入的、源图像中不存在的虚假边缘与噪声比例，是唯一"越小越好"的诊断指标。

其中 ↑ 表示越大越好、↓ 表示越小越好。前四项刻画"保留了多少有用信息与结构"，Nabf 则从反面约束"引入了多少不该有的伪影"，五者结合可较全面地评价融合结果。

**表 4-3　评测指标定义速查（↑ 越大越好，↓ 越小越好）**

| 指标 | 维度 | 方向 | 含义 | 文献 |
|---|---|---|---|---|
| MI | 信息量 | ↑ | 两源转移到融合图的信息总量 | [5] |
| SSIM | 结构 | ↑ | 对两源结构保真度均值 | [6] |
| Qabf | 边缘 | ↑ | 源边缘被融合图保留的比例 | [7] |
| VIF | 视觉感知 | ↑ | 基于人眼视觉的信息保真度 | [8] |
| Nabf | 伪影 | ↓ | 融合引入的伪影/噪声比例 | [9] |

**彩色任务的 RGB 计分协议。** 由于医学、显微为彩色（伪彩/荧光）任务，其融合结果需先由融合亮度 Y 与源色度 CbCr 重组、逆变换为 RGB（含 uint8 截断）后再转灰度计分——高饱和色区的 uint8 截断会改变灰度，直接用融合 Y 计分并不准确；红外–可见光任务的融合结果本为灰度，直接计分。为保证在无 MATLAB 的集群上评测的正确性，全部指标均由原始 MATLAB 评测套件逐项移植为 Python 实现，并通过单元性质测试（自融合 SSIM=1、常数图熵=0）与平凡基线自检加以校验。上述指标实现与协议细节详见评测文档。

## 4.1.4　对比方法

为覆盖不同技术路线并突出与最新进展的可比性，本文选取 9 个代表性图像融合方法作为对比基线，年份跨度从经典传统方法延伸至 2025 年，如表 4-4 所示。按技术路线可分为四类：

- **传统多尺度变换方法**：拉普拉斯金字塔（**LP**）<sup>[10]</sup> 与非下采样轮廓波变换（**NSCT**）<sup>[11]</sup>，作为无需训练的经典基线；
- **目标/注意力驱动的深度方法**：目标感知双对抗融合（**TarDAL**，CVPR 2022）<sup>[12]</sup> 与双注意力 Transformer（**DATFuse**，TCSVT 2023）<sup>[13]</sup>；
- **表示学习与生成式方法**：低秩表示引导融合（**LRRNet**，TPAMI 2023）<sup>[14]</sup>、去噪扩散融合（**DDFM**，ICCV 2023）<sup>[15]</sup> 与配准–融合联合（**MURF**，TPAMI 2023）<sup>[16]</sup>；
- **最新通用/多模态方法**：等变多模态融合（**EMMA**，CVPR 2024）<sup>[17]</sup> 与任务无关通用融合（**GIFNet**，CVPR 2025，"One Model for ALL"）<sup>[18]</sup>，用以检验本文方法相对领域最前沿进展的竞争力。

**统一复现协议。** 所有对比方法均在**同一标准化基准**上复现，以保证公平：统一的输入契约（8-bit 灰度源 A/B 加彩色源 CbCr）、统一的融合输出目录与共享的评测流水线；彩色任务（医学、显微）按前述 RGB 重组协议计分，红外–可见光保持灰度计分；每个方法使用独立虚拟环境隔离依赖、加载其官方预训练权重。对于原生仅面向红外–可见光的方法，在医学、显微任务上复用其 IR-VIS 权重进行跨域推理（反映真实的分布迁移，已在结果中如实说明）。

**表 4-4　对比方法一览**

| 方法 | 年份 · 来源 | 技术类型 | 文献 |
|---|---|---|---|
| LP | 1983 · IEEE T-COMM | 传统 · 拉普拉斯金字塔多尺度变换 | [10] |
| NSCT | 2006 · IEEE TIP | 传统 · 非下采样轮廓波变换 | [11] |
| TarDAL | CVPR 2022 | 目标感知双对抗融合 | [12] |
| DATFuse | TCSVT 2023 | 双注意力 Transformer | [13] |
| LRRNet | TPAMI 2023 | 低秩表示引导融合 | [14] |
| DDFM | ICCV 2023 | 去噪扩散模型融合 | [15] |
| MURF | TPAMI 2023 | 配准–融合联合 | [16] |
| EMMA | CVPR 2024 | 等变多模态融合 | [17] |
| GIFNet | CVPR 2025 | 任务无关通用融合 | [18] |
| **Ours** | 本文 | 统一多模态 MoE 融合 | — |

> 说明：本节实验设置对应 §4.2–§4.4 最终采用的对比配置（9 个对比方法 + 5 项判优指标，覆盖传统方法至 2025 年最新方法）。此外本文还在内部维护了一套更完整的 18 方法 × 12 指标标准化基准，用于更细粒度的诊断与排名，相关一手数据见根目录对比实验记录。

## 参考文献

> 本节文献编号为本节局部编号，供撰写期间引用；论文最终统一编排时并入全文参考文献表。

[1] Tang L, Yuan J, Zhang H, et al. PIAFusion: A progressive infrared and visible image fusion network based on illumination aware[J]. Information Fusion, 2022, 83–84: 79–92.

[2] Johnson K A, Becker J A. The Whole Brain Atlas (Harvard AANLIB)[DB/OL]. Harvard Medical School. http://www.med.harvard.edu/AANLIB/.

[3] John Innes Centre. GFP and phase-contrast microscopy image collection[DB/OL]. Norwich, UK.

[4] Ma J, Ma Y, Li C. Infrared and visible image fusion methods and applications: A survey[J]. Information Fusion, 2019, 45: 153–178.

[5] Qu G, Zhang D, Yan P. Information measure for performance of image fusion[J]. Electronics Letters, 2002, 38(7): 313–315.

[6] Wang Z, Bovik A C, Sheikh H R, et al. Image quality assessment: From error visibility to structural similarity[J]. IEEE Transactions on Image Processing, 2004, 13(4): 600–612.

[7] Xydeas C S, Petrovic V. Objective image fusion performance measure[J]. Electronics Letters, 2000, 36(4): 308–309.

[8] Han Y, Cai Y, Cao Y, et al. A new image fusion performance metric based on visual information fidelity[J]. Information Fusion, 2013, 14(2): 127–135.

[9] Kumar B K S. Multifocus and multispectral image fusion based on pixel significance using discrete cosine harmonic wavelet transform[J]. Signal, Image and Video Processing, 2013, 7(6): 1125–1143.

[10] Burt P J, Adelson E H. The Laplacian pyramid as a compact image code[J]. IEEE Transactions on Communications, 1983, 31(4): 532–540.

[11] da Cunha A L, Zhou J, Do M N. The nonsubsampled contourlet transform: Theory, design, and applications[J]. IEEE Transactions on Image Processing, 2006, 15(10): 3089–3101.

[12] Liu J, Fan X, Huang Z, et al. Target-aware dual adversarial learning and a multi-scenario multi-modality benchmark to fuse infrared and visible for object detection[C]//IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022: 5802–5811.

[13] Tang W, He F, Liu Y, et al. DATFuse: Infrared and visible image fusion via dual attention transformer[J]. IEEE Transactions on Circuits and Systems for Video Technology, 2023, 33(7): 3159–3172.

[14] Li H, Xu T, Wu X J, et al. LRRNet: A novel representation learning guided fusion network for infrared and visible images[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023, 45(9): 11040–11052.

[15] Zhao Z, Bai H, Zhu Y, et al. DDFM: Denoising diffusion model for multi-modality image fusion[C]//IEEE/CVF International Conference on Computer Vision (ICCV), 2023: 8082–8093.

[16] Xu H, Yuan J, Ma J. MURF: Mutually reinforcing multi-modal image registration and fusion[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023, 45(10): 12148–12166.

[17] Zhao Z, Bai H, Zhang J, et al. Equivariant multi-modality image fusion[C]//IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024: 25912–25921.

[18] Cheng C, Xu T, Wu X J, et al. One model for all: Low-level task interaction is a key to task-agnostic image fusion[C]//IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.
