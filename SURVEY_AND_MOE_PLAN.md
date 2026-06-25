# 图像融合领域调研 + MoE 跨模态统一融合方案

> 目的：为"把已中稿的 MDFNet(多尺度自适应卷积+Transformer,GFP–PC 融合)扩展为统一多任务/多模态 + MoE + 分布式框架"提供领域地图、数据集、跨领域对照、最新创新数字,以及我们框架内的 MoE 方案。
> 证据来源：deep-research(104 agents,24 条 3-0 核实结论)+ 数据集链接已逐一 HTTP 核实可达。日期 2026-06-23。

---

## 1. 领域地图(融合子领域)

| 子领域 | 目标 | 模态特性 | "好融合"= |
|---|---|---|---|
| 医学(MRI–CT/PET/SPECT) | 结构+功能互补 | MRI/CT 灰度结构;PET/SPECT 伪彩功能(代谢) | 结构清晰 + 功能伪彩保留 |
| 医学显微(**GFP–PC**,我们的) | 功能荧光+相位结构 | GFP 近黑稀疏绿荧光(功能);PC 灰度结构 | 结构保留 + 荧光位置/颜色保留 |
| 红外–可见光(IR-VIS) | 热目标+纹理 | IR 热辐射(显著目标);VIS 纹理细节、彩色 | 热目标显著 + 可见光纹理/色彩 |
| 多聚焦(MFF) | 全清晰 | 同场景不同对焦面 | 各区域取最清晰、无伪影/边界 |
| 多曝光(MEF) | 高动态 | 不同曝光 | 暗/亮区细节都在、自然 |
| 遥感(Pan-sharpen/高光谱) | 空间+光谱 | 全色高分;多/高光谱低分 | 高空间分辨 + 光谱保真 |

跨领域共性:**几乎都无真值(GT)→ 无监督/自监督**;都要"保留两源互补信息";多尺度+全局-局部建模通用。
关键差异在**前处理/后处理/损失/显著性定义**(见 §3)。

---

## 2. 数据集(链接已核实可达 2026-06-23)

| 子领域 | 数据集 | 规模 | 地址 |
|---|---|---|---|
| IR-VIS | **MSRS**(现代标准,带 train/test) | 1083 train+361 test(全 2999) | https://github.com/Linfeng-Tang/MSRS |
| IR-VIS | RoadScene | 221 对 | https://github.com/hanna-xu/RoadScene |
| IR-VIS | TNO | 261 对 | https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029 |
| IR-VIS | M3FD(检测,最大多模态) | 4200 对 | https://github.com/JinyuanLiu-CV/TarDAL |
| IR-VIS | LLVIP(监控) | 15488/16836 对 | https://github.com/bupt-ai-cz/LLVIP |
| 多聚焦 | **MFI-WHU**(带 train) | 120+ | https://github.com/HaoZhang1018/MFI-WHU |
| 多聚焦 | Lytro / MFFW | 20 / 13 | 见 https://github.com/Linfeng-Tang/Image-Fusion(聚合) |
| 多曝光 | MEFB(benchmark) | 100 对 | https://github.com/xingchenzhang/MEFB |
| 多曝光 | SICE | 多曝光序列 | https://github.com/csjcai/SICE |
| 医学 | Harvard AANLIB(全脑图谱) | MRI-CT/PET/SPECT | https://www.med.harvard.edu/AANLIB/home.html |
| 医学 | (聚合融合子集) | | https://github.com/Linfeng-Tang/Image-Fusion |
| GFP–PC | John Innes(已有) | 148 对 | 仓库内 `source images/GFP-PC` |
| 评测基准 | VIFB(首个 IR-VIS benchmark,13 指标) | 21 对/20 法 | https://github.com/xingchenzhang/VIFB |
| 聚合/综述 | IVIF_ZOO(TPAMI24 综述+数据汇编) | | https://github.com/RollingPlain/IVIF_ZOO |

> 正在后台下载到集群:MSRS、RoadScene、MFI-WHU(其余按需补)。

---

## 3. 跨领域对照(前处理 / 后处理 / 损失 / 网络)

| 维度 | 医学(含 GFP-PC)/MEF(含彩色) | IR-VIS | 多聚焦 |
|---|---|---|---|
| **前处理** | 彩色源 RGB→YCbCr,**只融 Y**;配准;归一化;重叠裁块 | VIS 取 Y(或 RGB),IR 灰度;配准(MSRS 已配准) | 灰度/Y;同机位无需配准 |
| **后处理** | **融合 Y + 原 CbCr → 逆变换 RGB**(需要) | 灰度任务多数无;彩色 VIS 需回填 CbCr | 灰度无;彩色回填 CbCr |
| **损失重点** | SSIM + 梯度 + 强度/RMI(信息保留);**功能/伪彩保留** | SSIM + 梯度 + **强度偏向(max-intensity 保热目标)** + **显著性/目标感知** | **decision-map / 清晰度(SF/AG)** 取最清晰、边界一致 |
| **网络** | CNN/Transformer 编码-融合-解码;双流 | 双流 + 显著性/检测分支(SeAFusion、TarDAL) | 聚焦判别 + 决策图 或 端到端回归 |

要点:**损失里的"强度/显著性"项是任务相关的最大差异**——IR-VIS 要偏向热目标的高强度,MFF 要偏向清晰区,GFP-PC 要偏向功能荧光。这正是我们 MoE"任务特异路由"应当承载的东西(对应 TC-MoA 的"dominant intensity"控制)。

---

## 4. 评测指标
通用:EN、MI、SF、AG、SD、SSIM、MS-SSIM、Qabf、VIF/VIFF、SCD、Nabf、CC(VIFB 13 指标的标准套件)。
任务特异:IR-VIS 常加目标检测/分割下游精度;MFF 加 Qw/边界伪影;医学/GFP-PC 加我们已实现的 **FUNCTION 轴(FuncCorr/FuncSal,显著区保留)**。我们的三轴协议(FUNCTION/FIDELITY/QUALITY)天然覆盖并可迁移到 IR-VIS 热目标显著性。

---

## 5. 最新创新数字(2023–2025)与"借什么"

| 方法 | 出处 | 核心创新 | 我们借什么 |
|---|---|---|---|
| **TC-MoA** | CVPR2024 [arxiv 2403.12494](https://arxiv.org/abs/2403.12494) | MoE 思想:专家=高效 adapter prompt 预训练基座;**任务特异路由网络**按任务定制共享 adapter,动态调"主导强度";adapter 用**互信息正则**保证跨任务兼容 | ✅ 任务/模态条件路由 + 共享专家 + MI 正则;"主导强度"= 我们 §3 的任务强度偏向 |
| **MoE-Fusion** | ICCV2023 [arxiv 2302.01392](https://arxiv.org/abs/2302.01392) | 多模态门控的**局部专家(MoLE)+全局专家(MoGE)** | ✅ 局部/全局专家分工 = 我们 ACM(局部)+Transformer(全局)天然对应 |
| **CDDFuse** | CVPR2023 [repo](https://github.com/Zhaozixiang1228/MMIF-CDDFuse) | 相关性驱动**双分支分解**:低频(模态共享)相关、高频(模态特异)不相关;一套结构做 IVF+MIF | ✅ 共享专家=低频/共性,路由专家=高频/任务特异 的结构先验 |
| **EMMA** | CVPR2024 [arxiv 2305.11443](https://arxiv.org/abs/2305.11443) | **等变成像**自监督先验,克服无 GT;支持下游检测/分割 | ✅ 无 GT 训练范式(等变/伪传感) |
| **DeFusion++** | 2024 [arxiv 2410.12274](https://arxiv.org/pdf/2410.12274) | 自监督预训练:CUD(共性/特有分解)+MFM(掩码特征建模)学任务无关融合表示 | ✅ 自监督预训练打底,克服小数据/跨任务泛化 |
| **FILM** | ICML2024 [arxiv 2402.02235](https://arxiv.org/abs/2402.02235) | 视觉-语言:文本描述跨注意力引导四任务融合 | ⭕ 可选:文本/任务语义作为路由条件 |
| **TITA** | ICCV2025 [arxiv 2504.05164](https://arxiv.org/html/2504.05164v1) | 显式分离**任务不变交互(IPA)**与**任务特异适配(OAF)**;OAF 用超网络从输入特征预测分支权重,**无需任务 ID**(作者自称类 MoE) | ✅ "隐式路由(看输入而非任务ID)"作为我们路由的对照/消融 |
| **U2Fusion** | TPAMI2020 [repo](https://github.com/hanna-xu/U2Fusion) | 首个单网络无监督统一多任务(多模态/多曝光/多聚焦) | ✅ 统一多任务的奠基 baseline(必须比它强) |
| **TIMFusion** | TPAMI2024 [arxiv 2305.15862](https://arxiv.org/abs/2305.15862) | 隐式架构搜索+元初始化,快速适配多任务 | ⭕ 可选:元初始化加速新任务适配 |

---

## 6. 我们框架内的 MoE 跨模态统一融合方案

把上面创新综合进 MDFNet,得到 **U-MDFNet(Unified-MoE-MDFNet)** 设计:

### 6.1 骨干(沿用并改造 MDFNet)
- 保留三分支多尺度 + ACM(自适应卷积,局部)+ Transformer(全局)。
- 修 0.8:启用真窗口注意力;修共享权重伪多尺度→独立尺度参数。
- 输入:任意两源的 **Y 通道**(彩色源走 YCbCr,§3);带 **任务/模态嵌入**。

### 6.2 MoE 注入(核心创新,综合 TC-MoA + MoE-Fusion + CDDFuse + DeepSeek-MoE)
- **位置**:Transformer 的 FFN → MoE-FFN(top-k 路由);可选 ACM 专家化。
- **专家结构(DeepSeek-MoE 式)**:1 个**共享专家**(always-on,承载跨任务共性融合能力)+ N 个**路由专家**(任务/模态特异)。
  - 与 **CDDFuse 分解先验**对齐:共享专家学"低频/模态共享"(相关),路由专家学"高频/模态特异"(不相关)。
  - 与 **MoE-Fusion 局部/全局**对齐:可设"局部专家(卷积)"与"全局专家(注意力)"两类,分别接 ACM/TM 支路。
- **路由(TC-MoA 式 + TITA 消融)**:路由条件 = token 特征 (+ 任务嵌入 + 模态嵌入);路由输出调节各源"**主导强度**"(IR 偏热目标 / MFF 偏清晰区 / GFP-PC 偏功能荧光)。对照实验:显式任务 ID 路由(TC-MoA) vs 隐式输入路由(TITA)。
- **正则**:负载均衡 aux loss(防专家坍塌)+ 共享专家 **MI 正则**(TC-MoA,保跨任务兼容)。
- **诊断**:专家-任务激活热力图、负载分布、利用率(对应阶段2 deliverable)。

### 6.3 损失(任务自适应,无 GT)
- 通用:SSIM + 梯度 + 强度/RMI(已实现)。
- 任务自适应权重/形式:IR-VIS 用 max-intensity + 显著性;MFF 用清晰度/decision-map;GFP-PC 用功能保留(FUNCTION)。用任务嵌入或不确定性自动加权(learnable σ)。
- 自监督打底(EMMA 等变 / DeFusion++ 分解预训练)克服无 GT、增强跨任务泛化。
- + MoE aux(负载均衡)+ MI 正则。

### 6.4 训练与评测
- 多任务混批训练(FusionDataset 已支持多 config);任务/模态嵌入区分。
- 评测:已建**三轴(FUNCTION/FIDELITY/QUALITY)+ Pareto 自检**协议,FUNCTION 迁移到 IR-VIS 热目标显著性、MFF 清晰区保留。
- 分布式(阶段4):FSDP2 分片 + **专家并行(EP)** + 激活检查点 + bf16(复用集群 A3B MoE 的 MFU/TPS 方法学)。

### 6.5 与我们已中稿工作的关系(创新落点)
- MDFNet 的 ACM/多尺度/内容-结构损失 → 作为骨干与共享能力;
- **新增量 = MoE 跨任务统一 + 任务特异路由(主导强度)+ 共享/路由专家(分解先验)+ 自监督无 GT + 分布式**;
- 相对 TC-MoA/MoE-Fusion 的差异化:① 以 **MDFNet 多尺度 ACM+TM** 为骨干(非 MAE 基座/非纯 IR-VIS);② 引入 **GFP-PC 等不对称模态**(一源近黑)的功能保留判据 FUNCTION 轴;③ 三轴 Pareto 评测协议。

---

## 7. 风险 / 开放问题(来自 research caveats)
- "统一"模型常用**每任务微调权重**(CDDFuse/FILM)而非一套权重;TC-MoA/U2Fusion 更接近真正单模型多任务——我们目标是后者,需注意。
- 路由用**显式任务 ID** vs **隐式输入特征**:小任务数(4-5)、任务内样本多样性高时,二者取舍未定→做消融。
- 小数据下 MoE 易专家坍塌→共享专家兜底 + aux loss。
- 非 IR-VIS 数据集链接/规模随版本变动,引用时标版本。

---

## 8. 参考(已核实主源)
TC-MoA https://arxiv.org/abs/2403.12494 · MoE-Fusion https://arxiv.org/abs/2302.01392 · CDDFuse https://github.com/Zhaozixiang1228/MMIF-CDDFuse · EMMA https://arxiv.org/abs/2305.11443 · DeFusion++ https://arxiv.org/pdf/2410.12274 · FILM https://arxiv.org/abs/2402.02235 · TITA https://arxiv.org/html/2504.05164v1 · U2Fusion https://github.com/hanna-xu/U2Fusion · TIMFusion https://arxiv.org/abs/2305.15862 · VIFB https://github.com/xingchenzhang/VIFB · IVIF_ZOO https://github.com/RollingPlain/IVIF_ZOO

> 完整 deep-research 原始结论(24 条核实)见会话工作区 `tasks/wwo0pebad.output`。本文件对应 MASTER_PLAN §1–§4 的领域依据与阶段2(MoE)设计输入。
