# Materials/ —— 论文用图素材

存放毕业论文正文各章节的图片素材（成品拼图 + 可自排的单图）。绘图脚本在仓库 `script/` 下。

## 目录规划

```
Materials/
├── figs/                方法章节总体框架图（SVG + PDF + PNG）
│   └── fig_u_moe_fusion_framework.*
├── comparison/          §4.2 与 SOTA 的定性对比图（本批已产出）
│   ├── irvis/
│   │   ├── individual/            # 12 张单图（带红框+左下角局部放大），供 PPT 自排
│   │   │   └── <样本>__<方法>.png  # 如 01506D__Ours.png、01506D__Visible.png
│   │   └── fig_irvis_qualitative.png   # 脚本拼好的成品图（2×6，可直接进论文/PPT）
│   ├── medical/
│   │   ├── pet/    (individual/ + fig_medical_pet_qualitative.png)   # PET–MRI
│   │   └── spect/  (individual/ + fig_medical_spect_qualitative.png) # SPECT–MRI
│   └── gfp_pc/  (individual/ + fig_gfp_pc_qualitative.png)
├── ablation/            §4.3 创新点消融定性图（本批已产出）
│   ├── irvis/   (individual/ + fig_irvis_ablation.png)
│   ├── medical/                # 按模态分开（与 §4.2 一致）
│   │   ├── pet/   (individual/ + fig_medical_pet_ablation.png)   # PET–MRI 样本
│   │   └── spect/ (individual/ + fig_medical_spect_ablation.png) # SPECT–MRI 样本
│   └── gfp_pc/  (individual/ + fig_gfp_pc_ablation.png)
├── hyperparam/          §4.4 超参数取值扫描定性图（本批已产出）
│   └── <param>/  (individual/ + fig_<param>_<task>[_<subtag>].png)  # 每超参 1 张，代表模态
└── efficiency/          §4.5 训练效率与分布式优化
    ├── data/            # 单卡、容量质量、NCCL、DDP 原始 JSON
    │   └── bottleneck/  # 通信分解、物理卡、真实 DataLoader 与采样实验
    └── figures/         # 原理图与数据证据图（SVG + PNG）
```

命名约定（后续各章沿用）：`<类别>/<子项>/{individual/, fig_<子项>_*.png}`。类别用途固定为 `comparison`（对比）、`ablation`（消融）、`hyperparam`（超参）。

## 方法总体框架图（figs/）

`figs/fig_u_moe_fusion_framework.{svg,pdf,png}` 展示 U-MoE-Fusion 从三任务统一亮度输入、三分支 ACM + 窗口 Transformer 骨干、任务条件 top-2 MoE，到决策图融合、色度重组与 maxfuse 训练目标的完整流程；底部另列 grouped-MoE、SDPA、compile 与 DDP 工程优化。脚本为 `script/make_framework_figure.py`，设计与实现映射详见根目录 `FIGURE-01-U-MOE-Fusion-framework.md`。

```bash
cd /ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper-md
/ytech_m2v4_hdd/lizhongyin/venv/gifnet/bin/python script/make_framework_figure.py
```

## 训练效率图（efficiency/）

由 `script/make_efficiency_figures.py` 从 `efficiency/data/` 生成。原理图包括 compile 融合、SDPA、分组容量 MoE、容量—负载均衡、DDP 分桶重叠与 rank 成本均衡；证据图包括 grouped×compile 交互、专家数扩展、容量 Pareto、NCCL 桶曲线、DDP 桶扫描、1/2/4/8 卡扩展，以及桶级通信分解、物理慢卡与同样本任务均衡受控对照。对应实验解释见 `EXP-INFRA-03-grouped-moe-ddp-evidence.md`。

```bash
cd /ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper-md
/opt/conda/envs/py310/bin/python3 script/make_efficiency_figures.py
```

## 本批对比图说明（comparison/）

- 风格对标参考论文：**2×6** 排布，每个面板在**同一关键区域画红框**，并在**左下角**贴红边**小尺寸局部放大子图**；面板下方编号标注 `(a) Visible … (l) Ours`（Ours 与其它方法**同样式**，不特殊标色）；无顶部标题。
- 面板顺序：`(a)源A | (b)源B | (c)LP | (d)NSCT | (e)TarDAL | (f)DATFuse | (g)LRRNet | (h)DDFM | (i)MURF | (j)EMMA | (k)GIFNet | (l)Ours`。
- 显示为**彩色**（社区标准 fuse-Y/recombine-CbCr）：**IR-VIS 用可见光色度重组为彩色**（脚本内 `recolor_irvis`，融合 Y 取自 `fused/`）；medical/gfp_pc 取 `fusion_bench/fused_final/`（已重组）。注：IR-VIS 的**指标**仍按协议在灰度 Y 上计算（见 `EVALUATION-metrics.md`），彩色仅用于**显示**。**Ours(v3) = 文件夹 `W96L`**。
- 医学**按模态分开画**：PET–MRI 与 SPECT–MRI 各一张图。
- 样本选取：优先挑本文 5 项指标（MI/SSIM/Qabf/VIF/Nabf）均优于全部对比方法的代表图。当前：
  - SPECT–MRI=`spect_18017`（5/5 全胜）、gfp_pc=`05-A02`（5/5）；
  - PET–MRI=`pet_25027`：PET 子集无 5/5 全胜图（每张都在 Nabf 上略输给某个过度平滑方法），此图为 4/5（MI/SSIM/Qabf/VIF 全胜，Nabf 0.011 近最低），为该子集最优代表；
  - IR-VIS=`00778N`：为兼顾视觉效果（彩色下可见光信息保留更直观）所选，本文在 MI/VIF/Nabf 上更优，SSIM/Qabf 与最优方法极接近（差 <0.01）。

## 如何重新生成 / 换样本 / 调红框

脚本：`script/make_qualitative_figure.py`（用带 matplotlib 的 venv，如 `/ytech_m2v4_hdd/lizhongyin/venv/gifnet/bin/python`）。

```bash
PY=/ytech_m2v4_hdd/lizhongyin/venv/gifnet/bin/python
cd script
# --box 关键区域相对坐标 x y w h（宜小）；--corner 放大子图所在角（默认 bl 左下）；--ncols 每行面板数（默认 6）
$PY make_qualitative_figure.py --task irvis   --sample 00778N      --box 0.40 0.45 0.16 0.16   # IR-VIS 自动按可见光色度重组为彩色
$PY make_qualitative_figure.py --task medical --subtag pet   --sample pet_25027   --box 0.38 0.40 0.18 0.18
$PY make_qualitative_figure.py --task medical --subtag spect --sample spect_18017 --box 0.38 0.40 0.18 0.18
$PY make_qualitative_figure.py --task gfp_pc  --sample 05-A02      --box 0.40 0.45 0.16 0.16
```

换样本：`--sample <stem>`（候选见 `fusion_bench/inputs/<task>/A/`）。选「本文全指标最优」的样本可参考 `select_samples.py` 的逻辑（逐图比对 W96L vs 9 方法）。

## 消融定性图（ablation/）

由 `script/make_ablation_figure.py` 生成，2×4 排布：`(a)源A | (b)源B | (c)Full(v3) | (d)−MoE | (e)−Decision head | (f)−Window attn | (g)−maxfuse | (h)−Task cond`。**仅创新点消融**（超参不在此）。变体文件夹映射：Full=`W96L`、−MoE=`abNoMoE`、−Decision head=`abDirect`、−Window attn=`abWs1`、−maxfuse=`abOrig`、−Task cond=`abNoTC`。

```bash
PY=/ytech_m2v4_hdd/lizhongyin/venv/gifnet/bin/python
cd script
# 样本与 §4.2 对比实验刻意不同（去重）；医学分 PET/SPECT 两图，与 §4.2 一致。
# 只做单创新点消融；双消融(abNoMoE_direct/abDirect_orig/abWs1_orig)与超参(abD3/abDeep 等)见 EXP-ABLATION-PARAM-v3.md。
$PY make_ablation_figure.py --task irvis   --sample 01506D     --box 0.40 0.45 0.16 0.16
$PY make_ablation_figure.py --task gfp_pc  --sample 05-B06     --box 0.40 0.45 0.16 0.16
$PY make_ablation_figure.py --task medical --subtag pet   --sample pet_25015  --box 0.38 0.40 0.18 0.18
$PY make_ablation_figure.py --task medical --subtag spect --sample spect_4010 --box 0.38 0.40 0.18 0.18
```

## 超参定性图（hyperparam/）

由 `script/make_hyperparam_figure.py` 生成，每个超参 1 张（选该参数效果最直观的代表模态）：source A | source B | 各扫描取值（v3 值标注 (v3)）。变体文件夹映射见脚本内 `PARAMS`（Full v3=`W96L`，其余 `hp*`/`abD3`/`abDeep`）。**8 个参数各用一张互不相同、且与 §4.2/§4.3 均不重复的样本**：崩点在 medical 的用医学图、崩点在 IR-VIS 的用红外-可见光图。

```bash
PY=/ytech_m2v4_hdd/lizhongyin/venv/gifnet/bin/python
cd script
# IR-VIS 类（各用不同样本）
$PY make_hyperparam_figure.py --param topk        --task irvis --sample 00032N --box 0.40 0.45 0.16 0.16
$PY make_hyperparam_figure.py --param depth       --task irvis --sample 00091D --box 0.40 0.45 0.16 0.16
$PY make_hyperparam_figure.py --param out_channel --task irvis --sample 00119D --box 0.40 0.45 0.16 0.16
$PY make_hyperparam_figure.py --param aux_weight  --task irvis --sample 00186D --box 0.40 0.45 0.16 0.16
$PY make_hyperparam_figure.py --param routing     --task irvis --sample 00218D --box 0.40 0.45 0.16 0.16
# 医学类（n_routed/window_size 崩点在 medical → SPECT；n_shared 用 PET）
$PY make_hyperparam_figure.py --param n_routed    --task medical --subtag spect --sample spect_11013 --box 0.38 0.40 0.18 0.18
$PY make_hyperparam_figure.py --param window_size --task medical --subtag spect --sample spect_15012 --box 0.38 0.40 0.18 0.18
$PY make_hyperparam_figure.py --param n_shared    --task medical --subtag pet   --sample pet_25022   --box 0.38 0.40 0.18 0.18
```

样本去重一览：IR-VIS 用 00032N/00091D/00119D/00186D/00218D，medical-spect 用 spect_11013/spect_15012，medical-pet 用 pet_25022；均不与 §4.2（00778N/pet_25027/spect_18017/05-A02）、§4.3（01506D/pet_25015/spect_4010/05-B06）重复。
