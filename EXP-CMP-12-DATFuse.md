# EXP-CMP-12：DATFuse 对比方法复现（三模态指标）

- 日期：2026-06-25
- 所属阶段：阶段 5（综合对比 / SOTA 复现）
- 结果级别：S2（完整测试子集，三任务）
- 关联代码：`code/ref/DATFuse`（作者官方仓库，含预训练权重 `model_10.pth`）；驱动 `code/Graduation-Paper/bench/run_datfuse.py`
- 评测器：`code/Graduation-Paper/bench/eval_method.py`（共享 `metrics/` 包）

## 0. 复现基础设施（所有对比方法共用）
统一基准管线见 `fusion_bench/BENCH_CONTRACT.md`：
- **标准化输入**：三任务测试对统一导出为 8-bit 灰度 `fusion_bench/inputs/<task>/{A,B}/<stem>.png`。A=彩色/功能源，B=灰度/结构源。
  - `gfp_pc`（30 对，GFP-Y / PC）、`irvis`（MSRS 抽 50 对，VIS-Y / IR）、`medical`（Harvard 48 对，PET/SPECT-Y / MRI）。
- **统一输出契约**：融合图按 stem 命名写到 `fusion_bench/fused/DATFuse/<task>/`。
- **统一评测**：`eval_method.py` 计算 EN/MI/SD/SF/AG/SSIM/MS_SSIM/Qabf/VIF + 诊断 SCD/Nabf/CC + 功能轴 FuncCorr/FuncSal。
- **环境隔离**：独立 venv `/ytech_m2v4_hdd/lizhongyin/venv/datfuse`，通过 `zzz_base.pth` 链接基础 venv 的 torch2.8（免重装）。
- **网络**：外网经 `proxy_env.sh` 代理。

## 1. 方法与权重来源
- 论文：Tang et al., *DATFuse: Infrared and Visible Image Fusion via Dual Attention Transformer*, IEEE TCSVT 2023.
- 仓库：作者官方 `https://github.com/tthinking/DATFuse`
  - commit `0cd93f15fb5552c7aef79b7be56726c863d6c5ba`（2024-09-08）
  - 克隆位置 `code/ref/DATFuse`
- 权重（仓库内自带）：`model_10.pth`（红外-可见光预训练，10 epoch）。`strict=True` 直接加载，无 missing/unexpected key。
- 网络结构（`Networks/network.py`，类 `MODEL`，`in_channel=2`）：
  5×5 卷积 stem → FEM(双 3×3) + 双注意力（通道 CAM + 空间 SAM）残差融合 → 单层 Swin Transformer（`window_size=1`，depth=2，num_heads=8）→ 对称解码（再过 FEM+CAM+SAM）→ 1×1 卷积 + Tanh 出单通道。轻量、无下采样、对输入尺寸无整除约束。
- 输入约定：仓库 `Test.py` 用 `torch.cat((ir, vi), 0)`，即通道顺序 **[IR, VIS]**。本契约映射 **B→IR 槽、A→VIS 槽**，与 `cat((B, A))` 一致。
- 任务-权重映射：三任务均用同一红外-可见光预训练权重（论文只发布此权重；medical/gfp_pc 为跨域迁移）。

## 2. 环境与运行
- venv 创建：
  ```
  /opt/conda/bin/python3.11 -m venv /ytech_m2v4_hdd/lizhongyin/venv/datfuse
  echo "/ytech_m2v4_hdd/lizhongyin/venv/lib/python3.11/site-packages" \
    > /ytech_m2v4_hdd/lizhongyin/venv/datfuse/lib/python3.11/site-packages/zzz_base.pth
  ```
  依赖（torch2.8 / timm1.0.27 / numpy / pillow / imageio）全部由基础 venv 经 `zzz_base.pth` 继承，无需额外 pip 安装。
- 推理：`CUDA_VISIBLE_DEVICES=0`，PIL 转灰度 → `ToTensor()`（归一 [0,1]）→ `cat((ir, vi))` → 前向 → Tanh 输出 `clip([0,1])` ×255 取 uint8 存灰度 PNG。全部为推理（无训练），三任务合计 128 张，单卡约 7s。
  ```
  export CUDA_VISIBLE_DEVICES=0
  /ytech_m2v4_hdd/lizhongyin/venv/datfuse/bin/python \
    /ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper/bench/run_datfuse.py
  ```
- 评测：
  ```
  /ytech_m2v4_hdd/lizhongyin/venv/bin/python \
    /ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper/bench/eval_method.py \
    --task <t> --name DATFuse --fused-dir /ytech_m2v4_hdd/lizhongyin/fusion_bench/fused/DATFuse/<t>
  ```

## 3. 结果（均值）

| 任务 | n | EN | MI | SD | SF | AG | SSIM | MS_SSIM | Qabf | VIF | SCD | Nabf | CC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| irvis   | 50 | 6.379 | 3.851 | 35.402 | 10.478 | 4.125 | 0.620 | 0.753 | 0.631 | 0.069 | 1.417 | 0.103 | 0.605 |
| medical | 48 | 5.680 | 3.026 | 63.760 | 29.299 | 11.248 | 0.314 | 0.735 | 0.605 | 0.049 | 0.970 | 0.064 | 0.859 |
| gfp_pc  | 30 | 6.143 | 2.577 | 18.714 |  9.398 |  4.442 | 0.523 | 0.664 | 0.555 | 0.059 | 1.291 | 0.068 | 0.531 |

功能轴：irvis FuncCorr 0.438 / FuncSal 0.809；medical 0.658 / 1.672；gfp_pc 0.242 / 0.293。
明细：`fusion_bench/reports/<task>/DATFuse__{per_image,means}.csv`，并入各任务 `leaderboard.csv`。

## 4. 分析
- DATFuse 是轻量 Transformer 方法（仅单层 window_size=1 的 Swin + 双注意力），推理极快（128 张约 7s）。其设计目标即红外-可见光，在 irvis 上 Qabf(0.63)/MS_SSIM(0.75) 表现稳健，边缘保真良好，Nabf 低（0.10），无明显伪影。
- medical/gfp_pc 用红外-可见光权重跨域迁移：medical 上 SD/SF/AG 高（结构对比强）但 SSIM 偏低(0.31)、VIF 低，说明输出与源的结构相似度下降——跨域分布差异明显；CC 高(0.86) 表明仍保留主体相关性。
- gfp_pc 上各质量轴整体偏低（SD 18.7、VIF 0.059），与 CDDFuse 的结论一致：显微域（GFP 大面积近黑背景）对通用红外-可见光模型是明显的分布外场景，为"通用 vs 专精"动机提供对照证据。
- 与 CDDFuse（EXP-CMP-01）相比，DATFuse 在 irvis 的 SD/MI 略低（轻量模型容量小），但 Qabf/Nabf 同档，是合理的轻量级对比锚点。

## 5. 问题与备注
- 权重仅有红外-可见光一套，medical/gfp_pc 无专用权重，统一跨域迁移（已在结果中体现差异）。
- 仓库 `Test.py` 直接 `(tanh_out*255)` 未做范围处理；本驱动改为 `clip([0,1])*255`，与训练时 GT 在 [0,1] 一致，避免负值截断引入暗噪。
- timm `models.layers` 导入有 FutureWarning（已弃用路径），不影响结果。

## 6. 下一步
- 与其余对比方法统一进 leaderboard 后做平均排名综合对比。


## 指标修订（RGB-final 协议，2026-06-28）

> 修订动机：原先对比方法直接对融合的 **Y 通道图** 计分。参照仓库 `infer_fusion.py` 与 原始 MATLAB `evaluation/main.m` 的约定——**彩色源任务的最终融合图是 Y 与源 CbCr 重组逆变换得到的 RGB 图，计分时对该 RGB 图做 `rgb2gray`（= PIL 'L'，BT.601）**。RGB 逆变换中的 uint8 截断会在高饱和色区（PET/SPECT 伪彩、GFP 绿色）改变灰度，因此直接用 Y 计分不严格。
>
> 修订范围：`output_mode=rgb` 的 **medical / gfp_pc** 两任务，对全部 18 方法的融合 Y 重组源 CbCr → RGB-final → `rgb2gray` 重算（RGB-final 图存于 `fusion_bench/fused_final/<方法>/<任务>/`）。**irvis 为 `output_mode=gray`（与 MDFNet 自身评测一致），维持灰度不变。** 重算后排名与原结论基本一致（个别名次 ±1）。

修订后核心指标（medical/gfp_pc 已按 RGB-final 协议；irvis 灰度不变）：

**DATFuse**

| 任务 | n | EN | MI | SD | SF | AG | SSIM | MS_SSIM | Qabf | VIF | SCD | Nabf | CC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| irvis | 50 | 6.379 | 3.851 | 35.402 | 10.478 | 4.125 | 0.620 | 0.753 | 0.631 | 0.069 | 1.417 | 0.103 | 0.605 |
| medical | 48 | 5.676 | 3.034 | 63.200 | 29.116 | 11.137 | 0.315 | 0.734 | 0.601 | 0.049 | 0.939 | 0.062 | 0.858 |
| gfp_pc | 30 | 6.144 | 2.561 | 18.712 | 9.426 | 4.477 | 0.523 | 0.663 | 0.549 | 0.058 | 1.284 | 0.071 | 0.529 |
