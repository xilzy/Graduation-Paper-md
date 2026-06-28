# EXP-CMP-13：DDFM 对比方法复现（扩散模型，三模态指标）

- 日期：2026-06-25
- 所属阶段：阶段 5（SOTA 复现）
- 结果级别：S2（完整测试子集，三任务）
- 关联代码：`code/ref/MMIF-DDFM`（作者仓库，commit 589f5cb）；驱动 `code/ref/MMIF-DDFM/run_ddfm.py`
- 评测器：`code/Graduation-Paper/bench/eval_method.py`

## 1. 方法与权重来源
- 论文：Zhao et al., *DDFM: Denoising Diffusion Model for Multi-Modality Image Fusion*, ICCV 2023 oral。
- 思路：以 OpenAI guided-diffusion 的 **ImageNet 256×256 无条件扩散先验** 为生成器，通过 EM + 扩散后验采样（diffusion posterior sampling）实现无监督融合，无需融合标签、无需在融合数据上训练。
- 仓库：`Zhaozixiang1228/MMIF-DDFM`（已在 `code/ref/MMIF-DDFM`）。
- 扩散权重：`256x256_diffusion_uncond.pt`（OpenAI，`openaipublic.blob.core.windows.net/diffusion/jul-2021/`），经代理下载。
  - **关键坑**：首次 wget 得到 2101067776 字节的**截断文件**，`torch.load` 失败后仓库会**静默回退随机初始化**（只打印 "Randomly initialize"，不报错），融合结果即噪声。改用 `curl -sL` 重下到完整 **2211383297 字节（~2.21GB）**，`torch.load` 成功（566 keys）后方可采样。本次结果均基于校验通过的完整权重。

## 2. 环境与运行
- venv：`/ytech_m2v4_hdd/lizhongyin/venv/ddfm`（base.pth 继承 torch2.8；私装 blobfile、matplotlib，guided_diffusion 依赖）。
- 采样：`CUDA_VISIBLE_DEVICES`/`--gpu 5`，`timestep_respacing=25`（DDIM，从 1000 步降到 25 步以在 128 张图上可控）；A→I 槽、B→V 槽（与仓库原 IVF 数据流一致）；256×256 采样后融合 Y 双三次插值回原尺寸存灰度 PNG。
- 速度：约 **2.5 s/图**（25 步，H800），三任务 128 张合计约 5–6 分钟。

## 3. 结果（均值）

| 任务 | n | EN | MI | SD | SF | AG | SSIM | MS_SSIM | Qabf | VIF | SCD | Nabf | CC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| irvis   | 50 | 6.090 | 2.357 | 28.537 | 6.358 | 2.392 | 0.704 | 0.787 | 0.316 | 0.036 | 1.446 | 0.076 | 0.661 |
| medical | 48 | 5.173 | 3.489 | 68.162 | 18.773 | 7.425 | 0.715 | 0.772 | 0.556 | 0.085 | 1.441 | 0.025 | 0.890 |
| gfp_pc  | 30 | 6.698 | 2.229 | 27.995 | 9.166 | 4.409 | 0.488 | 0.655 | 0.437 | 0.047 | 1.881 | 0.181 | 0.637 |

功能轴：irvis FuncCorr 0.482 / FuncSal 1.084；medical 0.405 / 1.937；gfp_pc 0.513 / 1.326。
明细：`fusion_bench/reports/<task>/DDFM__{per_image,means}.csv`，并入各任务 leaderboard。

## 4. 分析
- DDFM 在 **医学** 上较强（SSIM 0.715、CC 0.890、Nabf 0.025，结构忠实、伪影少），符合其扩散先验对自然/医学纹理的良好刻画。
- IR-VIS 上 Qabf/SF/AG 偏低：25 步 DDIM 与 256×256 重采样会平滑高频，边缘转移弱于 CDDFuse/SeAFusion；这是"采样步数–速度"折中代价（提步数可回升，但 128 图成本上升）。
- 平均排名：medical 第 6/18、irvis 第 15/18、gfp_pc 第 11/18 —— 作为扩散范式对照锚点，medical 体现其优势域，IR-VIS 体现降步代价。
- 复现要点已沉淀：**截断权重静默回退随机初始化**是本方法最隐蔽的坑，size 阈值校验不足，必须以 `torch.load` 成功为准。

## 5. 下一步
- 纳入 18 方法综合 leaderboard（见 `SUMMARY-comparison-18methods.md` / `fusion_bench/reports/COMPARISON.md`）做平均排名总评。
- 若论文需要更强 DDFM IR-VIS 数字，可单独将 irvis 提到 100 步重采样复跑（成本×4）。


## 指标修订（RGB-final 协议，2026-06-28）

> 修订动机：原先对比方法直接对融合的 **Y 通道图** 计分。参照仓库 `infer_fusion.py` 与 原始 MATLAB `evaluation/main.m` 的约定——**彩色源任务的最终融合图是 Y 与源 CbCr 重组逆变换得到的 RGB 图，计分时对该 RGB 图做 `rgb2gray`（= PIL 'L'，BT.601）**。RGB 逆变换中的 uint8 截断会在高饱和色区（PET/SPECT 伪彩、GFP 绿色）改变灰度，因此直接用 Y 计分不严格。
>
> 修订范围：`output_mode=rgb` 的 **medical / gfp_pc** 两任务，对全部 18 方法的融合 Y 重组源 CbCr → RGB-final → `rgb2gray` 重算（RGB-final 图存于 `fusion_bench/fused_final/<方法>/<任务>/`）。**irvis 为 `output_mode=gray`（与 MDFNet 自身评测一致），维持灰度不变。** 重算后排名与原结论基本一致（个别名次 ±1）。

修订后核心指标（medical/gfp_pc 已按 RGB-final 协议；irvis 灰度不变）：

**DDFM**

| 任务 | n | EN | MI | SD | SF | AG | SSIM | MS_SSIM | Qabf | VIF | SCD | Nabf | CC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| irvis | 50 | 6.090 | 2.357 | 28.537 | 6.358 | 2.392 | 0.704 | 0.787 | 0.316 | 0.036 | 1.446 | 0.076 | 0.661 |
| medical | 48 | 5.165 | 3.512 | 67.417 | 18.588 | 7.301 | 0.715 | 0.772 | 0.546 | 0.085 | 1.410 | 0.023 | 0.889 |
| gfp_pc | 30 | 6.692 | 2.223 | 27.766 | 9.136 | 4.429 | 0.488 | 0.655 | 0.433 | 0.046 | 1.873 | 0.180 | 0.636 |
