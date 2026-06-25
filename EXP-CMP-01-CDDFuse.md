# EXP-CMP-01：CDDFuse 对比方法复现（三模态指标）

- 日期：2026-06-25
- 所属阶段：阶段 5（综合对比 / SOTA 复现）
- 结果级别：S2（完整测试子集，三任务）
- 关联代码：`code/ref/MMIF-CDDFuse`（作者仓库，含预训练权重）；驱动 `code/Graduation-Paper/bench/run_cddfuse.py`
- 评测器：`code/Graduation-Paper/bench/eval_method.py`（共享 `metrics/` 包）

## 0. 复现基础设施（所有对比方法共用）
为保证所有对比方法可比，先搭建统一基准管线（详见 `fusion_bench/BENCH_CONTRACT.md`）：
- **标准化输入**：三任务测试对统一导出为 8-bit 灰度 `fusion_bench/inputs/<task>/{A,B}/<stem>.png`，
  彩色源的 CbCr 另存 `cbcr/`（供 RGB 还原）。A=彩色/功能源，B=灰度/结构源。
  - `gfp_pc`（30 对，GFP-Y / PC）、`irvis`（MSRS 测试集均匀抽 50 对，VIS-Y / IR）、`medical`（Harvard 48 对，PET/SPECT-Y / MRI）。
- **统一输出契约**：每方法把融合图（灰度或 RGB）按 stem 命名写到 `fusion_bench/fused/<Method>/<task>/`。
- **统一评测**：`eval_method.py` 用共享 `metrics/` 计算 EN/MI/SD/SF/AG/SSIM/MS_SSIM/Qabf/VIF + 诊断 SCD/Nabf/CC + 功能轴 FuncCorr/FuncSal，逐图 CSV + 均值 + 任务级 leaderboard。
- **环境隔离**：每方法独立 venv 于 `/ytech_m2v4_hdd/lizhongyin/venv/<method>`，通过 `zzz_base.pth` 链接基础 venv 的 torch2.8（免重装），方法私有依赖装入子 venv。不触碰系统 python。
- **网络**：外网（GitHub/HF/PyTorch/Drive）经 `proxy_env.sh` 代理；内网 PyPI 镜像直连。

## 1. 方法与权重来源
- 论文：Zhao et al., *CDDFuse: Correlation-Driven Dual-Branch Feature Decomposition for Multi-Modality Image Fusion*, CVPR 2023.
- 仓库：作者官方 `Zhaozixiang1228/MMIF-CDDFuse`（已在 `code/ref/MMIF-CDDFuse`）。
- 权重（仓库内自带）：`models/CDDFuse_IVF.pth`（红外-可见光）、`models/CDDFuse_MIF.pth`（医学）。
  checkpoint key：`DIDF_Encoder / DIDF_Decoder / BaseFuseLayer / DetailFuseLayer`。
- 任务-权重映射：`irvis→IVF`；`medical→MIF`；`gfp_pc→MIF`（显微/医学近似，无专用权重）。

## 2. 环境与运行
- venv：`/ytech_m2v4_hdd/lizhongyin/venv/cddfuse`（base.pth 继承 torch2.8 + 私装 einops）。
- 推理：`CUDA_VISIBLE_DEVICES=0`，Restormer 编码器双分支（base/detail）特征相加融合 → 解码 → min-max 归一化出灰度图。A→VIS 槽，B→IR 槽。
- 全部为推理（无训练），三任务合计 128 张，单卡数十秒。

## 3. 结果（均值）

| 任务 | n | EN | MI | SD | SF | AG | SSIM | MS_SSIM | Qabf | VIF | SCD | Nabf | CC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| irvis   | 50 | 6.603 | 4.921 | 42.005 | 11.045 | 4.136 | 0.715 | 0.759 | 0.681 | 0.097 | 1.657 | 0.111 | 0.606 |
| medical | 48 | 5.364 | 3.605 | 81.298 | 28.646 | 10.958 | 0.727 | 0.752 | 0.725 | 0.096 | 1.770 | 0.078 | 0.853 |
| gfp_pc  | 30 | 6.888 | 3.164 | 33.240 | 11.418 | 5.539 | 0.487 | 0.593 | 0.564 | 0.067 | 1.356 | 0.202 | 0.511 |

功能轴：irvis FuncCorr 0.453 / FuncSal 0.790；medical 0.336 / 1.551；gfp_pc 0.252 / 0.207。
明细：`fusion_bench/reports/<task>/CDDFuse__{per_image,means}.csv`，并入各任务 `leaderboard.csv`。

## 4. 分析
- IR-VIS、医学上 CDDFuse 表现强（高 SD/Qabf/SSIM、低 Nabf），符合其 CVPR'23 SOTA 定位，可作为强对比锚点。
- GFP-PC 用医学权重跨域迁移：SSIM/VIF 偏低、Nabf 偏高（0.20），说明显微域有分布差异（GFP 大面积近黑背景），为本课题"通用 vs 专精"动机提供对照证据。
- 评测尺子自洽：与 AvgBaseline 相比 CDDFuse 在 SD/SF/AG/Qabf 等质量轴显著占优。

## 5. 下一步
- 其余对比方法（SwinFusion/U2Fusion/IFCNN/DenseFuse/SeAFusion/TarDAL/PIAFusion/RFN-Nest/LRRNet/DATFuse/DDFM + 传统 GTF/LP/DWT/NSCT）按同一契约用子 agent 并行复现，统一进 leaderboard 后做平均排名综合对比。
