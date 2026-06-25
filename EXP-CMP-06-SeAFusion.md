# EXP-CMP-06：SeAFusion 对比方法复现（三模态指标）

- 日期：2026-06-25
- 所属阶段：阶段 5（综合对比 / SOTA 复现）
- 结果级别：S2（完整测试子集，三任务）
- 关联代码：`code/ref/SeAFusion`（作者仓库，含预训练权重）；驱动 `code/ref/SeAFusion/run_seafusion_bench.py`
- 评测器：`code/Graduation-Paper/bench/eval_method.py`（共享 `metrics/` 包）

## 0. 复现基础设施（所有对比方法共用）
统一基准管线见 `fusion_bench/BENCH_CONTRACT.md`：
- **标准化输入**：三任务测试对统一导出为 8-bit 灰度 `fusion_bench/inputs/<task>/{A,B}/<stem>.png`。A=可见光/功能源，B=红外/结构源。
  - `gfp_pc`（30 对，GFP-Y / PC）、`irvis`（MSRS 子集 50 对，VIS-Y / IR）、`medical`（Harvard 48 对，PET/SPECT-Y / MRI）。
- **统一输出契约**：融合图按 stem 命名写到 `fusion_bench/fused/SeAFusion/<task>/`。
- **统一评测**：`eval_method.py` 用共享 `metrics/` 计算 EN/MI/SD/SF/AG/SSIM/MS_SSIM/Qabf/VIF + 诊断 SCD/Nabf/CC + 功能轴 FuncCorr/FuncSal。
- **环境隔离**：独立 venv `/ytech_m2v4_hdd/lizhongyin/venv/seafusion`，经 `zzz_base.pth` 链接基础 venv 的 torch2.8（免重装）。
- **网络**：外网经 `proxy_env.sh`；内网 PyPI 镜像直连。

## 1. 方法与权重来源
- 论文：Tang et al., *Image fusion in the loop of high-level vision tasks: A semantic-aware real-time infrared and visible image fusion network*, Information Fusion 2022.
- 仓库：作者官方 `Linfeng-Tang/SeAFusion`，clone 至 `code/ref/SeAFusion`，commit `da0028d`（"Update CVPR and TIP publication links"）。
- 权重（仓库内自带，无需下载）：`model/Fusion/fusionmodel_final.pth`（MSRS 上训练的 IR-VIS 融合网络 `FusionNet`，`output=1`）。
- 网络结构（`FusionNet.py`）：VIS-Y 与 IR 各走 `ConvLeakyRelu → 2×RGBD（DenseBlock + Sobel 梯度支路）` 编码器，特征 concat 后经 4 段 decode（`ConvBnLeakyRelu×3 + ConvBnTanh`），输出 `tanh(·)/2+0.5 ∈ [0,1]`。
- 任务-权重映射：单一预训练 IR-VIS 权重用于全部三任务。A→VIS-Y 槽，B→IR 槽（契约约定 B→IR 一致）。

## 2. 环境与运行
环境搭建：
```
/opt/conda/bin/python3.11 -m venv /ytech_m2v4_hdd/lizhongyin/venv/seafusion
echo "/ytech_m2v4_hdd/lizhongyin/venv/lib/python3.11/site-packages" \
  > /ytech_m2v4_hdd/lizhongyin/venv/seafusion/lib/python3.11/site-packages/zzz_base.pth
# 继承基础 venv：torch 2.8.0+cu128 / torchvision 0.23 / numpy / PIL，无需额外安装
```
仓库 clone：
```
source /ytech_m2v4_hdd/lizhongyin/proxy_env.sh
git clone https://github.com/Linfeng-Tang/SeAFusion /ytech_m2v4_hdd/lizhongyin/code/ref/SeAFusion
```
推理（`CUDA_VISIBLE_DEVICES=4` → cuda:0，纯推理无训练）：
```
export CUDA_VISIBLE_DEVICES=4
cd /ytech_m2v4_hdd/lizhongyin/code/ref/SeAFusion
for t in irvis medical gfp_pc; do
  /ytech_m2v4_hdd/lizhongyin/venv/seafusion/bin/python run_seafusion_bench.py --task $t
done
```
驱动 `run_seafusion_bench.py` 关键处理：
- A、B 各以 `convert('L')` 读为单通道 [0,1]；A 作为 VIS-Y 直接喂入（`forward` 取 `image_vis[:,:1]`），B 作为 IR。
- 输出用仓库原生 `utils.save_img_single`（min-max 归一化 → uint8，灰度复制为 RGB png），与 SeAFusion 官方落盘一致。
- 因测试输入已是灰度 Y，省去官方 test.py 中 RGB2YCrCb/YCbCr2RGB 的色彩还原步骤，直接保存融合 Y。

评测：
```
for t in irvis medical gfp_pc; do
  /ytech_m2v4_hdd/lizhongyin/venv/bin/python \
    /ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper/bench/eval_method.py \
    --task $t --name SeAFusion --fused-dir /ytech_m2v4_hdd/lizhongyin/fusion_bench/fused/SeAFusion/$t
done
```

## 3. 结果（均值）

| 任务 | n | EN | MI | SD | SF | AG | SSIM | MS_SSIM | Qabf | VIF | SCD | Nabf | CC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| irvis   | 50 | 6.590 | 3.983 | 41.628 | 10.903 | 4.112 | 0.718 | 0.765 | 0.668 | 0.073 | 1.725 | 0.122 | 0.616 |
| medical | 48 | 5.741 | 2.912 | 77.573 | 24.229 | 9.723 | 0.662 | 0.752 | 0.640 | 0.053 | 1.591 | 0.083 | 0.855 |
| gfp_pc  | 30 | 6.867 | 3.362 | 30.170 | 12.878 | 6.494 | 0.507 | 0.605 | 0.581 | 0.072 | 1.415 | 0.267 | 0.510 |

功能轴：irvis FuncCorr 0.460 / FuncSal 0.831；medical 0.464 / 1.619；gfp_pc 0.276 / 0.142。
明细：`fusion_bench/reports/<task>/SeAFusion__{per_image,means}.csv`，并入各任务 `leaderboard.csv`。

## 4. 分析
- **IR-VIS（域内）**：SeAFusion 表现稳健——高 Qabf(0.668)/SSIM(0.718)/SCD(1.73)，符合其在 MSRS 上训练的本域优势，可作为强对比锚点。Sobel 梯度支路保证边缘传递（Qabf 高）。
- **医学（跨域）**：SD/SF/AG 三项最高（结构对比强），但 MI/VIF 偏低、EN 仅 5.74，说明用 IR-VIS 权重迁移到 PET-MRI 时信息保真有损失；CC 高(0.855) 表明融合图与源相关性仍强。
- **GFP-PC（跨域最远）**：SSIM(0.507)/MS_SSIM(0.605) 最低、Nabf 最高(0.267)，与 CDDFuse 在该任务的退化一致——显微域（GFP 大面积近黑背景）与 IR-VIS 分布差异大，伪影增多，为"通用 vs 专精"动机提供对照证据。
- 与 CDDFuse（EXP-CMP-01）相比：irvis 两者 SSIM/Qabf 接近（SeAFusion VIF 偏低 0.073 vs 0.097）；medical 上 CDDFuse 的 SD/SSIM 更高，SeAFusion 的 SF/AG 更突出。整体处于合理的 SOTA 区间，评测尺子自洽。

## 5. 问题与说明
- 单一 IR-VIS 预训练权重覆盖三任务（仓库未提供医学/显微专用权重），跨域结果含分布迁移影响，已在分析中标注。
- 输出沿用官方 min-max 归一化落盘，与 SeAFusion 原始视觉效果一致；评测器内部转灰度计算，RGB/灰度落盘不影响指标。

## 6. 结论
SeAFusion 预训练 `FusionNet` 在统一基准三任务上全部跑通（irvis 50 / medical 48 / gfp_pc 30，共 128 张），指标已并入 leaderboard。本域（IR-VIS）质量与边缘保持优秀，跨域（医学、显微）随分布差异递减，可作为对比基线之一。未做训练、未改共享基础设施、未 push。
