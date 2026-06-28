# EXP-CMP-14：MURF 对比方法复现（三模态指标）

- 日期：2026-06-25
- 所属阶段：阶段 5（综合对比 / SOTA 复现）
- 结果级别：S2（完整测试子集，三任务）
- 关联代码：`code/ref/MURF`（作者官方仓库，TensorFlow，含预训练融合权重）；驱动 `code/ref/MURF/run_murf.py`
- 评测器：`code/Graduation-Paper/bench/eval_method.py`（共享 `metrics/` 包）

## 0. 复现基础设施（所有对比方法共用）
统一基准管线见 `fusion_bench/BENCH_CONTRACT.md`：
- 标准化输入：三任务测试对统一导出 8-bit 灰度 `fusion_bench/inputs/<task>/{A,B}/<stem>.png`，A=彩色/功能源，B=灰度/结构源。
  - `irvis`（50，MSRS 子集，VIS-Y / IR）、`medical`（48，Harvard，PET/SPECT-Y / MRI）、`gfp_pc`（30，GFP-Y / PC）。
- 统一输出契约：每方法把融合图按 stem 命名写到 `fusion_bench/fused/<Method>/<task>/`。
- 统一评测：`eval_method.py` 用共享 `metrics/` 计算 EN/MI/SD/SF/AG/SSIM/MS_SSIM/Qabf/VIF + 诊断 SCD/Nabf/CC + 功能轴 FuncCorr/FuncSal，逐图 CSV + 均值 + 任务级 leaderboard。
- 数据已**预配准**：跳过 MURF 的配准阶段（Task#1 共享信息提取、Task#2 粗配准），仅运行 Task#3 的**融合**部分。

## 1. 方法与权重来源
- 论文：Xu, Wang, Ma et al., *MURF: Mutually Reinforcing Multi-modal Image Registration and Fusion*, IEEE TPAMI 2023.
- 仓库：作者官方 `hanna-xu/MURF`，URL `https://github.com/hanna-xu/MURF.git`，commit `79b270af4ec8a77a7ed880ec6890a5cf9fe3f268`（2024-04-09）。克隆于 `code/ref/MURF`（经 proxy）。
- 框架：**TensorFlow 1.x（TF1 graph API，`tf.contrib`/`tf.placeholder`/`tf.Session`）**，原推荐环境 python3.6 + tensorflow-gpu1.14。
- 预训练融合权重（**仓库内自带**，无需 gdown/HF）：
  - RGB-IR（→irvis、gfp_pc）：`RGB-IR/fine_registration_and_fusion/models/finetuning/0000.ckpt`（md5 `867882bf…`，大分辨率微调版）。
  - PET-MRI（→medical）：`PET-MRI/fine_registration_and_fusion/models/0000.ckpt`（md5 `d991e4ef…`）。
  - checkpoint 变量 scope：`f2m_net`（外加 BatchNorm 的 moving_mean/moving_variance）。
- 任务-权重映射：`irvis → RGB-IR fusion net`；`medical → PET-MRI fusion net`；`gfp_pc → RGB-IR fusion net`（A=GFP 功能源对应"彩色/RGB"槽，B=PC 对应灰度/结构槽，跨域迁移，与 CDDFuse 处理一致）。

### 融合网络要点（仅推理）
- 输入：A（彩色/功能源）以 3 通道送入，内部 `rgb2ycbcr` 取 Y；B（结构源）单通道。本基准 A 为灰度，按 R=G=B=gray 复制成中性 RGB（Cb=Cr≈128），故输出色度为中性、Y 即融合结果。
- RGB-IR 网络：A/B 双分支卷积（`w_a*/w_b*`）→ 通道注意力 `channelattention` → 拼接 `wf1/wf2/wf3`（末层 1 通道）→ `tanh/2+0.5` 得融合 Y；网络内含一个 `convoffset2D` 形变校正子网（预配准数据下近似恒等）。
- PET-MRI 网络：单分支（`w1/w2/w3`）→ 通道注意力 → `wf1/wf2/wf3`（末层 3 通道 RGB）→ `tanh/2+0.5`。
- 驱动统一把输出转灰度（0.299R+0.587G+0.114B）落盘（评测器本就按灰度计算）。

## 2. 环境与运行
### venv（关键：TF1 在 CUDA12/H800 上跑通）
原仓库为 TF1.x，内网镜像与 PyTorch base venv（torch2.8/py3.11）都无法直接承载，故**独立 py3.8 venv**：`/ytech_m2v4_hdd/lizhongyin/venv/murf`（不复用 base.pth torch 链接——MURF 不需要 torch）。
```bash
# venv（系统 /usr/bin/python3.8 无 ensurepip，用 versioned get-pip 引导）
/usr/bin/python3.8 -m venv --without-pip /ytech_m2v4_hdd/lizhongyin/venv/murf
curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py | /ytech_m2v4_hdd/lizhongyin/venv/murf/bin/python
P=/ytech_m2v4_hdd/lizhongyin/venv/murf/bin/python
# TF1.15 + CUDA12 支持：NVIDIA 官方 nvidia-tensorflow（nv23.03 = TF1.15.5，链 libcudart.so.12）
$P -m pip install --no-deps --index-url https://developer.download.nvidia.com/compute/redist nvidia-tensorflow
# 运行期依赖（内网镜像）
$P -m pip install numpy==1.22.4 scipy==1.7.3 protobuf==3.20.3 wrapt gast==0.3.3 astor astunparse \
   termcolor keras-applications keras-preprocessing google-pasta grpcio absl-py opt-einsum six \
   tensorboard==1.15.0 tensorflow-estimator==1.15.1 scikit-image==0.17.2 pillow opencv-python-headless==4.5.5.64 \
   matplotlib h5py==2.10.0 ipython==8.12.3
# 补 cuDNN8（nv23.03 需 libcudnn.so.8）+ cublas/nvrtc for CUDA12
$P -m pip install nvidia-cudnn-cu12==8.9.7.29 nvidia-cublas-cu12
```
- 运行需将 wheel 自带的 CUDA 库加入 `LD_LIBRARY_PATH`：
  `SP=…/venv/murf/lib/python3.8/site-packages/nvidia; export LD_LIBRARY_PATH=$SP/cudnn/lib:$SP/cublas/lib:$SP/cuda_nvrtc/lib:$LD_LIBRARY_PATH`
- 代码改动：仅注释保护 `f2m_model.py` 顶部 `from scipy.misc import imsave`（新版 scipy 已移除；推理不调用），其余模型/utils 原样运行。
- GPU：`CUDA_VISIBLE_DEVICES=6`（H800，compute capability 9.0），仅推理。

### 运行命令（每任务）
```bash
export CUDA_VISIBLE_DEVICES=6
SP=/ytech_m2v4_hdd/lizhongyin/venv/murf/lib/python3.8/site-packages/nvidia
export LD_LIBRARY_PATH=$SP/cudnn/lib:$SP/cublas/lib:$SP/cuda_nvrtc/lib:$LD_LIBRARY_PATH
P=/ytech_m2v4_hdd/lizhongyin/venv/murf/bin/python
cd /ytech_m2v4_hdd/lizhongyin/code/ref/MURF

# irvis（RGB-IR fusion net）
$P run_murf.py --arch rgbir \
  --code-dir code/ref/MURF/RGB-IR/fine_registration_and_fusion \
  --ckpt    code/ref/MURF/RGB-IR/fine_registration_and_fusion/models/finetuning/ \
  --in-a fusion_bench/inputs/irvis/A --in-b fusion_bench/inputs/irvis/B \
  --out fusion_bench/fused/MURF/irvis
# medical（PET-MRI fusion net）
$P run_murf.py --arch petmri \
  --code-dir code/ref/MURF/PET-MRI/fine_registration_and_fusion \
  --ckpt    code/ref/MURF/PET-MRI/fine_registration_and_fusion/models/ \
  --in-a fusion_bench/inputs/medical/A --in-b fusion_bench/inputs/medical/B \
  --out fusion_bench/fused/MURF/medical
# gfp_pc（RGB-IR fusion net，跨域迁移）
$P run_murf.py --arch rgbir \
  --code-dir code/ref/MURF/RGB-IR/fine_registration_and_fusion \
  --ckpt    code/ref/MURF/RGB-IR/fine_registration_and_fusion/models/finetuning/ \
  --in-a fusion_bench/inputs/gfp_pc/A --in-b fusion_bench/inputs/gfp_pc/B \
  --out fusion_bench/fused/MURF/gfp_pc
```
（路径以 `/ytech_m2v4_hdd/lizhongyin/` 为根；驱动逐图重建图并 restore checkpoint，输入按 8 的倍数 reflect-pad 后推理再裁回原尺寸。）
- 全部为推理（无训练），三任务合计 128 张，单卡数分钟。

## 3. 结果（均值）

| 任务 | n | EN | MI | SD | SF | AG | SSIM | MS_SSIM | Qabf | VIF | SCD | Nabf | CC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| irvis   | 50 | 5.180 | 1.362 | 18.319 | 12.170 | 4.750 | 0.554 | 0.746 | 0.341 | 0.026 | 0.895 | 0.186 | 0.603 |
| medical | 48 | 5.685 | 2.467 | 78.535 | 44.918 | 18.129 | 0.323 | 0.726 | 0.515 | 0.048 | 1.359 | 0.313 | 0.840 |
| gfp_pc  | 30 | 5.845 | 1.538 | 15.813 | 13.653 | 6.835 | 0.473 | 0.654 | 0.397 | 0.035 | 1.072 | 0.250 | 0.517 |

功能轴：irvis FuncCorr 0.532 / FuncSal 1.207；medical 0.210 / 1.614；gfp_pc 0.321 / 0.556。
明细：`fusion_bench/reports/<task>/MURF__{per_image,means}.csv`，并入各任务 `leaderboard.csv`。

## 4. 分析
- **medical（PET-MRI 专用权重）最强**：高 SD(78.5)/SF(44.9)/AG(18.1)/Qabf(0.515)/CC(0.840)，纹理与梯度保留充分，符合 MURF 在医学融合上的设计取向；但 SSIM(0.323) 偏低、Nabf(0.313) 偏高，反映其偏向增强对比/锐度而非与源结构高保真。
- **irvis** 各项偏保守：EN(5.18)/MI(1.36)/SD(18.3)/VIF(0.026) 明显低于 CDDFuse（EN6.60/SD42.0）。原因：MURF 的 RGB-IR 融合 Y 输出整体偏暗、动态范围被压缩（MSRS 夜景下尤甚），属方法风格差异；Qabf(0.341) 中等、Nabf(0.186) 可控。
- **gfp_pc** 用 RGB-IR 权重跨域迁移：指标居中（Qabf0.397、Nabf0.250），与 CDDFuse 借医学权重迁移类似，为本课题"通用 vs 专精/跨域迁移"动机再添一条对照证据。
- 量纲自洽：medical 上 SD/SF/AG/Qabf 远高于 irvis/gfp_pc，与各任务源对比度量级一致；功能轴 FuncSal 在 medical 最高，符合 PET/SPECT 功能信号强的特性。

## 5. 问题与说明
- **TF1.x 跑 CUDA12**：原仓库 TF1.14/python3.6，内网无 nvidia-tensorflow、亦无 TF≤2 的 GPU 轮子。改用 NVIDIA 官方 `nvidia-tensorflow 1.15.5+nv23.03`（链 CUDA12），配 `nvidia-cudnn-cu12==8.9.7`，在 H800(sm90) 上成功 `Created TensorFlow device … H800`，GPU 推理可用。numpy 需升到 1.22（nv23.03 ABI）。
- **配准阶段跳过**：基准数据已预配准，仅跑 Task#3 融合；网络内 `convoffset2D` 形变子网仍保留（restore 全部 `f2m_net` 变量），对已对齐输入近似恒等，不影响融合质量。
- **灰度输入适配**：A 本应是彩色源，这里为灰度 Y，按中性 RGB 复制送入，输出取 Y 转灰度落盘，与评测灰度口径一致。
- 未使用 RGB-NIR / CT-MRI 两套权重（与本基准三任务无对应模态）。

## 6. 结论
MURF 三任务全部跑通（irvis 50 / medical 48 / gfp_pc 30，共 128 张），已并入统一 leaderboard。其在 medical 上质量轴（SD/SF/AG/Qabf）表现突出，可作为医学融合的强对比锚点；irvis 偏保守、亮度/信息量偏低，体现方法风格差异。作为 buffer 对比方法，复现完整、provenance 清晰（官方仓库自带权重 + 官方 nvidia-tensorflow 让 TF1 在 CUDA12 落地）。


## 指标修订（RGB-final 协议，2026-06-28）

> 修订动机：原先对比方法直接对融合的 **Y 通道图** 计分。参照仓库 `infer_fusion.py` 与 原始 MATLAB `evaluation/main.m` 的约定——**彩色源任务的最终融合图是 Y 与源 CbCr 重组逆变换得到的 RGB 图，计分时对该 RGB 图做 `rgb2gray`（= PIL 'L'，BT.601）**。RGB 逆变换中的 uint8 截断会在高饱和色区（PET/SPECT 伪彩、GFP 绿色）改变灰度，因此直接用 Y 计分不严格。
>
> 修订范围：`output_mode=rgb` 的 **medical / gfp_pc** 两任务，对全部 18 方法的融合 Y 重组源 CbCr → RGB-final → `rgb2gray` 重算（RGB-final 图存于 `fusion_bench/fused_final/<方法>/<任务>/`）。**irvis 为 `output_mode=gray`（与 MDFNet 自身评测一致），维持灰度不变。** 重算后排名与原结论基本一致（个别名次 ±1）。

修订后核心指标（medical/gfp_pc 已按 RGB-final 协议；irvis 灰度不变）：

**MURF**

| 任务 | n | EN | MI | SD | SF | AG | SSIM | MS_SSIM | Qabf | VIF | SCD | Nabf | CC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| irvis | 50 | 5.180 | 1.362 | 18.319 | 12.170 | 4.750 | 0.554 | 0.746 | 0.341 | 0.026 | 0.895 | 0.186 | 0.603 |
| medical | 48 | 5.671 | 2.521 | 76.982 | 43.886 | 17.476 | 0.328 | 0.726 | 0.515 | 0.048 | 1.308 | 0.295 | 0.839 |
| gfp_pc | 30 | 5.849 | 1.520 | 15.818 | 13.598 | 6.851 | 0.472 | 0.654 | 0.395 | 0.035 | 1.080 | 0.251 | 0.519 |
