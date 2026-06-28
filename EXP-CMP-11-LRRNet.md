# EXP-CMP-11：LRRNet 对比方法复现（三模态指标）

- 日期：2026-06-25
- 所属阶段：阶段 5（综合对比 / SOTA 复现）
- 结果级别：S2（完整测试子集，三任务）
- 关联代码：`code/ref/LRRNet`（作者官方仓库，含预训练 `.model` 权重）；自写干净驱动 `code/ref/LRRNet/run_lrrnet_clean.py`（复用 `net_lista.LRR_NET` 网络定义 + state_dict）
- 评测器：`code/Graduation-Paper/bench/eval_method.py`（共享 `metrics/` 包）

## 0. 复现基础设施（所有对比方法共用）
统一基准管线见 `fusion_bench/BENCH_CONTRACT.md`：
- **标准化输入**：三任务测试对统一导出为 8-bit 灰度 `fusion_bench/inputs/<task>/{A,B}/<stem>.png`。A=彩色/功能源（VIS/PET/GFP-Y），B=灰度/结构源（IR/MRI/PC）。
  - `gfp_pc`（30 对，GFP-Y / PC）、`irvis`（MSRS 子集 50 对，VIS-Y / IR）、`medical`（Harvard 48 对，PET/SPECT-Y / MRI）。
- **统一输出契约**：每方法把融合图按 stem 写到 `fusion_bench/fused/<Method>/<task>/`。
- **统一评测**：`eval_method.py` 共享 `metrics/` 计算 EN/MI/SD/SF/AG/SSIM/MS_SSIM/Qabf/VIF + 诊断 SCD/Nabf/CC + 功能轴 FuncCorr/FuncSal。
- **环境隔离**：方法独立 venv，通过 `zzz_base.pth` 链接基础 venv 的 torch2.8（免重装），私有依赖装入子 venv。
- **网络**：外网经 `proxy_env.sh` 代理；内网 PyPI 镜像直连。

## 1. 方法与权重来源
- 论文：Li et al., *LRRNet: A Novel Representation Learning Guided Fusion Network for Infrared and Visible Images*, TPAMI 2023.
- 仓库：作者官方 `hli1221/imagefusion-LRRNet`，clone 至 `code/ref/LRRNet`，commit `e489fc325ea6acb9b7327b1de2c0cc4a09172ba4`（"Update README.md"）。
- 权重（仓库内自带）：`model/final_lrr_net_lam2_1.5_wir_3.0_lam3_gram_2000_epoch_4_block_4.model`（218 KB，约 0.049M 参数）。
  网络配置：`LRR_NET(s=3, n=128, channel=1, stride=1, num_block=4, fusion_type='cat')`，state_dict 与该配置 **All keys matched successfully**（strict 加载）。
- 任务-权重映射：唯一一个红外-可见光预训练模型，三任务复用（B→IR 槽，A→VIS 槽）。

## 2. 环境与运行
- venv 创建：
  ```
  /opt/conda/bin/python3.11 -m venv /ytech_m2v4_hdd/lizhongyin/venv/lrrnet
  echo "/ytech_m2v4_hdd/lizhongyin/venv/lib/python3.11/site-packages" \
    > /ytech_m2v4_hdd/lizhongyin/venv/lrrnet/lib/python3.11/site-packages/zzz_base.pth   # 继承 torch2.8
  /ytech_m2v4_hdd/lizhongyin/venv/lrrnet/bin/pip install opencv-python-headless pillow numpy
  ```
  torch 2.8.0+cu128（继承基础 venv），cv2 4.13.0，PIL/numpy 私装。
- 干净驱动（仓库 `testing_fusion_lrr.py` 用了硬编码 TNO 路径 + cv2 旧约定，故自写 `run_lrrnet_clean.py`，仅复用 `net_lista.LRR_NET` 定义 + 预训练 state_dict）：
  - 读图：PIL `convert("L")` 取 8-bit 灰度 [0,255] → tensor [1,1,H,W]。
  - 预处理：`normalize_tensor`（逐图 min-max 到 [0,1]），与仓库 `utils.normalize_tensor` 一致。
  - 前向：`model(B_ir, A_vis)['fuse']`（仓库约定 `model(img_ir, img_vi)`，即 B→IR、A→VIS 槽）。
  - 出图：对 fuse 做 min-max 归一化 → [0,255] uint8 灰度 PNG，与 `utils.save_image` 一致。
  - 网络全卷积 stride=1，无固定输入尺寸，输出尺寸 == 输入（无需 pad/crop）。
- GPU：`CUDA_VISIBLE_DEVICES=2`，纯推理（无训练），三任务合计 128 张，单卡数十秒。
- 运行命令：
  ```
  export CUDA_VISIBLE_DEVICES=2
  for t in irvis medical gfp_pc; do
    /ytech_m2v4_hdd/lizhongyin/venv/lrrnet/bin/python run_lrrnet_clean.py --task $t
  done
  ```
- 评测命令：
  ```
  for t in irvis medical gfp_pc; do
    /ytech_m2v4_hdd/lizhongyin/venv/bin/python \
      /ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper/bench/eval_method.py \
      --task $t --name LRRNet \
      --fused-dir /ytech_m2v4_hdd/lizhongyin/fusion_bench/fused/LRRNet/$t
  done
  ```

## 3. 结果（均值）

| 任务 | n | EN | MI | SD | SF | AG | SSIM | MS_SSIM | Qabf | VIF | SCD | Nabf | CC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| irvis   | 50 | 6.109 | 2.935 | 31.147 | 8.140 | 2.929 | 0.609 | 0.721 | 0.413 | 0.046 | 0.740 | 0.046 | 0.516 |
| medical | 48 | 5.426 | 2.493 | 50.673 | 13.977 | 6.050 | 0.195 | 0.692 | 0.170 | 0.030 | 0.348 | 0.077 | 0.836 |
| gfp_pc  | 30 | 6.325 | 1.779 | 29.201 | 13.456 | 6.487 | 0.380 | 0.640 | 0.317 | 0.034 | 1.638 | 0.249 | 0.604 |

功能轴：irvis FuncCorr 0.311 / FuncSal 0.200；medical 0.523 / 2.334；gfp_pc 0.846 / 2.236。
明细：`fusion_bench/reports/<task>/LRRNet__{per_image,means}.csv`，并入各任务 `leaderboard.csv`。

## 4. 分析
- IR-VIS（域内，预训练任务）表现最稳：高 SSIM/MS_SSIM、低 Nabf（0.046，几乎无伪影），Qabf 0.413 中等。LRRNet 走低秩(L)+稀疏(S)表示分解再 cat 重构，倾向"温和保边"，故 EN/SD/SF 等绝对强度指标低于 CDDFuse 一档——这是其表示学习驱动、轻量(0.05M)架构的固有取向。
- medical（跨域）SSIM 仅 0.195：PET-Y 与 MRI 的强度分布差异大，单一 IR-VIS 权重 + 逐图 min-max 重构后整体偏暗、对比塌缩，结构相似度被拉低；但 CC 高(0.836)、Nabf 低(0.077)说明无明显伪影，属"保守融合"而非"破坏性融合"。
- gfp_pc（跨域显微）SCD 高(1.638)、FuncCorr 高(0.846) 说明功能信息（GFP）保留较好，但 Nabf 偏高(0.249)、MI 低(1.779)，显微域分布差异引入更多噪声边——与 CDDFuse 在该域的结论一致，为本课题"通用 vs 专精"动机提供又一对照证据。
- 评测尺子自洽：strict state_dict 全键匹配、输出尺寸与输入一致、范围有界，复现可信。

## 5. 问题与限制
- 仓库 `testing_fusion_lrr.py` 含硬编码 TNO 路径与 cv2 旧式读写约定，直接跑会失败；按契约改用自写驱动复用网络定义 + 权重，规避死 API。
- 仅有一个 IR-VIS 预训练模型，medical/gfp_pc 为跨域迁移，无专用权重（作者未提供）；训练成本与契约对"无专用权重则用现有预训练"的指引一致，故未额外训练。

## 6. 下一步
- 结果已并入三任务 leaderboard，与 CDDFuse 等方法一并做平均排名综合对比。


## 指标修订（RGB-final 协议，2026-06-28）

> 修订动机：原先对比方法直接对融合的 **Y 通道图** 计分。参照仓库 `infer_fusion.py` 与 原始 MATLAB `evaluation/main.m` 的约定——**彩色源任务的最终融合图是 Y 与源 CbCr 重组逆变换得到的 RGB 图，计分时对该 RGB 图做 `rgb2gray`（= PIL 'L'，BT.601）**。RGB 逆变换中的 uint8 截断会在高饱和色区（PET/SPECT 伪彩、GFP 绿色）改变灰度，因此直接用 Y 计分不严格。
>
> 修订范围：`output_mode=rgb` 的 **medical / gfp_pc** 两任务，对全部 18 方法的融合 Y 重组源 CbCr → RGB-final → `rgb2gray` 重算（RGB-final 图存于 `fusion_bench/fused_final/<方法>/<任务>/`）。**irvis 为 `output_mode=gray`（与 MDFNet 自身评测一致），维持灰度不变。** 重算后排名与原结论基本一致（个别名次 ±1）。

修订后核心指标（medical/gfp_pc 已按 RGB-final 协议；irvis 灰度不变）：

**LRRNet**

| 任务 | n | EN | MI | SD | SF | AG | SSIM | MS_SSIM | Qabf | VIF | SCD | Nabf | CC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| irvis | 50 | 6.109 | 2.935 | 31.147 | 8.140 | 2.929 | 0.609 | 0.721 | 0.413 | 0.046 | 0.740 | 0.046 | 0.515 |
| medical | 48 | 5.407 | 2.500 | 50.045 | 13.825 | 5.954 | 0.196 | 0.692 | 0.169 | 0.031 | 0.314 | 0.072 | 0.836 |
| gfp_pc | 30 | 6.316 | 1.763 | 28.827 | 13.359 | 6.501 | 0.379 | 0.641 | 0.313 | 0.034 | 1.638 | 0.249 | 0.604 |
