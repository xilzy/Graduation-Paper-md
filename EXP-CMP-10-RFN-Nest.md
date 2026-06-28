# EXP-CMP-10：RFN-Nest 对比方法复现（三模态指标）

- 日期：2026-06-25
- 所属阶段：阶段 5（综合对比 / SOTA 复现）
- 结果级别：S2（完整测试子集，三任务）
- 关联代码：`code/ref/RFN-Nest`（作者官方仓库 `hli1221/imagefusion-rfn-nest`，含预训练权重）；驱动 `code/Graduation-Paper/bench/run_rfnnest.py`
- 评测器：`code/Graduation-Paper/bench/eval_method.py`（共享 `metrics/` 包）

## 0. 复现基础设施（所有对比方法共用）
统一基准管线见 `fusion_bench/BENCH_CONTRACT.md`：标准化 8-bit 灰度输入 `fusion_bench/inputs/<task>/{A,B}/<stem>.png`（A=可见/功能源，B=红外/结构源），统一输出 `fusion_bench/fused/<Method>/<task>/<stem>.png`，统一评测 `eval_method.py`，每方法独立 venv（`zzz_base.pth` 继承基础 venv 的 torch2.8），外网经 `proxy_env.sh` 代理、内网 PyPI 镜像直连。

## 1. 方法与权重来源（provenance）
- 论文：Li, Wu & Kittler, *RFN-Nest: An end-to-end residual fusion network for infrared and visible images*, Information Fusion 2021.
- **来源 = 官方仓库（非 re-impl，非训练）**：作者本人 `hli1221/imagefusion-rfn-nest`。
  - 仓库克隆至 `code/ref/RFN-Nest`，commit `76e4e0500a3d818068f31942d7d5ed874c36c37b`。
  - 预训练权重随仓库自带，以 zip 形式提供，解压后：
    - **Nest 自编码器**：`models/nestfuse/nestfuse_gray_1e2.model`（灰度 NestFuse 编/解码器，第一阶段在 MS-COCO 上以像素+SSIM 重建损失训练）。
    - **RFN 残差融合模块（两阶段）**：`models/rfn_twostage/6.0/Final_epoch_2_alpha_700_wir_6.0_wvi_3.0_ssim_vi.model`（第二阶段在 KAIST IR/VIS 上训练的 4 组 `FusionBlock_res`，alpha=700, w_ir=6.0, w_vi=3.0）。
  - 网络结构（`net.py`）：编码器 `NestFuse_light2_nodense(nb_filter=[64,112,160,208,256], in=out=1, deepsupervision=False)`，融合 `Fusion_network(nb_filter, 'res')`；`load_state_dict` 直接可载。
- 端口适配说明：原仓库 `utils.py` 依赖已废弃的 `scipy.misc.imread/imsave/imresize`，`test_*.py` 使用 Python2 风格的 `is` 字符串/浮点比较；在 torch2.8/新版 scipy 下不可用。**只复用承载权重的 `net.py`（网络结构）与两个 `.model`（state_dict）**，I/O 与融合用自写干净驱动 `bench/run_rfnnest.py`（PIL + torch），不改动仓库网络定义。
- 输入/输出尺度约定（与原仓库一致，关键）：
  - 输入：灰度按 **[0,255] 像素域**直接喂入（`get_train_images`/`get_test_image` 均为 `mode='L'` 的 raw float，**不做 /255 归一化**）。
  - 输出：原仓库 `utils.save_image_test` 的实际行为是 **逐图 min-max 归一化到 [0,255]**（`(x-min)/(max-min)*255`，clamp 那一行已注释掉），训练损失同样对 decoder 输出先 min-max 再 ×255。驱动严格照此实现。
  - 踩坑：最初误用 `clamp(0,255)` 直存导致三任务输出大面积饱和到 255（decoder 原始动态范围远超 [0,255]），改为 min-max 后分布恢复正常（无 255 饱和）。

## 2. 环境与运行
- venv：`/ytech_m2v4_hdd/lizhongyin/venv/rfnnest`
  ```
  /opt/conda/bin/python3.11 -m venv /ytech_m2v4_hdd/lizhongyin/venv/rfnnest
  echo "/ytech_m2v4_hdd/lizhongyin/venv/lib/python3.11/site-packages" \
    > /ytech_m2v4_hdd/lizhongyin/venv/rfnnest/lib/python3.11/site-packages/zzz_base.pth
  ```
  通过 `zzz_base.pth` 继承基础 venv 的 torch 2.8.0+cu128，无需额外 pip 安装（仅用 torch/numpy/PIL）。
- 权重解压（仓库内 zip）：
  ```
  cd code/ref/RFN-Nest/models && python -c \
    "import zipfile;[zipfile.ZipFile(z).extractall('.') for z in ['nestfuse.zip','rfn_twostage.zip']]"
  ```
- 推理：`export CUDA_VISIBLE_DEVICES=3`（单卡 H800），全部为推理无训练，三任务合计 128 张，单卡数秒完成。
- 融合策略：论文测试期默认 **RFN（`fs_type='res'`, `use_strategy=False`）** —— 4 组 `FusionBlock_res` 残差融合编码特征，再经 NestFuse `decoder_eval` 解码（非 add/avg/max/spa/nuclear 静态策略）。
- 尺寸处理：所有标准输入均 ≤512（irvis 640×480、medical 256×256、gfp_pc 358×358），走非分块路径；驱动将 H/W 反射 pad 到 8 的倍数（编码器三次 2× 下采样需要），解码后裁回原尺寸再 min-max。
- 运行命令：
  ```
  export CUDA_VISIBLE_DEVICES=3
  PY=/ytech_m2v4_hdd/lizhongyin/venv/rfnnest/bin/python
  for t in irvis medical gfp_pc; do
    $PY /ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper/bench/run_rfnnest.py --task $t
  done
  ```
- 任务-槽位映射：RFN-Nest 为非对称 IR/VIS 方法。按契约 **B→IR 槽（`en_ir`），A→VIS 槽（`en_vi`）**：irvis A=可见、B=红外天然契合；medical A=PET/SPECT→VIS、B=MRI→IR；gfp_pc A=GFP-Y→VIS、B=PC→IR（跨域零样本）。

## 3. 结果（均值）

| 任务 | n | EN | MI | SD | SF | AG | SSIM | MS_SSIM | Qabf | VIF | SCD | Nabf | CC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| irvis   | 50 | 6.097 | 2.447 | 28.309 | 5.878 | 2.134 | 0.711 | 0.793 | 0.382 | 0.046 | 1.490 | 0.023 | 0.670 |
| medical | 48 | 5.636 | 2.766 | 70.603 | 8.775 | 3.766 | 0.643 | 0.760 | 0.182 | 0.041 | 1.453 | 0.029 | 0.883 |
| gfp_pc  | 30 | 6.783 | 2.172 | 30.369 | 6.964 | 3.599 | 0.470 | 0.649 | 0.339 | 0.045 | 1.858 | 0.144 | 0.639 |

功能轴：irvis FuncCorr 0.452 / FuncSal 1.159；medical 0.457 / 2.051；gfp_pc 0.606 / 1.561。
明细：`fusion_bench/reports/<task>/RFN-Nest__{per_image,means}.csv`，并入各任务 `leaderboard.csv`。

## 4. 分析
- RFN-Nest 是 2021 年"端到端残差融合 + Nest 多尺度自编码器"方法，整体呈 **温和/平滑融合**特征：**SF/AG/Qabf 三任务均偏低**（如 irvis Qabf 0.382、AG 2.13，明显低于 CDDFuse 0.681/4.14 与 DenseFuse 0.633/3.31），**Nabf 很低**（irvis 0.023 / medical 0.029，几乎不引入伪影）；SSIM/MS-SSIM 稳健（irvis MS-SSIM 0.793，三方法中最高之一）。这与其"特征域残差融合 + min-max 输出归一化"的设计一致——保结构、低伪影，但细节锐度与边缘转移弱于显式细节增强类方法。
- 医学任务上 **SD 70.6、CC 0.883** 表现强（高对比、与源高相关），但 Qabf 仅 0.182 偏低——min-max 拉伸提升了全局对比/标准差，却未显著提升梯度级边缘保真。
- GFP-PC（跨域零样本，RFN 在 KAIST IR/VIS 上训练）：**SCD 1.858（三任务最高）、FuncCorr 0.606（显著高于 CDDFuse 0.252 / DenseFuse 0.262）**，说明残差融合较好保留了 GFP 功能信号与两源差异成分；但 Nabf 0.144 偏高、SSIM 0.470 偏低，反映显微域分布差异（GFP 大面积近黑背景）下伪影上升。可作为"低伪影、温和、强源相关"一端的对比锚点。
- 与同期对比锚点（CDDFuse 强增强、DenseFuse 温和重建）相比，RFN-Nest 介于二者之间偏"温和"：质量轴（SF/AG/Qabf）弱于 CDDFuse，但 SCD/CC/功能相关性更突出，体现"特征级残差融合更重源信息保真而非锐度增强"。

## 5. 问题与说明
- 原仓库 `utils.py`/`test_*.py` 依赖废弃 API（`scipy.misc`）与 Python2 风格 `is` 比较，直接跑不通；已绕开，仅复用 `net.py` + 两个 state_dict，融合/IO 自写，确保与契约一致且可复现。
- **输出归一化坑**：误用 clamp 会导致全饱和；原仓库实为逐图 min-max，已对齐（详见 §1）。
- 未训练：作者预训练 NestFuse 自编码器 + RFN 两阶段融合权重均可用且匹配论文，按契约 §6 优先用预训练，无需在本地数据上重训。

## 6. 下一步
- 与其余对比方法（CDDFuse、DenseFuse 已完成、SwinFusion/U2Fusion/IFCNN/SeAFusion/PIAFusion/LRRNet/DATFuse/DDFM + 传统 GTF/LP/DWT/NSCT）统一并入 leaderboard，做三任务平均排名综合对比；RFN-Nest 可作为"低伪影、温和、强源相关"一端的参照点。


## 指标修订（RGB-final 协议，2026-06-28）

> 修订动机：原先对比方法直接对融合的 **Y 通道图** 计分。参照仓库 `infer_fusion.py` 与 原始 MATLAB `evaluation/main.m` 的约定——**彩色源任务的最终融合图是 Y 与源 CbCr 重组逆变换得到的 RGB 图，计分时对该 RGB 图做 `rgb2gray`（= PIL 'L'，BT.601）**。RGB 逆变换中的 uint8 截断会在高饱和色区（PET/SPECT 伪彩、GFP 绿色）改变灰度，因此直接用 Y 计分不严格。
>
> 修订范围：`output_mode=rgb` 的 **medical / gfp_pc** 两任务，对全部 18 方法的融合 Y 重组源 CbCr → RGB-final → `rgb2gray` 重算（RGB-final 图存于 `fusion_bench/fused_final/<方法>/<任务>/`）。**irvis 为 `output_mode=gray`（与 MDFNet 自身评测一致），维持灰度不变。** 重算后排名与原结论基本一致（个别名次 ±1）。

修订后核心指标（medical/gfp_pc 已按 RGB-final 协议；irvis 灰度不变）：

**RFN-Nest**

| 任务 | n | EN | MI | SD | SF | AG | SSIM | MS_SSIM | Qabf | VIF | SCD | Nabf | CC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| irvis | 50 | 6.097 | 2.447 | 28.309 | 5.878 | 2.134 | 0.711 | 0.792 | 0.382 | 0.046 | 1.490 | 0.023 | 0.670 |
| medical | 48 | 5.604 | 2.785 | 69.438 | 8.809 | 3.737 | 0.643 | 0.759 | 0.181 | 0.041 | 1.417 | 0.027 | 0.883 |
| gfp_pc | 30 | 6.773 | 2.153 | 29.983 | 6.952 | 3.630 | 0.469 | 0.649 | 0.333 | 0.044 | 1.850 | 0.145 | 0.638 |
