# EXP-CMP-03：U2Fusion 对比方法复现（三模态指标）

- 日期：2026-06-25
- 所属阶段：阶段 5（综合对比 / SOTA 复现）
- 结果级别：S2（完整测试子集，三任务）
- 关联代码：`code/ref/U2Fusion`（作者官方 TF1.x 仓库 + 预训练 ckpt）；自实现 PyTorch 端口 `code/ref/U2Fusion-pytorch`（`u2fusion_torch.py` + 由作者权重转换的 `u2fusion.pth`）
- 评测器：`code/Graduation-Paper/bench/eval_method.py`（共享 `metrics/` 包）

## 0. 复现基础设施（所有对比方法共用）
统一基准管线见 `fusion_bench/BENCH_CONTRACT.md`：三任务测试对统一导出为 8-bit 灰度
`fusion_bench/inputs/<task>/{A,B}/<stem>.png`（A=彩色/功能源，B=灰度/结构源），
融合图按 stem 写到 `fusion_bench/fused/U2Fusion/<task>/`，由 `eval_method.py` 统一计算
EN/MI/SD/SF/AG/SSIM/MS_SSIM/Qabf/VIF + 诊断 SCD/Nabf/CC + 功能轴 FuncCorr/FuncSal。
- `gfp_pc`（30 对，GFP-Y / PC）、`irvis`（MSRS 子集 50 对，VIS-Y / IR）、`medical`（Harvard 48 对，PET/SPECT-Y / MRI）。
- 环境隔离：方法独立 venv `/ytech_m2v4_hdd/lizhongyin/venv/u2fusion`，经 `zzz_base.pth` 继承基础 venv 的 torch2.8（cu128），方法私有依赖另装。
- 网络：外网（GitHub/HF/PyTorch/Drive）经 `proxy_env.sh` 代理；内网 PyPI 镜像直连。

## 1. 方法与权重来源
- 论文：Xu, Ma, Jiang, Guo, Ling, *U2Fusion: A Unified Unsupervised Image Fusion Network*, IEEE TPAMI 2020。
- 仓库：作者官方 `hanna-xu/U2Fusion`（TensorFlow 1.x），已在 `code/ref/U2Fusion`。
  自带统一预训练权重 `model/model.ckpt`（TF v1 checkpoint，单一统一模型，覆盖 VIS-IR/医学/多曝光/多聚焦）。
- **权重来源 = 作者原始权重**（非第三方端口、非自训练）。
  原 TF1.x 在 CUDA12 / H800 上无法运行；亦无可用的现成 PyTorch 预训练端口。
  采取的方案：**自实现一份与作者 `Net.py` 逐层等价的 PyTorch DenseNet 生成器**，并把作者 TF ckpt 的卷积核/偏置
  （核 `[kh,kw,in,out]` → torch `[out,in,kh,kw]` 转置）转入该端口（`convert_weights.py`，20 个张量全部一一对应）。
  这样推理跑在 PyTorch/GPU 上，但权重是作者的，不是重训。
- **数值忠实性验证**：在 TF2.21 eager 下按 `Net.py` 逐层复刻前向（reflect pad / VALID conv /
  `max(x,0.2x)` LReLU / dense concat / 末层 `tanh/2+0.5`），与 PyTorch 端口在同一真实样本上对比，
  **max abs diff = 3.6e-7，mean abs diff = 5.9e-8**（纯浮点舍入）——证明端口与作者前向完全一致。
- 任务-权重映射：U2Fusion 是统一无监督模型，单套权重通吃。源映射按作者 VIS-IR 约定
  **A→SOURCE1（vis 槽）、B→SOURCE2（ir 槽）**，对 medical/gfp_pc 一致沿用。

## 2. 环境与运行
- 建 venv：
  ```
  /opt/conda/bin/python3.11 -m venv /ytech_m2v4_hdd/lizhongyin/venv/u2fusion
  echo "/ytech_m2v4_hdd/lizhongyin/venv/lib/python3.11/site-packages" \
      > /ytech_m2v4_hdd/lizhongyin/venv/u2fusion/lib/python3.11/site-packages/zzz_base.pth   # 继承 torch2.8
  source /ytech_m2v4_hdd/lizhongyin/proxy_env.sh
  /ytech_m2v4_hdd/lizhongyin/venv/u2fusion/bin/pip install --default-timeout=300 tensorflow-cpu  # 仅用于读 ckpt
  ```
  （torch/numpy/PIL 由基础 venv 提供；tensorflow-cpu 2.21 只用于一次性导出权重，推理不依赖 TF。）
- 转权重（一次性）：`TF_CPP_MIN_LOG_LEVEL=3 .../u2fusion/bin/python convert_weights.py`
  → 产出 `code/ref/U2Fusion-pytorch/u2fusion.pth`。
- 推理：`export CUDA_VISIBLE_DEVICES=1`，对 `irvis/medical/gfp_pc` 各跑
  `.../u2fusion/bin/python infer.py <task>`。预处理：灰度 /255 归一到 [0,1]，
  `cat([A,B])` 送统一网络，输出 [0,1] → ×255 存灰度 PNG。
- 全部为推理（无训练），三任务合计 128 张，单卡 GPU 1 上约 5s。

## 3. 结果（均值）

| 任务 | n | EN | MI | SD | SF | AG | SSIM | MS_SSIM | Qabf | VIF | SCD | Nabf | CC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| irvis   | 50 | 4.882 | 1.961 | 19.495 | 6.846 | 2.421 | 0.689 | 0.782 | 0.328 | 0.038 | 1.053 | 0.028 | 0.654 |
| medical | 48 | 5.119 | 2.688 | 49.013 | 17.257 | 6.809 | 0.301 | 0.739 | 0.438 | 0.052 | 0.315 | 0.036 | 0.873 |
| gfp_pc  | 30 | 5.738 | 2.305 | 16.576 | 7.493 | 3.540 | 0.451 | 0.691 | 0.382 | 0.048 | 1.491 | 0.058 | 0.634 |

功能轴：irvis FuncCorr 0.509 / FuncSal 1.186；medical 0.327 / 1.937；gfp_pc 0.542 / 1.484。
明细：`fusion_bench/reports/<task>/U2Fusion__{per_image,means}.csv`，并入各任务 `leaderboard.csv`。

## 4. 分析
- U2Fusion 作为 2020 年的**统一无监督**基线，整体走"信息保真、低伪影"路线：三任务 **Nabf 极低**
  （0.028–0.058）、CC 偏高，融合干净不引入额外噪声边缘；但对比/锐度类指标（SD/SF/AG/Qabf）明显弱于
  CDDFuse（CVPR'23）。这正符合"通用统一网络 vs 专精强网络"的代际差距，是有价值的对比锚点。
- irvis 上 SD 仅 19.5、EN 4.88，整图偏暗低对比——U2Fusion 在 MSRS（大量夜景、IR 主导）上的已知特性：
  自适应权重倾向保留更"平"的可见光强度分布，导致暗区被压。这与 CDDFuse 的高 SD 形成鲜明对照。
- medical 上 SSIM 仅 0.30（但 MS_SSIM 0.74 正常）：PET/SPECT 与 MRI 强度差异大，统一网络对单源结构 SSIM
  牺牲较多；SD/SF/AG 反而是三任务最高，说明它在医学上把 MRI 高频结构注入得较充分。
- gfp_pc 跨域（无专用权重，统一模型直接迁移）：指标居中、Nabf 仍低，SCD 1.49 较高（融合保留了两源互补内容），
  作为显微域的"通用网络迁移"对照，证据自洽。

## 5. 复现要点 / 踩坑
- 原 TF1.x 仓库在 CUDA12/H800 上不可运行（无 TF1 wheel），但 ckpt 完好。回避方式：不跑 TF1，
  改为「PyTorch 逐层等价端口 + 转入作者 ckpt 权重」，并用 TF2 eager 前向做数值对拍（diff 3.6e-7）确认忠实。
- tensorflow-cpu 2.21 wheel（~数百 MB）在内网镜像首拉超时，需 `--default-timeout=300` 重试方成功；
  TF 仅用于一次性读取 ckpt，推理与评测均不依赖 TF。
- 端口关键对齐点：reflect pad=1 + VALID conv 等价 same 尺寸；LReLU 用 `max(x,0.2x)`；
  dense 块 concat 顺序为 `[新特征, 累积]`（与 TF `concat([out,x],3)` 一致）；末层 `tanh/2+0.5`。

## 6. 下一步
- 已写入三任务 leaderboard，可与 CDDFuse 等一并做平均排名综合对比。
- 后续 SwinFusion / IFCNN / DenseFuse / SeAFusion / TarDAL / PIAFusion / RFN-Nest / DDFM 等按同一契约并行复现。
