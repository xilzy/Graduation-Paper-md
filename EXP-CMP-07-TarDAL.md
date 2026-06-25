# EXP-CMP-07：TarDAL 对比方法复现（三模态指标）

- 日期：2026-06-25
- 所属阶段：阶段 5（综合对比 / SOTA 复现）
- 结果级别：S2（完整测试子集，三任务）
- 关联代码：`code/ref/TarDAL`（作者官方仓库，commit `6a9edd744b44fc03344fe8fb0fd930f5df47b00b`）；驱动脚本 `code/ref/TarDAL/run_tardal_bench.py`（本任务新增的最小前向脚本）
- 评测器：`code/Graduation-Paper/bench/eval_method.py`（共享 `metrics/` 包）

## 0. 复现基础设施（所有对比方法共用）
遵循统一基准管线（详见 `fusion_bench/BENCH_CONTRACT.md`）：
- **标准化输入**：三任务测试对统一为 8-bit 灰度 `fusion_bench/inputs/<task>/{A,B}/<stem>.png`。A=彩色/功能源，B=灰度/结构源。
  - `gfp_pc`（30 对，GFP-Y / PC）、`irvis`（MSRS 50 对，VIS-Y / IR）、`medical`（Harvard 48 对，PET/SPECT-Y / MRI）。
- **统一输出契约**：融合图按 stem 命名写到 `fusion_bench/fused/TarDAL/<task>/`。
- **统一评测**：`eval_method.py` 用共享 `metrics/` 计算 EN/MI/SD/SF/AG/SSIM/MS_SSIM/Qabf/VIF + 诊断 SCD/Nabf/CC + 功能轴 FuncCorr/FuncSal。
- **环境隔离**：独立 venv `/ytech_m2v4_hdd/lizhongyin/venv/tardal`，通过 `zzz_base.pth` 链接基础 venv 的 torch2.8（免重装），方法私有依赖私装。不触碰系统 python。
- **GPU**：全程仅用 `CUDA_VISIBLE_DEVICES=5`，纯推理。

## 1. 方法与权重来源
- 论文：Liu et al., *Target-aware Dual Adversarial Learning and a Multi-scenario Multi-Modality Benchmark to Fuse Infrared and Visible for Object Detection*, CVPR 2022.
- 仓库：作者官方 `JinyuanLiu-CV/TarDAL`，克隆至 `code/ref/TarDAL`，commit `6a9edd74`。
- 融合生成器：`module/fuse/generator.py` 的 `Generator(dim=32, depth=3)`，结构为 DenseNet 风格编码 + 融合解码：
  - 输入 `cat([ir, vi], dim=1)`（2 通道，IR + VIS），输出 1 通道 + `Tanh`。
- **采用权重变体：TarDAL-DT（`tardal-dt.pth`）** —— 纯融合"直接训练"(Direct-Train) 变体，是 TarDAL 论文中报告融合质量的标准权重（不含检测分支耦合）。
  - 来源：GitHub Release v1.0.0
    `https://github.com/JinyuanLiu-CV/TarDAL/releases/download/v1.0.0/tardal-dt.pth`（经 `proxy_env.sh` 代理下载，~1.2 MB）。
  - 同时下载了 `tardal-tt.pth` / `tardal-ct.pth` 备用，均能无 missing/unexpected key 加载（46 keys 完整匹配）。
  - 关键细节：`tardal-dt.pth` 的 checkpoint 内含 `use_eval=False` 标记（v0 风格权重），其 BatchNorm 在推理时应使用 **train 模式（batch 统计）** 而非 eval（running stats），否则输出退化。本复现严格按官方 `pipeline/fuse.py` 的 `use_eval` 逻辑处理：DT 走 `net.train()`，tt/ct 走 `net.eval()`。
- 任务-权重映射：三任务**统一使用 TarDAL-DT**。TarDAL 角色固定 IR/VIS，按契约 **B→IR 槽，A→VIS 槽**。

## 2. 环境与运行
环境搭建命令：
```bash
/opt/conda/bin/python3.11 -m venv /ytech_m2v4_hdd/lizhongyin/venv/tardal
echo "/ytech_m2v4_hdd/lizhongyin/venv/lib/python3.11/site-packages" \
  > /ytech_m2v4_hdd/lizhongyin/venv/tardal/lib/python3.11/site-packages/zzz_base.pth   # 继承 torch2.8
/ytech_m2v4_hdd/lizhongyin/venv/tardal/bin/pip install kornia opencv-python-headless pyyaml  # 内网镜像
```
克隆 + 权重下载（经代理）：
```bash
source /ytech_m2v4_hdd/lizhongyin/proxy_env.sh
git clone https://github.com/JinyuanLiu-CV/TarDAL code/ref/TarDAL
mkdir -p code/ref/TarDAL/weights/v1
curl -sL -o code/ref/TarDAL/weights/v1/tardal-dt.pth \
  https://github.com/JinyuanLiu-CV/TarDAL/releases/download/v1.0.0/tardal-dt.pth
```
推理（最小前向，复刻官方预处理/保存约定）：
```bash
export CUDA_VISIBLE_DEVICES=5
cd code/ref/TarDAL
/ytech_m2v4_hdd/lizhongyin/venv/tardal/bin/python run_tardal_bench.py
```
- 预处理：`cv2.IMREAD_GRAYSCALE` 读入 → `/255` → `(1,1,H,W)` float，与官方 `loader/utils/reader.py:gray_read` 一致；TarDAL 为全卷积网络，任意尺寸直推、无需 padding/resize。
- 前向：`fus = Generator(ir=B, vi=A)`；Tanh 输出按官方 `img_write` 约定 `clamp(0,1)*255` 存 uint8（负值天然截断为 0）。
- 三任务合计 128 张，单卡总耗时约 3.5s（irvis 2.3s / medical 0.6s / gfp_pc 0.6s）。全部为推理，无训练。

## 3. 结果（均值）

| 任务 | n | EN | MI | SD | SF | AG | SSIM | MS_SSIM | Qabf | VIF | SCD | Nabf | CC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| irvis   | 50 | 6.378 | 2.601 | 36.202 | 10.193 | 4.038 | 0.495 | 0.727 | 0.423 | 0.041 | 1.540 | 0.176 | 0.637 |
| medical | 48 | 5.817 | 2.927 | 62.679 | 21.789 | 8.010 | 0.261 | 0.686 | 0.431 | 0.047 | 0.718 | 0.054 | 0.837 |
| gfp_pc  | 30 | 7.176 | 2.628 | 40.515 | 14.490 | 6.437 | 0.476 | 0.564 | 0.447 | 0.049 | 1.561 | 0.293 | 0.571 |

功能轴：irvis FuncCorr 0.604 / FuncSal 1.123；medical 0.814 / 1.661；gfp_pc 0.692 / 0.935。
明细：`fusion_bench/reports/<task>/TarDAL__{per_image,means}.csv`，并入各任务 `leaderboard.csv`。

## 4. 分析
- TarDAL-DT 在三任务上均给出结构合理、动态范围充分的融合图（fused min/mean/max 覆盖 0–255，std 26–64），无塌黑/饱和退化。
- 相对 CDDFuse（EXP-CMP-01）：TarDAL 的 SD/SF/AG 等"细节锐度"轴在 irvis/medical 上偏低，SSIM/VIF 也较低——符合 TarDAL 以"目标显著性 + 对抗细节"为导向、而非以结构相似度为最优化目标的特性；其低 Nabf（medical 0.054）说明伪影抑制良好。
- medical 上 SSIM 偏低（0.261）：TarDAL 用 IR/VIS 域权重跨域到 PET-MRI，融合更偏向 MRI(B/IR 槽)结构、对 PET(A/VIS 槽)的低频功能信息保留弱，CC 0.837 但 SSIM 低，体现跨域迁移的偏置。
- gfp_pc：Nabf 偏高（0.293）、SSIM 中等，显微域（GFP 大面积近黑背景）与 IR/VIS 自然图像分布差异显著，为本课题"通用 vs 专精"动机提供又一组对照证据。

## 5. 问题与备注
- `weights/` 目录在仓库内为空，权重需从 GitHub Release v1.0.0 手动拉取（代理可达；GitHub API `/releases` 列表经代理返回空，但直接拼 release 下载 URL 可用）。
- 官方完整 `infer.py` 管线绑定 M3FD dataloader / wandb / yolo 检测分支，配置 plumbing 繁琐；按契约允许，本复现直接将 generator 权重载入网络做最小前向，完全复刻官方预处理与输出约定（gray_read 归一化、Tanh→clamp→uint8 保存），结果可比。
- `use_eval=False`（DT/v0 权重）这一隐藏约定若忽略会导致 BatchNorm 用错统计、输出退化——已正确处理。

## 6. 下一步
- 其余对比方法按同一契约复现，统一进 leaderboard 后做平均排名综合对比。
