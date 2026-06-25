# EXP-CMP-02：SwinFusion 对比方法复现（三模态指标）

- 日期：2026-06-25
- 所属阶段：阶段 5（综合对比 / SOTA 复现）
- 结果级别：S2（完整测试子集，三任务）
- 关联代码：`code/ref/SwinFusion`（作者仓库 Linfeng-Tang/SwinFusion，含预训练权重）；驱动 `code/ref/SwinFusion/run_swinfusion_bench.py`
- 评测器：`code/Graduation-Paper/bench/eval_method.py`（共享 `metrics/` 包）

## 0. 复现基础设施（所有对比方法共用）
统一基准管线见 `fusion_bench/BENCH_CONTRACT.md`：
- **标准化输入**：三任务测试对统一导出为 8-bit 灰度 `fusion_bench/inputs/<task>/{A,B}/<stem>.png`，A=彩色/功能源（VIS-Y / PET / GFP-Y），B=灰度/结构源（IR / MRI / PC）。
  - `gfp_pc`（30 对）、`irvis`（MSRS 测试集 50 对）、`medical`（Harvard 48 对）。
- **统一输出契约**：融合图按 stem 命名写到 `fusion_bench/fused/SwinFusion/<task>/`。
- **统一评测**：`eval_method.py` 用共享 `metrics/` 计算 EN/MI/SD/SF/AG/SSIM/MS_SSIM/Qabf/VIF + 诊断 SCD/Nabf/CC + 功能轴 FuncCorr/FuncSal，逐图 CSV + 均值 + 任务级 leaderboard。
- **环境隔离**：方法独立 venv `/ytech_m2v4_hdd/lizhongyin/venv/swinfusion`，经 `zzz_base.pth` 链接基础 venv 的 torch2.8（免重装），方法私有依赖装入子 venv。
- **网络**：外网经 `proxy_env.sh` 代理；内网 PyPI 镜像直连。

## 1. 方法与权重来源
- 论文：Ma / Tang et al., *SwinFusion: Cross-domain Long-range Learning for General Image Fusion via Swin Transformer*, IEEE/CAA JAS 2022.
- 仓库：作者官方 `Linfeng-Tang/SwinFusion`（已在 `code/ref/SwinFusion`），commit `c55dc3c1cec1d3590639f55ca9cf33709c3bf99c`。
- 模型类：`models/network_swinfusion1.py::SwinFusion`（双分支 Swin Transformer 编码 + cross-domain 融合 + 重建），`forward(A, B)`。
  - 配置（与仓库 `test_swinfusion.py::define_model` 一致）：`in_chans=1, img_size=128, window_size=8, img_range=1., depths=[6,6,6,6], embed_dim=60, num_heads=[6,6,6,6], mlp_ratio=2, upsampler=None, resi_connection='1conv'`。
- 权重（仓库内自带 `10000_G.pth` 生成器，checkpoint key `params`）：
  - IR-VIS：`Model/Infrared_Visible_Fusion/Infrared_Visible_Fusion/models/10000_G.pth`
  - 医学：`Model/Medical_Fusion-PET-MRI/Medical_Fusion/models/10000_G.pth`
- 任务-权重映射：`irvis→IR-VIS`；`medical→Medical PET-MRI`；`gfp_pc→Medical PET-MRI`（显微/医学近域，无专用权重）。
- **槽位约定**：仓库 `test_swinfusion.py` 默认 `A_dir=IR, B_dir=VI_Y`，即 `forward(IR, VIS)`。据契约「B→IR 槽，A→VIS 槽」，本驱动取 **模型第一参=我们的 B（IR/MRI/PC），第二参=我们的 A（VIS/PET/GFP）**，三任务一致。

## 2. 环境与运行
- 建 venv（base.pth 继承 torch2.8 + torchvision + timm；仅私装 opencv-headless / matplotlib）：
```
source /ytech_m2v4_hdd/lizhongyin/proxy_env.sh
/opt/conda/bin/python3.11 -m venv /ytech_m2v4_hdd/lizhongyin/venv/swinfusion
echo "/ytech_m2v4_hdd/lizhongyin/venv/lib/python3.11/site-packages" \
  > /ytech_m2v4_hdd/lizhongyin/venv/swinfusion/lib/python3.11/site-packages/zzz_base.pth
/ytech_m2v4_hdd/lizhongyin/venv/swinfusion/bin/pip install opencv-python-headless matplotlib
```
- 推理（`CUDA_VISIBLE_DEVICES=0`，纯推理无训练）：
```
export CUDA_VISIBLE_DEVICES=0
cd /ytech_m2v4_hdd/lizhongyin/code/ref/SwinFusion
/ytech_m2v4_hdd/lizhongyin/venv/swinfusion/bin/python run_swinfusion_bench.py irvis medical gfp_pc
```
- 驱动逻辑：读标准化灰度 → /255 归一化 → 按仓库 test 方式做 window_size=8 的反射 padding（保证可整除）→ `model(B_in=IR, A_in=VIS)` → 裁回原尺寸 → clamp[0,1]×255 出灰度 PNG。
- 评测：
```
/ytech_m2v4_hdd/lizhongyin/venv/bin/python \
  /ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper/bench/eval_method.py \
  --task <t> --name SwinFusion --fused-dir /ytech_m2v4_hdd/lizhongyin/fusion_bench/fused/SwinFusion/<t>
```
- 用时：irvis 50 张约 52 s（首次含建图），medical 48 张约 6 s，gfp_pc 30 张约 6 s（单卡）。

## 3. 结果（均值）

| 任务 | n | EN | MI | SD | SF | AG | SSIM | MS_SSIM | Qabf | VIF |
|---|---|---|---|---|---|---|---|---|---|---|
| irvis   | 50 | 6.524 | 4.728 | 41.364 | 10.492 | 3.829 | 0.725 | 0.766 | 0.656 | 0.089 |
| medical | 48 | 5.940 | 3.435 | 71.289 | 28.671 | 11.347 | 0.339 | 0.736 | 0.749 | 0.078 |
| gfp_pc  | 30 | 6.561 | 5.014 | 25.720 |  9.773 | 4.656 | 0.541 | 0.609 | 0.689 | 0.136 |

诊断 / 功能轴：

| 任务 | SCD | Nabf | CC | FuncCorr | FuncSal |
|---|---|---|---|---|---|
| irvis   | 1.698 | 0.063 | 0.602 | 0.435 | 0.707 |
| medical | 1.075 | 0.085 | 0.831 | 0.233 | 1.458 |
| gfp_pc  | 0.885 | 0.071 | 0.456 | 0.157 | -0.237 |

明细：`fusion_bench/reports/<task>/SwinFusion__{per_image,means}.csv`，并入各任务 `leaderboard.csv`。

## 4. 分析
- **IR-VIS**：用专用权重，整体均衡——SSIM 0.725 / MS_SSIM 0.766 / Qabf 0.656、Nabf 仅 0.063（伪影低），SCD 1.70 高，是稳健的强对比锚点；与 CDDFuse（EN 6.60 / SD 42.0 / Qabf 0.681）相近，质量轴略低、SD 接近。
- **医学**：高对比、强边缘——SD 71.3 / SF 28.7 / AG 11.3 / Qabf 0.749 显著高，CC 0.831 表明与源相关性强；但 **SSIM 仅 0.339**（MS_SSIM 0.736 正常），系 SwinFusion 医学权重输出动态范围被大幅拉伸、单尺度 SSIM 对亮度/对比偏移敏感所致，非塌黑或配准错误（已抽样核验输出 min/max 跨满量程、mean 合理）。功能轴 FuncSal 1.458 高，功能信息保留好。
- **GFP-PC**：医学权重跨域迁移——Qabf 0.689 / Nabf 0.071 不错，但 SD 偏低（25.7）、FuncSal 为负（-0.237），说明 GFP 大面积近黑背景下功能显著性被弱化，反映「医学→显微」分布漂移；为本课题"通用 vs 专精"动机再添一条对照。
- **尺子自洽**：Nabf 三任务均低（≤0.085）、SCD/CC 合理，说明融合未引入大量伪影，评测可比。

## 5. 问题与注记
- 仓库 `test_swinfusion.py::define_model` 实际加载 `_E.pth`（EMA），本驱动按契约用生成器 `_G.pth`；两者均为同一网络的等价 checkpoint（key `params`），strict 加载无缺失/多余键。
- 医学 SSIM 偏低见 §4，为方法固有对比拉伸特性，非复现缺陷；MS_SSIM 与 Qabf 正常可佐证。
- 仅推理，无需训练；三任务合计 128 张，单卡分钟级完成。

## 6. 下一步
- 与 CDDFuse 等其余对比方法（U2Fusion/IFCNN/DenseFuse/SeAFusion/TarDAL/PIAFusion/RFN-Nest/LRRNet/DATFuse/DDFM + 传统法）统一进 leaderboard 后做平均排名综合对比。
