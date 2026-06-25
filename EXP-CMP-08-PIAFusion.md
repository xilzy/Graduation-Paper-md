# EXP-CMP-08：PIAFusion 对比方法复现（三模态指标）

- 日期：2026-06-25
- 所属阶段：阶段 5（综合对比 / SOTA 复现）
- 结果级别：S2（完整测试子集，三任务）
- 关联代码：`code/ref/PIAFusion_pytorch`（非官方 PyTorch 移植，含预训练权重）；驱动 `code/Graduation-Paper/bench/run_piafusion.py`
- 评测器：`code/Graduation-Paper/bench/eval_method.py`（共享 `metrics/` 包）

## 0. 复现基础设施（所有对比方法共用）
统一基准管线见 `fusion_bench/BENCH_CONTRACT.md`：
- **标准化输入**：三任务测试对统一导出为 8-bit 灰度 `fusion_bench/inputs/<task>/{A,B}/<stem>.png`。A=彩色/功能源（取 Y），B=灰度/结构源。
  - `gfp_pc`（30 对，GFP-Y / PC）、`irvis`（MSRS 测试集 50 对，VIS-Y / IR）、`medical`（Harvard 48 对，PET/SPECT-Y / MRI）。
- **统一输出契约**：每方法把融合图按 stem 命名写到 `fusion_bench/fused/<Method>/<task>/`。
- **统一评测**：`eval_method.py` 计算 EN/MI/SD/SF/AG/SSIM/MS_SSIM/Qabf/VIF + 诊断 SCD/Nabf/CC + 功能轴 FuncCorr/FuncSal。
- **环境隔离**：独立 venv `/ytech_m2v4_hdd/lizhongyin/venv/piafusion`，通过 `zzz_base.pth` 链接基础 venv 的 torch2.8（免重装），方法私有依赖装入子 venv。不触碰系统 python。
- **网络**：外网（GitHub/HF/Drive）经 `proxy_env.sh` 代理；内网 PyPI 镜像直连。
- **GPU**：仅用 `CUDA_VISIBLE_DEVICES=6`。

## 1. 方法与权重来源
- 论文：Tang et al., *PIAFusion: A progressive infrared and visible image fusion network based on illumination aware*, Information Fusion 2022, vol. 83-84, pp. 79-92.
- **provenance：pytorch-port pretrained（非 re-train）**。作者官方仓库 `Linfeng-Tang/PIAFusion` 为 TensorFlow 实现，与 CUDA12/H800 不兼容，已放弃。
- 采用社区官方 PyTorch 移植 `linklist2/PIAFusion_pytorch`（README 自述 unofficial，但与论文 loss 对齐，作者协助调试）。
  - 仓库：`code/ref/PIAFusion_pytorch`，commit `dc1abc3371ee8f468b306a9ed3138187e39decc4`（"fix some bugs in readme"）。
  - 注意：作者预期 repo `Linfeng-Tang/PIAFusion_pytorch` **不存在**（GitHub API 返回 Repository not found），正确移植仓库为 `linklist2/PIAFusion_pytorch`。
- **权重直接随仓库提交**（无需额外下载）：
  - 照度感知融合网 `pretrained/fusion_model_epoch_29.pth`（4.7 MB，state_dict key：`encoder.* / decoder.*`）。
  - 照度分类子网 `pretrained/best_cls.pth`（测试用不到，仅训练时为 loss 提供照度权重）。
- 网络结构：双分支 Encoder（VIS/IR 各 5 层 conv + CMDAF 跨模态差分注意力）→ 通道拼接 → Decoder（5 层 conv，末层 Tanh/2+0.5 出 [0,1]）。输出单通道 Y。
- 任务-权重映射：三任务均用同一融合权重（仅 MSRS IR-VIS 训练得到，medical/gfp_pc 为跨域迁移）。

## 2. 环境与运行
- venv 创建：
  ```
  /opt/conda/bin/python3.11 -m venv /ytech_m2v4_hdd/lizhongyin/venv/piafusion
  echo /ytech_m2v4_hdd/lizhongyin/venv/lib/python3.11/site-packages \
    > /ytech_m2v4_hdd/lizhongyin/venv/piafusion/lib/python3.11/site-packages/zzz_base.pth
  ```
  继承 base venv 的 torch 2.8.0+cu128 / torchvision 0.23 / numpy / PIL / tqdm，**无需额外安装**（模型仅含 Conv/LeakyReLU/Tanh，torch2.8 完全兼容，绕过 requirements.txt 里的 torch1.9）。
- 推理：`export CUDA_VISIBLE_DEVICES=6`，驱动 `bench/run_piafusion.py`：
  - A（已是 8-bit 灰度的 VIS-Y）直接作为模型 Y 输入；B（IR/结构灰度）作为 IR 输入；两者归一化到 [0,1]。
  - `fused = clamp(model(vis_y, ir))`，输出 Y∈[0,1] → 直接存为灰度 PNG（契约允许灰度，评测器内部转灰）。
  - 因输入已是灰度，跳过移植仓库 dataloader 的 RGB→YCrCb 流程，避免对单通道源做错误色彩分离。
  ```
  for t in irvis medical gfp_pc; do
    /ytech_m2v4_hdd/lizhongyin/venv/piafusion/bin/python \
      code/Graduation-Paper/bench/run_piafusion.py --task $t
  done
  ```
- 全部为推理（无训练），三任务合计 128 张，单卡数秒级完成。

## 3. 结果（均值）

| 任务 | n | EN | MI | SD | SF | AG | SSIM | MS_SSIM | Qabf | VIF | SCD | Nabf | CC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| irvis   | 50 | 6.466 | 4.555 | 41.119 | 10.958 | 4.132 | 0.706 | 0.761 | 0.678 | 0.086 | 1.621 | 0.093 | 0.602 |
| medical | 48 | 5.657 | 3.077 | 74.801 | 27.127 | 10.847 | 0.360 | 0.748 | 0.692 | 0.065 | 1.474 | 0.087 | 0.856 |
| gfp_pc  | 30 | 6.562 | 4.087 | 25.039 | 10.767 | 5.214 | 0.535 | 0.645 | 0.676 | 0.094 | 1.612 | 0.096 | 0.524 |

功能轴：irvis FuncCorr 0.429 / FuncSal 0.755；medical 0.480 / 1.679；gfp_pc 0.364 / 0.239。
明细：`fusion_bench/reports/<task>/PIAFusion__{per_image,means}.csv`，并入各任务 `leaderboard.csv`。

## 4. 分析
- **IR-VIS（本征域）**：PIAFusion 表现稳健，Qabf 0.678 / SSIM 0.706 / MS_SSIM 0.761，与 CDDFuse 同档；SD/EN 略低于 CDDFuse，因其设计偏向保边缘梯度而非最大化对比度，Nabf 低（0.093）说明伪影少。
- **医学（跨域）**：SD/SF/AG 显著拉高（高频结构强），但 SSIM 仅 0.360——融合 Y 与 PET-Y 单源结构差异大（MRI 结构主导），且照度感知权重是按 MSRS 夜/昼场景训练，在医学域失配；FuncSal 高（1.679）说明功能成分被放大。
- **GFP-PC（跨域）**：各项中规中矩，Qabf 0.676 与本征域持平，但 SD 仅 25（GFP 大面积近黑背景压低全局对比），SSIM 0.535，体现显微域分布差异——为本课题"通用 vs 专精"动机提供对照。
- **跨域共性**：照度感知子网仅在 IR-VIS 训练，medical/gfp_pc 属迁移，metrics 自洽但非该方法强项；与 CDDFuse 相比 PIAFusion 在 irvis 上 MI 略低、Nabf 更低，整体可作为可见光-红外强对比锚点。

## 5. 问题与坑
1. **官方 repo 名错误**：任务给定 `Linfeng-Tang/PIAFusion_pytorch` 实际不存在；TF 官方仓库为 `Linfeng-Tang/PIAFusion`，PyTorch 移植在 `linklist2/PIAFusion_pytorch`。经 GitHub search API 定位后改用后者。
2. **git clone 通过代理需显式处理**：直接 clone 报 "could not read Username"，本质是仓库名错（Repository not found）；纠正仓库名后正常 clone。
3. **预训练权重无需外部下载**：移植仓库已把 `fusion_model_epoch_29.pth` / `best_cls.pth` 直接提交进 repo，省去百度网盘（被墙）下载。
4. **torch 版本**：requirements 指定 torch1.9+cu111，与 H800/CUDA12 不兼容；模型为纯卷积，直接用 base venv 的 torch2.8 推理无误（已 smoke-test 权重加载与前向）。
5. **灰度输入适配**：移植仓库测试流程假设 VIS 为 RGB 并做 RGB→YCrCb；本基准 A 已是灰度 Y，故驱动里直接喂单通道，跳过色彩转换。

## 6. 结论
PIAFusion（PyTorch 移植，作者预训练融合权重）已在三任务全部完成推理与统一评测：irvis 50 / medical 48 / gfp_pc 30，共 128 张。指标进入各任务 leaderboard，可作为 IR-VIS 域强对比锚点；医学/显微为跨域迁移，结果合理但非强项，支撑"通用 vs 专精"对比论证。

## 7. 下一步
- 其余对比方法按同一契约并行复现，统一进 leaderboard 后做平均排名综合对比。
