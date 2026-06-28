# EXP-CMP-04：DenseFuse 对比方法复现（三模态指标）

- 日期：2026-06-25
- 所属阶段：阶段 5（综合对比 / SOTA 复现）
- 结果级别：S2（完整测试子集，三任务）
- 关联代码：`code/ref/DenseFuse-pytorch`（作者官方 PyTorch 端口，含预训练权重）；驱动 `code/Graduation-Paper/bench/run_densefuse.py`
- 评测器：`code/Graduation-Paper/bench/eval_method.py`（共享 `metrics/` 包）

## 0. 复现基础设施（所有对比方法共用）
统一基准管线见 `fusion_bench/BENCH_CONTRACT.md`：标准化 8-bit 灰度输入 `fusion_bench/inputs/<task>/{A,B}/<stem>.png`（A=可见/功能源，B=红外/结构源），统一输出 `fusion_bench/fused/<Method>/<task>/<stem>.png`，统一评测 `eval_method.py`，每方法独立 venv（`zzz_base.pth` 继承基础 venv 的 torch2.8），外网经 `proxy_env.sh` 代理、内网 PyPI 镜像直连。

## 1. 方法与权重来源（provenance）
- 论文：Li & Wu, *DenseFuse: A Fusion Approach to Infrared and Visible Images*, IEEE TIP 2019.
- **来源 = 官方 PyTorch 端口（非 re-impl，非训练）**：作者本人 `hli1221/densefuse-pytorch`（138★，与原 TF 仓库同一作者）。
  - 仓库克隆至 `code/ref/DenseFuse-pytorch`，commit `4394b63e9295db1c6b7a5c3664551c90f0605f2b`。
  - 预训练权重随仓库自带：`models/densefuse_gray.model`（灰度，MS-COCO 上以"像素 + SSIM 重建损失"训练的自编码器），`load_state_dict` 直接可载，参数量 74,193（约 0.07M），与论文一致（C1 + DenseBlock(3 层 dense conv) 编码器；4 层 conv 解码器）。
  - 未使用 RGB 权重（基准输入为灰度）。
- 端口适配说明：原仓库 `utils.py` 依赖已废弃的 `scipy.misc.imread/imsave` 与 `torch.utils.serialization.load_lua`，在 torch2.8/新版 scipy 下不可用。**只复用承载权重的 `net.py`（网络结构）与 `models/densefuse_gray.model`（state_dict）**，I/O 与融合用自写干净驱动 `bench/run_densefuse.py`（PIL + torch），不改动仓库网络定义。
- 输入尺度约定：灰度预训练模型在 **[0,255] 像素域**运算（与原仓库 `get_test_images` 的 `mode='L'` 一致，不做 /255 归一化），输出 `clamp(0,255)` 直接存 8-bit。

## 2. 环境与运行
- venv：`/ytech_m2v4_hdd/lizhongyin/venv/densefuse`
  ```
  /opt/conda/bin/python3.11 -m venv /ytech_m2v4_hdd/lizhongyin/venv/densefuse
  echo "/ytech_m2v4_hdd/lizhongyin/venv/lib/python3.11/site-packages" \
    > /ytech_m2v4_hdd/lizhongyin/venv/densefuse/lib/python3.11/site-packages/zzz_base.pth
  ```
  通过 `zzz_base.pth` 继承基础 venv 的 torch 2.8.0+cu128，无需额外 pip 安装（仅用 torch/numpy/PIL）。
- 推理：`export CUDA_VISIBLE_DEVICES=3`（单卡 H800），全部为推理无训练，三任务合计 128 张，单卡数秒完成。
- 融合策略：采用论文测试期 **L1-norm（soft，块内 l1 活动度 + 3×3 均值 + soft-max 权重）** 策略；驱动同时支持 `add`（(en1+en2)/2）。本次结果用 `--strategy l1`。
- 运行命令（在 `code/ref/DenseFuse-pytorch` 目录下，使 `import net` 可用）：
  ```
  export CUDA_VISIBLE_DEVICES=3
  PY=/ytech_m2v4_hdd/lizhongyin/venv/densefuse/bin/python
  for t in irvis medical gfp_pc; do
    $PY /ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper/bench/run_densefuse.py --task $t --strategy l1
  done
  ```
- 任务-槽位映射：DenseFuse 为对称自编码器（左右源等价），A、B 各自编码后特征融合再解码；A→VIS 槽，B→IR 槽与契约一致。

## 3. 结果（均值）

| 任务 | n | EN | MI | SD | SF | AG | SSIM | MS_SSIM | Qabf | VIF | SCD | Nabf | CC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| irvis   | 50 | 6.374 | 3.861 | 37.062 | 9.117 | 3.307 | 0.738 | 0.772 | 0.633 | 0.075 | 1.380 | 0.016 | 0.598 |
| medical | 48 | 5.113 | 3.691 | 67.757 | 24.901 | 9.288 | 0.742 | 0.756 | 0.662 | 0.101 | 0.896 | 0.006 | 0.832 |
| gfp_pc  | 30 | 6.555 | 3.429 | 24.984 | 9.073 | 4.414 | 0.546 | 0.572 | 0.636 | 0.115 | 0.490 | 0.042 | 0.397 |

功能轴：irvis FuncCorr 0.469 / FuncSal 0.774；medical 0.505 / 1.497；gfp_pc 0.262 / -0.427。
明细：`fusion_bench/reports/<task>/DenseFuse__{per_image,means}.csv`，并入各任务 `leaderboard.csv`。

## 4. 分析
- DenseFuse 是 2019 年轻量自编码器基线：**Nabf 极低**（irvis 0.016 / medical 0.006 / gfp_pc 0.042，远低于 CDDFuse），说明几乎不引入伪影、融合平滑；Qabf（边缘信息保真）三任务均 0.63–0.66，稳健。
- 但 **SD/SF/AG 等对比度与细节锐度指标明显弱于 CDDFuse**（如 irvis SD 37.1 vs CDDFuse 42.0；medical SD 67.8 vs 81.3），符合其"温和重建型融合、不做显式细节增强"的定位——作为弱-中强度的经典 CNN 对比锚点恰当。
- 医学任务表现最好（高 CC 0.832、低 Nabf）；GFP-PC 上 SSIM/CC 偏低（0.55/0.40、SCD 0.49），与该域分布特殊（GFP 大面积近黑背景、与 PC 结构差异大）一致，且 DenseFuse 仅在 MS-COCO 自然图像上自监督重建、未见显微域，跨域为零样本迁移。FuncSal 在 gfp_pc 为负，反映融合图功能显著性相对源功能图被弱化（平滑化所致）。

## 5. 问题与说明
- 原仓库 `utils.py`/`test_image.py` 依赖废弃 API（`scipy.misc`、`torch.utils.serialization.load_lua`），直接跑不通；已绕开，仅复用 `net.py` + state_dict，融合/IO 自写，确保与契约一致且可复现。
- 未训练：作者预训练灰度权重可用且匹配论文，按契约 §6 优先用预训练，无需在本地数据上重训。

## 6. 下一步
- 与其余对比方法（CDDFuse 已完成、SwinFusion/U2Fusion/IFCNN/SeAFusion/PIAFusion/RFN-Nest/LRRNet/DATFuse/DDFM + 传统 GTF/LP/DWT/NSCT）统一并入 leaderboard，做三任务平均排名综合对比；DenseFuse 可作为"低伪影、温和"一端的参照点。


## 指标修订（RGB-final 协议，2026-06-28）

> 修订动机：原先对比方法直接对融合的 **Y 通道图** 计分。参照仓库 `infer_fusion.py` 与 原始 MATLAB `evaluation/main.m` 的约定——**彩色源任务的最终融合图是 Y 与源 CbCr 重组逆变换得到的 RGB 图，计分时对该 RGB 图做 `rgb2gray`（= PIL 'L'，BT.601）**。RGB 逆变换中的 uint8 截断会在高饱和色区（PET/SPECT 伪彩、GFP 绿色）改变灰度，因此直接用 Y 计分不严格。
>
> 修订范围：`output_mode=rgb` 的 **medical / gfp_pc** 两任务，对全部 18 方法的融合 Y 重组源 CbCr → RGB-final → `rgb2gray` 重算（RGB-final 图存于 `fusion_bench/fused_final/<方法>/<任务>/`）。**irvis 为 `output_mode=gray`（与 MDFNet 自身评测一致），维持灰度不变。** 重算后排名与原结论基本一致（个别名次 ±1）。

修订后核心指标（medical/gfp_pc 已按 RGB-final 协议；irvis 灰度不变）：

**DenseFuse**

| 任务 | n | EN | MI | SD | SF | AG | SSIM | MS_SSIM | Qabf | VIF | SCD | Nabf | CC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| irvis | 50 | 6.374 | 3.861 | 37.062 | 9.117 | 3.307 | 0.738 | 0.772 | 0.633 | 0.075 | 1.380 | 0.016 | 0.598 |
| medical | 48 | 5.131 | 3.708 | 67.492 | 24.813 | 9.232 | 0.741 | 0.756 | 0.655 | 0.101 | 0.869 | 0.006 | 0.832 |
| gfp_pc | 30 | 6.559 | 3.360 | 25.010 | 9.157 | 4.469 | 0.545 | 0.573 | 0.630 | 0.111 | 0.512 | 0.049 | 0.397 |
