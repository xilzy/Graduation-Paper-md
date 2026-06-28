# EXP-CMP-05：IFCNN 对比方法复现（三模态指标）

- 日期：2026-06-25
- 所属阶段：阶段 5（综合对比 / SOTA 复现）
- 结果级别：S2（完整测试子集，三任务）
- 关联代码：`code/ref/IFCNN`（作者官方仓库，含预训练权重 `Code/snapshots/IFCNN-MAX.pth`）；驱动 `code/ref/IFCNN/Code/run_bench_ifcnn.py`
- 评测器：`code/Graduation-Paper/bench/eval_method.py`（共享 `metrics/` 包）

## 0. 复现基础设施（所有对比方法共用）
统一基准管线见 `fusion_bench/BENCH_CONTRACT.md`：
- **标准化输入**：三任务测试对统一导出为 8-bit 灰度 `fusion_bench/inputs/<task>/{A,B}/<stem>.png`。A=彩色/功能源（的 Y），B=灰度/结构源。
  - `gfp_pc`（30 对，GFP-Y / PC）、`irvis`（50 对，MSRS 子集，VIS-Y / IR）、`medical`（48 对，Harvard PET/SPECT-Y / MRI）。
- **统一输出契约**：每方法把融合图（灰度）按 stem 命名写到 `fusion_bench/fused/IFCNN/<task>/`。
- **统一评测**：`eval_method.py` 用共享 `metrics/` 计算 EN/MI/SD/SF/AG/SSIM/MS_SSIM/Qabf/VIF + 诊断 SCD/Nabf/CC + 功能轴 FuncCorr/FuncSal。
- **环境隔离**：方法独立 venv 于 `/ytech_m2v4_hdd/lizhongyin/venv/ifcnn`，通过 `zzz_base.pth` 链接基础 venv 的 torch2.8（免重装）。不触碰系统 python。
- **网络**：外网（GitHub）经 `proxy_env.sh` 代理；内网 PyPI 镜像直连。

## 1. 方法与权重来源
- 论文：Zhang et al., *IFCNN: A General Image Fusion Framework Based on Convolutional Neural Network*, Information Fusion 54 (2020) 99–118.
- 仓库：作者官方 `uzeful/IFCNN`，clone 到 `code/ref/IFCNN`，commit `7dd4a2aa8b6bea7e5d8b14c2a2dfb0645c12cab5`（"Update IV_filenames and datasets_num values"）。
- 模型定义：`Code/model.py`（`IFCNN` 类）。结构：conv1 取自 ResNet101 首层（stride=1, padding=0，权重冻结）→ conv2（ConvBlock）→ 特征融合（element-wise MAX）→ conv3 → conv4（1×1 → 3 通道重建）。
- 权重（仓库内自带，无需下载）：`Code/snapshots/IFCNN-MAX.pth`（通用模型，作者推荐用于 multi-focus / IR-VIS / 医学融合）。该 checkpoint 含 `conv1.weight`，会覆盖 ResNet101 首层，**因此构建模型时无需联网下载 ResNet101 预训练权重**（用 `weights=None` 仅取架构）。
- 任务-权重映射：三任务全部使用 **IFCNN-MAX**（通用变体），符合任务要求。

## 2. 环境与运行
### 2.1 venv 搭建
```bash
cd /ytech_m2v4_hdd/lizhongyin/venv
/opt/conda/bin/python3.11 -m venv ifcnn --without-pip
# base.pth 继承基础 venv 的 torch2.8 / torchvision0.23 / numpy / PIL / cv2，无需任何 pip 安装
echo "/ytech_m2v4_hdd/lizhongyin/venv/lib/python3.11/site-packages" \
  > ifcnn/lib/python3.11/site-packages/zzz_base.pth
```
依赖全部来自基础 venv：torch 2.8.0+cu128、torchvision 0.23.0+cu128、opencv 4.13.0、numpy、Pillow。**无私有 pip 安装**。

### 2.2 仓库克隆
```bash
source /ytech_m2v4_hdd/lizhongyin/proxy_env.sh
cd /ytech_m2v4_hdd/lizhongyin/code/ref
git clone https://github.com/uzeful/IFCNN
```

### 2.3 推理运行
```bash
cd /ytech_m2v4_hdd/lizhongyin/code/ref/IFCNN/Code
export CUDA_VISIBLE_DEVICES=2
/ytech_m2v4_hdd/lizhongyin/venv/ifcnn/bin/python run_bench_ifcnn.py
```
- GPU：`CUDA_VISIBLE_DEVICES=2`，纯推理。三任务合计 128 张，单卡约 3.6s。
- 预处理严格对齐作者 demo（`Code/IFCNN_Main.py` 的 IV/medical 灰度分支）：
  1. `PIL.Image.open(stem.png).convert('RGB')` — 8-bit 灰度自动复制为 3 通道；
  2. `ToTensor` + `Normalize(mean=[0,0,0], std=[1,1,1])`（即仅 /255，**不做 ImageNet 归一化**，与作者灰度分支一致；ImageNet mean/std 仅用于其彩色 multi-focus 分支）；
  3. `model(imgA, imgB)`，`fuse_scheme=0`，特征逐元素 MAX 融合；
  4. `denorm + clamp(0,1)*255 → uint8 RGB → cv2.COLOR_RGB2GRAY → 单通道灰度 PNG`。
- 喂入顺序 (A, B)；MAX 融合对称，顺序无关。
- 输出：`fusion_bench/fused/IFCNN/<task>/<stem>.png`。

## 3. 结果（均值）

| 任务 | n | EN | MI | SD | SF | AG | SSIM | MS_SSIM | Qabf | VIF | SCD | Nabf | CC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| irvis   | 50 | 6.336 | 2.876 | 34.088 | 11.535 | 4.443 | 0.722 | 0.783 | 0.617 | 0.054 | 1.526 | 0.132 | 0.629 |
| medical | 48 | 5.126 | 2.867 | 68.028 | 29.473 | 11.516 | 0.730 | 0.768 | 0.675 | 0.068 | 1.109 | 0.092 | 0.851 |
| gfp_pc  | 30 | 6.373 | 2.295 | 22.143 | 12.982 | 6.399 | 0.514 | 0.671 | 0.574 | 0.056 | 1.561 | 0.197 | 0.557 |

功能轴：irvis FuncCorr 0.461 / FuncSal 0.899；medical 0.426 / 1.612；gfp_pc 0.325 / 0.499。
明细：`fusion_bench/reports/<task>/IFCNN__{per_image,means}.csv`，并入各任务 `leaderboard.csv`。

## 4. 遇到的问题
- **ResNet101 联网下载**：`myIFCNN()` 默认 `models.resnet101(pretrained=True)` 会触发权重下载。但 IFCNN-MAX.pth 自带 `conv1.weight`，加载后即覆盖 conv1，故改用 `models.resnet101(weights=None)` 仅取架构，**避免联网**、且不影响结果。
- **灰度归一化口径**：作者 demo 对彩色 multi-focus 用 ImageNet mean/std，对 IR-VIS / 医学灰度用 mean=0/std=1。本基准输入均为 8-bit 灰度，统一采用灰度分支口径（mean=0/std=1），与作者一致。
- **3 通道还原**：`convert('RGB')` 将灰度复制为 3 等同通道，模型输出 3 通道再经 `RGB2GRAY` 回灰度，过程无信息损失。

## 5. 分析与结论
- IFCNN 是轻量通用融合框架（仅 4 个卷积层、单一通用权重），三任务均跑通且速度极快。
- IR-VIS、医学上指标稳健（SSIM≈0.72–0.73、Qabf 0.62–0.68、Nabf 较低 0.09–0.13），作为"通用网络"对比锚点表现合理，但 MI/VIF 等信息保真轴弱于 CDDFuse（如 medical MI 2.87 vs CDDFuse 3.61），符合 IFCNN 早期通用模型 vs 后续专精 SOTA 的预期差距。
- GFP-PC 跨域：SSIM 0.514、Nabf 0.197 偏高、MI 最低（2.29），说明显微域（GFP 大面积近黑背景）对该通用模型也存在分布差异，为本课题"通用 vs 专精/跨域泛化"动机提供又一对照证据。
- 评测尺子自洽：MAX 融合倾向保留双源高响应，SF/AG（细节/梯度）在 medical 上较高（29.47/11.52），符合 MAX 策略特性。


## 指标修订（RGB-final 协议，2026-06-28）

> 修订动机：原先对比方法直接对融合的 **Y 通道图** 计分。参照仓库 `infer_fusion.py` 与 原始 MATLAB `evaluation/main.m` 的约定——**彩色源任务的最终融合图是 Y 与源 CbCr 重组逆变换得到的 RGB 图，计分时对该 RGB 图做 `rgb2gray`（= PIL 'L'，BT.601）**。RGB 逆变换中的 uint8 截断会在高饱和色区（PET/SPECT 伪彩、GFP 绿色）改变灰度，因此直接用 Y 计分不严格。
>
> 修订范围：`output_mode=rgb` 的 **medical / gfp_pc** 两任务，对全部 18 方法的融合 Y 重组源 CbCr → RGB-final → `rgb2gray` 重算（RGB-final 图存于 `fusion_bench/fused_final/<方法>/<任务>/`）。**irvis 为 `output_mode=gray`（与 MDFNet 自身评测一致），维持灰度不变。** 重算后排名与原结论基本一致（个别名次 ±1）。

修订后核心指标（medical/gfp_pc 已按 RGB-final 协议；irvis 灰度不变）：

**IFCNN**

| 任务 | n | EN | MI | SD | SF | AG | SSIM | MS_SSIM | Qabf | VIF | SCD | Nabf | CC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| irvis | 50 | 6.336 | 2.876 | 34.088 | 11.535 | 4.443 | 0.722 | 0.783 | 0.617 | 0.054 | 1.526 | 0.132 | 0.629 |
| medical | 48 | 5.183 | 2.948 | 67.529 | 29.071 | 11.284 | 0.731 | 0.767 | 0.671 | 0.068 | 1.078 | 0.083 | 0.851 |
| gfp_pc | 30 | 6.373 | 2.266 | 22.112 | 12.990 | 6.440 | 0.513 | 0.671 | 0.567 | 0.055 | 1.555 | 0.200 | 0.558 |
