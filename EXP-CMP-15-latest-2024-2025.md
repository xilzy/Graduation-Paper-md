# EXP-CMP-15：2024–2025 最新方法复现（EMMA / GIFNet / TC-MoA）

- 日期：2026-07-06　评测协议同 `EVALUATION-metrics.md`（核心 9 + 诊断；irvis 灰度、medical/gfp_pc RGB-final）。
- 目标：为对比实验补充领域最新（2024–2025）通用/多模态融合方法，回答「对比方法最新只到 2023」的时效性短板。
- 数据：`fusion_bench/`（irvis 50 / medical 48 / gfp_pc 30），统一标准化输入 + 共享 `metrics/` 计分，与既有 18 方法严格可比。
- 结论：**EMMA、GIFNet 均复现成功，且在三模态 × 5 项判优指标（MI/SSIM/Qabf/VIF/Nabf）上被 Ours(v3) 全面压制**，已纳入 §4.2 对比表；**TC-MoA 因上游权重下架无法复现**。

---

## 1. 候选方法与选取

经网络检索（CVPR/ICCV/TPAMI 2024–2025 通用与多模态融合），优先选取「通用/多任务 + 有开源代码与预训练权重 + 可覆盖三模态」的方法：

| 方法 | 会议/年份 | 类型 | 结果 |
|---|---|---|---|
| EMMA | CVPR 2024 | 等变多模态融合（IR-VIS + 医学）| ✅ 复现成功 |
| GIFNet | CVPR 2025 | 任务无关通用融合（One Model for ALL）| ✅ 复现成功 |
| TC-MoA | CVPR 2024 | 任务定制混合适配器（MoE 统一融合）| ❌ 权重下架，无法复现 |

## 2. 复现设置（均在跳板机 H800，各占单卡）

统一遵循 `fusion_bench/BENCH_CONTRACT.md`：A→可见光/功能槽、B→红外/结构槽；输出灰度融合 Y 到 `fused/<M>/<task>`；medical/gfp_pc 经 `recombine_rescore.py` 重组 RGB-final 后再计分；一律用 base venv 的共享评测器，不自写指标。

### 2.1 EMMA（CVPR 2024）
- 仓库：`github.com/Zhaozixiang1228/MMIF-EMMA` @ `9983ca2`，`code/ref/MMIF-EMMA`。
- 权重：`model/EMMA.pth`（U-Fuser 融合模块，仓库自带）；Ai.pth/Av.pth 仅训练用，推理不需。
- 环境：venv `/ytech_m2v4_hdd/lizhongyin/venv/emma`（py3.11），torch 2.5.1+cu124（H800 可用；注意 mirror 默认 2.12/cu13 与驱动 CUDA12.9 不兼容，需固定 2.5.1）。
- 推理：自写 `emma_infer.py`（避开 test.py 的 Windows 路径假设），forward = `model(ir, vi)`，reflect-pad 到 32 倍数后裁回、min-max 归一化、存 'L' 灰度。GPU 1。
- 状态：irvis 50/50、medical 48/48、gfp_pc 30/30 全部完成。gfp_pc 为跨域应用（EMMA 训练面向 IR-VIS）。

### 2.2 GIFNet（CVPR 2025）
- 仓库：`github.com/AWCXV/GIFNet` @ `e9c83a6`，`code/ref/GIFNet`。
- 权重：仓库自带单一 checkpoint `model/Final.model`（3.4MB，state_dict）。
- 环境：venv `/ytech_m2v4_hdd/lizhongyin/venv/gifnet`（py3.11），torch 2.8.0+cu128（弃用仓库 cu111 旧 torch）。GPU 0。
- 推理：直接用 `test.py`，不加 `--*_IS_RGB`（输入已是灰度 Y），cv2 读写单通道；A→vis、B→ir，一套任务无关模型跑三任务。
- 状态：irvis 50/50、medical 48/48、gfp_pc 30/30 全部完成。medical/gfp_pc 为跨域（模型面向 IVIF/MFIF）。

### 2.3 TC-MoA（CVPR 2024）——权重下架，复现受阻
- 仓库：`github.com/YangSun22/TC-MoA` @ `48e4a54`，环境（venv `tcmoa`、torch 2.5.1+cu124、timm 0.3.2 打补丁 `torch._six`→`collections.abc`、`np.float`→`np.float64`）、MAE ViT-Large backbone（1.3GB，fbaipublicfiles 下载 OK）、standalone `predict_bench.py` **均已搭好并冒烟通过**。
- 阻塞点：TC-MoA 训练权重（Google Drive id `1S23P6Sw…`）**HTTP 404**，`/file/d/…/view` 页与训练数据文件夹亦 404（=私有/移除，非配额 403）；同一代理下 gdown 可正常下载 gdown 官方公开测试文件，证明非代理/配额问题；HuggingFace/ModelScope/GitHub Release **均无镜像**；仓库 issue #10/#16/#25（2026-01）为作者未回复的失效链接反馈。
- 结论：**仅缺权重导致无法产出融合图**。只加载 MAE backbone 会缺 258 个 TC-MoA 训练模块键（`blocks_MoA`×150 等），随机初始化的输出无意义，故未产出任何 `fused/TC-MoA` 图像（不伪造）。重训需同样已丢失的训练数据 + 训 MAE 大模型，代价过高，暂放弃。

## 3. 结果：三模态 × 5 指标（vs Ours v3）

约定 ↑ 越大越好，Nabf ↓ 越小越好；**加粗 = 该指标全场最优**。

**irvis 红外-可见光 (n=50)**

| 方法 | MI ↑ | SSIM ↑ | Qabf ↑ | VIF ↑ | Nabf ↓ |
|---|---|---|---|---|---|
| EMMA | 4.173 | 0.716 | 0.635 | 0.073 | 0.128 |
| GIFNet | 1.939 | 0.718 | 0.410 | 0.036 | 0.128 |
| **Ours (v3)** | **5.200** | **0.724** | **0.646** | **0.106** | **0.026** |

**medical 医学 (n=48)**

| 方法 | MI ↑ | SSIM ↑ | Qabf ↑ | VIF ↑ | Nabf ↓ |
|---|---|---|---|---|---|
| EMMA | 2.875 | 0.668 | 0.562 | 0.046 | 0.071 |
| GIFNet | 2.728 | 0.367 | 0.393 | 0.039 | 0.146 |
| **Ours (v3)** | **4.556** | **0.726** | **0.691** | **0.111** | **0.022** |

**gfp_pc 显微 (n=30)**

| 方法 | MI ↑ | SSIM ↑ | Qabf ↑ | VIF ↑ | Nabf ↓ |
|---|---|---|---|---|---|
| EMMA | 2.722 | 0.501 | 0.517 | 0.057 | 0.293 |
| GIFNet | 1.672 | 0.364 | 0.278 | 0.032 | 0.250 |
| **Ours (v3)** | **5.445** | **0.538** | **0.680** | **0.135** | **0.060** |

**判读**：EMMA、GIFNet 在三模态 × 5 指标共 30 个对比点上无一超过 Ours（irvis SSIM Ours 0.724 亦高于 GIFNet 0.718）。两者作为跨域/通用模型，在医学、显微域结构保真明显下降（GIFNet SSIM 掉到 0.36–0.37），佐证「无单一通用模型三模态通吃、需任务特异路由」的动机。

## 4. 落地

- 按用户决策：**EMMA、GIFNet 直接加入 §4.2 对比表（不删旧方法），最终 9 个对比方法 + Ours**；加入后 Ours 仍在 5 项指标 × 3 模态上全部第一。见 `content/section-comparison.md`。
- TC-MoA 因权重不可得未纳入（本文档留档说明，非回避）。
- 复现产物：`fusion_bench/fused[_final]/{EMMA,GIFNet}/<task>`、`fusion_bench/reports/<task>/leaderboard.csv`；环境 venv `emma`/`gifnet`/`tcmoa`；`code/ref/{MMIF-EMMA,GIFNet,TC-MoA}`。
