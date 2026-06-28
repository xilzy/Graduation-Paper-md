# 指标计算方法与评测流程总文档（EVALUATION）

> 本文件汇总：指标计算的 **Python 脚本位置**、**每个指标的定义/公式/方向/对应 MATLAB 源**、**彩色图 RGB 计分协议**、**聚合（平均排名）** 与 **运行方式**。是 MASTER_PLAN §0 中规划的 `EVALUATION.md` 的落地。
>
> 日期：2026-06-28。所有指标为 `evaluation/*.m`（GBK 编码的原始 MATLAB 套件）的 Python 移植，公式逐一对齐。

---

## 0. 代码位置（计算的"真源"）

| 文件 | 作用 |
|---|---|
| `code/Graduation-Paper/metrics/fusion_metrics.py` | **所有指标的逐项实现**（核心9 + 诊断 + 平衡感知 *_hm + 功能轴）。输入为 2-D float 灰度数组，值域 [0,255]。 |
| `code/Graduation-Paper/metrics/__init__.py` | 指标注册表：`CORE_METRICS / DIAGNOSTIC_METRICS / RANK_METRICS / HIGHER_IS_BETTER / FIDELITY_AXIS / QUALITY_AXIS / FUNCTION_AXIS`；`compute_all()`。 |
| `code/Graduation-Paper/metrics/aggregate.py` | `average_rank_table()`：按方向给每个指标排名→平均排名。 |
| `code/Graduation-Paper/bench/prep_inputs.py` | 把三任务测试对导出为标准化 8-bit 灰度 `inputs/<task>/{A,B}`，并存彩色源 `cbcr/`。 |
| `code/Graduation-Paper/bench/eval_method.py` | **单方法×单任务评测驱动**：对一个方法的融合图目录算全指标→`per_image.csv / means.csv / leaderboard.csv`。 |
| `code/Graduation-Paper/bench/recombine_rescore.py` | 彩色任务的 RGB-final 重组（融合Y + 源CbCr→RGB）。 |
| `code/Graduation-Paper/bench/consolidate.py` | 跨方法汇总→平均排名 `comparison.csv` + `COMPARISON.md`。 |
| `evaluation/*.m` | 原始 MATLAB 套件（参考与对拍来源；集群无 MATLAB，故用 Python 移植）。 |

---

## 1. 输入约定与彩色图 RGB 计分协议（关键）

- 度量对象统一为 **灰度 [0,255] 三元组 `(A, B, F)`**：A=源1、B=源2、F=融合图。
- **源 A/B**：彩色源转灰度取亮度 Y（PIL `'L'` = BT.601 `rgb2gray` = 0.299R+0.587G+0.114B），与 MATLAB `main.m` 的 `rgb2gray` 一致。
- **融合图 F 的计分基准（与仓库 `infer_fusion.py` + MATLAB `evaluation/main.m` 对齐）**：
  - 任务 `output_mode=rgb`（**medical / gfp_pc**）：最终融合图 = `ycbcr_to_rgb(融合Y, fuse_chroma(源CbCr))`（含 uint8 截断），计分时对该 **RGB 图做 `rgb2gray`**。
    > MATLAB `main.m`：`if size(fused,3)==3, fused = rgb2gray(fused)`；RGB 逆变换的 uint8 截断在高饱和色区（PET/SPECT 伪彩、GFP 绿）会改变灰度，故不能直接用融合 Y 计分。
  - 任务 `output_mode=gray`（**irvis**）：融合图本就是灰度，直接计分（与 MDFNet 自身评测一致）。
- 尺寸不一致时把 B、F 双线性对齐到 A（数据集已配准，仅尺寸兜底）。

---

## 2. 核心 9 指标（恒报）

约定：↑=越大越好，↓=越小越好。Python 函数均在 `metrics/fusion_metrics.py`。

| 指标 | 含义 | 方向 | 计算要点（Python 实现） | MATLAB 源 |
|---|---|---|---|---|
| **EN** 信息熵 | 融合图自身信息量 | ↑ | `en(F)`：uint8 直方图 256 bin，`-Σ p log2 p` | `MyEntroy.m` |
| **MI** 互信息 | 从两源转移到 F 的信息总量 | ↑ | `mi=MI(A,F)+MI(B,F)`；`_mi_pair` 用 256×256 联合直方图算 `Hx+Hy-Hxy` | `MI.m`/`mutual_info.m` |
| **SD** 标准差 | 对比度 | ↑ | `F.std()` | `SD.m` |
| **SF** 空间频率 | 行/列梯度能量 | ↑ | `sqrt(RF²+CF²)`，RF/CF 为行/列一阶差分 RMS | `MySF.m` |
| **AG** 平均梯度 | 清晰度/纹理 | ↑ | `mean( sqrt((gx²+gy²)/2) )` | `AverageGradient.m` |
| **SSIM** 结构相似 | F 对两源结构保真均值 | ↑ | `(SSIM(F,A)+SSIM(F,B))/2`，11×11 高斯窗 σ1.5，C1/C2 标准 | `ssim_index.m` |
| **MS_SSIM** 多尺度SSIM | 多尺度结构保真均值 | ↑ | 5 级权重 (0.0448…0.1333)，逐级高斯下采样 | `msssim.m` |
| **Qabf** 梯度转移质量 | 源边缘被 F 保留的比例 | ↑(0~1) | Xydeas–Petrovic：Sobel 梯度幅值/方向 → Qg·Qa，按源梯度强度加权 | `Qabf.m` |
| **VIF** 视觉信息保真 | 人眼信息保真（像素域 VIFp） | ↑ | `(VIFp(A,F)+VIFp(B,F))/2`，4 尺度高斯金字塔 | `vifvec.m`/`VIFF_Public.m` |

> 注：VIF 为像素域 VIFp（非小波域 VIFF），数值偏小但单调一致，可用于排名。

---

## 3. 诊断指标（定位问题用）

| 指标 | 含义 | 方向 | 实现 | MATLAB 源 |
|---|---|---|---|---|
| **Nabf** | 融合引入的伪影/噪声（F 梯度超过两源处的未转移信息） | ↓ | `nabf(A,B,F)`，复用 Qabf 的梯度框架，artifact mask=`gF>gA & gF>gB` | `analysis_nabf.m` |
| **SCD** | 差分相关和（互补信息保留） | ↑ | `corr(F-B,A)+corr(F-A,B)` | `analysis_SCD.m` |
| **CC** | F 与两源相关系数均值 | ↑ | `(corr(A,F)+corr(B,F))/2` | `my_cc.m` |
| **PSNR** | F 对两源均值的峰信噪比 | ↑ | `10·log10(255²/MSE)`，MSE 对 (A+B)/2 | `psnr.m` |
| **Balance** | 模态平衡度 `min(SSIM_A,SSIM_B)/max(...)` | ↑(0~1) | 暴露模态偏置 | EXP-0-02 设计 |
| **MI_A/MI_B, SSIM_A/SSIM_B** | 分源拆解 | — | 诊断单源偏向 | — |

---

## 4. 平衡感知 *_hm 与两轴（抗刷分，S1 排名用）

直接用"求和/平均"的保真度，会被"复制单源"（如 naive Max）刷高。故用 **逐源调和平均（soft-min）**：弱源主导，复制单源会塌掉。

- `*_hm`：`SSIM_hm / MS_SSIM_hm / Qabf_hm / VIF_hm / MI_hm`，`_hm(a,b)=2ab/(a+b)`（任一≤0 则 0）。
- **两轴分开排名**（避免指标数量失衡偏置；要求 Pareto 改进而非以一换一）：
  - `FIDELITY_AXIS = [SSIM_hm, MS_SSIM_hm, Qabf_hm, VIF_hm, MI_hm]`（对两源的平衡保真，杀"复制/模糊"作弊）
  - `QUALITY_AXIS = [EN, SD, SF, AG]`（无参考图像丰富度，杀"全图模糊"作弊）
- GFP-PC 警示：一源近黑（GFP）时保真轴被另一源主导，平凡 copy/blend 会占优，此时以质量轴为判别——详见 `EXP-0-02-balance-aware-eval.md`。

---

## 5. 功能轴（FUNCTION，任务相关）

通用指标看不出"功能源的稀疏显著信号是否被注入"（结构源会主导）。故只在**功能源的显著区**（最亮 10% 像素：GFP 荧光点 / IR 热目标 / PET-SPECT 代谢亮区）度量：

- **FuncCorr** ↑：显著区内 F 与功能源的 Pearson 相关（功能图样是否被跟随）。
- **FuncSal** ↑：`(显著区均值 - 非显著区均值)/全局std`（功能区在 F 中是否凸显）。
- func 源由任务配置：gfp_pc/medical=源A，irvis=源B（热目标）。

---

## 6. 聚合：平均排名（MASTER_PLAN §5.4 拍板）

`metrics/aggregate.py: average_rank_table()` 与 `bench/consolidate.py`：
- {全部方法} 在核心 9 指标各自排名（rank 1=最好，按 `HIGHER_IS_BETTER` 方向，↓类如 Nabf 反向），取**平均排名 AvgRank**为单一总分，**越低越好**。
- 对量纲鲁棒、最抗刷分；配套同看：指标族雷达图 + 分源平衡条 + 逐图折线。
- 跨方法总表输出：`fusion_bench/reports/<task>/comparison.csv` 与 `COMPARISON.md`（= 仓库 `COMPARISON-leaderboard.md`）。

---

## 7. 正确性保障（集群无 MATLAB）

1. **公式逐项对齐** `evaluation/*.m`（见各函数 docstring 引用的 .m）。
2. **单元性质测试**：自融合 SSIM=1、常数图熵=0、尺度/朝向不变性。
3. **去欺骗自检**：平凡 Avg/Max 基线作为"探测器"，真实方法须在质量轴上压过它们（`eval_fusion.py` 的 PARETO 自检）。
4. 如有可跑 MATLAB 的机器，终评前两边对拍一次。

---

## 8. 运行方式（CLI）

```bash
PY=/ytech_m2v4_hdd/lizhongyin/venv/bin/python
BENCH=/ytech_m2v4_hdd/lizhongyin/fusion_bench
CODE=/ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper

# 0) 一次性：导出标准化测试输入
$PY $CODE/bench/prep_inputs.py

# 1) 某方法把融合 Y 写到 fused/<M>/<task>/<stem>.png 后：
#    彩色任务先做 RGB-final 重组（irvis 跳过）
$PY $CODE/bench/recombine_rescore.py --method <M>            # 生成 fused_final/<M>/{medical,gfp_pc}

# 2) 评测（irvis 用 fused/，medical|gfp_pc 用 fused_final/）
$PY $CODE/bench/eval_method.py --task irvis   --name <M> --fused-dir $BENCH/fused/<M>/irvis
$PY $CODE/bench/eval_method.py --task medical --name <M> --fused-dir $BENCH/fused_final/<M>/medical
$PY $CODE/bench/eval_method.py --task gfp_pc  --name <M> --fused-dir $BENCH/fused_final/<M>/gfp_pc

# 3) 跨方法汇总 + 平均排名
$PY $CODE/bench/consolidate.py
```

直接在 Python 里算单张：
```python
import sys; sys.path.insert(0, "code/Graduation-Paper")
import metrics as M
vals = M.compute_all(A, B, F)   # A,B,F: 2-D float [0,255]；返回 {指标:值}
```

---

## 9. 相关文档
- 指标选取理由与四级测试阶段：`MASTER_PLAN.md §5`。
- 平衡感知评测设计与证据：`EXP-0-02-balance-aware-eval.md`。
- 18 方法对比结果与 RGB 协议修订：`SUMMARY-comparison-18methods.md`、`COMPARISON-leaderboard.md`、各 `EXP-CMP-*`。
