# 对比方法复现综合总结（18 方法 × 3 模态）

- 日期：2026-06-25
- 范围：在统一标准化基准上复现 **18 个图像融合对比方法**，在 **IR-VIS(MSRS) / 医学(Harvard) / 显微(GFP-PC)** 三模态测试集上统一融合与计算指标。
- 逐方法过程记录见 `EXP-CMP-01..14`；机器可读结果见 `fusion_bench/reports/<task>/{leaderboard,comparison}.csv` 与 `fusion_bench/reports/COMPARISON.md`。

## 1. 复现基础设施（统一、可比、可复算）
- **标准化输入**：三任务测试对统一导出 8-bit 灰度 `fusion_bench/inputs/<task>/{A,B}/<stem>.png`（A=彩色/功能源，B=灰度/结构源），并存彩色源 `cbcr/<stem>.npy`。规模：irvis 50（MSRS 测试均匀抽样）、medical 48、gfp_pc 30。
- **统一输出契约**：每方法把融合 Y 按 stem 写到 `fusion_bench/fused/<Method>/<task>/`。
- **RGB 计分协议（重要，2026-06-28 修订）**：参照仓库 `infer_fusion.py` 与原始 MATLAB `evaluation/main.m`——**彩色源任务（`output_mode=rgb`：medical、gfp_pc）的最终融合图 = 融合 Y 与源 CbCr 重组逆变换得到的 RGB 图，计分时对该 RGB 做 `rgb2gray`（=PIL 'L'，BT.601）**。RGB 逆变换的 uint8 截断会在高饱和色区改变灰度，故不能直接用 Y 计分。RGB-final 图存 `fusion_bench/fused_final/<Method>/<task>/`。**irvis 为 `output_mode=gray`（与 MDFNet 自身评测一致），保持灰度。** 全部 18 方法已按此协议重算 medical/gfp_pc。
- **统一评测**：共享 `metrics/` 包计算核心 9 项（EN/MI/SD/SF/AG/SSIM/MS_SSIM/Qabf/VIF）+ 诊断（SCD/Nabf/CC）+ 功能轴（FuncCorr/FuncSal），逐图 CSV→均值→任务级 leaderboard→平均排名 `comparison.csv`。
- **环境隔离**：每方法独立 venv 于 `/ytech_m2v4_hdd/lizhongyin/venv/<method>`（base.pth 继承 torch2.8，私装方法依赖），不污染系统 python。
- **算力**：8×H800，每方法子 agent 指定单卡并行复现；外网经代理。

## 2. 18 方法清单与来源

| # | 方法 | 年份/类型 | 复现来源 | 权重 |
|---|---|---|---|---|
| 1 | CDDFuse | CVPR'23 Transformer分解 | 作者repo | 自带 IVF/MIF |
| 2 | SwinFusion | JAS'22 Swin | 作者repo | 自带（IVF/Med/MFF）|
| 3 | U2Fusion | TPAMI'20 统一无监督 | 作者TF权重→PyTorch逐层移植（误差3.6e-7）| 作者权重转换 |
| 4 | DenseFuse | TIP'19 自编码 | 作者pytorch port | 自带 gray |
| 5 | IFCNN | InfFus'20 通用CNN | uzeful/IFCNN | 自带 IFCNN-MAX |
| 6 | SeAFusion | InfFus'22 语义感知 | 作者repo | 自带 |
| 7 | TarDAL | CVPR'22 目标感知GAN | 作者repo | release tardal-dt |
| 8 | PIAFusion | InfFus'22 光照感知 | linklist2 pytorch port | 自带 |
| 9 | RFN-Nest | InfFus'21 残差融合 | 作者repo | 自带 RFN |
| 10 | LRRNet | TPAMI'23 低秩表示 | 作者repo | 自带 |
| 11 | DATFuse | TCSVT'23 双注意力Transformer | tthinking/DATFuse | 自带 |
| 12 | MURF | TPAMI'23 配准+融合 | 作者repo（TF1.15@H800 via nvidia-tensorflow）| 自带 finetuning ckpt |
| 13 | DDFM | ICCV'23 扩散 | 作者repo + OpenAI 256×256 先验 | 扩散权重(代理下载) |
| 14 | LP | 传统 拉普拉斯金字塔 | 自实现 | 无需 |
| 15 | DWT | 传统 离散小波 | 自实现(pywt) | 无需 |
| 16 | DTCWT | 传统 双树复小波 | 自实现(dtcwt) | 无需 |
| 17 | GTF | 传统 梯度转移(ADMM) | 自实现 | 无需 |
| 18 | NSCT* | 传统 非下采样轮廓波(近似) | 自实现 | 无需 |

> 论文 `initial_paper.tex` 列出的对比方法（NSCT/MSTSR/CNN/IFCNN/EMFusion/MURF/MATR/DPCN…）中，IFCNN、MURF、NSCT 已直接复现；其余传统/特定方法以本批 LP/DWT/DTCWT/GTF/NSCT* + 深度 SOTA 覆盖同一指标族。

## 3. 平均排名总评（核心 9 指标，方向感知，越低越好；18 方法；RGB-final 协议）

**IR-VIS (MSRS, n=50，灰度)** — Top: CDDFuse(3.56) · SeAFusion(4.89) · SwinFusion(5.00) · PIAFusion(5.22) · IFCNN(5.56)。尾部 U2Fusion/GTF（偏暗、低对比）。

**医学 (Harvard, n=48，RGB-final)** — Top: CDDFuse(4.22) · IFCNN(5.33) · SwinFusion(5.67) · PIAFusion(6.89) · DenseFuse(7.22)；DDFM(8.22) 在医学域明显优于其 IR-VIS 表现。

**显微 GFP-PC (n=30，RGB-final)** — Top: SeAFusion(5.22) · SwinFusion(6.11) · PIAFusion(6.33) · TarDAL(6.89) · IFCNN(7.00)；CDDFuse(7.33) 因用医学权重跨域、Nabf 偏高而退位。

> 注：medical/gfp_pc 为 RGB-final 协议重算结果（见 §1）；与旧的"直接 Y 计分"相比，单指标有小幅变化（如 CDDFuse medical MI 3.61→3.73、SD 81.3→79.5），平均排名顺序基本不变（个别名次 ±1）。完整逐指标表见 `COMPARISON-leaderboard.md` 与各 `comparison.csv`。

## 4. 跨方法观察（为本课题服务）
- **没有单一方法三模态通吃**：CDDFuse 统治 IR-VIS/医学，但 GFP-PC 让位于 SeAFusion/PIAFusion——印证 MASTER_PLAN 的"通用 vs 专精"动机与 MoE 任务特异路由的必要性。
- **跨域迁移普遍掉点**：多数方法仅有 IR-VIS 权重，迁到医学/显微时 SSIM↓、Nabf↑（尤以 GFP-PC 大面积近黑背景最明显），是统一多任务/任务自适应损失要解决的核心难点。
- **传统方法仍是硬基线**：LP/DTCWT/NSCT* 在 SD/SF/Qabf 等"边缘+对比"指标上不输部分深度法；GTF 强 B-偏置（Nabf 极低但抑制功能源，GFP-PC 上 FuncSal 转负），是良好的"退化对照"。
- **评测尺子自洽**：所有真实方法在质量轴上均优于平凡 Avg 基线（去欺骗自检通过），平均排名对量纲鲁棒，可直接用于论文 SOTA 对比与消融锚点。

## 5. 复现可靠性备注
- U2Fusion：作者 TF1 权重逐层移植到 PyTorch，前向对拍误差 3.6e-7（即作者真实权重，非重训）。
- MURF：TF1.15 经 `nvidia-tensorflow` 跑通 H800/CUDA12。
- DDFM：扩散权重**截断会静默回退随机初始化**，已以 `torch.load` 成功校验完整 2.21GB 权重；25 步 DDIM。
- 跨域任务（medical/gfp_pc）多数复用方法的 IR-VIS 权重，结果反映分布迁移而非该域专训，已在各 EXP 文档注明。
