# 阶段清单对照（Progress Checklist）

> 与 MASTER_PLAN.md 配套的"进度对账表"。每个阶段列出 deliverable 勾选项;每完成一项就更新状态并在对应 EXP 记录里留证据。
>
> 图例：✅ 完成 · ⚠️ 部分/未达标 · ⏳ 进行中 · ❌ 未开始
>
> 规则：① 一个阶段的全部 deliverable ✅ 才算该阶段完成,方可进入下一阶段(硬前提项见标注)。② 每次状态变化都要能指到一份 EXP 记录或 reports/ 证据。③ 本表随项目持续更新。

---

## 阶段 0 — 基线复现与多任务底座

| # | Deliverable | 状态 | 证据 |
|---|---|---|---|
| 0.1 | 环境配置(venv + torch + 依赖) | ✅ | EXP-0-01 §环境;`requirements_lock.txt` |
| 0.2 | 评测指标 Python 化 + 性质自测 | ✅ | EXP-0-01;`selftest_metrics.py`(15/15) |
| 0.3 | 平衡感知 + 两轴评测协议 | ✅ | EXP-0-02 |
| 0.4 | 指标口径与 MATLAB 对齐(rgb2gray 灰度) | ✅ | EXP-0-03 §7 |
| 0.5 | YCbCr 颜色流水线(RGB 任务两次转换) | ✅ | EXP-0-03 |
| 0.6 | 训练管线重构(真 batch + FusionDataset + in_channel 参数化) | ✅ | EXP-0-01;`train_fusion.py` |
| 0.7 | GFP-PC 锚点 S1 报表 | ✅ | EXP-0-01;`reports/gfp_pc_all` |
| 0.8 | 补回真窗口注意力(window_size>1) | ❌ | 当前 window=1 |
| 0.9 | 接入 IR-VIS / 医学数据(多任务底座)**[阶段1硬前提]** | ✅ | EXP-0-06;三任务(GFP-PC/IR-VIS-MSRS/医学-Harvard)统一数据集打通。新增 `mm_fusion_data.py`(suffix+folder 双配对)、`mm_fusion_dataset.py`(Y域统一/170裁块/任务配额平衡)、`train_mm.py`、configs;医学下载 `Havard-Medical-Image-Fusion-Datasets`(810 对)→ `data/Harvard-Medical`(578/48)。冒烟+8ep 训练无报错。数据/预处理学习见 `DATASETS_AND_PREPROCESSING.md`。**注**:MFI-WHU 多聚焦仍 .rar 待解压(本批用医学替代第三任务);ACM(layers.py)假设方形输入,非方形(MSRS 640×480)整图推理需先方裁/修 ACM |
| 0.10 | 验证单任务不退化(管线正确/无伪影/产出合理) | ✅ | EXP-0-04;LR衰减修复伪影(Nabf 0.49→0.11);官方 checkpoint 作冻结参照,`p0_gfp_pc_lrd` 作可训 baseline(不强求逐位复刻) |
| 0.11 | 功能/显著区保留指标(GFP-PC 评测要点) | ✅ | EXP-0-05;FUNCTION 轴(FuncCorr/FuncSal)+ 三轴 + Pareto 自检 PASS;Max 功能垫底被识破 |
| 0.12 | 与 MATLAB 指标数值对拍(可选) | ❌ | 集群无 MATLAB |

**收尾顺序**：0.10(#6) → 0.11(#7) → 0.8 + 0.9 → (可选 0.12)。

---

## 阶段 1 — 单一稠密模型多任务联合训练（无 MoE 对照）

| # | Deliverable | 状态 | 证据 |
|---|---|---|---|
| 1.1 | 共享 backbone + 任务/模态嵌入 | ❌ | |
| 1.2 | 各任务单独训 baseline | ❌ | |
| 1.3 | 三任务联合训 | ❌ | |
| 1.4 | 任务干扰矩阵(联合 vs 单训掉点量化) | ❌ | |
| 1.5 | 通用/专精权衡 baseline 数字(论证 MoE 必要性) | ❌ | |

## 阶段 2 — 引入 MoE（论文主创新）

| # | Deliverable | 状态 | 证据 |
|---|---|---|---|
| 2.1 | TM 的 FFN→MoE-FFN(top-k 路由) | ⏳ | EXP-0-06;`Networks/net_moe.py: MoEFFN`(top-k 路由)已实现并训练无报错,待 S1 评测对照 |
| 2.2 | 路由条件对比(token / +任务 / +模态) | ⏳ | `net_moe.py` 任务嵌入条件路由 + `--no-task-cond` 隐式路由消融开关已就位,待跑对照 |
| 2.3 | 共享专家+路由专家 结构对比 | ⏳ | `net_moe.py` 1 共享专家 always-on + N 路由专家已实现,待结构消融 |
| 2.4 | 负载均衡(aux loss / 容量因子) | ⏳ | `net_moe.py` Switch/GShard aux loss + `train_moe.py --aux-weight` 已实现并训练,待专家负载诊断 |
| 2.5 | 专家负载分布 / 专家-任务激活热力图 / 利用率 | ❌ | |
| 2.6 | 扩展性曲线(专家数×top-k → 质量-参数量-FLOPs) | ❌ | |

## 阶段 3 — 无 GT 的统一多任务损失

| # | Deliverable | 状态 | 证据 |
|---|---|---|---|
| 3.1 | 任务自适应损失(IR 显著性 / MFF 清晰区 / GFP-PC 功能+对比) | ❌ | |
| 3.2 | 固定权重 vs 任务条件权重 vs 不确定性自动加权 对比 | ❌ | |
| 3.3 | 加 aux loss 后优化稳定性对比 | ❌ | |

## 阶段 4 — 分布式训练与效率

| # | Deliverable | 状态 | 证据 |
|---|---|---|---|
| 4.1 | FSDP2 分片 + 专家并行(EP) + 激活检查点 + bf16 | ❌ | |
| 4.2 | 单卡 vs FSDP8(±EP) 的 MFU/TPS/显存/扩展效率 | ❌ | |

## 阶段 5 — 综合对比 + 消融（成文）

| # | Deliverable | 状态 | 证据 |
|---|---|---|---|
| 5.1 | 各任务 SOTA 对比(统一报表) | ❌ | |
| 5.2 | 逐项消融(真窗口注意力/独立多尺度/MoE/共享专家/任务自适应损失/负载均衡) | ❌ | |
| 5.3 | 跨任务泛化(zero-shot / 少样本) | ❌ | |
| 5.4 | 显著性检验 | ❌ | |

---

*最后更新：2026-06-23(阶段 0 收尾进行中)。*
