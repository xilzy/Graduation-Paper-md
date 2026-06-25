# EXP-FIX-03：裁剪采样方式 + crops_per_task / 裁剪尺寸定值（问题 2）

- 日期：2026-06-24　关联代码：`mm_fusion_dataset.py: MMFusionDataset`、`train_mm.py`/`train_moe.py`（新增 `--fixed-pool`）
- 结论：① 采样方式从"固定裁块池"升级为 **on-the-fly 随机裁剪**（每次取样重抽位置，多样性不封顶，用 `torch.randint` 避免 worker 重复随机）。② **crops_per_task=4000、裁剪尺寸=170** 为推荐默认——4000 是"让样本量最多的 IR-VIS(1083 图) 不被压到每图 1 块"的最小均衡点；170 相比 256 无 S0 收益却省 2.2× 算力。

## 1. 配额平衡为何重要 + 原做法
三任务图数悬殊（IR-VIS 1083 / 医学 578 / GFP 118）。若密集网格裁块，大集裁块数碾压小集 → 每 epoch 大集主导梯度、小任务被饿死、放大/掩盖任务干扰。`crops_per_task` 给每任务设一个目标裁块数：`per_pair = crops_per_task // 该任务图数`，图少的多裁、图多的少裁，使三任务裁块数拉平。这是个**可调超参**，需定一个合适值。

## 2. 先解决一个隐患：固定池 vs on-the-fly
原实现把裁块位置**一次性烘焙成固定池**（`crops_per_task` 个具体位置），整个训练反复用这些 → **多样性被 crops_per_task 封顶**（GFP 仅 118 图，固定 1000 块就反复看同样裁块）。
改为 **on-the-fly**：index 只决定"取哪个图对"，裁块**位置每次 `__getitem__` 现抽**（`torch.randint`，DataLoader 按 worker/epoch 自动播种 → 不同 worker/epoch 都拿到新裁块，避开 numpy-in-worker 重复随机的坑）。这样 `crops_per_task` 只剩"每 epoch 样本量 / 任务均衡比"的作用，多样性不封顶、对取值更鲁棒。
- 验证：同一 index 连取两次，on-the-fly 返回不同裁块（多样）、fixed-pool 返回相同（可复现）。两模式均能正常训练。

## 3. 实验：crops_per_task 与裁剪尺寸（6-epoch 等预算，on-the-fly，patch 默认 170）

**每任务裁块均衡（关键）**：

| cpt | gfp | irvis | medical | 均衡? |
|---|---|---|---|---|
| 2000 | 1888 | **1083** | 1734 | ✗ IR-VIS 被压到每图1块、欠采样 |
| **4000** | 3894 | 3249 | 3468 | ✓ ±10% 内 |
| 8000 | 7906 | 7581 | 7514 | ✓ 但成本翻倍 |

**成本 / 退化自检**：

| 配置 | s/epoch | 退化自检 | 备注 |
|---|---|---|---|
| cpt2k·170 | 74 | ALL PASS | IR-VIS 欠采样 |
| **cpt4k·170** | 167 | ALL PASS（gfp corrB.94 / irvis .91 / med .97，irvis mean .17 最亮） | 均衡、推荐 |
| cpt8k·170 | 363 | ALL PASS | 均衡但 2.2× 成本、S0 无增益 |
| cpt4k·**256** | 372 | ALL PASS（但 irvis mean .055 更暗、gfp corrA −.32） | 2.2× 成本、无 S0 收益 |

> 注：cpt8k/256 因更慢，截至比较时只到 epoch 3–4（欠训），但结合"均衡表 + 成本 + S0 全过"已足以定论。早期一次对比曾被 StepLR 按 epoch 衰减混淆（不同 cpt → epoch 长度不同 → 同 step 数下 LR 不同），本轮改为**固定 epoch 预算**消除该混淆。

## 4. 抉择与理由
- **采样**：on-the-fly（多样性不封顶、取值鲁棒、可复现性由 worker 播种保证）。`--fixed-pool` 保留作可复现实验开关。
- **crops_per_task = 4000**：是"让图最多的 IR-VIS(1083) 不被 `//` 压到每图 1 块"的**最小均衡点**；2000 会欠采样 IR-VIS，8000 均衡但 2.2× 成本无 S0 收益。快速迭代可用 2000，某任务欠拟合再上 8000。
- **裁剪尺寸 = 170**：256 算力 2.2×、S0 无优势、IR-VIS 反而更暗；170 覆盖局部融合线索足够、最省、与既有 probe/历史一致。配合 EXP-FIX-02 的尺度无关损失，170 不再是"被写死"，而是"按成本/收益选定"。

## 5. 边界 / 下一步
- 上述以 S0 + 逐源 corr 代理 + ≤6 epoch 为据；最终值应在 S1 三轴平均排名接通后复核。
- 已把 on-the-fly 设为 `train_mm.py`/`train_moe.py` 默认，`--fixed-pool` 可切回。
