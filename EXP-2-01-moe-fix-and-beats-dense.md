# EXP-2-01：修好 MoE（结构塌陷根因）并量化超过稠密基线

- 日期：2026-06-25　所属阶段：阶段 2（MoE 主创新）　结果级别：S1（avg-rank 三任务对照）
- 关联代码：`Networks/net_moe.py`（块 forward 布局修复 + MoEFFN 重写 + out_scale）、`train_moe.py`、`mm_eval_compare.py`（新增 S1 对照）、`mm_infer_check.py`
- 关联上一实验：EXP-FIX-04（放大 MoE 无效，判断"集成本身有问题"）。本次定位并修复了那个"集成问题"。

## 0. 一句话
MoE 此前逐源结构相关性≈0、放大也无改善的根因**不是容量、不是损失、不是专家求和幅度**，而是我在重写骨干时 **TransformerBlock 的张量布局 view/permute 不一致**。修复后 MoE 结构相关性从≈0 恢复到 0.92–0.99；再加 `out_scale` 稳定性控制 + 任务自适应损失，**在 S1 平均排名上三任务全胜、总分 1.63 < 稠密 2.19**——创新点成立、保留。

## 1. 问题回顾
EXP-0-06/FIX-04：MoE（含任务自适应）虽解除塌黑、S0 过，但**逐源 Y-corr≈0**（融合不跟结构），且 out_channel 16→64、专家 4→8 **均无改善**，更大反更不稳。判断"瓶颈在 MoE 集成"，但当时把矛头指向"共享+路由专家求和放大 FFN 残差"。

## 2. 诊断：阶梯实验定位根因
对称损失下跑 5 个变体（6ep，对照稠密的 corrB≈0.9+）：
| 变体 | corrB(gfp/irvis/med) |
|---|---|
| 在位修复(functional combine) | ≈0 / ≈0 / ≈0 |
| +out_scale | ≈0 | 
| **shared_only(n_routed=0，≈稠密 Mlp)** | **≈0** ← 关键 |
| aux=0 | ≈0 |

**关键证据**：连"只有共享专家、完全无路由"（等价于把稠密 Mlp 套进我的 MoE 壳）都 corrB≈0。→ 问题与 MoE FFN/路由/aux/在位写法**全无关**，出在 **MoETransformerBlock 骨干重写**本身。

## 3. 根因：view/permute 布局不一致
对比原始 `net.TransformerBlock`：它进出都用 **view**（`x.view(B,H,W,C)` 进、`x.view(B,C,H,W)` 出，是"重解释"内存而非转置），全程一致——网络据此学到固定布局。
我的 `MoETransformerBlock`：进用 view（重解释），**出却用 `permute(0,3,1,2)`（真转置）**——进出不一致 → 通道/空间布局被错位打乱 → 后续 conv 收到错乱特征 → 结构信息被冲掉、corrB≈0。
**修复**：让块进出严格镜像原始（view 进、view 出），仅把 FFN 换成 MoEFFN：
```python
x = x + y                       # y = MoEFFN(norm2(x))
return x.view(B, C, H, W)       # 重解释，与原始一致（不是 permute）
```

## 4. 修复验证（结构恢复）
对称损失，6ep：
| 变体（修复后） | structure 损失 | corrB(gfp/irvis/med) |
|---|---|---|
| 修复前 | ~3.25 | ≈0/≈0/≈0 |
| shared_only | **0.98** | 0.97/0.92/0.99 |
| full MoE | 1.26 | 0.97/0.96/0.99 |
| MoE+out_scale+任务自适应 | 1.07 | 0.98/0.97/0.99 |
→ structure 从 3.25 掉到 ~1.0，corrB 全面恢复到稠密水平（0.9+）。**布局 bug 即根因坐实。**

## 5. S1 量化对照：MoE vs 稠密（8ep，三任务 probe 各 15 图，平均排名）
用 `mm_eval_compare.py`（balance-aware RANK_METRICS：SSIM_hm/MS_SSIM_hm/Qabf_hm/VIF_hm/MI_hm/EN/SD/SF/AG → 平均排名，越低越好）：

| 方法 | 总平均排名 | gfp / irvis / med |
|---|---|---|
| **MoE+out_scale+任务自适应** | **1.63** | 1.67 / 1.67 / 1.56（全胜） |
| 稠密 | 2.19 | 2.22 / 2.22 / 2.11 |
| MoE(无 out_scale, 对称) | 2.19 | **8ep 退化成高频噪声**（SSIM_hm≈0.006、SF 84–136） |

## 6. 分析与诚实边界
- **创新点成立**：修复后的 MoE + 任务自适应在三任务上平均排名全面优于稠密（1.63 vs 2.19），且 SF（7.7/2.7/18.5）与稠密同量级——是真增益，非"靠噪声刷 SF/SD"。
- **out_scale 是稳定性必需**：去掉 out_scale 的对称 MoE 到 8ep 退化成高频噪声（SSIM_hm≈0、SF 暴涨）。这说明"专家求和放大残差幅度"的担忧**对稳定性成立**（虽不是 corr 塌陷的根因——那是布局 bug）。故最终配置 = 布局修复 + out_scale + 任务自适应。
- **指标陷阱记录**：平均排名含 SF/SD/AG（越大越好），高频噪声会刷高这三项——噪声 MoE 因此排名没垫底（与稠密并列 2.19），但其 SSIM_hm≈0.006 暴露真相。**说明 avg-rank 必须与退化检测同看**（与评测协议 §5.1-3 一致）；本次 S0 退化检测应补"高频噪声/伪影"判据。
- **结构 vs 锐度权衡**：moe_ta 总分赢，但 SSIM_hm 在 gfp/medical 上稠密略高（0.29/0.21 vs 0.21/0.16）——任务自适应 max-强度以少量结构相似度换取锐度/对比/信息量（SF/SD/MI 更高）。按既定 avg-rank 聚合，moe_ta 胜。
- **规模边界**：probe 15/任务、单 seed、8ep，是方向性结论非终评；需全测试集 + 多 seed + 显著性复核（S3）。

## 7. 结论 / 下一步
- **保留 MoE 创新**：根因已修、量化已超稠密。最终推荐配置：`net_moe.py` 布局修复 + `--out-scale --task-adaptive`（n_routed=4,k=2,shared=1,aux=0.01）。
- 下一步：① S0 退化检测补"高频噪声/伪影"判据（本次暴露）；② 全测试集 + 多 seed 的 S3 复核 + 显著性；③ 阶段2 路由诊断（专家-任务激活热力图、负载分布）、显式 vs 隐式路由消融（`--no-task-cond`）、专家数扩展性曲线（现在 corr 不再饱和，可重做）；④ 拆"out_scale / 任务自适应 / 路由"各自贡献的消融。
