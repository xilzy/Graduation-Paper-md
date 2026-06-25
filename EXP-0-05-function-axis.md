# EXP-0-05：FUNCTION 轴（功能/显著区保留指标）

- 日期：2026-06-23
- 所属阶段：阶段 0（deliverable 0.11）
- 结果级别：S1（probe 15）
- 关联代码：`metrics/fusion_metrics.py`（`func_corr`/`func_sal`/`FUNCTION_AXIS`），`eval_fusion.py`（三轴排名 + Pareto 自检），`configs/gfp_pc.json`（`func_source`）
- 证据：`reports/gfp_pc_func_probe/`

## 1. 动机（承 EXP-0-02/03）
通用指标(FIDELITY/QUALITY)在 GFP-PC 上都被 PC 主导,测不到"有没有把 GFP 功能信息注入"。需要一条只看 GFP 信号区的指标。

## 2. 设计
- **功能源**由配置 `func_source` 指定(GFP-PC = A=GFP;IR-VIS 将设为 IR)。
- **显著掩码** = 功能源 Y 的最亮 10%(top percentile=90)像素 = 荧光/热目标所在。
- **FuncCorr** = 融合图与功能源在掩码内的皮尔逊相关(融合是否跟随功能图样)。
- **FuncSal** = (融合图掩码内均值 − 掩码外均值)/全局std(功能区在融合图里是否突出)。
- 两者越大越好。`FUNCTION_AXIS=[FuncCorr,FuncSal]`,GFP-PC 下作**主判据**。

## 3. 结果（probe 15）
| 方法 | FuncCorr | FuncSal | (FidRank/QualRank/FuncRank, 组合) |
|---|---|---|---|
| Avg(朴素) | **0.732** | **1.448** | 1.2 / 3.75 / 1.0 |
| Retrained_LRD | 0.641 | 0.853 | 1.2 / 1.75 / **1.0** → Composite **1.32** |
| MDFNet(官方) | 0.509 | 0.696 | 1.8 / 1.25 / 2.0 → Composite 1.68 |
| **Max(朴素)** | **0.249** | **0.135** | 3.0 / 2.0 / **4.0** |

## 4. 分析（评测协议至此可信）
1. **FUNCTION 轴成功抓到要害**:**Max(=复制 PC) 功能保留最差**(FuncCorr 0.249、FuncSal 0.135)——它丢掉了 GFP,这正是之前所有通用指标看不到的失败。✓
2. **三轴下每个朴素基线都是"退化极端"**:
   - **Max** = 全结构/无功能 → FUNCTION 垫底;
   - **Avg** = 全功能/无锐度(字面混入 GFP 所以功能/保真都高,但糊)→ QUALITY 垫底。
3. **正确的自检 = Pareto 不被支配**:要求没有任何真方法在三条轴上被某个朴素基线全面压制。结果 **PASS**——两个学习型方法都没被 Max/Avg 支配,占据"非退化中间带"。
4. 顺带:**Retrained_LRD 三轴组合分最佳(1.32)**,优于官方 MDFNet(1.68)——更平衡/保真且不糊。

## 5. 结论
GFP-PC 的评测协议补完:**FUNCTION(主) + FIDELITY + QUALITY 三轴 + Pareto 自检**。能同时识破"复制单源"(Max)与"糊成一团"(Avg)两类作弊,并正确把真融合放在中间带。这把"尺子"现在可信,可作为后续 MoE/多任务所有迭代的判据。

## 6. 下一步（阶段 0 收尾）
- 0.8 真窗口注意力(window_size>1) + 0.9 接入 IR-VIS/多聚焦数据(IR-VIS 用 output_mode=gray、func_source=IR)。
- 多任务接入后,FUNCTION 轴天然迁移到 IR-VIS 的"热目标显著性保留"。

> 备注：对应 MASTER_PLAN §6 / PROGRESS_CHECKLIST 0.11。
