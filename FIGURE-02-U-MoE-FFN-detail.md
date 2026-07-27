# U-MoE-Fusion 的 MoE-FFN 详细图

## 1. 产物

- `Materials/figs/fig_u_moe_ffn_detail.pdf`：论文排版用矢量版本；
- `Materials/figs/fig_u_moe_ffn_detail.png`：高分辨率预览与 Word/PPT 兼容版本；
- `script/make_moe_ffn_detail_figure.py`：可复现绘图脚本。

该图参考原论文 `Graduation-Paper/figs/3.pdf` 的横向分栏、浅色大背景、虚线分组边框和神经元示意风格，但模块内容完全依据当前 `Networks/net_moe.py` 与最终 W96L 配置重新绘制。

## 2. 图中信息与实现对应关系

### A. Transformer 中的 MoE-FFN

左侧从输入特征 `F'` 经 LayerNorm 得到 token `h`，再进入 MoE-FFN；模块输出与原始 `F'` 做残差相加得到 `F''`。这对应 `MoETransformerBlock.forward()` 中的：

```python
y = self.norm2(x.reshape(B * H * W, C)).view(B, H, W, C)
y = self.mlp(y.reshape(B, H * W, C), task_emb).view(B, H, W, C)
x = x + y
```

图中 `T=B×H×W` 表示送入专家计算前展平后的总 token 数。

### B. 任务条件路由

每个 token 与对应任务嵌入相加形成路由条件：

```python
gate_in = flat + task_emb[:, None, :]
logits = self.gate(gate_in)
probs = softmax(logits)
topv, topi = probs.topk(2)
topv = topv / topv.sum(...)
```

因此图中依次画出 `Task ID → e_t`、`z=h+e_t`、线性门控、softmax、top-2 与选中权重重归一化。E3 和 E9 只用于示意一次 token 级选择，实际被选专家随 token、样本和任务变化，并非固定路由。

### C. 共享专家与路由专家

最终配置包含 1 个常开共享专家和 12 个路由专家。共享专家对每个 token 始终执行；路由分支只执行 top-2 专家，输出为：

```text
F_MoE = E_s(h) + sum_{i in Top-2} alpha_i E_i(h)
```

路由概率同时产生 Switch/GShard 风格的负载均衡辅助项：

```text
L_balance = 12 * sum_i(f_i * p_i)
```

训练总损失中的权重为 `0.01`。红色虚线仅表示训练辅助约束，不属于推理数据流。

### D. Grouped-capacity dispatch

黄色 dispatch 模块对应 `combine="grouped"` 的固定容量实现。每个 token 仍只进入 top-2 专家，专家容量为：

```text
cap = max(1, floor(1.25 * T * k / E)),  k=2, E=12
```

该实现把 token 按专家分组并填充到固定容量，再通过两次批量 GEMM 完成专家计算；它优化执行方式，但不改变上方共享专家加 top-2 路由专家的模型语义。

### E. 单个 FFN 专家

右侧蓝色放大框展示共享专家与各路由专家共同采用的拓扑：

```text
Linear(C→4C) → GELU → Dropout → Linear(4C→C) → Dropout
```

各专家结构相同、参数相互独立。当前实现默认 `drop=0` 时 Dropout 等价于恒等映射，但图中保留该层以完整表示 `Expert.forward()`。

## 3. 版式与视觉约束

- 主 MoE 路径采用参考图的浅橙背景，Expert 放大框采用浅蓝背景；
- Router、共享专家、路由专家、选中 top-2、汇聚/残差和训练辅助分支使用不同颜色；
- 全部数据流箭头位于模块边框下方，并在方框端点前后退让，避免箭头或线尾穿框；
- 仅必要的共享旁路与残差旁路使用直角折线，其余连接使用水平、垂直或短直线；
- E3/E9 使用独立橙色突出，未选专家不绘制执行箭头；
- 底部图例覆盖图中所有颜色和路径语义；
- 全部英文使用 Times New Roman，输出仅保留 PDF 与 PNG。

## 4. 重新生成

```bash
cd /ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper-md
/ytech_m2v4_hdd/lizhongyin/venv/gifnet/bin/python \
  script/make_moe_ffn_detail_figure.py
```

## 5. 建议图题

> **图 3-x　U-MoE-Fusion 中任务条件 MoE-FFN 的详细结构。** 归一化 token 与任务嵌入共同生成 softmax 路由分数；一个共享专家始终参与计算，12 个路由专家中仅 top-2 被稀疏激活并按归一化权重汇聚。右侧给出单个 FFN 专家的两层前馈结构，红色虚线表示仅训练时使用的负载均衡辅助约束。
