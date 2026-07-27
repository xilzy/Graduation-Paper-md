# 分布式训练机制对比原理图

## 1. 产物

- `Materials/efficiency/figures/distributed_parallelism_comparison.pdf`：论文排版用矢量版本；
- `Materials/efficiency/figures/distributed_parallelism_comparison.png`：高分辨率预览版本；
- `script/make_distributed_parallelism_figure.py`：可复现绘图脚本。

该图以 `content/section-efficiency.md` 表 4-21 为基础，完整比较表中六类机制，并补充 Tensor Parallel、Pipeline Parallel、Ulysses Parallelism 和 Context Parallel，共十类分布式训练方法。

## 2. 图的组织方式

图采用两行五列布局，每个模块都用相同的三个问题解释：

1. **Shard**：究竟切分数据、参数、梯度、优化器状态、层、专家还是序列；
2. **Comm**：主要依赖 All-Reduce、Reduce-Scatter/All-Gather、All-to-All、P2P 还是 Ring P2P；
3. **U-MoE**：该机制是否针对当前 U-MoE-Fusion 的实测瓶颈。

第一行对应表 4-21 的数据并行和状态切分机制；第二行补充模型、专家和长序列并行轴。底部总结栏将当前实测资源构成与最终选择直接连接。

## 3. 表 4-21 中的六类机制

### 3.1 PyTorch DDP

每个 GPU 保存完整模型，不同 rank 处理不同 mini-batch，反向传播后对梯度做 All-Reduce。当前模型采用：

- `gradient_as_bucket_view=True`；
- static graph；
- 约 8 MiB 的两桶配置；
- fused Adam；
- 正常的异步梯度归约。

这与“参数很小、激活很大”的资源构成匹配。

### 3.2 Megatron overlap-grad-reduce

该机制不是新的切分维度，而是当某个梯度桶就绪时立即异步发起 All-Reduce，使其与后续反向计算重叠。PyTorch DDP reducer 已提供同类能力，因此图中标为 **IN DDP**。

### 3.3 Megatron Distributed Optimizer / Parameter Gather

优化器状态和归约后的梯度按 data-parallel rank 切分，通过 Reduce-Scatter 分发梯度，并在需要时 All-Gather 参数。它适合参数和优化器状态占用很大的模型；U-MoE-Fusion 仅约 4.11 M 参数，状态总量不足 0.1 GB，节省有限。

### 3.4 DeepSpeed ZeRO / FSDP

ZeRO 的不同 stage 或 FSDP 可进一步切分参数、梯度和优化器状态。计算某层前需要 All-Gather 参数，反向后使用 Reduce-Scatter。当前峰值显存主要来自激活、融合损失和 MoE 容量缓冲，因此切分模型状态不能解决主要问题，反而增加层级 collective。

### 3.5 Expert Parallel

不同 GPU 保存不同专家，token 根据路由结果经 All-to-All 分发，专家计算完成后再通过 All-to-All 合并。当前 12 个小专家能够全部放在单卡，并已通过 grouped-capacity 形成规则批量 GEMM；改为 EP 会把本地 bmm 变成网络上的 token 交换。

### 3.6 固定 shape 与成本感知数据分片

该机制不切分模型，而是在进入普通 DDP 前按可预测成本或历史耗时重新分配样本，使各 rank 的预计总工作量接近。当前固定形状数据仅按任务标签均衡没有收益，因此图中强调：只有观测到真实成本异构后才采用成本感知分配。

## 4. 补充的四类并行机制

### 4.1 Tensor Parallel（TP）

把单层大矩阵沿行或列切到多个 GPU，每层 GEMM 后需要 All-Reduce 或 All-Gather。它适合单层矩阵无法放入单卡的大宽度模型；当前主干宽度仅 `C=96`，小矩阵上的高频 collective 不划算。

### 4.2 Pipeline Parallel（PP）

把连续层划分为多个 stage，不同 GPU 之间通过 P2P 传递激活和梯度，并用多个 micro-batch 填充流水线。当前骨干较浅，stage 间计算量不足以摊薄 pipeline bubble。

### 4.3 Ulysses Parallelism（UP）

这里将用户给出的 **UP** 按分布式长序列训练中的通行含义解释为 **Ulysses Parallelism**：

1. 输入先沿序列维切分为 `S/P × H`；
2. 第一次 All-to-All 把布局转换为完整序列、部分注意力头，即 `S × H/P`；
3. 每个 rank 对自己的注意力头执行完整序列注意力；
4. 第二次 All-to-All 恢复序列切分布局。

当前窗口注意力的单窗序列长度只有 64，且总头数为 8，不存在超长序列激活瓶颈。

### 4.4 Context Parallel（CP）

每个 GPU 保留本地 query 和一段上下文 K/V；K/V 块通过环形 P2P 在各 rank 间轮转，并用在线 softmax 累积完整注意力结果。它避免复制超长上下文，但当前模型已把空间划分为独立 8×8 局部窗口，不需要再跨 GPU 切分上下文。

## 5. 当前模型的选择依据

实测资源构成为：

- 模型约 4.11 M 参数；
- FP32 参数、梯度和 Adam 状态总量不足 0.1 GB；
- 训练峰值显存约 61–76 GB；
- 主要占用来自激活、融合损失分支和 grouped-MoE 容量缓冲；
- 12 个路由专家可以单卡本地执行；
- 每个注意力窗口仅 64 个 token。

因此当前推荐组合是 **DDP + 异步分桶重叠 + bucket view + static graph + 8 MiB 两桶 + fused Adam**。模型状态、张量、流水线、专家和长上下文并行都在解决当前并不存在的瓶颈，暂不采用。

## 6. 视觉约束

- 十种机制都用四个 GPU/rank 的统一尺度表示；
- 数据、模型状态、Tensor、Pipeline、Expert、Ulysses 和 Context 使用不同颜色；
- All-Reduce、Reduce-Scatter/All-Gather、All-to-All、P2P 和 Ring P2P 在底部提供完整图例；
- 所有常规流程使用直线或正交线，只有 Expert Parallel 的交叉 token 路由刻意保留交叉关系；
- 箭头端点在模块边界前退让，不穿过模块文字；
- 全部英文使用 Times New Roman，仅输出 PDF 和 PNG。

## 7. 重新生成

```bash
cd /ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper-md
/ytech_m2v4_hdd/lizhongyin/venv/gifnet/bin/python \
  script/make_distributed_parallelism_figure.py
```

## 8. 建议图题

> **图 4-x　分布式训练机制的切分对象、通信模式及其与 U-MoE-Fusion 的匹配。** 图中比较 DDP、梯度归约重叠、分布式优化器、ZeRO/FSDP、成本感知数据分片、TP、PP、EP、Ulysses Parallelism 和 Context Parallelism。U-MoE-Fusion 的模型状态很小，而峰值显存和步时主要由激活与本地计算主导，因此当前采用 DDP 及其分桶重叠优化；其余并行轴会引入与现有瓶颈不匹配的额外 collective。
