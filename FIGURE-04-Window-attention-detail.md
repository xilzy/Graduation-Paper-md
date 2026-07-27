# U-MoE-Fusion 的 8×8 Window Attention 详细图

## 1. 产物

- `Materials/figs/fig_u_moe_window_attention_detail.pdf`：论文排版用矢量版本；
- `Materials/figs/fig_u_moe_window_attention_detail.png`：高分辨率预览与 Word/PPT 兼容版本；
- `script/make_window_attention_figure.py`：可复现绘图脚本。

该图展开总体框架图 `fig_u_moe_fusion_framework.pdf` 中的 **8×8 Window Attention**，内容依据 `Networks/net_moe.py` 的 `MoETransformerBlock`、`SDPAWindowAttention` 以及 `Networks/net.py` 的窗口划分/还原函数绘制。

## 2. 空间分窗

输入特征先从 `(B,C,H,W)` 重排为 `(B,H,W,C)`，再执行 LayerNorm。最终 W96L 配置中 `C=96`、`window_size=8`。

若 H 或 W 不是 8 的倍数，则只在右侧和下侧做反射填充：

```text
pad_h = (8 - H mod 8) mod 8
pad_w = (8 - W mod 8) mod 8
```

训练裁块为 `170×170`，因此实际注意力计算前填充为 `176×176`。随后划分为 `22×22=484` 个互不重叠的 8×8 窗口；每个窗口展平为 64 个 token：

```text
x_window shape = (B * nW, 64, 96)
```

最终模型使用普通非移位窗口，`shift_size=0`、`mask=None`；图中没有画 Swin 的循环移位或跨窗掩码，因为当前实现并未启用它们。

## 3. Q/K/V 与八头注意力

每个窗口先经过一个共享线性层，将通道从 96 投影为 288，再拆分为 Q、K、V：

```text
Q, K, V shape = (B * nW, 8, 64, 12)
```

其中 `num_heads=8`，每头维度 `d=96/8=12`。每个头只在同一 8×8 窗口内的 64 个 token 之间建立关系，不与其他窗口直接交互。

## 4. 相对位置偏置与 fused SDPA

8×8 窗口的二维相对位移范围为 `[-7,7]×[-7,7]`，因此每个头维护：

```text
(2*8-1) * (2*8-1) = 15 * 15 = 225
```

个可学习偏置。通过 `relative_position_index` 将其展开为每头 `64×64` 的加性偏置矩阵。单头注意力为：

```text
A_h = softmax(Q_h K_h^T / sqrt(12) + B_rel)
O_h = A_h V_h
```

实现使用 `torch.nn.functional.scaled_dot_product_attention`，将缩放、偏置、softmax、dropout 与 `A·V` 融合到 SDPA 内核；在支持的 GPU 上可使用 FlashAttention / memory-efficient 路径。该实现与手写 `QK^T → softmax → AV` 数值语义一致。

## 5. 多头合并、窗口还原与残差

八个头的输出先拼接回每个 token 的 96 维表示，再做线性投影：

```text
(8,64,12) → concat → (64,96) → Linear(96→96)
```

之后把 64 个 token 还原为 8×8 窗口，通过 `window_reverse` 重建填充后的完整特征图；若前面发生过填充，则裁回原始 `H×W`。最后与窗口注意力前的 shortcut 相加：

```text
F_attn = F + WindowAttention(LayerNorm(F))
```

该输出随后进入第二个 LayerNorm 和任务条件 MoE-FFN，后者见 `FIGURE-02-U-MoE-FFN-detail.md`。

## 6. 版式与视觉约束

- 左侧按空间分窗、Q/K/V 与八头注意力、窗口还原与残差三段组织；
- 右侧放大单个 8×8 窗口，展示 64 token、单头 Q/K/V、相对位置偏置和 64×64 注意力矩阵；
- Q、K、V、相对位置偏置、SDPA、窗口与残差使用互不相同的颜色；
- 全部箭头位于模块边框下方并在端点退让，不穿文字或方框；
- 全部英文使用 Times New Roman，输出仅保留 PDF 与 PNG。

## 7. 重新生成

```bash
cd /ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper-md
/ytech_m2v4_hdd/lizhongyin/venv/gifnet/bin/python \
  script/make_window_attention_figure.py
```

## 8. 建议图题

> **图 3-x　U-MoE-Fusion 中 8×8 窗口注意力的详细结构。** 输入特征经归一化和反射填充后被划分为非移位 8×8 窗口，每窗 64 个 token 通过八头 Q/K/V 投影和带相对位置偏置的 fused SDPA 建模局部空间关系；随后多头特征拼接、线性投影、窗口还原并裁回原尺寸，最终通过残差连接输出。
