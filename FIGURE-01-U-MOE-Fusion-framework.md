# U-MoE-Fusion 方法框架图

## 1. 产物

本轮在 `Materials/figs/` 生成整套可交付格式：

- `fig_u_moe_fusion_framework.svg`：首选排版源，文字与框线保持矢量、示例图已内嵌；
- `fig_u_moe_fusion_framework.pdf`：论文直接插入版本；
- `fig_u_moe_fusion_framework.png`：高分辨率预览与 Word/PPT 兼容版本；
- 绘图脚本：`script/make_framework_figure.py`。

该图以原论文 `Graduation-Paper/figs/1.pdf` 的“三分支 ACM + Transformer + 输出模块”关系为结构参照，但没有直接复刻旧图；输入范围、Transformer 内部、融合头、训练目标与工程优化均按当前 U-MoE-Fusion v3 实现重新组织。

## 2. 图中信息与实现对应关系

### A. 统一多任务输入

图左以三组真实样本展示同一套权重覆盖的任务：

1. IR–VIS：Visible + Infrared；
2. Medical：PET/SPECT + MRI；
3. Microscopy：GFP + Phase Contrast。

彩色源执行 `RGB → YCbCr`，只将两源亮度 `X=[Y_A,Y_B]` 作为双通道输入；源色度 CbCr 留到推理结束后重组。训练端同时标出 170×170 不缩放裁块、按任务配额平衡以及任务标识 `t`。

### B. U-MoE-Fusion 主干

主干保留 MDFNet 的三分支 ACM/Transformer 组织，并按实际前向过程画成浅层、中层、深层上下文三路。三路特征最终逐元素求和。图中五个创新点与 §4.3 消融编号一致：

| 标记 | 创新点 | 图中位置 | 实现依据 |
|---|---|---|---|
| I1 | MoE 混合专家 FFN | Transformer 与下方放大框 | 1 个常开共享专家 + 12 个路由专家，top-2 稀疏激活 |
| I2 | 决策图融合头 | 右侧橙色模块 | `F_Y=w⊙Y_A+(1−w)⊙Y_B` |
| I3 | 真实窗口注意力 | 每个 Transformer 块左半部 | 8×8 window attention |
| I4 | maxfuse 无监督目标 | 右下训练目标框 | SSIM-to-max、max intensity、joint gradient、RMI content |
| I5 | 任务条件路由 | Task ID/embedding 至各 MoE 的虚线 | token 特征与任务嵌入共同决定专家选择 |

MoE 放大框区分了两条计算路径：共享专家始终参与，softmax 路由器仅选择 12 个路由专家中的 top-2；路由负载由 `L_balance` 约束。该表示对应最终 W96L 配置，不把实验过但未采用的 DeepSeek 路由、INN detail head 或 per-task head 画入主图。

### C. 融合与重建

三路特征相加后，由 1×1 卷积与 sigmoid 预测像素级决策图 `w`，再对两路亮度做凸组合。IR–VIS 输出亮度图；医学与 GFP–PC 将融合亮度与彩色源 CbCr 重组为 RGB。右侧三张真实输出与左侧任务一一对应。

### 高效训练条带

底部黄色条带单独标出 grouped-capacity MoE dispatch、fused SDPA、`torch.compile` 与 DDP overlap/rank balancing。它们属于 §4.5 的训练实现优化，不改变上方推理语义，因此没有画成新的网络分支。

## 3. 最终模型配置口径

图中采用与 `models/W_96d4L/args.txt`、`content/section-setup.md` 和 `content/section-ablation.md` 一致的核心口径：

- 一套共享权重覆盖 IR–VIS、Medical、GFP–PC；
- `out_channel=96`，`depth=4`，`window_size=8`；
- 1 个共享专家、12 个路由专家、top-2；
- softmax 路由 + `aux_weight=0.01`；
- decision-map blend head；
- maxfuse 损失。

## 4. 重新生成

```bash
cd /ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper-md
/ytech_m2v4_hdd/lizhongyin/venv/gifnet/bin/python \
  script/make_framework_figure.py
```

脚本默认读取论文实验使用的三组代表样本；可用 `--code-root`、`--data-root` 和 `--output-dir` 改路径。生成后的 SVG/PDF 已内嵌示例图，不依赖外部图片即可移动或排版。

## 5. 建议图题

> **图 3-x　U-MoE-Fusion 统一多模态图像融合方法总体框架。** 三类任务经统一亮度契约进入共享多分支主干；任务条件路由在每个 Transformer 块内激活共享专家与 top-2 路由专家；像素级决策图完成双源凸组合，彩色任务再重组源色度。下方为训练目标与不改变推理语义的工程加速路径。
