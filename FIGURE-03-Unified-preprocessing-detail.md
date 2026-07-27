# U-MoE-Fusion 统一预处理详细图

## 1. 产物

- `Materials/figs/fig_u_moe_unified_preprocessing_detail.pdf`：论文排版用矢量版本；
- `Materials/figs/fig_u_moe_unified_preprocessing_detail.png`：高分辨率预览与 Word/PPT 兼容版本；
- `script/make_unified_preprocessing_figure.py`：可复现绘图脚本。

该图展开总体框架图 `fig_u_moe_fusion_framework.pdf` 中的 **Unified Preprocessing** 模块，内容依据 `mm_fusion_data.py`、`mm_fusion_dataset.py`、`ycbcr.py` 与 `infer_fusion.py` 的实际数据路径绘制。

## 2. 三任务如何统一为同一输入契约

### A. 成对输入与任务标识

三类任务均组织为有序源对：

| 任务 | Source A（彩色源） | Source B（灰度源） | task_id |
|---|---|---|---:|
| IR–VIS | Visible RGB | Infrared | 0 |
| Medical | PET / SPECT | MRI | 1 |
| Microscopy | GFP | Phase Contrast | 2 |

`mm_fusion_data.py` 支持两种配对方式：MSRS/Medical 通过文件夹内相同文件名配对，GFP–PC 通过去除源后缀后的共同 stem 配对。配对后始终保留 A/B 的源顺序，并同时返回 `task_id`。

### B. 彩色与灰度源的统一亮度表示

彩色源通过 PIL 的 BT.601 全范围 YCbCr 转换提取亮度：

```text
Y = 0.299 R + 0.587 G + 0.114 B
```

灰度源直接使用灰度值作为 Y；在需要色度重组时，其 `Cb=Cr=128`，即中性色度。两个 Y 通道都除以 255，归一化到 `[0,1]`。因此网络只看到统一的亮度表示，而不会因输入原本是 RGB 或灰度而改变接口。

### C. 尺寸对齐

加载 B 时，以 A 的高宽为参照；若形状不同，则仅将 B 双线性对齐到 A。推理时 B 的 Cb/Cr 使用同样的目标尺寸。已配准且尺寸一致的源对不会经过该缩放步骤。

## 3. 训练与推理分流

### 训练路径

1. 若任一边小于 170，则反射填充到至少 170；
2. A/B 使用完全相同的 `(top,left)` 坐标裁取 `170×170`；
3. 不对裁块做额外缩放，避免破坏亮度与梯度统计；
4. 每个任务约取 4000 个裁块，并使用固定随机种子建立任务平衡索引，避免大数据集淹没小数据集。

训练输入为：

```text
X_train = concat(Y_A, Y_B),  shape = 2 x 170 x 170
```

### 推理路径

测试/探测阶段保留对齐后的完整图像，不裁块，批大小为 1：

```text
X_infer = concat(Y_A, Y_B),  shape = 2 x H x W
```

训练和推理均同时向模型提供 `task_id=t`；三任务共用同一套 U-MoE 参数。

## 4. 仅推理使用的色度旁路

Cb/Cr 不进入主干。对于 RGB 输出任务，推理阶段先按色度偏离中性值 128 的幅度融合两个源的色度：

```text
c_f = [c_A |c_A-128| + c_B |c_B-128|]
      / [|c_A-128| + |c_B-128|]
```

当某个源为灰度图时，其色度等于 128，因此对最终颜色不产生贡献。得到模型输出的 fused Y 后：

- RGB 任务：`YCbCr(fused Y, Cb_f, Cr_f) → RGB`；
- 灰度任务：直接保存 fused Y。

评价指标始终可在单独保存的 fused-Y 域计算，不受最终显示颜色影响。

## 5. 版式与视觉约束

- 四个主分区依次为异构成对输入、共享亮度契约、对齐与采样、统一模型接口；
- RGB、灰度、Y、Cb/Cr、训练、推理和模型契约使用不同颜色；
- 色度旁路沿主面板底部独立布线，不与亮度主路径混淆；
- 训练和推理路径分别以绿色和蓝色标识，并在统一输入框中注明各自尺寸；
- 箭头位于模块边框下方并在端点退让，不穿过文字或方框；
- 全部英文使用 Times New Roman，输出仅保留 PDF 与 PNG。

## 6. 重新生成

```bash
cd /ytech_m2v4_hdd/lizhongyin/code/Graduation-Paper-md
/ytech_m2v4_hdd/lizhongyin/venv/gifnet/bin/python \
  script/make_unified_preprocessing_figure.py
```

## 7. 建议图题

> **图 3-x　U-MoE-Fusion 的统一多任务亮度预处理。** 三类异构源对经 stem 配对、BT.601 亮度提取、尺寸对齐和归一化后，统一拼接为双通道 Y 域输入；训练阶段采用同坐标 170×170 裁块与任务平衡采样，推理阶段保留完整分辨率。源 Cb/Cr 仅通过旁路参与最终颜色重组，不进入共享主干。
