# Materials/ —— 论文用图素材

存放毕业论文正文各章节的图片素材（成品拼图 + 可自排的单图）。绘图脚本在仓库 `script/` 下。

## 目录规划

```
Materials/
├── comparison/          §4.2 与 SOTA 的定性对比图（本批已产出）
│   ├── irvis/
│   │   ├── individual/            # 单图（带红框+局部放大），供 PPT 自行排版
│   │   │   └── <样本>__<方法>.png  # 如 00004N__Ours.png、00004N__Visible.png
│   │   └── fig_irvis_qualitative.png   # 脚本自动拼好的成品图（可直接进论文/PPT）
│   ├── medical/  （同上，彩色）
│   └── gfp_pc/   （同上，彩色）
├── ablation/            预留：消融实验图（待产出）
└── hyperparam/          预留：超参实验图（待产出）
```

命名约定（后续各章沿用）：`<类别>/<子项>/{individual/, fig_<子项>_*.png}`。类别用途固定为 `comparison`（对比）、`ablation`（消融）、`hyperparam`（超参）。

## 本批对比图说明（comparison/）

- 风格对标参考论文：每个面板在**同一关键区域**画红框，并在角落贴红边**局部放大子图**。
- 每张成品图为一行 12 面板：`源A | 源B | LP | NSCT | TarDAL | DATFuse | LRRNet | DDFM | MURF | EMMA | GIFNet | Ours`，Ours 标题红色高亮。
- 数据来源（与评测协议一致）：irvis 取 `fusion_bench/fused/`（灰度）；medical/gfp_pc 取 `fusion_bench/fused_final/`（RGB 彩色重组）。**Ours(v3) = 文件夹 `W96L`**。
- 当前样本：irvis=`00004N`（夜景，行人热目标）、medical=`pet_25027`、gfp_pc=`05-A06`。

## 如何重新生成 / 换样本 / 调红框

脚本：`script/make_qualitative_figure.py`（用带 matplotlib 的 venv，如 `/ytech_m2v4_hdd/lizhongyin/venv/gifnet/bin/python`）。

```bash
PY=/ytech_m2v4_hdd/lizhongyin/venv/gifnet/bin/python
cd script
# --box 为关键区域的相对坐标 x y w h；--corner 放大子图所在角；--ncols 每行面板数
$PY make_qualitative_figure.py --task irvis   --sample 00004N   --box 0.35 0.40 0.28 0.28 --corner br
$PY make_qualitative_figure.py --task medical --sample pet_25027 --box 0.35 0.35 0.30 0.30 --corner br
$PY make_qualitative_figure.py --task gfp_pc  --sample 05-A06    --box 0.35 0.35 0.30 0.30 --corner br
```

换样本：`--sample <stem>`（可选样本见 `fusion_bench/inputs/<task>/A/`）。同时会刷新 `individual/` 下的单图与成品拼图。
