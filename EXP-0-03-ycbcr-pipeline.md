# EXP-0-03：补全 YCbCr 颜色流水线（RGB 任务的两次转换）

- 日期：2026-06-22
- 所属阶段：阶段 0（评测协议/推理流水线修正，承用户补充说明）
- 结果级别：S1（probe 15）
- 关联代码：`ycbcr.py`（新增），`infer_fusion.py`（改为 YCbCr-aware、任务自适应出图），`configs/gfp_pc.json`（加 `output_mode`），`eval_fusion.py`（优先用精确的 `_Y` 目录算指标）

## 1. 用户补充的评判/处理标准
- 网络对 **RGB 输入**先做 `RGB→YCbCr`，**只取 Y 通道**喂网络（如 GFP 取其 Y）。
- 网络输出只是"融合后的 Y"，**不是最终融合图**；要把融合 Y 与**原始 CbCr** 重新组合再 `逆变换→RGB`，才是最终融合图。
- **灰度↔灰度**任务（如红外-可见光）**没有这两步**，直接处理。
- 一句话：涉及 RGB 的任务有"前向取 Y" + "回填 CbCr 逆变换"两次转换。

## 2. 数据事实（确认谁带颜色）
GFP(A) 带颜色（chroma std 13~31），**PCI(B) 是纯灰度（chroma=0）**。故最终 RGB 的颜色来自 GFP。CbCr 融合规则照搬 `Utils/.../fusedY2RGB.m`：偏离 128 越远权重越大；灰度源(=128)不贡献，于是 GFP-PC 的结果≈GFP 的 CbCr。

## 3. 本次实现
1. `ycbcr.py`：`load_y`/`load_ycbcr`（BT.601，PIL 'YCbCr'，灰度像素 Y==灰度值，故与原 `Test.py` 的 `convert('L')` 一致）、`fuse_chroma`（偏差加权）、`ycbcr_to_rgb`。
2. `infer_fusion.py` 任务自适应：输入恒为两源的 **Y**；`output_mode=rgb`（GFP-PC）→ 融合 Y + 融合 CbCr → 逆变换 → **RGB 最终图**；`output_mode=gray`（IR-VIS）→ 直接存灰度 Y。无论哪种都另存一份融合 Y 到 `<out>_Y/`，保证 Y 域指标口径一致。
3. `configs/gfp_pc.json` 加 `"output_mode":"rgb"`。
4. `eval_fusion.py`：算指标时**优先读 `<dir>_Y`**（精确融合 Y），避免 RGB 往返 ±1 误差。

## 4. 验证（样本 03-C09）
- **RGB 最终图保住了荧光颜色**：R/G/B 均值 59.7/67.5/59.3（绿>红蓝），chroma std=12.1（有色）。✓
- **Y 域指标不变**：新（正规 YCbCr 的 Y）vs 旧（PIL 'L'）融合 Y 平均绝对差 **0.204**（≈0）。✓
- RGB 最终图亮度 vs 融合 Y 差 1.55（YCbCr↔RGB 往返取整），故指标改用 `_Y` 目录读取。
- probe 复评：MDFNet SSIM_hm 0.174 等与 EXP-0-02 一致 → 结论不变。

## 5. 对此前发现的影响（重要）
- **此前所有 Y 域指标与结论仍成立**（指标本就在 Y 上算，数值未变）。
- 但对"GFP 功能信息是否可见"的讨论有修正：**GFP 的功能位置主要由颜色(CbCr)承载，而 CbCr 在最终 RGB 中是原样回填保留的**——也就是说"功能/荧光在哪"在最终图里天然保住了。Y 域指标看不到颜色，所以它们测不到这部分价值是"预期之内"，不是缺陷。
- → 对任务 #7（功能/显著区保留指标）的定位更清楚：应在 **RGB 最终图的颜色/显著区**或 **GFP 高亮区**上度量，而不是只在 Y 上。

## 6. 下一步
- 任务 #7：在 RGB 最终图/GFP 信号区上做"功能保留"度量（颜色保真 + 高亮区结构保留）。
- 多任务接入时：IR-VIS / 多聚焦用 `output_mode=gray`；含彩色的可见光用 `rgb`。

> 备注：对应 MASTER_PLAN §6。评测流水线第 3 次修正；推理产物：`outputs/*_rgb`（最终 RGB）+ `outputs/*_rgb_Y`（评测用 Y）。

---

## 7. 更正（指标口径与 MATLAB 对齐）
查看了 `evaluation/main.m`(及 main2/main3):**MATLAB 是在灰度图上算指标**——读入(可能为 RGB 的)融合图后,`if size(...,3)==3: fused=rgb2gray(fused)`,源图同理,再送 `evalution()`。`rgb2gray = 0.299R+0.587G+0.114B`,即全范围 BT.601 亮度,正好等于 PIL `convert(L)`。

因此把 `eval_fusion.py` 改为:**对 RGB 最终图取 `convert(L)`(=rgb2gray)算指标**,撤销上一轮"优先读网络直出 `_Y`"的做法(那与 MATLAB 口径不符)。

实测差异**不只是取整噪声**:对梯度类指标影响明显,例如 MDFNet `Qabf_hm` 0.120→0.158、`MI_hm` 0.242→0.295、`SCD` 0.921→1.002。原因:`YCbCr→RGB→灰度` 往返在荧光高色度处发生 uint8 截断,恰好改变了功能区的亮度/梯度。结论:**评测必须用"最终 RGB 图的 rgb2gray"口径**,与论文 MATLAB 完全一致;`_Y` 仅作中间产物保留、不用于评分。
