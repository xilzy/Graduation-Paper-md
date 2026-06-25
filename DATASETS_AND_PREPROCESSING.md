# 三类数据集与预处理学习笔记（IR-VIS / 医学 / GFP-PC）

> 目的：在把 MDFNet 扩展为统一多模态融合框架前，先弄清三类源图像（MSRS 红外-可见光、Harvard AANLIB 脑医学、John Innes GFP-PC 显微）各自**怎么获取、怎么预处理、怎么喂进网络**，为统一数据集方案（见 `EXP-0-06-unified-multitask-dataset.md`）提供依据。
>
> 证据来源：① 直接拉取并阅读各方法源码（`code/ref/` 下 PIAFusion / SwinFusion / U2Fusion / CDDFuse / DDFM / Image-Fusion）；② 一个 deep-research 子代理对论文/仓库的逐条 HTTP 核实。日期 2026-06-24。

---

## 0. 一句话结论（统一预处理共识）

**彩色源 → YCbCr，只融亮度 Y，色度 CbCr 原样保留；融合得到 Y 后再与原 CbCr 拼回、逆变换成 RGB。** 这条"fuse-Y / recombine-CbCr"在 IR-VIS（彩色可见光）、医学（伪彩 PET/SPECT）、显微（彩色 GFP）三域**一致成立**，是社区标准做法（Dif-Fusion、MDC-RHT、IR-VIS NN 综述均有明述；PIAFusion/SeAFusion/SwinFusion/GeSeNet 源码 `RGB2YCrCb`/`ycbcr2rgb` 即如此）。归一化主流为 **[0,1]（/255）**。三类数据集**都已配准**，无需额外对齐。

> 这正是仓库里 `ycbcr.py` + `infer_fusion.py` 已经实现的流程 —— 我们的统一管线与领域共识吻合。

---

## 1. 三类数据集：获取方式与规格

| 任务 | 数据集 | 获取（已落地集群） | 规模 | 分辨率 | 模态 A（彩色/功能） | 模态 B（灰度/结构） |
|---|---|---|---|---|---|---|
| IR-VIS | **MSRS** | 已有 `data/MSRS`（PIAFusion 同源，文件夹式 train/test） | train 1083 / test 361 | 640×480 | 可见光 VIS（RGB） | 红外 IR（L 单通道） |
| 医学 | **Harvard AANLIB**（脑图谱派生子集） | 本次下载 `code/ref/Harvard-AANLIB`（`xianming-gu/Havard-Medical-Image-Fusion-Datasets`，**810 对在仓库内**）→ 整理到 `data/Harvard-Medical` | PET-MRI 269 + SPECT-MRI 357 + CT-MRI 184 = 810 | 256×256 | PET/SPECT 伪彩（RGB，代谢/功能） | MRI（L，结构）；CT 也是 L |
| 显微 | **GFP-PC**（John Innes） | 已有 `source images/GFP-PC` | 148 对（118 train / 30 test） | 358×358 | GFP 绿色荧光（RGB，功能） | 相位衬度 PC（灰度，结构） |

要点：
- **Harvard AANLIB 原站**（`med.harvard.edu/AANLIB/`）是逐切片 GIF 的可浏览图谱，**不是干净的配对集**。社区统一用「预配准、预裁剪到 256×256」的子集：MRI/CT 为 8-bit 灰度，PET/SPECT 为 8-bit RGB 伪彩，**同名文件即配对**。`xianming-gu` 仓库把这 810 对直接提交在库内，一条 `git clone` 即得（无需 Google Drive/百度网盘）。其 `MyDatasets/` 给了官方 train/test 划分。
- 备选医学源（已核实，规模更小，留作交叉验证）：CDDFuse `test_img`（PET/SPECT/CT 共 136 对，本次也已下载）、GeSeNet（每模态 5 对 demo）、EMFusion（zip 内每模态 10 对，仅 test）。
- 我们当前医学任务用 **MRI-PET ∪ MRI-SPECT 的功能伪彩对**（A=PET/SPECT 彩色，B=MRI 灰度），采用仓库官方划分：**train 578 / test 48**。CT-MRI（双灰度、结构-结构）暂留作扩展。

---

## 2. 各方法的预处理配方（源码核实）

### 2.1 IR-VIS（PIAFusion / SeAFusion，MSRS）
- **VIS 作为 Y 融合**：RGB→YCrCb，仅 Y 进网络；Cb/Cr 从可见光原样带出，输出时拼回成彩色。
- **IR 单通道灰度**：`cv2.imread(ir, 0)` / `.convert('L')`。
- **归一化 [0,1]**（`/255`，`ToTensor`），非 [-1,1]。
- **裁块**：PIAFusion 官方 64×64 patch、stride 24；SeAFusion 不裁块、整帧 480×640 训练。
- **配准**：MSRS 已配准（由 MFNet 剔除 125 对错位后得到对齐对），无需 warp。

### 2.2 医学（SwinFusion / MATR / GeSeNet）
- **YCbCr 融 Y、拼回 Cb/Cr→RGB**：GeSeNet `test.py` 是最清晰可引用的范例（`RGB2YCrCb` 融 Y 与灰度 MRI，再 `YCrCb2RGB`）。SwinFusion 用 `rgb2ycbcr(only_y=True)`。
- **分辨率 256×256**；**归一化 [0,1]**（SwinFusion/GeSeNet）；**MATR/DPCN 用 [-1,1]** 是异类（我们统一到 [0,1]）。
- **配对**：同名文件（GeSeNet/SwinFusion）或文件名内编号（MATR `MRI_001↔SPECT_001`）。
- ⚠️ 注意：CDDFuse/SwinFusion/MATR 仓库本身**多为代码 + 少量样例**，完整集需外链；`xianming-gu` 仓库是少见的"图像直接在库内"。

### 2.3 GFP-PC 显微（MDFNet，我们的起点）
- 与医学/IR-VIS **同构**：彩色 GFP→亮度/色度，**只融亮度** Y 与灰度 PC，色度保留→逆变换 RGB。
- 命名小差异：经典 GFP-PC 论文常说 **YUV**（与 YCbCr 实践等价）。绿色荧光的功能/颜色信息在色度（保留），待融合的结构细节在亮度。
- 源：John Innes GFP 库（历史 `data.jic.ac.uk/Gfp/`，现已 CAPTCHA 门控）；我们仓库内已有 148 对 358×358。**无公开 GitHub 镜像**，故沿用仓库内数据。

---

## 3. 对"统一数据集方案"的指导

1. **统一到亮度域训练**：三类源都化为单通道 Y∈[0,1]，网络只学亮度融合（`in_channel=2`，concat Y_a/Y_b）。色彩在推理时按 `ycbcr.py` 的偏差加权规则拼回 —— 训练端无需改网络通道，三任务共用同一契约。
2. **分辨率不一 → 裁不缩放**：358 / 640×480 / 256 三种尺寸。因为梯度损失 `joint_grad` 把归一化写死成 `/170²`，**统一裁 170×170**（不 resize，避免亮度统计被缩放扭曲）；小于 patch 的反射 padding。
3. **任务不均 → 按任务配额平衡**：MSRS 1083 对 vs GFP 118 对，直接网格裁切会让 IR-VIS 淹没其它任务。按 `crops_per_task` 给每个任务相近的裁块数（固定 seed 可复现）。
4. **配准已就绪**：三集都已配准，B 仅在尺寸不一致时双线性对齐到 A，无需配准模块。
5. **颜色/功能轴可迁移**：GFP 荧光显著性（FUNCTION 轴 FuncCorr/FuncSal）可平移到 IR 热目标显著性、PET/SPECT 功能代谢区，三轴评测协议天然通用。
6. **任务相关的强度偏向**是最大跨任务差异：IR-VIS 偏热目标高强度、医学偏功能亮区、GFP 偏功能荧光。这正是 MoE 任务特异路由 / 任务自适应损失（max-intensity）要承载的（见 SURVEY §3、§6.3）。

---

## 4. 参考实现落地清单（`code/ref/`）

| 仓库 | 用途 | 关键路径 |
|---|---|---|
| `Harvard-AANLIB`（xianming-gu） | **医学配对数据源**（810 对，含官方 train/test 划分） | `MyDatasets/{PET-MRI,SPECT-MRI,CT-MRI}/{train,test}/{MRI,PET/SPECT/CT}` |
| `PIAFusion` | MSRS 同源 + IR-VIS 预处理（RGB2YCbCr、stride24、/255） | `utils.py: RGB2YCbCr/YCbCr2RGB`，`main.py` |
| `SwinFusion` | 统一 IVIF+医学+MEF，YCbCr only_y 原语 | `utils/utils_image.py`，`Model/Medical_Fusion-*` |
| `U2Fusion` | 统一多任务奠基 baseline（必须比它强） | `test_imgs/medical`（4 对样例） |
| `MMIF-CDDFuse` | 共享/特异分解先验 + 医学 test 集（136 对，交叉验证） | `test_img/{MRI_PET,MRI_SPECT,MRI_CT}` |
| `MMIF-DDFM` | 扩散无监督融合（参考） | — |
| `Image-Fusion` | Linfeng-Tang 聚合 + 通用评测指标 | `General Evaluation Metric/` |

> 论文出处与"借什么"已在 `SURVEY_AND_MOE_PLAN.md §5` 列出（TC-MoA / MoE-Fusion / CDDFuse / EMMA / U2Fusion 等）。本笔记聚焦"数据怎么来、怎么处理"，与之互补。
