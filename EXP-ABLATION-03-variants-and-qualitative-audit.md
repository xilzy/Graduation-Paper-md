# EXP-ABLATION-03：消融变体清点 + 定性图去重与 PET–MRI 补图（审计记录）

- 日期：2026-07-07
- 背景：`content/section-ablation.md`（§4.3 消融实验）初版定稿后，核对时发现三个问题：
  1. `fusion_bench/fused_final/` 下的 `ab*` 变体文件夹**多于**表 4-10~4-12 里的 5 个，`abNoMoE_direct / abDirect_orig / abD3 / abDeep` 等是干嘛的、为什么不在表里；
  2. §4.3.2 主观分析选的样本**和 §4.2 对比实验完全一样**（00778N / spect_18017 / 05-A02）；
  3. 医学消融只画了 **SPECT–MRI**，没有 **PET–MRI**（而 §4.2 是 PET、SPECT 各一张）。
- 本文档回答这三个问题，并记录已做的修正。

---

## 1. `fused_final/ab*` 变体清点（回答问题①）

`ab*` 系列共 11 个文件夹，分三类；**只有第一类（5 个单消融 + Full 基线）进 §4.3 主表**，其余两类各有归属，不进主表。

变体来自两个训练脚本 `run_sweep.sh` / `run_finish.sh`：每个变体都在 v3 基线（`W96L` = oc96-depth4 MoE，4.11M）之上**只改被测的那一项**，其余超参全同、同样重训 20 epoch。

| 文件夹 | 训练日志 | 相对 v3 改动（train_moe.py 参数） | 类别 | 进 §4.3 表？ | 归属/记录处 |
|---|---|---|---|---|---|
| `W96L` | logs_W_96d4L | 无（完整 v3 基线） | 基线 | 是（完整 v3） | §4.3 表4-10~12 |
| `abNoMoE` | logs_ab_noMoE | `--n-routed 0` | 单消融 I1（−MoE） | 是 | §4.3 |
| `abDirect` | logs_ab_direct | `--fusion-head direct` | 单消融 I2（−决策图头） | 是 | §4.3 |
| `abWs1` | logs_ab_ws1 | `--window-size 1` | 单消融 I3（−窗口注意力） | 是 | §4.3 |
| `abOrig` | logs_ab_orig | `--loss-mode orig` | 单消融 I4（−maxfuse） | 是 | §4.3 |
| `abNoTC` | logs_ab_notaskcond | `--no-task-cond` | 单消融 I5（−任务条件路由） | 是 | §4.3 |
| `abNoMoE_direct` | logs_ab_noMoE_direct | `--n-routed 0 --fusion-head direct` | **双消融** I1+I2 | 否 | EXP-ABLATION-PARAM-v3 §2.1 |
| `abDirect_orig` | logs_ab_direct_orig | `--fusion-head direct --loss-mode orig` | **双消融** I2+I4 | 否 | EXP-ABLATION-PARAM-v3 §2.1 |
| `abWs1_orig` | logs_ab_ws1_orig | `--window-size 1 --loss-mode orig` | **双消融** I3+I4 | 否 | EXP-ABLATION-PARAM-v3 §2.1 |
| `abD3` | logs_ab_d3 | `--depth 3`（3.10M） | **超参对比点** depth=3 | 否 | EXP-ABLATION-PARAM-v3 §3（骨干深度）·§4.4 |
| `abDeep` | logs_ab_deepseek | 路由改 deepseek（无辅助损失，bias 负载均衡） | **超参对比点** routing=deepseek | 否 | EXP-ABLATION-PARAM-v3 §3（路由方式）·§4.4 |

### 为什么不进 §4.3 主表

- **§4.3 主表按设计只做「单创新点」消融**：每次移除一个创新点、其余全同，这样才能**干净地隔离每个创新点的独立贡献**，读者一眼看清"去掉 I2 掉多少、去掉 I4 掉多少"。这是消融实验的标准做法。
- **双消融（3 组）属于「创新点交互/互补性」分析**：一次去掉两个，用来说明创新点之间**互补、非冗余**（去两个比去一个掉得更多，说明增益叠加、彼此不可替代）。这类结论放在内部记录 `EXP-ABLATION-PARAM-v3.md` §2.1，作为主表的补充证据；正文主表不混入，避免"移除项组合"把单因子结论搅乱。
- **`abD3` / `abDeep` 不是"创新点消融"，而是"超参/架构取值对比"**：`abD3` 是 depth=3 这个**取值**（对照 depth=2/4/5），`abDeep` 是 routing=deepseek 这种**路由方式**（对照 softmax）。它们回答的是"参数取多少最好 / 哪种路由更好"，属于 §4.4 超参数分析，与"某个创新点有没有用"是两码事。§4.3 开头已声明"创新点消融与超参取值分析二者不混"。
- 补充：这两个是**探索期**（7-02）先跑的，正式超参扫（`run_sweep.sh`，7-04）里 depth 只补了 hpD2/hpD5、routing 复用 abDeep，所以 depth=3 直接复用了 `abD3`——它们都已 benchmark（`reports/*/abD3__means.csv`、`abDeep__means.csv`），数据进了 §3 的深度表与路由表，不是"废弃跑"。

> 一句话结论：`fused_final/ab*` 里没有多余/无用的文件夹——5 个进主表（单消融），3 个是双消融（互补性分析，内部文档），2 个是超参对比点（depth=3 / deepseek，§4.4）。

（另：`fused_final/` 里还有大量**非 `ab` 前缀**的文件夹，如 `hp*` 是 §4.4 的 8 组超参扫描、`D*/W*/TG*/IR*` 等是更早期的探索批次与 9 个 SOTA 对比方法目录，均不属于 §4.3 消融。）

---

## 2. 定性样本去重（回答问题②）

**问题**：§4.3.2 消融定性图原本用的样本与 §4.2 对比实验完全相同，读者会在两节看到同一张图。

**修正**：§4.3.2 四张图**全部另选**与 §4.2 不同的样本。选样沿用 `script/select_samples.py` 的口径——挑本文 5 项指标（MI/SSIM/Qabf/VIF↑、Nabf↓）**优于全部 9 个对比方法**的代表图（保证"完整 v3"确实是强代表，这样移除创新点后的退化才看得清），取其中**排名靠前但未被 §4.2 占用**的那一张。

| 图 | §4.2 对比用样本 | §4.3 消融用样本（改后） | 选取依据 |
|---|---|---|---|
| IR-VIS | 00778N | **01506D** | 全指标全胜候选 #1（§4.2 未用） |
| 医学 PET–MRI | pet_25027 | **pet_25015** | PET 无 5/5 全胜图，取 4/5（仅 Nabf 略输）最优候选，§4.2 未用 |
| 医学 SPECT–MRI | spect_18017 | **spect_4010** | 全指标全胜候选 #2（#1 为 §4.2 用的 spect_18017） |
| 显微 GFP–PC | 05-A02 | **05-B06** | 全指标全胜候选 #2（#1 为 §4.2 用的 05-A02） |

---

## 3. 医学 PET–MRI 消融补图（回答问题③）

**问题**：§4.2 医学分 PET–MRI、SPECT–MRI 各一张；§4.3.2 却只画了 SPECT–MRI，缺 PET–MRI。

**原因**：初版 `make_ablation_figure.py` 对 medical 只生成单张 `fig_medical_ablation.png`（当时只跑了 spect_18017 一个样本），没有像对比脚本那样按模态分图；并非数据缺失——**6 个单消融变体文件夹里 PET 融合结果齐全**（`W96L/abNoMoE/abDirect/abWs1/abOrig/abNoTC` 各有 24 张 PET + 24 张 SPECT）。

**修正**：给 `make_ablation_figure.py` 加了 `--subtag pet|spect`（与 `make_qualitative_figure.py` 一致）：medical 按模态分别输出到 `Materials/ablation/medical/{pet,spect}/fig_medical_{pet,spect}_ablation.png`，源 A 标签自动为 PET / SPECT。据此**补出 PET–MRI 消融图**（pet_25015），与 SPECT–MRI（spect_4010）并列，和 §4.2 的呈现方式一致。

---

## 4. 本次改动清单
- `script/make_ablation_figure.py`：新增 `--subtag`，medical 按 PET/SPECT 分图。
- `Materials/ablation/`：重生成 4 张图——`irvis/fig_irvis_ablation.png`(01506D)、`medical/pet/fig_medical_pet_ablation.png`(pet_25015，新增)、`medical/spect/fig_medical_spect_ablation.png`(spect_4010)、`gfp_pc/fig_gfp_pc_ablation.png`(05-B06)；删除旧的 `medical/fig_medical_ablation.png` 及被替换样本的 individual 面板。
- `content/section-ablation.md`：§4.3.2 改为 4 图（新增 PET–MRI）、样本改为去重后的新样本、图号 4-5~4-8；表 4-9 下新增「消融范围说明」直接解释 fused_final 里多出来的变体各是什么、为什么不进主表。
- `Materials/README.md`：ablation 目录树补 pet/spect 子目录，重生成命令改为新样本 + `--subtag`，并注明样本与 §4.2 去重、双消融/超参见 EXP-ABLATION-PARAM-v3。

---

## 5. 为什么 5 个单消融只组合出 3 组双消融（设计考量）

5 个创新点两两组合共 C(5,2)=10 种。**双消融不是要穷举交互网格，只是要证明一件事：去掉两个比去掉一个掉得更多（超可加退化）→ 创新点互补、彼此不可替代。** 这个结论用少数几组有代表性的组合就够了，跑满 10 组（每组都是三模态 20-epoch 重训）性价比很低。据此只挑了 3 组，规则如下：

| 组合 | 跑？ | 文件夹 | 参数叠加 | 选它的理由 |
|---|---|---|---|---|
| I1+I2（−MoE −决策图头） | ✅ | `abNoMoE_direct` | `--n-routed 0 --fusion-head direct` | 容量(MoE) × 融合机制(决策图头) 是否各自独立起效 |
| I2+I4（−决策图头 −maxfuse） | ✅ | `abDirect_orig` | `--fusion-head direct --loss-mode orig` | **两大主导机制**互测：证明它俩不是做同一件事 |
| I3+I4（−窗口 −maxfuse） | ✅ | `abWs1_orig` | `--window-size 1 --loss-mode orig` | 空间上下文(窗口) × 细节对齐(maxfuse) |
| I1+I5（−MoE −任务条件） | ❌ | — | — | **逻辑冗余**：`--n-routed 0` 已让路由失效，任务条件路由无从存在，等于 ≈ 单独 −MoE |
| I1+I3 / I1+I4 / I2+I3 / I2+I5 / I3+I5 / I4+I5 | ❌ | — | — | 边际信息低，不改变「互补非冗余」结论，省算力 |

设计上的三条取舍原则：

1. **围绕两个"主导机制"展开。** 单消融保持度排序里 I2 决策图头（14→4）、I4 maxfuse（14→9）掉得最狠，是承重构件。3 组双消融正是以它俩为中心：`--fusion-head direct`(I2) 与 `--loss-mode orig`(I4) 各出现在 2 组里；`I2+I4` 还把两者直接对撞，验证「最强的两个也不重叠」。
2. **每个"独立结构件"覆盖一次即可。** I1(MoE 容量) 与 I3(窗口注意力) 各与一个主导机制配一次（I1+I2、I3+I4），足以看清它们与主导机制的叠加效应；再多配对只是重复同一类证据。
3. **排除不独立/低价值的组合。** I5(任务条件路由) 是 MoE 路由的**子机制**（代码里 `task_cond` 只作用于路由器，`--n-routed 0` 时路由器不存在），所以任何含 I5 的双消融要么冗余(I1+I5)、要么价值低(I5 单删本就最温和、只是 MoE 内部精修)，一律不做。

**构造方式印证了这套逻辑**：3 组双消融的命名就是「已有单消融 + 再叠一个主导删除」——`abNoMoE`→`abNoMoE_direct`、`abDirect`→`abDirect_orig`、`abWs1`→`abWs1_orig`（7-04 那一波在 7-02 单消融基础上追加 `direct`/`orig`），是一次**定向补充**而非组合扫描。

> 一句话：双消融的目的是"证明互补非冗余"，不是"画满交互矩阵"；因此只需围绕两个主导机制(I2/I4)、每个独立创新点覆盖一次、剔除含 I5 的冗余组合，3 组就足以支撑结论，10 组是浪费。若答辩需要更完整，可再补 I1+I4、I2+I3 等，但预期只会进一步强化同一结论。
