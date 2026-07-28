# U-MoE-Fusion 公式汇总与源码对应

本文档集中整理 U-MoE-Fusion 最终 W96L 模型在论文方法、训练、分布式实现和实验评价中需要使用的公式。公式以当前源码和最终配置 `models/W_96d4L/args.txt` 为最高优先级，并参考旧论文 `initial_paper.tex` 的组织方式重新书写；旧论文中已经被决策图融合头、任务条件 MoE、真实窗口注意力或 maxfuse 损失替代的公式不直接沿用。

为便于后续移入论文正文，公式按“P：预处理、A：主干与 ACM、W：窗口注意力、M：MoE、H：融合头、L：损失、D：优化与分布式、E：评价指标”编号。正式排版时可再统一替换为章节连续编号。

## 1　符号约定与最终配置

| 符号 | 含义 | 最终取值或形状 |
|---|---|---|
| $I_A,I_B$ | Source A、Source B 的归一化亮度图 | $[0,1]^{H\times W}$ |
| $I_F$ | 网络输出的融合亮度图 | $[0,1]^{H\times W}$ |
| $t$ | 任务编号 | 0：GFP–PC；1：IR–VIS；2：Medical |
| $N_b$ | batch size | 单卡训练为 10 |
| $P_c$ | 训练裁块边长 | 170 |
| $N_g$ | DDP rank/GPU 数 | 1、2、4 或 8 |
| $C$ | 主干特征通道数 | 96 |
| $M$ | 窗口边长 | 8 |
| $N=M^2$ | 每个窗口的 token 数 | 64 |
| $n_h$ | 注意力头数 | 8 |
| $d=C/n_h$ | 单头维度 | 12 |
| $E$ | 路由专家数 | 12 |
| $k$ | 每个 token 激活的路由专家数 | 2 |
| $n_s$ | 常开共享专家数 | 1 |
| $L$ | 每条尺度路径中的 Transformer 块数 | 4 |
| $\rho$ | grouped-capacity 容量因子 | 1.25（仅效率实验性能档） |
| $\odot$ | 逐元素乘法 | — |

最终模型采用 `fusion_head=blend`、`res_scale=0`、`routing=softmax`、`out_scale=True`、`loss_mode=maxfuse` 和 `ssim_target=max`。因此正文主公式应描述“纯决策图凸组合、softmax top-2 路由、MoE 输出缩放和逐像素最大值监督”，不应混入 DeepSeek 路由、INN detail head、per-task head 或多尺度 Sobel 辅助项等未进入最终配置的实验分支。

任务编号由最终参数文件中的配置顺序确定：

$$
t=0:\mathrm{GFP\!-\!PC},\qquad
t=1:\mathrm{IR\!-\!VIS},\qquad
t=2:\mathrm{Medical}.
$$

W96L 普通训练入口没有传入 `attn_impl` 或调用 `set_combine`，因此该 checkpoint 的可证实路径是 **vanilla attention + sparse MoE dispatch**。SDPA 与 grouped-capacity 是后续效率实验中的可选执行优化，应单独陈述，不能写成 W96L 权重训练时已启用的默认组件。

## 2　统一预处理与颜色重建

### 2.1　彩色源的 BT.601 全范围 YCbCr 变换

当前数据管线通过 PIL 的 `convert("YCbCr")` 处理彩色源，对应 JPEG/BT.601 全范围变换。对 8-bit RGB 像素，前向变换写为：

**式（P-1）　RGB 到 YCbCr**

$$
\begin{bmatrix}
Y\\ Cb\\ Cr
\end{bmatrix}
=
\begin{bmatrix}
0.299000 & 0.587000 & 0.114000\\
-0.168736 & -0.331264 & 0.500000\\
0.500000 & -0.418688 & -0.081312
\end{bmatrix}
\begin{bmatrix}
R\\G\\B
\end{bmatrix}
+
\begin{bmatrix}
0\\128\\128
\end{bmatrix}.
$$

旧论文使用的是带 $Y$ 偏置 16 的视频限幅范围写法；当前源码使用全范围 PIL 变换，二者系数和偏置不同。旧式中的绿色通道系数还写成了 0.564，而标准 BT.601 limited-range 通常约为 0.504，因此旧矩阵本身也不应作为标准公式引用。新论文应采用式（P-1），不能直接复制旧论文的 $0.257/0.564/0.098$ 矩阵。

对灰度源，不引入虚假颜色，直接令：

**式（P-2）　灰度源的统一亮度与中性色度**

$$
Y=I_{\mathrm{gray}},\qquad Cb=Cr=128.
$$

### 2.2　尺寸对齐、归一化和双通道输入

以 Source A 的空间尺寸为基准；当 Source B 尺寸不一致时，对 B 做双线性插值：

**式（P-3）　尺寸对齐**

$$
\widehat Y_B=
\begin{cases}
\mathrm{Resize}_{\mathrm{bilinear}}(Y_B;H_A,W_A),
& (H_B,W_B)\neq(H_A,W_A),\\
Y_B,&\text{otherwise}.
\end{cases}
$$

两路亮度统一除以 255：

**式（P-4）　亮度归一化**

$$
I_A=\frac{Y_A}{255},\qquad
I_B=\frac{\widehat Y_B}{255},\qquad
I_A,I_B\in[0,1].
$$

训练阶段若某一边小于裁块尺寸 $P_c=170$，先在右侧或下侧做反射填充，再从两路图像的同一坐标 $(u,v)$ 裁取区域：

**式（P-5）　同坐标裁块**

$$
\begin{aligned}
u&\sim\mathcal U\{0,\ldots,H'-P_c\},\\
v&\sim\mathcal U\{0,\ldots,W'-P_c\},\\
I_A^{c}&=I_A'[u:u+P_c,\;v:v+P_c],\\
I_B^{c}&=I_B'[u:u+P_c,\;v:v+P_c].
\end{aligned}
$$

其中 $I_A',I_B'$ 是反射填充后的图像。数据集为每项任务设置名义裁块配额 $C_t=4000$。若任务 $t$ 有 $n_t$ 对训练图像，当前索引构造的精确长度为：

**式（P-6）　按任务配额构造训练索引**

$$
\begin{aligned}
m_t&=\max\!\left(1,\left\lfloor\frac{C_t}{n_t}\right\rfloor\right),\\
N_t&=\min(C_t,n_tm_t),\\
\Pr(t=q)&=\frac{N_q}{\sum_{t=0}^{2}N_t}.
\end{aligned}
$$

该策略使三项任务贡献同量级样本，但在 $C_t$ 不能被 $n_t$ 整除时不会自动补齐余数，因此严格概率不一定恰好为 $1/3$。按文档记录的 118、1083、578 对训练样本估算，三个索引长度分别为 3894、3249、3468；论文宜表述为“按任务近似平衡”，不宜写成完全等量。

网络输入始终是两路亮度的通道拼接：

**式（P-7）　统一模型输入**

$$
\mathbf X=\mathrm{Concat}_{c}(I_A,I_B)
\in\mathbb R^{N_b\times2\times H\times W}.
$$

训练时 $H=W=170$；推理时保持完整分辨率。任务编号 $t$ 与 $\mathbf X$ 一起送入网络，但不作为第三个图像通道。

### 2.3　推理阶段色度融合与 RGB 重建

对 $c\in\{Cb,Cr\}$，先计算两源色度相对中性值 128 的偏离：

**式（P-8）　色度偏离权重**

$$
d_A=|c_A-128|,\qquad d_B=|c_B-128|.
$$

色度按偏离幅度加权；两路都为中性色时返回 128：

**式（P-9）　色度融合**

$$
c_F=
\begin{cases}
128,&d_A+d_B=0,\\[2mm]
\dfrac{c_A d_A+c_B d_B}{d_A+d_B},&d_A+d_B>0.
\end{cases}
$$

因此灰度源 $c=128$ 时权重为 0，不会给结果引入颜色。融合亮度与 $Cb_F,Cr_F$ 组合后，按下式近似逆变换为 RGB：

**式（P-10）　YCbCr 到 RGB**

$$
\begin{aligned}
R_F&=I_F^{255}+1.402(Cr_F-128),\\
G_F&=I_F^{255}-0.344136(Cb_F-128)-0.714136(Cr_F-128),\\
B_F&=I_F^{255}+1.772(Cb_F-128),
\end{aligned}
$$

其中 $I_F^{255}=255I_F$，最终结果裁剪到 $[0,255]$ 并转为 `uint8`。IR–VIS 灰度任务直接保存 $I_F^{255}$，不执行颜色重建。

## 3　任务感知多尺度主干与自适应卷积

### 3.1　ACM 的外层表达

旧论文只给出了 ACM 的整体形式，该式仍适用于当前 `Basic3x3`：

**式（A-1）　自适应卷积模块**

$$
\mathrm{ACM}(\mathbf U)
=\mathrm{ReLU}\!\left(
\mathrm{BN}\!\left(
\mathrm{AC}(\mathbf U)
\right)\right).
$$

这里 $\mathrm{AC}$ 不是固定 3×3 卷积，而是根据当前样本的全局上下文对基础卷积核逐元素门控。

### 3.2　样本自适应动态卷积核

设基础核为
$\mathbf W\in\mathbb R^{C_o\times C_i\times K\times K}$，当前模型取 $K=3$。首先把每个输入通道自适应池化为 $K\times K$ 并展平：

**式（A-2）　全局上下文描述**

$$
\mathbf g_{b,c}
=\mathrm{vec}\!\left(
\mathrm{AAP}_{K\times K}(\mathbf U_{b,c})
\right)\in\mathbb R^{K^2}.
$$

上下文编码层把 $K^2$ 维描述压缩到
$m=\lfloor K^2/2\rfloor+1$ 维；对 $K=3$，$m=5$：

**式（A-3）　潜在上下文编码**

$$
\mathbf z_{b,c}
=\mathrm{CE}(\mathbf g_{b,c})\in\mathbb R^m.
$$

源码分别生成输入通道相关项 $\mathbf u_{b,c}$ 与输出通道相关项 $\mathbf v_{b,o}$：

**式（A-4）　输入/输出通道门控分量**

$$
\begin{aligned}
\mathbf u_{b,c}
&=\mathrm{GD}\!\left(
\mathrm{ReLU}(\mathrm{BN}(\mathbf z_{b,c}))
\right)\in\mathbb R^{K^2},\\
\mathbf v_{b,o}
&=\mathrm{GD}_2\!\left(
\mathrm{ReLU}(\mathrm{BN}(\mathrm{CI}(\mathbf Z_b)))
\right)\in\mathbb R^{K^2}.
\end{aligned}
$$

其中 $\mathrm{CI}$ 按输入通道分组建立到输出通道的交互。在 W96L 中，stem 的 $2\to96$ ACM 使用
$\mathrm{Linear}_{2\to96}$；后续 $96\to96$ ACM 将输入划为 6 组，并在各组复用
$\mathrm{Linear}_{16\to16}$。位置 $\delta\in\{1,\ldots,K^2\}$ 的动态门控和样本特异卷积核为：

**式（A-5）　动态核门控**

$$
a_{b,o,c,\delta}
=\sigma(u_{b,c,\delta}+v_{b,o,\delta}),
\qquad
\widetilde W_{b,o,c,\delta}
=a_{b,o,c,\delta}W_{o,c,\delta}.
$$

最终自适应卷积可写为：

**式（A-6）　自适应卷积输出**

$$
\mathrm{AC}(\mathbf U)_{b,o,p}
=\sum_{c=1}^{C_i}
\sum_{\delta\in\Omega_K}
\widetilde W_{b,o,c,\delta}\,
\mathbf U_{b,c,p+\delta},
$$

其中 $\Omega_K$ 是 $K\times K$ 邻域。与普通卷积相比，基础权重 $\mathbf W$ 仍然共享，但门控 $a_{b,o,c,\delta}$ 随样本和全局上下文改变；同一样本生成的动态核在全部空间位置 $p$ 上共享。

### 3.3　任务偏置、三尺度路径与特征汇聚

双通道输入经 stem ACM 映射到 $C=96$ 通道，并加入任务相关通道偏置：

**式（A-7）　任务感知 stem**

$$
\mathbf F_0
=\mathrm{ACM}_{\mathrm{in}}(\mathbf X)
+\mathbf b_t^{\mathrm{stem}},
\qquad
\mathbf b_t^{\mathrm{stem}}\in\mathbb R^{C\times1\times1}.
$$

$\mathbf b_t^{\mathrm{stem}}$ 与后文路由嵌入 $\mathbf e_t$ 是两张独立的可学习任务查找表。卷积、注意力、专家集合和融合头在任务间共享，但这两张小型 embedding 表包含任务专属参数，因此“一套共享模型”不等于“所有参数均与任务编号无关”。

对尺度路径 $s\in\{1,2,3\}$，源码递归增加 ACM 深度，再调用同一组四层 Transformer 权重：

**式（A-8）　三尺度特征递推**

$$
\begin{aligned}
\mathbf C_s
&=\mathrm{ACM}(\mathbf C_{s-1}),
\qquad \mathbf C_0=\mathbf F_0,\\
\mathbf T_s
&=\mathcal T^{(L=4)}(\mathbf C_s;\mathbf e_t),\\
\mathbf O_s
&=\mathrm{ACM}(\mathbf T_s).
\end{aligned}
$$

当前实现中的 `self.conv` 和 `self.basicLayer` 在三个尺度调用之间共享参数；尺度差异来自前置 ACM 递推深度和不同输入特征，而不是三套独立权重。三路输出在特征域逐元素相加：

**式（A-9）　多尺度特征汇聚**

$$
\mathbf F_{\Sigma}
=\mathbf O_1+\mathbf O_2+\mathbf O_3.
$$

## 4　源码布局下的 8×8 非移位窗口注意力

### 4.1　BCHW 到 BHWC 的源码实际布局

这一处必须区分“论文意图”和“当前实现”。对连续存储的
$\mathbf F\in\mathbb R^{N_b\times C\times H\times W}$，
`MoETransformerBlock.forward` 使用 `view(B,H,W,C)`，而不是
`permute(0,2,3,1)`：

**式（W-10）　源码的线性布局重解释**

$$
\mathbf X^{\mathrm{view}}
=\mathrm{view}_{N_b,H,W,C}(\mathbf F),
$$

其元素由相同的线性存储下标对应：

$$
(cH+h)W+w
=
(\widehat hW+\widehat w)C+\widehat c.
$$

因此一般有
$\mathbf X^{\mathrm{view}}_{b,\widehat h,\widehat w,\widehat c}
\neq\mathbf F_{b,\widehat c,\widehat h,\widehat w}$。
严格的语义转置本应满足
$\mathbf X^{\mathrm{spatial}}_{b,h,w,c}=\mathbf F_{b,c,h,w}$，
这需要 `permute`。所以下文的窗口尺寸、Q/K/V 和相对位置公式与源码一致，但其中的“空间坐标”属于
$\mathbf X^{\mathrm{view}}$ 的重解释网格，不能直接宣称为原始特征图上的真实 8×8 邻域。若后续修复实现为显式 `permute`，数学式保持不变，只需把
$\mathbf X^{\mathrm{view}}$ 替换为
$\mathbf X^{\mathrm{spatial}}$。

### 4.2　反射填充与窗口划分

对任一位置的通道向量 $\mathbf x\in\mathbb R^C$，窗口注意力和 MoE 前使用的 LayerNorm 为：

**式（W-0）　LayerNorm**

$$
\mu(\mathbf x)=\frac1C\sum_{c=1}^{C}x_c,\qquad
\nu(\mathbf x)=\frac1C\sum_{c=1}^{C}(x_c-\mu)^2,
$$

$$
\mathrm{LN}(\mathbf x)
=\gamma\odot
\frac{\mathbf x-\mu(\mathbf x)}
{\sqrt{\nu(\mathbf x)+\epsilon_{\mathrm{LN}}}}
+\beta.
$$

窗口边长为 $M=8$。对输入空间尺寸 $H\times W$，右侧和下侧填充量为：

**式（W-1）　窗口整除填充**

$$
p_h=(M-H\bmod M)\bmod M,\qquad
p_w=(M-W\bmod M)\bmod M.
$$

于是 $H_p=H+p_h,\;W_p=W+p_w$。170×170 训练特征被填充为 176×176。经 LayerNorm 后，特征划分为：

**式（W-2）　窗口 token 张量**

$$
\mathbf X_w
=\mathrm{Partition}_M(\mathrm{LN}(\mathbf X^{\mathrm{view}}))
\in\mathbb R^{
(N_b n_W)\times N\times C},
$$

其中

$$
n_W=\frac{H_pW_p}{M^2},\qquad
N=M^2=64,\qquad C=96.
$$

该模型使用固定非移位窗口，故 `shift_size=0` 且 attention mask 为 `None`。

### 4.3　Q/K/V 投影与多头拆分

每个窗口通过一次线性投影生成 Q、K、V：

**式（W-3）　Q/K/V 投影**

$$
[\mathbf Q,\mathbf K,\mathbf V]
=\mathbf X_w\mathbf W_{qkv}+\mathbf b_{qkv}.
$$

重排后每个张量的形状为：

**式（W-4）　多头张量形状**

$$
\mathbf Q,\mathbf K,\mathbf V
\in\mathbb R^{(N_bn_W)\times n_h\times N\times d},
\qquad
n_h=8,\quad d=\frac{C}{n_h}=12.
$$

### 4.4　二维相对位置偏置

设窗口内 token $p$ 的坐标为 $(r_p,c_p)$，则 token 对 $(p,q)$ 的相对位置索引为：

**式（W-5）　相对位置索引**

$$
\mathrm{idx}(p,q)
=\big(r_p-r_q+M-1\big)(2M-1)
+\big(c_p-c_q+M-1\big).
$$

索引范围为 $0,\ldots,(2M-1)^2-1$。当 $M=8$ 时，共有
$15^2=225$ 种相对位移。设第 $h$ 个头的可学习偏置表为
$\Theta^{(h)}\in\mathbb R^{225}$，则：

**式（W-6）　注意力偏置矩阵**

$$
\mathbf B_{\mathrm{rel},pq}^{(h)}
=\Theta_{\mathrm{idx}(p,q)}^{(h)},
\qquad
\mathbf B_{\mathrm{rel}}^{(h)}
\in\mathbb R^{64\times64}.
$$

### 4.5　注意力计算、可选 Fused SDPA 与残差

每个头的窗口注意力为：

**式（W-7）　带相对位置偏置的窗口注意力**

$$
\mathbf A^{(h)}
=\mathrm{softmax}\!\left(
\frac{\mathbf Q^{(h)}\mathbf K^{(h)\top}}{\sqrt d}
+\mathbf B_{\mathrm{rel}}^{(h)}
\right),
\qquad
\mathbf O^{(h)}=\mathbf A^{(h)}\mathbf V^{(h)}.
$$

W96L 普通训练入口实际调用手写 vanilla 路径。可选的 `scaled_dot_product_attention` 只是融合式（W-7）的缩放、加偏置、softmax、dropout 和 $\mathbf A\mathbf V$ 执行过程，不改变该数学定义；它在后续效率实验中才通过显式参数启用。八头结果拼接并线性投影：

**式（W-8）　多头合并**

$$
\mathbf O_w
=\mathrm{Concat}_{h=1}^{n_h}
\left(\mathbf O^{(h)}\right)\mathbf W_o+\mathbf b_o.
$$

窗口还原并裁去填充后，与注意力前的 shortcut 相加：

**式（W-9）　窗口注意力残差**

$$
\mathbf Z
=\mathbf X^{\mathrm{view}}+
\mathrm{Crop}_{H,W}\!\left[
\mathrm{WindowReverse}_M(\mathbf O_w)
\right].
$$

## 5　任务条件 MoE-FFN

### 5.1　Token 与任务条件

将注意力残差输出 $\mathbf Z$ 经第二个 LayerNorm 并展平空间位置：

**式（M-1）　MoE 输入 token**

$$
\mathbf H
=\mathrm{Flatten}_{HW}(\mathrm{LN}(\mathbf Z))
\in\mathbb R^{N_b\times HW\times C}.
$$

任务编号通过独立于 stem bias 的嵌入表得到
$\mathbf e_t\in\mathbb R^C$，并广播到同一样本的全部 token。第 $j$ 个 token 的路由条件为：

**式（M-2）　任务条件路由输入**

$$
\mathbf z_j=\mathbf h_j+\mathbf e_t.
$$

### 5.2　单个专家

所有共享专家和路由专家使用相同拓扑、独立参数：

**式（M-3）　FFN 专家**

$$
\mathcal E_i(\mathbf h)
=\mathrm{Drop}\!\left(
\mathbf W_{2,i}\,
\mathrm{Drop}\!\left[
\mathrm{GELU}(\mathbf W_{1,i}\mathbf h+\mathbf b_{1,i})
\right]
+\mathbf b_{2,i}
\right),
$$

其中
$\mathbf W_{1,i}:\mathbb R^{C}\rightarrow\mathbb R^{4C}$，
$\mathbf W_{2,i}:\mathbb R^{4C}\rightarrow\mathbb R^{C}$。
最终配置中即 $96\rightarrow384\rightarrow96$，其中
$\mathrm{GELU}(x)=x\Phi(x)$；`drop=0` 时 Dropout 为恒等映射。

### 5.3　Softmax top-2 路由

路由 logits 和完整专家概率为：

**式（M-4）　路由概率**

$$
\ell_j=\mathbf W_g\mathbf z_j,
\qquad
p_{j,i}
=\frac{\exp(\ell_{j,i})}
{\sum_{r=1}^{E}\exp(\ell_{j,r})}.
$$

令
$\mathcal S_j=\mathrm{TopK}(\mathbf p_j,k)$，最终 $k=2$。选中概率重新归一化：

**式（M-5）　Top-k 门控权重**

$$
g_{j,i}
=
\begin{cases}
\dfrac{p_{j,i}}
{\sum_{r\in\mathcal S_j}p_{j,r}+\varepsilon},
&i\in\mathcal S_j,\\[3mm]
0,&i\notin\mathcal S_j.
\end{cases}
$$

### 5.4　共享专家、路由专家与输出缩放

MoE 的逻辑输出由共享路径和 top-2 路由路径相加。由于最终配置启用 `out_scale=True`，源码还除以“共享分支数 + 一个聚合路由分支”：

**式（M-6）　MoE 聚合**

$$
\mathbf m_j
=\frac{
\displaystyle\sum_{q=1}^{n_s}\mathcal E_q^{(s)}(\mathbf h_j)
+
\displaystyle\sum_{i=1}^{E}g_{j,i}\mathcal E_i^{(r)}(\mathbf h_j)
}{n_s+1}.
$$

最终 $n_s=1$，因此：

**式（M-7）　最终配置的 MoE 输出**

$$
\mathbf m_j
=\frac{1}{2}\left[
\mathcal E_s(\mathbf h_j)
+\sum_{i\in\mathcal S_j}
g_{j,i}\mathcal E_i(\mathbf h_j)
\right].
$$

该 $1/2$ 缩放在示意图中被省略，但属于最终 W96L 源码的真实前向过程。

### 5.5　负载均衡辅助损失

源码使用 top-1 选中结果统计离散负载，而不是把 top-2 的两个位置都计入 $f_i$。设当前调用共有 $T=N_bHW$ 个 token：

**式（M-8）　专家负载与平均概率**

$$
f_i=\frac{1}{T}\sum_{j=1}^{T}
\mathbf1\!\left[
i=\arg\max_r p_{j,r}
\right],
\qquad
P_i=\frac{1}{T}\sum_{j=1}^{T}p_{j,i}.
$$

Switch/GShard 风格辅助项为：

**式（M-9）　单层路由负载均衡**

$$
\mathcal L_{\mathrm{bal}}
=E\sum_{i=1}^{E}f_iP_i.
$$

三条尺度路径、每条四个 MoE 块的辅助项在模型前向中求和：

**式（M-10）　全模型路由辅助项**

$$
\mathcal L_{\mathrm{aux}}
=\sum_{s=1}^{3}\sum_{\ell=1}^{4}
\mathcal L_{\mathrm{bal}}^{(s,\ell)}.
$$

三个尺度调用共享同一组块参数，但输入特征不同，因此式（M-10）按调用分别累计。

### 5.6　效率实验中的可选 Grouped-capacity 调度

W96L 普通训练入口保持 `combine="sparse"`；只有显式调用 `set_combine("grouped", cap_factor=...)` 时才执行本节公式。设总 token 数为 $T$，每个 token 产生 $k$ 个 dispatch，故：

**式（M-11）　Dispatch 数量**

$$
D=Tk.
$$

每个专家的固定容量为：

**式（M-12）　专家容量**

$$
\mathrm{cap}
=\max\!\left(
1,\left\lfloor
\rho\frac{Tk}{E}
\right\rfloor
\right),
\qquad \rho=1.25.
$$

按专家排序后，第 $d$ 个 dispatch 仅在其桶内位置
$\mathrm{pos}_d<\mathrm{cap}$ 时保留：

**式（M-13）　容量保留掩码**

$$
\kappa_d
=\mathbf1[
\mathrm{pos}_d<\mathrm{cap}
].
$$

保留 token 被整理为
$\mathbf B_{\mathrm{exp}}\in\mathbb R^{E\times\mathrm{cap}\times C}$，
随后所有专家通过两次 batched GEMM 同时执行。超过容量的 dispatch 被丢弃，保留下来的另一条 top-2 权重不会重新归一化；因此 grouped 路径只在无溢出时与 sparse 路径数值等价，有限容量会对路由输出形成近似。

### 5.7　MoE 残差

MoE 输出恢复空间布局后，与注意力残差输出相加：

**式（M-14）　Transformer 的第二个残差**

$$
\mathbf Y
=\mathbf Z+
\mathrm{Reshape}_{H,W}(\mathbf M),
\qquad
\mathbf F^{+}
=\mathrm{view}_{N_b,C,H,W}(\mathbf Y).
$$

式（W-9）和式（M-14）共同替代旧论文中“MSA 残差 + 普通 MLP 残差”的 Transformer 公式。末端同样使用 `view` 恢复 BCHW；它与式（W-10）的入口重解释互为形状恢复，但不会把窗口内部已经发生的 token 混合改回真正的原图空间窗口。

## 6　像素级决策图融合头

汇聚特征经 1×1 卷积输出两通道
$[\mathbf a,\mathbf r]$：

**式（H-1）　融合头 logits**

$$
[\mathbf a,\mathbf r]
=\mathrm{Conv}_{1\times1}(\mathbf F_{\Sigma}).
$$

第一通道经 sigmoid 得到 Source A 的逐像素权重：

**式（H-2）　决策图**

$$
\mathbf w=\sigma(\mathbf a),\qquad
0\leq w_{ij}\leq1.
$$

源码中 `blend` 分支的一般形式为：

**式（H-3）　带可选残差的融合输出**

$$
\widetilde I_F
=\mathbf w\odot I_A
+(1-\mathbf w)\odot I_B
+s_r\tanh(\mathbf r),
\qquad
I_F=\mathrm{clip}(\widetilde I_F,0,1).
$$

最终 W96L 取 `res_scale` $s_r=0$，所以论文主公式应化简为：

**式（H-4）　最终决策图凸组合**

$$
\boxed{
I_F
=\mathbf w\odot I_A
+(1-\mathbf w)\odot I_B
}.
$$

由于 $I_A,I_B,w\in[0,1]$，式（H-4）天然位于 $[0,1]$，末端 clip 不改变数值。旧论文“1×1 卷积 + tanh 直接回归融合图”的输出公式已经被式（H-4）替代。

## 7　Maxfuse 无监督训练目标

### 7.1　逐像素最大值目标

最终配置对三个任务统一使用逐像素最大亮度：

**式（L-1）　Maxfuse 目标**

$$
I_M=\max(I_A,I_B),
$$

其中 max 为逐像素运算，而不是整幅图的全局最大值。

### 7.2　SSIM 结构损失

对图像 $X,Y$，使用 11×11、$\sigma=1.5$ 的高斯窗且不做边界 padding，计算局部统计量：

**式（L-2）　SSIM 局部统计量**

$$
\begin{aligned}
\mu_X&=G_\sigma*X,&
\mu_Y&=G_\sigma*Y,\\
\sigma_X^2&=G_\sigma*(X^2)-\mu_X^2,&
\sigma_Y^2&=G_\sigma*(Y^2)-\mu_Y^2,\\
\sigma_{XY}&=G_\sigma*(XY)-\mu_X\mu_Y.
\end{aligned}
$$

对归一化亮度 $L_{\mathrm{range}}=1$，有
$C_1=(0.01L_{\mathrm{range}})^2$、
$C_2=(0.03L_{\mathrm{range}})^2$。SSIM 为：

**式（L-3）　结构相似度**

$$
\mathrm{SSIM}(X,Y)
=\frac{(2\mu_X\mu_Y+C_1)(2\sigma_{XY}+C_2)}
{(\mu_X^2+\mu_Y^2+C_1)(\sigma_X^2+\sigma_Y^2+C_2)}.
$$

训练函数 `ssim()` 返回的是 $1-\mathrm{mean}(\mathrm{SSIM})$，因此 maxfuse 的原始 SSIM 损失为：

**式（L-4）　Max-SSIM 损失**

$$
\mathcal L_{\mathrm{SSIM}}
=1-\mathrm{mean}\!\left[
\mathrm{SSIM}(I_F,I_M)
\right].
$$

### 7.3　联合梯度损失

当前损失代码使用如下 3×3 高通核，而不是旧论文文字中笼统的 $\nabla^2$：

**式（L-5）　梯度核**

$$
\mathbf K_g=
\begin{bmatrix}
\frac18&\frac18&\frac18\\
\frac18&-1&\frac18\\
\frac18&\frac18&\frac18
\end{bmatrix},
\qquad
\mathcal G(X)=\mathbf K_g*X.
$$

该训练卷积使用一像素零填充，使梯度图保持 $H\times W$。

两源的联合梯度目标和损失为：

**式（L-6）　联合最大梯度**

$$
\mathcal G_M
=\max\!\left(
|\mathcal G(I_A)|,\,
|\mathcal G(I_B)|
\right),
$$

**式（L-7）　梯度保持损失**

$$
\mathcal L_{\mathrm{Grad}}
=\frac{1}{N_bHW}
\left\|
|\mathcal G(I_F)|-\mathcal G_M
\right\|_1.
$$

源码按实际 $H\times W$ 归一化，不再把分母写死为 $170^2$。

### 7.4　最大强度损失

**式（L-8）　最大强度 MSE**

$$
\mathcal L_{\mathrm{Int}}
=\frac{1}{N_bHW}
\left\|I_F-I_M\right\|_2^2.
$$

这取代旧论文对两源分别计算 MSE 后取平均的对称强度项。

### 7.5　RMI 区域互信息损失

最终源码对 $I_F$ 与每个源分别计算 RMI。默认参数为区域半径 $r=3$、向量维度 $d_r=r^2=9$、max-pool stride 3、BCE 权重 $\gamma=0.5$ 和
$\epsilon=5\times10^{-4}$。

由于当前 `RMILoss(with_logits=True)`，第一输入先按 logit 计算 BCE，并再经过 sigmoid：

**式（L-9）　RMI 的 BCE 分量**

$$
\mathcal L_{\mathrm{BCE}}(F,S)
=-\frac{1}{N_{\mathrm{pix}}}
\sum_n
\left[
S_n\log\sigma(F_n)
+(1-S_n)\log(1-\sigma(F_n))
\right].
$$

其中 $N_{\mathrm{pix}}$ 为参与 BCE 平均的 batch、通道与空间元素总数。

对 max-pool 后的 $\sigma(F)$ 与目标 $S$ 做 3×3 `unfold`，得到预测区域向量
$\mathbf P$ 和目标区域向量 $\mathbf Y$。中心化后构造：

**式（L-10）　RMI 区域向量与协方差项**

$$
\begin{aligned}
\mathcal D(X)
&=\mathrm{MaxPool}_{3\times3,\,s=3,\,p=1}(X),\\
\mathbf P
&=\mathrm{Unfold}_{3\times3}(\mathcal D(\sigma(F))),\\
\mathbf Y
&=\mathrm{Unfold}_{3\times3}(\mathcal D(S)),\\
\widetilde{\mathbf P}&=\mathbf P-\mathrm{mean}_{\mathrm{region}}(\mathbf P),\\
\widetilde{\mathbf Y}&=\mathbf Y-\mathrm{mean}_{\mathrm{region}}(\mathbf Y),\\
\Sigma_{YY}&=\widetilde{\mathbf Y}\widetilde{\mathbf Y}^{\top},\\
\Sigma_{PP}&=\widetilde{\mathbf P}\widetilde{\mathbf P}^{\top},\\
\Sigma_{YP}&=\widetilde{\mathbf Y}\widetilde{\mathbf P}^{\top}.
\end{aligned}
$$

条件协方差近似为：

**式（L-11）　RMI 条件协方差**

$$
\mathbf M
=\Sigma_{YY}
-\Sigma_{YP}
(\Sigma_{PP}+\epsilon\mathbf I)^{-T}
\Sigma_{PY}.
$$

这里的 ${}^{-T}$ 忠实对应源码对逆矩阵再次执行 `transpose`；由于
$\Sigma_{PP}$ 理论上对称，它与通常写法中的逆矩阵等价。

源码通过 Cholesky 分解计算 log-det，区域项可写为：

**式（L-12）　区域互信息下界项**

$$
\mathcal I_l(F,S)
=\frac{1}{2d_r}
\log\det(\mathbf M+\epsilon\mathbf I).
$$

按 batch/通道聚合后，实际 RMI 损失为：

**式（L-13）　源码中的 RMI 损失**

$$
\mathcal L_{\mathrm{RMI}}(F,S)
=\gamma\,\mathcal L_{\mathrm{BCE}}(F,S)
+(1-\gamma)
\left|
\mathrm{mean}\big[\mathcal I_l(F,S)\big]
\right|,
\qquad \gamma=0.5.
$$

训练代码的调用顺序是
$\mathcal L_{\mathrm{RMI}}(I_F,I_A)$ 和
$\mathcal L_{\mathrm{RMI}}(I_F,I_B)$：第一参数是模型输出，第二参数是源图目标。

### 7.6　内容、结构与最终数值权重

保留旧论文“内容项 + 结构项”的外层组织，但内部目标改为 maxfuse：

**式（L-14）　内容损失**

$$
\mathcal L_{\mathrm{content}}
=w_2\mathcal L_{\mathrm{RMI}}(I_F,I_A)
+w_3\mathcal L_{\mathrm{RMI}}(I_F,I_B)
+w_4\mathcal L_{\mathrm{Int}}.
$$

**式（L-15）　结构损失**

$$
\mathcal L_{\mathrm{structure}}
=\frac{w_0+w_1}{2}\mathcal L_{\mathrm{SSIM}}
+w_6\mathcal L_{\mathrm{Grad}}.
$$

完整目标为：

**式（L-16）　总训练损失**

$$
\mathcal L_{\mathrm{total}}
=\mathcal L_{\mathrm{content}}
+\alpha\mathcal L_{\mathrm{structure}}
+\lambda_{\mathrm{aux}}\mathcal L_{\mathrm{aux}}.
$$

最终配置

$$
(w_0,\ldots,w_6)=(2,2,2,2,4,0,3),
\qquad
\alpha=2,
\qquad
\lambda_{\mathrm{aux}}=0.01,
$$

代入后得到论文可直接使用的最终展开式：

**式（L-17）　W96L 最终损失**

$$
\boxed{
\begin{aligned}
\mathcal L_{\mathrm{total}}
={}&2\mathcal L_{\mathrm{RMI}}(I_F,I_A)
+2\mathcal L_{\mathrm{RMI}}(I_F,I_B)\\
&+4\mathcal L_{\mathrm{Int}}
+4\mathcal L_{\mathrm{SSIM}}
+6\mathcal L_{\mathrm{Grad}}
+0.01\mathcal L_{\mathrm{aux}}.
\end{aligned}
}
$$

最终配置中 `ms_grad=0`，所以额外的多尺度 Sobel 损失不进入式（L-17）。

## 8　优化器与分布式训练公式

### 8.1　Adam 与学习率衰减

对第 $n$ 步梯度
$\mathbf g_n=\nabla_\theta\mathcal L_n$，Adam 更新为：

**式（D-1）　Adam 一、二阶矩**

$$
\mathbf m_n
=\beta_1\mathbf m_{n-1}+(1-\beta_1)\mathbf g_n,
\qquad
\mathbf v_n
=\beta_2\mathbf v_{n-1}+(1-\beta_2)\mathbf g_n^2,
$$

**式（D-2）　偏差校正与参数更新**

$$
\widehat{\mathbf m}_n=\frac{\mathbf m_n}{1-\beta_1^n},
\qquad
\widehat{\mathbf v}_n=\frac{\mathbf v_n}{1-\beta_2^n},
\qquad
\theta_n
=\theta_{n-1}
-\eta_n
\frac{\widehat{\mathbf m}_n}
{\sqrt{\widehat{\mathbf v}_n}+\epsilon}.
$$

源码取
$\beta_1=0.9,\beta_2=0.999,\epsilon=10^{-8}$，初始学习率
$\eta_0=10^{-3}$，且不使用 weight decay。效率配置中的 fused Adam 只合并优化器内核，不改变式（D-1）和式（D-2）。每个 epoch 后执行 StepLR：

**式（D-3）　按 epoch 衰减的学习率**

$$
\eta_e=\eta_0\gamma^e,\qquad \gamma=0.8.
$$

最终 W96L 配置文件记录 26 个 epoch。

### 8.2　DDP 梯度同步

设数据并行 world size 为 $N_g$，第 $r$ 个 rank 在本地 mini-batch 上得到梯度
$\mathbf g^{(r)}$。DDP All-Reduce 后每个 rank 使用：

**式（D-4）　数据并行平均梯度**

$$
\overline{\mathbf g}
=\frac{1}{N_g}\sum_{r=1}^{N_g}\mathbf g^{(r)}.
$$

若单卡 batch 为 $B_{\mathrm{local}}=10$，则：

**式（D-5）　全局 batch**

$$
B_{\mathrm{global}}=N_g B_{\mathrm{local}}.
$$

梯度 bucket 就绪后异步发起式（D-4），只改变通信与反向计算的时间重叠，不改变最终平均梯度。
需要注意，打印出的 loss 是 rank 0 本地标量；MoE 的
$f_i,P_i$ 也先在各 rank 本地统计，再通过参数梯度的 DDP 平均产生联合更新，并没有先做跨 rank 的路由计数归约。

### 8.3　吞吐、加速比和扩展效率

设 $N_g$ 卡平均一步耗时为 $T_{N_g}$，每卡 batch 固定为 $B_{\mathrm{local}}$，则：

**式（D-6）　全局吞吐**

$$
Q_{N_g}=\frac{N_g B_{\mathrm{local}}}{T_{N_g}}.
$$

相对单卡的加速比和扩展效率为：

**式（D-7）　多卡扩展效率**

$$
S_{N_g}=\frac{Q_{N_g}}{Q_1},
\qquad
\mathcal E_{N_g}=\frac{S_{N_g}}{N_g}\times100\%.
$$

这里固定每卡 batch=10，随 GPU 数增加而增大全局 batch，因此式（D-7）衡量的是弱扩展效率，不是固定全局 batch 的强扩展效率。

同步训练每步必须等待最慢 rank，临界路径可概括为：

**式（D-8）　Straggler 决定的步时**

$$
T_{\mathrm{step}}
\approx
\max_{r\in\{1,\ldots,N_g\}}T_r
+T_{\mathrm{comm}}^{\mathrm{exposed}},
$$

其中
$T_{\mathrm{comm}}^{\mathrm{exposed}}$
是未被反向计算覆盖的通信时间。

## 9　最终采用的五项评价指标

评价前把源图与融合图统一为 $[0,255]$ 灰度。医学和显微任务先执行式（P-9）和式（P-10），将最终 RGB 结果转灰度后计分；IR–VIS 直接使用融合亮度。除特别说明外，评价实现中的数值稳定项取 $\varepsilon=10^{-10}$。

### 9.1　互信息 MI

MI 先按
$$
Q_8(X)=\mathrm{uint8}\!\left(
\mathrm{clip}(\mathrm{round}X,0,255)
\right)
$$
量化，再由两幅量化图的 256×256 联合直方图估计离散联合概率和边缘概率：

**式（E-1）　两图互信息**

$$
\mathrm{MI}(X,Y)
=\sum_{x,y}p_{XY}(x,y)
\log_2\frac{p_{XY}(x,y)}
{p_X(x)p_Y(y)}
=H(X)+H(Y)-H(X,Y).
$$

融合指标为两源互信息之和：

**式（E-2）　融合 MI**

$$
\mathrm{MI}_{\mathrm{fusion}}
=\mathrm{MI}(I_A,I_F)
+\mathrm{MI}(I_B,I_F).
$$

### 9.2　双源平均 SSIM

单对图像使用式（L-2）和式（L-3）计算 SSIM；实验报告两源平均：

**式（E-3）　融合 SSIM**

$$
\mathrm{SSIM}_{\mathrm{fusion}}
=\frac{
\mathrm{SSIM}(I_F,I_A)
+\mathrm{SSIM}(I_F,I_B)
}{2}.
$$

评价值域为 $[0,255]$，故
$C_1=(0.01\times255)^2$、
$C_2=(0.03\times255)^2$。

### 9.3　梯度转移质量 Qabf

使用 Sobel 算子获得图像 $X$ 的梯度幅值 $g_X$ 和方向 $a_X$：

**式（E-4）　Sobel 梯度幅值与方向**

$$
g_X=\sqrt{G_{x,X}^2+G_{y,X}^2},
\qquad
a_X=\mathrm{atan2}(G_{y,X},G_{x,X}).
$$

对源 $S\in\{A,B\}$ 与融合图 F，幅值和方向保持度为：

**式（E-5）　边缘幅值/方向保持度**

$$
G^{SF}
=
\begin{cases}
\dfrac{g_F}{g_S+\varepsilon},&g_S>g_F,\\[2mm]
\dfrac{g_S}{g_F+\varepsilon},&g_F>g_S,\\[2mm]
1,&g_S=g_F,
\end{cases}
\qquad
A^{SF}
=1-\frac{|a_S-a_F|}{\pi/2}.
$$

通过 Logistic 映射：

**式（E-6）　边缘质量映射**

$$
Q_g^{SF}
=\frac{T_g}{1+\exp[k_g(G^{SF}-D_g)]},
\qquad
Q_a^{SF}
=\frac{T_a}{1+\exp[k_a(A^{SF}-D_a)]},
$$

其中
$(T_g,k_g,D_g)=(0.9994,-15,0.5)$，
$(T_a,k_a,D_a)=(0.9879,-22,0.8)$，且
$Q^{SF}=Q_g^{SF}Q_a^{SF}$。最终：

**式（E-7）　Qabf**

$$
Q_{\mathrm{abf}}
=
\frac{
\sum_{i,j}
\left[
Q^{AF}_{ij}g_{A,ij}
+Q^{BF}_{ij}g_{B,ij}
\right]
}{
\sum_{i,j}(g_{A,ij}+g_{B,ij})+\varepsilon
}.
$$

### 9.4　融合伪影率 Nabf

源码把“融合梯度同时强于两源”的位置视为伪影候选：

**式（E-8）　伪影掩码**

$$
\mathcal A_{ij}
=\mathbf1[
g_{F,ij}>g_{A,ij}
\;\land\;
g_{F,ij}>g_{B,ij}
].
$$

复用 Qabf 的边缘质量后：

**式（E-9）　Nabf**

$$
N_{\mathrm{abf}}
=
\frac{
\sum_{i,j}\mathcal A_{ij}
\left[
(1-Q^{AF}_{ij})g_{A,ij}
+(1-Q^{BF}_{ij})g_{B,ij}
\right]
}{
\sum_{i,j}(g_{A,ij}+g_{B,ij})+\varepsilon
}.
$$

Qabf 越大越好，Nabf 越小越好。

### 9.5　视觉信息保真度 VIF

VIFp 在四个高斯尺度上计算。第 $s$ 个尺度使用
$K_s=2^{5-s}+1$ 和
$\sigma_s=K_s/5$ 的高斯核，$s>1$ 时先滤波并二倍下采样。对参考图 $R$ 与失真/融合图 $D$，局部增益和残差方差为：

**式（E-10）　VIF 局部信号模型**

$$
g_s
=\frac{\sigma_{RD,s}}
{\sigma_{R,s}^2+\varepsilon},
\qquad
\sigma_{v,s}^2
=\max\!\left(
\sigma_{D,s}^2-g_s\sigma_{RD,s},
\varepsilon
\right).
$$

源码还对退化区域做稳定化处理：当
$\sigma_{R,s}^2<\varepsilon$ 或
$\sigma_{D,s}^2<\varepsilon$ 时令 $g_s=0$，并将残差方差限制为至少 $\varepsilon$。

单源 VIFp 为：

**式（E-11）　单源 VIFp**

$$
\mathrm{VIFp}(R,D)
=
\frac{
\displaystyle\sum_{s=1}^{4}\sum_{i,j}
\log_{10}\!\left(
1+\frac{g_{s,ij}^2\sigma_{R,s,ij}^2}
{\sigma_{v,s,ij}^2+\varepsilon}
\right)
}{
\displaystyle\sum_{s=1}^{4}\sum_{i,j}
\log_{10}\!\left(
1+\frac{\sigma_{R,s,ij}^2}{\varepsilon}
\right)
+\varepsilon
}.
$$

融合 VIF 取两源平均：

**式（E-12）　融合 VIF**

$$
\mathrm{VIF}_{\mathrm{fusion}}
=\frac{
\mathrm{VIFp}(I_A,I_F)
+\mathrm{VIFp}(I_B,I_F)
}{2}.
$$

## 10　旧论文公式到新论文公式的替换关系

| 旧论文公式 | 旧含义 | 新论文对应 | 处理方式 |
|---|---|---|---|
| Eq. (1) | 限幅范围 RGB→YCbCr | P-1、P-2、P-9、P-10 | 改为当前 PIL 全范围变换并补全灰度与色度重建 |
| Eq. (2) | `ReLU(BN(AC(In)))` | A-1 至 A-6 | 保留外层式，按源码展开动态核 |
| Eq. (3) | MSA 第一次残差 | W-0 至 W-10 | 展开窗口、相对偏置和注意力，并单列源码 `view` 布局限制 |
| Eq. (4) | 普通 MLP 第二次残差 | M-1 至 M-14 | 替换为任务条件共享+top-2 MoE |
| Eq. (5) | 内容项 + $\alpha$ 结构项 | L-16、L-17 | 保留外层组织并加入 MoE aux |
| Eq. (6)–(9) | 对称 RMI 与平均强度 | L-8 至 L-14 | 改为两源 RMI + 逐像素最大强度 |
| Eq. (10)–(12) | 双源 SSIM 与旧梯度项 | L-2 至 L-7、L-15 | 改为 max-SSIM 和当前高通核联合梯度 |
| 旧输出层 | 1×1 卷积 + tanh 直接回归 | H-1 至 H-4 | 替换为像素级决策图凸组合 |

## 11　源码一致性注意事项

1. **MoE 输出存在 $1/2$ 缩放。** 最终配置 `out_scale=True`，故应采用式（M-7）；仅写“共享专家 + top-2 加权和”会遗漏源码中的幅度归一化。
2. **负载均衡的 $f_i$ 按 top-1 统计。** 路由前向仍执行 top-2，但 `net_moe.py` 的辅助项只用 `topi[:,0]` 统计离散负载，见式（M-8）。
3. **最终 maxfuse 对所有任务统一取逐像素 max。** `task_adaptive=False` 且 `loss_mode=maxfuse`，不是按任务切换平均/最大目标。
4. **最终融合头没有 detail residual。** 虽然 head 生成第二通道，`res_scale=0` 使其不进入输出，正文主公式必须使用式（H-4）。
5. **SDPA 与 grouped 都不是 W96L 普通训练入口的默认路径。** 可选 SDPA 与式（W-7）数学同义；grouped 只有在无容量溢出时才与 sparse 数值等价，溢出 dispatch 被丢弃且剩余权重不重新归一化。
6. **RMI 的实现口径需要在最终定稿前再次确认。** 当前融合输出已位于 $[0,1]$，但 `RMILoss` 默认 `with_logits=True`，会在 BCE 和区域分支中把 $I_F$ 当作 logits 再做 sigmoid。本文档式（L-9）至式（L-13）忠实记录现有代码；若后续将实现改为 `with_logits=False`，这些公式也必须同步修改。
7. **训练轮数以最终配置文件为准。** `W_96d4L/args.txt` 记录 26 epoch，而较早的 `content/section-setup.md` 写作稿仍写 20 epoch；公式 D-3 本身不受影响，但实验设置定稿时应统一。
8. **任务配额是近似平衡而非严格等量。** 当前数据索引按整除后的 `per_pair` 重复样本且不补齐余数，精确分布应使用式（P-6），不宜在论文中直接写成三任务各占 $1/3$。
9. **当前窗口布局不是严格的 BCHW→BHWC 语义转置。** 源码使用式（W-10）的 `view`，因此“原图空间 8×8 窗口”的论文表述与当前执行不完全一致；若要采用该表述，应先把入口和出口改为对应的 `permute` 并重新验证模型。
10. **任务编号必须按最终配置顺序书写。** 当前映射是 GFP–PC=0、IR–VIS=1、Medical=2，不能沿用早期图解中的 IR–VIS=0、Medical=1、Microscopy=2。
11. **任务专属参数不止路由条件。** Stem 后的 $\mathbf b_t^{\mathrm{stem}}$ 与 router embedding $\mathbf e_t$ 是两张独立查找表；“共享网络”应解释为共享主干和专家集合，而不是全部参数对任务编号完全无关。

## 12　主要源码依据

| 公式范围 | 主要依据 |
|---|---|
| P-1 至 P-10 | `mm_fusion_dataset.py`、`ycbcr.py`、`FIGURE-03-Unified-preprocessing-detail.md` |
| A-1 至 A-9 | `Networks/layers.py`、`Networks/net.py`、`Networks/net_moe.py` |
| W-0 至 W-10 | `Networks/net.py::WindowAttention`、`Networks/net_moe.py::SDPAWindowAttention/MoETransformerBlock` |
| M-1 至 M-14 | `Networks/net_moe.py::Expert/MoEFFN/MODEL_MoE` |
| H-1 至 H-4 | `Networks/net_moe.py::MODEL_MoE.forward`、最终 `W_96d4L/args.txt` |
| L-1 至 L-17 | `train_moe.py`、`train_moe_ddp.py`、`losses/__init__.py`、最终 `W_96d4L/args.txt` |
| D-1 至 D-8 | `train_moe.py`、`train_moe_ddp.py`、`content/section-efficiency.md` |
| E-1 至 E-12 | `metrics/fusion_metrics.py`、`EVALUATION-metrics.md`、`content/section-setup.md` |
