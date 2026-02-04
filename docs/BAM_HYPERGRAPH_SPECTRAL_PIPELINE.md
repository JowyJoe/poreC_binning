# BAM → Hypergraph → Spectral Coarse Binning → Refine：论文级方法学草案

> 本文档用于把我们讨论的“完全抛弃 PPL `.contacts`，从 name-sorted BAM 直接构建超图并分箱”的理论与数学定义整理成可写入论文 Methods 的版本。  
> 公式使用 LaTeX 书写（建议用支持 MathJax/KaTeX 的渲染器查看）。

## 1. 目标与设计原则

**目标**：利用 Nanopore Pore-C 的 multi-way contacts 做宏基因组分箱，尽可能获得 **低污染（contamination）** 与 **高完整度（completeness）**（以 CheckM2 等评估为准）。

**核心原则**：

1. **保留 multi-way（超边）结构**：每条 read（QNAME）对应一条超边，不做 clique expansion。
2. **证据可解释、可复现**：过滤规则与边权从 BAM 中的 `MAPQ/AS/NM/对齐长度/比对类型(primary/secondary/supplementary)` 等量严格定义；所有决策记录到 `run.json`。
3. **避免不稳定的 K 选择**：谱方法保留，但不把结果押在 `eigengap → K → k-means`；优先采用“递归谱二分 + 模型选择停止（BIC）”实现自适应簇数。

---

## 2. 记号与对象

- 顶点集合 \(V\)：contigs（每个 contig 一个顶点 \(v\)）。
- 超边集合 \(E\)：reads（每条 read \(r\) 一个超边 \(e=r\)）。
- 对齐记录集合 \(A_r\)：BAM 中 query name 为 \(r\) 的全部 alignment 记录（name-sorted BAM 可流式聚合）。

我们将构建带权 incidence 矩阵 \(H\in\mathbb{R}^{|V|\times |E|}\)，并据此构建超图拉普拉斯进行谱分箱。

---

## 3. 从 name-sorted BAM 构建带权超图（BAM → weighted incidence）

### 3.1 预处理：保留哪些 alignment（降低污染的关键）

为抑制重复序列与多重比对造成的跨物种“假连接”，建议：

- **丢弃 secondary alignments**（`is_secondary=True`）。
- **保留 primary + supplementary**：Pore-C 的 multi-way contact 依赖 split alignment；supplementary 是有效信号。
- 可选丢弃：`is_qcfail`、`is_duplicate`（视数据情况）。

> 以上属于“定义性选择”，不是经验阈值；目的在于让超边主要由可信的映射片段组成。

### 3.2 单条 alignment 的连续证据权重（不依赖硬阈值）

设 \(a\in A_r\) 为一条 alignment，令：

- \(MAPQ(a)\)：mapping quality（Phred 量纲）
- \(\ell(a)\)：aligned length（`query_alignment_length`）
- \(NM(a)\)：错配数（若 BAM 含 NM tag）

由 MAPQ 的定义得到“正确映射概率”权重：

$$
p_{\text{ok}}(a)=1-10^{-MAPQ(a)/10}.
$$

若存在 NM，可定义近似 identity：

$$
id(a)=\max\!\left(0, 1-\frac{NM(a)}{\ell(a)}\right).
$$

定义单条 alignment 的证据强度：

$$
e(a)=p_{\text{ok}}(a)\cdot id(a)\cdot \ell(a).
$$

> 解释：与其用 `MAPQ≥X` 这类经验阈值，不如把 MAPQ 转成概率并连续化地影响权重，便于写进论文且更稳健。

### 3.3 read → contig 的软分配（soft incidence）

对 read \(r\) 与 contig \(c\)，聚合证据：

$$
E_{r,c}=\sum_{a\in A_r,\ c(a)=c} e(a).
$$

归一化为软分配：

$$
P_{r,c}=\frac{E_{r,c}}{\sum_{c'}E_{r,c'}}.
$$

则 \(P_{r,\cdot}\) 是 read \(r\) 在各 contig 上的概率分布；低质量/多重映射产生的“尾巴 contig”将被自然抑制。

### 3.4 用有效阶数 \(k_{\text{eff}}\) 替代硬阶数 \(k\)

直接用“命中 contig 个数”作为阶数 \(k\) 对噪声很敏感。我们使用概率分布的有效支持大小（inverse Simpson index）：

$$
k_{\text{eff}}(r)=\frac{1}{\sum_c P_{r,c}^2}.
$$

性质：

- 若 \(P\) 近似集中在 2 个 contig，则 \(k_{\text{eff}}\approx 2\)
- 若 \(P\) 接近均匀分散在很多 contig，则 \(k_{\text{eff}}\) 才会显著增大

### 3.5 超边权重（连续化 OrderNorm）

沿用 v0.1 的“高阶 contact 降权”思想，但为了避免在 \(k_{\text{eff}}\to 1^+\)（概率极度集中在单一 contig）时出现数值放大，我们建议：

1) 用 **硬阶数** \(k(r)\) 做 OrderNorm（只依赖“该 read 命中了多少个 contig”这一事实）；
2) 用 BAM 证据定义一个 **read-level 可靠性因子** \(R(r)\in(0,1]\) 来连续化抑制噪声 read（而不是靠经验阈值）。

硬阶数定义为（在过滤 secondary 后）：

$$
k(r)=\left|\{c:\ E_{r,c}>0\}\right|.
$$

OrderNorm（pair-normalization）：

$$
\mathrm{OrderNorm}(k)=\frac{2}{k(k-1)} \quad (k\ge 2).
$$

一个无阈值、可解释的可靠性因子示例是“平均映射置信度 × 映射集中度”：

- 平均映射置信度：

$$
\bar p_{\text{ok}}(r)=\frac{1}{|A_r|}\sum_{a\in A_r} p_{\text{ok}}(a).
$$

- 映射集中度（Simpson concentration）：

$$
C(r)=\sum_c P_{r,c}^2=\frac{1}{k_{\text{eff}}(r)}.
$$

于是可定义：

$$
w(r)=\mathrm{OrderNorm}\!\left(k(r)\right)\cdot \bar p_{\text{ok}}(r)\cdot C(r).
$$

最终定义 incidence：

$$
H_{c,r}=w(r)\cdot P_{r,c}.
$$

> 这一步是“低污染”的另一个关键：它让“看起来很高阶但其实很噪”的 read 不会产生超强连接；并且在上述定义下 \(w(r)\le 1\) 有利于数值稳定。

---

## 4. Zhou-style 归一化超图拉普拉斯（理论骨架）

参考：Zhou, Huang, Schölkopf (2007), *Learning with Hypergraphs: Clustering, Classification, and Embedding*.

为了与 Zhou et al. (2007) 形式对齐，我们先引入“未加权”的软 incidence：

$$
\tilde H_{c,r}=P_{r,c},
$$

以及超边权重矩阵：

$$
W=\mathrm{diag}(w(r)).
$$

**实现约定（与代码一致）**：我们把权重吸收进 incidence，直接存储

$$
H=\tilde H\,W \quad\Longleftrightarrow\quad H_{c,r}=w(r)\,P_{r,c}.
$$

在这种记号下，可不显式写出 \(W\)（两种写法是代数等价的）。

给定加权 incidence \(H\)，定义：

- 顶点度（contig 度）：

$$
d(v)=\sum_{r} H_{v,r}, \qquad D_v=\mathrm{diag}(d(v)).
$$

- 超边度（read 度）：

$$
\delta(r)=\sum_{v} H_{v,r}, \qquad D_e=\mathrm{diag}(\delta(r)).
$$

归一化“扩散/相似度”矩阵：

$$
\Theta = D_v^{-1/2}\,H\,D_e^{-1}\,H^{\top}\,D_v^{-1/2}.
$$

归一化拉普拉斯：

$$
L = I-\Theta.
$$

直觉：\(\Theta\) 描述了超图上的“归一化扩散/随机游走式相似度”；谱分解提供一个低维嵌入，使得经常共现于同一超边的顶点在嵌入空间中更接近。

---

## 5. Coarse 分箱：递归谱二分（不显式选 K）

### 5.1 为什么不用 `eigengap → K → k-means` 作为主流程

宏基因组 + Pore-C 图结构常出现：

- hub read / hub contig
- 重复序列导致的跨物种弱连接
- coverage 接近的多个基因组

在这种情况下，`eigengap` 往往偏向极小 K（例如 2），不稳健且难以保证论文级可重复性。

### 5.2 每次二分：Fiedler 向量 + sweep cut（谱二分的经典 rounding）

对当前簇 \(S\subseteq V\)，构建其诱导子超图/子矩阵 \(L_S\)。求第二小特征向量 \(u_2\)（Fiedler vector），按 \(u_2\) 排序后进行 sweep cut，选择使 normalized cut（或其等价形式）最优的分割。

> 原理：normalized cut 的谱松弛 + rounding（与 Shi–Malik (2000) 的 Ncut、Cheeger-type 不等式相关）。

### 5.3 是否接受二分：用 BIC 做模型选择停止规则（自适应簇数）

为了不引入人为阈值，我们用统计模型选择判断“这次切分是否有意义”。一种可实现且可解释的做法：

1) 从超图构造 contig–contig 的非负相似度矩阵（去掉对角）：

$$
A = H\,D_e^{-1}\,H^{\top}.
$$

2) 在簇内 \(A_S\) 上，把观测定义为边存在性：

$$
x_{ij}=\mathbb{1}[A_{ij}>0],\quad i<j.
$$

3) 比较：

- 1-block：\(x_{ij}\sim \mathrm{Bernoulli}(p_0)\)
- 2-block（由谱二分给出左右两侧）：簇内边概率 \(p_{\text{in}}\)，簇间边概率 \(p_{\text{out}}\)

计算最大似然与 BIC：

$$
BIC = -2\log\mathcal{L} + k\log N,
$$

其中 \(N\) 是观测对数（例如 \(n(n-1)/2\)），\(k\) 是参数数目（1-block 为 1，2-block 为 2）。

**触发规则**：

$$
\text{accept split if } BIC_{2\text{-block}} < BIC_{1\text{-block}}.
$$

可选：coverage 辅助触发（来自 BAM）：

- 在簇内 contig 的 \(y=\log(1+\mathrm{cov})\) 上比较 1-Gauss vs 2-GMM 的 BIC；
- 作为“额外证据”，但不应作为唯一触发（否则 coverage 接近的混合簇拆不动）。

> 原理：Schwarz (1978) 的 BIC；用“模型是否显著更好”替代“猜 K”。

递归执行直到不再触发切分，即得到 coarse bins（簇数自适应）。

---

## 6. Refine：提纯（降污染）+ 召回（提完整）+ 拆分（修复混合）

### 6.1 超图接触支持：contig 对某 bin 的支持分数

设当前 bins 为 \(b\in\mathcal{B}\)，每个 bin 是 contig 的集合。定义 contig \(v\) 对 bin \(b\) 的“接触支持”：

$$
S(v,b)=\sum_{r\in E} w(r)\,P_{r,v}\,\Big(\sum_{u\in b} P_{r,u}\Big).
$$

直觉：同一 read 中 \(v\) 与 bin 内 contig 的软共现期望；是超图的自然统计量。

### 6.2 reassign（关键）：用对数似然比把错分 contig 迁移到更合理的 bin

令当前归属为 \(b_0\)，最佳替代为 \(b_1=\arg\max_{b\neq b_0}S(v,b)\)。定义：

$$
LLR(v)=\log(S(v,b_0)+\epsilon)-\log(S(v,b_1)+\epsilon).
$$

若 \(LLR(v)\ll 0\)，说明 \(v\) 更符合 \(b_1\) 而非 \(b_0\)，属于“潜在污染来源”。

阈值不拍脑袋：对所有 \(LLR(v)\) 用 1/2 成分混合模型做 BIC 选择并求后验，自动确定“迁移/不迁移”分界。

### 6.3 coverage 作为 veto/辅助：用似然而非硬阈值

从 BAM 直接计算 contig coverage（例如 aligned_bases / contig_length），在 log 空间建模：

$$
y(v)=\log(1+\mathrm{cov}(v)).
$$

每个 bin \(b\) 拟合 \(p(y\mid b)\)（可用稳健分布如 Student-t，或中位数+MAD 的近似），得到对数似然 \(\log p(y(v)\mid b)\)。

将接触证据与 coverage 证据合并（朴素贝叶斯式）：

$$
score(v,b)=\alpha\log(S(v,b)+\epsilon) + (1-\alpha)\log p(y(v)\mid b).
$$

\(\alpha\) 可通过数据驱动选择（例如在高置信 contig 上最大化一致性/似然）。

### 6.4 split：对疑似混合 bin 进行局部递归谱二分

触发建议使用：

- 图结构触发：bin 内 1-block vs 2-block 的 BIC（同 §5.3）
- coverage 触发：bin 内 1 vs 2 成分的 BIC

一旦触发，bin 内再运行递归谱二分，得到子 bin；随后在局部再次运行 reassign/recruit，以提升边界一致性。

### 6.5 recruit：对 unbinned contig 进行后验高置信招回

对 unbinned 的 \(v\)，取 \(b^\*=\arg\max_b score(v,b)\)，若其后验概率落入“高置信成分”，则招回至 \(b^\*\)；否则保持 unbinned（高纯度优先）。

---

## 7. 建议的工程化输入/输出（为后续编码做准备）

### 7.1 新增的内部标准输入（替代 `.contacts`）

**输入**：name-sorted BAM（按 QNAME 排序）。  
**输出**：内部标准 `contacts.parquet`（一行一个 read/contact/hyperedge）：

- `contact_id`：整数 id
- `contigs`：list[str]（建议存“软分配后保留的 contig 列表”，并可另存权重）
- `k_eff`：float
- `weight`：float（即 \(w(r)\)）
- `support_count`：int（可设为 1）
- 可选：`contig_weights`：list[float]（对应 \(P_{r,c}\)，用于构图时写入 \(H_{c,r}\)）
- 可选证据：`mapq_mean`, `mapq_min`, `aligned_len_sum`, `nm_sum`, `n_segments` 等（用于 QC/消融）

> 注意：如果要实现软 incidence \(H_{c,r}=w(r)\cdot P_{r,c}\)，则需要在构图阶段读到每个 contig 的 \(P_{r,c}\)。这可以通过 `contig_weights` 存下来实现。

### 7.2 pipeline 的目录结构（建议）

- `out/contacts/contacts.parquet`（BAM→contacts）
- `out/graph/`（build_graph 输出：`contig_index.tsv`, `edges.tsv`, `graph_meta.json` 等）
- `out/bins.tsv`（coarse）
- `out/refined/`（refine：`bins.refined.tsv`, `reassign.tsv`, `decontam_removed.tsv`, `split_map.tsv`, `run_refine.json`）
- `out/final_bins/`（export FASTA）

### 7.3 运行记录（run.json 的关键字段建议）

需要记录：

- BAM 输入与过滤规则（是否丢 secondary、是否保留 supplementary）
- 证据权重定义（MAPQ→\(p_{\text{ok}}\)、是否使用 NM/identity）
- \(k_{\text{eff}}\) 与 \(w(r)\) 的定义
- coarse 每次切分的 \(\Delta BIC\)、簇大小、停止原因
- refine 的 reassign/recruit/split 决策统计与 \(\Delta BIC\)/后验阈值

---

## 8. 参考文献（可在论文中引用）

- Zhou, D., Huang, J., & Schölkopf, B. (2007). *Learning with Hypergraphs: Clustering, Classification, and Embedding*. NIPS.
- Shi, J., & Malik, J. (2000). *Normalized Cuts and Image Segmentation*. IEEE TPAMI.
- Ng, A. Y., Jordan, M. I., & Weiss, Y. (2002). *On Spectral Clustering: Analysis and an algorithm*. NeurIPS.
- Schwarz, G. (1978). *Estimating the dimension of a model*. Annals of Statistics. (BIC)
