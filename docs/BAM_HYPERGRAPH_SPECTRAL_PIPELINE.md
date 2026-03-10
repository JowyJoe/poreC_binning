# BAM → Hypergraph → Spectral Coarse Binning → Refine：论文级方法学草案

> **说明（2026-03-02）**：本文档作为开发日志/方法学草稿，仅供参考；不再作为实现的强制规范。以仓库中的实际代码与 CLI 行为为准。

> 本文档用于把我们讨论的“完全抛弃 PPL `.contacts`，从 name-sorted BAM 直接构建超图并分箱”的理论与数学定义整理成可写入论文 Methods 的版本。  
> 公式使用 LaTeX 书写（建议用支持 MathJax/KaTeX 的渲染器查看）。

## 1. 目标与设计原则

**目标**：利用 Nanopore Pore-C 的 multi-way contacts 做**宿主中心（host-centric）**的宏基因组分箱，尽可能获得 **低污染（contamination）** 与 **高完整度（completeness）**（以 CheckM2 等评估为准）。

**次目标**：在不做“硬分类”的前提下，为 accessory / MGE-like contigs 输出保守的结构性宿主关联摘要（association head）。

**核心原则**：

1. **保留 multi-way（超边）结构**：每条 read（QNAME）对应一条超边，不做 clique expansion。
2. **证据可解释、可复现**：过滤规则与边权从 BAM 中的 `MAPQ/AS/NM/对齐长度/比对类型(primary/secondary/supplementary)` 等量严格定义；所有决策记录到 `run.json`。
3. **避免不稳定的 K 选择**：`coarse-method=spectral` 已切换为 **联合超图谱谱嵌入（contact hypergraph + feature hypergraph） + HDBSCAN 自动聚类**（无需指定 K）。旧的“递归谱二分 + BIC 停机”仅作为历史对照，不再是默认实现路径。

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

实现约定（与 `porebin/bam_contacts.py` 一致）：对齐长度 \(\ell(a)\) 优先取 `query_alignment_length`；若不可得则取 `reference_end-reference_start`；若仍不可得则跳过该片段并计入 `len_missing_count`。

由 MAPQ 的定义得到“正确映射概率”权重：

$$
p_{\text{ok}}(a)=1-10^{-MAPQ(a)/10}.
$$

实现约定（与 `porebin/bam_contacts.py` 一致）：若 `MAPQ==255`（SAM 规范：mapping quality not available / unknown）或 MAPQ 缺失/异常（None、<0 等），则按缺失值处理：
\(p_{\text{ok}}(a)=0.5\)，并计入 `mapq_missing_count`。

若存在 NM，可定义近似 identity：

$$
id(a)=\max\!\left(0, 1-\frac{NM(a)}{\ell(a)}\right).
$$

实现约定（与 `porebin/bam_contacts.py` 一致）：若 BAM 中没有 `NM` tag，则取 \(id(a)=1\)，并计入 `nm_missing_count`。

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
\pi_{r,c}=\frac{E_{r,c}}{\sum_{c'}E_{r,c'}}.
$$

其中 \(\pi_{r,c}\) 是 read \(r\) 向 contig \(c\) 的**归一化证据份额**（soft incidence weight）：
\(\pi_{r,c}\ge 0\) 且对同一条 read 命中的 contigs 有 \(\sum_c \pi_{r,c}=1\)。

> 重要术语：\(\pi_{r,c}\) 是归一化证据份额（用于 incidence 权重），不是后验概率；后验样式的量应在 refine 推断层定义。

### 3.4 用有效阶数 \(k_{\text{eff}}\) 替代硬阶数 \(k\)

直接用“命中 contig 个数”作为阶数 \(k\) 对噪声很敏感。我们使用归一化证据份额 \(\pi_{r,\cdot}\) 的有效支持大小（inverse Simpson index；QC only）：

$$
k_{\text{eff}}(r)=\frac{1}{\sum_c \pi_{r,c}^2}.
$$

性质：

- 若 \(\pi\) 近似集中在 2 个 contig，则 \(k_{\text{eff}}\approx 2\)
- 若 \(\pi\) 接近均匀分散在很多 contig，则 \(k_{\text{eff}}\) 才会显著增大

### 3.5 read 质量权重 \(q(r)\) 与阶数归一化（只在下游做一次）

**证据层（BAM→contacts.parquet）的核心原则**：multi-way（高阶）是信号，不应被当作歧义强惩罚。
因此在 evidence layer 我们把 read 的总信息量主要交给 read-level 的对齐可信度 \(q(r)\) 控制，而不是再乘映射集中度或阶数惩罚项。

在 `contacts.parquet` 中：
- `weight` 字段存的是 \(q(r)\in[0,1]\)（不乘 \(C(r)\)）。
- `contig_weights` 存的是 \(\pi_{r,c}\)（归一化证据份额）。

如果某个下游模块需要对阶数 \(k\) 做一次**温和**的归一化（避免“一个 read 的贡献随着阶数线性变大”），应在该模块内显式定义并只做一次，例如：
- build_graph 的二部图边：`edge_weight(c,r)=OrderNorm(k(r)) * q(r) * \pi_{r,c}`（OrderNorm 可选 pair/star）
- spectral coarse 的 contact 超边权重：\(W_c(e)=q(e)/(k(e)-1)\)（见 §5）

硬阶数定义为（在过滤 secondary 后）：

$$
k(r)=\left|\{c:\ E_{r,c}>0\}\right|.
$$

常用的两种 OrderNorm（仅用于下游某些模块，不改变 \(q(r)\) 的语义）：

$$
\mathrm{OrderNorm}_{\text{pair}}(k)=\frac{2}{k(k-1)} \quad (k\ge 2),\qquad
\mathrm{OrderNorm}_{\text{star}}(k)=\frac{1}{k-1} \quad (k\ge 2).
$$

实现里我们同时输出：read 质量权重 \(q(r)\)（用于 `weight`）与映射集中度 \(C(r)\)（仅 QC）。
映射集中度（Simpson concentration；QC only）：

$$
C(r)=\sum_c \pi_{r,c}^2=\frac{1}{k_{\text{eff}}(r)}.
$$

实现说明：当前 `contacts.parquet` 中 `C(r)`/`k_eff` 仅作为 QC 输出；`weight` 使用 \(q(r)\)，不再乘 \(C(r)\)（不惩罚 multi-way）。

read-level 质量权重定义（与实现一致）：

$$
q(r)=\frac{\sum_{a\in A_r} p_{\text{ok}}(a)\,id(a)\,\ell(a)}{\sum_{a\in A_r}\ell(a)}.
$$

---

## 4. Zhou-style 归一化超图拉普拉斯（理论骨架）

参考：Zhou, Huang, Schölkopf (2007), *Learning with Hypergraphs: Clustering, Classification, and Embedding*.

为了与 Zhou et al. (2007) 形式对齐，我们先引入“未加权”的软 incidence：

$$
\tilde H_{c,r}=\pi_{r,c},
$$

以及超边权重矩阵：

$$
W=\mathrm{diag}(w(r)).
$$

工程上有两种**代数等价**的写法（代码里两者都会出现，取决于模块/缓存格式）：

1) **把超边标量权重吸收进 incidence**（常见于二部图缓存），直接存储加权 incidence：

$$
H=\tilde H\,W \quad\Longleftrightarrow\quad H_{c,r}=w(r)\,\pi_{r,c}.
$$

2) **保留未加权 incidence \(\tilde H\) 与权重 \(W\) 分开存储**（spectral v2 的 contact 超图按此实现）：
在 matvec 中显式使用 \(H\) 与 \(\mathrm{diag}(W)\)。

两种写法代数等价；关键是：\(\pi_{r,c}\) 的语义始终是 evidence share，\(w(r)\) 的语义由下游模块显式定义并只应用一次。

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

## 5. Coarse 分箱（spectral v2：joint hypergraph embedding + HDBSCAN）

> **实现更新（spectral v2）**：当前仓库的 `--coarse-method spectral / --method spectral` 已切换为
> **联合超图谱嵌入（contact hypergraph + feature hypergraph） + HDBSCAN 自动聚类**，不再使用“递归谱二分 + BIC 停机”作为默认实现。
>
> 联合定义概览：
> - contact 超图：从 `contacts.parquet` 读软 incidence \(H_c[v,e]=\pi_{e,v}\)，超边权重 \(W_c(e)=q(e)/(k(e)-1)\)（丢弃 \(k<2\)）。
> - feature 超图：每个顶点 \(v\) 生成超边 \(e_v=\{v\}\cup kNN(v)\)，incidence 为二值 \(H_f[u,e_v]=\mathbb{1}[u\in e_v]\)，权重全 1。
>   - feature 向量：canonical TNF(136)（列顺序固定为 `TNF136_LIST`；按 `canonical_4mer=min(s,revcomp(s))` 合并反向互补；忽略含 N 的窗口）+ 可选 `log1p(coverage)`，拼接后逐维 z-score。
>   - `knn_k=15`（内部常量，不暴露 CLI 参数）。
> - Zhou-style：\(\Theta=D_v^{-1/2}H\,\mathrm{diag}(W)\,\mathrm{diag}(D_e^{-1})H^\top D_v^{-1/2}\)（通过 LinearOperator matvec 实现，不构造稠密 \(|V|\times|V|\)）。
> - joint：\(\Theta_{\text{joint}}=\lambda\,\Theta_c+(1-\lambda)\,\Theta_f\)，取最大特征向量前 \(d+1\) 个，丢掉第 1 个平凡向量得到嵌入 \(Z\)，对行做 L2 归一化。
>   - \(\lambda\)：若存在 `coverage.tsv` 则 0.6，否则 0.7（自动选择并记录）。
>   - \(d\)：\(d=\min(128,\max(32,\lfloor\log_2|V|\rfloor\cdot 4))\)，并确保 \(d\le |V|-2\)。
> - 聚类：对 \(Z\) 运行 HDBSCAN（无需指定簇数；噪声 label=-1 视为 unbinned）。
>
> 下文的“递归谱二分 + BIC”可视为历史方法学草案/对照，并非当前默认实现路径。

### 5.0 Spectral v2：精确定义与工程实现（当前默认）

**输入/索引**

- `contacts.parquet`：至少包含 `contigs/contig_weights/k/weight`（其中 `weight=q(e)`，Part 1 定义：\(q=\sum(p_{\text{ok}}id\ell)/\sum\ell\)，且不再乘 \(C(r)\)）。
- `graph/contigs.tsv`：必须存在，格式固定（无表头）：
  - `contig_idx<TAB>contig_name`
  - `contig_idx` 与 build_graph 的 contig 节点编号一致（0..|V|-1）。

**Contact hypergraph（Pore‑C multi‑way contacts）**

- 软 incidence：\(H_c\in\mathbb{R}^{|V|\times|E_c|}\)，定义 \(H_c[v,e]=\pi_{e,v}\)（来自 `contig_weights`，不是 0/1）。
- 丢弃 singleton：若 `k<2`，该 read 不进入 \(H_c/W_c\)，并计数 `dropped_edges_singleton_contact += 1`。
- 超边权重：\(W_c[e]=q(e)/(k(e)-1)\)（只在此处做一次 \(1/(k-1)\) 阶数归一化；其他地方不得再次按 k 惩罚）。
- 超边度：\(D_{e_c}[e]=\sum_v H_c[v,e]\)。
- 顶点度：\(D_{v_c}[v]=\sum_e W_c[e]\cdot H_c[v,e]\)。
- Zhou‑style 对称算子：
  \[
  \Theta_c=D_{v_c}^{-1/2}\,H_c\,\mathrm{diag}(W_c)\,\mathrm{diag}(D_{e_c}^{-1})\,H_c^\top\,D_{v_c}^{-1/2}.
  \]

**Feature hypergraph（序列组成 + coverage 可选）**

- 每个 contig 生成特征向量 \(x_v\)：
  - canonical TNF(136)：列顺序固定为 `TNF136_LIST`；只统计全为 A/C/G/T 的 4‑mer，含 N 的窗口忽略；按 `canonical_4mer=min(s,revcomp(s))` 合并反向互补后计数归一化为频率。
  - coverage（可选）：若 `coverage.tsv` 存在且包含该 contig：`cov_feat=log1p(cov)`；否则 `cov_feat=0` 并记录 `coverage_missing_count`。
  - 拼接并逐维 z‑score：\(x_v=[\mathrm{freq}_{136},\ \mathrm{cov\_feat}]\)（方差为 0 的维度保持 0）。
- kNN：`knn_k=15`（内部常量，不暴露 CLI 参数）。
- 一顶点一超边：对每个顶点 \(v\) 生成超边 \(e_v=\{v\}\cup kNN(v)\)，因此 \(|E_f|=|V|\)。
- 二值 incidence：\(H_f[u,e_v]=\mathbb{1}[u\in e_v]\)，权重 \(W_f[e]=1\)。
- \(D_{e_f}[e]=|e|=\texttt{knn\_k}+1\)，\(D_{v_f}[v]=\sum_e H_f[v,e]\)。
- \[
  \Theta_f=D_{v_f}^{-1/2}\,H_f\,\mathrm{diag}(W_f)\,\mathrm{diag}(D_{e_f}^{-1})\,H_f^\top\,D_{v_f}^{-1/2}.
  \]

**Joint + 谱嵌入**

- \(\Theta_{\text{joint}}=\lambda\,\Theta_c+(1-\lambda)\,\Theta_f\)。
- \(\lambda\) 自动设定（不暴露 CLI 参数）：若存在 `coverage.tsv` 取 0.6，否则取 0.7（记录到 `run.json`）。
- 嵌入维度：
  \[
  d=\min(128,\max(32,\lfloor\log_2|V|\rfloor\cdot 4)),\quad d\le |V|-2.
  \]
- 用 `eigsh(which="LA")` 求 \(\Theta_{\text{joint}}\) 的前 \(d+1\) 个最大特征向量，丢掉第 1 个平凡向量，得到 \(Z\in\mathbb{R}^{|V|\times d}\)，对每行做 L2 归一化（全 0 行保持 0）。

**工程实现要求（避免 clique expansion / 避免稠密矩阵）**

- 不做 clique expansion（不把每条 read 展开成所有 contig 对边）。
- 不显式构造 \(|V|\times|V|\) 稠密矩阵；\(\Theta_c/\Theta_f/\Theta_{\text{joint}}\) 均用 LinearOperator 的 `matvec` 实现：
  - 给定向量 \(x\in\mathbb{R}^{|V|}\)，contact `matvec`：
    1) \(y=D_{v_c}^{-1/2}\odot x\)
    2) \(z=H_c^\top y\)
    3) \(z=z\odot (W_c/D_{e_c})\)
    4) \(y_2=H_c z\)
    5) `out = D_{v_c}^{-1/2} ⊙ y2`
  - feature `matvec` 同理，且 \(W_f=1\Rightarrow z=z/D_{e_f}\)。
  - joint：\(\lambda\Theta_c(x)+(1-\lambda)\Theta_f(x)\)。
- 当 \(D_{v_c}[v]=0\) 时，令 \(D_{v_c}^{-1/2}[v]=0\)，视为 contact‑isolated；最终强制该 contig `label=-1` 并记录 `isolated_contigs_count_contact`。

**聚类（HDBSCAN）**

- 对 \(Z\) 运行 HDBSCAN：`min_cluster_size=5`、`min_samples=None`、`metric=euclidean`；若实现支持并有 `--threads`，则使用并行。
- 输出 `bins.tsv`：对 `label!=-1` 的 contig 输出，bin_id 重映射为 0..num_bins-1；`label=-1` 视为 unbinned。
- 后处理（contact 连通性一致性）：为了避免 feature 超图把 **contact 不连通** 的 contig 合并进同一 bin，实际实现会：
  - 将同一 HDBSCAN label 按 contact 超图的连通分量拆分（一个 bin 不跨越多个 contact 分量）；
  - 将 contact 分量内被标为 `-1` 的 contig 重新分配到该分量的多数标签；
  - 若某个 contact 分量整体为 `-1` 且分量大小 \(\ge\) `min_cluster_size`，则提升为一个新标签（避免“大量可连接 contig 全部 unbinned”）。
  相关统计会写入 `run.json` 的 `contact_component_postprocess` 字段以便审计与复现。

**审计记录（run.json / coarse meta）**

必须记录：`spectral_v2_joint_enabled=true`、`lambda_contact`、`d`、`knn_k`、`dropped_edges_singleton_contact`、`isolated_contigs_count_contact`、`num_bins`、`unbinned_count`，以及 HDBSCAN 实现来源与版本信息。

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

## 6. Refine：host-assignment inference（主任务）+ accessory/MGE-like association（次任务）

> **语义更新（v0.1.0）**：Refine 不再被描述为“又一次聚类”或“启发式细分”，而是一个**宿主归属推断层**：
> coarse 阶段的 `bins.tsv` 只是**候选宿主群落**（candidate host communities），refine 负责给出每个 contig 的宿主支持分布与不确定性摘要，并对 accessory/MGE-like contigs 给出结构性关联输出。

### 6.1 输入量与符号（必须区分 evidence vs inference）

Evidence layer（来自 `contacts.parquet`）：

- \(E_{r,c}\)：read \(r\) 向 contig \(c\) 的（去重叠后）证据质量质量（见 §3）。
- \(\pi_{r,c}\)：归一化证据份额（`contig_weights`），满足 \(\sum_c \pi_{r,c}=1\)（在该 read 命中的 contigs 上）。
  - **注意**：\(\pi_{r,c}\) 不是后验概率。
- \(q(r)\)：read-level 对齐可信度权重（`weight`），范围 \([0,1]\)。
- \(k(r)\)：该 read 的阶数（命中 contig 数；`k`）。

Coarse layer（来自 `bins.tsv`）：

- coarse 给每个 contig 一个候选宿主群落标签 \(b_0(c)\)（candidate host）。

Refine layer（本节定义的新量）：

- \(\theta_{c,b}\)：contig \(c\) 对候选宿主 \(b\) 的**posterior-like 支持分数**（refine 推断层量）。
- （可选概念）\(\gamma_{r,b}\)：read \(r\) 对候选宿主 \(b\) 的 dominant/mixed 支持（本实现第一版不显式输出，只用一阶代理量）。

### 6.2 HySBM-inspired 的一阶支持聚合（当前实现：one-pass \(\theta\)）

本实现使用一种 **HySBM-inspired**（仅“启发/定位”，非精确实现）的一次扫描聚合：

对每条 read \(r\)，先计算它在 coarse 宿主上的质量份额（一个 \(\gamma\)-like 代理量）：

$$
m_{r,b}=\sum_{u:\ b_0(u)=b} \pi_{r,u}.
$$

然后对该 read 中每个 contig \(c\)，更新 contig→host 支持：

$$
\theta_{c,b}\mathrel{+}= q(r)\,\pi_{r,c}\,m_{r,b}^{(-c)},
\qquad
m_{r,b}^{(-c)} = m_{r,b} - \mathbb{1}[b_0(c)=b]\cdot \pi_{r,c}.
$$

直觉：
- \(q(r)\) 控制该 read 的总信息量（高阶 read 不会因为 \(k\) 大被强惩罚）。
- \(\pi_{r,c}\) 是该 read 给 contig \(c\) 的证据份额。
- \(m_{r,b}^{(-c)}\) 代表“该 read 中属于宿主 \(b\) 的其他 contigs 的证据质量份额”，因此它衡量 contig \(c\) 与宿主 \(b\) 的结构性共现支持（并去掉自支持）。

同时记录每个 contig 的总证据支撑：

$$
\mathrm{total\_support}(c)=\sum_r q(r)\,\pi_{r,c}.
$$

Feature 在 refine 中只做 prior/regularizer：当前实现使用一个非常弱的 coarse-label prior：
\(\theta_{c,b_0(c)}\mathrel{+}=\texttt{prior\_add}\)，其中 `prior_add` 是 `total_support` 的一个小比例（具体常量写入 `run_refine.json`）。

### 6.3 不确定性摘要与 contig 类型标记

对每个 contig \(c\)，把 \(\theta_{c,\cdot}\) 归一化为一个 posterior-like 分布（当前实现保留 top‑L 宿主的稀疏表示）：

$$
\hat\theta_{c,b}=\frac{\theta_{c,b}}{\sum_{b'}\theta_{c,b'}}.
$$

由 \(\hat\theta\) 计算不确定性摘要：
- `top1_host/top2_host`
- `margin(c)=\hat\theta_{c,top1}-\hat\theta_{c,top2}`
- `entropy(c)=-\sum_b \hat\theta_{c,b}\log \hat\theta_{c,b}`（自然对数）
- `effective_hosts(c)=\exp(\mathrm{entropy}(c))`

并据此标记：
- `is_core_like(c)`：高支持、低不确定（用于最终宿主 bins）
- `is_ambiguous(c)`：不确定性较高（可能是边界/重复/弱证据 contig）
- `is_accessory_candidate(c)`：不确定性显著且非 core（进入结构性关联 head）

阈值均为内部常量，并写入 `run_refine.json` 以便审计/复现。

### 6.4 accessory/MGE-like association head（结构性关联输出）

对 `is_accessory_candidate` 的 contig，输出其与多个宿主的结构性关联（不做硬分类）：
- `top_hosts`：关联最强的前 3 个宿主
- `host_weights`：对应的归一化权重（来自 \(\hat\theta\)）
- `host_entropy` 与 `effective_hosts`
- `association_confidence`：由归一化熵得到的保守置信度（当前实现：\(1-\mathrm{entropy}/\log K\)）
- `single_host_like/broad_host_like`：基于 `effective_hosts` 与 top1 权重的粗分型（仅描述关联形态，不等于生物学鉴定）

---

## 7. 工程化输入/输出（以当前实现为准）

### 7.1 标准 evidence 接口：`contacts.parquet`

name-sorted BAM → `contacts.parquet`（一行一个 read/contact/hyperedge）：

- `contact_id`：整数 id
- `contigs`：list[str]
- `contig_weights`：list[float]（对应 \(\pi_{r,c}\)，与 `contigs` 对齐且归一化为 1；**evidence share，不是后验**）
- `k`：int（硬阶数，等于 `len(contigs)`）
- `k_eff`：float（有效阶数 \(1/\sum_c \pi_{r,c}^2\)；QC only）
- `weight`：float（read 质量权重 \(q(r)\in[0,1]\)，不再乘映射集中度/阶数惩罚）
- 证据/QC：`mapq_min`, `p_ok_mean`, `aligned_len_sum`, `deoverlap_query_union_len_sum`, `nm_sum`, `n_segments`,
  `mapq_missing_count`, `nm_missing_count`, `len_missing_count`

### 7.2 目录结构（当前默认）

- `out/contacts/contacts.parquet`（evidence）
- `out/coverage/coverage.tsv`（可选 feature；定义保持不变）
- `out/graph/`（build_graph 输出审计与缓存：`contig_index.tsv`, `contigs.tsv`, `edges.tsv`, `graph_meta.json` 等）
- `out/bins.tsv`（coarse：candidate host communities）
- `out/refined/`（refine：`bins.refined.tsv`, `contig_host_scores.tsv`, `accessory_associations.tsv`, `run_refine.json`）
- `out/final_bins/`（export FASTA）

### 7.3 审计记录（run.json / run_refine.json）

必须记录：
- evidence：BAM flag 过滤语义（skip unmapped、drop secondary、keep supplementary）、de-overlap、\(q(r)\) 与 \(\pi\) 定义与缺失值策略
- coarse（spectral v2）：\(\lambda\)、嵌入维度 \(d\)、`knn_k`、HDBSCAN 版本信息、singleton/isolated 统计、contact component postprocess 统计
- refine：\(\theta\) 聚合方法摘要、prior/regularizer 是否启用、阈值（core/ambiguous/accessory）与输出文件路径

> 设计原则：**inference 与 export 解耦**。export 的 “≥200kb” 等展示/导出策略不应反向定义推断逻辑。

---

## 8. 参考文献（可在论文中引用）

- Zhou, D., Huang, J., & Schölkopf, B. (2007). *Learning with Hypergraphs: Clustering, Classification, and Embedding*. NIPS.
- Shi, J., & Malik, J. (2000). *Normalized Cuts and Image Segmentation*. IEEE TPAMI.
- Ng, A. Y., Jordan, M. I., & Weiss, Y. (2002). *On Spectral Clustering: Analysis and an algorithm*. NeurIPS.
- Schwarz, G. (1978). *Estimating the dimension of a model*. Annals of Statistics. (BIC)
