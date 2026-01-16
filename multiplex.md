# Project HyperBin-X: 基于多层超图耦合谱聚类的宏基因组分箱算法

## 1. 项目背景与核心理念 (Core Philosophy)

### 1.1 问题定义
我们要开发一个宏基因组分箱 (Binning) 工具，旨在解决单一模态数据的局限性：
*   **Pore-C (物理视图)**: 提供高精度的长距离接触信息，但数据稀疏，存在大量孤立点。
*   **TNF (化学视图)**: 提供全覆盖的序列组成特征，但分辨率低，难以区分种内差异。

### 1.2 解决方案：双层多路复用网络 (Bi-layer Multiplex Network)
我们不采用简单的特征拼接 (Early Fusion)，而是构建一个**双层网络模型**，通过**超拉普拉斯矩阵 (Supra-Laplacian)** 将两层耦合。
*   **理论依据**: *Mucha et al. (Science, 2010)* 提出的多层网络扩散动力学。
*   **核心优势**: 允许物理层和化学层保留各自的拓扑结构，同时通过耦合参数 $\beta$ 强制同一 Contig 在两层中的嵌入坐标趋同。

---

## 2. 核心数学模型 (Mathematical Framework)

### 2.1 超级矩阵构造 (The Supra-Laplacian)
系统建模为 $2N \times 2N$ 的稀疏矩阵 $\mathcal{L}_{supra}$ ($N$ 为 Contig 数量)：

$$
\mathcal{L}_{supra} = \begin{pmatrix}
\Delta_{phy} + \beta I & -\beta I \\
-\beta I & \Delta_{chem} + \beta I
\end{pmatrix}
$$

*   **$\Delta_{phy}$**: 物理层的归一化超图拉普拉斯矩阵。
*   **$\Delta_{chem}$**: 化学层的归一化超图拉普拉斯矩阵。
*   **$\beta$ (耦合系数)**: 默认设为 `0.5`。物理意义为层间扩散系数，起到“垂直弹簧”的作用。
*   **$I$**: 单位矩阵。

### 2.2 单层超图定义
对于每一层，采用 *Zhou et al. (2006)* 定义的拉普拉斯算子：
$$ \Delta = I - D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} $$

---

## 3. 详细实现步骤 (Implementation Steps)

请使用 Python (`scipy.sparse`, `numpy`, `sklearn`) 实现以下模块。**严禁将稀疏矩阵转为稠密矩阵 (.toarray())**。
# Project HyperBin-X: 基于多层超图耦合谱聚类的宏基因组分箱算法

## 1. 项目背景与核心理念 (Core Philosophy)

### 1.1 问题定义
我们要开发一个宏基因组分箱 (Binning) 工具，旨在解决单一模态数据的局限性：
*   **Pore-C (物理视图)**: 提供高精度的长距离接触信息，但数据稀疏，存在大量孤立点。
*   **TNF (化学视图)**: 提供全覆盖的序列组成特征，但分辨率低，难以区分种内差异。

### 1.2 解决方案：双层多路复用网络 (Bi-layer Multiplex Network)
我们不采用简单的特征拼接 (Early Fusion)，而是构建一个**双层网络模型**，通过**超拉普拉斯矩阵 (Supra-Laplacian)** 将两层耦合。
*   **理论依据**: *Mucha et al. (Science, 2010)* 提出的多层网络扩散动力学。
*   **核心优势**: 允许物理层和化学层保留各自的拓扑结构，同时通过耦合参数 $\beta$ 强制同一 Contig 在两层中的嵌入坐标趋同。

---

## 2. 核心数学模型 (Mathematical Framework)

### 2.1 超级矩阵构造 (The Supra-Laplacian)
系统建模为 $2N \times 2N$ 的稀疏矩阵 $\mathcal{L}_{supra}$ ($N$ 为 Contig 数量)：

$$
\mathcal{L}_{supra} = \begin{pmatrix}
\Delta_{phy} + \beta I & -\beta I \\
-\beta I & \Delta_{chem} + \beta I
\end{pmatrix}
$$

*   **$\Delta_{phy}$**: 物理层的归一化超图拉普拉斯矩阵。
*   **$\Delta_{chem}$**: 化学层的归一化超图拉普拉斯矩阵。
*   **$\beta$ (耦合系数)**: 默认设为 `0.5`。物理意义为层间扩散系数，起到“垂直弹簧”的作用。
*   **$I$**: 单位矩阵。

### 2.2 单层超图定义
对于每一层，采用 *Zhou et al. (2006)* 定义的拉普拉斯算子：
$$ \Delta = I - D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} $$

---

## 3. 详细实现步骤 (Implementation Steps)

请使用 Python (`scipy.sparse`, `numpy`, `sklearn`) 实现以下模块。**严禁将稀疏矩阵转为稠密矩阵 (.toarray())**。

### 模块 1: 数据预处理与权重设计 (Preprocessing)
*   **输入**: `contigs.fasta`, `alignment.paf`.
*   **TNF 计算**: 计算并归一化 136维 TNF 向量。
*   **权重设计 (Key Requirement)**:
    *   **物理层权重 ($W_{phy}$)**: 使用用户提供的**原创比对系数** (从 PAF/BAM 中提取或计算)。
    *   **化学层权重 ($W_{chem}$)**: 必须反映“相似度”。在 KNN 建图时，推荐使用**高斯核权重 (Gaussian Kernel)** 或 **反距离权重 (Inverse Distance)**。
        *   **公式**: $w_{ij} = \exp(-\frac{||x_i - x_j||^2}{2\sigma^2})$ 或 $w_{ij} = \frac{1}{1 + \text{dist}(i, j)}$
        *   **理由**: 距离越近的邻居，权重越大，代表这条“化学键”越强。这与物理层的“比对系数越高越可靠”在逻辑上是对齐的。

### 模块 2: 双层超图构建 (Graph Construction)
编写函数 `build_laplacian(incidence_matrix, weights)`:

1.  **构建 $H_{phy}$**: 行=Contig, 列=Read。
    *   利用 `scipy.sparse.lil_matrix` 构建。
    *   填入物理权重。
2.  **构建 $H_{chem}$**:
    *   使用 `sklearn.NearestNeighbors(k=5)`。
    *   构建 $N \times N$ 矩阵，第 $j$ 列代表以 Contig $j$ 为中心的超边。
    *   填入化学权重 (反距离)。
3.  **计算 $\Delta_{phy}$ 和 $\Delta_{chem}$**:
    *   严格按照 2.2 中的公式进行稀疏矩阵运算。
    *   **注意**: 处理度为 0 的孤立节点 (避免除零错误)。

### 模块 3: 多层耦合与特征分解 (Coupling & Embedding)
编写函数 `run_multiplex_embedding(L_phy, L_chem, beta=0.5, n_clusters=K)`:

1.  **拼接大矩阵**:
    *   使用 `scipy.sparse.bmat` 按照 2.1 的公式构建 $\mathcal{L}_{supra}$。
    *   **$\beta$ 选择**: 这是一个平衡参数。
        *   **推荐策略**: 不要只选 0.5。建议设置一个**默认范围** (例如 `[0.1, 0.5, 1.0, 2.0]`)。
        *   **理由**: 不同的数据集，物理信号和化学信号的强弱不同。如果 Pore-C 数据质量极高，$\beta$ 应该小一点（让物理层主导）；如果 Pore-C 很稀疏，$\beta$ 应该大一点（强行拉近化学层相似的点）。
        *   **开发建议**: 写一个简单的 Grid Search 循环，用内部指标（如 Silhouette Score）自动选最佳的 $\beta$。
2.  **特征分解 (Eigendecomposition)**:
    *   使用 `scipy.sparse.linalg.eigsh`。
    *   求解 $K$ 个**最小**的非零特征向量 (Smallest Algebraic)。
    *   输出矩阵 $U$ 形状为 $(2N, K)$。

### 模块 4: 坐标融合与聚类 (Clustering)
**这是谱聚类的关键步骤：**

1.  **坐标分离**:
    *   从 $2N \times K$ 的特征向量矩阵 $U$ 中拆分：
    *   $U_{phy} = U[0:N, :]$ (前 N 行)
    *   $U_{chem} = U[N:2N, :]$ (后 N 行)
2.  **融合 (Fusion)**:
    *   $U_{final} = (U_{phy} + U_{chem}) / 2$
    *   *注: 这一步融合了物理拓扑和化学拓扑的信息。*
3.  **行归一化 (Row Normalization)**:
    *   **至关重要**: 对 $U_{final}$ 的每一行进行 L2 归一化。
    *   $$ U_{norm}[i] = \frac{U_{final}[i]}{||U_{final}[i]||_2} $$
4.  **K-Means**:
    *   对 $U_{norm}$ 进行 K-Means 聚类。
    *   得到 Labels，即为最终的分箱结果。

---

## 4. 代码质量要求 (Quality Control)
1.  **内存优化**: 始终保持 `csr_matrix` 格式，只在必要时 (如特征向量计算后) 转为 `numpy array`。
2.  **模块化**: 将 `Laplacian Construction`, `Coupling`, `Clustering` 封装为独立的类或函数。
3.  **鲁棒性**: 必须处理图中可能存在的 **不连通分量 (Disconnected Components)**，防止特征分解不收敛。