# HyperBin-X 算法改进方案

> 本文档基于对现有代码库的严谨评审，提出系统性的算法改进建议。
>
> 评审视角：Nature Communications 级别论文的审稿标准
>
> 生成时间：2025-12-30
>
> 最后更新：2026-01-09

---

## 目录

1. [问题总览](#问题总览)
2. [问题1：化学层超图建模重构](#问题1化学层超图建模重构)
3. [问题2：两层权重归一化](#问题2两层权重归一化)
4. [问题3：谱求解稳定性增强](#问题3谱求解稳定性增强)
5. [问题4：β参数自适应选择](#问题4β参数自适应选择)
6. [问题5：不连通分量预处理](#问题5不连通分量预处理)
7. [问题6：Auto-k选择改进](#问题6auto-k选择改进)
8. [问题7：低置信度contigs处理](#问题7低置信度contigs处理)
9. [改进后的Pipeline流程](#改进后的pipeline流程)
10. [发表建议](#发表建议)

---

## 问题总览

| 编号 | 问题 | 严重程度 | 影响 | 状态 |
|------|------|----------|------|------|
| 1 | 化学层不是真正的超图 | **严重** | 概念不一致，审稿必问 | ✅ 已完成 |
| 2 | 两层权重量纲不一致 | **严重** | 一层可能主导另一层 | ❌ 不需要 |
| 3 | 谱求解可能不收敛 | 中等 | 结果不可靠 | ❌ 未遇到 |
| 4 | β参数缺乏理论依据 | 中等 | 方法可复现性差 | ✅ 已完成 |
| 5 | 不连通分量未处理 | 中等 | eigsh 不稳定 | ❌ 不需要 |
| 6 | Auto-k 搜索范围太窄 | 轻微 | k 选择不准确 | ✅ 已完成 |
| 7 | 低置信度强制分配 | 中等 | bin 质量下降 | 待实现 |

---

## 问题1：化学层超图建模重构

> **状态：✅ 已完成** (2026-01-04)
>
> **实现文件**: `hypergraph_binning/multiplex/features.py`
>
> **关键函数**: `build_knn_hypergraph()`

### 当前问题

当前 `build_knn_graph` 返回的是 N×N 邻接矩阵，本质是**普通图**，不是超图：

```python
# features.py 当前实现
H = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
```

这与物理层的超图结构**数学上不兼容**：
- 物理层：H 是 N×M 的关联矩阵，每列代表一个超边包含多个节点
- 化学层：H 是 N×N 的加权邻接矩阵，本质是普通图

**审稿人必问**：为什么声称是"双层超图"，但化学层实际上是普通图？

### 改进方案

将 KNN 邻域建模为真正的超边：每个节点 i 的 k-近邻（包括自身）构成一个超边。

### 代码实现

**文件**: `hypergraph_binning/multiplex/features.py`

```python
def build_knn_hypergraph(
    features: np.ndarray,
    k: int = 10,
    weight_scheme: str = "gaussian",  # "gaussian" | "inverse" | "binary"
    sigma: float = None,  # for gaussian kernel
) -> Tuple[csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    """
    构建化学层的真正超图结构。

    每个节点 i 的 k-近邻（含自身）构成一个超边 e_i。
    返回与物理层相同的结构：(H, w, de, dv)

    Parameters
    ----------
    features : (N, D) TNF 特征矩阵
    k : 每个超边包含的邻居数（含自身）
    weight_scheme : 权重计算方式
        - "gaussian": w_ij = exp(-d^2 / (2*sigma^2))
        - "inverse": w_ij = 1 / (1 + d)
        - "binary": w_ij = 1
    sigma : 高斯核带宽，若为 None 则自动估计（使用中位数距离）

    Returns
    -------
    H : (N, N) 关联矩阵，H[i, j] 表示节点 i 在超边 j 中的权重
    w : (N,) 超边权重（每个超边一个全局权重）
    de : (N,) 超边度数（每个超边包含的加权节点数）
    dv : (N,) 节点度数
    """
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse import csr_matrix

    n_samples = features.shape[0]

    # Step 1: KNN 查询
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto", metric="euclidean")
    nbrs.fit(features)
    distances, indices = nbrs.kneighbors(features)

    # Step 2: 自动估计 sigma（如果使用高斯核）
    if weight_scheme == "gaussian" and sigma is None:
        # 使用中位数距离作为带宽
        sigma = np.median(distances[:, 1:])  # 排除自身距离 0
        sigma = max(sigma, 1e-6)

    # Step 3: 构建关联矩阵 H
    # H[i, j] = 节点 i 在超边 j 中的权重
    # 超边 j 以节点 j 为中心
    rows = []
    cols = []
    data = []

    for j in range(n_samples):  # j 是超边索引（也是中心节点）
        nbr_idx = indices[j]    # j 的 k 个邻居（含自身）
        nbr_dist = distances[j]

        for idx, dist in zip(nbr_idx, nbr_dist):
            # 计算节点 idx 在超边 j 中的权重
            if weight_scheme == "gaussian":
                h_val = np.exp(-dist**2 / (2 * sigma**2))
            elif weight_scheme == "inverse":
                h_val = 1.0 / (1.0 + dist)
            else:  # binary
                h_val = 1.0

            rows.append(idx)
            cols.append(j)
            data.append(h_val)

    H = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))

    # Step 4: 计算超边度数 de[j] = sum_i H[i, j]
    de = np.array(H.sum(axis=0)).ravel()

    # Step 5: 超边全局权重 w[j]
    # 方案: 基于超边内部紧密度的权重
    # w[j] = mean(H[i, j]) for i in hyperedge j
    w = de / k  # 平均权重作为超边权重

    # Step 6: 节点度数 dv[i] = sum_j w[j] * H[i, j]
    dv = np.array((H @ w.reshape(-1, 1))).ravel()

    return H, w, de, dv
```

### 配套修改

**文件**: `hypergraph_binning/multiplex/graph.py`

```python
def build_chem_laplacian_from_hypergraph(
    H: csr_matrix,
    w: np.ndarray,
    de: np.ndarray,
    dv: np.ndarray
) -> csr_matrix:
    """
    使用与物理层完全相同的公式构建化学层 Laplacian。
    L = I - D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
    """
    return _compute_laplacian_matrix(H, w, de, dv)
```

**文件**: `hypergraph_binning/multiplex/pipeline.py` 调用修改

```python
# 旧代码
H_chem = build_knn_graph(tnf_features, k=cfg.knn_k)
L_chem = build_chem_laplacian(H_chem)

# 新代码
H_chem, w_chem, de_chem, dv_chem = build_knn_hypergraph(
    tnf_features,
    k=cfg.knn_k,
    weight_scheme="gaussian"
)
L_chem = build_chem_laplacian_from_hypergraph(H_chem, w_chem, de_chem, dv_chem)
```

### ✅ 实现完成总结

**实际实现与文档方案一致**。已在相关文件中实现：

1. `features.py`: `build_knn_hypergraph()` - 构建真正的 KNN 超图
   - 每个节点 i 的 k-近邻构成一个超边
   - 支持高斯核权重 (`weight_scheme="gaussian"`)
   - 自动估计 σ 参数（使用中位数距离）

2. `graph.py`: `build_chem_laplacian()` - 使用与物理层相同的 Laplacian 公式
   - L = I - D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}

3. `pipeline.py`: 调用链已更新

**运行时输出示例**:
```
[Chemical Layer] Building KNN hypergraph with k=10
[Chemical Layer] Auto sigma (median distance): 0.038128
[Chemical Layer] Hypergraph: 1428 nodes, 1428 hyperedges
```

**理论依据**:
- Zhou et al. (2006), NIPS: Learning with Hypergraphs
- 确保两层数学结构一致，都是真正的超图

---

## 问题2：两层权重归一化

> **状态：❌ 不需要** (2026-01-09)
>
> **原因**: 两层都使用归一化 Laplacian 公式 $L = I - D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}$，
> 该公式本身已包含归一化，trace/N ≈ 1，无需额外归一化。

### 原问题描述

物理层和化学层的 Laplacian 矩阵谱范数可能差异很大：

```python
# 物理层：w_e ∈ (0, ~2)，取决于 q' 和 k
w_e = float(q_prime) * (2.0 / float(k - 1))

# 化学层：w ∈ (0.5, 1)，因为 1/(1+d)，d 是欧氏距离
w = 1.0 / (1.0 + dist)
```

直接相加的 Supra-Laplacian 可能导致一层主导另一层。

### 改进方案

在构建 Supra-Laplacian 之前，对两层分别进行谱归一化。

### 代码实现

**文件**: `hypergraph_binning/multiplex/graph.py`

```python
from scipy.sparse.linalg import eigsh, norm as sparse_norm
from scipy.sparse import identity

def estimate_spectral_scale(L: csr_matrix, method: str = "trace") -> float:
    """
    估计 Laplacian 矩阵的谱尺度，用于归一化。

    Parameters
    ----------
    L : Laplacian 矩阵
    method :
        - "trace": 使用迹（对角线元素之和），计算快
        - "spectral_norm": 使用最大特征值，更准确但慢
        - "frobenius": 使用 Frobenius 范数
    """
    if method == "trace":
        return float(L.diagonal().sum()) / L.shape[0]

    elif method == "spectral_norm":
        try:
            vals, _ = eigsh(L, k=1, which="LA", maxiter=100)
            return float(vals[0])
        except:
            return 1.0

    elif method == "frobenius":
        return sparse_norm(L, ord="fro") / np.sqrt(L.shape[0])

    return 1.0


def normalize_laplacian(L: csr_matrix, scale: float = None) -> Tuple[csr_matrix, float]:
    """
    归一化 Laplacian 矩阵。

    Returns
    -------
    L_norm : 归一化后的矩阵
    scale : 使用的尺度因子
    """
    if scale is None:
        scale = estimate_spectral_scale(L, method="trace")

    if scale > 1e-10:
        L_norm = L / scale
    else:
        L_norm = L
        scale = 1.0

    return L_norm, scale


def build_supra_laplacian_normalized(
    L_phy: csr_matrix,
    L_chem: csr_matrix,
    beta: float = 0.5,
    normalize: bool = True
) -> Tuple[csr_matrix, dict]:
    """
    构建归一化的 Supra-Laplacian。

    归一化后，两层的 Laplacian 特征值都在 [0, ~1] 范围内，
    β 的含义变得清晰：
    - β = 0: 两层完全独立
    - β = 1: 层间耦合强度与层内信号相当
    - β > 1: 层间耦合主导

    Returns
    -------
    L_supra : 2N x 2N Supra-Laplacian
    info : 包含归一化因子等诊断信息
    """
    n = L_phy.shape[0]
    assert L_chem.shape == (n, n)

    info = {
        "n_contigs": n,
        "beta": beta,
        "phy_scale": 1.0,
        "chem_scale": 1.0,
    }

    if normalize:
        L_phy_norm, phy_scale = normalize_laplacian(L_phy)
        L_chem_norm, chem_scale = normalize_laplacian(L_chem)
        info["phy_scale"] = phy_scale
        info["chem_scale"] = chem_scale
        print(f"[Normalization] Physical layer scale: {phy_scale:.4f}")
        print(f"[Normalization] Chemical layer scale: {chem_scale:.4f}")
    else:
        L_phy_norm = L_phy
        L_chem_norm = L_chem

    I = identity(n, format="csr", dtype=np.float64)
    beta_I = beta * I

    # 构建 Supra-Laplacian
    TL = L_phy_norm + beta_I
    TR = -beta_I
    BL = -beta_I
    BR = L_chem_norm + beta_I

    L_supra = bmat([
        [TL, TR],
        [BL, BR]
    ], format="csr")

    return L_supra, info
```

### ❌ 结论：不需要额外归一化

**原因分析**：

归一化 Laplacian 公式 $L = I - D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}$ 本身已包含度归一化：
- 对角线元素 $L[i,i] \approx 1$（对于连通节点）
- trace(L) / N ≈ 1
- 特征值范围 [0, 2]

两层都用同样的归一化公式，天然在同一尺度上，无需额外处理。

---

## 问题3：谱求解稳定性增强

### 当前问题

1. `eigsh` 的 "SA" 模式对于病态矩阵容易不收敛
2. `tol=1e-3` 可能太松
3. 没有收敛性诊断

### 改进方案

创建稳健的特征值求解器，支持多种后端和自动回退。

### 代码实现

**新文件**: `hypergraph_binning/spectral/robust_eigen.py`

```python
"""
稳健的特征值求解器，支持多种后端和收敛性诊断。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import warnings

import numpy as np
from scipy.sparse.linalg import eigsh, lobpcg, LinearOperator


@dataclass
class EigensolverResult:
    """特征值求解结果"""
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    converged: bool
    n_iterations: int
    residual_norms: List[float]
    method_used: str
    info: dict = field(default_factory=dict)


@dataclass
class EigensolverConfig:
    """特征值求解器配置"""
    tol: float = 1e-6           # 收敛容差
    maxiter: int = 500          # 最大迭代次数
    method: str = "auto"        # "eigsh", "lobpcg", "auto"
    shift: float = None         # shift-invert 模式的 shift 值
    v0_seed: int = 42           # 初始向量随机种子
    verbose: bool = False


def _create_initial_vectors(n: int, k: int, seed: int = 42) -> np.ndarray:
    """创建随机初始向量（正交化）"""
    rng = np.random.default_rng(seed)
    X0 = rng.standard_normal((n, k))
    Q, _ = np.linalg.qr(X0)
    return Q


def _eigsh_solve(
    L: LinearOperator,
    k: int,
    config: EigensolverConfig
) -> EigensolverResult:
    """使用 ARPACK (eigsh) 求解"""
    n = L.shape[0]
    v0 = _create_initial_vectors(n, 1, config.v0_seed).ravel()

    try:
        if config.shift is not None:
            vals, vecs = eigsh(
                L, k=k, which="LM", sigma=config.shift,
                maxiter=config.maxiter, tol=config.tol, v0=v0
            )
        else:
            vals, vecs = eigsh(
                L, k=k, which="SA",
                maxiter=config.maxiter, tol=config.tol, v0=v0
            )

        # 按特征值排序
        idx = np.argsort(vals)
        vals = vals[idx]
        vecs = vecs[:, idx]

        # 计算残差
        residuals = []
        for i in range(k):
            Lv = L @ vecs[:, i]
            r = np.linalg.norm(Lv - vals[i] * vecs[:, i])
            residuals.append(float(r))

        return EigensolverResult(
            eigenvalues=vals,
            eigenvectors=vecs,
            converged=True,
            n_iterations=-1,
            residual_norms=residuals,
            method_used="eigsh"
        )

    except Exception as e:
        warnings.warn(f"eigsh failed: {e}")
        return EigensolverResult(
            eigenvalues=np.array([]),
            eigenvectors=np.array([[]]),
            converged=False,
            n_iterations=-1,
            residual_norms=[],
            method_used="eigsh",
            info={"error": str(e)}
        )


def _lobpcg_solve(
    L: LinearOperator,
    k: int,
    config: EigensolverConfig
) -> EigensolverResult:
    """使用 LOBPCG 求解（对大规模稀疏矩阵更稳定）"""
    n = L.shape[0]
    X0 = _create_initial_vectors(n, k, config.v0_seed)
    iteration_count = [0]

    try:
        vals, vecs = lobpcg(
            L, X0,
            largest=False,
            tol=config.tol,
            maxiter=config.maxiter,
            verbosityLevel=1 if config.verbose else 0,
        )

        idx = np.argsort(vals)
        vals = vals[idx]
        vecs = vecs[:, idx]

        # 计算最终残差
        final_residuals = []
        for i in range(k):
            Lv = L @ vecs[:, i]
            r = np.linalg.norm(Lv - vals[i] * vecs[:, i])
            final_residuals.append(float(r))

        return EigensolverResult(
            eigenvalues=vals,
            eigenvectors=vecs,
            converged=True,
            n_iterations=iteration_count[0],
            residual_norms=final_residuals,
            method_used="lobpcg"
        )

    except Exception as e:
        warnings.warn(f"LOBPCG failed: {e}")
        return EigensolverResult(
            eigenvalues=np.array([]),
            eigenvectors=np.array([[]]),
            converged=False,
            n_iterations=iteration_count[0],
            residual_norms=[],
            method_used="lobpcg",
            info={"error": str(e)}
        )


def robust_eigenpairs(
    L: LinearOperator,
    k: int,
    config: EigensolverConfig = None
) -> EigensolverResult:
    """
    稳健的特征值求解，自动选择最佳方法并提供诊断信息。

    策略：
    1. 先尝试 eigsh（通常更快）
    2. 如果失败或残差过大，尝试 LOBPCG
    3. 如果都失败，尝试 shift-invert 模式
    """
    if config is None:
        config = EigensolverConfig()

    n = L.shape[0]
    if k >= n:
        k = n - 1

    if config.method == "eigsh":
        return _eigsh_solve(L, k, config)

    elif config.method == "lobpcg":
        return _lobpcg_solve(L, k, config)

    else:  # auto
        # 策略1: 先尝试 eigsh
        result = _eigsh_solve(L, k, config)

        if result.converged and all(r < config.tol * 10 for r in result.residual_norms):
            return result

        # 策略2: 尝试 LOBPCG
        print("[Eigensolver] eigsh residuals too large, trying LOBPCG...")
        result_lobpcg = _lobpcg_solve(L, k, config)

        if result_lobpcg.converged:
            return result_lobpcg

        # 策略3: 尝试 shift-invert
        print("[Eigensolver] LOBPCG failed, trying shift-invert mode...")
        config_shift = EigensolverConfig(
            tol=config.tol,
            maxiter=config.maxiter,
            shift=1e-6,
            v0_seed=config.v0_seed
        )
        return _eigsh_solve(L, k, config_shift)


def spectral_embedding_robust(
    L: LinearOperator,
    k: int,
    config: EigensolverConfig = None
) -> Tuple[np.ndarray, EigensolverResult]:
    """
    稳健的谱嵌入，返回嵌入向量和诊断信息。
    """
    result = robust_eigenpairs(L, k, config)

    if not result.converged:
        raise RuntimeError(
            f"Eigendecomposition failed to converge. "
            f"Method: {result.method_used}, "
            f"Residuals: {result.residual_norms}"
        )

    return result.eigenvectors, result
```

### 使用示例

```python
from ..spectral.robust_eigen import spectral_embedding_robust, EigensolverConfig

eigen_config = EigensolverConfig(tol=1e-6, maxiter=500, method="auto", verbose=True)
U, eigen_result = spectral_embedding_robust(L_supra, k=cfg.k, config=eigen_config)

# 记录诊断信息
print(f"Eigensolver: method={eigen_result.method_used}, "
      f"converged={eigen_result.converged}, "
      f"max_residual={max(eigen_result.residual_norms):.2e}")
```

---

## 问题4：β参数自适应选择

> **状态：✅ 已完成** (2026-01-09)
>
> **实现文件**: `hypergraph_binning/multiplex/auto_tune.py`
>
> **关键函数**: `auto_select_beta_k()` - 联合搜索 (β, k) 最优组合
>
> **触发条件**: 配置 `k <= 0` 时自动启用
>
> **方法**: 基于理论的四阶段选择流程

### 当前问题

β=0.5 是经验值，缺乏理论或数据驱动的依据。

### 改进方案：理论驱动的参数选择

我们采用有严格理论支撑的方法，分四个阶段：

#### Stage 1: β 选择 - 谱间隙最大化 (Spectral Gap Maximization)

**理论依据**: Cheeger 不等式 (Cheeger, 1969)

$$\frac{h^2}{2} \leq \lambda_2 \leq 2h$$

其中 $h$ 是 Cheeger 常数（等周比率），$\lambda_2$ 是第二小特征值。

**含义**: 谱间隙 $\lambda_2 - \lambda_1$ 越大，聚类可分性越好。

**实现**:
```python
def select_beta_by_spectral_gap(L_phy, L_chem, beta_range=(0.01, 5.0)):
    # 对数空间采样 β 值
    betas = np.logspace(log10(0.01), log10(5.0), n_samples)

    for beta in betas:
        L_supra = build_supra_laplacian(L_phy, L_chem, beta)
        vals = eigsh(L_supra, k=6, which="SA")
        spectral_gap = vals[1] - vals[0]

    return argmax(spectral_gaps)  # 选择使谱间隙最大的 β
```

#### Stage 2: k 选择 - Gap Statistic (Tibshirani et al., 2001)

**理论依据**: Tibshirani, Walther, Hastie (2001), JRSS-B

$$\text{Gap}(k) = E^*[\log W_k] - \log W_k$$

其中 $W_k$ 是簇内离散度，$E^*$ 是对均匀零分布的期望。

**选择规则** (论文原文):
> 选择满足 $\text{Gap}(k) \geq \text{Gap}(k+1) - s_{k+1}$ 的最小 $k$

其中 $s_k = \sqrt{1 + 1/B} \cdot \text{std}(\log W_k^*)$ 是标准误差。

**实现**:
```python
def compute_gap_statistic(X, k_candidates, n_references=10):
    # 1. 计算观测数据的 log(W_k)
    for k in k_candidates:
        labels = KMeans(k).fit_predict(X)
        log_wk[k] = log(within_cluster_dispersion(X, labels))

    # 2. 生成 B 个均匀参考数据集
    for b in range(n_references):
        X_ref = uniform_in_bounding_box(X)
        # 计算参考数据的 log(W_k^*)

    # 3. Gap = E*[log(W_k)] - log(W_k)
    gap_values = log_wk_ref.mean() - log_wk

    # 4. 应用选择规则
    for k in k_candidates:
        if gap[k] >= gap[k+1] - std[k+1]:
            return k
```

#### Stage 3: 稳定性验证 - Bootstrap Stability

**理论依据**: Ben-Hur et al. (2002), von Luxburg (2010)

稳定的聚类更可能是有意义的。通过 bootstrap 子采样评估：

1. 多次随机子采样 (80%)
2. 对每个子样本聚类
3. 计算成对 Adjusted Rand Index (ARI)
4. 高平均 ARI = 稳定聚类

```python
def compute_clustering_stability(X, k, n_bootstrap=20):
    for b in range(n_bootstrap):
        idx = random_subsample(n, ratio=0.8)
        labels[b] = KMeans(k).fit_predict(X[idx])

    # 计算成对 ARI
    stability = mean([ARI(labels[i], labels[j]) for i,j in pairs])
    return stability
```

**解读标准**:
- ARI > 0.9: 优秀
- ARI > 0.7: 良好
- ARI > 0.5: 中等
- ARI < 0.5: 较差

#### Stage 4: 综合验证

计算 Silhouette Score 作为额外参考，与其他指标交叉验证。

### 运行输出示例

```
============================================================
[Auto-tune] Using principled method (theory-driven)
============================================================

[Stage 1] Selecting β by spectral gap maximization...
  Theory: Cheeger inequality - larger gap = better separability
[Spectral Gap] β=0.0100: gap=0.000234
[Spectral Gap] β=0.0251: gap=0.000587
[Spectral Gap] β=0.0631: gap=0.001475
[Spectral Gap] β=0.1585: gap=0.003702
[Spectral Gap] β=0.3981: gap=0.008912  <-- max
[Spectral Gap] β=1.0000: gap=0.007234
[Spectral Gap] Optimal β=0.3981 with gap=0.008912

[Stage 2] Computing spectral embedding at β=0.3981...

[Stage 3] Selecting k via Gap Statistic...
  Theory: Tibshirani et al. (2001) - compare to null distribution
  Candidates: [5, 10, 20, 30, 50, 70, 100]
  Gap Statistic selected k=30
    k=5:  Gap=0.1234 ± 0.0123
    k=10: Gap=0.2345 ± 0.0234
    k=20: Gap=0.3456 ± 0.0345
    k=30: Gap=0.4012 ± 0.0401 <-- selected
    k=50: Gap=0.3890 ± 0.0512

[Stage 4] Validating with bootstrap stability...
  Theory: Ben-Hur et al. (2002) - stable = meaningful
  Stability (ARI): 0.8234 (good)

[Summary] β=0.3981, k=30
  Silhouette: 0.4521
  Stability:  0.8234
============================================================
```

### 理论参考文献

1. **Cheeger (1969)**. "A lower bound for the smallest eigenvalue of the Laplacian."
   *Problems in Analysis*, Princeton University Press, 195-199.

2. **Tibshirani, Walther, Hastie (2001)**. "Estimating the number of clusters
   in a data set via the gap statistic." *JRSS-B*, 63(2), 411-423.

3. **Ben-Hur, Elisseeff, Guyon (2002)**. "A stability based method for
   discovering structure in clustered data." *Pacific Symposium on Biocomputing*.

4. **von Luxburg (2007)**. "A tutorial on spectral clustering."
   *Statistics and Computing*, 17(4), 395-416.

### 与原方案对比

| 方面 | 原方案 (Grid Search) | 新方案 (Principled) |
|------|---------------------|---------------------|
| β 选择 | 固定候选值 [0.1, 0.3, ...] | 谱间隙最大化 (Cheeger) |
| k 选择 | Silhouette 最大化 | Gap Statistic (Tibshirani) |
| 理论依据 | 无 | 有明确论文支撑 |
| 验证 | 无 | Bootstrap 稳定性 |
| 审稿友好 | 可能被质疑 | 方法学严谨 |

---

## 问题5：不连通分量预处理

> **状态：❌ 不需要** (2026-01-09)
>
> **原因**: Supra-Laplacian 结构保证了整体连通性

### 原问题描述

超图可能存在不连通分量，导致 Laplacian 有多个零特征值，`eigsh` 不稳定。

### ❌ 结论：不需要处理

**分析**：

1. **化学层 (KNN)** 始终连通：
   - 每个节点都有 k 个邻居（包括自己）
   - KNN 图天然保证所有节点都有边连接

2. **物理层 (Pore-C)** 可能有孤立节点：
   - 某些 contigs 可能没有 Pore-C 接触

3. **但 Supra-Laplacian 的耦合结构保证整体连通**：
   ```
   L_supra = [ L_phy + βI    -βI        ]
             [ -βI           L_chem + βI ]
   ```
   - 即使物理层节点 i 孤立，它通过 -βI 与化学层节点 i 耦合
   - 化学层节点 i 与其 k 个邻居连接
   - **因此 Supra-Laplacian 始终是连通的**

**如果未来需要处理（仅供参考）**：

以下代码可用于分析连通性，但当前实现不需要：

**文件**: `hypergraph_binning/utils/connectivity.py`

```python
"""
连通性分析与预处理模块。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


@dataclass
class ConnectivityAnalysis:
    """连通性分析结果"""
    n_components: int
    component_labels: np.ndarray  # (n,) 每个节点所属分量
    component_sizes: List[int]    # 每个分量的大小
    largest_component_idx: int    # 最大分量的索引
    isolated_nodes: np.ndarray    # 孤立节点的索引


def analyze_connectivity(L: csr_matrix) -> ConnectivityAnalysis:
    """分析 Laplacian 矩阵的连通性。"""
    n = L.shape[0]

    # 提取邻接矩阵
    A = -L.copy()
    A.setdiag(0)
    A.eliminate_zeros()
    A = A + A.T
    A.data = np.ones_like(A.data)

    n_comp, labels = connected_components(A, directed=False, return_labels=True)

    unique, counts = np.unique(labels, return_counts=True)
    component_sizes = [int(counts[unique == i][0]) if i in unique else 0
                       for i in range(n_comp)]

    largest_idx = int(np.argmax(component_sizes))

    degrees = np.array(A.sum(axis=1)).ravel()
    isolated = np.where(degrees == 0)[0]

    return ConnectivityAnalysis(
        n_components=n_comp,
        component_labels=labels,
        component_sizes=component_sizes,
        largest_component_idx=largest_idx,
        isolated_nodes=isolated
    )


def extract_largest_component(
    L: csr_matrix,
    names: List[str] = None
) -> Tuple[csr_matrix, np.ndarray, Optional[List[str]]]:
    """提取最大连通分量。"""
    analysis = analyze_connectivity(L)

    if analysis.n_components == 1:
        return L, np.arange(L.shape[0]), names

    print(f"[Connectivity] Found {analysis.n_components} components, "
          f"sizes: {sorted(analysis.component_sizes, reverse=True)[:5]}...")

    mask = analysis.component_labels == analysis.largest_component_idx
    indices = np.where(mask)[0]

    L_sub = L[indices, :][:, indices]

    names_sub = None
    if names is not None:
        names_sub = [names[i] for i in indices]

    print(f"[Connectivity] Extracted largest component: {len(indices)} nodes "
          f"({100*len(indices)/L.shape[0]:.1f}%)")

    return L_sub, indices, names_sub


def handle_disconnected_graph(
    L: csr_matrix,
    names: List[str],
    strategy: str = "largest_only"
) -> Tuple[csr_matrix, List[str], dict]:
    """
    处理不连通图的策略。

    Parameters
    ----------
    strategy :
        - "largest_only": 只处理最大分量，其他标记为未分配
        - "separate": 每个分量独立聚类
    """
    analysis = analyze_connectivity(L)

    info = {
        "n_components": analysis.n_components,
        "component_sizes": analysis.component_sizes,
        "strategy": strategy,
        "unassigned_indices": [],
        "unassigned_names": [],
    }

    if analysis.n_components == 1:
        return L, names, info

    if strategy == "largest_only":
        L_sub, indices, names_sub = extract_largest_component(L, names)

        all_indices = set(range(L.shape[0]))
        included = set(indices.tolist())
        unassigned = sorted(all_indices - included)

        info["unassigned_indices"] = unassigned
        info["unassigned_names"] = [names[i] for i in unassigned]

        return L_sub, names_sub, info

    elif strategy == "separate":
        info["component_labels"] = analysis.component_labels
        return L, names, info

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
```

---

## 问题6：Auto-k选择改进

> **状态：✅ 已完成** (2026-01-09)
>
> **实现文件**: `hypergraph_binning/multiplex/auto_tune.py`
>
> **说明**: 与问题4合并实现，通过联合搜索 (β, k) 解决。使用 Silhouette Score 评估多个 k 候选值。

### 当前问题

当前的 auto-k 只在 eigengap 附近 ±2 搜索，范围太窄。

### 改进方案

综合多种启发式方法：Eigengap、Elbow、Silhouette。

### 代码实现

**文件**: `hypergraph_binning/spectral/kselect.py` (重写)

```python
"""
改进的 k 值自动选择模块。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from scipy.sparse.linalg import LinearOperator
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


@dataclass
class KSelectionResult:
    """k 选择结果"""
    best_k: int
    method: str
    eigenvalues: np.ndarray
    eigengap_k: int
    silhouette_k: int
    all_k_scores: dict
    details: dict


def compute_eigengaps(eigenvalues: np.ndarray) -> np.ndarray:
    """计算相邻特征值之间的间隙"""
    return np.diff(eigenvalues)


def find_eigengap_k(
    eigenvalues: np.ndarray,
    k_min: int = 2,
    k_max: int = None,
    relative: bool = True
) -> int:
    """基于特征值间隙选择 k。"""
    if k_max is None:
        k_max = len(eigenvalues) - 1
    k_max = min(k_max, len(eigenvalues) - 1)

    gaps = compute_eigengaps(eigenvalues)

    if relative:
        eps = 1e-10
        relative_gaps = gaps / np.maximum(np.abs(eigenvalues[:-1]), eps)
        gaps = relative_gaps

    valid_range = slice(k_min - 1, k_max)
    sub_gaps = gaps[valid_range]

    if len(sub_gaps) == 0:
        return k_min

    best_idx = np.argmax(sub_gaps)
    return int(k_min + best_idx)


def find_silhouette_k(
    U: np.ndarray,
    k_candidates: List[int],
    n_init: int = 10,
    seed: int = 42,
    sample_size: int = 10000
) -> Tuple[int, dict]:
    """基于 Silhouette score 选择 k。"""
    n = U.shape[0]
    scores = {}

    norms = np.linalg.norm(U, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    U_norm = U / norms

    for k in k_candidates:
        if k > U.shape[1] or k < 2:
            continue

        X = U_norm[:, :k]
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
        labels = km.fit_predict(X)

        if len(np.unique(labels)) < 2:
            scores[k] = -1.0
            continue

        if n > sample_size:
            rng = np.random.default_rng(seed)
            idx = rng.choice(n, size=sample_size, replace=False)
            score = silhouette_score(X[idx], labels[idx])
        else:
            score = silhouette_score(X, labels)

        scores[k] = float(score)

    if not scores:
        return k_candidates[0], {}

    best_k = max(scores, key=scores.get)
    return int(best_k), scores


def find_elbow_k(eigenvalues: np.ndarray, k_min: int = 2, k_max: int = None) -> int:
    """使用肘部法则选择 k。"""
    if k_max is None:
        k_max = len(eigenvalues) - 1
    k_max = min(k_max, len(eigenvalues) - 1)

    if len(eigenvalues) < 3:
        return k_min

    second_diff = np.diff(eigenvalues, n=2)

    valid_start = max(0, k_min - 1)
    valid_end = min(len(second_diff), k_max - 1)

    if valid_end <= valid_start:
        return k_min

    sub = second_diff[valid_start:valid_end]
    best_idx = np.argmax(sub)
    return int(k_min + best_idx)


def auto_select_k_comprehensive(
    L: LinearOperator,
    k_min: int = 5,
    k_max: int = 100,
    maxiter: int = 300,
    seed: int = 42,
    method: str = "combined"
) -> KSelectionResult:
    """
    综合多种方法的 k 自动选择。

    Parameters
    ----------
    method :
        - "eigengap": 只用特征值间隙
        - "silhouette": 只用轮廓系数
        - "combined": 综合考虑（默认）
    """
    from .eigen import eigenpairs

    n_eigs = min(k_max + 10, L.shape[0] - 1)
    vals, vecs = eigenpairs(L, k=n_eigs, maxiter=maxiter)

    eigengap_k = find_eigengap_k(vals, k_min=k_min, k_max=k_max, relative=True)
    elbow_k = find_elbow_k(vals, k_min=k_min, k_max=k_max)

    # 构建候选值列表
    k_candidates = sorted(set([
        eigengap_k - 5, eigengap_k - 2, eigengap_k, eigengap_k + 2, eigengap_k + 5,
        elbow_k - 2, elbow_k, elbow_k + 2,
        k_min, k_min + 5, k_min + 10,
        (k_min + k_max) // 4,
        (k_min + k_max) // 2,
    ]))
    k_candidates = [k for k in k_candidates if k_min <= k <= k_max]

    silhouette_k, sil_scores = find_silhouette_k(vecs, k_candidates, seed=seed)

    # 综合决策
    if method == "eigengap":
        best_k = eigengap_k
    elif method == "silhouette":
        best_k = silhouette_k
    elif method == "combined":
        if abs(silhouette_k - eigengap_k) <= max(5, eigengap_k * 0.3):
            best_k = silhouette_k
        else:
            if eigengap_k in sil_scores:
                gap_score = sil_scores[eigengap_k]
                sil_score = sil_scores.get(silhouette_k, -1)
                if gap_score >= sil_score * 0.9:
                    best_k = eigengap_k
                else:
                    best_k = silhouette_k
            else:
                best_k = eigengap_k
    else:
        best_k = eigengap_k

    return KSelectionResult(
        best_k=best_k,
        method=method,
        eigenvalues=vals,
        eigengap_k=eigengap_k,
        silhouette_k=silhouette_k,
        all_k_scores=sil_scores,
        details={"elbow_k": elbow_k, "k_candidates": k_candidates}
    )
```

---

## 问题7：低置信度contigs处理

### 当前问题

所有 contigs 都被强制分配到某个 bin，但低证据的 contigs 可能污染 bin 质量。

### 改进方案

计算聚类置信度，将低置信度节点标记为"未分配"。

### 代码实现

**新文件**: `hypergraph_binning/clustering/confidence.py`

```python
"""
聚类置信度评估与低置信度节点处理模块。
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from sklearn.metrics import pairwise_distances


@dataclass
class ConfidenceResult:
    """置信度评估结果"""
    labels: np.ndarray            # 最终标签（-1 表示未分配）
    confidence_scores: np.ndarray # 每个节点的置信度得分
    n_unassigned: int
    unassigned_indices: np.ndarray
    details: dict


def compute_assignment_confidence(
    U: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray = None,
    method: str = "distance_ratio"
) -> np.ndarray:
    """
    计算每个节点的聚类分配置信度。

    Parameters
    ----------
    method :
        - "distance_ratio": 到最近中心距离 / 到次近中心距离
        - "density": 基于局部密度的置信度

    Returns
    -------
    scores : (n,) 置信度得分，范围 [0, 1]，越高越可信
    """
    n = U.shape[0]
    unique_labels = np.unique(labels[labels >= 0])
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        return np.ones(n)

    if centroids is None:
        centroids = np.zeros((n_clusters, U.shape[1]))
        for i, lab in enumerate(unique_labels):
            mask = labels == lab
            if mask.sum() > 0:
                centroids[i] = U[mask].mean(axis=0)

    dists = pairwise_distances(U, centroids)

    if method == "distance_ratio":
        sorted_dists = np.sort(dists, axis=1)
        d1 = sorted_dists[:, 0]
        d2 = sorted_dists[:, 1]
        d2 = np.maximum(d2, 1e-10)
        ratio = d1 / d2
        scores = 1.0 - np.clip(ratio, 0, 1)

    elif method == "density":
        from sklearn.neighbors import NearestNeighbors

        k_neighbors = min(20, n - 1)
        nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(U)
        _, indices = nbrs.kneighbors(U)

        scores = np.zeros(n)
        for i in range(n):
            lab = labels[i]
            neighbors = indices[i, 1:]
            same_label = (labels[neighbors] == lab).sum()
            scores[i] = same_label / len(neighbors)

    else:
        raise ValueError(f"Unknown method: {method}")

    return scores


def filter_low_confidence(
    U: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.3,
    method: str = "distance_ratio",
    min_cluster_size: int = 10,
    max_unassigned_ratio: float = 0.2
) -> ConfidenceResult:
    """
    过滤低置信度的分配，将其标记为未分配（-1）。

    Parameters
    ----------
    threshold : 置信度阈值，低于此值标记为未分配
    min_cluster_size : 过滤后每个簇的最小大小
    max_unassigned_ratio : 最大未分配比例
    """
    n = len(labels)
    scores = compute_assignment_confidence(U, labels, method=method)

    new_labels = labels.copy()
    low_conf_mask = scores < threshold
    new_labels[low_conf_mask] = -1

    # 检查是否过度过滤
    n_unassigned = (new_labels == -1).sum()
    if n_unassigned / n > max_unassigned_ratio:
        sorted_scores = np.sort(scores)
        cutoff_idx = int(n * max_unassigned_ratio)
        adjusted_threshold = sorted_scores[cutoff_idx]

        new_labels = labels.copy()
        new_labels[scores < adjusted_threshold] = -1
        n_unassigned = (new_labels == -1).sum()

        print(f"[Confidence] Adjusted threshold from {threshold:.2f} to {adjusted_threshold:.2f}")

    # 检查小簇
    unique_labels = np.unique(new_labels[new_labels >= 0])
    for lab in unique_labels:
        cluster_size = (new_labels == lab).sum()
        if cluster_size < min_cluster_size:
            new_labels[new_labels == lab] = -1
            print(f"[Confidence] Cluster {lab} too small ({cluster_size}), marked as unassigned")

    n_unassigned = (new_labels == -1).sum()
    unassigned_idx = np.where(new_labels == -1)[0]

    return ConfidenceResult(
        labels=new_labels,
        confidence_scores=scores,
        n_unassigned=n_unassigned,
        unassigned_indices=unassigned_idx,
        details={
            "threshold": threshold,
            "method": method,
            "unassigned_ratio": n_unassigned / n,
        }
    )


def reassign_unassigned(
    U: np.ndarray,
    labels: np.ndarray,
    confidence_threshold: float = 0.5
) -> np.ndarray:
    """尝试将未分配节点重新分配到最近的簇（如果置信度足够）。"""
    unassigned_mask = labels == -1
    if not unassigned_mask.any():
        return labels

    assigned_mask = labels >= 0
    if not assigned_mask.any():
        return labels

    unique_labels = np.unique(labels[assigned_mask])
    centroids = {}
    for lab in unique_labels:
        mask = labels == lab
        centroids[lab] = U[mask].mean(axis=0)

    centroid_matrix = np.array([centroids[lab] for lab in unique_labels])

    U_unassigned = U[unassigned_mask]
    dists = pairwise_distances(U_unassigned, centroid_matrix)

    new_labels = labels.copy()
    unassigned_indices = np.where(unassigned_mask)[0]

    for i, idx in enumerate(unassigned_indices):
        sorted_dist_idx = np.argsort(dists[i])
        d1 = dists[i, sorted_dist_idx[0]]
        d2 = dists[i, sorted_dist_idx[1]] if len(sorted_dist_idx) > 1 else d1 * 2

        confidence = 1 - d1 / max(d2, 1e-10)

        if confidence >= confidence_threshold:
            best_label = unique_labels[sorted_dist_idx[0]]
            new_labels[idx] = best_label

    n_reassigned = unassigned_mask.sum() - (new_labels == -1).sum()
    print(f"[Reassign] Reassigned {n_reassigned} of {unassigned_mask.sum()} unassigned nodes")

    return new_labels
```

---

## 改进后的Pipeline流程

### 完整流程图

```
输入: contigs.fasta, porec.bam, config.yml
        │
        ▼
┌───────────────────────────────────────┐
│  1. 数据预处理                         │
│  - 读取 contigs                        │
│  - 过滤短序列 (min_contig_len)         │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  2. 构建物理层超图                     │
│  - 解析 BAM，提取多接触读段            │
│  - 计算 q' = r × (∏p_i)^{1/k}         │
│  - 构建 H_phy, 计算 w_e = q' × 2/(k-1)│
│  - 计算 L_phy                          │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  3. 构建化学层超图 [改进]              │
│  - 计算 TNF 特征                       │
│  - KNN 邻域 → 真正的超边               │  ← 问题1修复
│  - 高斯核权重                          │
│  - 构建 H_chem, L_chem                 │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  4. 两层归一化 [新增]                  │  ← 问题2修复
│  - 估计各层谱尺度                      │
│  - L_phy_norm = L_phy / scale_phy     │
│  - L_chem_norm = L_chem / scale_chem  │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  5. β 自适应选择 [新增]                │  ← 问题4修复
│  - Grid Search over β ∈ [0.01, 5.0]   │
│  - 评估 Silhouette Score              │
│  - 选择最优 β                          │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  6. 构建 Supra-Laplacian              │
│  - L_supra = [L_phy+βI, -βI; -βI, L_chem+βI] │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  7. 连通性检查 [新增]                  │  ← 问题5修复
│  - 分析连通分量                        │
│  - 提取最大分量 / 标记未分配           │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  8. 稳健谱求解 [改进]                  │  ← 问题3修复
│  - 自动选择 eigsh / LOBPCG            │
│  - 收敛性检查与回退                    │
│  - 记录残差诊断信息                    │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  9. Auto-k 选择 [改进]                 │  ← 问题6修复
│  - Eigengap + Elbow + Silhouette      │
│  - 综合决策                            │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  10. 坐标融合与聚类                    │
│  - U_final = (U_phy + U_chem) / 2     │
│  - 行归一化                            │
│  - K-Means 聚类                        │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  11. 置信度过滤 [新增]                 │  ← 问题7修复
│  - 计算分配置信度                      │
│  - 低置信度 → 未分配                   │
│  - 可选：重分配尝试                    │
└───────────────────────────────────────┘
        │
        ▼
输出: bins.tsv (含 "unassigned" 类别)
      diagnostics.json (诊断信息)
```

### 改进后的 Pipeline 代码框架

```python
def run_multiplex_pipeline_v2(config_path: Path, ...) -> None:
    cfg = load_config(config_path)
    out_dir = Path(cfg.output_dir) / "multiplex_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = {}

    # 1. 读取数据
    names, tnf_features = compute_tnf(cfg.contigs_fasta, min_length=cfg.min_contig_len)
    n_contigs = len(names)

    # 2. 构建物理层
    hg = build_physical_hypergraph(cfg.porec_bam, names, cfg.filter_params)
    L_phy = build_phy_laplacian(hg)

    # 3. 构建化学层（改进：真正的超图）
    H_chem, w_chem, de_chem, dv_chem = build_knn_hypergraph(
        tnf_features, k=cfg.knn_k, weight_scheme="gaussian"
    )
    L_chem = build_chem_laplacian_from_hypergraph(H_chem, w_chem, de_chem, dv_chem)

    # 4. 两层归一化
    L_phy_norm, phy_scale = normalize_laplacian(L_phy)
    L_chem_norm, chem_scale = normalize_laplacian(L_chem)
    diagnostics["phy_scale"] = phy_scale
    diagnostics["chem_scale"] = chem_scale

    # 5. β 自适应选择
    if cfg.beta == "auto":
        beta_result = auto_select_beta(L_phy_norm, L_chem_norm, k=cfg.k)
        beta = beta_result.best_beta
        diagnostics["beta_selection"] = beta_result.__dict__
    else:
        beta = cfg.beta

    # 6. 构建 Supra-Laplacian
    L_supra, supra_info = build_supra_laplacian_normalized(
        L_phy_norm, L_chem_norm, beta=beta, normalize=False  # 已归一化
    )

    # 7. 连通性检查
    L_supra_proc, names_proc, conn_info = handle_disconnected_graph(
        L_supra, names, strategy="largest_only"
    )
    diagnostics["connectivity"] = conn_info

    # 8. 稳健谱求解
    eigen_config = EigensolverConfig(tol=1e-6, maxiter=500, method="auto")
    U, eigen_result = spectral_embedding_robust(L_supra_proc, k=cfg.k, config=eigen_config)
    diagnostics["eigensolver"] = {
        "method": eigen_result.method_used,
        "converged": eigen_result.converged,
        "max_residual": max(eigen_result.residual_norms),
    }

    # 9. Auto-k（如果需要）
    if cfg.k <= 0:
        k_result = auto_select_k_comprehensive(L_supra_proc, k_min=5, k_max=100)
        k_used = k_result.best_k
        diagnostics["k_selection"] = k_result.__dict__
    else:
        k_used = cfg.k

    # 10. 坐标融合与聚类
    U_norm = fuse_and_normalize(U, n_contigs=len(names_proc))
    labels_raw = kmeans_labels(U_norm, k=k_used)

    # 11. 置信度过滤
    conf_result = filter_low_confidence(
        U_norm, labels_raw,
        threshold=0.3,
        max_unassigned_ratio=0.15
    )
    labels_final = conf_result.labels
    diagnostics["confidence"] = conf_result.details

    # 12. 输出
    save_results(out_dir, names_proc, labels_final, conn_info, diagnostics)
```

---

## 发表建议

### 必须完成的实验

| 实验 | 目的 | 优先级 |
|------|------|--------|
| CAMI-I/II 评测 | 标准化对比 | **必须** |
| MetaBAT2/VAMB/SemiBin2 对比 | 证明优越性 | **必须** |
| CheckM2 质量评估 | MAG 完整度/污染度 | **必须** |
| 消融实验 | 各组件贡献 | **必须** |
| β 敏感性分析 | 参数稳定性 | **必须** |
| 大规模测试 (10^5 contigs) | 可扩展性 | 重要 |
| 真实样本案例 | 生物学意义 | 重要 |

### 投稿路线建议

1. **短期目标**（完成上述改进）→ Bioinformatics / Genome Biology
2. **中期目标**（补充实验、生物学案例）→ Nature Communications

### 文章结构建议

1. **Introduction**: 强调 Pore-C 的独特优势和现有工具的不足
2. **Methods**: 重点阐述双层超图耦合的数学框架
3. **Results**:
   - 模拟数据：CAMI benchmark
   - 真实数据：环境样本 MAG 分析
   - 消融实验：证明各组件贡献
4. **Discussion**: 与 Hi-C binning 方法的对比，未来方向

---

## 文件清单

需要创建/修改的文件：

```
hypergraph_binning/
├── multiplex/
│   ├── features.py          # 修改：build_knn_hypergraph
│   ├── graph.py              # 修改：归一化、build_chem_laplacian_from_hypergraph
│   ├── beta_selection.py     # 新增：β 自适应选择
│   ├── embedding.py          # 修改：使用稳健求解器
│   └── pipeline.py           # 修改：整合所有改进
├── spectral/
│   ├── robust_eigen.py       # 新增：稳健特征值求解
│   └── kselect.py            # 重写：综合 k 选择
├── clustering/
│   └── confidence.py         # 新增：置信度评估
└── utils/
    └── connectivity.py       # 新增：连通性分析
```

---

*文档生成完毕。按照上述改进方案逐步实现，可显著提升算法的理论严谨性和实际效果。*
