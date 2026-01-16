"""
Adaptive (β, k) auto-selection for multiplex spectral clustering.

This module implements a two-stage automatic parameter selection:

Stage 1 (Coarse): Spectral Gap Maximization for β selection
  - Theory: Cheeger inequality - larger gap = better cluster separability
  - Fast: only compute a few eigenvalues per β
  - Reference: Cheeger (1969)

Stage 2 (Fine): Grid Search with Silhouette Score for k selection
  - Theory: Silhouette measures cluster cohesion vs separation
  - Standard method in clustering literature
  - Reference: Rousseeuw (1987)

Design Philosophy:
- Similar to SemiBin2/MetaBAT2: search over parameter space + internal metric
- Two-stage approach: fast coarse filtering + accurate fine selection
- β range [0.01, 5.0]: covers weak coupling to strong coupling
- k range: adaptive based on data size

References:
- Cheeger (1969). "A lower bound for the smallest eigenvalue of the Laplacian."
- Rousseeuw (1987). "Silhouettes: a graphical aid to the interpretation and
  validation of cluster analysis." J Comput Appl Math, 20, 53-65.
- von Luxburg (2007). "A tutorial on spectral clustering."
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .graph import build_supra_laplacian


@dataclass
class AutoTuneResult:
    """Result of joint (β, k) auto-selection."""
    best_beta: float
    best_k: int
    best_score: float
    all_results: List[Tuple[float, int, float]]  # [(beta, k, score), ...]
    eigenvectors: np.ndarray  # Eigenvectors for best (β, k)
    eigenvalues: np.ndarray   # Eigenvalues for best β
    method_details: Dict = field(default_factory=dict)


# =============================================================================
# Core Functions
# =============================================================================

def _generate_beta_candidates(n_samples: int = 20) -> np.ndarray:
    """
    Generate β candidates in log-space.

    Range [0.01, 5.0] covers:
    - β → 0: two layers nearly independent
    - β → ∞: two layers strongly coupled

    Log-space ensures good coverage across different magnitudes.
    """
    return np.logspace(-2, 0.7, n_samples)  # [0.01, ~5.0]


def _generate_k_candidates(n: int) -> List[int]:
    """
    Generate k candidates adaptively based on data size.

    Heuristic (similar to other binning tools):
    - Small datasets: finer granularity
    - Large datasets: coarser granularity, cap at reasonable max
    """
    if n <= 50:
        return list(range(2, min(n - 1, 20)))
    elif n <= 200:
        return [2, 5, 10, 15, 20, 30, 40, 50]
    elif n <= 1000:
        return [5, 10, 20, 30, 50, 70, 100]
    elif n <= 5000:
        return [10, 20, 30, 50, 70, 100, 150, 200]
    else:
        return [20, 30, 50, 70, 100, 150, 200, 300]


def _compute_spectral_gap(
    L_phy: csr_matrix,
    L_chem: csr_matrix,
    beta: float,
    maxiter: int = 200
) -> float:
    """
    Compute spectral gap (λ₂ - λ₁) for a given β.

    Theoretical basis (Cheeger inequality):
        h²/2 ≤ λ₂ ≤ 2h
    where h is the Cheeger constant (isoperimetric ratio).

    Larger spectral gap indicates better cluster separability.
    This is a FAST operation - only needs first few eigenvalues.

    Parameters
    ----------
    L_phy, L_chem : Layer Laplacians
    beta : coupling strength
    maxiter : eigensolver iterations

    Returns
    -------
    spectral_gap : λ₂ - λ₁ (0 if computation fails)
    """
    L_supra = build_supra_laplacian(L_phy, L_chem, beta=beta)

    try:
        # Only compute first 3 eigenvalues (very fast)
        vals, _ = eigsh(L_supra, k=min(3, L_supra.shape[0] - 1),
                        which="SA", maxiter=maxiter, tol=1e-3)
        vals = np.sort(vals)
        return float(vals[1] - vals[0]) if len(vals) > 1 else 0.0
    except Exception:
        return 0.0


def _fuse_and_normalize(U: np.ndarray, n: int) -> np.ndarray:
    """
    Fuse physical and chemical layer embeddings, then row-normalize.

    For Supra-Laplacian eigenvectors U of shape (2N, k):
    - First N rows: physical layer embedding
    - Last N rows: chemical layer embedding

    Fusion: simple average (can be extended to weighted average)
    Normalization: project onto unit sphere for K-means
    """
    U_phy = U[:n, :]
    U_chem = U[n:, :]
    U_fused = (U_phy + U_chem) / 2.0

    # Row normalization (important for spectral clustering)
    norms = np.linalg.norm(U_fused, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return U_fused / norms


def _compute_silhouette(
    X: np.ndarray,
    k: int,
    seed: int = 42,
    max_samples: int = 5000
) -> Tuple[float, np.ndarray]:
    """
    Run K-means and compute Silhouette Score.

    Silhouette Score (Rousseeuw, 1987):
    - Measures how similar a point is to its own cluster vs other clusters
    - Range: [-1, 1], higher is better
    - Standard method for cluster quality evaluation

    Parameters
    ----------
    X : (N, d) embedding matrix
    k : number of clusters
    seed : random seed for K-means
    max_samples : subsample for large datasets (speed)

    Returns
    -------
    score : Silhouette Score
    labels : cluster assignments
    """
    n = X.shape[0]

    if k < 2 or k >= n:
        return -1.0, np.zeros(n, dtype=int)

    # Use first k dimensions (spectral clustering convention)
    X_k = X[:, :k] if X.shape[1] >= k else X

    # K-means clustering
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(X_k)

    # Check valid clustering
    if len(np.unique(labels)) < 2:
        return -1.0, labels

    # Compute Silhouette (subsample if large)
    if n > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_samples, replace=False)
        score = silhouette_score(X_k[idx], labels[idx])
    else:
        score = silhouette_score(X_k, labels)

    return float(score), labels


# =============================================================================
# Main Entry Point
# =============================================================================

def auto_select_beta_k(
    L_phy: csr_matrix,
    L_chem: csr_matrix,
    beta_candidates: Optional[List[float]] = None,
    k_candidates: Optional[List[int]] = None,
    k_max: int = 100,
    maxiter: int = 300,
    seed: int = 42,
    verbose: bool = True,
    n_top_betas: int = 5
) -> AutoTuneResult:
    """
    Two-stage automatic selection of optimal (β, k).

    Stage 1 (Coarse - Fast):
        Compute spectral gap for all β candidates.
        Select top N candidates with highest spectral gap.
        Theory: Cheeger inequality - larger gap = better separability.

    Stage 2 (Fine - Accurate):
        For each selected β, search over k candidates.
        Use Silhouette Score to evaluate clustering quality.
        Theory: Rousseeuw (1987) - standard cluster quality metric.

    This two-stage approach:
    - Is faster than full grid search (Stage 1 filters out poor β values)
    - Has dual theoretical support (Cheeger + Silhouette)
    - Similar to SemiBin2/MetaBAT2 design philosophy

    Parameters
    ----------
    L_phy : (N, N) physical layer normalized Laplacian
    L_chem : (N, N) chemical layer normalized Laplacian
    beta_candidates : β values to search (default: log-space [0.01, 5.0])
    k_candidates : k values to search (default: adaptive based on N)
    k_max : maximum eigenvectors to compute
    maxiter : eigensolver max iterations
    seed : random seed
    verbose : print progress
    n_top_betas : number of top β values to keep after Stage 1

    Returns
    -------
    AutoTuneResult with optimal (β, k) and diagnostic info

    References
    ----------
    - Cheeger (1969). A lower bound for the smallest eigenvalue.
    - Rousseeuw (1987). Silhouettes. J Comput Appl Math.
    - von Luxburg (2007). A tutorial on spectral clustering.
    """
    n = L_phy.shape[0]

    # =========================================
    # Generate search space
    # =========================================
    if beta_candidates is None:
        beta_candidates = _generate_beta_candidates(n_samples=20)
    beta_candidates = np.asarray(beta_candidates)

    if k_candidates is None:
        k_candidates = _generate_k_candidates(n)

    # Filter valid k values
    k_candidates = [k for k in k_candidates if 2 <= k < n]
    if not k_candidates:
        k_candidates = [max(2, n // 2)]

    k_max = min(k_max, max(k_candidates) + 1, n - 1, 2 * n - 2)

    if verbose:
        print("=" * 60)
        print("[Auto-tune] Two-stage adaptive parameter selection")
        print("=" * 60)
        print(f"  Data size: N = {n}")
        print(f"  β search space: [{beta_candidates.min():.3f}, {beta_candidates.max():.3f}] "
              f"({len(beta_candidates)} values)")
        print(f"  k search space: {k_candidates}")
        print()

    # =========================================
    # Stage 1: Coarse β selection via Spectral Gap
    # =========================================
    if verbose:
        print("[Stage 1] Coarse β selection (Spectral Gap - Cheeger)")
        print("-" * 60)

    beta_gaps = []
    for beta in beta_candidates:
        gap = _compute_spectral_gap(L_phy, L_chem, float(beta), maxiter=200)
        beta_gaps.append((float(beta), gap))
        if verbose:
            print(f"  β={beta:.4f}: spectral_gap={gap:.6f}")

    # Sort by gap and select top N
    beta_gaps.sort(key=lambda x: x[1], reverse=True)
    top_betas = [bg[0] for bg in beta_gaps[:n_top_betas]]

    if verbose:
        print()
        print(f"  Top {n_top_betas} β values: {[f'{b:.4f}' for b in top_betas]}")
        print()

    # =========================================
    # Stage 2: Fine (β, k) selection via Silhouette
    # =========================================
    if verbose:
        print("[Stage 2] Fine selection (Silhouette - Rousseeuw)")
        print("-" * 60)

    all_results = []
    best_beta = top_betas[0]
    best_k = k_candidates[0]
    best_score = -1.0
    best_eigenvectors = None
    best_eigenvalues = None

    for beta in top_betas:
        # Build Supra-Laplacian and compute eigenvectors
        L_supra = build_supra_laplacian(L_phy, L_chem, beta=beta)

        try:
            vals, vecs = eigsh(L_supra, k=k_max, which="SA",
                              maxiter=maxiter, tol=1e-3)
            idx = np.argsort(vals)
            vals = vals[idx]
            vecs = vecs[:, idx]
        except Exception as e:
            if verbose:
                print(f"  β={beta:.4f}: eigsh failed ({e}), skipping")
            continue

        # Fuse and normalize embedding
        U_norm = _fuse_and_normalize(vecs, n)

        # Evaluate each k
        scores_this_beta = []
        for k in k_candidates:
            if k > U_norm.shape[1]:
                continue

            score, _ = _compute_silhouette(U_norm, k, seed=seed)
            all_results.append((beta, k, score))
            scores_this_beta.append((k, score))

            # Update best
            if score > best_score:
                best_score = score
                best_beta = beta
                best_k = k
                best_eigenvectors = vecs.copy()
                best_eigenvalues = vals.copy()

        if verbose and scores_this_beta:
            best_k_this = max(scores_this_beta, key=lambda x: x[1])
            print(f"  β={beta:.4f}: best k={best_k_this[0]} (silhouette={best_k_this[1]:.3f})")

    # =========================================
    # Return result
    # =========================================
    if best_eigenvectors is None:
        raise RuntimeError("Auto-tuning failed: no valid (β, k) found")

    if verbose:
        print()
        print("=" * 60)
        print("[Result] Optimal parameters:")
        print(f"  β = {best_beta:.4f}")
        print(f"  k = {best_k}")
        print(f"  Silhouette Score = {best_score:.4f}")
        print("=" * 60)

    return AutoTuneResult(
        best_beta=best_beta,
        best_k=best_k,
        best_score=best_score,
        all_results=all_results,
        eigenvectors=best_eigenvectors,
        eigenvalues=best_eigenvalues,
        method_details={
            "method": "two_stage",
            "stage1": {
                "name": "Spectral Gap (Cheeger)",
                "all_betas": [bg[0] for bg in beta_gaps],
                "all_gaps": [bg[1] for bg in beta_gaps],
                "top_betas": top_betas,
            },
            "stage2": {
                "name": "Silhouette (Rousseeuw)",
                "k_candidates": k_candidates,
                "n_evaluated": len(all_results),
            },
            "references": [
                "Cheeger (1969). A lower bound for the smallest eigenvalue.",
                "Rousseeuw (1987). Silhouettes. J Comput Appl Math.",
            ]
        }
    )
