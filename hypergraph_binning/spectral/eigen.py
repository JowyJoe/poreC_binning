from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator


def eigenpairs(L: LinearOperator, k: int, maxiter: int = 300, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Return (vals, vecs) for k smallest eigenpairs of symmetric operator L.
    vals shape (k,), vecs shape (n, k), both sorted ascending by eigenvalues.

    Args:
        L: Linear operator (Laplacian matrix)
        k: Number of eigenpairs to compute
        maxiter: Maximum iterations for eigsh
        seed: Random seed for reproducibility (controls v0 initialization)
    """
    if k < 1:
        raise ValueError("k must be >=1")
    # Use fixed v0 for reproducibility (eigsh uses random init if v0=None)
    rng = np.random.RandomState(seed)
    n = L.shape[0]
    v0 = rng.randn(n)
    vals, vecs = eigsh(L, k=k, which="SA", maxiter=maxiter, tol=1e-3, v0=v0)
    idx = np.argsort(vals)
    return vals[idx], vecs[:, idx]


def spectral_embedding(L: LinearOperator, k: int, maxiter: int = 300, seed: int = 42) -> np.ndarray:
    """Compute k smallest eigenvectors of L (symmetric), return U (n x k).
    Note: the first eigenvalue may be ~0; downstream k-means is applied on row-normalized U.

    Args:
        L: Linear operator (Laplacian matrix)
        k: Number of eigenvectors to compute
        maxiter: Maximum iterations for eigsh
        seed: Random seed for reproducibility
    """
    vals, vecs = eigenpairs(L, k=k, maxiter=maxiter, seed=seed)
    return vecs
