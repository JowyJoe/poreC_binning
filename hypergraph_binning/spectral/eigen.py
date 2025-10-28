from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator


def eigenpairs(L: LinearOperator, k: int, maxiter: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    """Return (vals, vecs) for k smallest eigenpairs of symmetric operator L.
    vals shape (k,), vecs shape (n, k), both sorted ascending by eigenvalues.
    """
    if k < 1:
        raise ValueError("k must be >=1")
    vals, vecs = eigsh(L, k=k, which="SA", maxiter=maxiter, tol=1e-3, v0=None)
    idx = np.argsort(vals)
    return vals[idx], vecs[:, idx]


def spectral_embedding(L: LinearOperator, k: int, maxiter: int = 300, seed: int = 42) -> np.ndarray:
    """Compute k smallest eigenvectors of L (symmetric), return U (n x k).
    Note: the first eigenvalue may be ~0; downstream k-means is applied on row-normalized U.
    """
    vals, vecs = eigenpairs(L, k=k, maxiter=maxiter)
    return vecs
