from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse.linalg import LinearOperator
from sklearn.metrics import silhouette_score

from .eigen import eigenpairs
from ..clustering.kmeans_cluster import kmeans_labels


@dataclass
class KSelectParams:
    k_min: int = 10
    k_max: int = 200
    eig_extra: int = 5  # compute a bit more to evaluate gaps around candidate k
    maxiter: int = 300
    seeds: int = 5      # stability: number of k-means restarts considered in silhouette


def largest_eigengap(vals: np.ndarray, k_min: int, k_max: int) -> int:
    """Pick k by the largest eigengap between consecutive eigenvalues (λ_k and λ_{k+1}).
    vals must be ascending; length must be >= k_max+1.
    """
    lo = max(1, k_min)
    hi = min(k_max, len(vals) - 1)
    gaps = vals[1:hi+1] - vals[:hi]
    # We consider gaps from indices [lo-1 .. hi-1] corresponding to k in [lo..hi]
    idx_range = np.arange(lo-1, hi)
    sub = gaps[idx_range]
    k_candidates = np.arange(lo, hi+1)
    best_idx = int(np.argmax(sub))
    return int(k_candidates[best_idx])


def auto_select_k(
    L: LinearOperator,
    params: KSelectParams,
) -> Tuple[int, np.ndarray]:
    """
    Return (k, U) where k is auto-selected via eigengap, and U is embedding with that k.
    Then refine k within ±2 by checking silhouette in embedding space.
    """
    want = params.k_max + 1  # need one more for gap at k_max
    vals, vecs = eigenpairs(L, k=want, maxiter=params.maxiter)
    k_gap = largest_eigengap(vals, params.k_min, params.k_max)

    # candidate ks around k_gap (±2 within bounds)
    ks = sorted(set([k_gap + d for d in [-2, -1, 0, 1, 2] if params.k_min <= k_gap + d <= params.k_max]))

    def row_norm(X: np.ndarray) -> np.ndarray:
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        nrm[nrm == 0] = 1.0
        return X / nrm

    best_k = k_gap
    best_score = -1.0
    best_U = vecs[:, :best_k]

    for k in ks:
        U = vecs[:, :k]
        X = row_norm(U)
        # try multiple seeds but reuse our kmeans wrapper which itself does multi init
        labels = kmeans_labels(U, k=k, n_init=20, seed=42)
        # silhouette can be expensive on very large n; we can sample
        n = X.shape[0]
        if n > 20000:
            idx = np.random.default_rng(42).choice(n, size=20000, replace=False)
            score = silhouette_score(X[idx], labels[idx])
        else:
            score = silhouette_score(X, labels)
        if score > best_score:
            best_score = float(score)
            best_k = k
            best_U = U

    return best_k, best_U
