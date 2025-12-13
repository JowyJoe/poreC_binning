from __future__ import annotations
from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from ..spectral.kselect import largest_eigengap


def run_multiplex_embedding(L_supra: csr_matrix, k: int, maxiter: int = 300, seed: int = 42) -> np.ndarray:
    """
    Compute k smallest eigenvectors of Supra-Laplacian.
    Returns U_final (N, k) after fusion and normalization.
    If k <= 0, auto-select k using eigengap (searching up to k=60).
    """
    n2 = L_supra.shape[0]
    n = n2 // 2
    
    # Determine how many eigenvectors to compute
    if k <= 0:
        auto_cap = min(200, n - 1)
        k_search = max(2, min(auto_cap, n2 - 2))
        if k_search < 2:
            raise ValueError("Cannot auto-select k with fewer than 2 contigs.")
        print(f"Auto-selecting k (computing top {k_search} eigenvectors)...")
    else:
        k_search = min(k, n2 - 1)
        if k_search <= 0:
            raise ValueError("k must be greater than 0.")

    # Solve eigenproblem
    # We want smallest algebraic connectivity.
    vals, vecs = eigsh(L_supra, k=k_search, which="SA", maxiter=maxiter, tol=1e-3, v0=None)
    
    # Sort by eigenvalues
    idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:, idx]
    
    # Auto-select k if needed
    if k <= 0:
        # Use heuristic: largest gap in [k_min, k_max]
        # k_min=2 because k=1 is trivial (connected component)
        best_k = largest_eigengap(vals, k_min=2, k_max=k_search - 1)
        best_k = max(2, min(best_k, k_search))
        print(f"Auto-selected k={best_k} based on eigengap.")
        vecs = vecs[:, :best_k]
        # Update k for subsequent logic if needed (though we just use vecs shape)
        k = best_k

    # U is (2N, k)
    U = vecs
    
    # Coordinate Fusion
    U_phy = U[:n, :]
    U_chem = U[n:, :]
    
    # Late Fusion: Average the coordinates
    U_final = (U_phy + U_chem) / 2.0
    
    # Row Normalization (Project to unit sphere)
    norms = np.linalg.norm(U_final, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    U_norm = U_final / norms
    
    return U_norm
