from __future__ import annotations
from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def run_multiplex_embedding(L_supra: csr_matrix, k: int, maxiter: int = 300, seed: int = 42) -> np.ndarray:
    """
    Compute k smallest eigenvectors of Supra-Laplacian.
    Returns U_final (N, k) after fusion and normalization.
    """
    n2 = L_supra.shape[0]
    n = n2 // 2
    
    # Solve eigenproblem
    # We want smallest algebraic connectivity.
    # Note: L_supra is symmetric.
    # k+1 because the first one is usually 0 (connected component).
    # But for clustering k classes, we usually need k eigenvectors.
    # If the graph is connected, lambda_0 = 0, v_0 = const.
    # We usually use v_1 ... v_k for clustering.
    # Let's compute k eigenvectors.
    # If k is small, we might get the null space.
    # Standard spectral clustering uses bottom k eigenvectors.
    
    vals, vecs = eigsh(L_supra, k=k, which="SA", maxiter=maxiter, tol=1e-3, v0=None)
    
    # Sort by eigenvalues
    idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:, idx]
    
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
