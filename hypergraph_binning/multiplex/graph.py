from __future__ import annotations
from typing import Tuple, Optional

import numpy as np
from scipy.sparse import csr_matrix, diags, bmat, identity
from scipy.sparse.linalg import LinearOperator

from ..hypergraph.build import Hypergraph

def _compute_laplacian_matrix(H: csr_matrix, w: np.ndarray, de: np.ndarray, dv: np.ndarray) -> csr_matrix:
    """
    Compute the normalized Laplacian matrix L = I - Dv^-1/2 H W De^-1 H^T Dv^-1/2
    Returns a sparse matrix (CSR).
    """
    n, m = H.shape
    
    # Avoid division by zero
    dv_safe = np.where(dv > 0, dv, 1.0)
    inv_sqrt_dv = 1.0 / np.sqrt(dv_safe)
    # If degree is 0, the entry in inv_sqrt_dv is 1.0, but we want the final operator to be 0 for that node?
    # Usually for isolated nodes, L[i,i] should be 0 or 1 depending on convention.
    # If we stick to I - A_norm, and A_norm is 0 for isolated, then L=I.
    # Let's keep it simple.
    
    de_safe = np.where(de > 0, de, 1.0)
    inv_de = 1.0 / de_safe
    
    # Diagonal matrices
    # D_v^{-1/2}
    Dv_inv_sqrt = diags(inv_sqrt_dv, format="csr")
    
    # W * D_e^{-1}
    # w is (m,), inv_de is (m,)
    # We can combine them into one diagonal matrix for the middle
    mid_vals = w * inv_de
    Mid = diags(mid_vals, format="csr")
    
    # Compute A_norm = Dv^-1/2 H W De^-1 H^T Dv^-1/2
    # Order: (Dv^-1/2 H) * (W De^-1) * (H^T Dv^-1/2)
    
    # 1. Left part: L_part = Dv^-1/2 @ H
    L_part = Dv_inv_sqrt @ H
    
    # 2. Middle: L_part @ Mid
    # This scales columns of L_part
    L_part = L_part @ Mid
    
    # 3. Full: L_part @ H.T @ Dv^-1/2
    # Note: H.T @ Dv^-1/2 is (Dv^-1/2 @ H).T
    # So we are doing A @ A.T essentially (if weights were symmetric/identity)
    
    # A_norm = L_part @ H.T @ Dv_inv_sqrt
    # To keep sparsity, we should do: (L_part @ H.T) @ Dv_inv_sqrt
    A_norm = L_part @ H.T @ Dv_inv_sqrt
    
    # L = I - A_norm
    I = identity(n, format="csr", dtype=np.float64)
    L = I - A_norm
    
    return L

def build_phy_laplacian(hg: Hypergraph) -> csr_matrix:
    """Build physical layer Laplacian from Hypergraph object."""
    return _compute_laplacian_matrix(hg.H, hg.w, hg.de, hg.dv)

def build_chem_laplacian(H_knn: csr_matrix) -> csr_matrix:
    """
    Build chemical layer Laplacian from KNN incidence matrix.
    H_knn: (N, N) where col j is hyperedge centered at j.
    Weights are already in H_knn values.
    """
    n = H_knn.shape[0]
    m = H_knn.shape[1] # should be n
    
    # Compute degrees
    # Edge degree de[j] = sum_i H[i, j]
    # Since H contains weights, this is weighted degree.
    # Formula: delta(e) = sum_v h(v,e)
    de = np.array(H_knn.sum(axis=0)).ravel()
    
    # Vertex degree dv[i] = sum_e w(e) h(v,e)
    # Here w(e) is assumed 1.0 because weights are in h(v,e).
    # So dv[i] = sum_j H[i, j]
    dv = np.array(H_knn.sum(axis=1)).ravel()
    
    # Hyperedge weights w: all 1.0
    w = np.ones(m, dtype=np.float64)
    
    return _compute_laplacian_matrix(H_knn, w, de, dv)

def build_supra_laplacian(L_phy: csr_matrix, L_chem: csr_matrix, beta: float = 0.5) -> csr_matrix:
    """
    Construct the 2N x 2N Supra-Laplacian matrix.
    L_supra = [ L_phy + beta*I   -beta*I ]
              [ -beta*I          L_chem + beta*I ]
    """
    n = L_phy.shape[0]
    assert L_chem.shape == (n, n)
    
    I = identity(n, format="csr", dtype=np.float64)
    beta_I = beta * I
    
    # Blocks
    TL = L_phy + beta_I
    TR = -beta_I
    BL = -beta_I
    BR = L_chem + beta_I
    
    L_supra = bmat([
        [TL, TR],
        [BL, BR]
    ], format="csr")
    
    return L_supra
