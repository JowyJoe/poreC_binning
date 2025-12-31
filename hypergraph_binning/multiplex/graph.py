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

def build_chem_laplacian(H: csr_matrix, w: np.ndarray, de: np.ndarray, dv: np.ndarray) -> csr_matrix:
    """
    Build chemical layer Laplacian using the same formula as physical layer.
    L = I - D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}

    Parameters
    ----------
    H : (N, N) hypergraph incidence matrix
    w : (N,) hyperedge weights
    de : (N,) hyperedge degrees
    dv : (N,) vertex degrees

    Returns
    -------
    L_chem : (N, N) normalized Laplacian matrix
    """
    return _compute_laplacian_matrix(H, w, de, dv)

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
