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

    de_safe = np.where(de > 0, de, 1.0)
    inv_de = 1.0 / de_safe

    # Diagonal matrices
    Dv_inv_sqrt = diags(inv_sqrt_dv, format="csr")

    # W * D_e^{-1}
    mid_vals = w * inv_de
    Mid = diags(mid_vals, format="csr")

    # Compute A_norm = Dv^-1/2 H W De^-1 H^T Dv^-1/2
    L_part = Dv_inv_sqrt @ H
    L_part = L_part @ Mid
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

def build_supra_laplacian(
    L_phy: csr_matrix,
    L_chem: csr_matrix,
    beta: float = 0.5
) -> csr_matrix:
    """
    Construct the 2N x 2N Supra-Laplacian matrix.

    The Supra-Laplacian couples two network layers (physical and chemical) into
    a unified matrix for multiplex spectral clustering.

    Parameters
    ----------
    L_phy : csr_matrix
        Physical layer Laplacian (N x N), from Pore-C contacts
    L_chem : csr_matrix
        Chemical layer Laplacian (N x N), from TNF KNN hypergraph
    beta : float
        Inter-layer coupling strength. Controls how strongly the two layers
        are coupled. Default 0.5.

    Returns
    -------
    L_supra : csr_matrix
        Supra-Laplacian matrix (2N x 2N)

    Mathematical Formulation
    ------------------------
    L_supra = [ L_phy + βI    -βI        ]
              [ -βI           L_chem + βI ]

    Physical Interpretation
    -----------------------
    - Diagonal blocks (L + βI): Intra-layer diffusion + coupling potential
    - Off-diagonal blocks (-βI): Inter-layer coupling ("vertical springs")
    - β controls coupling strength:
        - β → 0: Layers are independent
        - β → ∞: Forces identical embeddings in both layers

    Note
    ----
    Both L_phy and L_chem use the normalized Laplacian formula
    L = I - D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}, so they are already
    on the same scale (trace/N ≈ 1). No additional normalization needed.

    References
    ----------
    - Mucha et al. (2010), Science: "Community structure in multiplex networks"
    - Gomez et al. (2013), Phys Rev Lett: "Diffusion dynamics on multiplex networks"
    """
    n = L_phy.shape[0]
    assert L_chem.shape == (n, n), f"Shape mismatch: L_phy {L_phy.shape} vs L_chem {L_chem.shape}"

    # Construct Supra-Laplacian
    I = identity(n, format="csr", dtype=np.float64)
    beta_I = beta * I

    # Block matrix construction
    TL = L_phy + beta_I      # Top-left: Physical + coupling
    TR = -beta_I             # Top-right: Inter-layer coupling
    BL = -beta_I             # Bottom-left: Inter-layer coupling
    BR = L_chem + beta_I     # Bottom-right: Chemical + coupling

    L_supra = bmat([
        [TL, TR],
        [BL, BR]
    ], format="csr")

    print(f"[Supra-Laplacian] Constructed {L_supra.shape[0]}x{L_supra.shape[1]} matrix, β={beta}")

    return L_supra
