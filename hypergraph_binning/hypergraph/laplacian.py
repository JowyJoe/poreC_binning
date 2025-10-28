from __future__ import annotations
from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import LinearOperator


def make_L_linear_operator(H: csr_matrix, w: np.ndarray, de: np.ndarray, dv: np.ndarray) -> LinearOperator:
    """
    Return a LinearOperator representing L = I - Dv^{-1/2} H W De^{-1} H^T Dv^{-1/2}
    without forming dense matrices.
    Shapes: H (n x m), w (m,), de (m,), dv (n,)
    """
    n, m = H.shape
    assert w.shape == (m,)
    assert de.shape == (m,)
    assert dv.shape == (n,)

    # Precompute scaling vectors
    dv_safe = np.where(dv > 0, dv, 1.0)
    inv_sqrt_dv = 1.0 / np.sqrt(dv_safe)
    inv_de = np.where(de > 0, 1.0 / de, 0.0)

    def matvec(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float64, copy=False)
        y = inv_sqrt_dv * x  # Dv^{-1/2} x
        y = H.T.dot(y)       # m
        y = y * inv_de       # De^{-1}
        y = y * w            # W
        y = H.dot(y)         # n
        y = inv_sqrt_dv * y  # Dv^{-1/2}
        return x - y         # I - M

    return LinearOperator(shape=(n, n), matvec=matvec, dtype=np.float64)
