from __future__ import annotations
from typing import Iterable, List

import numpy as np
from scipy.sparse.linalg import LinearOperator


class StreamingHypergraphOperator(LinearOperator):
    """
    Matrix-free operator for L = I - Dv^{-1/2} H W De^{-1} H^T Dv^{-1/2}
    using chunked hyperedge storage on disk.

    Each chunk file (.npz) must contain arrays:
      - idx_flat: int32, concatenated vertex indices for all edges in chunk
      - ptr: int64, length (m_chunk+1), CSR-style pointers into idx_flat
      - w: float32/float64, length (m_chunk,), per-edge weights
      - de: int32, length (m_chunk,), per-edge degrees
    """

    def __init__(self, n: int, dv: np.ndarray, chunk_paths: List[str]):
        self._n = int(n)
        dv_safe = np.where(dv > 0, dv, 1.0).astype(np.float64)
        self._inv_sqrt_dv = 1.0 / np.sqrt(dv_safe)
        self._chunk_paths = list(map(str, chunk_paths))
        super().__init__(dtype=np.float64, shape=(self._n, self._n))

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float64, copy=False)
        # y = Dv^{-1/2} x
        y = self._inv_sqrt_dv * x
        # accum = H W De^{-1} H^T y  (computed in chunks)
        accum = np.zeros((self._n,), dtype=np.float64)
        for p in self._chunk_paths:
            with np.load(p) as npz:
                idx_flat = npz["idx_flat"]  # (nnz_chunk,)
                ptr = npz["ptr"]            # (m_chunk+1,)
                w = npz["w"].astype(np.float64, copy=False)    # (m_chunk,)
                de = npz["de"].astype(np.float64, copy=False)  # (m_chunk,)
            # s_e = sum_{v in e} y[v]
            y_sel = y[idx_flat]
            # reduceat sums y_sel at edges boundaries
            s_e = np.add.reduceat(y_sel, ptr[:-1])
            gain = (w / np.where(de > 0, de, 1.0)) * s_e  # per-edge scalar
            # scatter-add: for each edge, add gain[e] to all its members
            rep = np.repeat(gain, np.diff(ptr))
            # accum[idx_flat] += rep (vectorized)
            np.add.at(accum, idx_flat, rep)
        # z = Dv^{-1/2} accum
        z = self._inv_sqrt_dv * accum
        return x - z
