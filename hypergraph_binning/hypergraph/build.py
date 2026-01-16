from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class Hypergraph:
    H: csr_matrix  # n x m incidence (0/1)
    w: np.ndarray  # (m,) hyperedge weights (>0)
    de: np.ndarray  # (m,) hyperedge degrees (|e|)
    dv: np.ndarray  # (n,) vertex degrees (sum_e w_e * H_{ve})
    contig_names: List[str]

    def to_adjacency(self) -> csr_matrix:
        """
        Convert hypergraph to weighted adjacency matrix (for contact ratio calculation).

        A[i,j] = sum_e (w_e / d_e) * H[i,e] * H[j,e]  for i != j

        This represents the total contact strength between node i and j
        across all shared hyperedges, normalized by hyperedge size.

        Returns:
            csr_matrix: n x n sparse adjacency matrix (self-loops removed)
        """
        n = self.H.shape[0]
        m = self.H.shape[1]

        if m == 0:
            return csr_matrix((n, n), dtype=np.float64)

        # W = diag(w / de) to normalize by hyperedge size
        # This gives each pair in a hyperedge equal share of the weight
        w_normalized = self.w / np.maximum(self.de, 1)

        # A = H @ W @ H^T
        # Efficient: (H @ diag(w_norm)) @ H^T = H @ (w_norm * H^T)
        HW = self.H.multiply(w_normalized)  # broadcast: each column scaled by w_norm
        A = HW @ self.H.T

        # Remove self-loops (diagonal)
        A = A.tocsr()
        A.setdiag(0)
        A.eliminate_zeros()

        return A


def build_from_porec(
    contig_names: List[str],
    hyperedges: Iterable[Tuple[List[int], float]],
) -> Hypergraph:
    """
    Build hypergraph from iterable of (member_indices, q_prime) per read.
    We set w_e = q' * 2/(|e|-1) with |e|>=2, where q' is the per-read reliability
    computed upstream from MAPQ (see io.bam.iterate_porec_hyperedges).
    """
    n = len(contig_names)
    data: List[int] = []
    rows: List[int] = []
    cols: List[int] = []
    w_list: List[float] = []
    de_list: List[int] = []

    m = 0
    for members, q_prime in hyperedges:
        k = len(members)
        if k < 2:
            continue
        # Hyperedge weight: w_e = q' * 2/(k-1)
        # Rationale: A hyperedge of size k contains C(k,2) = k(k-1)/2 implicit pairs.
        # We assign equal weight 1/(k-1) to each pair (so total = k/2).
        # The factor 2/(k-1) normalizes so that each contig pair within the hyperedge
        # receives contribution q'/C(k,2) = 2q'/(k(k-1)) per pair, totaling q' per edge.
        # This avoids the "dilution effect" where large hyperedges dominate small ones.
        w_e = float(q_prime) * (2.0 / float(k - 1))
        for v in members:
            rows.append(v)
            cols.append(m)
            data.append(1)
        w_list.append(w_e)
        de_list.append(k)
        m += 1

    if m == 0:
        # empty hypergraph
        H = csr_matrix((n, 0), dtype=np.float64)
        w = np.zeros((0,), dtype=np.float64)
        de = np.zeros((0,), dtype=np.float64)
        dv = np.zeros((n,), dtype=np.float64)
        return Hypergraph(H=H, w=w, de=de, dv=dv, contig_names=contig_names)

    H = csr_matrix((np.array(data, dtype=np.float64), (np.array(rows), np.array(cols))), shape=(n, m))
    w = np.array(w_list, dtype=np.float64)
    de = np.array(de_list, dtype=np.float64)
    # dv = H @ w (each vertex degree is sum over incident edges of w_e)
    dv = (H @ w.reshape(-1, 1)).ravel()
    return Hypergraph(H=H, w=w, de=de, dv=dv, contig_names=contig_names)
