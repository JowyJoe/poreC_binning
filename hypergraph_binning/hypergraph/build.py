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
        # per-edge weight with pairwise equal-share principle
        # TODO: Review normalization factor. Future work may require adjusting this formula.
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
