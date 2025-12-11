from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import json
import numpy as np


@dataclass
class EdgeChunks:
    n_vertices: int
    chunk_paths: List[Path]
    dv: np.ndarray  # (n,) vertex degrees


def write_edge_chunks(
    n_vertices: int,
    edge_iter: Iterable[Tuple[List[int], float]],
    out_dir: Path,
    q_cap: float = 0.95,
    chunk_size: int = 200_000,
    dtype_float: str = "float32",
) -> EdgeChunks:
    """
    Stream edges to chunked npz files to avoid keeping full incidence in memory.
    Each edge: (members_idx, q_r). We store per-chunk arrays:
      - idx_flat: np.int32 flattened member indices concatenated
      - ptr: np.int64 array of length (m_chunk+1) pointing into idx_flat (CSR-like)
      - w: np.float32 per-edge weights
      - de: np.int32 per-edge degrees (|e|)
    Also accumulate dv (vertex degrees) over all chunks as sum_e w_e for incident edges.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    dv = np.zeros((n_vertices,), dtype=np.float64)
    chunk_paths: List[Path] = []

    cur_ptr: List[int] = [0]
    cur_idx: List[int] = []
    cur_w: List[float] = []
    cur_de: List[int] = []
    m_in_chunk = 0
    chunk_id = 0

    def flush_chunk():
        nonlocal cur_ptr, cur_idx, cur_w, cur_de, m_in_chunk, chunk_id
        if m_in_chunk == 0:
            return
        idx_flat = np.asarray(cur_idx, dtype=np.int32)
        ptr = np.asarray(cur_ptr, dtype=np.int64)
        w = np.asarray(cur_w, dtype=np.float32 if dtype_float == "float32" else np.float64)
        de = np.asarray(cur_de, dtype=np.int32)
        path = out_dir / f"edges_chunk_{chunk_id:05d}.npz"
        np.savez_compressed(path, idx_flat=idx_flat, ptr=ptr, w=w, de=de)
        chunk_paths.append(path)
        # reset
        cur_ptr = [0]
        cur_idx = []
        cur_w = []
        cur_de = []
        m_in_chunk = 0
        chunk_id += 1

    for members, q_prime in edge_iter:
        k = len(members)
        if k < 2:
            continue
        # per-edge weight with pairwise equal-share principle
        # TODO: Review normalization factor. Future work may require adjusting this formula.
        w_e = float(q_prime) * (2.0 / float(k - 1))
        # update dv
        for v in members:
            dv[v] += w_e
        # append edge to chunk
        cur_idx.extend(members)
        cur_ptr.append(cur_ptr[-1] + k)
        cur_w.append(w_e)
        cur_de.append(k)
        m_in_chunk += 1
        if m_in_chunk >= chunk_size:
            flush_chunk()

    flush_chunk()

    return EdgeChunks(n_vertices=n_vertices, chunk_paths=chunk_paths, dv=dv.astype(np.float64))
