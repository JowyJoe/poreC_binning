from __future__ import annotations

import csv
import gzip
import json
import logging
from pathlib import Path
from typing import Optional


class GraphClusterError(RuntimeError):
    pass


def cluster_leiden(
    *,
    graph_dir: Path,
    out_bins_tsv: Path,
    resolution: float = 1.0,
    seed: int = 0,
    logger: Optional[logging.Logger] = None,
) -> None:
    logger = logger or logging.getLogger("porebin")
    graph_dir = graph_dir.resolve()
    out_bins_tsv = out_bins_tsv.resolve()

    contig_names = _read_contig_index(graph_dir / "contig_index.tsv")
    if not contig_names:
        raise GraphClusterError(f"No contigs found in {graph_dir / 'contig_index.tsv'}")

    meta = _read_graph_meta(graph_dir / "graph_meta.json")
    num_contigs = len(contig_names)
    num_contacts = int(meta.get("num_contacts", 0))
    if num_contacts <= 0:
        raise GraphClusterError(f"Invalid num_contacts in {graph_dir / 'graph_meta.json'}: {num_contacts}")

    edges, weights = _read_edges(graph_dir / "edges.tsv", contig_offset=num_contigs)
    if not edges:
        raise GraphClusterError(f"No edges found in {graph_dir / 'edges.tsv'}")

    logger.info(
        f"Clustering bipartite graph: contigs={num_contigs}, contacts={num_contacts}, edges={len(edges)}"
    )

    try:
        import igraph as ig
        import leidenalg
    except Exception as exc:  # pragma: no cover
        raise GraphClusterError(
            "cluster requires 'python-igraph' and 'leidenalg'. Install them, e.g. pip install python-igraph leidenalg."
        ) from exc

    g = ig.Graph(n=num_contigs + num_contacts, edges=edges, directed=False)
    g.es["weight"] = weights
    g.vs["type"] = [False] * num_contigs + [True] * num_contacts

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
    )

    membership = partition.membership[:num_contigs]
    out_bins_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_bins_tsv.open("w", encoding="utf-8", newline="") as fh:
        fh.write("contig_name\tbin_id\n")
        for name, bin_id in zip(contig_names, membership, strict=True):
            fh.write(f"{name}\t{bin_id}\n")

    logger.info(f"Wrote bins: {out_bins_tsv}")


def cluster_spectral_hypergraph(
    *,
    graph_dir: Path,
    out_bins_tsv: Path,
    seed: int = 0,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """
    Hypergraph spectral clustering using a Zhou-style normalized hypergraph Laplacian.

    We treat the contig-contact bipartite incidence as a hypergraph incidence matrix H,
    with hyperedge weights already encoded by `edges.tsv` weights (typically OrderNorm(k)*weight).
    We compute:
      A = H * D_e^{-1} * H^T   (contig-contig "2-step" similarity)
      S = D_v^{-1/2} * A * D_v^{-1/2}
      L = I - S
    Then run k-means on the first K eigenvectors of L (K chosen by an eigengap heuristic).

    This is an experimental coarse clustering alternative to Leiden.
    """
    logger = logger or logging.getLogger("porebin")
    graph_dir = graph_dir.resolve()
    out_bins_tsv = out_bins_tsv.resolve()

    contig_names = _read_contig_index(graph_dir / "contig_index.tsv")
    if not contig_names:
        raise GraphClusterError(f"No contigs found in {graph_dir / 'contig_index.tsv'}")

    meta = _read_graph_meta(graph_dir / "graph_meta.json")
    num_contigs = len(contig_names)
    num_contacts = int(meta.get("num_contacts", 0))
    if num_contacts <= 0:
        raise GraphClusterError(f"Invalid num_contacts in {graph_dir / 'graph_meta.json'}: {num_contacts}")

    rows, cols, data = _read_incidence(graph_dir / "edges.tsv")
    if not rows:
        raise GraphClusterError(f"No edges found in {graph_dir / 'edges.tsv'}")

    try:
        import numpy as np
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla
        from sklearn.cluster import KMeans
    except Exception as exc:  # pragma: no cover
        raise GraphClusterError(
            "spectral clustering requires optional dependencies: scipy, scikit-learn. "
            "Install them (e.g. pip install 'porebin[spectral]' or conda install scipy scikit-learn)."
        ) from exc

    logger.info(
        "Spectral clustering hypergraph incidence: contigs=%s, contacts=%s, incidences=%s",
        num_contigs,
        num_contacts,
        len(rows),
    )

    H = sp.coo_matrix((np.asarray(data, dtype=float), (np.asarray(rows), np.asarray(cols))), shape=(num_contigs, num_contacts)).tocsr()

    dv = np.asarray(H.sum(axis=1)).ravel()
    if not np.any(dv > 0):
        raise GraphClusterError("All contigs have zero incidence degree; cannot run spectral clustering.")
    de = np.asarray(H.sum(axis=0)).ravel()
    if not np.any(de > 0):
        raise GraphClusterError("All contacts have zero incidence degree; cannot run spectral clustering.")

    # A = H * D_e^{-1} * H^T
    de_inv = np.zeros_like(de, dtype=float)
    nz = de > 0
    de_inv[nz] = 1.0 / de[nz]
    A = H.multiply(de_inv) @ H.T
    A.setdiag(0.0)
    A.eliminate_zeros()

    dv_inv_sqrt = np.zeros_like(dv, dtype=float)
    nzv = dv > 0
    dv_inv_sqrt[nzv] = 1.0 / np.sqrt(dv[nzv])
    Dv_inv_sqrt = sp.diags(dv_inv_sqrt, format="csr")
    S = Dv_inv_sqrt @ A @ Dv_inv_sqrt

    # L = I - S
    L = sp.eye(num_contigs, format="csr") - S

    # Compute a small spectral basis; K will be chosen via eigengap.
    # Keep this conservative/fast by default.
    max_eigs = 50
    k_eigs = min(max_eigs, max(2, num_contigs - 1))
    try:
        evals, evecs = spla.eigsh(L, k=k_eigs, which="SA")
    except Exception as exc:
        raise GraphClusterError(f"eigsh failed on hypergraph Laplacian (n={num_contigs}, k={k_eigs}).") from exc

    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]

    K, k_meta = _choose_k_by_eigengap(evals, conservative=True)
    K = int(min(max(2, K), evecs.shape[1]))

    X = evecs[:, :K]
    # Row-normalize embedding.
    row_norm = np.linalg.norm(X, axis=1, keepdims=True)
    row_norm[row_norm == 0] = 1.0
    X = X / row_norm

    km = KMeans(n_clusters=K, n_init=10, random_state=seed)
    labels = km.fit_predict(X)

    logger.info("Spectral clustering chose K=%s (%s)", K, k_meta.get("method", "eigengap"))

    out_bins_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_bins_tsv.open("w", encoding="utf-8", newline="") as fh:
        fh.write("contig_name\tbin_id\n")
        for name, bin_id in zip(contig_names, labels.tolist(), strict=True):
            fh.write(f"{name}\t{bin_id}\n")

    logger.info(f"Wrote bins: {out_bins_tsv}")

    return {
        "method": "spectral_hypergraph_laplacian",
        "laplacian": "L = I - Dv^{-1/2} * (H * De^{-1} * H^T) * Dv^{-1/2}",
        "k_selected": K,
        "k_selection": k_meta,
        "n_eigs": int(k_eigs),
        "seed": seed,
    }


def cluster_leiden_pairwise(
    *,
    contig_index_tsv: Path,
    pairwise_edges_tsv_gz: Path,
    out_bins_tsv: Path,
    resolution: float = 1.0,
    seed: int = 0,
    logger: Optional[logging.Logger] = None,
) -> None:
    logger = logger or logging.getLogger("porebin")
    contig_index_tsv = contig_index_tsv.resolve()
    pairwise_edges_tsv_gz = pairwise_edges_tsv_gz.resolve()
    out_bins_tsv = out_bins_tsv.resolve()

    contig_names = _read_contig_index(contig_index_tsv)
    if not contig_names:
        raise GraphClusterError(f"No contigs found in {contig_index_tsv}")
    contig_to_idx = {name: i for i, name in enumerate(contig_names)}

    if not pairwise_edges_tsv_gz.exists():
        raise FileNotFoundError(f"Missing pairwise edges file: {pairwise_edges_tsv_gz}")

    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    with gzip.open(pairwise_edges_tsv_gz, "rt", encoding="utf-8", newline="") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = line.rstrip("\n").split("\t")
            if len(row) < 3:
                raise GraphClusterError(f"Invalid edge row in {pairwise_edges_tsv_gz}:{line_no}: {row}")
            a, b, w_s = row[0], row[1], row[2]
            if a == b:
                continue
            try:
                u = contig_to_idx[a]
            except KeyError as exc:
                raise GraphClusterError(
                    f"Contig '{a}' in {pairwise_edges_tsv_gz} not found in {contig_index_tsv}."
                ) from exc
            try:
                v = contig_to_idx[b]
            except KeyError as exc:
                raise GraphClusterError(
                    f"Contig '{b}' in {pairwise_edges_tsv_gz} not found in {contig_index_tsv}."
                ) from exc
            try:
                w = float(w_s)
            except ValueError as exc:
                raise GraphClusterError(f"Invalid edge weight in {pairwise_edges_tsv_gz}:{line_no}: {w_s!r}") from exc
            edges.append((u, v))
            weights.append(w)

    if not edges:
        raise GraphClusterError(f"No edges found in {pairwise_edges_tsv_gz}")

    logger.info(f"Clustering pairwise graph: contigs={len(contig_names)}, edges={len(edges)}")

    try:
        import igraph as ig
        import leidenalg
    except Exception as exc:  # pragma: no cover
        raise GraphClusterError(
            "cluster requires 'python-igraph' and 'leidenalg'. Install them, e.g. pip install python-igraph leidenalg."
        ) from exc

    g = ig.Graph(n=len(contig_names), edges=edges, directed=False)
    g.es["weight"] = weights

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
    )

    out_bins_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_bins_tsv.open("w", encoding="utf-8", newline="") as fh:
        fh.write("contig_name\tbin_id\n")
        for name, bin_id in zip(contig_names, partition.membership, strict=True):
            fh.write(f"{name}\t{bin_id}\n")

    logger.info(f"Wrote bins: {out_bins_tsv}")


def _read_graph_meta(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing graph meta file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_contig_index(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing contig index file: {path}")

    names_by_idx: dict[int, str] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if row[0] == "contig_name":
                continue
            if len(row) < 2:
                raise GraphClusterError(f"Invalid contig_index row in {path}: {row}")
            name, idx_s = row[0], row[1]
            try:
                idx = int(idx_s)
            except ValueError as exc:
                raise GraphClusterError(f"Invalid contig_idx in {path}: {idx_s!r}") from exc
            names_by_idx[idx] = name

    if not names_by_idx:
        return []
    max_idx = max(names_by_idx)
    contigs: list[str] = [""] * (max_idx + 1)
    for idx, name in names_by_idx.items():
        contigs[idx] = name
    if any(not n for n in contigs):
        raise GraphClusterError(f"Non-contiguous contig_idx values in {path}")
    return contigs


def _read_edges(path: Path, *, contig_offset: int) -> tuple[list[tuple[int, int]], list[float]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing edges file: {path}")

    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if row[0] == "contig_idx":
                continue
            if len(row) < 3:
                raise GraphClusterError(f"Invalid edge row in {path}: {row}")
            try:
                c_idx = int(row[0])
                contact_idx = int(row[1])
                w = float(row[2])
            except ValueError as exc:
                raise GraphClusterError(f"Invalid edge row in {path}: {row}") from exc
            edges.append((c_idx, contig_offset + contact_idx))
            weights.append(w)
    return edges, weights


def _read_incidence(path: Path) -> tuple[list[int], list[int], list[float]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing edges file: {path}")

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if row[0] == "contig_idx":
                continue
            if len(row) < 3:
                raise GraphClusterError(f"Invalid edge row in {path}: {row}")
            try:
                c_idx = int(row[0])
                contact_idx = int(row[1])
                w = float(row[2])
            except ValueError as exc:
                raise GraphClusterError(f"Invalid edge row in {path}: {row}") from exc
            rows.append(c_idx)
            cols.append(contact_idx)
            data.append(w)
    return rows, cols, data


def _choose_k_by_eigengap(evals, *, conservative: bool) -> tuple[int, dict]:
    """
    Pick K clusters from ascending Laplacian eigenvalues via eigengap.

    conservative=True chooses the *largest* K whose eigengap is close to the maximum gap,
    which tends to avoid overly coarse (potentially mixed) clusters.
    """
    import numpy as np

    vals = np.asarray(evals, dtype=float)
    if vals.size < 3:
        return 2, {"method": "fallback_small_n", "n_eigs": int(vals.size)}

    # gaps between consecutive eigenvalues; K corresponds to the index AFTER the gap.
    gaps = np.diff(vals)
    # ignore the first gap at i=0 (often near 0), consider K >= 2
    start = 1
    if gaps.size <= start:
        return 2, {"method": "fallback_no_gaps", "n_eigs": int(vals.size)}

    usable = gaps[start:]
    max_gap = float(np.max(usable))
    if max_gap <= 0:
        return 2, {"method": "fallback_nonpos_gap", "max_gap": max_gap, "n_eigs": int(vals.size)}

    ratio = 0.8 if conservative else 1.0
    candidates = np.where(gaps >= (ratio * max_gap))[0]
    candidates = candidates[candidates >= start]
    if candidates.size == 0:
        i = int(np.argmax(usable) + start)
        return i + 1, {"method": "eigengap_argmax", "gap_ratio": ratio, "max_gap": max_gap, "n_eigs": int(vals.size)}

    i = int(np.max(candidates) if conservative else np.min(candidates))
    return i + 1, {"method": "eigengap_conservative" if conservative else "eigengap", "gap_ratio": ratio, "max_gap": max_gap, "n_eigs": int(vals.size)}
