from __future__ import annotations

import csv
import gzip
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

from porebin.export import MIN_BIN_BP
from porebin.utils import iter_fasta_records


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
    bam: Optional[Path] = None,
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
    Default (no BAM): run k-means on the first K eigenvectors of L (K chosen by an eigengap heuristic).

    If `bam` is provided: run divisive (top-down) spectral bisection, using BAM coverage BIC (1-Gaussian vs 2-GMM
    on log1p coverage) as an automatic "stop" signal (no user thresholds). Small splits are discarded using the
    same 200kb minimum bin size as export/refine.

    This is an experimental coarse clustering alternative to Leiden.
    """
    logger = logger or logging.getLogger("porebin")
    graph_dir = graph_dir.resolve()
    out_bins_tsv = out_bins_tsv.resolve()
    bam = bam.resolve() if bam is not None else None

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

    if bam is not None:
        contigs_fasta_s = meta.get("input_contigs_fasta")
        contigs_fasta = Path(contigs_fasta_s).expanduser().resolve() if contigs_fasta_s else None
        if contigs_fasta is None or not contigs_fasta.exists():
            raise GraphClusterError(
                "Spectral clustering with BAM needs the original contigs FASTA path in graph_meta.json (input_contigs_fasta). "
                f"Got: {contigs_fasta_s!r}"
            )

        contig_len = _read_contig_lengths(contigs_fasta)
        if not contig_len:
            raise GraphClusterError(f"No contigs found in FASTA: {contigs_fasta}")

        lengths = np.asarray([contig_len.get(n, 0) for n in contig_names], dtype=int)
        min_contig_len, min_contig_len_meta = _auto_min_contig_len(lengths)
        keep_mask = lengths >= min_contig_len
        keep_idx = np.where(keep_mask)[0]
        if keep_idx.size < 2:
            raise GraphClusterError(
                f"Too few contigs >= MIN_CONTIG_LEN={min_contig_len} for spectral clustering (n={keep_idx.size})."
            )

        if not bam.exists():
            raise FileNotFoundError(f"BAM not found: {bam}")
        cov = _coverage_from_bam(bam, contig_names=contig_names, contig_len=contig_len, min_contig_len=min_contig_len, logger=logger)
        cov_arr = np.asarray([cov.get(n, 0.0) for n in contig_names], dtype=float)

        # Cache coverage for later refine (avoid scanning BAM twice).
        cov_out = out_bins_tsv.parent / "coverage" / "coverage.tsv"
        cov_out.parent.mkdir(parents=True, exist_ok=True)
        with cov_out.open("w", encoding="utf-8", newline="") as fh:
            fh.write("contig_name\tcoverage\n")
            for i, name in enumerate(contig_names):
                if contig_len.get(name, 0) >= min_contig_len:
                    fh.write(f"{name}\t{cov_arr[i]:.12g}\n")

        labels_keep, meta_bisect = _divisive_bisect_by_coverage_bic(
            A=A[keep_idx][:, keep_idx].tocsr(),
            contig_names=[contig_names[i] for i in keep_idx.tolist()],
            lengths=lengths[keep_idx],
            coverage=cov_arr[keep_idx],
            seed=seed,
            logger=logger,
        )

        out_bins_tsv.parent.mkdir(parents=True, exist_ok=True)
        with out_bins_tsv.open("w", encoding="utf-8", newline="") as fh:
            fh.write("contig_name\tbin_id\n")
            for i, b in enumerate(labels_keep):
                if b is None:
                    continue
                fh.write(f"{contig_names[keep_idx[i]]}\t{b}\n")

        logger.info(f"Wrote bins: {out_bins_tsv}")
        return {
            "method": "spectral_hypergraph_bisection",
            "laplacian": "L = I - Dv^{-1/2} * (H * De^{-1} * H^T) * Dv^{-1/2}",
            "seed": seed,
            "min_contig_len": int(min_contig_len),
            "min_contig_len_method": min_contig_len_meta,
            "min_bin_bp": int(MIN_BIN_BP),
            "stop_rule": "split if BIC2 < BIC1 on log1p(coverage), else stop",
            "coverage_tsv": str(cov_out),
            **meta_bisect,
        }

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


def _read_contig_lengths(contigs_fasta: Path) -> dict[str, int]:
    lengths: dict[str, int] = {}
    for name, _header, seq in iter_fasta_records(contigs_fasta):
        lengths[name] = len(seq)
    return lengths


def _auto_min_contig_len(lengths) -> tuple[int, dict]:
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise GraphClusterError("spectral clustering requires numpy (via scipy).") from exc

    vals = np.asarray(lengths, dtype=int)
    total_bp = int(np.sum(vals))
    bp_1000_2500 = int(np.sum(vals[(vals >= 1000) & (vals <= 2500)]))
    ratio = (bp_1000_2500 / total_bp) if total_bp else 0.0
    min_len = 1000 if ratio >= 0.05 else 2500
    meta = {
        "type": "semiBin_ratio_1000_2500",
        "ratio_1000_2500_bp": ratio,
        "ratio_threshold": 0.05,
        "rule": "min_contig_len=1000 if ratio>=0.05 else 2500",
    }
    return min_len, meta


def _coverage_from_bam(
    bam_path: Path,
    *,
    contig_names: list[str],
    contig_len: dict[str, int],
    min_contig_len: int,
    logger: logging.Logger,
) -> dict[str, float]:
    """
    Coverage proxy (fast, no per-base depth): aligned_bases_sum / contig_length.

    This mirrors refine's coverage computation, but is kept local to clustering so spectral can use BAM
    as an automatic stopping signal.
    """
    try:
        import pysam
    except Exception as exc:  # pragma: no cover
        raise GraphClusterError(
            "spectral clustering with BAM requires 'pysam'. Install it (pip install 'porebin[bam]' or conda install pysam)."
        ) from exc

    logger.info("Spectral clustering: computing coverage from BAM (aligned_bases_sum/contig_length)")
    bam = pysam.AlignmentFile(str(bam_path), "rb")
    ref_names = list(bam.references)
    aligned_bases: defaultdict[str, int] = defaultdict(int)
    for aln in bam.fetch(until_eof=True):
        if aln.is_unmapped:
            continue
        rid = aln.reference_id
        if rid is None or rid < 0 or rid >= len(ref_names):
            continue
        ref = ref_names[rid]
        aln_len = aln.query_alignment_length
        if not aln_len and aln.reference_end is not None and aln.reference_start is not None:
            aln_len = aln.reference_end - aln.reference_start
        if not aln_len:
            continue
        aligned_bases[ref] += int(aln_len)
    bam.close()

    cov: dict[str, float] = {}
    for name in contig_names:
        L = contig_len.get(name, 0)
        if L < min_contig_len:
            continue
        cov[name] = aligned_bases.get(name, 0) / float(L) if L else 0.0
    return cov


def _divisive_bisect_by_coverage_bic(
    *,
    A,
    contig_names: list[str],
    lengths,
    coverage,
    seed: int,
    logger: logging.Logger,
) -> tuple[list[Optional[int]], dict]:
    """
    Top-down spectral bisection on the contig-contig similarity graph, driven by BAM coverage multi-modality.

    Stop rule (automatic, no user thresholds):
      - If BIC(2-GMM) < BIC(1-Gaussian) on log1p(coverage): split (attempt).
      - Else: stop and emit the cluster as a bin.

    Clusters with total_bp < MIN_BIN_BP are treated as unbinned (labels=None).
    """
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.csgraph as csgraph

    A = A.tocsr()
    n = int(A.shape[0])
    if n != len(contig_names):
        raise GraphClusterError("Internal error: contig_names length does not match A shape.")

    lens = np.asarray(lengths, dtype=int)
    cov = np.asarray(coverage, dtype=float)

    labels: list[Optional[int]] = [None] * n
    final_clusters: list[np.ndarray] = []

    splits_attempted = 0
    splits_accepted = 0
    bic_triggered = 0
    stopped_unimodal = 0
    stopped_small = 0
    max_depth = 0

    # Stack items are (indices, depth)
    stack: list[tuple[np.ndarray, int]] = [(np.arange(n, dtype=int), 0)]
    while stack:
        idx, depth = stack.pop()
        max_depth = max(max_depth, depth)
        if idx.size < 2:
            stopped_small += 1
            continue

        bp = int(np.sum(lens[idx]))
        if bp < MIN_BIN_BP:
            stopped_small += 1
            continue

        trig, _bic_meta = _coverage_bic_trigger(cov[idx])
        if not trig:
            stopped_unimodal += 1
            final_clusters.append(idx)
            continue
        bic_triggered += 1

        # Build induced adjacency for this cluster
        A_sub = A[idx][:, idx].tocsr()
        if A_sub.nnz == 0:
            final_clusters.append(idx)
            continue

        # If already disconnected, split by components (parameter-free).
        n_comp, comp = csgraph.connected_components(A_sub, directed=False, return_labels=True)
        if n_comp > 1:
            for c in range(n_comp):
                sub = idx[np.where(comp == c)[0]]
                if sub.size >= 2:
                    stack.append((sub, depth + 1))
            splits_accepted += 1
            continue

        splits_attempted += 1
        left_local, right_local, _split_meta = _spectral_sweep_bisect(A_sub, seed=seed)
        if left_local.size == 0 or right_local.size == 0:
            final_clusters.append(idx)
            continue
        splits_accepted += 1
        stack.append((idx[left_local], depth + 1))
        stack.append((idx[right_local], depth + 1))

    # Keep only clusters that meet the minimum bin bp.
    kept: list[tuple[int, np.ndarray]] = []
    for idx in final_clusters:
        bp = int(np.sum(lens[idx]))
        if bp >= MIN_BIN_BP:
            kept.append((bp, idx))

    kept.sort(key=lambda x: (-x[0], int(np.min(x[1]))))
    for bin_id, (_bp, idx) in enumerate(kept):
        for i in idx.tolist():
            labels[int(i)] = int(bin_id)

    contigs_assigned = sum(1 for x in labels if x is not None)
    logger.info(
        "Spectral bisection: bins=%s assigned_contigs=%s/%s splits_attempted=%s splits_accepted=%s",
        len(kept),
        contigs_assigned,
        n,
        splits_attempted,
        splits_accepted,
    )

    return labels, {
        "k_selected": int(len(kept)),
        "bins_written": int(len(kept)),
        "contigs_assigned": int(contigs_assigned),
        "contigs_unbinned": int(n - contigs_assigned),
        "splits": {
            "attempted": int(splits_attempted),
            "accepted": int(splits_accepted),
            "bic_triggered": int(bic_triggered),
            "stopped_unimodal": int(stopped_unimodal),
            "stopped_small": int(stopped_small),
            "max_depth": int(max_depth),
        },
        "coverage_bic": {"min_n": 20, "method": "bic_1gauss_vs_2gmm_on_log1p_cov"},
    }


def _spectral_sweep_bisect(A_sub, *, seed: int) -> tuple["np.ndarray", "np.ndarray", dict]:
    import numpy as np
    import scipy.sparse.linalg as spla

    n = int(A_sub.shape[0])
    if n < 2:
        return np.asarray([], dtype=int), np.asarray([], dtype=int), {"method": "fallback_small_n", "n": n}

    deg = np.asarray(A_sub.sum(axis=1)).ravel()
    d_inv_sqrt = np.zeros_like(deg, dtype=float)
    nz = deg > 0
    d_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])

    def matvec(x):
        y = d_inv_sqrt * x
        y = A_sub @ y
        y = d_inv_sqrt * y
        return x - y

    L = spla.LinearOperator((n, n), matvec=matvec, dtype=float)
    evals, evecs = spla.eigsh(L, k=2, which="SA")
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]

    f = evecs[:, 1]
    node_order = np.argsort(f)
    best_t, best_phi = _sweep_best_conductance(A_sub, degrees=deg, order=node_order)
    left = node_order[:best_t]
    right = node_order[best_t:]
    return left, right, {"lambda2": float(evals[1]), "phi": float(best_phi), "method": "spectral_sweep_cut"}


def _sweep_best_conductance(A_sub, *, degrees, order):
    import numpy as np

    n = int(A_sub.shape[0])
    if n < 2:
        return 0, float("inf")

    deg = np.asarray(degrees, dtype=float)
    vol_total = float(np.sum(deg))
    if vol_total <= 0:
        return n // 2, float("inf")

    in_s = np.zeros(n, dtype=bool)
    cut = 0.0
    vol_s = 0.0
    best_phi = float("inf")
    best_t = n // 2

    for t in range(1, n):
        v = int(order[t - 1])
        in_s[v] = True
        vol_s += float(deg[v])

        row_start = int(A_sub.indptr[v])
        row_end = int(A_sub.indptr[v + 1])
        neigh = A_sub.indices[row_start:row_end]
        wts = A_sub.data[row_start:row_end]
        for u, w in zip(neigh, wts, strict=True):
            cut += -float(w) if in_s[int(u)] else float(w)

        vol_t = vol_total - vol_s
        if vol_s <= 0 or vol_t <= 0:
            continue
        phi = cut / min(vol_s, vol_t)
        if phi < best_phi:
            best_phi = phi
            best_t = t

    return best_t, best_phi


def _coverage_bic_trigger(values) -> tuple[bool, dict]:
    xs = [math.log1p(float(v)) for v in values if float(v) >= 0.0]
    n = len(xs)
    if n < 20:
        return False, {"type": "skip_small_n", "n": n}
    bic1 = _bic(xs, k=1)
    bic2 = _bic(xs, k=2)
    return (bic2 < bic1), {"type": "bic_1v2", "n": n, "bic1": bic1, "bic2": bic2, "delta": bic1 - bic2}


def _bic(xs: list[float], *, k: int) -> float:
    if not xs:
        return 0.0
    if k == 1:
        ll = _loglik_1gauss(xs)
        p = 2
    elif k == 2:
        ll = _loglik_2gmm(xs)
        p = 5
    else:
        raise ValueError("k must be 1 or 2")
    n = len(xs)
    return -2.0 * ll + p * math.log(n)


def _loglik_1gauss(xs: list[float]) -> float:
    n = len(xs)
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / n
    var = max(var, 1e-6)
    return sum(_log_norm_pdf(x, mu, var) for x in xs)


def _loglik_2gmm(xs: list[float]) -> float:
    n = len(xs)
    xs_sorted = sorted(xs)
    mu1 = xs_sorted[n // 4]
    mu2 = xs_sorted[(3 * n) // 4]
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / n
    var1 = var2 = max(var, 1e-3)
    w1 = 0.5

    for _ in range(50):
        r1_sum = 0.0
        r2_sum = 0.0
        mu1_num = 0.0
        mu2_num = 0.0
        var1_num = 0.0
        var2_num = 0.0

        for x in xs:
            l1 = math.log(w1) + _log_norm_pdf(x, mu1, var1)
            l2 = math.log(1.0 - w1) + _log_norm_pdf(x, mu2, var2)
            m = max(l1, l2)
            d = math.exp(l1 - m) + math.exp(l2 - m)
            r1 = math.exp(l1 - m) / d
            r2 = 1.0 - r1
            r1_sum += r1
            r2_sum += r2
            mu1_num += r1 * x
            mu2_num += r2 * x

        if r1_sum <= 1e-9 or r2_sum <= 1e-9:
            break

        w1 = max(1e-3, min(1.0 - 1e-3, r1_sum / n))
        mu1 = mu1_num / r1_sum
        mu2 = mu2_num / r2_sum

        for x in xs:
            l1 = math.log(w1) + _log_norm_pdf(x, mu1, var1)
            l2 = math.log(1.0 - w1) + _log_norm_pdf(x, mu2, var2)
            m = max(l1, l2)
            d = math.exp(l1 - m) + math.exp(l2 - m)
            r1 = math.exp(l1 - m) / d
            r2 = 1.0 - r1
            var1_num += r1 * (x - mu1) ** 2
            var2_num += r2 * (x - mu2) ** 2

        var1 = max(var1_num / r1_sum, 1e-6)
        var2 = max(var2_num / r2_sum, 1e-6)

    ll = 0.0
    for x in xs:
        l1 = math.log(w1) + _log_norm_pdf(x, mu1, var1)
        l2 = math.log(1.0 - w1) + _log_norm_pdf(x, mu2, var2)
        m = max(l1, l2)
        ll += m + math.log(math.exp(l1 - m) + math.exp(l2 - m))
    return ll


def _log_norm_pdf(x: float, mu: float, var: float) -> float:
    return -0.5 * (math.log(2.0 * math.pi * var) + ((x - mu) ** 2) / var)
