from __future__ import annotations

import csv
import gzip
import json
import logging
import math
from pathlib import Path
from typing import Optional


class GraphClusterError(RuntimeError):
    pass


def cluster_spectral_hypergraph(
    *,
    graph_dir: Path,
    out_bins_tsv: Path,
    seed: int = 0,
    bam: Optional[Path] = None,
    threads: int = 1,
    logger: Optional[logging.Logger] = None,
) -> dict:
    """
    Joint hypergraph spectral embedding + HDBSCAN (v2).

    This replaces the legacy "recursive bisection + BIC stop rule" spectral coarse clustering.
    It does NOT construct a dense |V|x|V| matrix and does NOT do clique expansion.

    Hypergraphs:
      1) Contact hypergraph from contacts.parquet (soft incidence P_{e,c})
      2) Feature hypergraph from tetranucleotide composition (+ optional coverage) via kNN

    Joint operator:
      Theta_joint = lambda_contact * Theta_contact + (1-lambda_contact) * Theta_feature

    Embedding:
      top (d+1) eigenvectors of Theta_joint (which="LA"), drop the first, L2-normalize rows.

    Clustering:
      HDBSCAN on the embedding (min_cluster_size=5), labels==-1 => unbinned.
    """
    logger = logger or logging.getLogger("porebin")
    graph_dir = graph_dir.resolve()
    out_bins_tsv = out_bins_tsv.resolve()
    bam = bam.resolve() if bam is not None else None
    threads = max(1, int(threads))

    meta = _read_graph_meta(graph_dir / "graph_meta.json")
    try:
        import numpy as np
        import scipy
    except Exception as exc:  # pragma: no cover
        raise GraphClusterError(
            "spectral clustering requires optional dependencies: scipy + scikit-learn + hdbscan. "
            "Install them (e.g. pip install 'porebin[spectral]' or conda install scipy)."
        ) from exc

    from porebin.hypergraph_joint_spectral import (
        JointSpectralError,
        build_contact_incidence_from_parquet,
        build_feature_incidence,
        build_feature_knn_edges,
        hdbscan_cluster,
        load_contig_index,
        load_coverage_feature_optional,
        load_or_build_tnf136_features,
        make_theta_operator,
        postprocess_labels_by_contact_components,
        spectral_embed_joint,
        write_bins_tsv,
        zscore_features,
    )

    contig_name_to_idx, idx_to_name = load_contig_index(graph_dir)
    V = len(idx_to_name)
    if V < 2:
        raise GraphClusterError("Too few contigs for spectral clustering.")

    contigs_fasta_s = meta.get("input_contigs_fasta")
    contacts_parquet_s = meta.get("input_contacts_parquet")
    if not contigs_fasta_s or not contacts_parquet_s:
        raise GraphClusterError(
            "spectral_v2 requires graph_meta.json to contain input_contigs_fasta and input_contacts_parquet."
        )
    contigs_fasta = Path(contigs_fasta_s).expanduser().resolve()
    contacts_parquet = Path(contacts_parquet_s).expanduser().resolve()

    # Optional coverage.tsv is used only as a feature; no BAM scan in v2.
    cov_path = graph_dir.parent / "coverage" / "coverage.tsv"
    cov_path = cov_path if cov_path.exists() else None
    lambda_contact = 0.6 if cov_path is not None else 0.7
    knn_k = 15

    try:
        contact = build_contact_incidence_from_parquet(contacts_parquet, contig_name_to_idx, logger=logger)
        X_comp = load_or_build_tnf136_features(
            graph_dir=graph_dir, contigs_fasta=contigs_fasta, contig_name_to_idx=contig_name_to_idx, logger=logger
        )
        x_cov, coverage_missing_count, coverage_used = load_coverage_feature_optional(cov_path, contig_name_to_idx)
        X = np.concatenate([X_comp.astype(np.float32, copy=False), x_cov.reshape(-1, 1)], axis=1)
        X = zscore_features(X)

        neighbors = build_feature_knn_edges(X, knn_k)
        feature = build_feature_incidence(neighbors, knn_k)

        contact_theta = make_theta_operator(contact.H_csr, contact.W, contact.De, contact.Dv)
        feature_theta = make_theta_operator(feature.H_csr, feature.W, feature.De, feature.Dv)

        d = min(128, max(32, int(math.floor(math.log2(float(V)))) * 4))
        d = min(d, max(1, V - 2))

        Z = spectral_embed_joint(
            contact_op=contact_theta.op,
            feature_op=feature_theta.op,
            lambda_contact=lambda_contact,
            d=int(d),
            seed=seed,
        )

        labels, hmeta = hdbscan_cluster(Z, min_cluster_size=5, threads=threads)
        # Contact-isolated contigs are treated as unbinned.
        labels = np.asarray(labels, dtype=int)
        labels[np.asarray(contact.isolated_mask_contact, dtype=bool)] = -1
        labels, pp = postprocess_labels_by_contact_components(
            labels, contact.H_csr, min_cluster_size=5
        )

        num_bins, unbinned = write_bins_tsv(out_bins_tsv, idx_to_name, labels)
        logger.info("Wrote bins: %s", out_bins_tsv)

        sklearn_version = None
        try:
            import sklearn

            sklearn_version = getattr(sklearn, "__version__", None)
        except Exception:
            sklearn_version = None

        return {
            "method": "spectral_v2_joint_contact_feature_hdbscan",
            "spectral_v2_joint_enabled": True,
            "lambda_contact": float(lambda_contact),
            "d": int(d),
            "knn_k": int(knn_k),
            "dropped_edges_singleton_contact": int(contact.dropped_edges_singleton_contact),
            "isolated_contigs_count_contact": int(np.sum(contact.isolated_mask_contact)),
            "coverage_tsv": (str(cov_path) if cov_path is not None else None),
            "coverage_used": bool(coverage_used),
            "coverage_missing_count": int(coverage_missing_count),
            "hdbscan": hmeta,
            "contact_component_postprocess": {
                "components_total": pp.components_total,
                "components_promoted_from_noise": pp.components_promoted_from_noise,
                "noise_reassigned_by_component": pp.noise_reassigned_by_component,
                "labels_split_by_component": pp.labels_split_by_component,
            },
            "versions": {"scipy": getattr(scipy, "__version__", None), "sklearn": sklearn_version},
            "num_bins": int(num_bins),
            "unbinned_count": int(unbinned),
        }
    except JointSpectralError as exc:
        raise GraphClusterError(str(exc)) from exc


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

from porebin.cluster_legacy import (
    _auto_min_contig_len,
    _bic,
    _choose_k_by_eigengap,
    _coverage_bic_trigger,
    _coverage_from_bam,
    _coverage_from_tsv,
    _divisive_bisect_by_auto_bic,
    _divisive_bisect_by_coverage_bic,
    _graph_bic_trigger,
    _log_norm_pdf,
    _loglik_1gauss,
    _loglik_2gmm,
    _read_contig_lengths,
    _spectral_sweep_bisect,
    _sweep_best_conductance,
)
