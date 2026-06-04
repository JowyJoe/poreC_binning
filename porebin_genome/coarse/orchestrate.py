"""Genome-centric coarse discovery orchestration for candidate genome bins."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from porebin_genome.coarse.baseline import run_pairwise_leiden_baseline
from porebin_genome.coarse.cluster import hdbscan_cluster, write_coarse_bins_tsv
from porebin_genome.coarse.contact import build_contact_incidence_from_parquet
from porebin_genome.coarse.embed import auto_embedding_dim, spectral_embed_joint
from porebin_genome.coarse.features import build_feature_matrix
from porebin_genome.coarse.hyperedge_embedding import (
    DEFAULT_HYPEREDGE_EMBEDDING_DIM,
    DEFAULT_HYPEREDGE_EMBEDDING_EPOCHS,
    DEFAULT_HYPEREDGE_EMBEDDING_LR,
    DEFAULT_HYPEREDGE_FEATURE_GUARD,
    DEFAULT_HYPEREDGE_VAE_BETA,
    DEFAULT_HYPEREDGE_VAE_BATCH_SIZE,
    DEFAULT_HYPEREDGE_VAE_LAMBDA,
    load_hyperedge_embedding_tsv,
    train_hyperedge_embedding,
)
from porebin_genome.coarse.hyperedge_weight import (
    DEFAULT_HYPERGRAPH_WEIGHT_ETA,
    normalize_contact_weight_mode,
)
from porebin_genome.coarse.metadata import build_coarse_run_record
from porebin_genome.coarse.operator import (
    DEFAULT_ADAPTIVE_DROP_RATIO,
    DEFAULT_ADAPTIVE_K,
    DEFAULT_ADAPTIVE_K_MAX,
    DEFAULT_ADAPTIVE_MIN_K,
    DEFAULT_ADAPTIVE_MUTUAL_KNN,
    build_adaptive_feature_knn_edges,
    build_adaptive_knn_fallback_report_rows,
    build_adaptive_knn_meta,
    build_feature_incidence,
    build_feature_knn_edges,
    make_theta_operator,
    write_adaptive_knn_report,
)
from porebin_genome.coarse.pairwise import DEFAULT_PAIRWISE_ALPHA_MIN, build_clique_pairwise_contacts
from porebin_genome.coarse.pairwise_normalize import normalize_pairwise_contacts
from porebin_genome.io.contacts import validate_contacts_parquet_core_schema
from porebin_genome.io.contracts import build_pipeline_layout
from porebin_genome.io.coverage import read_coverage_tsv
from porebin_genome.io.coverage import validate_coverage_tsv
from porebin_genome.io.fasta import iter_fasta_names, read_contig_lengths
from porebin_genome.io.runtime import ensure_dir, write_json


DEFAULT_FEATURE_MODE = "tnf_plus_cov"
DEFAULT_FEATURE_KNN_K = 15
ADAPTIVE_FEATURE_KNN_MODE = "adaptive"
DEFAULT_LAMBDA_CONTACT = 0.50
DEFAULT_HDBSCAN_SELECTION_METHOD = "leaf"
DEFAULT_CONTACT_WEIGHT_MODE = "hypergraph-native"
DEFAULT_COARSE_METHOD = "spectral"
COARSE_METHODS = ("spectral", "hgvae")


@dataclass(frozen=True)
class CoarseRunResult:
    """Outputs emitted by the coarse discovery layer."""

    bins_tsv: Path
    run_json: Path
    implemented: bool
    contigs_total: int
    n_bins: int
    n_contigs_clustered: int
    n_contigs_unbinned: int
    hyperedge_embedding_tsv: Path | None = None
    hyperedge_embedding_meta_json: Path | None = None


def _read_contig_index(contigs_fasta: Path) -> tuple[dict[str, int], list[str]]:
    idx_to_name = list(iter_fasta_names(contigs_fasta))
    if len(idx_to_name) < 2:
        raise RuntimeError("At least two contigs are required for coarse discovery.")
    if len(set(idx_to_name)) != len(idx_to_name):
        raise RuntimeError("contigs.fasta contains duplicate contig names.")
    return {name: idx for idx, name in enumerate(idx_to_name)}, idx_to_name


def _default_hdbscan_min_cluster_size(n_contigs: int) -> int:
    if n_contigs < 10:
        return 2
    if n_contigs < 50:
        return 3
    return 5


def _get_logger(logger: Optional[object]) -> object:
    return logger if logger is not None else logging.getLogger("porebin_genome")


def _log(logger: Optional[object], level: str, message: str) -> None:
    target = _get_logger(logger)
    log_fn = getattr(target, level, None)
    if callable(log_fn):
        log_fn(message)


def _parse_feature_knn_k(feature_knn_k: int | str) -> tuple[str, int]:
    if isinstance(feature_knn_k, str):
        text = feature_knn_k.strip().lower()
        if text == ADAPTIVE_FEATURE_KNN_MODE:
            return ADAPTIVE_FEATURE_KNN_MODE, DEFAULT_FEATURE_KNN_K
        try:
            fixed_k = int(text)
        except ValueError as exc:
            raise ValueError("--knn-k must be a positive integer or 'adaptive'.") from exc
    else:
        fixed_k = int(feature_knn_k)
    if fixed_k < 1:
        raise ValueError("--knn-k must be a positive integer or 'adaptive'.")
    return "fixed", fixed_k


def normalize_coarse_method(value: str) -> str:
    """Normalize coarse discovery method names."""
    method = str(value).strip().lower().replace("-", "_")
    if method == "hg_vae":
        method = "hgvae"
    if method not in COARSE_METHODS:
        raise ValueError("--coarse-method must be 'spectral' or 'hgvae'.")
    return method


def _count_neighbor_edges(neighbors: "object") -> int:
    import numpy as np

    arr = np.asarray(neighbors, dtype=np.int32)
    if arr.size == 0:
        return 0
    return int(np.sum(arr >= 0))


def run_coarse_discovery(
    *,
    contigs_fasta: Path,
    contacts_parquet: Path,
    coverage_tsv: Path,
    out_dir: Path,
    logger: Optional[object] = None,
    seed: int = 0,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    feature_knn_k: int | str = ADAPTIVE_FEATURE_KNN_MODE,
    coarse_method: str = DEFAULT_COARSE_METHOD,
    embedding_dim: Optional[int] = None,
    hdbscan_min_cluster_size: Optional[int] = None,
    hdbscan_min_samples: Optional[int] = None,
    hdbscan_selection_method: str = DEFAULT_HDBSCAN_SELECTION_METHOD,
    threads: int = 1,
    contact_weight_mode: str = DEFAULT_CONTACT_WEIGHT_MODE,
    hypergraph_weight_eta: float = DEFAULT_HYPERGRAPH_WEIGHT_ETA,
    run_pairwise_baseline: bool = False,
    pairwise_alpha_min: float = DEFAULT_PAIRWISE_ALPHA_MIN,
    run_hyperedge_embedding: bool = False,
    hyperedge_embedding_dim: int = DEFAULT_HYPEREDGE_EMBEDDING_DIM,
    hyperedge_embedding_epochs: int = DEFAULT_HYPEREDGE_EMBEDDING_EPOCHS,
    hyperedge_embedding_lr: float = DEFAULT_HYPEREDGE_EMBEDDING_LR,
    hyperedge_vae_beta: float = DEFAULT_HYPEREDGE_VAE_BETA,
    hyperedge_vae_lambda: float = DEFAULT_HYPEREDGE_VAE_LAMBDA,
    hyperedge_vae_batch_size: int = DEFAULT_HYPEREDGE_VAE_BATCH_SIZE,
    hyperedge_feature_guard: bool = DEFAULT_HYPEREDGE_FEATURE_GUARD,
) -> CoarseRunResult:
    """Run real coarse candidate genome-bin discovery without any component/noise postprocess."""
    layout = build_pipeline_layout(out_dir)
    ensure_dir(layout.coarse_dir)

    contigs_fasta = contigs_fasta.resolve()
    contacts_parquet = contacts_parquet.resolve()
    coverage_tsv = coverage_tsv.resolve()

    if not contigs_fasta.exists():
        raise FileNotFoundError(f"Contigs FASTA not found: {contigs_fasta}")
    validate_contacts_parquet_core_schema(contacts_parquet)
    validate_coverage_tsv(coverage_tsv)

    contig_name_to_idx, idx_to_name = _read_contig_index(contigs_fasta)
    contigs_total = len(idx_to_name)
    weight_mode = normalize_contact_weight_mode(contact_weight_mode)
    coarse_method_name = normalize_coarse_method(coarse_method)
    feature_matrix = build_feature_matrix(
        contigs_fasta=contigs_fasta,
        coverage_tsv=coverage_tsv,
        contig_name_to_idx=contig_name_to_idx,
        feature_mode=feature_mode,
        logger=logger,
    )

    pairwise_outputs: dict[str, object] = {}
    if bool(run_pairwise_baseline):
        contig_lengths = read_contig_lengths(contigs_fasta)
        coverage_by_contig = read_coverage_tsv(coverage_tsv)
        pairwise_result = build_clique_pairwise_contacts(
            contacts_path=contacts_parquet,
            contig_name_to_idx=contig_name_to_idx,
            out_parquet=layout.pairwise_clique_contacts_parquet,
            meta_json=layout.pairwise_clique_meta_json,
            alpha_min=float(pairwise_alpha_min),
        )
        pairwise_outputs.update(
            {
                "pairwise_clique_contacts_parquet": str(pairwise_result.contacts_parquet),
                "pairwise_clique_meta_json": str(pairwise_result.meta_json),
            }
        )
        normalization_result = normalize_pairwise_contacts(
            pairwise_contacts_path=pairwise_result.contacts_parquet,
            contig_lengths=contig_lengths,
            coverage_by_contig=coverage_by_contig,
            out_parquet=layout.pairwise_normalized_contacts_parquet,
            meta_json=layout.pairwise_normalization_meta_json,
        )
        pairwise_outputs.update(
            {
                "pairwise_normalized_contacts_parquet": str(normalization_result.normalized_contacts_parquet),
                "pairwise_normalization_meta_json": str(normalization_result.meta_json),
            }
        )
        if bool(run_pairwise_baseline):
            baseline_result = run_pairwise_leiden_baseline(
                pairwise_normalized_contacts_path=normalization_result.normalized_contacts_parquet,
                idx_to_name=idx_to_name,
                bins_tsv=layout.pairwise_leiden_bins_tsv,
                sweep_tsv=layout.pairwise_leiden_sweep_tsv,
                meta_json=layout.pairwise_baseline_meta_json,
            )
            pairwise_outputs.update(
                {
                    "pairwise_leiden_bins_tsv": str(baseline_result.bins_tsv),
                    "pairwise_leiden_sweep_tsv": str(baseline_result.sweep_tsv),
                    "pairwise_baseline_meta_json": str(baseline_result.meta_json),
                }
            )
    hyperedge_embedding_outputs: dict[str, object] = {}
    hyperedge_embedding_tsv: Path | None = None
    hyperedge_embedding_meta_json: Path | None = None
    train_hgvae_embedding = bool(run_hyperedge_embedding or coarse_method_name == "hgvae")
    if train_hgvae_embedding:
        embedding_result = train_hyperedge_embedding(
            contacts_path=contacts_parquet,
            contig_name_to_idx=contig_name_to_idx,
            idx_to_name=idx_to_name,
            out_tsv=layout.hyperedge_embedding_tsv,
            meta_json=layout.hyperedge_embedding_meta_json,
            contact_weight_mode=weight_mode,
            hypergraph_weight_eta=float(hypergraph_weight_eta),
            alpha_min=float(pairwise_alpha_min),
            embedding_dim=int(hyperedge_embedding_dim),
            epochs=int(hyperedge_embedding_epochs),
            learning_rate=float(hyperedge_embedding_lr),
            feature_matrix=feature_matrix.X,
            feature_guard=bool(hyperedge_feature_guard),
            coverage_feature_present=bool(feature_matrix.coverage_used and feature_matrix.feature_mode == "tnf_plus_cov"),
            beta_kl=float(hyperedge_vae_beta),
            lambda_hypergraph=float(hyperedge_vae_lambda),
            hyperedge_batch_size=int(hyperedge_vae_batch_size),
            seed=int(seed),
        )
        hyperedge_embedding_tsv = embedding_result.embedding_tsv
        hyperedge_embedding_meta_json = embedding_result.meta_json
        hyperedge_embedding_outputs.update(
            {
                "hyperedge_embedding_tsv": str(embedding_result.embedding_tsv),
                "hyperedge_embedding_meta_json": str(embedding_result.meta_json),
            }
        )

    contact = build_contact_incidence_from_parquet(
        contacts_parquet,
        contig_name_to_idx,
        weight_mode=weight_mode,
        hypergraph_weight_eta=float(hypergraph_weight_eta),
        logger=logger,
    )
    feature_knn_mode, fixed_feature_knn_k = _parse_feature_knn_k(feature_knn_k)
    adaptive_k_meta: dict[str, object] | None = None
    feature = None
    if coarse_method_name == "spectral" and feature_knn_mode == ADAPTIVE_FEATURE_KNN_MODE:
        try:
            adaptive = build_adaptive_feature_knn_edges(
                feature_matrix.X,
                contig_ids=idx_to_name,
                min_k=DEFAULT_ADAPTIVE_MIN_K,
                default_k=DEFAULT_ADAPTIVE_K,
                k_max=DEFAULT_ADAPTIVE_K_MAX,
                drop_ratio=DEFAULT_ADAPTIVE_DROP_RATIO,
                mutual_knn=DEFAULT_ADAPTIVE_MUTUAL_KNN,
            )
            neighbors = adaptive.neighbors
            if contigs_total > 1 and _count_neighbor_edges(neighbors) == 0:
                raise RuntimeError("adaptive mutual-kNN filtering removed all feature-neighbor edges")
            adaptive_k_meta = adaptive.meta
            write_adaptive_knn_report(layout.adaptive_k_report_tsv, adaptive.report_rows)
            write_json(layout.adaptive_k_meta_json, adaptive_k_meta)
            _log(
                logger,
                "info",
                (
                    "[adaptive-k] built local adaptive feature graph: "
                    f"min_k={DEFAULT_ADAPTIVE_MIN_K}, default_k={DEFAULT_ADAPTIVE_K}, "
                    f"k_max={DEFAULT_ADAPTIVE_K_MAX}, mutual_knn={DEFAULT_ADAPTIVE_MUTUAL_KNN}"
                ),
            )
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
            _log(logger, "warning", f"[adaptive-k] fallback to fixed k=15: {fallback_reason}")
            neighbors = build_feature_knn_edges(feature_matrix.X, DEFAULT_FEATURE_KNN_K)
            adaptive_k_meta = build_adaptive_knn_meta(
                min_k=DEFAULT_ADAPTIVE_MIN_K,
                default_k=DEFAULT_ADAPTIVE_K,
                k_max=DEFAULT_ADAPTIVE_K_MAX,
                drop_ratio=DEFAULT_ADAPTIVE_DROP_RATIO,
                mutual_knn=DEFAULT_ADAPTIVE_MUTUAL_KNN,
                fallback_used=True,
                fallback_reason=fallback_reason,
                fallback_k=DEFAULT_FEATURE_KNN_K,
                n_contigs=contigs_total,
            )
            write_adaptive_knn_report(
                layout.adaptive_k_report_tsv,
                build_adaptive_knn_fallback_report_rows(
                    contig_ids=idx_to_name,
                    fallback_k=DEFAULT_FEATURE_KNN_K,
                    n_contigs=contigs_total,
                ),
            )
            write_json(layout.adaptive_k_meta_json, adaptive_k_meta)
    else:
        neighbors = (
            build_feature_knn_edges(feature_matrix.X, fixed_feature_knn_k)
            if coarse_method_name == "spectral"
            else None
        )
    if neighbors is not None:
        feature = build_feature_incidence(neighbors)

    if coarse_method_name == "hgvae":
        if hyperedge_embedding_tsv is None:
            raise RuntimeError("HG-VAE coarse method requires a trained hyperedge embedding.")
        embedding_contigs, Z, _support = load_hyperedge_embedding_tsv(hyperedge_embedding_tsv)
        if embedding_contigs != idx_to_name:
            raise RuntimeError("HG-VAE embedding contig order does not match coarse contig order.")
        embedding_source = "hgvae"
        lambda_contact_for_record = None
        feature_knn_k_for_record = 0
    else:
        if feature is None:
            raise RuntimeError("Spectral coarse method requires a feature incidence graph.")
        contact_theta = make_theta_operator(contact.H_csr, contact.W, contact.De, contact.Dv)
        feature_theta = make_theta_operator(feature.H_csr, feature.W, feature.De, feature.Dv)
        target_embedding_dim = auto_embedding_dim(contigs_total) if embedding_dim is None else int(embedding_dim)
        Z = spectral_embed_joint(
            contact_op=contact_theta.op,
            feature_op=feature_theta.op,
            lambda_contact=DEFAULT_LAMBDA_CONTACT,
            d=target_embedding_dim,
            seed=int(seed),
        )
        embedding_source = "joint_spectral"
        lambda_contact_for_record = DEFAULT_LAMBDA_CONTACT
        feature_knn_k_for_record = int(feature.feature_knn_k)

    labels, hdbscan_meta = hdbscan_cluster(
        Z,
        min_cluster_size=(
            _default_hdbscan_min_cluster_size(contigs_total)
            if hdbscan_min_cluster_size is None
            else int(hdbscan_min_cluster_size)
        ),
        min_samples=hdbscan_min_samples,
        selection_method=hdbscan_selection_method,
        threads=int(max(1, threads)),
    )

    cluster_result = write_coarse_bins_tsv(
        out_path=layout.coarse_bins_tsv,
        idx_to_name=idx_to_name,
        labels=labels,
    )
    run_record = build_coarse_run_record(
        contigs_fasta=str(contigs_fasta),
        contacts_parquet=str(contacts_parquet),
        coverage_tsv=str(coverage_tsv),
        bins_tsv=str(layout.coarse_bins_tsv),
        labels=cluster_result.labels,
        feature_mode=feature_matrix.feature_mode,
        embedding_dim=int(Z.shape[1]),
        lambda_contact=lambda_contact_for_record,
        contact_hyperedge_count=int(contact.contact_hyperedge_count),
        feature_knn_k=int(feature_knn_k_for_record),
        dropped_singleton_contacts=int(contact.dropped_singleton_contacts),
        coverage_used=bool(feature_matrix.coverage_used),
        coverage_missing_count=int(feature_matrix.coverage_missing_count),
        hdbscan_meta=hdbscan_meta,
    )
    run_record["coarse_method"] = coarse_method_name
    run_record["embedding_source"] = embedding_source
    run_record["clustering_method"] = "hdbscan"
    run_record["feature_knn_mode"] = feature_knn_mode if coarse_method_name == "spectral" else "not_used"
    run_record["contact_weight_mode"] = weight_mode
    run_record["hypergraph_weight_eta"] = float(hypergraph_weight_eta)
    run_record["pairwise_alpha_min"] = float(pairwise_alpha_min)
    run_record["pairwise_baseline_enabled"] = bool(run_pairwise_baseline)
    run_record["hyperedge_embedding_enabled"] = bool(train_hgvae_embedding)
    run_record["hyperedge_embedding_requested"] = bool(run_hyperedge_embedding)
    run_record["hyperedge_embedding_dim"] = int(hyperedge_embedding_dim)
    run_record["hyperedge_embedding_epochs"] = int(hyperedge_embedding_epochs)
    run_record["hyperedge_embedding_lr"] = float(hyperedge_embedding_lr)
    run_record["hyperedge_vae_beta"] = float(hyperedge_vae_beta)
    run_record["hyperedge_vae_lambda"] = float(hyperedge_vae_lambda)
    run_record["hyperedge_vae_batch_size"] = int(hyperedge_vae_batch_size)
    run_record["hyperedge_feature_guard"] = bool(hyperedge_feature_guard)
    run_record["outputs"].update(pairwise_outputs)
    run_record["outputs"].update(hyperedge_embedding_outputs)
    if adaptive_k_meta is not None:
        run_record["outputs"]["adaptive_k_report_tsv"] = str(layout.adaptive_k_report_tsv)
        run_record["outputs"]["adaptive_k_meta_json"] = str(layout.adaptive_k_meta_json)
        run_record["adaptive_k"] = dict(adaptive_k_meta)
    write_json(layout.coarse_run_json, run_record)

    return CoarseRunResult(
        bins_tsv=layout.coarse_bins_tsv,
        run_json=layout.coarse_run_json,
        implemented=True,
        contigs_total=contigs_total,
        n_bins=cluster_result.n_bins,
        n_contigs_clustered=cluster_result.n_contigs_clustered,
        n_contigs_unbinned=cluster_result.n_contigs_unbinned,
        hyperedge_embedding_tsv=hyperedge_embedding_tsv,
        hyperedge_embedding_meta_json=hyperedge_embedding_meta_json,
    )
