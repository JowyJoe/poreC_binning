"""Genome-centric coarse discovery orchestration for candidate genome bins."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from porebin_genome.coarse.cluster import hdbscan_cluster, write_coarse_bins_tsv
from porebin_genome.coarse.contact import build_contact_incidence_from_parquet
from porebin_genome.coarse.embed import auto_embedding_dim, spectral_embed_joint
from porebin_genome.coarse.features import build_feature_matrix
from porebin_genome.coarse.metadata import build_coarse_run_record
from porebin_genome.coarse.operator import build_feature_incidence, build_feature_knn_edges, make_theta_operator
from porebin_genome.io.contacts import validate_contacts_parquet_core_schema
from porebin_genome.io.contracts import build_pipeline_layout
from porebin_genome.io.coverage import validate_coverage_tsv
from porebin_genome.io.fasta import iter_fasta_names
from porebin_genome.io.runtime import ensure_dir, write_json


DEFAULT_FEATURE_MODE = "tnf_plus_cov"
DEFAULT_FEATURE_KNN_K = 15
DEFAULT_LAMBDA_CONTACT = 0.60
DEFAULT_HDBSCAN_SELECTION_METHOD = "leaf"


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


def run_coarse_discovery(
    *,
    contigs_fasta: Path,
    contacts_parquet: Path,
    coverage_tsv: Path,
    out_dir: Path,
    logger: Optional[object] = None,
    seed: int = 0,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    feature_knn_k: int = DEFAULT_FEATURE_KNN_K,
    lambda_contact: float = DEFAULT_LAMBDA_CONTACT,
    embedding_dim: Optional[int] = None,
    hdbscan_min_cluster_size: Optional[int] = None,
    hdbscan_min_samples: Optional[int] = None,
    hdbscan_selection_method: str = DEFAULT_HDBSCAN_SELECTION_METHOD,
    threads: int = 1,
) -> CoarseRunResult:
    """Run real coarse candidate genome-bin discovery without any component/noise postprocess."""
    _ = logger
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

    contact = build_contact_incidence_from_parquet(
        contacts_parquet,
        contig_name_to_idx,
        logger=logger,
    )
    feature_matrix = build_feature_matrix(
        contigs_fasta=contigs_fasta,
        coverage_tsv=coverage_tsv,
        contig_name_to_idx=contig_name_to_idx,
        feature_mode=feature_mode,
        logger=logger,
    )
    neighbors = build_feature_knn_edges(feature_matrix.X, feature_knn_k)
    feature = build_feature_incidence(neighbors)

    contact_theta = make_theta_operator(contact.H_csr, contact.W, contact.De, contact.Dv)
    feature_theta = make_theta_operator(feature.H_csr, feature.W, feature.De, feature.Dv)

    target_embedding_dim = auto_embedding_dim(contigs_total) if embedding_dim is None else int(embedding_dim)
    Z = spectral_embed_joint(
        contact_op=contact_theta.op,
        feature_op=feature_theta.op,
        lambda_contact=float(lambda_contact),
        d=target_embedding_dim,
        seed=int(seed),
    )
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
        lambda_contact=float(lambda_contact),
        contact_hyperedge_count=int(contact.contact_hyperedge_count),
        feature_knn_k=int(feature.feature_knn_k),
        dropped_singleton_contacts=int(contact.dropped_singleton_contacts),
        coverage_used=bool(feature_matrix.coverage_used),
        coverage_missing_count=int(feature_matrix.coverage_missing_count),
        hdbscan_meta=hdbscan_meta,
    )
    write_json(layout.coarse_run_json, run_record)

    return CoarseRunResult(
        bins_tsv=layout.coarse_bins_tsv,
        run_json=layout.coarse_run_json,
        implemented=True,
        contigs_total=contigs_total,
        n_bins=cluster_result.n_bins,
        n_contigs_clustered=cluster_result.n_contigs_clustered,
        n_contigs_unbinned=cluster_result.n_contigs_unbinned,
    )
