from __future__ import annotations

from pathlib import Path

import pytest

from tests.porebin_genome_testkit import write_dual_community_fixture


def test_contact_incidence_shape_and_indexing(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for coarse contact test")

    from porebin_genome.coarse.contact import build_contact_incidence_from_parquet

    fixture = write_dual_community_fixture(tmp_path, community_size=2)
    names = fixture["all_names"]
    contig_name_to_idx = {name: idx for idx, name in enumerate(names)}
    contact = build_contact_incidence_from_parquet(fixture["contacts"], contig_name_to_idx)

    assert contact.H_csr.shape == (4, 2)
    assert contact.contact_hyperedge_count == 2
    assert contact.dropped_singleton_contacts == 0
    assert contact.W.shape == (2,)
    assert contact.De.shape == (2,)
    assert contact.Dv.shape == (4,)


def test_feature_matrix_operator_embedding_and_clustering_shapes(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for coarse feature test")
    pytest.importorskip("hdbscan", reason="hdbscan required for coarse clustering test")

    from porebin_genome.coarse.cluster import hdbscan_cluster
    from porebin_genome.coarse.contact import build_contact_incidence_from_parquet
    from porebin_genome.coarse.embed import auto_embedding_dim, spectral_embed_joint
    from porebin_genome.coarse.features import build_feature_matrix
    from porebin_genome.coarse.operator import build_feature_incidence, build_feature_knn_edges, make_theta_operator

    fixture = write_dual_community_fixture(tmp_path, community_size=3)
    names = fixture["all_names"]
    contig_name_to_idx = {name: idx for idx, name in enumerate(names)}

    contact = build_contact_incidence_from_parquet(fixture["contacts"], contig_name_to_idx)
    feature_matrix = build_feature_matrix(
        contigs_fasta=fixture["contigs"],
        coverage_tsv=fixture["coverage"],
        contig_name_to_idx=contig_name_to_idx,
    )
    neighbors = build_feature_knn_edges(feature_matrix.X, knn_k=4)
    feature = build_feature_incidence(neighbors)
    contact_theta = make_theta_operator(contact.H_csr, contact.W, contact.De, contact.Dv)
    feature_theta = make_theta_operator(feature.H_csr, feature.W, feature.De, feature.Dv)
    embedding = spectral_embed_joint(
        contact_op=contact_theta.op,
        feature_op=feature_theta.op,
        lambda_contact=0.6,
        d=auto_embedding_dim(len(names)),
        seed=0,
    )
    labels, meta = hdbscan_cluster(
        embedding,
        min_cluster_size=2,
        min_samples=None,
        selection_method="leaf",
        threads=1,
    )

    assert feature_matrix.X.shape[0] == len(names)
    assert feature_matrix.X.shape[1] == 137
    assert neighbors.shape == (len(names), min(4, len(names) - 1))
    assert feature.H_csr.shape == (len(names), len(names))
    assert contact_theta.op.shape == (len(names), len(names))
    assert feature_theta.op.shape == (len(names), len(names))
    assert embedding.shape[0] == len(names)
    assert embedding.shape[1] >= 1
    assert labels.shape == (len(names),)
    assert meta["impl"] == "hdbscan"
