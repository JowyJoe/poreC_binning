from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_joint_operator_shapes_and_eigsh(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for contacts.parquet")
    pytest.importorskip("scipy", reason="spectral optional deps not installed")
    pytest.importorskip("sklearn", reason="spectral v2 features require scikit-learn")

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from porebin.hypergraph_joint_spectral import (
        build_contact_incidence_from_parquet,
        build_feature_incidence,
        build_feature_knn_edges,
        load_or_build_tnf136_features,
        load_contig_index,
        load_coverage_feature_optional,
        make_theta_operator,
        spectral_embed_joint,
        zscore_features,
    )

    graph_dir = tmp_path / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)

    contigs = [f"c{i}" for i in range(10)]
    (graph_dir / "contigs.tsv").write_text(
        "".join([f"{i}\t{name}\n" for i, name in enumerate(contigs)]),
        encoding="utf-8",
    )

    contigs_fasta = tmp_path / "contigs.fasta"
    contigs_fasta.write_text(
        "".join([f">c{i}\n{('ACGT' * 200)}\n" for i in range(10)]),
        encoding="utf-8",
    )

    # contacts.parquet: include one singleton contact that should be dropped by incidence builder.
    rows_contigs = [
        ["c0", "c1", "c2"],
        ["c2", "c3", "c4"],
        ["c5", "c6"],
        ["c7", "c8", "c9"],
        ["c0"],  # dropped (k<2)
    ]
    rows_p = [
        [0.4, 0.3, 0.3],
        [1 / 3, 1 / 3, 1 / 3],
        [0.5, 0.5],
        [0.2, 0.3, 0.5],
        [1.0],
    ]
    rows_k = [len(x) for x in rows_contigs]
    rows_q = [1.0, 0.8, 0.9, 1.0, 1.0]

    contacts_parquet = tmp_path / "contacts.parquet"
    pq.write_table(
        pa.Table.from_pydict(
            {"contigs": rows_contigs, "contig_weights": rows_p, "k": rows_k, "weight": rows_q}
        ),
        contacts_parquet,
    )

    (graph_dir / "graph_meta.json").write_text(
        json.dumps(
            {"input_contigs_fasta": str(contigs_fasta), "input_contacts_parquet": str(contacts_parquet)},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    contig_name_to_idx, idx_to_name = load_contig_index(graph_dir)
    assert len(idx_to_name) == 10

    contact = build_contact_incidence_from_parquet(contacts_parquet, contig_name_to_idx)
    X_comp = load_or_build_tnf136_features(
        graph_dir=graph_dir, contigs_fasta=contigs_fasta, contig_name_to_idx=contig_name_to_idx
    )
    x_cov, _missing, _used = load_coverage_feature_optional(None, contig_name_to_idx)
    X = np.concatenate([X_comp, x_cov.reshape(-1, 1)], axis=1)
    X = zscore_features(X)

    neighbors = build_feature_knn_edges(X, knn_k=15)
    feature = build_feature_incidence(neighbors, knn_k=15)

    contact_theta = make_theta_operator(contact.H_csr, contact.W, contact.De, contact.Dv)
    feature_theta = make_theta_operator(feature.H_csr, feature.W, feature.De, feature.Dv)

    x = np.random.default_rng(0).standard_normal(len(idx_to_name))
    assert (contact_theta.op @ x).shape == (10,)
    assert (feature_theta.op @ x).shape == (10,)

    Z = spectral_embed_joint(
        contact_op=contact_theta.op,
        feature_op=feature_theta.op,
        lambda_contact=0.7,
        d=32,
        seed=0,
    )
    assert Z.shape[0] == 10
    # With V=10: k=min(d+1,V-1)=9, drop 1 => 8 dims.
    assert Z.shape[1] == 8
