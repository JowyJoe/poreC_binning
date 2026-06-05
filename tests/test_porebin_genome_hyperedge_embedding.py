from __future__ import annotations

from pathlib import Path

import pytest

from tests.porebin_genome_testkit import read_json, read_tsv_rows, write_dual_community_fixture, write_reassign_refine_fixture


def _write_feature_guard_fixture(tmp_path: Path) -> dict[str, Path]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    contacts = tmp_path / "contacts.parquet"
    pq.write_table(
        pa.table(
            {
                "contact_id": pa.array([0, 1], type=pa.int64()),
                "contigs": pa.array([["a", "b"], ["a", "c"]], type=pa.list_(pa.string())),
                "contig_weights": pa.array([[0.5, 0.5], [0.5, 0.5]], type=pa.list_(pa.float64())),
                "k": pa.array([2, 2], type=pa.int32()),
                "k_eff": pa.array([2.0, 2.0], type=pa.float64()),
                "weight": pa.array([1.0, 1.0], type=pa.float64()),
            }
        ),
        contacts,
    )
    return {"contacts": contacts}


def _write_reassign_embedding(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "contig_id\tz0\tz1\tsupport_weight",
                "a\t1\t0\t10",
                "b\t1\t0\t10",
                "c\t0\t1\t10",
                "d\t0\t1\t10",
                "e\t1\t0\t10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_feature_reconstruction_loss_balances_tnf_and_coverage() -> None:
    torch = pytest.importorskip("torch", reason="torch required for HG-VAE loss unit test")

    from porebin_genome.coarse.hyperedge_embedding import _torch_feature_reconstruction_loss

    target = torch.zeros((2, 3), dtype=torch.float32)
    reconstruction = torch.tensor([[2.0, 0.0, 4.0], [0.0, 0.0, 4.0]], dtype=torch.float32)
    loss, tnf_loss, coverage_loss = _torch_feature_reconstruction_loss(
        reconstruction=reconstruction,
        target=target,
        coverage_feature_present=True,
    )

    assert float(tnf_loss) == pytest.approx(1.0)
    assert float(coverage_loss) == pytest.approx(16.0)
    assert float(loss) == pytest.approx(8.5)


def test_hypergraph_loss_full_batch_is_strength_weighted_average() -> None:
    torch = pytest.importorskip("torch", reason="torch required for HG-VAE loss unit test")

    import numpy as np

    from porebin_genome.coarse.hyperedge_embedding import _TrainingEdge, _torch_hypergraph_loss

    z = torch.tensor([[0.0], [2.0], [10.0], [12.0]], dtype=torch.float32)
    edges = [
        _TrainingEdge(
            contact_id=0,
            members=np.asarray([0, 1], dtype=np.int64),
            alpha=np.asarray([0.5, 0.5], dtype=np.float32),
            weight=3.0,
            base_weight=3.0,
            feature_dispersion=0.0,
            feature_compatibility=1.0,
        ),
        _TrainingEdge(
            contact_id=1,
            members=np.asarray([2, 3], dtype=np.int64),
            alpha=np.asarray([0.5, 0.5], dtype=np.float32),
            weight=1.0,
            base_weight=1.0,
            feature_dispersion=0.0,
            feature_compatibility=1.0,
        ),
    ]
    loss = _torch_hypergraph_loss(z=z, edges=edges, batch_size=0, rng=np.random.default_rng(1))

    assert float(loss) == pytest.approx(1.0)


def test_hgvae_embedding_training_outputs_vectors(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for hyperedge embedding test")
    pytest.importorskip("torch", reason="torch required for HG-VAE embedding test")

    import numpy as np

    from porebin_genome.coarse.features import build_feature_matrix
    from porebin_genome.coarse.hyperedge_embedding import train_hyperedge_embedding

    fixture = write_dual_community_fixture(tmp_path, community_size=3)
    names = list(fixture["all_names"])
    contig_to_idx = {name: idx for idx, name in enumerate(names)}
    feature_matrix = build_feature_matrix(
        contigs_fasta=fixture["contigs"],
        coverage_tsv=fixture["coverage"],
        contig_name_to_idx=contig_to_idx,
    )
    result = train_hyperedge_embedding(
        contacts_path=fixture["contacts"],
        contig_name_to_idx=contig_to_idx,
        idx_to_name=names,
        out_tsv=tmp_path / "embedding.tsv",
        meta_json=tmp_path / "embedding.json",
        embedding_dim=8,
        epochs=5,
        feature_matrix=feature_matrix.X,
        hyperedge_batch_size=0,
        seed=7,
    )

    rows = read_tsv_rows(result.embedding_tsv)
    meta = read_json(result.meta_json)
    vectors = np.asarray([[float(row[f"z{idx}"]) for idx in range(8)] for row in rows])

    assert len(rows) == len(names)
    assert result.n_training_edges > 0
    assert meta["method"] == "feature_guarded_feature_anchored_hypergraph_vae_embedding"
    assert meta["final_loss"] is not None
    assert np.linalg.norm(vectors, axis=1).min() > 0.0


def test_feature_guard_downweights_incompatible_contact(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for hyperedge embedding test")
    pytest.importorskip("torch", reason="torch required for HG-VAE embedding test")

    import numpy as np

    from porebin_genome.coarse.hyperedge_embedding import train_hyperedge_embedding

    fixture = _write_feature_guard_fixture(tmp_path)
    names = ["a", "b", "c"]
    feature_matrix = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [10.0, 0.0],
        ],
        dtype=np.float32,
    )
    result = train_hyperedge_embedding(
        contacts_path=fixture["contacts"],
        contig_name_to_idx={name: idx for idx, name in enumerate(names)},
        idx_to_name=names,
        out_tsv=tmp_path / "embedding.tsv",
        meta_json=tmp_path / "embedding.json",
        embedding_dim=4,
        epochs=1,
        feature_matrix=feature_matrix,
        feature_guard=True,
        seed=3,
    )

    rows = {row["contig_id"]: row for row in read_tsv_rows(result.embedding_tsv)}
    meta = read_json(result.meta_json)

    assert meta["method"] == "feature_guarded_feature_anchored_hypergraph_vae_embedding"
    assert meta["feature_guard_enabled"] is True
    assert meta["feature_compatibility_scale"] == pytest.approx(12.5)
    assert meta["mean_feature_compatibility"] == pytest.approx(0.75)
    assert float(rows["c"]["support_weight"]) < float(rows["b"]["support_weight"])


def test_hgvae_can_drive_coarse_clustering(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for HG-VAE coarse integration test")
    pytest.importorskip("hdbscan", reason="hdbscan required for HG-VAE coarse integration test")
    pytest.importorskip("torch", reason="torch required for HG-VAE coarse integration test")

    from porebin_genome.coarse.orchestrate import run_coarse_discovery

    fixture = write_dual_community_fixture(tmp_path, community_size=4)
    result = run_coarse_discovery(
        contigs_fasta=fixture["contigs"],
        contacts_parquet=fixture["contacts"],
        coverage_tsv=fixture["coverage"],
        out_dir=tmp_path / "out",
        coarse_method="hgvae",
        hyperedge_embedding_dim=8,
        hyperedge_embedding_epochs=5,
        hyperedge_vae_batch_size=0,
        hdbscan_min_cluster_size=2,
        seed=11,
    )
    coarse_run = read_json(result.run_json)

    assert result.hyperedge_embedding_tsv is not None
    assert result.hyperedge_embedding_tsv.exists()
    assert result.n_bins >= 1
    assert coarse_run["coarse_method"] == "hgvae"
    assert coarse_run["embedding_source"] == "hgvae"
    assert coarse_run["hyperedge_embedding_enabled"] is True
    assert coarse_run["lambda_contact"] is None


def test_embedding_report_scores_without_changing_reassign(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for embedding scorer integration test")

    from porebin_genome.refine.orchestrate import run_refinement

    fixture = write_reassign_refine_fixture(tmp_path)
    embedding_tsv = _write_reassign_embedding(tmp_path / "embedding.tsv")
    result = run_refinement(
        contigs_fasta=fixture["contigs"],
        coarse_bins_tsv=fixture["coarse_bins"],
        contacts_parquet=fixture["contacts"],
        coverage_tsv=fixture["coverage"],
        enable_scg=False,
        embedding_scorer_mode="report",
        embedding_tsv=embedding_tsv,
        out_dir=tmp_path / "out",
    )

    assignment = {row["contig_id"]: row["bin_id"] for row in read_tsv_rows(result.bins_refined_tsv)}
    scores = read_tsv_rows(result.refine_embedding_scores_tsv)
    reassign_scores = [row for row in scores if row["action_type"] == "reassign" and row["contig_id"] == "e"]

    assert assignment["e"] == assignment["c"]
    assert reassign_scores
    assert reassign_scores[0]["embedding_decision"] == "would_veto"
    assert reassign_scores[0]["final_accepted"] == "1"


def test_embedding_veto_blocks_low_margin_reassign(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for embedding scorer integration test")

    from porebin_genome.refine.orchestrate import run_refinement

    fixture = write_reassign_refine_fixture(tmp_path)
    embedding_tsv = _write_reassign_embedding(tmp_path / "embedding.tsv")
    result = run_refinement(
        contigs_fasta=fixture["contigs"],
        coarse_bins_tsv=fixture["coarse_bins"],
        contacts_parquet=fixture["contacts"],
        coverage_tsv=fixture["coverage"],
        enable_scg=False,
        embedding_scorer_mode="veto",
        embedding_tsv=embedding_tsv,
        out_dir=tmp_path / "out",
    )

    assignment = {row["contig_id"]: row["bin_id"] for row in read_tsv_rows(result.bins_refined_tsv)}
    actions = read_tsv_rows(result.refine_actions_tsv)
    scores = read_tsv_rows(result.refine_embedding_scores_tsv)
    meta = read_json(result.refine_meta_json)

    assert assignment["e"] == "0"
    assert any(row["reason"] == "embedding_veto_low_margin" for row in actions if row["action_type"] == "reassign")
    assert any(row["embedding_decision"] == "veto" for row in scores if row["action_type"] == "reassign")
    assert meta["n_embedding_veto_actions"] >= 1
