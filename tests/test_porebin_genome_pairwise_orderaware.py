from __future__ import annotations

from pathlib import Path

import pytest

from tests.porebin_genome_testkit import read_json, read_tsv_rows, write_dual_community_fixture


def _write_single_hyperedge_fixture(tmp_path: Path) -> dict[str, Path]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    contigs = tmp_path / "contigs.fasta"
    coverage = tmp_path / "coverage.tsv"
    contacts = tmp_path / "contacts.parquet"
    names = ["a", "b", "c", "d"]
    contigs.write_text(
        "\n".join(line for name in names for line in (f">{name}", "ACGT" * 50)) + "\n",
        encoding="utf-8",
    )
    coverage.write_text(
        "contig_name\tcoverage\n" + "\n".join(f"{name}\t10" for name in names) + "\n",
        encoding="utf-8",
    )
    pq.write_table(
        pa.table(
            {
                "contact_id": pa.array([42], type=pa.int64()),
                "contigs": pa.array([names], type=pa.list_(pa.string())),
                "contig_weights": pa.array([[0.25, 0.25, 0.25, 0.25]], type=pa.list_(pa.float64())),
                "k": pa.array([4], type=pa.int32()),
                "k_eff": pa.array([4.0], type=pa.float64()),
                "weight": pa.array([2.0], type=pa.float64()),
            }
        ),
        contacts,
    )
    return {"contigs": contigs, "coverage": coverage, "contacts": contacts}


def test_canonical_contacts_preserve_contact_id(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for contact_id test")

    from porebin_genome.evidence.canonical import iter_canonical_contacts

    fixture = _write_single_hyperedge_fixture(tmp_path)
    rows = list(iter_canonical_contacts(fixture["contacts"], require_contig_weights=True))

    assert len(rows) == 1
    assert rows[0].contact_id == 42


def test_clique_pairwise_expansion_records_inflation(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for pairwise test")

    import pyarrow.parquet as pq

    from porebin_genome.coarse.pairwise import build_clique_pairwise_contacts

    fixture = _write_single_hyperedge_fixture(tmp_path)
    names = ["a", "b", "c", "d"]
    result = build_clique_pairwise_contacts(
        contacts_path=fixture["contacts"],
        contig_name_to_idx={name: idx for idx, name in enumerate(names)},
        out_parquet=tmp_path / "pairwise.parquet",
        meta_json=tmp_path / "pairwise.json",
    )
    data = pq.read_table(result.contacts_parquet).to_pydict()
    meta = read_json(result.meta_json)

    assert result.n_contacts_used == 1
    assert result.n_pair_edges == 6
    assert result.mean_clique_inflation == 6
    assert meta["max_clique_inflation"] == 6
    assert set(data["n_contacts"]) == {1}
    assert set(data["q_sum"]) == {2.0}


def test_hypergraph_native_weight_uses_effective_order_not_pairwise_graph() -> None:
    from porebin_genome.coarse.hyperedge_weight import hypergraph_native_weight

    balanced_four_way = hypergraph_native_weight(
        read_weight=1.0,
        alpha_values=[0.25, 0.25, 0.25, 0.25],
        eta=0.5,
    )
    concentrated_four_member = hypergraph_native_weight(
        read_weight=1.0,
        alpha_values=[0.85, 0.05, 0.05, 0.05],
        eta=0.5,
    )

    assert balanced_four_way == pytest.approx(1.0 / (3.0 ** 0.5))
    assert concentrated_four_member == pytest.approx(1.0)


def test_pairwise_normalization_keeps_pairwise_baseline_separate(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for normalization test")

    import pyarrow.parquet as pq

    from porebin_genome.coarse.pairwise import build_clique_pairwise_contacts
    from porebin_genome.coarse.pairwise_normalize import normalize_pairwise_contacts
    from porebin_genome.io.coverage import read_coverage_tsv
    from porebin_genome.io.fasta import read_contig_lengths

    fixture = _write_single_hyperedge_fixture(tmp_path)
    names = ["a", "b", "c", "d"]
    pairwise = build_clique_pairwise_contacts(
        contacts_path=fixture["contacts"],
        contig_name_to_idx={name: idx for idx, name in enumerate(names)},
        out_parquet=tmp_path / "pairwise.parquet",
        meta_json=tmp_path / "pairwise.json",
    )
    normalized = normalize_pairwise_contacts(
        pairwise_contacts_path=pairwise.contacts_parquet,
        contig_lengths=read_contig_lengths(fixture["contigs"]),
        coverage_by_contig=read_coverage_tsv(fixture["coverage"]),
        out_parquet=tmp_path / "normalized.parquet",
        meta_json=tmp_path / "normalized.json",
    )
    normalized_data = pq.read_table(normalized.normalized_contacts_parquet).to_pydict()

    assert normalized.n_pair_edges_in == 6
    assert normalized.total_observed_contacts == 6
    assert normalized.ice_converged is True
    assert all(value == pytest.approx(0.01) for value in normalized_data["length_corrected"])
    assert all(value == pytest.approx(1.0 / 3.0) for value in normalized_data["balanced_weight"])
    assert all(value == pytest.approx(1.0) for value in normalized_data["reliability"])


def test_hypergraph_native_coarse_does_not_write_pairwise_feedback_outputs(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for hypergraph-native coarse test")
    pytest.importorskip("hdbscan", reason="hdbscan required for hypergraph-native coarse test")

    from porebin_genome.coarse.orchestrate import run_coarse_discovery

    fixture = write_dual_community_fixture(tmp_path, community_size=4)
    out_dir = tmp_path / "out"
    result = run_coarse_discovery(
        contigs_fasta=fixture["contigs"],
        contacts_parquet=fixture["contacts"],
        coverage_tsv=fixture["coverage"],
        out_dir=out_dir,
        feature_knn_k=2,
        contact_weight_mode="hypergraph-native",
    )
    run_record = read_json(result.run_json)

    assert run_record["contact_weight_mode"] == "hypergraph_native"
    assert "hyperedge_reliability_parquet" not in run_record["outputs"]
    assert "pairwise_normalized_contacts_parquet" not in run_record["outputs"]
    assert not (out_dir / "coarse" / "hyperedge_reliability.parquet").exists()
    assert not (out_dir / "coarse" / "pairwise_normalized_contacts.parquet").exists()


def test_pairwise_leiden_baseline_outputs_sweep(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for pairwise baseline test")
    pytest.importorskip("igraph", reason="igraph required for pairwise baseline test")
    pytest.importorskip("leidenalg", reason="leidenalg required for pairwise baseline test")

    from porebin_genome.coarse.orchestrate import run_coarse_discovery

    fixture = write_dual_community_fixture(tmp_path, community_size=4)
    out_dir = tmp_path / "out"
    result = run_coarse_discovery(
        contigs_fasta=fixture["contigs"],
        contacts_parquet=fixture["contacts"],
        coverage_tsv=fixture["coverage"],
        out_dir=out_dir,
        feature_knn_k=2,
        run_pairwise_baseline=True,
    )
    run_record = read_json(result.run_json)
    sweep_rows = read_tsv_rows(out_dir / "coarse" / "pairwise_leiden_sweep.tsv")

    assert run_record["pairwise_baseline_enabled"] is True
    assert (out_dir / "coarse" / "bins.pairwise_leiden.tsv").exists()
    assert sweep_rows
    assert any(row["selected"] == "True" for row in sweep_rows)
