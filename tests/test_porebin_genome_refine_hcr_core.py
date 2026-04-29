from __future__ import annotations

from pathlib import Path

import pytest


def test_load_hyperedges_reads_canonical_alpha_and_indexes(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for HCR hyperedge tests")
    import pyarrow as pa
    import pyarrow.parquet as pq

    from porebin_genome.refine.hyperedge import collect_edge_ids_for_contig, load_hyperedges

    contacts = tmp_path / "contacts.parquet"
    pq.write_table(
        pa.table(
            {
                "contact_id": pa.array([0], type=pa.int64()),
                "contigs": pa.array([["a", "b", "a"]], type=pa.list_(pa.string())),
                "contig_weights": pa.array([[0.4, 0.2, 0.4]], type=pa.list_(pa.float64())),
                "k": pa.array([3], type=pa.int32()),
                "k_eff": pa.array([0.0], type=pa.float64()),
                "weight": pa.array([0.9], type=pa.float64()),
            }
        ),
        contacts,
    )

    store = load_hyperedges(contacts)

    assert len(store.edges) == 1
    edge = store.edges[0]
    assert edge.members == ("a", "b")
    assert edge.alpha == pytest.approx((0.8, 0.2))
    assert sum(edge.alpha) == pytest.approx(1.0)
    assert edge.k == 2
    assert edge.k_eff == pytest.approx(1.0 / (0.8 * 0.8 + 0.2 * 0.2))
    assert collect_edge_ids_for_contig(store, "a") == (0,)
    assert collect_edge_ids_for_contig(store, "b") == (0,)


def test_compute_bin_coherence_is_one_for_perfect_absorption() -> None:
    from porebin_genome.refine.coherence import compute_bin_coherence
    from porebin_genome.refine.hyperedge import HyperedgeRecord, HyperedgeStore

    edge = HyperedgeRecord(
        edge_id=0,
        members=("a", "b"),
        alpha=(0.5, 0.5),
        read_weight=1.0,
        k=2,
        k_eff=2.0,
    )
    store = HyperedgeStore(
        edges=(edge,),
        edges_by_id={0: edge},
        edge_ids_by_contig={"a": (0,), "b": (0,)},
    )

    stats = compute_bin_coherence(store, {"a": "0", "b": "0"})

    assert stats["0"].numerator == pytest.approx(1.0)
    assert stats["0"].denominator == pytest.approx(1.0)
    assert stats["0"].coherence == pytest.approx(1.0)


def test_compute_bin_coherence_drops_when_hyperedge_mass_leaks() -> None:
    from porebin_genome.refine.coherence import compute_bin_coherence
    from porebin_genome.refine.hyperedge import HyperedgeRecord, HyperedgeStore

    edge = HyperedgeRecord(
        edge_id=0,
        members=("a", "b"),
        alpha=(0.5, 0.5),
        read_weight=1.0,
        k=2,
        k_eff=2.0,
    )
    store = HyperedgeStore(
        edges=(edge,),
        edges_by_id={0: edge},
        edge_ids_by_contig={"a": (0,), "b": (0,)},
    )

    stats = compute_bin_coherence(store, {"a": "0", "b": "1"})

    assert stats["0"].coherence == pytest.approx(0.5)
    assert stats["1"].coherence == pytest.approx(0.5)


def test_mass_conserved_projection_preserves_total_edge_mass() -> None:
    from porebin_genome.refine.hyperedge import HyperedgeRecord
    from porebin_genome.refine.local_graph import project_edge_to_local_pairs

    edge = HyperedgeRecord(
        edge_id=0,
        members=("a", "b", "c"),
        alpha=(0.6, 0.3, 0.1),
        read_weight=0.8,
        k=3,
        k_eff=1.0 / (0.6 * 0.6 + 0.3 * 0.3 + 0.1 * 0.1),
    )

    pairs = project_edge_to_local_pairs(
        edge,
        focus_contigs={"a", "b", "c"},
        reliability_fn=lambda _: 0.8,
    )

    assert len(pairs) == 3
    assert sum(weight for _left, _right, weight in pairs) == pytest.approx(0.8)


def test_project_local_pair_graph_accumulates_pair_weights() -> None:
    from porebin_genome.refine.hyperedge import HyperedgeRecord, HyperedgeStore
    from porebin_genome.refine.local_graph import project_local_pair_graph

    edge0 = HyperedgeRecord(
        edge_id=0,
        members=("a", "b", "c"),
        alpha=(0.6, 0.3, 0.1),
        read_weight=0.8,
        k=3,
        k_eff=1.0 / (0.6 * 0.6 + 0.3 * 0.3 + 0.1 * 0.1),
    )
    edge1 = HyperedgeRecord(
        edge_id=1,
        members=("a", "b"),
        alpha=(0.5, 0.5),
        read_weight=0.5,
        k=2,
        k_eff=2.0,
    )
    store = HyperedgeStore(
        edges=(edge0, edge1),
        edges_by_id={0: edge0, 1: edge1},
        edge_ids_by_contig={"a": (0, 1), "b": (0, 1), "c": (0,)},
    )

    graph = project_local_pair_graph(
        store,
        edge_ids=(0, 1),
        focus_contigs={"a", "b"},
        reliability_fn=lambda edge: edge.read_weight,
    )

    assert graph.nodes == ("a", "b")
    assert len(graph.edges) == 1
    left_id, right_id, weight = graph.edges[0]
    assert (left_id, right_id) == ("a", "b")
    assert weight == pytest.approx((0.8 * 0.6 * 0.3 / (0.6 * 0.3 + 0.6 * 0.1 + 0.3 * 0.1)) + 0.5)
    assert graph.total_projected_mass == pytest.approx(weight)


def test_partial_local_projection_does_not_renormalize_missing_members() -> None:
    from porebin_genome.refine.hyperedge import HyperedgeRecord
    from porebin_genome.refine.local_graph import project_edge_to_local_pairs

    edge = HyperedgeRecord(
        edge_id=0,
        members=("a", "b", "c"),
        alpha=(0.6, 0.3, 0.1),
        read_weight=0.8,
        k=3,
        k_eff=1.0 / (0.6 * 0.6 + 0.3 * 0.3 + 0.1 * 0.1),
    )

    pairs = project_edge_to_local_pairs(
        edge,
        focus_contigs={"a", "b"},
        reliability_fn=lambda _: 0.8,
    )

    assert len(pairs) == 1
    assert pairs[0][0:2] == ("a", "b")
    assert pairs[0][2] == pytest.approx(0.8 * 0.6 * 0.3 / (0.6 * 0.3 + 0.6 * 0.1 + 0.3 * 0.1))


def test_delta_contact_coherence_reports_positive_improvement() -> None:
    from porebin_genome.refine.coherence import BinCoherenceStat, delta_contact_coherence

    before = {"0": BinCoherenceStat("0", 0.25, 0.5, 0.5)}
    after = {"0": BinCoherenceStat("0", 1.0, 1.0, 1.0)}

    assert delta_contact_coherence(before, after) == pytest.approx(0.5)
