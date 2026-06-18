from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from porebin_genome.coarse.hyperedge_weight import hypergraph_native_weight
from porebin_genome.evidence.canonical import CanonicalContact
from porebin_genome.refinement.contact_index import (
    ContactIndexError,
    build_contact_index,
    build_contact_index_from_contacts,
    encode_assignment,
)


def _contacts() -> list[CanonicalContact]:
    return [
        CanonicalContact(
            contact_id=10,
            contigs=["a", "b"],
            contig_weights=[0.6, 0.4],
            k_input=2,
            k_valid=2,
            weight=1.0,
        ),
        CanonicalContact(
            contact_id=11,
            contigs=["a", "c", "d"],
            contig_weights=[0.2, 0.5, 0.3],
            k_input=3,
            k_valid=3,
            weight=0.8,
        ),
        CanonicalContact(
            contact_id=12,
            contigs=["b", "c"],
            contig_weights=[0.5, 0.5],
            k_input=2,
            k_valid=2,
            weight=0.5,
        ),
        CanonicalContact(
            contact_id=13,
            contigs=["x"],
            contig_weights=[1.0],
            k_input=1,
            k_valid=1,
            weight=1.0,
        ),
    ]


def _brute_force_coherence(
    assignment: dict[str, str],
    *,
    eta: float,
) -> dict[str, tuple[float, float, float]]:
    numerator: dict[str, float] = {}
    denominator: dict[str, float] = {}
    for contact in _contacts():
        if contact.k_valid < 2:
            continue
        reliability = hypergraph_native_weight(
            read_weight=contact.weight,
            alpha_values=contact.contig_weights or (),
            eta=eta,
        )
        masses: dict[str, float] = {}
        for contig, alpha in zip(
            contact.contigs,
            contact.contig_weights or (),
            strict=True,
        ):
            bin_id = assignment.get(contig)
            if bin_id is None:
                continue
            masses[bin_id] = masses.get(bin_id, 0.0) + float(alpha)
        for bin_id, mass in masses.items():
            denominator[bin_id] = (
                denominator.get(bin_id, 0.0) + reliability * mass
            )
            numerator[bin_id] = (
                numerator.get(bin_id, 0.0)
                + reliability * mass * mass
            )
    return {
        bin_id: (
            numerator.get(bin_id, 0.0),
            denominator.get(bin_id, 0.0),
            numerator.get(bin_id, 0.0)
            / (denominator.get(bin_id, 0.0) + 1e-12),
        )
        for bin_id in set(numerator) | set(denominator)
    }


def _brute_force_contig_support(
    contig_id: str,
    assignment: dict[str, str],
    *,
    eta: float,
) -> dict[str, tuple[float, int]]:
    support: dict[str, float] = {}
    edge_count: dict[str, int] = {}
    for contact in _contacts():
        if contact.k_valid < 2 or contig_id not in contact.contigs:
            continue
        alpha = dict(
            zip(
                contact.contigs,
                contact.contig_weights or (),
                strict=True,
            )
        )
        reliability = hypergraph_native_weight(
            read_weight=contact.weight,
            alpha_values=contact.contig_weights or (),
            eta=eta,
        )
        other_mass: dict[str, float] = {}
        for other_contig, other_alpha in alpha.items():
            if other_contig == contig_id:
                continue
            bin_id = assignment.get(other_contig)
            if bin_id is None:
                continue
            other_mass[bin_id] = (
                other_mass.get(bin_id, 0.0) + float(other_alpha)
            )
        for bin_id, mass in other_mass.items():
            support[bin_id] = (
                support.get(bin_id, 0.0)
                + reliability * float(alpha[contig_id]) * mass
            )
            edge_count[bin_id] = edge_count.get(bin_id, 0) + 1
    return {
        bin_id: (value, edge_count[bin_id])
        for bin_id, value in support.items()
    }


def test_contact_index_csr_reconstructs_hyperedges_and_reverse_incidence() -> None:
    index = build_contact_index_from_contacts(
        _contacts(),
        contig_names=("a", "b", "c", "d", "x"),
        eta=0.5,
    )

    assert index.contig_names == ("a", "b", "c", "d", "x")
    assert index.edge_ids.tolist() == [10, 11, 12]
    assert index.edge_offsets.tolist() == [0, 2, 5, 7]
    assert index.n_edges == 3
    assert index.n_incidences == 7

    members, alpha = index.edge_members_view(1)
    assert [index.contig_names[int(idx)] for idx in members] == ["a", "c", "d"]
    assert alpha.tolist() == pytest.approx([0.2, 0.5, 0.3])

    assert index.incident_edges(index.contig_index("a")).tolist() == [0, 1]
    assert index.incident_edges(index.contig_index("b")).tolist() == [0, 2]
    assert index.incident_edges(index.contig_index("c")).tolist() == [1, 2]
    assert index.incident_edges(index.contig_index("d")).tolist() == [1]
    assert index.incident_edges(index.contig_index("x")).tolist() == []


def test_contact_index_caches_hypergraph_native_reliability_once() -> None:
    index = build_contact_index_from_contacts(_contacts(), eta=0.5)

    expected = [
        hypergraph_native_weight(
            read_weight=contact.weight,
            alpha_values=contact.contig_weights or (),
            eta=0.5,
        )
        for contact in _contacts()
        if contact.k_valid >= 2
    ]
    assert index.edge_reliability.tolist() == pytest.approx(expected)
    assert index.edge_effective_order.tolist() == pytest.approx(
        [1.0 / (0.6**2 + 0.4**2), 1.0 / (0.2**2 + 0.5**2 + 0.3**2), 2.0]
    )
    assert not index.edge_reliability.flags.writeable


def test_contact_index_coherence_matches_direct_hypergraph_formula() -> None:
    assignment = {"a": "0", "b": "0", "c": "1", "d": "1"}
    index = build_contact_index_from_contacts(
        _contacts(),
        contig_names=("a", "b", "c", "d", "x"),
        eta=0.5,
    )
    labels, bin_names = encode_assignment(index, assignment)
    cache = index.build_coherence_cache(labels, n_bins=len(bin_names))

    expected = _brute_force_coherence(assignment, eta=0.5)
    for bin_idx, bin_name in enumerate(bin_names):
        assert cache.numerator[bin_idx] == pytest.approx(
            expected[bin_name][0]
        )
        assert cache.denominator[bin_idx] == pytest.approx(
            expected[bin_name][1]
        )
        assert cache.coherence[bin_idx] == pytest.approx(
            expected[bin_name][2]
        )

    assert cache.edge_masses(0) == pytest.approx({0: 1.0})
    assert cache.edge_masses(1) == pytest.approx({0: 0.2, 1: 0.8})
    assert cache.edge_masses(2) == pytest.approx({0: 0.5, 1: 0.5})
    assert index.counters.full_edge_scans == 1
    assert index.counters.full_edges_visited == index.n_edges


def test_contact_index_contig_support_matches_direct_local_formula() -> None:
    assignment = {"a": "0", "b": "0", "c": "1", "d": "1"}
    index = build_contact_index_from_contacts(
        _contacts(),
        contig_names=("a", "b", "c", "d", "x"),
        eta=0.5,
    )
    labels, bin_names = encode_assignment(index, assignment)
    support = index.contig_bin_support(index.contig_index("a"), labels)
    expected = _brute_force_contig_support("a", assignment, eta=0.5)

    by_name = {bin_names[bin_idx]: value for bin_idx, value in support.items()}
    assert set(by_name) == set(expected)
    for bin_name, (expected_support, expected_edge_count) in expected.items():
        assert by_name[bin_name].support == pytest.approx(expected_support)
        assert by_name[bin_name].edge_count == expected_edge_count

    assert index.counters.full_edge_scans == 0
    assert index.counters.local_queries == 1
    assert index.counters.local_edges_visited == 2


def test_affected_edges_uses_only_reverse_incidence() -> None:
    index = build_contact_index_from_contacts(
        _contacts(),
        contig_names=("a", "b", "c", "d", "x"),
    )

    affected = index.affected_edges(
        [index.contig_index("a"), index.contig_index("c")]
    )

    assert affected.tolist() == [0, 1, 2]
    assert index.counters.full_edge_scans == 0
    assert index.counters.local_queries == 2
    assert index.counters.local_edges_visited == 4


def test_contact_index_rejects_contact_contigs_absent_from_fixed_fasta_order() -> None:
    with pytest.raises(ContactIndexError, match="absent from the FASTA"):
        build_contact_index_from_contacts(
            _contacts(),
            contig_names=("a", "b"),
        )


def test_contact_index_requires_normalized_positive_alpha() -> None:
    bad = CanonicalContact(
        contact_id=20,
        contigs=["a", "b"],
        contig_weights=[0.8, 0.8],
        k_input=2,
        k_valid=2,
        weight=1.0,
    )

    with pytest.raises(ContactIndexError, match="sum to one"):
        build_contact_index_from_contacts([bad])


def test_contact_index_values_are_finite() -> None:
    index = build_contact_index_from_contacts(_contacts())
    assert all(math.isfinite(float(value)) for value in index.edge_reliability)
    assert all(math.isfinite(float(value)) for value in index.edge_effective_order)
    assert np.allclose(
        [
            float(index.edge_alpha[index.edge_member_slice(edge_idx)].sum())
            for edge_idx in range(index.n_edges)
        ],
        1.0,
    )


def test_contact_index_streams_canonical_parquet(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "contacts.parquet"
    table = pa.table(
        {
            "contact_id": pa.array([30, 31], type=pa.int64()),
            "contigs": pa.array(
                [["a", "b"], ["b", "c"]],
                type=pa.list_(pa.string()),
            ),
            "contig_weights": pa.array(
                [[0.75, 0.25], [0.4, 0.6]],
                type=pa.list_(pa.float64()),
            ),
            "k": pa.array([2, 2], type=pa.int32()),
            "weight": pa.array([1.0, 0.5], type=pa.float64()),
        }
    )
    pq.write_table(table, path)

    index = build_contact_index(
        path,
        contig_names=("a", "b", "c", "unconnected"),
        parquet_batch_size=1,
    )

    assert index.edge_ids.tolist() == [30, 31]
    assert index.edge_offsets.tolist() == [0, 2, 4]
    assert index.incident_edges(index.contig_index("b")).tolist() == [0, 1]
    assert index.incident_edges(index.contig_index("unconnected")).tolist() == []
