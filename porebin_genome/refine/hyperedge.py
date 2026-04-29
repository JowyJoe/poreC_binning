"""Hyperedge-native contact evidence for HCR-style refinement."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Collection, Mapping

from porebin_genome.evidence.canonical import iter_canonical_contacts


@dataclass(frozen=True)
class HyperedgeRecord:
    """One canonical Pore-C read interpreted as a refinement hyperedge."""

    edge_id: int
    members: tuple[str, ...]
    alpha: tuple[float, ...]
    read_weight: float
    k: int
    k_eff: float


@dataclass(frozen=True)
class HyperedgeStore:
    """Materialized hyperedges plus reverse indices for local refinement queries."""

    edges: tuple[HyperedgeRecord, ...]
    edges_by_id: Mapping[int, HyperedgeRecord]
    edge_ids_by_contig: Mapping[str, tuple[int, ...]]


ReliabilityFn = Callable[[HyperedgeRecord], float]


def load_hyperedges(contacts_parquet: Path, *, min_k: int = 2) -> HyperedgeStore:
    """Load canonical contacts.parquet rows into refinement hyperedges."""
    edges: list[HyperedgeRecord] = []
    edge_ids_by_contig: dict[str, list[int]] = {}

    for edge_id, row in enumerate(
        iter_canonical_contacts(contacts_parquet, require_contig_weights=True)
    ):
        if row.k_valid < int(min_k):
            continue
        if row.contig_weights is None:
            continue
        if len(row.contigs) != len(row.contig_weights):
            continue

        members = tuple(str(name) for name in row.contigs)
        alpha = tuple(float(value) for value in row.contig_weights)
        if not members or not alpha:
            continue

        concentration = sum(value * value for value in alpha)
        if concentration <= 0.0:
            continue
        k_eff = 1.0 / concentration

        edge = HyperedgeRecord(
            edge_id=int(edge_id),
            members=members,
            alpha=alpha,
            read_weight=float(row.weight),
            k=int(row.k_valid),
            k_eff=float(k_eff),
        )
        edges.append(edge)
        for contig_id in members:
            edge_ids_by_contig.setdefault(contig_id, []).append(int(edge_id))

    edges_tuple = tuple(edges)
    return HyperedgeStore(
        edges=edges_tuple,
        edges_by_id={edge.edge_id: edge for edge in edges_tuple},
        edge_ids_by_contig={
            contig_id: tuple(sorted(edge_ids))
            for contig_id, edge_ids in edge_ids_by_contig.items()
        },
    )


def default_edge_reliability(edge: HyperedgeRecord) -> float:
    """MVP edge reliability: clip the upstream read-level weight into [0, 1]."""
    return _clip01(edge.read_weight)


def collect_edge_ids_for_contig(store: HyperedgeStore, contig_id: str) -> tuple[int, ...]:
    """Return all hyperedge ids containing one contig."""
    return tuple(store.edge_ids_by_contig.get(str(contig_id), ()))


def collect_edge_ids_for_contigs(
    store: HyperedgeStore,
    contig_ids: Collection[str],
) -> tuple[int, ...]:
    """Return the deduplicated hyperedge ids touching any contig in a collection."""
    out: set[int] = set()
    for contig_id in contig_ids:
        out.update(store.edge_ids_by_contig.get(str(contig_id), ()))
    return tuple(sorted(out))


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, float(value))))
