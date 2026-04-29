"""Hyperedge-aware contig-to-bin contact evidence for refinement actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from porebin_genome.refine.hyperedge import (
    HyperedgeRecord,
    HyperedgeStore,
    ReliabilityFn,
    collect_edge_ids_for_contig,
    default_edge_reliability,
)


@dataclass(frozen=True)
class BinContactEvidence:
    """Local hyperedge-aware support of one contig toward one bin."""

    bin_id: str
    support: float
    edge_count: int


def collect_contig_bin_evidence(
    store: HyperedgeStore,
    assignment: Mapping[str, str],
    contig_id: str,
    *,
    reliability_fn: ReliabilityFn | None = None,
) -> dict[str, BinContactEvidence]:
    """Aggregate hyperedge-aware support from one contig toward assigned bins."""
    reliability_fn = reliability_fn or default_edge_reliability

    target_contig = str(contig_id)
    support_by_bin: dict[str, float] = {}
    edges_by_bin: dict[str, set[int]] = {}

    for edge_id in collect_edge_ids_for_contig(store, target_contig):
        edge = store.edges_by_id.get(int(edge_id))
        if edge is None:
            continue
        reliability = float(reliability_fn(edge))
        if reliability <= 0.0:
            continue
        alpha_self = _edge_member_alpha(edge, target_contig)
        if alpha_self <= 0.0:
            continue

        other_mass_by_bin: dict[str, float] = {}
        for member_id, alpha in zip(edge.members, edge.alpha, strict=True):
            if member_id == target_contig:
                continue
            bin_id = str(assignment.get(member_id, "")).strip()
            if not bin_id:
                continue
            other_mass_by_bin[bin_id] = float(other_mass_by_bin.get(bin_id, 0.0) + float(alpha))

        for bin_id, other_mass in other_mass_by_bin.items():
            if other_mass <= 0.0:
                continue
            support = reliability * alpha_self * float(other_mass)
            support_by_bin[bin_id] = float(support_by_bin.get(bin_id, 0.0) + support)
            edges_by_bin.setdefault(bin_id, set()).add(int(edge.edge_id))

    return {
        bin_id: BinContactEvidence(
            bin_id=bin_id,
            support=float(support_by_bin[bin_id]),
            edge_count=len(edges_by_bin.get(bin_id, ())),
        )
        for bin_id in sorted(support_by_bin.keys(), key=str)
    }


def _edge_member_alpha(edge: HyperedgeRecord, contig_id: str) -> float:
    for member_id, alpha in zip(edge.members, edge.alpha, strict=True):
        if member_id == contig_id:
            return float(alpha)
    return 0.0
