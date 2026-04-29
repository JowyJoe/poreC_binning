"""Hyperedge-native contact coherence utilities for HCR-style refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Collection

from porebin_genome.refine.hyperedge import (
    HyperedgeRecord,
    HyperedgeStore,
    ReliabilityFn,
    default_edge_reliability,
)


@dataclass(frozen=True)
class BinCoherenceStat:
    """Numerator, denominator, and final coherence for one bin."""

    bin_id: str
    numerator: float
    denominator: float
    coherence: float


def edge_bin_mass(
    edge: HyperedgeRecord,
    assignment: Mapping[str, str],
    bin_id: str,
) -> float:
    """Return how much hyperedge mass falls into one assigned bin."""
    target_bin = str(bin_id)
    mass = 0.0
    for contig_id, alpha in zip(edge.members, edge.alpha, strict=True):
        if assignment.get(contig_id, "") == target_bin:
            mass += float(alpha)
    return float(mass)


def bins_touched_by_edge(
    edge: HyperedgeRecord,
    assignment: Mapping[str, str],
) -> dict[str, float]:
    """Return assigned bins touched by one hyperedge plus their absorbed mass."""
    out: dict[str, float] = {}
    for contig_id, alpha in zip(edge.members, edge.alpha, strict=True):
        bin_id = str(assignment.get(contig_id, "")).strip()
        if not bin_id:
            continue
        out[bin_id] = float(out.get(bin_id, 0.0) + float(alpha))
    return out


def compute_bin_coherence(
    store: HyperedgeStore,
    assignment: Mapping[str, str],
    *,
    reliability_fn: ReliabilityFn | None = None,
    bins: Collection[str] | None = None,
    eps: float = 1e-12,
) -> dict[str, BinCoherenceStat]:
    """Compute C(b) for bins in the current assignment."""
    reliability_fn = reliability_fn or default_edge_reliability

    tracked_bins: set[str] = {
        str(bin_id).strip()
        for bin_id in assignment.values()
        if str(bin_id).strip()
    }
    if bins is not None:
        tracked_bins.update(str(bin_id).strip() for bin_id in bins if str(bin_id).strip())
    tracked_bins.discard("")

    numerator: dict[str, float] = {bin_id: 0.0 for bin_id in tracked_bins}
    denominator: dict[str, float] = {bin_id: 0.0 for bin_id in tracked_bins}

    for edge in store.edges:
        reliability = float(reliability_fn(edge))
        if reliability <= 0.0:
            continue
        for bin_id, mass in bins_touched_by_edge(edge, assignment).items():
            if bin_id not in tracked_bins:
                continue
            numerator[bin_id] += reliability * float(mass) * float(mass)
            denominator[bin_id] += reliability * float(mass)

    return {
        bin_id: BinCoherenceStat(
            bin_id=bin_id,
            numerator=float(numerator.get(bin_id, 0.0)),
            denominator=float(denominator.get(bin_id, 0.0)),
            coherence=float(numerator.get(bin_id, 0.0) / (denominator.get(bin_id, 0.0) + float(eps))),
        )
        for bin_id in sorted(tracked_bins, key=str)
    }


def delta_contact_coherence(
    before: Mapping[str, BinCoherenceStat],
    after: Mapping[str, BinCoherenceStat],
    *,
    bin_lengths: Mapping[str, int] | None = None,
) -> float:
    """Return the mean coherence improvement across affected bins."""
    affected_bins = sorted(set(before.keys()) | set(after.keys()), key=str)
    if not affected_bins:
        return 0.0

    total_weight = 0.0
    total_delta = 0.0
    for bin_id in affected_bins:
        weight = 1.0
        if bin_lengths is not None:
            weight = float(max(1, int(bin_lengths.get(bin_id, 1))))
        before_value = float(before.get(bin_id, BinCoherenceStat(bin_id, 0.0, 0.0, 0.0)).coherence)
        after_value = float(after.get(bin_id, BinCoherenceStat(bin_id, 0.0, 0.0, 0.0)).coherence)
        total_weight += weight
        total_delta += weight * (after_value - before_value)
    return float(total_delta / total_weight) if total_weight > 0.0 else 0.0
