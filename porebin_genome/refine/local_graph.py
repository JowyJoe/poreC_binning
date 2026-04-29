"""Mass-conserved local pairwise projections for HCR-style refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection, Mapping

from porebin_genome.refine.coherence import edge_bin_mass
from porebin_genome.refine.hyperedge import (
    HyperedgeRecord,
    HyperedgeStore,
    ReliabilityFn,
    default_edge_reliability,
)


@dataclass(frozen=True)
class LocalPairGraph:
    """Local pairwise graph derived from hyperedges without mass inflation."""

    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str, float], ...]
    total_projected_mass: float


def project_edge_to_local_pairs(
    edge: HyperedgeRecord,
    *,
    focus_contigs: Collection[str],
    reliability_fn: ReliabilityFn,
) -> tuple[tuple[str, str, float], ...]:
    """Project one hyperedge into local pairwise edges using mass conservation."""
    focus = {str(contig_id) for contig_id in focus_contigs}
    local_members = [
        (contig_id, float(alpha))
        for contig_id, alpha in zip(edge.members, edge.alpha, strict=True)
        if contig_id in focus
    ]
    if len(local_members) < 2:
        return ()

    denominator = 0.0
    for idx in range(len(edge.alpha)):
        for jdx in range(idx + 1, len(edge.alpha)):
            denominator += float(edge.alpha[idx]) * float(edge.alpha[jdx])
    if denominator <= 0.0:
        return ()

    reliability = float(reliability_fn(edge))
    if reliability <= 0.0:
        return ()

    out: list[tuple[str, str, float]] = []
    for idx in range(len(local_members)):
        left_id, left_alpha = local_members[idx]
        for jdx in range(idx + 1, len(local_members)):
            right_id, right_alpha = local_members[jdx]
            if left_id <= right_id:
                pair = (left_id, right_id)
            else:
                pair = (right_id, left_id)
            weight = reliability * float(left_alpha) * float(right_alpha) / denominator
            out.append((pair[0], pair[1], float(weight)))
    return tuple(out)


def project_local_pair_graph(
    store: HyperedgeStore,
    edge_ids: Collection[int],
    *,
    focus_contigs: Collection[str],
    reliability_fn: ReliabilityFn | None = None,
) -> LocalPairGraph:
    """Build a local pairwise graph from a selected hyperedge subset.

    Only pairwise mass captured inside ``focus_contigs`` is retained. When the
    focus set omits some hyperedge members, the projected local mass is a
    partial view of the full hyperedge reliability mass rather than a
    re-normalized copy of it.
    """
    reliability_fn = reliability_fn or default_edge_reliability

    pair_weights: dict[tuple[str, str], float] = {}
    nodes = {str(contig_id) for contig_id in focus_contigs}
    total_projected_mass = 0.0

    for edge_id in sorted(set(int(value) for value in edge_ids)):
        edge = store.edges_by_id.get(int(edge_id))
        if edge is None:
            continue
        for left_id, right_id, weight in project_edge_to_local_pairs(
            edge,
            focus_contigs=focus_contigs,
            reliability_fn=reliability_fn,
        ):
            pair_weights[(left_id, right_id)] = float(pair_weights.get((left_id, right_id), 0.0) + float(weight))
            total_projected_mass += float(weight)

    return LocalPairGraph(
        nodes=tuple(sorted(nodes)),
        edges=tuple(
            (left_id, right_id, float(weight))
            for (left_id, right_id), weight in sorted(pair_weights.items())
        ),
        total_projected_mass=float(total_projected_mass),
    )


def select_edges_for_bin(
    store: HyperedgeStore,
    assignment: Mapping[str, str],
    *,
    bin_id: str,
    min_mass_in_bin: float = 0.6,
    reliability_fn: ReliabilityFn | None = None,
) -> tuple[int, ...]:
    """Select hyperedges that place substantial reliable mass inside one bin."""
    reliability_fn = reliability_fn or default_edge_reliability
    target_bin = str(bin_id)
    out: list[int] = []
    for edge in store.edges:
        if float(reliability_fn(edge)) <= 0.0:
            continue
        if edge_bin_mass(edge, assignment, target_bin) >= float(min_mass_in_bin):
            out.append(int(edge.edge_id))
    return tuple(sorted(out))
