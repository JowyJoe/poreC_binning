"""Suspect-bin detection and bin-level snapshot metrics for HCR refinement."""

from __future__ import annotations

from porebin_genome.refine.coherence import compute_bin_coherence
from porebin_genome.refine.hyperedge import HyperedgeStore, ReliabilityFn, default_edge_reliability
from porebin_genome.refine.local_graph import project_local_pair_graph, select_edges_for_bin
from porebin_genome.refine.markers import ContigScgProfile, build_bin_scg_states, summarize_bin_scg_state
from porebin_genome.refine.models import BinFeatureProfile, BinSnapshot, RefineState, SuspectBin


def build_bin_snapshots(
    *,
    state: RefineState,
    store: HyperedgeStore,
    profiles: dict[str, BinFeatureProfile],
    reliability_fn: ReliabilityFn | None = None,
    contig_scg_profiles: dict[str, ContigScgProfile] | None = None,
    min_mass_in_bin: float = 0.6,
) -> dict[str, BinSnapshot]:
    """Build bin-level metric snapshots for suspect detection and QC."""
    reliability_fn = reliability_fn or default_edge_reliability
    coherence_stats = compute_bin_coherence(
        store,
        state.current_assignment,
        reliability_fn=reliability_fn,
    )
    bin_scg_states = build_bin_scg_states(
        assignment=state.current_assignment,
        contig_profiles=(contig_scg_profiles or {}),
    )

    snapshots: dict[str, BinSnapshot] = {}
    for bin_id, members in state.bin_to_contigs().items():
        total_length = int(sum(state.contig_lengths.get(contig_id, 0) for contig_id in members))
        profile = profiles.get(bin_id)
        scg_status, scg_duplicate_marker_count = summarize_bin_scg_state(bin_scg_states.get(bin_id))
        edge_ids = select_edges_for_bin(
            store,
            state.current_assignment,
            bin_id=bin_id,
            min_mass_in_bin=min_mass_in_bin,
            reliability_fn=reliability_fn,
        )
        graph = project_local_pair_graph(
            store,
            edge_ids=edge_ids,
            focus_contigs=members,
            reliability_fn=reliability_fn,
        )
        contact_components = _count_components(members=members, edges=graph.edges)
        low_support_ratio = _low_support_ratio(members=members, edges=graph.edges)
        contact_coherence = float(coherence_stats.get(bin_id, _empty_snapshot_stat(bin_id)).coherence)

        suspect_reasons = _detect_suspect_reasons(
            members=members,
            contact_coherence=contact_coherence,
            contact_components=contact_components,
            low_support_ratio=low_support_ratio,
            feature_dispersion=(float(profile.feature_dispersion) if profile is not None else 0.0),
            coverage_dispersion=(profile.coverage_mad if profile is not None else None),
        )
        snapshots[bin_id] = BinSnapshot(
            bin_id=bin_id,
            members=tuple(sorted(members)),
            n_contigs=len(members),
            total_length=total_length,
            median_coverage=(
                float(profile.median_coverage)
                if profile is not None and profile.median_coverage is not None
                else None
            ),
            coverage_dispersion=(
                float(profile.coverage_mad)
                if profile is not None and profile.coverage_mad is not None
                else None
            ),
            feature_dispersion=(float(profile.feature_dispersion) if profile is not None else 0.0),
            contact_coherence=contact_coherence,
            scg_status=str(scg_status),
            scg_duplicate_marker_count=int(scg_duplicate_marker_count),
            contact_components=int(contact_components),
            low_support_ratio=float(low_support_ratio),
            suspect_flag=bool(suspect_reasons),
            suspect_reasons=tuple(suspect_reasons),
            refine_status=("suspect" if suspect_reasons else "stable"),
        )
    return snapshots


def detect_suspect_bins(*, snapshots: dict[str, BinSnapshot]) -> list[SuspectBin]:
    """Return suspect bins that qualify for local split attempts."""
    out: list[SuspectBin] = []
    for snapshot in snapshots.values():
        if snapshot.suspect_flag:
            out.append(SuspectBin(bin_id=snapshot.bin_id, reasons=snapshot.suspect_reasons))
    return sorted(out, key=lambda item: item.bin_id)


def _detect_suspect_reasons(
    *,
    members: list[str],
    contact_coherence: float,
    contact_components: int,
    low_support_ratio: float,
    feature_dispersion: float,
    coverage_dispersion: float | None,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if len(members) < 4:
        return ()
    if contact_components > 1:
        reasons.append("multiple_local_contact_components")
    if contact_coherence < 0.60:
        reasons.append("low_contact_coherence")
    if low_support_ratio >= 0.25:
        reasons.append("high_low_support_ratio")
    if feature_dispersion > 0.85:
        reasons.append("high_feature_dispersion")
    if coverage_dispersion is not None and coverage_dispersion > 5.0:
        reasons.append("high_coverage_dispersion")

    if "multiple_local_contact_components" in reasons:
        return tuple(reasons)
    if len(reasons) >= 2:
        return tuple(reasons)
    return ()


def _count_components(
    *,
    members: list[str],
    edges: tuple[tuple[str, str, float], ...],
) -> int:
    adjacency: dict[str, set[str]] = {str(member): set() for member in members}
    for left_id, right_id, weight in edges:
        if float(weight) <= 0.0:
            continue
        adjacency.setdefault(str(left_id), set()).add(str(right_id))
        adjacency.setdefault(str(right_id), set()).add(str(left_id))

    seen: set[str] = set()
    n_components = 0
    for node in sorted(adjacency.keys()):
        if node in seen:
            continue
        n_components += 1
        stack = [node]
        seen.add(node)
        while stack:
            current = stack.pop()
            for neighbor in sorted(adjacency.get(current, ())):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
    return int(n_components)


def _low_support_ratio(
    *,
    members: list[str],
    edges: tuple[tuple[str, str, float], ...],
) -> float:
    degree_by_contig = {str(member): 0.0 for member in members}
    for left_id, right_id, weight in edges:
        if float(weight) <= 0.0:
            continue
        degree_by_contig[str(left_id)] = float(degree_by_contig.get(str(left_id), 0.0) + float(weight))
        degree_by_contig[str(right_id)] = float(degree_by_contig.get(str(right_id), 0.0) + float(weight))
    if not degree_by_contig:
        return 0.0
    low_support = sum(1 for value in degree_by_contig.values() if float(value) <= 0.0)
    return float(low_support / len(degree_by_contig))


def _empty_snapshot_stat(bin_id: str):
    from porebin_genome.refine.coherence import BinCoherenceStat

    return BinCoherenceStat(bin_id=bin_id, numerator=0.0, denominator=0.0, coherence=0.0)
