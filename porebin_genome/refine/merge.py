"""Conservative merge refinement for coarse over-splitting."""

from __future__ import annotations

from porebin_genome.refine.coherence import compute_bin_coherence
from porebin_genome.refine.features import mad, median
from porebin_genome.refine.hyperedge import HyperedgeStore, ReliabilityFn, default_edge_reliability
from porebin_genome.refine.markers import ContigScgProfile, evaluate_merge_scg_transition
from porebin_genome.refine.models import BinFeatureProfile, MergeCandidate, MergeDecision, RefineState


MIN_MERGE_SUPPORT_EDGES = 2
FEATURE_WORSENING_TOLERANCE = 0.05
COVERAGE_WORSENING_TOLERANCE = 0.50


def generate_merge_candidates(
    *,
    state: RefineState,
    store: HyperedgeStore,
    profiles: dict[str, BinFeatureProfile],
    reliability_fn: ReliabilityFn | None = None,
) -> list[MergeCandidate]:
    """Generate conservative bin-merge candidates from cross-bin hyperedge evidence."""
    reliability_fn = reliability_fn or default_edge_reliability

    pair_support: dict[tuple[str, str], float] = {}
    pair_edges: dict[tuple[str, str], set[int]] = {}
    for edge in store.edges:
        reliability = float(reliability_fn(edge))
        if reliability <= 0.0:
            continue
        mass_by_bin: dict[str, float] = {}
        for contig_id, alpha in zip(edge.members, edge.alpha, strict=True):
            bin_id = str(state.current_assignment.get(contig_id, "")).strip()
            if not bin_id:
                continue
            mass_by_bin[bin_id] = float(mass_by_bin.get(bin_id, 0.0) + float(alpha))
        bins = sorted(mass_by_bin.keys(), key=str)
        for idx in range(len(bins)):
            left_bin = bins[idx]
            for jdx in range(idx + 1, len(bins)):
                right_bin = bins[jdx]
                support = reliability * float(mass_by_bin[left_bin]) * float(mass_by_bin[right_bin])
                if support <= 0.0:
                    continue
                pair = (left_bin, right_bin)
                pair_support[pair] = float(pair_support.get(pair, 0.0) + support)
                pair_edges.setdefault(pair, set()).add(int(edge.edge_id))

    candidates: list[MergeCandidate] = []
    bin_members = state.bin_to_contigs()
    for (left_bin, right_bin), cross_support in sorted(
        pair_support.items(),
        key=lambda item: (-float(item[1]), tuple(item[0])),
    ):
        support_edges = len(pair_edges.get((left_bin, right_bin), ()))
        if support_edges < MIN_MERGE_SUPPORT_EDGES:
            continue
        if left_bin not in bin_members or right_bin not in bin_members:
            continue

        source_bin, target_bin = _choose_merge_direction(state=state, left_bin=left_bin, right_bin=right_bin)
        feature_compatible = _profiles_feature_compatible(
            profiles.get(source_bin),
            profiles.get(target_bin),
        )
        coverage_compatible = _profiles_coverage_compatible(
            profiles.get(source_bin),
            profiles.get(target_bin),
        )
        if not feature_compatible or not coverage_compatible:
            continue

        candidates.append(
            MergeCandidate(
                source_bin=source_bin,
                target_bin=target_bin,
                cross_support=float(cross_support),
                support_edges=int(support_edges),
                feature_compatible=bool(feature_compatible),
                coverage_compatible=bool(coverage_compatible),
            )
        )
    return candidates


def evaluate_merge_candidate(
    *,
    candidate: MergeCandidate,
    state: RefineState,
    store: HyperedgeStore,
    profiles: dict[str, BinFeatureProfile],
    feature_matrix: "object",
    contig_name_to_idx: dict[str, int],
    reliability_fn: ReliabilityFn | None = None,
    contig_scg_profiles: dict[str, ContigScgProfile] | None = None,
) -> MergeDecision:
    """Accept a merge only if it improves contact coherence and preserves consistency."""
    reliability_fn = reliability_fn or default_edge_reliability

    if not candidate.feature_compatible:
        return MergeDecision(
            candidate=candidate,
            accepted=False,
            reason="merge_feature_incompatible",
            confidence=0.0,
            note="candidate bins fail feature compatibility",
        )
    if not candidate.coverage_compatible:
        return MergeDecision(
            candidate=candidate,
            accepted=False,
            reason="merge_coverage_incompatible",
            confidence=0.0,
            note="candidate bins fail coverage compatibility",
        )

    source_bin = candidate.source_bin
    target_bin = candidate.target_bin
    source_members = state.bin_to_contigs().get(source_bin, [])
    target_members = state.bin_to_contigs().get(target_bin, [])
    if not source_members or not target_members:
        return MergeDecision(
            candidate=candidate,
            accepted=False,
            reason="merge_bin_members_unavailable",
            confidence=0.0,
            note="one merge candidate bin has no current members",
        )

    before_stats = compute_bin_coherence(
        store,
        state.current_assignment,
        reliability_fn=reliability_fn,
        bins=(source_bin, target_bin),
    )
    before_weighted = _weighted_before_coherence(
        state=state,
        before_stats=before_stats,
        source_bin=source_bin,
        target_bin=target_bin,
    )
    after_assignment = dict(state.current_assignment)
    for contig_id in target_members:
        after_assignment[contig_id] = source_bin
    after_stats = compute_bin_coherence(
        store,
        after_assignment,
        reliability_fn=reliability_fn,
        bins=(source_bin,),
    )
    after_coherence = float(after_stats.get(source_bin, _empty_stat(source_bin)).coherence)
    delta_c = float(after_coherence - before_weighted)
    if delta_c <= 0.0:
        return MergeDecision(
            candidate=candidate,
            accepted=False,
            reason="merge_does_not_improve_contact_coherence",
            confidence=0.0,
            note=(
                f"pre_contact={before_weighted:.3f};"
                f"post_contact={after_coherence:.3f};"
                f"delta_contact={delta_c:.3f}"
            ),
        )

    merged_members = tuple(sorted(source_members + target_members))
    merged_feature = _group_feature_dispersion(
        group=merged_members,
        feature_matrix=feature_matrix,
        contig_name_to_idx=contig_name_to_idx,
    )
    pre_feature = max(
        float(profiles.get(source_bin).feature_dispersion if profiles.get(source_bin) is not None else 0.0),
        float(profiles.get(target_bin).feature_dispersion if profiles.get(target_bin) is not None else 0.0),
    )
    if merged_feature > pre_feature + FEATURE_WORSENING_TOLERANCE:
        return MergeDecision(
            candidate=candidate,
            accepted=False,
            reason="merge_feature_dispersion_worsened",
            confidence=0.0,
            note=f"pre_feature={pre_feature:.3f};post_feature={merged_feature:.3f}",
        )

    merged_coverage = _group_coverage_dispersion(group=merged_members, coverage_by_contig=state.coverage_by_contig)
    pre_coverage = max(
        float(profiles.get(source_bin).coverage_mad if profiles.get(source_bin) and profiles.get(source_bin).coverage_mad is not None else 0.0),
        float(profiles.get(target_bin).coverage_mad if profiles.get(target_bin) and profiles.get(target_bin).coverage_mad is not None else 0.0),
    )
    if merged_coverage is not None and merged_coverage > pre_coverage + COVERAGE_WORSENING_TOLERANCE:
        return MergeDecision(
            candidate=candidate,
            accepted=False,
            reason="merge_coverage_dispersion_worsened",
            confidence=0.0,
            note=f"pre_coverage={pre_coverage:.3f};post_coverage={merged_coverage:.3f}",
        )

    scg_note = ""
    if contig_scg_profiles is not None:
        scg_decision = evaluate_merge_scg_transition(
            source_bin=source_bin,
            target_bin=target_bin,
            state=state,
            contig_profiles=contig_scg_profiles,
        )
        scg_note = f";scg={scg_decision.status}:{scg_decision.reason}:{scg_decision.note}"
        if scg_decision.status == "veto":
            return MergeDecision(
                candidate=candidate,
                accepted=False,
                reason=scg_decision.reason,
                confidence=0.0,
                note=scg_decision.note,
            )

    confidence = min(1.0, max(0.0, 0.5 + delta_c))
    return MergeDecision(
        candidate=candidate,
        accepted=True,
        reason="merge_improves_contact_coherence",
        confidence=float(confidence),
        note=(
            f"delta_contact={delta_c:.3f};"
            f"cross_support={candidate.cross_support:.3f};"
            f"support_edges={candidate.support_edges}"
            f"{scg_note}"
        ),
    )


def apply_merge_decision(*, state: RefineState, decision: MergeDecision) -> None:
    """Apply an accepted merge by moving target-bin members into the source bin."""
    if not decision.accepted:
        return
    source_bin = decision.candidate.source_bin
    target_bin = decision.candidate.target_bin
    for contig_id in sorted(state.bin_to_contigs().get(target_bin, [])):
        state.assign(
            contig_id,
            source_bin,
            stage="merge",
            reason="merged_into_source_bin",
        )


def _choose_merge_direction(
    *,
    state: RefineState,
    left_bin: str,
    right_bin: str,
) -> tuple[str, str]:
    left_length = sum(int(state.contig_lengths.get(contig_id, 0)) for contig_id in state.bin_to_contigs().get(left_bin, []))
    right_length = sum(int(state.contig_lengths.get(contig_id, 0)) for contig_id in state.bin_to_contigs().get(right_bin, []))
    if left_length > right_length:
        return left_bin, right_bin
    if right_length > left_length:
        return right_bin, left_bin
    return min(left_bin, right_bin), max(left_bin, right_bin)


def _profiles_feature_compatible(
    left: BinFeatureProfile | None,
    right: BinFeatureProfile | None,
) -> bool:
    import numpy as np

    if left is None or right is None:
        return True
    centroid_distance = float(np.linalg.norm(left.centroid - right.centroid))
    limit = max(0.5, 3.0 * max(float(left.distance_mad), float(right.distance_mad), 0.1))
    return centroid_distance <= limit


def _profiles_coverage_compatible(
    left: BinFeatureProfile | None,
    right: BinFeatureProfile | None,
) -> bool:
    if (
        left is None
        or right is None
        or left.median_coverage is None
        or right.median_coverage is None
    ):
        return True
    left_mad = float(left.coverage_mad or 0.0)
    right_mad = float(right.coverage_mad or 0.0)
    diff = abs(float(left.median_coverage) - float(right.median_coverage))
    if left_mad > 0.0 or right_mad > 0.0:
        return diff <= 3.0 * max(left_mad, right_mad, 0.5)
    return diff <= max(1.0, 0.25 * max(float(left.median_coverage), float(right.median_coverage)))


def _weighted_before_coherence(
    *,
    state: RefineState,
    before_stats,
    source_bin: str,
    target_bin: str,
) -> float:
    source_length = float(sum(int(state.contig_lengths.get(contig_id, 0)) for contig_id in state.bin_to_contigs().get(source_bin, [])))
    target_length = float(sum(int(state.contig_lengths.get(contig_id, 0)) for contig_id in state.bin_to_contigs().get(target_bin, [])))
    total = source_length + target_length
    if total <= 0.0:
        return 0.0
    source_value = float(before_stats.get(source_bin, _empty_stat(source_bin)).coherence)
    target_value = float(before_stats.get(target_bin, _empty_stat(target_bin)).coherence)
    return float((source_length / total) * source_value + (target_length / total) * target_value)


def _group_feature_dispersion(
    *,
    group: tuple[str, ...],
    feature_matrix: "object",
    contig_name_to_idx: dict[str, int],
) -> float:
    import numpy as np

    member_idx = [contig_name_to_idx[contig_id] for contig_id in group if contig_id in contig_name_to_idx]
    if len(member_idx) < 2:
        return 0.0
    X = np.asarray(feature_matrix[member_idx, :], dtype=float)
    centroid = X.mean(axis=0)
    distances = np.linalg.norm(X - centroid, axis=1)
    return float(median(distances.tolist()))


def _group_coverage_dispersion(
    *,
    group: tuple[str, ...],
    coverage_by_contig: dict[str, float],
) -> float | None:
    values = [float(coverage_by_contig[contig_id]) for contig_id in group if contig_id in coverage_by_contig]
    if len(values) < 2:
        return 0.0 if values else None
    return float(mad(values))


def _empty_stat(bin_id: str):
    from porebin_genome.refine.coherence import BinCoherenceStat

    return BinCoherenceStat(bin_id=bin_id, numerator=0.0, denominator=0.0, coherence=0.0)
