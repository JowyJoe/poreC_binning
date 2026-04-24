"""Local split generation and evaluation for suspect genome bins."""

from __future__ import annotations

from collections import defaultdict

from porebin_genome.refine.models import BinSnapshot, RefineState, SplitCandidate, SplitDecision, SupportSummary, SuspectBin
from porebin_genome.refine.support import mad, median


MIN_SPLIT_CHILD_CONTIGS = 2
MIN_SPLIT_CHILD_TOTAL_LENGTH = 1000


def generate_split_candidates(
    *,
    state: RefineState,
    suspects: list[SuspectBin],
    support_summary: SupportSummary,
    feature_matrix: "object",
    contig_name_to_idx: dict[str, int],
) -> list[SplitCandidate]:
    """Generate at most one local two-way split candidate per suspect bin."""
    candidates: list[SplitCandidate] = []
    for suspect in suspects:
        members = state.bin_to_contigs().get(suspect.bin_id, [])
        if len(members) < 4:
            continue
        components = list(support_summary.contact_components_by_bin.get(suspect.bin_id, ()))
        if len(components) > 1:
            primary = tuple(sorted(components[0]))
            secondary_members = sorted({contig_id for group in components[1:] for contig_id in group})
            secondary = tuple(secondary_members)
            if primary and secondary:
                candidates.append(
                    SplitCandidate(
                        source_bin=suspect.bin_id,
                        groups=(primary, secondary),
                        method="contact_components",
                        reasons=suspect.reasons,
                    )
                )
                continue
        kmeans_groups = _kmeans_groups(
            members=members,
            feature_matrix=feature_matrix,
            contig_name_to_idx=contig_name_to_idx,
        )
        if kmeans_groups is not None:
            candidates.append(
                SplitCandidate(
                    source_bin=suspect.bin_id,
                    groups=kmeans_groups,
                    method="feature_kmeans2",
                    reasons=suspect.reasons,
                )
            )
    return candidates


def evaluate_split_candidate(
    *,
    candidate: SplitCandidate,
    state: RefineState,
    snapshot: BinSnapshot,
    support_summary: SupportSummary,
    feature_matrix: "object",
    contig_name_to_idx: dict[str, int],
    new_bin_id: str,
) -> SplitDecision:
    """Accept or reject a split candidate using size and consistency-improvement gates."""
    import numpy as np

    groups = [tuple(sorted(group)) for group in candidate.groups if group]
    if len(groups) != 2:
        return SplitDecision(
            candidate=candidate,
            accepted=False,
            new_bin_id="",
            reason="split_candidate_not_two_way",
            confidence=0.0,
            note="candidate did not resolve to two non-empty child groups",
        )

    child_lengths = [sum(state.contig_lengths.get(contig_id, 0) for contig_id in group) for group in groups]
    child_counts = [len(group) for group in groups]
    if any(count < MIN_SPLIT_CHILD_CONTIGS for count in child_counts):
        return SplitDecision(
            candidate=candidate,
            accepted=False,
            new_bin_id="",
            reason="split_child_too_small_by_contig_count",
            confidence=0.0,
            note=f"child_counts={child_counts}",
        )
    if any(total_length < MIN_SPLIT_CHILD_TOTAL_LENGTH for total_length in child_lengths):
        return SplitDecision(
            candidate=candidate,
            accepted=False,
            new_bin_id="",
            reason="split_child_too_small_by_total_length",
            confidence=0.0,
            note=f"child_total_length={child_lengths}",
        )

    pre_contact = float(snapshot.contact_consistency)
    pre_feature = float(snapshot.feature_dispersion)
    pre_coverage = snapshot.coverage_dispersion

    child_contact_values = [
        _group_contact_consistency(
            source_bin=candidate.source_bin,
            group=group,
            members=list(snapshot.members),
            support_summary=support_summary,
        )
        for group in groups
    ]
    child_feature_values = [
        _group_feature_dispersion(
            group=group,
            feature_matrix=feature_matrix,
            contig_name_to_idx=contig_name_to_idx,
        )
        for group in groups
    ]
    child_coverage_values = [
        _group_coverage_dispersion(group=group, coverage_by_contig=state.coverage_by_contig)
        for group in groups
    ]

    total_length = float(sum(child_lengths))
    weighted_contact = sum(
        (float(child_lengths[idx]) / total_length) * float(child_contact_values[idx])
        for idx in range(len(groups))
    )
    weighted_feature = sum(
        (float(child_lengths[idx]) / total_length) * float(child_feature_values[idx])
        for idx in range(len(groups))
    )
    if pre_coverage is None:
        weighted_coverage = None
    else:
        coverage_weights = []
        for idx in range(len(groups)):
            value = child_coverage_values[idx]
            coverage_weights.append(0.0 if value is None else float(value))
        weighted_coverage = sum(
            (float(child_lengths[idx]) / total_length) * float(coverage_weights[idx])
            for idx in range(len(groups))
        )

    contact_improved = weighted_contact > (pre_contact + 0.05)
    feature_improved = weighted_feature < max(0.0, pre_feature - 0.05)
    coverage_improved = (
        pre_coverage is not None
        and weighted_coverage is not None
        and weighted_coverage < max(0.0, float(pre_coverage) - 0.5)
    )

    if not any((contact_improved, feature_improved, coverage_improved)):
        return SplitDecision(
            candidate=candidate,
            accepted=False,
            new_bin_id="",
            reason="split_consistency_not_improved",
            confidence=0.0,
            note=(
                f"pre_contact={pre_contact:.3f};post_contact={weighted_contact:.3f};"
                f"pre_feature={pre_feature:.3f};post_feature={weighted_feature:.3f};"
                f"pre_coverage={pre_coverage};post_coverage={weighted_coverage}"
            ),
        )

    confidence = min(
        1.0,
        0.55
        + (0.20 if contact_improved else 0.0)
        + (0.15 if feature_improved else 0.0)
        + (0.10 if coverage_improved else 0.0),
    )
    return SplitDecision(
        candidate=candidate,
        accepted=True,
        new_bin_id=str(new_bin_id),
        reason="split_improves_bin_consistency",
        confidence=float(confidence),
        note=(
            f"method={candidate.method};contact_improved={int(contact_improved)};"
            f"feature_improved={int(feature_improved)};coverage_improved={int(coverage_improved)}"
        ),
    )


def apply_split_decision(*, state: RefineState, decision: SplitDecision) -> None:
    """Apply an accepted split decision to the refine state."""
    if not decision.accepted:
        return

    groups = [tuple(sorted(group)) for group in decision.candidate.groups if group]
    groups.sort(key=lambda group: (-len(group), tuple(group)))
    keep_group = groups[0]
    move_group = groups[1]
    source_bin = decision.candidate.source_bin
    target_bin = decision.new_bin_id

    for contig_id in keep_group:
        state.assign(
            contig_id,
            source_bin,
            stage="split",
            reason="split_primary_child_retained",
        )
    for contig_id in move_group:
        state.assign(
            contig_id,
            target_bin,
            stage="split",
            reason="split_target_child_created",
        )


def _kmeans_groups(
    *,
    members: list[str],
    feature_matrix: "object",
    contig_name_to_idx: dict[str, int],
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    try:
        import numpy as np
        from sklearn.cluster import KMeans
    except Exception:  # pragma: no cover
        return None

    member_idx = [contig_name_to_idx[contig_id] for contig_id in members if contig_id in contig_name_to_idx]
    if len(member_idx) < 4:
        return None
    X = np.asarray(feature_matrix[member_idx, :], dtype=float)
    labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
    groups: list[tuple[str, ...]] = []
    for label in (0, 1):
        group = tuple(sorted(members[idx] for idx, assigned in enumerate(labels.tolist()) if int(assigned) == label))
        if group:
            groups.append(group)
    if len(groups) != 2:
        return None
    if min(len(group) for group in groups) < MIN_SPLIT_CHILD_CONTIGS:
        return None
    return groups[0], groups[1]


def _group_contact_consistency(
    *,
    source_bin: str,
    group: tuple[str, ...],
    members: list[str],
    support_summary: SupportSummary,
) -> float:
    ratios: list[float] = []
    for contig_id in group:
        in_group = 0.0
        total = 0.0
        for other_id in members:
            if other_id == contig_id:
                continue
            pair_key = tuple(sorted((contig_id, other_id)))
            pair_weight = float(support_summary.pair_support_by_bin.get(source_bin, {}).get(pair_key, 0.0))
            total += pair_weight
            if other_id in group:
                in_group += pair_weight
        if total <= 0.0:
            ratios.append(0.0)
        else:
            ratios.append(in_group / total)
    return float(median(ratios))


def _group_feature_dispersion(
    *,
    group: tuple[str, ...],
    feature_matrix: "object",
    contig_name_to_idx: dict[str, int],
) -> float:
    import numpy as np

    member_idx = [contig_name_to_idx[contig_id] for contig_id in group]
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
    if not values:
        return None
    return float(mad(values))
