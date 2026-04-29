"""Local split generation and evaluation for HCR-style refinement."""

from __future__ import annotations

from porebin_genome.refine.coherence import compute_bin_coherence
from porebin_genome.refine.features import mad, median
from porebin_genome.refine.hyperedge import HyperedgeStore, ReliabilityFn, default_edge_reliability
from porebin_genome.refine.local_graph import project_local_pair_graph, select_edges_for_bin
from porebin_genome.refine.markers import ContigScgProfile, evaluate_split_scg_transition
from porebin_genome.refine.models import BinFeatureProfile, RefineState, SplitCandidate, SplitDecision, SuspectBin


MIN_SPLIT_CHILD_CONTIGS = 2
MIN_SPLIT_CHILD_TOTAL_LENGTH = 1000
FEATURE_WORSENING_TOLERANCE = 0.05
COVERAGE_WORSENING_TOLERANCE = 0.50


def generate_split_candidates(
    *,
    state: RefineState,
    suspects: list[SuspectBin],
    store: HyperedgeStore,
    feature_matrix: "object",
    contig_name_to_idx: dict[str, int],
    reliability_fn: ReliabilityFn | None = None,
    min_mass_in_bin: float = 0.6,
) -> list[SplitCandidate]:
    """Generate at most one local two-way split candidate per suspect bin."""
    reliability_fn = reliability_fn or default_edge_reliability

    candidates: list[SplitCandidate] = []
    for suspect in suspects:
        members = sorted(state.bin_to_contigs().get(suspect.bin_id, []))
        if len(members) < 4:
            continue

        edge_ids = select_edges_for_bin(
            store,
            state.current_assignment,
            bin_id=suspect.bin_id,
            min_mass_in_bin=min_mass_in_bin,
            reliability_fn=reliability_fn,
        )
        graph = project_local_pair_graph(
            store,
            edge_ids=edge_ids,
            focus_contigs=members,
            reliability_fn=reliability_fn,
        )
        components = _connected_components_from_graph(nodes=members, edges=graph.edges)
        leiden_groups = _leiden_two_way_groups(graph=graph)
        if leiden_groups is not None:
            method = "leiden_two_way"
            if len(components) > 1 and _partition_matches_components(leiden_groups, components):
                method = "leiden_two_way_components"
            candidates.append(
                SplitCandidate(
                    source_bin=suspect.bin_id,
                    groups=leiden_groups,
                    method=method,
                    reasons=suspect.reasons,
                )
            )
            continue

        if len(components) > 1:
            primary = tuple(sorted(components[0]))
            secondary_members = sorted({contig_id for group in components[1:] for contig_id in group})
            secondary = tuple(secondary_members)
            if primary and secondary:
                candidates.append(
                    SplitCandidate(
                        source_bin=suspect.bin_id,
                        groups=(primary, secondary),
                        method="local_graph_components",
                        reasons=suspect.reasons,
                    )
                )
                continue

        kmeans_groups = _feature_kmeans_groups(
            members=members,
            feature_matrix=feature_matrix,
            contig_name_to_idx=contig_name_to_idx,
        )
        if kmeans_groups is not None:
            candidates.append(
                SplitCandidate(
                    source_bin=suspect.bin_id,
                    groups=kmeans_groups,
                    method="feature_kmeans2_fallback",
                    reasons=suspect.reasons,
                )
            )
    return candidates


def evaluate_split_candidate(
    *,
    candidate: SplitCandidate,
    state: RefineState,
    store: HyperedgeStore,
    feature_matrix: "object",
    contig_name_to_idx: dict[str, int],
    profiles: dict[str, BinFeatureProfile],
    new_bin_id: str,
    reliability_fn: ReliabilityFn | None = None,
    contig_scg_profiles: dict[str, ContigScgProfile] | None = None,
) -> SplitDecision:
    """Accept or reject a split candidate with HCR-style contact improvement."""
    reliability_fn = reliability_fn or default_edge_reliability

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

    groups.sort(
        key=lambda group: (
            -sum(int(state.contig_lengths.get(contig_id, 0)) for contig_id in group),
            -len(group),
            tuple(group),
        )
    )
    keep_group, move_group = groups
    child_lengths = [
        sum(int(state.contig_lengths.get(contig_id, 0)) for contig_id in keep_group),
        sum(int(state.contig_lengths.get(contig_id, 0)) for contig_id in move_group),
    ]
    child_counts = [len(keep_group), len(move_group)]
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

    source_bin = candidate.source_bin
    before_stats = compute_bin_coherence(
        store,
        state.current_assignment,
        reliability_fn=reliability_fn,
        bins=(source_bin,),
    )
    before_coherence = float(before_stats.get(source_bin, _empty_stat(source_bin)).coherence)

    after_assignment = dict(state.current_assignment)
    for contig_id in move_group:
        after_assignment[contig_id] = str(new_bin_id)
    after_stats = compute_bin_coherence(
        store,
        after_assignment,
        reliability_fn=reliability_fn,
        bins=(source_bin, str(new_bin_id)),
    )
    after_keep = float(after_stats.get(source_bin, _empty_stat(source_bin)).coherence)
    after_move = float(after_stats.get(str(new_bin_id), _empty_stat(str(new_bin_id))).coherence)

    total_length = float(sum(child_lengths))
    weighted_after_coherence = (
        (float(child_lengths[0]) / total_length) * after_keep
        + (float(child_lengths[1]) / total_length) * after_move
    )
    delta_contact = float(weighted_after_coherence - before_coherence)
    resolves_disconnected_components = candidate.method in {
        "local_graph_components",
        "leiden_two_way_components",
    }
    if delta_contact <= 0.0 and not resolves_disconnected_components:
        return SplitDecision(
            candidate=candidate,
            accepted=False,
            new_bin_id="",
            reason="split_does_not_improve_contact_coherence",
            confidence=0.0,
            note=(
                f"pre_contact={before_coherence:.3f};"
                f"post_contact={weighted_after_coherence:.3f};"
                f"delta_contact={delta_contact:.3f}"
            ),
        )

    profile = profiles.get(source_bin)
    child_feature_values = [
        _group_feature_dispersion(
            group=group,
            feature_matrix=feature_matrix,
            contig_name_to_idx=contig_name_to_idx,
        )
        for group in (keep_group, move_group)
    ]
    weighted_feature = sum(
        (float(child_lengths[idx]) / total_length) * float(child_feature_values[idx])
        for idx in range(2)
    )
    if profile is not None and weighted_feature > float(profile.feature_dispersion + FEATURE_WORSENING_TOLERANCE):
        return SplitDecision(
            candidate=candidate,
            accepted=False,
            new_bin_id="",
            reason="split_child_feature_dispersion_worsened",
            confidence=0.0,
            note=(
                f"pre_feature={profile.feature_dispersion:.3f};"
                f"post_feature={weighted_feature:.3f}"
            ),
        )

    child_coverage_values = [
        _group_coverage_dispersion(group=group, coverage_by_contig=state.coverage_by_contig)
        for group in (keep_group, move_group)
    ]
    weighted_coverage = None
    if any(value is not None for value in child_coverage_values):
        weighted_coverage = sum(
            (float(child_lengths[idx]) / total_length) * float(child_coverage_values[idx] or 0.0)
            for idx in range(2)
        )
    if (
        profile is not None
        and profile.coverage_mad is not None
        and weighted_coverage is not None
        and weighted_coverage > float(profile.coverage_mad + COVERAGE_WORSENING_TOLERANCE)
    ):
        return SplitDecision(
            candidate=candidate,
            accepted=False,
            new_bin_id="",
            reason="split_child_coverage_dispersion_worsened",
            confidence=0.0,
            note=(
                f"pre_coverage={profile.coverage_mad:.3f};"
                f"post_coverage={weighted_coverage:.3f}"
            ),
        )

    scg_note = ""
    if contig_scg_profiles is not None:
        scg_decision = evaluate_split_scg_transition(
            source_bin=source_bin,
            child_groups=(keep_group, move_group),
            state=state,
            contig_profiles=contig_scg_profiles,
            new_bin_id=str(new_bin_id),
        )
        scg_note = f";scg={scg_decision.status}:{scg_decision.reason}:{scg_decision.note}"
        if scg_decision.status == "veto":
            return SplitDecision(
                candidate=candidate,
                accepted=False,
                new_bin_id="",
                reason=scg_decision.reason,
                confidence=0.0,
                note=scg_decision.note,
            )

    confidence = min(1.0, max(0.0, 0.5 + delta_contact))
    reason = (
        "split_resolves_disconnected_contact_components"
        if delta_contact <= 0.0 and resolves_disconnected_components
        else "split_improves_contact_coherence"
    )
    return SplitDecision(
        candidate=candidate,
        accepted=True,
        new_bin_id=str(new_bin_id),
        reason=reason,
        confidence=float(confidence),
        note=(
            f"method={candidate.method};"
            f"pre_contact={before_coherence:.3f};"
            f"post_contact={weighted_after_coherence:.3f};"
            f"delta_contact={delta_contact:.3f}"
            f"{scg_note}"
        ),
    )


def apply_split_decision(*, state: RefineState, decision: SplitDecision) -> None:
    """Apply an accepted split decision to the refine state."""
    if not decision.accepted:
        return

    groups = [tuple(sorted(group)) for group in decision.candidate.groups if group]
    groups.sort(
        key=lambda group: (
            -sum(int(state.contig_lengths.get(contig_id, 0)) for contig_id in group),
            -len(group),
            tuple(group),
        )
    )
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


def _connected_components_from_graph(
    *,
    nodes: list[str],
    edges: tuple[tuple[str, str, float], ...],
) -> list[tuple[str, ...]]:
    adjacency: dict[str, set[str]] = {str(node): set() for node in nodes}
    for left_id, right_id, weight in edges:
        if float(weight) <= 0.0:
            continue
        adjacency.setdefault(str(left_id), set()).add(str(right_id))
        adjacency.setdefault(str(right_id), set()).add(str(left_id))

    seen: set[str] = set()
    groups: list[tuple[str, ...]] = []
    for node in sorted(adjacency.keys()):
        if node in seen:
            continue
        stack = [node]
        seen.add(node)
        component: list[str] = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency.get(current, ())):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
        groups.append(tuple(sorted(component)))

    groups.sort(key=lambda group: (-len(group), tuple(group)))
    return groups


def _feature_kmeans_groups(
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

    filtered_members = [contig_id for contig_id in members if contig_id in contig_name_to_idx]
    member_idx = [contig_name_to_idx[contig_id] for contig_id in filtered_members]
    if len(member_idx) < 4:
        return None
    X = np.asarray(feature_matrix[member_idx, :], dtype=float)
    labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
    groups: list[tuple[str, ...]] = []
    for label in (0, 1):
        group = tuple(
            sorted(
                filtered_members[idx]
                for idx, assigned in enumerate(labels.tolist())
                if int(assigned) == label
            )
        )
        if group:
            groups.append(group)
    if len(groups) != 2:
        return None
    if min(len(group) for group in groups) < MIN_SPLIT_CHILD_CONTIGS:
        return None
    return groups[0], groups[1]


def _leiden_two_way_groups(
    *,
    graph,
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    try:
        import igraph as ig
        import leidenalg
    except Exception:  # pragma: no cover
        return None

    if len(graph.nodes) < 4 or not graph.edges:
        return None

    node_order = list(graph.nodes)
    node_index = {node_id: idx for idx, node_id in enumerate(node_order)}
    ig_graph = ig.Graph()
    ig_graph.add_vertices(node_order)

    edge_pairs: list[tuple[int, int]] = []
    edge_weights: list[float] = []
    for left_id, right_id, weight in graph.edges:
        if float(weight) <= 0.0:
            continue
        edge_pairs.append((node_index[left_id], node_index[right_id]))
        edge_weights.append(float(weight))
    if not edge_pairs:
        return None
    ig_graph.add_edges(edge_pairs)
    ig_graph.es["weight"] = edge_weights

    best_groups: tuple[tuple[str, ...], tuple[str, ...]] | None = None
    best_quality: float | None = None
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    for gamma in (0.25, 0.5, 1.0, 2.0, 4.0):
        partition = leidenalg.find_partition(
            ig_graph,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=float(gamma),
            seed=0,
        )
        groups = _groups_from_membership(node_order=node_order, membership=partition.membership)
        if groups is None or groups in seen:
            continue
        seen.add(groups)
        quality = float(partition.quality())
        if best_quality is None or quality > best_quality:
            best_quality = quality
            best_groups = groups
    return best_groups


def _groups_from_membership(
    *,
    node_order: list[str],
    membership: list[int],
) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    grouped: dict[int, list[str]] = {}
    for node_id, label in zip(node_order, membership, strict=True):
        grouped.setdefault(int(label), []).append(str(node_id))
    groups = tuple(
        sorted(
            (tuple(sorted(group)) for group in grouped.values() if group),
            key=lambda group: (-len(group), tuple(group)),
        )
    )
    if len(groups) != 2:
        return None
    if min(len(group) for group in groups) < MIN_SPLIT_CHILD_CONTIGS:
        return None
    return groups


def _partition_matches_components(
    groups: tuple[tuple[str, ...], tuple[str, ...]],
    components: list[tuple[str, ...]],
) -> bool:
    collapsed = (
        tuple(sorted(components[0])),
        tuple(sorted({contig_id for component in components[1:] for contig_id in component})),
    )
    normalized_groups = tuple(sorted((tuple(sorted(groups[0])), tuple(sorted(groups[1]))), key=lambda group: (-len(group), tuple(group))))
    normalized_collapsed = tuple(sorted((tuple(sorted(collapsed[0])), tuple(sorted(collapsed[1]))), key=lambda group: (-len(group), tuple(group))))
    return normalized_groups == normalized_collapsed


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
