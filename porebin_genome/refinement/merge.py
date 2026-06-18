"""One-pass Pore-C merge candidate generation for stable genome bins."""

from __future__ import annotations

import math
from dataclasses import dataclass

from porebin_genome.refinement.actions import (
    ActionType,
    Decision,
    DecisionStatus,
    Proposal,
)
from porebin_genome.refinement.profiles import (
    ActionEvidenceUpdate,
    EvidenceProfileState,
)
from porebin_genome.refinement.state import (
    ActionDelta,
    AssignmentChange,
    RefineState,
)


@dataclass(frozen=True)
class BinPairSupport:
    """Direct Pore-C support accumulated for one unordered bin pair."""

    left_bin_idx: int
    right_bin_idx: int
    raw_support: float
    normalized_support: float
    supporting_edge_count: int

    @property
    def pair(self) -> tuple[int, int]:
        return self.left_bin_idx, self.right_bin_idx


@dataclass(frozen=True)
class MergeCandidate:
    """One mutual-best complete-bin merge proposal."""

    left_bin_idx: int
    right_bin_idx: int
    source_bin_idx: int
    target_bin_idx: int
    raw_support: float
    normalized_support: float
    supporting_edge_count: int
    proposal: Proposal

    @property
    def pair(self) -> tuple[int, int]:
        return self.left_bin_idx, self.right_bin_idx


@dataclass(frozen=True)
class MergeCandidateBatch:
    """Auditable result of one global merge-support scan."""

    state_version: int
    eligible_bin_indices: tuple[int, ...]
    supported_pair_count: int
    candidates: tuple[MergeCandidate, ...]


@dataclass(frozen=True)
class MergeBatchUpdate:
    """One synchronized transaction for accepted disjoint merge decisions."""

    decisions: tuple[Decision, ...]
    contact_delta: ActionDelta
    profile_update: ActionEvidenceUpdate
    state_version: int


def generate_merge_candidates(
    *,
    state: RefineState,
    profiles: EvidenceProfileState,
    min_supporting_edges: int = 2,
    eps: float = 1e-12,
) -> MergeCandidateBatch:
    """Find disjoint mutual-best bin pairs in one hyperedge scan."""
    _validate_inputs(
        state=state,
        profiles=profiles,
        min_supporting_edges=min_supporting_edges,
        eps=eps,
    )
    eligible = tuple(
        bin_idx
        for bin_idx in range(state.n_bins)
        if _is_merge_eligible(bin_idx, state=state, profiles=profiles)
    )
    if len(eligible) < 2:
        return MergeCandidateBatch(
            state_version=state.version,
            eligible_bin_indices=eligible,
            supported_pair_count=0,
            candidates=(),
        )
    eligible_set = set(eligible)
    raw_support: dict[tuple[int, int], float] = {}
    edge_count: dict[tuple[int, int], int] = {}

    for edge_idx in state.contact_index.scan_edge_indices():
        masses = tuple(
            sorted(
                (
                    (bin_idx, float(mass))
                    for bin_idx, mass in state.edge_masses(edge_idx).items()
                    if bin_idx in eligible_set and float(mass) > 0.0
                ),
                key=lambda item: item[0],
            )
        )
        if len(masses) < 2:
            continue
        reliability = float(
            state.contact_index.edge_reliability[edge_idx]
        )
        if reliability <= 0.0:
            continue
        for left_offset in range(len(masses)):
            left_bin, left_mass = masses[left_offset]
            for right_offset in range(left_offset + 1, len(masses)):
                right_bin, right_mass = masses[right_offset]
                contribution = reliability * left_mass * right_mass
                if contribution <= 0.0:
                    continue
                pair = (left_bin, right_bin)
                raw_support[pair] = float(
                    raw_support.get(pair, 0.0) + contribution
                )
                edge_count[pair] = edge_count.get(pair, 0) + 1

    pair_support = tuple(
        _pair_support(
            pair,
            raw_support=support,
            supporting_edge_count=edge_count[pair],
            state=state,
            eps=eps,
        )
        for pair, support in sorted(raw_support.items())
        if support > 0.0 and edge_count[pair] >= int(min_supporting_edges)
    )
    best_neighbor = _best_neighbors(pair_support)
    mutual_pairs = tuple(
        support
        for support in pair_support
        if (
            best_neighbor.get(support.left_bin_idx)
            == support.right_bin_idx
            and best_neighbor.get(support.right_bin_idx)
            == support.left_bin_idx
        )
    )
    candidates = tuple(
        _merge_candidate(support, state=state)
        for support in mutual_pairs
    )
    return MergeCandidateBatch(
        state_version=state.version,
        eligible_bin_indices=eligible,
        supported_pair_count=len(pair_support),
        candidates=candidates,
    )


def prepare_merge_batch(
    *,
    state: RefineState,
    profiles: EvidenceProfileState,
    decisions: tuple[Decision, ...],
) -> MergeBatchUpdate:
    """Combine accepted disjoint merges evaluated on one state snapshot."""
    _validate_shared_state(state=state, profiles=profiles)
    normalized = tuple(decisions)
    if not normalized:
        raise ValueError("A merge batch must contain at least one decision.")

    used_bins: set[int] = set()
    changes: list[AssignmentChange] = []
    for decision in normalized:
        if decision.status is not DecisionStatus.ACCEPT:
            raise ValueError("A merge batch may contain only accepted decisions.")
        proposal = decision.proposal
        if proposal.action_type is not ActionType.MERGE:
            raise ValueError("A merge batch may contain only merge decisions.")
        if proposal.state_version != state.version:
            raise ValueError("Merge decisions must share the current state version.")
        pair_bins = set(proposal.source_bins) | set(proposal.target_bins)
        overlap = used_bins & pair_bins
        if overlap:
            raise ValueError(
                "Merge decisions in one batch must be bin-disjoint; "
                f"overlap={sorted(overlap)}"
            )
        used_bins.update(pair_bins)
        changes.extend(proposal.changes)

    contact_delta = state.evaluate(
        sorted(
            changes,
            key=lambda change: (
                change.old_bin_idx,
                change.new_bin_idx,
                change.contig_idx,
            ),
        )
    )
    profile_update = profiles.evaluate(contact_delta)
    return MergeBatchUpdate(
        decisions=normalized,
        contact_delta=contact_delta,
        profile_update=profile_update,
        state_version=state.version,
    )


def apply_merge_batch(
    *,
    state: RefineState,
    profiles: EvidenceProfileState,
    update: MergeBatchUpdate,
) -> None:
    """Apply one prepared merge batch as a two-state transaction."""
    _validate_shared_state(state=state, profiles=profiles)
    if update.state_version != state.version:
        raise ValueError("MergeBatchUpdate is stale.")
    state.apply(update.contact_delta)
    try:
        profiles.apply(update.profile_update)
    except Exception:
        state.rollback(update.contact_delta)
        raise


def rollback_merge_batch(
    *,
    state: RefineState,
    profiles: EvidenceProfileState,
    update: MergeBatchUpdate,
) -> None:
    """Rollback the most recently applied merge batch."""
    profiles.rollback(update.profile_update)
    state.rollback(update.contact_delta)


def _is_merge_eligible(
    bin_idx: int,
    *,
    state: RefineState,
    profiles: EvidenceProfileState,
) -> bool:
    profile = profiles.profile(bin_idx)
    return (
        bool(state.bin_members(bin_idx))
        and profile.scg.duplicate_burden == 0
        and profile.embedding.observed_count == profile.member_count
        and profile.embedding.centroid is not None
        and profile.embedding.radius is not None
    )


def _pair_support(
    pair: tuple[int, int],
    *,
    raw_support: float,
    supporting_edge_count: int,
    state: RefineState,
    eps: float,
) -> BinPairSupport:
    left_bin, right_bin = pair
    denominator = math.sqrt(
        float(state.contact_denominator[left_bin])
        * float(state.contact_denominator[right_bin])
        + float(eps)
    )
    normalized = (
        float(raw_support) / denominator
        if denominator > 0.0
        else 0.0
    )
    return BinPairSupport(
        left_bin_idx=left_bin,
        right_bin_idx=right_bin,
        raw_support=float(raw_support),
        normalized_support=float(normalized),
        supporting_edge_count=int(supporting_edge_count),
    )


def _best_neighbors(
    pair_support: tuple[BinPairSupport, ...],
) -> dict[int, int]:
    neighbors: dict[int, list[tuple[int, BinPairSupport]]] = {}
    for support in pair_support:
        neighbors.setdefault(support.left_bin_idx, []).append(
            (support.right_bin_idx, support)
        )
        neighbors.setdefault(support.right_bin_idx, []).append(
            (support.left_bin_idx, support)
        )
    return {
        bin_idx: min(
            options,
            key=lambda item: (
                -item[1].normalized_support,
                item[0],
            ),
        )[0]
        for bin_idx, options in neighbors.items()
    }


def _merge_candidate(
    support: BinPairSupport,
    *,
    state: RefineState,
) -> MergeCandidate:
    left_bin = support.left_bin_idx
    right_bin = support.right_bin_idx
    left_length = int(state.bin_total_length[left_bin])
    right_length = int(state.bin_total_length[right_bin])
    if left_length > right_length:
        target_bin, source_bin = left_bin, right_bin
    elif right_length > left_length:
        target_bin, source_bin = right_bin, left_bin
    else:
        target_bin, source_bin = min(left_bin, right_bin), max(
            left_bin,
            right_bin,
        )
    changes = tuple(
        AssignmentChange(
            contig_idx=contig_idx,
            old_bin_idx=source_bin,
            new_bin_idx=target_bin,
        )
        for contig_idx in sorted(state.bin_members(source_bin))
    )
    proposal = Proposal(
        proposal_id=(
            f"merge:v{state.version}:bin{source_bin}->bin{target_bin}"
        ),
        action_type=ActionType.MERGE,
        changes=changes,
        state_version=state.version,
    )
    return MergeCandidate(
        left_bin_idx=left_bin,
        right_bin_idx=right_bin,
        source_bin_idx=source_bin,
        target_bin_idx=target_bin,
        raw_support=support.raw_support,
        normalized_support=support.normalized_support,
        supporting_edge_count=support.supporting_edge_count,
        proposal=proposal,
    )


def _validate_inputs(
    *,
    state: RefineState,
    profiles: EvidenceProfileState,
    min_supporting_edges: int,
    eps: float,
) -> None:
    _validate_shared_state(state=state, profiles=profiles)
    if int(min_supporting_edges) < 1:
        raise ValueError("min_supporting_edges must be positive.")
    if not math.isfinite(float(eps)) or float(eps) <= 0.0:
        raise ValueError("eps must be finite and positive.")


def _validate_shared_state(
    *,
    state: RefineState,
    profiles: EvidenceProfileState,
) -> None:
    if profiles.refine_state is not state:
        raise ValueError(
            "Merge generator and EvidenceProfileState must share one "
            "RefineState instance."
        )
    if profiles.version != state.version:
        raise ValueError(
            "Merge generator requires synchronized assignment and profiles."
        )
