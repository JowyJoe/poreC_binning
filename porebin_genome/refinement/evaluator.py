"""Local, side-effect-free evaluation of refinement proposals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from porebin_genome.refinement.actions import (
    ActionEvidence,
    ActionType,
    BinPairEmbeddingEvidence,
    BinContactTransition,
    ContactEvidence,
    ContigContactEvidence,
    ContigEmbeddingEvidence,
    Decision,
    DecisionStatus,
    GateStatus,
    MetricEvidence,
    Proposal,
    ScgEvidence,
    StructuralEvidence,
)
from porebin_genome.refinement.profiles import (
    ActionEvidenceUpdate,
    BinEvidenceTransition,
    EvidenceProfileState,
)
from porebin_genome.refinement.state import (
    ActionDelta,
    RefineState,
    RefineStateError,
)


@dataclass(frozen=True)
class EvaluationConfig:
    """Only structural bounds; empirical decision thresholds live in Policy."""

    min_bin_contigs: int = 1
    min_bin_length: int = 1
    eps: float = 1e-12

    def __post_init__(self) -> None:
        min_bin_contigs = int(self.min_bin_contigs)
        min_bin_length = int(self.min_bin_length)
        eps = float(self.eps)
        if min_bin_contigs < 1:
            raise ValueError("min_bin_contigs must be positive.")
        if min_bin_length < 1:
            raise ValueError("min_bin_length must be positive.")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError("eps must be a finite positive number.")
        object.__setattr__(self, "min_bin_contigs", min_bin_contigs)
        object.__setattr__(self, "min_bin_length", min_bin_length)
        object.__setattr__(self, "eps", eps)


class ActionEvaluator:
    """Evaluate proposals from exact local contact and profile deltas."""

    def __init__(
        self,
        *,
        state: RefineState,
        profiles: EvidenceProfileState,
        config: EvaluationConfig | None = None,
    ) -> None:
        if profiles.refine_state is not state:
            raise ValueError(
                "ActionEvaluator state and EvidenceProfileState must share "
                "the same RefineState instance."
            )
        self.state = state
        self.profiles = profiles
        self.config = config or EvaluationConfig()

    def evaluate(self, proposal: Proposal) -> ActionEvidence:
        """Return raw independent evidence without mutating either state."""
        structural = self._validate_shape(proposal)
        if structural.status is not GateStatus.PASS:
            return _unevaluable(proposal, structural)

        try:
            contact_delta = self.state.evaluate(proposal.changes)
            profile_update = self.profiles.evaluate(contact_delta)
        except RefineStateError as exc:
            return _unevaluable(
                proposal,
                StructuralEvidence(
                    status=GateStatus.CONFLICT,
                    reason=f"invalid_assignment_transition:{exc}",
                ),
            )

        structural = self._validate_output_bins(
            proposal,
            profile_update,
        )
        if structural.status is not GateStatus.PASS:
            return ActionEvidence(
                proposal=proposal,
                structural=structural,
                contact=_contact_evidence(
                    proposal,
                    self.state,
                    contact_delta,
                    eps=self.config.eps,
                ),
                embedding=_split_embedding_evidence(
                    proposal,
                    profile_update.transitions,
                    eps=self.config.eps,
                ),
                contig_embedding=_contig_embedding_evidence(
                    proposal,
                    self.profiles,
                ),
                bin_pair_embedding=_bin_pair_embedding_evidence(
                    proposal,
                    self.profiles,
                ),
                scg=_scg_evidence(
                    proposal,
                    profile_update,
                    self.profiles,
                ),
                contact_delta=contact_delta,
                profile_update=profile_update,
            )

        return ActionEvidence(
            proposal=proposal,
            structural=structural,
            contact=_contact_evidence(
                proposal,
                self.state,
                contact_delta,
                eps=self.config.eps,
            ),
            embedding=_split_embedding_evidence(
                proposal,
                profile_update.transitions,
                eps=self.config.eps,
            ),
            contig_embedding=_contig_embedding_evidence(
                proposal,
                self.profiles,
            ),
            bin_pair_embedding=_bin_pair_embedding_evidence(
                proposal,
                self.profiles,
            ),
            scg=_scg_evidence(
                proposal,
                profile_update,
                self.profiles,
            ),
            contact_delta=contact_delta,
            profile_update=profile_update,
        )

    def _validate_shape(self, proposal: Proposal) -> StructuralEvidence:
        if proposal.state_version != self.state.version:
            return StructuralEvidence(
                status=GateStatus.UNINFORMATIVE,
                reason=(
                    "stale_proposal:"
                    f"evaluated_at={proposal.state_version},"
                    f"current={self.state.version}"
                ),
            )
        if self.profiles.version != self.state.version:
            return StructuralEvidence(
                status=GateStatus.UNINFORMATIVE,
                reason=(
                    "profile_state_out_of_sync:"
                    f"profiles={self.profiles.version},state={self.state.version}"
                ),
            )

        shape_error = _action_shape_error(proposal, self.state)
        if shape_error is not None:
            return StructuralEvidence(
                status=GateStatus.CONFLICT,
                reason=shape_error,
            )
        return StructuralEvidence(
            status=GateStatus.PASS,
            reason="valid_action_shape",
        )

    def _validate_output_bins(
        self,
        proposal: Proposal,
        update: ActionEvidenceUpdate,
    ) -> StructuralEvidence:
        checked_bins = _retained_output_bins(proposal)
        transitions = {
            transition.bin_idx: transition
            for transition in update.transitions
        }
        for bin_idx in checked_bins:
            transition = transitions.get(bin_idx)
            if transition is None:
                return StructuralEvidence(
                    status=GateStatus.CONFLICT,
                    reason=f"missing_output_bin_transition:{bin_idx}",
                    checked_bins=checked_bins,
                )
            after = transition.after
            if after.member_count < self.config.min_bin_contigs:
                return StructuralEvidence(
                    status=GateStatus.CONFLICT,
                    reason=(
                        f"output_bin_too_small:{bin_idx}:"
                        f"contigs={after.member_count}:"
                        f"minimum={self.config.min_bin_contigs}"
                    ),
                    checked_bins=checked_bins,
                )
            if after.total_length < self.config.min_bin_length:
                return StructuralEvidence(
                    status=GateStatus.CONFLICT,
                    reason=(
                        f"output_bin_too_short:{bin_idx}:"
                        f"length={after.total_length}:"
                        f"minimum={self.config.min_bin_length}"
                    ),
                    checked_bins=checked_bins,
                )
        return StructuralEvidence(
            status=GateStatus.PASS,
            reason="valid_output_bins",
            checked_bins=checked_bins,
        )


def apply_decision(
    *,
    state: RefineState,
    profiles: EvidenceProfileState,
    decision: Decision,
) -> None:
    """Commit one accepted decision as a synchronized two-state transaction."""
    if decision.status is not DecisionStatus.ACCEPT:
        raise ValueError("Only an accepted decision can be applied.")
    contact_delta = decision.evidence.contact_delta
    profile_update = decision.evidence.profile_update
    if contact_delta is None or profile_update is None:
        raise ValueError("Accepted decision is missing local state deltas.")

    state.apply(contact_delta)
    try:
        profiles.apply(profile_update)
    except Exception:
        state.rollback(contact_delta)
        raise


def rollback_decision(
    *,
    state: RefineState,
    profiles: EvidenceProfileState,
    decision: Decision,
) -> None:
    """Rollback the most recently applied accepted decision exactly."""
    if decision.status is not DecisionStatus.ACCEPT:
        raise ValueError("Only an accepted decision can be rolled back.")
    contact_delta = decision.evidence.contact_delta
    profile_update = decision.evidence.profile_update
    if contact_delta is None or profile_update is None:
        raise ValueError("Accepted decision is missing local state deltas.")
    profiles.rollback(profile_update)
    state.rollback(contact_delta)


def _unevaluable(
    proposal: Proposal,
    structural: StructuralEvidence,
) -> ActionEvidence:
    return ActionEvidence(
        proposal=proposal,
        structural=structural,
        contact=None,
        embedding=None,
        contig_embedding=None,
        bin_pair_embedding=None,
        scg=None,
        contact_delta=None,
        profile_update=None,
    )


def _action_shape_error(
    proposal: Proposal,
    state: RefineState,
) -> str | None:
    changes = proposal.changes
    action_type = proposal.action_type

    if action_type is ActionType.SPLIT:
        if len(proposal.source_bins) != 1:
            return "split_requires_one_source_bin"
        source_bin = proposal.source_bins[0]
        if any(change.old_bin_idx != source_bin for change in changes):
            return "split_changes_must_share_source_bin"
        if any(change.new_bin_idx < state.n_bins for change in changes):
            return "split_targets_must_be_new_bins"
        if not proposal.target_bins:
            return "split_requires_new_child_bin"
        return None

    if action_type is ActionType.MERGE:
        if len(proposal.source_bins) != 1 or len(proposal.target_bins) != 1:
            return "merge_requires_one_source_and_one_target_bin"
        source_bin = proposal.source_bins[0]
        target_bin = proposal.target_bins[0]
        if source_bin == target_bin or target_bin >= state.n_bins:
            return "merge_requires_distinct_existing_bins"
        if any(
            change.old_bin_idx != source_bin
            or change.new_bin_idx != target_bin
            for change in changes
        ):
            return "merge_changes_do_not_match_source_target"
        if set(proposal.contig_indices) != set(state.bin_members(source_bin)):
            return "merge_must_move_every_source_member"
        return None

    if action_type is ActionType.RECRUIT:
        if len(changes) != 1:
            return "recruit_requires_one_contig"
        change = changes[0]
        if (
            change.old_bin_idx != -1
            or change.new_bin_idx < 0
            or change.new_bin_idx >= state.n_bins
        ):
            return "recruit_requires_unassigned_to_existing_bin_change"
        return None

    return f"unsupported_action_type:{action_type.value}"


def _retained_output_bins(proposal: Proposal) -> tuple[int, ...]:
    if proposal.action_type is ActionType.MERGE:
        return proposal.target_bins
    return tuple(sorted(set(proposal.source_bins) | set(proposal.target_bins)))


def _contact_evidence(
    proposal: Proposal,
    state: RefineState,
    delta: ActionDelta,
    *,
    eps: float,
) -> ContactEvidence:
    transitions = tuple(
        BinContactTransition(
            bin_idx=bin_idx,
            numerator_before=delta.numerator_before[offset],
            numerator_after=delta.numerator_after[offset],
            denominator_before=delta.denominator_before[offset],
            denominator_after=delta.denominator_after[offset],
            coherence_before=delta.coherence_before[offset],
            coherence_after=delta.coherence_after[offset],
        )
        for offset, bin_idx in enumerate(delta.affected_bins)
    )

    single_contig = None
    if proposal.action_type is ActionType.RECRUIT:
        single_contig = _single_contig_contact(proposal, state, eps=eps)

    merge_pair_support = None
    merge_normalized_support = None
    merge_edge_count = 0
    if proposal.action_type is ActionType.MERGE:
        (
            merge_pair_support,
            merge_normalized_support,
            merge_edge_count,
        ) = _merge_pair_support(
            proposal,
            state,
            delta,
            eps=eps,
        )

    split_separation = None
    split_within = None
    split_cross = None
    if proposal.action_type is ActionType.SPLIT:
        split_separation, split_within, split_cross = _split_separation(
            proposal,
            state,
            delta,
            eps=eps,
        )

    return ContactEvidence(
        transitions=transitions,
        affected_edge_count=len(delta.affected_edges),
        single_contig=single_contig,
        merge_pair_support=merge_pair_support,
        merge_normalized_support=merge_normalized_support,
        merge_supporting_edge_count=merge_edge_count,
        split_separation=split_separation,
        split_within_support=split_within,
        split_cross_support=split_cross,
    )


def _single_contig_contact(
    proposal: Proposal,
    state: RefineState,
    *,
    eps: float,
) -> ContigContactEvidence:
    change = proposal.changes[0]
    supports = state.contact_index.contig_bin_support(
        change.contig_idx,
        state.assignment,
    )
    total = sum(item.support for item in supports.values())
    source = supports.get(change.old_bin_idx)
    target = supports.get(change.new_bin_idx)
    source_support = 0.0 if source is None else float(source.support)
    target_support = 0.0 if target is None else float(target.support)
    other_supports = [
        item.support
        for bin_idx, item in supports.items()
        if bin_idx != change.new_bin_idx
    ]
    runner_up = max(other_supports, default=0.0)
    source_share = (
        source_support / (total + eps)
        if change.old_bin_idx >= 0 and total > 0.0
        else None
    )
    target_share = (
        target_support / (total + eps)
        if change.new_bin_idx >= 0 and total > 0.0
        else None
    )
    target_margin = (
        target_share - runner_up / (total + eps)
        if target_share is not None
        else None
    )
    return ContigContactEvidence(
        contig_idx=change.contig_idx,
        source_bin_idx=change.old_bin_idx,
        target_bin_idx=change.new_bin_idx,
        source_support=source_support,
        target_support=target_support,
        runner_up_support=float(runner_up),
        source_share=source_share,
        target_share=target_share,
        target_margin=target_margin,
        source_edge_count=0 if source is None else int(source.edge_count),
        target_edge_count=0 if target is None else int(target.edge_count),
    )


def _merge_pair_support(
    proposal: Proposal,
    state: RefineState,
    delta: ActionDelta,
    *,
    eps: float,
) -> tuple[float, float, int]:
    source_bin = proposal.source_bins[0]
    target_bin = proposal.target_bins[0]
    support = 0.0
    edge_count = 0
    for update in delta.edge_updates:
        masses = dict(update.before)
        source_mass = float(masses.get(source_bin, 0.0))
        target_mass = float(masses.get(target_bin, 0.0))
        if source_mass <= 0.0 or target_mass <= 0.0:
            continue
        reliability = float(
            state.contact_index.edge_reliability[update.edge_idx]
        )
        support += reliability * source_mass * target_mass
        edge_count += 1
    denominator = math.sqrt(
        float(state.contact_denominator[source_bin])
        * float(state.contact_denominator[target_bin])
        + float(eps)
    )
    normalized_support = (
        float(support) / denominator
        if denominator > 0.0
        else 0.0
    )
    return float(support), float(normalized_support), edge_count


def _split_separation(
    proposal: Proposal,
    state: RefineState,
    delta: ActionDelta,
    *,
    eps: float,
) -> tuple[float | None, float, float]:
    child_bins = set(proposal.source_bins) | set(proposal.target_bins)
    within = 0.0
    cross = 0.0
    for update in delta.edge_updates:
        masses = [
            float(mass)
            for bin_idx, mass in update.after
            if bin_idx in child_bins and float(mass) > 0.0
        ]
        if not masses:
            continue
        reliability = float(
            state.contact_index.edge_reliability[update.edge_idx]
        )
        within += reliability * sum(mass * mass for mass in masses)
        cross += reliability * sum(
            masses[left] * masses[right]
            for left in range(len(masses))
            for right in range(left + 1, len(masses))
        )
    denominator = within + 2.0 * cross
    separation = (
        within / (denominator + eps)
        if denominator > 0.0
        else None
    )
    return separation, float(within), float(cross)


def _split_embedding_evidence(
    proposal: Proposal,
    transitions: Iterable[BinEvidenceTransition],
    *,
    eps: float,
) -> MetricEvidence | None:
    if proposal.action_type is not ActionType.SPLIT:
        return None
    before_values: list[tuple[float, int]] = []
    after_values: list[tuple[float, int]] = []
    for transition in transitions:
        before_value = transition.before.embedding.dispersion
        after_value = transition.after.embedding.dispersion
        before_count = transition.before.embedding.observed_count
        after_count = transition.after.embedding.observed_count

        if before_value is not None and before_count > 0:
            before_values.append((float(before_value), int(before_count)))
        if after_value is not None and after_count > 0:
            after_values.append((float(after_value), int(after_count)))

    before = _weighted_mean(before_values)
    after = _weighted_mean(after_values)
    relative_gain = (
        (before - after) / (abs(before) + eps)
        if before is not None and after is not None
        else None
    )
    return MetricEvidence(
        before=before,
        after=after,
        relative_gain=relative_gain,
        observed_before=sum(count for _value, count in before_values),
        observed_after=sum(count for _value, count in after_values),
    )


def _contig_embedding_evidence(
    proposal: Proposal,
    profiles: EvidenceProfileState,
) -> ContigEmbeddingEvidence | None:
    if proposal.action_type is not ActionType.RECRUIT:
        return None
    change = proposal.changes[0]
    target_profile = profiles.profile(change.new_bin_idx)
    metrics = profiles.contig_bin_metrics(
        change.contig_idx,
        change.new_bin_idx,
    )
    return ContigEmbeddingEvidence(
        contig_idx=change.contig_idx,
        target_bin_idx=change.new_bin_idx,
        distance=metrics.embedding_distance,
        target_radius=target_profile.embedding.radius,
    )


def _bin_pair_embedding_evidence(
    proposal: Proposal,
    profiles: EvidenceProfileState,
) -> BinPairEmbeddingEvidence | None:
    if proposal.action_type is not ActionType.MERGE:
        return None
    source_bin = proposal.source_bins[0]
    target_bin = proposal.target_bins[0]
    source = profiles.profile(source_bin).embedding
    target = profiles.profile(target_bin).embedding
    centroid_distance = None
    if source.centroid is not None and target.centroid is not None:
        centroid_distance = float(
            math.sqrt(
                float(
                    ((source.centroid - target.centroid) ** 2).sum()
                )
            )
        )
    return BinPairEmbeddingEvidence(
        source_bin_idx=source_bin,
        target_bin_idx=target_bin,
        centroid_distance=centroid_distance,
        source_radius=source.radius,
        target_radius=target.radius,
    )


def _weighted_mean(values: Iterable[tuple[float, int]]) -> float | None:
    entries = tuple(values)
    total_weight = sum(weight for _value, weight in entries)
    if total_weight <= 0:
        return None
    return float(
        sum(value * weight for value, weight in entries) / total_weight
    )


def _scg_evidence(
    proposal: Proposal,
    update: ActionEvidenceUpdate,
    profiles: EvidenceProfileState,
) -> ScgEvidence:
    before_burden = sum(
        transition.before.scg.duplicate_burden
        for transition in update.transitions
    )
    after_burden = sum(
        transition.after.scg.duplicate_burden
        for transition in update.transitions
    )
    new_duplicates = tuple(
        sorted(
            {
                marker
                for transition in update.transitions
                for marker in (
                    set(transition.after.scg.duplicate_marker_ids)
                    - set(transition.before.scg.duplicate_marker_ids)
                )
            }
        )
    )
    resolved_duplicates = tuple(
        sorted(
            {
                marker
                for transition in update.transitions
                for marker in (
                    set(transition.before.scg.duplicate_marker_ids)
                    - set(transition.after.scg.duplicate_marker_ids)
                )
            }
        )
    )
    observed_markers = {
        marker_idx
        for transition in update.transitions
        for profile in (transition.before.scg, transition.after.scg)
        for marker_idx, count in enumerate(profile.marker_contig_counts)
        if count > 0
    }
    duplicate_change = int(after_burden - before_burden)

    moved_contig_has_scg = any(
        profiles.inputs.scg_markers[change.contig_idx]
        for change in proposal.changes
    )
    single_contig_action = proposal.action_type is ActionType.RECRUIT

    if single_contig_action and not moved_contig_has_scg:
        status = GateStatus.UNINFORMATIVE
        reason = "changed_contig_has_no_scg"
    elif not observed_markers:
        status = GateStatus.UNINFORMATIVE
        reason = "no_scg_information"
    elif (
        proposal.action_type is ActionType.SPLIT
        and duplicate_change >= 0
    ):
        status = GateStatus.CONFLICT
        reason = "split_did_not_reduce_scg_duplication"
    elif (
        proposal.action_type
        in {
            ActionType.MERGE,
            ActionType.RECRUIT,
        }
        and (duplicate_change > 0 or new_duplicates)
    ):
        status = GateStatus.CONFLICT
        reason = "target_bin_gained_scg_duplication"
    else:
        status = GateStatus.PASS
        reason = (
            "scg_duplication_reduced"
            if duplicate_change < 0 or resolved_duplicates
            else "scg_nonconflicting"
        )
    return ScgEvidence(
        status=status,
        reason=reason,
        duplicate_burden_before=int(before_burden),
        duplicate_burden_after=int(after_burden),
        duplicate_change=duplicate_change,
        new_duplicate_markers=new_duplicates,
        resolved_duplicate_markers=resolved_duplicates,
        observed_marker_count=len(observed_markers),
    )
