"""Conservative Pore-C/HG-VAE agreement recruitment for unbinned contigs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from porebin_genome.refinement.actions import (
    ActionType,
    Decision,
    DecisionStatus,
    Proposal,
)
from porebin_genome.refinement.evaluator import (
    ActionEvaluator,
    apply_decision,
)
from porebin_genome.refinement.policy import (
    ActionPolicy,
    PolicyConfig,
)
from porebin_genome.refinement.profiles import EvidenceProfileState
from porebin_genome.refinement.state import AssignmentChange, RefineState


class RecruitCandidateStatus(str, Enum):
    """Outcome of considering one contig for conservative recruitment."""

    READY = "ready"
    NOT_UNBINNED = "not_unbinned"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class RecruitGeneratorConfig:
    """Direct evidence bounds used before creating a recruit proposal."""

    min_target_share: float = 0.5
    min_supporting_edges: int = 2
    tie_tolerance: float = 1e-12
    eps: float = 1e-12

    def __post_init__(self) -> None:
        share = float(self.min_target_share)
        if not math.isfinite(share) or not 0.0 <= share <= 1.0:
            raise ValueError("min_target_share must be between zero and one.")
        if int(self.min_supporting_edges) < 1:
            raise ValueError("min_supporting_edges must be positive.")
        for name in ("tie_tolerance", "eps"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        object.__setattr__(self, "min_target_share", share)
        object.__setattr__(
            self,
            "min_supporting_edges",
            int(self.min_supporting_edges),
        )
        object.__setattr__(
            self,
            "tie_tolerance",
            float(self.tie_tolerance),
        )
        object.__setattr__(self, "eps", float(self.eps))


@dataclass(frozen=True)
class RecruitCandidate:
    """One auditable recruit proposal or an explicit abstention."""

    contig_idx: int
    status: RecruitCandidateStatus
    reason: str
    contact_target_bin_idx: int | None = None
    latent_target_bin_idx: int | None = None
    target_support: float | None = None
    runner_up_support: float | None = None
    target_share: float | None = None
    target_margin: float | None = None
    target_edge_count: int = 0
    latent_distance: float | None = None
    target_radius: float | None = None
    proposal: Proposal | None = None


@dataclass(frozen=True)
class RecruitCandidateBatch:
    """Results for all currently unbinned contigs on one state snapshot."""

    state_version: int
    eligible_bin_indices: tuple[int, ...]
    outcomes: tuple[RecruitCandidate, ...]
    candidates: tuple[RecruitCandidate, ...]


@dataclass(frozen=True)
class RecruitAttempt:
    """One candidate after optional refresh against the current state."""

    initial_candidate: RecruitCandidate
    evaluated_candidate: RecruitCandidate
    decision: Decision | None


@dataclass(frozen=True)
class RecruitPassResult:
    """One conservative pass over initially ready unbinned contigs."""

    initial_batch: RecruitCandidateBatch
    attempts: tuple[RecruitAttempt, ...]
    accepted_count: int
    final_state_version: int


def generate_recruit_candidates(
    *,
    state: RefineState,
    profiles: EvidenceProfileState,
    config: RecruitGeneratorConfig | None = None,
) -> RecruitCandidateBatch:
    """Consider every current unbinned contig using only local contacts."""
    _validate_shared_state(state=state, profiles=profiles)
    resolved = config or RecruitGeneratorConfig()
    eligible = _eligible_target_bins(state=state, profiles=profiles)
    outcomes = tuple(
        generate_recruit_candidate(
            contig_idx=contig_idx,
            state=state,
            profiles=profiles,
            config=resolved,
            eligible_bin_indices=eligible,
        )
        for contig_idx, bin_idx in enumerate(state.assignment)
        if int(bin_idx) == -1
    )
    candidates = tuple(
        sorted(
            (
                outcome
                for outcome in outcomes
                if outcome.status is RecruitCandidateStatus.READY
            ),
            key=_candidate_priority,
        )
    )
    return RecruitCandidateBatch(
        state_version=state.version,
        eligible_bin_indices=eligible,
        outcomes=outcomes,
        candidates=candidates,
    )


def generate_recruit_candidate(
    *,
    contig_idx: int,
    state: RefineState,
    profiles: EvidenceProfileState,
    config: RecruitGeneratorConfig | None = None,
    eligible_bin_indices: tuple[int, ...] | None = None,
) -> RecruitCandidate:
    """Generate one proposal only when contact and latent targets agree."""
    _validate_shared_state(state=state, profiles=profiles)
    resolved = config or RecruitGeneratorConfig()
    contig_idx = int(contig_idx)
    if contig_idx < 0 or contig_idx >= state.n_contigs:
        raise IndexError(f"Contig index out of range: {contig_idx}")
    if int(state.assignment[contig_idx]) != -1:
        return RecruitCandidate(
            contig_idx=contig_idx,
            status=RecruitCandidateStatus.NOT_UNBINNED,
            reason="contig_is_already_assigned",
        )

    eligible = (
        _eligible_target_bins(state=state, profiles=profiles)
        if eligible_bin_indices is None
        else tuple(int(value) for value in eligible_bin_indices)
    )
    if not eligible:
        return _abstain(contig_idx, "no_stable_target_bins")

    inputs = profiles.inputs
    if (
        inputs.embedding is None
        or not bool(inputs.embedding_present[contig_idx])
    ):
        return _abstain(contig_idx, "hgvae_embedding_unavailable")

    supports = state.contact_index.contig_bin_support(
        contig_idx,
        state.assignment,
    )
    if not supports:
        return _abstain(contig_idx, "no_assigned_bin_contact_support")
    ranked_supports = tuple(
        sorted(
            supports.values(),
            key=lambda item: (-item.support, item.bin_idx),
        )
    )
    contact_best = ranked_supports[0]
    tied_contact_bins = tuple(
        item.bin_idx
        for item in ranked_supports
        if math.isclose(
            item.support,
            contact_best.support,
            rel_tol=0.0,
            abs_tol=resolved.tie_tolerance,
        )
    )
    if len(tied_contact_bins) > 1:
        return _abstain(
            contig_idx,
            "contact_best_target_tied",
            contact_target=contact_best.bin_idx,
            target_support=contact_best.support,
            runner_up_support=ranked_supports[1].support,
            target_edge_count=contact_best.edge_count,
        )
    eligible_set = set(eligible)
    if contact_best.bin_idx not in eligible_set:
        return _abstain(
            contig_idx,
            "contact_best_target_not_stable",
            contact_target=contact_best.bin_idx,
            target_support=contact_best.support,
            runner_up_support=(
                ranked_supports[1].support
                if len(ranked_supports) > 1
                else 0.0
            ),
            target_edge_count=contact_best.edge_count,
        )

    total_support = float(sum(item.support for item in ranked_supports))
    runner_up = float(
        ranked_supports[1].support
        if len(ranked_supports) > 1
        else 0.0
    )
    target_share = float(
        contact_best.support / (total_support + resolved.eps)
    )
    target_margin = float(
        target_share - runner_up / (total_support + resolved.eps)
    )
    if contact_best.edge_count < resolved.min_supporting_edges:
        return _abstain(
            contig_idx,
            "contact_supporting_edges_insufficient",
            contact_target=contact_best.bin_idx,
            target_support=contact_best.support,
            runner_up_support=runner_up,
            target_share=target_share,
            target_margin=target_margin,
            target_edge_count=contact_best.edge_count,
        )
    if target_share + resolved.eps < resolved.min_target_share:
        return _abstain(
            contig_idx,
            "contact_target_share_below_minimum",
            contact_target=contact_best.bin_idx,
            target_support=contact_best.support,
            runner_up_support=runner_up,
            target_share=target_share,
            target_margin=target_margin,
            target_edge_count=contact_best.edge_count,
        )

    latent_target, latent_distance, latent_tied = _nearest_latent_target(
        contig_idx=contig_idx,
        eligible_bin_indices=eligible,
        profiles=profiles,
        tie_tolerance=resolved.tie_tolerance,
    )
    if latent_tied:
        return _abstain(
            contig_idx,
            "latent_best_target_tied",
            contact_target=contact_best.bin_idx,
            latent_target=latent_target,
            target_support=contact_best.support,
            runner_up_support=runner_up,
            target_share=target_share,
            target_margin=target_margin,
            target_edge_count=contact_best.edge_count,
            latent_distance=latent_distance,
        )
    if latent_target != contact_best.bin_idx:
        return _abstain(
            contig_idx,
            "contact_latent_target_disagree",
            contact_target=contact_best.bin_idx,
            latent_target=latent_target,
            target_support=contact_best.support,
            runner_up_support=runner_up,
            target_share=target_share,
            target_margin=target_margin,
            target_edge_count=contact_best.edge_count,
            latent_distance=latent_distance,
            target_radius=profiles.profile(
                contact_best.bin_idx
            ).embedding.radius,
        )

    target_radius = profiles.profile(latent_target).embedding.radius
    proposal = Proposal(
        proposal_id=(
            f"recruit:v{state.version}:contig{contig_idx}->bin{latent_target}"
        ),
        action_type=ActionType.RECRUIT,
        changes=(
            AssignmentChange(
                contig_idx=contig_idx,
                old_bin_idx=-1,
                new_bin_idx=latent_target,
            ),
        ),
        state_version=state.version,
    )
    return RecruitCandidate(
        contig_idx=contig_idx,
        status=RecruitCandidateStatus.READY,
        reason="porec_hgvae_target_agreement",
        contact_target_bin_idx=contact_best.bin_idx,
        latent_target_bin_idx=latent_target,
        target_support=float(contact_best.support),
        runner_up_support=runner_up,
        target_share=target_share,
        target_margin=target_margin,
        target_edge_count=int(contact_best.edge_count),
        latent_distance=latent_distance,
        target_radius=target_radius,
        proposal=proposal,
    )


def run_recruitment_pass(
    *,
    state: RefineState,
    profiles: EvidenceProfileState,
    config: RecruitGeneratorConfig | None = None,
    policy: ActionPolicy | None = None,
) -> RecruitPassResult:
    """Evaluate initially ready contigs, refreshing only after state changes."""
    resolved = config or RecruitGeneratorConfig()
    resolved_policy = policy or ActionPolicy(
        PolicyConfig(
            recruit_min_target_share=resolved.min_target_share,
            recruit_min_supporting_edges=resolved.min_supporting_edges,
        )
    )
    initial_batch = generate_recruit_candidates(
        state=state,
        profiles=profiles,
        config=resolved,
    )
    attempts: list[RecruitAttempt] = []
    accepted_count = 0
    for initial in initial_batch.candidates:
        current = initial
        proposal = current.proposal
        if proposal is None:
            continue
        if proposal.state_version != state.version:
            current = generate_recruit_candidate(
                contig_idx=initial.contig_idx,
                state=state,
                profiles=profiles,
                config=resolved,
            )
            proposal = current.proposal
        if (
            current.status is not RecruitCandidateStatus.READY
            or proposal is None
        ):
            attempts.append(
                RecruitAttempt(
                    initial_candidate=initial,
                    evaluated_candidate=current,
                    decision=None,
                )
            )
            continue

        decision = resolved_policy.decide(
            ActionEvaluator(
                state=state,
                profiles=profiles,
            ).evaluate(proposal)
        )
        attempts.append(
            RecruitAttempt(
                initial_candidate=initial,
                evaluated_candidate=current,
                decision=decision,
            )
        )
        if decision.status is DecisionStatus.ACCEPT:
            apply_decision(
                state=state,
                profiles=profiles,
                decision=decision,
            )
            accepted_count += 1
    return RecruitPassResult(
        initial_batch=initial_batch,
        attempts=tuple(attempts),
        accepted_count=accepted_count,
        final_state_version=state.version,
    )


def _eligible_target_bins(
    *,
    state: RefineState,
    profiles: EvidenceProfileState,
) -> tuple[int, ...]:
    eligible: list[int] = []
    for bin_idx in range(state.n_bins):
        profile = profiles.profile(bin_idx)
        if (
            bool(state.bin_members(bin_idx))
            and profile.scg.duplicate_burden == 0
            and profile.embedding.observed_count == profile.member_count
            and profile.embedding.centroid is not None
            and profile.embedding.radius is not None
        ):
            eligible.append(bin_idx)
    return tuple(eligible)


def _nearest_latent_target(
    *,
    contig_idx: int,
    eligible_bin_indices: tuple[int, ...],
    profiles: EvidenceProfileState,
    tie_tolerance: float,
) -> tuple[int, float, bool]:
    embedding = profiles.inputs.embedding
    if embedding is None:
        raise ValueError("HG-VAE embedding is unavailable.")
    vector = embedding[contig_idx]
    distances = tuple(
        (
            bin_idx,
            float(
                np.linalg.norm(
                    vector - profiles.profile(bin_idx).embedding.centroid
                )
            ),
        )
        for bin_idx in eligible_bin_indices
    )
    ranked = tuple(sorted(distances, key=lambda item: (item[1], item[0])))
    best_bin, best_distance = ranked[0]
    tied = sum(
        math.isclose(
            distance,
            best_distance,
            rel_tol=0.0,
            abs_tol=tie_tolerance,
        )
        for _bin_idx, distance in ranked
    ) > 1
    return best_bin, best_distance, tied


def _candidate_priority(candidate: RecruitCandidate) -> tuple[float, int, int]:
    return (
        -float(candidate.target_share or 0.0),
        -int(candidate.target_edge_count),
        int(candidate.contig_idx),
    )


def _abstain(
    contig_idx: int,
    reason: str,
    *,
    contact_target: int | None = None,
    latent_target: int | None = None,
    target_support: float | None = None,
    runner_up_support: float | None = None,
    target_share: float | None = None,
    target_margin: float | None = None,
    target_edge_count: int = 0,
    latent_distance: float | None = None,
    target_radius: float | None = None,
) -> RecruitCandidate:
    return RecruitCandidate(
        contig_idx=contig_idx,
        status=RecruitCandidateStatus.ABSTAIN,
        reason=reason,
        contact_target_bin_idx=contact_target,
        latent_target_bin_idx=latent_target,
        target_support=target_support,
        runner_up_support=runner_up_support,
        target_share=target_share,
        target_margin=target_margin,
        target_edge_count=target_edge_count,
        latent_distance=latent_distance,
        target_radius=target_radius,
    )


def _validate_shared_state(
    *,
    state: RefineState,
    profiles: EvidenceProfileState,
) -> None:
    if profiles.refine_state is not state:
        raise ValueError(
            "Recruit generator and EvidenceProfileState must share one "
            "RefineState instance."
        )
    if profiles.version != state.version:
        raise ValueError(
            "Recruit generator requires synchronized assignment and profiles."
        )
