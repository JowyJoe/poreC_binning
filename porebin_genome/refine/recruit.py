"""Hyperedge-aware recruitment of currently unbinned contigs."""

from __future__ import annotations

from porebin_genome.refine.coherence import compute_bin_coherence, delta_contact_coherence
from porebin_genome.refine.contact_evidence import collect_contig_bin_evidence
from porebin_genome.refine.features import evaluate_feature_gate
from porebin_genome.refine.hyperedge import HyperedgeStore, ReliabilityFn, default_edge_reliability
from porebin_genome.refine.markers import (
    ContigScgProfile,
    evaluate_recruit_scg_transition,
)
from porebin_genome.refine.models import BinFeatureProfile, RecruitCandidate, RecruitDecision, RefineState


MIN_RECRUIT_SUPPORT_EDGES = 2


def generate_recruit_candidates(
    *,
    state: RefineState,
    store: HyperedgeStore,
    profiles: dict[str, BinFeatureProfile],
    feature_matrix: "object",
    contig_name_to_idx: dict[str, int],
    reliability_fn: ReliabilityFn | None = None,
) -> list[RecruitCandidate]:
    """Generate conservative recruitment candidates for unbinned contigs."""
    reliability_fn = reliability_fn or default_edge_reliability

    candidates: list[RecruitCandidate] = []
    assigned_contigs = set(state.current_assignment.keys())
    for contig_id in sorted(state.contig_lengths.keys()):
        if contig_id in assigned_contigs:
            continue
        evidence_by_bin = collect_contig_bin_evidence(
            store,
            state.current_assignment,
            contig_id,
            reliability_fn=reliability_fn,
        )
        if not evidence_by_bin:
            continue

        ordered = sorted(
            evidence_by_bin.values(),
            key=lambda item: (-float(item.support), str(item.bin_id)),
        )
        target = ordered[0]
        runner_up_support = float(ordered[1].support) if len(ordered) > 1 else 0.0
        feature_gate, _feature_score, feature_note = evaluate_feature_gate(
            contig_id=contig_id,
            target_bin=target.bin_id,
            feature_matrix=feature_matrix,
            contig_name_to_idx=contig_name_to_idx,
            profiles=profiles,
            coverage_by_contig=state.coverage_by_contig,
        )
        candidates.append(
            RecruitCandidate(
                contig_id=contig_id,
                target_bin=str(target.bin_id),
                target_bin_support=float(target.support),
                runner_up_support=runner_up_support,
                target_support_edges=int(target.edge_count),
                feature_gate=feature_gate,
                feature_note=feature_note,
            )
        )
    return candidates


def evaluate_recruit_candidate(
    *,
    candidate: RecruitCandidate,
    state: RefineState,
    store: HyperedgeStore,
    reliability_fn: ReliabilityFn | None = None,
    contig_scg_profiles: dict[str, ContigScgProfile] | None = None,
) -> RecruitDecision:
    """Accept a recruitment only if local contact coherence improves under constraints."""
    reliability_fn = reliability_fn or default_edge_reliability

    if candidate.target_bin_support <= 0.0:
        return RecruitDecision(
            candidate=candidate,
            accepted=False,
            reason="no_positive_target_bin_support",
            confidence=0.0,
            note="target_bin_support is not positive",
        )
    if candidate.target_support_edges < MIN_RECRUIT_SUPPORT_EDGES:
        return RecruitDecision(
            candidate=candidate,
            accepted=False,
            reason="recruit_requires_multiple_independent_hyperedges",
            confidence=0.0,
            note=f"support_edges={candidate.target_support_edges}",
        )
    if not candidate.feature_gate:
        return RecruitDecision(
            candidate=candidate,
            accepted=False,
            reason="target_bin_feature_gate_failed",
            confidence=0.0,
            note=candidate.feature_note,
        )

    before_stats = compute_bin_coherence(
        store,
        state.current_assignment,
        reliability_fn=reliability_fn,
        bins=(candidate.target_bin,),
    )
    after_assignment = dict(state.current_assignment)
    after_assignment[candidate.contig_id] = candidate.target_bin
    after_stats = compute_bin_coherence(
        store,
        after_assignment,
        reliability_fn=reliability_fn,
        bins=(candidate.target_bin,),
    )
    delta_c = float(delta_contact_coherence(before_stats, after_stats))
    if delta_c <= 0.0:
        return RecruitDecision(
            candidate=candidate,
            accepted=False,
            reason="recruit_does_not_improve_contact_coherence",
            confidence=0.0,
            note=f"delta_contact={delta_c:.3f}",
        )

    scg_note = ""
    if contig_scg_profiles is not None:
        scg_decision = evaluate_recruit_scg_transition(
            contig_id=candidate.contig_id,
            target_bin=candidate.target_bin,
            state=state,
            contig_profiles=contig_scg_profiles,
        )
        scg_note = f";scg={scg_decision.status}:{scg_decision.reason}:{scg_decision.note}"
        if scg_decision.status == "veto":
            return RecruitDecision(
                candidate=candidate,
                accepted=False,
                reason=scg_decision.reason,
                confidence=0.0,
                note=scg_decision.note,
            )

    confidence = min(1.0, max(0.0, 0.5 + delta_c))
    return RecruitDecision(
        candidate=candidate,
        accepted=True,
        reason="recruited_to_target_bin",
        confidence=float(confidence),
        note=(
            f"delta_contact={delta_c:.3f};"
            f"target_bin_support={candidate.target_bin_support:.3f};"
            f"support_edges={candidate.target_support_edges}"
            f"{scg_note}"
        ),
    )


def apply_recruit_decision(*, state: RefineState, decision: RecruitDecision) -> None:
    """Apply an accepted recruitment decision."""
    if not decision.accepted:
        return
    state.assign(
        decision.candidate.contig_id,
        decision.candidate.target_bin,
        stage="recruit",
        reason="recruited_to_target_bin",
    )
