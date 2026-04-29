"""Hyperedge-aware reassignment for boundary contigs."""

from __future__ import annotations

from porebin_genome.refine.coherence import compute_bin_coherence, delta_contact_coherence
from porebin_genome.refine.contact_evidence import collect_contig_bin_evidence
from porebin_genome.refine.features import evaluate_feature_gate
from porebin_genome.refine.hyperedge import HyperedgeStore, ReliabilityFn, default_edge_reliability
from porebin_genome.refine.markers import (
    ContigScgProfile,
    evaluate_reassign_scg_transition,
)
from porebin_genome.refine.models import BinFeatureProfile, ReassignCandidate, ReassignDecision, RefineState


def generate_reassign_candidates(
    *,
    state: RefineState,
    store: HyperedgeStore,
    profiles: dict[str, BinFeatureProfile],
    feature_matrix: "object",
    contig_name_to_idx: dict[str, int],
    reliability_fn: ReliabilityFn | None = None,
) -> list[ReassignCandidate]:
    """Generate boundary-contig move candidates using hyperedge-aware evidence."""
    reliability_fn = reliability_fn or default_edge_reliability

    candidates: list[ReassignCandidate] = []
    for contig_id, source_bin in sorted(state.current_assignment.items()):
        evidence_by_bin = collect_contig_bin_evidence(
            store,
            state.current_assignment,
            contig_id,
            reliability_fn=reliability_fn,
        )
        if not evidence_by_bin:
            continue

        current_support = float(evidence_by_bin.get(source_bin).support) if source_bin in evidence_by_bin else 0.0
        ordered = sorted(
            evidence_by_bin.values(),
            key=lambda item: (-float(item.support), str(item.bin_id)),
        )
        alternatives = [item for item in ordered if item.bin_id != source_bin and float(item.support) > 0.0]
        if not alternatives:
            continue

        target = alternatives[0]
        if float(target.support) + 1e-12 < current_support:
            continue

        runner_up_support = current_support
        for item in alternatives[1:]:
            runner_up_support = max(runner_up_support, float(item.support))

        feature_gate, _feature_score, feature_note = evaluate_feature_gate(
            contig_id=contig_id,
            target_bin=target.bin_id,
            feature_matrix=feature_matrix,
            contig_name_to_idx=contig_name_to_idx,
            profiles=profiles,
            coverage_by_contig=state.coverage_by_contig,
        )
        candidates.append(
            ReassignCandidate(
                contig_id=contig_id,
                source_bin=source_bin,
                target_bin=str(target.bin_id),
                current_bin_support=float(current_support),
                target_bin_support=float(target.support),
                runner_up_support=float(runner_up_support),
                target_support_edges=int(target.edge_count),
                feature_gate=feature_gate,
                feature_note=feature_note,
            )
        )
    return candidates


def evaluate_reassign_candidate(
    *,
    candidate: ReassignCandidate,
    state: RefineState,
    store: HyperedgeStore,
    profiles: dict[str, BinFeatureProfile],
    feature_matrix: "object",
    contig_name_to_idx: dict[str, int],
    reliability_fn: ReliabilityFn | None = None,
    contig_scg_profiles: dict[str, ContigScgProfile] | None = None,
) -> ReassignDecision:
    """Accept a reassignment only if it improves contact coherence and respects constraints."""
    reliability_fn = reliability_fn or default_edge_reliability

    if candidate.target_bin_support <= 0.0:
        return ReassignDecision(
            candidate=candidate,
            accepted=False,
            move_to_unbinned=False,
            reason="no_positive_target_bin_support",
            confidence=0.0,
            note="target_bin_support is not positive",
        )
    if not candidate.feature_gate:
        move_to_unbinned = candidate.target_bin_support <= candidate.current_bin_support + 1e-12
        return ReassignDecision(
            candidate=candidate,
            accepted=False,
            move_to_unbinned=move_to_unbinned,
            reason="target_bin_feature_gate_failed",
            confidence=0.0,
            note=candidate.feature_note,
        )
    if not _evaluate_non_worsening_gate(
        contig_id=candidate.contig_id,
        target_bin=candidate.target_bin,
        profiles=profiles,
        feature_matrix=feature_matrix,
        contig_name_to_idx=contig_name_to_idx,
    ):
        move_to_unbinned = candidate.target_bin_support <= candidate.current_bin_support + 1e-12
        return ReassignDecision(
            candidate=candidate,
            accepted=False,
            move_to_unbinned=move_to_unbinned,
            reason="target_bin_non_worsening_gate_failed",
            confidence=0.0,
            note="adding contig would worsen target feature dispersion",
        )

    affected_bins = (candidate.source_bin, candidate.target_bin)
    before_stats = compute_bin_coherence(
        store,
        state.current_assignment,
        reliability_fn=reliability_fn,
        bins=affected_bins,
    )
    after_assignment = dict(state.current_assignment)
    after_assignment[candidate.contig_id] = candidate.target_bin
    after_stats = compute_bin_coherence(
        store,
        after_assignment,
        reliability_fn=reliability_fn,
        bins=affected_bins,
    )
    delta_c = float(delta_contact_coherence(before_stats, after_stats))
    if delta_c <= 0.0:
        move_to_unbinned = candidate.target_bin_support <= candidate.current_bin_support + 1e-12
        return ReassignDecision(
            candidate=candidate,
            accepted=False,
            move_to_unbinned=move_to_unbinned,
            reason="move_does_not_improve_contact_coherence",
            confidence=0.0,
            note=f"delta_contact={delta_c:.3f}",
        )

    scg_note = ""
    if contig_scg_profiles is not None:
        scg_decision = evaluate_reassign_scg_transition(
            contig_id=candidate.contig_id,
            source_bin=candidate.source_bin,
            target_bin=candidate.target_bin,
            state=state,
            contig_profiles=contig_scg_profiles,
        )
        scg_note = f";scg={scg_decision.status}:{scg_decision.reason}:{scg_decision.note}"
        if scg_decision.status == "veto":
            return ReassignDecision(
                candidate=candidate,
                accepted=False,
                move_to_unbinned=False,
                reason=scg_decision.reason,
                confidence=0.0,
                note=scg_decision.note,
            )

    confidence = min(1.0, max(0.0, 0.5 + delta_c))
    return ReassignDecision(
        candidate=candidate,
        accepted=True,
        move_to_unbinned=False,
        reason="move_to_target_bin",
        confidence=float(confidence),
        note=(
            f"delta_contact={delta_c:.3f};"
            f"target_bin_support={candidate.target_bin_support:.3f};"
            f"current_bin_support={candidate.current_bin_support:.3f};"
            f"support_edges={candidate.target_support_edges}"
            f"{scg_note}"
        ),
    )


def apply_reassign_decision(*, state: RefineState, decision: ReassignDecision) -> None:
    """Apply an accepted target-bin move or abstain to unbinned."""
    if decision.accepted:
        state.assign(
            decision.candidate.contig_id,
            decision.candidate.target_bin,
            stage="reassign",
            reason="moved_to_target_bin",
        )
        return
    if decision.move_to_unbinned:
        state.unassign(
            decision.candidate.contig_id,
            stage="reassign",
            reason=decision.reason,
            source_bin=decision.candidate.source_bin,
            note=decision.note,
        )


def _evaluate_non_worsening_gate(
    *,
    contig_id: str,
    target_bin: str,
    profiles: dict[str, BinFeatureProfile],
    feature_matrix: "object",
    contig_name_to_idx: dict[str, int],
) -> bool:
    import numpy as np

    profile = profiles.get(target_bin)
    if profile is None or contig_id not in contig_name_to_idx:
        return True
    contig_vec = np.asarray(feature_matrix[contig_name_to_idx[contig_id], :], dtype=float)
    distances = []
    for member_id in profile.members:
        if member_id not in contig_name_to_idx:
            continue
        member_vec = np.asarray(feature_matrix[contig_name_to_idx[member_id], :], dtype=float)
        distances.append(member_vec)
    if not distances:
        return True
    member_matrix = np.vstack(distances + [contig_vec])
    centroid = member_matrix.mean(axis=0)
    new_distances = np.linalg.norm(member_matrix - centroid, axis=1)
    new_dispersion = float(np.median(new_distances))
    return new_dispersion <= float(profile.feature_dispersion + 0.15)
