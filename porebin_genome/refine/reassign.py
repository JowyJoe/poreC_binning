"""Conservative target-bin reassignment and filtering for refine MVP."""

from __future__ import annotations

from porebin_genome.refine.models import BinFeatureProfile, ReassignCandidate, ReassignDecision, RefineState, SupportSummary
from porebin_genome.refine.support import compute_assignment_confidence, evaluate_feature_gate, runner_up_margin


REASSIGN_MARGIN_MIN = 0.20
REASSIGN_CONFIDENCE_MIN = 0.70
FILTER_CONFIDENCE_MIN = 0.40


def generate_reassign_candidates(
    *,
    state: RefineState,
    support_summary: SupportSummary,
    profiles: dict[str, BinFeatureProfile],
    feature_matrix: "object",
    contig_name_to_idx: dict[str, int],
) -> list[ReassignCandidate]:
    """Generate conservative target-bin move candidates for weakly assigned contigs."""
    candidates: list[ReassignCandidate] = []
    for contig_id, source_bin in sorted(state.current_assignment.items()):
        scores = dict(support_summary.support_by_contig_bin.get(contig_id, {}))
        if not scores:
            continue
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        target_bin = str(ordered[0][0])
        target_bin_support = float(ordered[0][1])
        if target_bin == source_bin:
            if len(ordered) < 2:
                continue
            target_bin = str(ordered[1][0])
            target_bin_support = float(ordered[1][1])
        current_bin_support = float(scores.get(source_bin, 0.0))
        if target_bin_support <= current_bin_support:
            continue
        secondary_scores = [
            float(score)
            for bin_id, score in ordered
            if str(bin_id) != target_bin
        ]
        runner_up_support = float(secondary_scores[0]) if secondary_scores else 0.0
        margin = runner_up_margin(target_bin_support=target_bin_support, runner_up_support=runner_up_support)
        feature_gate, _feature_score, feature_note = evaluate_feature_gate(
            contig_id=contig_id,
            target_bin=target_bin,
            feature_matrix=feature_matrix,
            contig_name_to_idx=contig_name_to_idx,
            profiles=profiles,
            coverage_by_contig=state.coverage_by_contig,
        )
        non_worsening_gate = _evaluate_non_worsening_gate(
            contig_id=contig_id,
            target_bin=target_bin,
            profiles=profiles,
            feature_matrix=feature_matrix,
            contig_name_to_idx=contig_name_to_idx,
        )
        candidates.append(
            ReassignCandidate(
                contig_id=contig_id,
                source_bin=source_bin,
                target_bin=target_bin,
                current_bin_support=current_bin_support,
                target_bin_support=target_bin_support,
                runner_up_support=runner_up_support,
                runner_up_margin=margin,
                feature_gate=feature_gate,
                non_worsening_gate=non_worsening_gate,
                feature_note=feature_note,
            )
        )
    return candidates


def evaluate_reassign_candidate(candidate: ReassignCandidate) -> ReassignDecision:
    """Accept or reject a target-bin move candidate with conservative gates."""
    confidence = compute_assignment_confidence(
        target_bin_support=candidate.target_bin_support,
        runner_up_support=candidate.runner_up_support,
        feature_gate=candidate.feature_gate,
    )
    if candidate.runner_up_margin < REASSIGN_MARGIN_MIN:
        return ReassignDecision(
            candidate=candidate,
            accepted=False,
            reason="target_bin_margin_too_small",
            confidence=float(confidence),
            note=f"runner_up_margin={candidate.runner_up_margin:.3f}",
        )
    if not candidate.feature_gate:
        return ReassignDecision(
            candidate=candidate,
            accepted=False,
            reason="target_bin_feature_gate_failed",
            confidence=float(confidence),
            note=candidate.feature_note,
        )
    if not candidate.non_worsening_gate:
        return ReassignDecision(
            candidate=candidate,
            accepted=False,
            reason="target_bin_non_worsening_gate_failed",
            confidence=float(confidence),
            note="adding contig would worsen target feature dispersion",
        )
    if confidence < REASSIGN_CONFIDENCE_MIN:
        return ReassignDecision(
            candidate=candidate,
            accepted=False,
            reason="target_bin_assignment_confidence_too_low",
            confidence=float(confidence),
            note=f"assignment_confidence={confidence:.3f}",
        )
    return ReassignDecision(
        candidate=candidate,
        accepted=True,
        reason="move_to_target_bin",
        confidence=float(confidence),
        note=(
            f"target_bin_support={candidate.target_bin_support:.3f};"
            f"current_bin_support={candidate.current_bin_support:.3f};"
            f"runner_up_margin={candidate.runner_up_margin:.3f}"
        ),
    )


def apply_reassign_decision(*, state: RefineState, decision: ReassignDecision) -> None:
    """Apply an accepted target-bin move to the refine state."""
    if not decision.accepted:
        return
    state.assign(
        decision.candidate.contig_id,
        decision.candidate.target_bin,
        stage="reassign",
        reason="moved_to_target_bin",
    )


def filter_low_confidence_assignments(
    *,
    state: RefineState,
    support_summary: SupportSummary,
    profiles: dict[str, BinFeatureProfile],
    feature_matrix: "object",
    contig_name_to_idx: dict[str, int],
) -> list[tuple[str, str, float, str]]:
    """Return contigs that should be moved back to unbinned after refine."""
    filtered: list[tuple[str, str, float, str]] = []
    for contig_id, current_bin in sorted(list(state.current_assignment.items())):
        own_support = float(support_summary.support_by_contig_bin.get(contig_id, {}).get(current_bin, 0.0))
        runner_up_support = float(support_summary.runner_up_support.get(contig_id, 0.0))
        feature_gate, _feature_score, feature_note = evaluate_feature_gate(
            contig_id=contig_id,
            target_bin=current_bin,
            feature_matrix=feature_matrix,
            contig_name_to_idx=contig_name_to_idx,
            profiles=profiles,
            coverage_by_contig=state.coverage_by_contig,
        )
        confidence = compute_assignment_confidence(
            target_bin_support=own_support,
            runner_up_support=runner_up_support,
            feature_gate=feature_gate,
        )
        if confidence < FILTER_CONFIDENCE_MIN:
            filtered.append(
                (
                    contig_id,
                    current_bin,
                    float(confidence),
                    feature_note if not feature_gate else "assignment_confidence_below_threshold",
                )
            )
    return filtered


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
