"""Conservative recruitment of high-confidence unbinned contigs."""

from __future__ import annotations

from porebin_genome.refine.models import BinFeatureProfile, RecruitCandidate, RecruitDecision, RefineState, SupportSummary
from porebin_genome.refine.support import compute_assignment_confidence, evaluate_feature_gate, runner_up_margin


RECRUIT_MARGIN_MIN = 0.35
RECRUIT_CONFIDENCE_MIN = 0.75


def generate_recruit_candidates(
    *,
    state: RefineState,
    support_summary: SupportSummary,
    profiles: dict[str, BinFeatureProfile],
    feature_matrix: "object",
    contig_name_to_idx: dict[str, int],
) -> list[RecruitCandidate]:
    """Generate recruitment candidates for currently unbinned contigs."""
    candidates: list[RecruitCandidate] = []
    assigned_contigs = set(state.current_assignment.keys())
    for contig_id in sorted(state.contig_lengths.keys()):
        if contig_id in assigned_contigs:
            continue
        scores = dict(support_summary.support_by_contig_bin.get(contig_id, {}))
        if not scores:
            continue
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        target_bin = str(ordered[0][0])
        target_bin_support = float(ordered[0][1])
        runner_up_support = float(ordered[1][1]) if len(ordered) > 1 else 0.0
        margin = runner_up_margin(target_bin_support=target_bin_support, runner_up_support=runner_up_support)
        feature_gate, _feature_score, feature_note = evaluate_feature_gate(
            contig_id=contig_id,
            target_bin=target_bin,
            feature_matrix=feature_matrix,
            contig_name_to_idx=contig_name_to_idx,
            profiles=profiles,
            coverage_by_contig=state.coverage_by_contig,
        )
        candidates.append(
            RecruitCandidate(
                contig_id=contig_id,
                target_bin=target_bin,
                target_bin_support=target_bin_support,
                runner_up_support=runner_up_support,
                runner_up_margin=margin,
                feature_gate=feature_gate,
                feature_note=feature_note,
            )
        )
    return candidates


def evaluate_recruit_candidate(candidate: RecruitCandidate) -> RecruitDecision:
    """Accept or reject a recruitment candidate conservatively."""
    confidence = compute_assignment_confidence(
        target_bin_support=candidate.target_bin_support,
        runner_up_support=candidate.runner_up_support,
        feature_gate=candidate.feature_gate,
    )
    if candidate.target_bin_support <= 0.0:
        return RecruitDecision(
            candidate=candidate,
            accepted=False,
            reason="no_positive_target_bin_support",
            confidence=float(confidence),
            note="target_bin_support is not positive",
        )
    if candidate.runner_up_margin < RECRUIT_MARGIN_MIN:
        return RecruitDecision(
            candidate=candidate,
            accepted=False,
            reason="target_bin_margin_too_small",
            confidence=float(confidence),
            note=f"runner_up_margin={candidate.runner_up_margin:.3f}",
        )
    if not candidate.feature_gate:
        return RecruitDecision(
            candidate=candidate,
            accepted=False,
            reason="target_bin_feature_gate_failed",
            confidence=float(confidence),
            note=candidate.feature_note,
        )
    if confidence < RECRUIT_CONFIDENCE_MIN:
        return RecruitDecision(
            candidate=candidate,
            accepted=False,
            reason="target_bin_assignment_confidence_too_low",
            confidence=float(confidence),
            note=f"assignment_confidence={confidence:.3f}",
        )
    return RecruitDecision(
        candidate=candidate,
        accepted=True,
        reason="recruited_to_target_bin",
        confidence=float(confidence),
        note=(
            f"target_bin_support={candidate.target_bin_support:.3f};"
            f"runner_up_margin={candidate.runner_up_margin:.3f}"
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
