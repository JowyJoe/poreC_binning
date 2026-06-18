"""Ordered, interpretable policy for refinement action evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass

from porebin_genome.refinement.actions import (
    ActionEvidence,
    ActionType,
    Decision,
    DecisionStatus,
    GateResult,
    GateStatus,
    MetricEvidence,
)


@dataclass(frozen=True)
class PolicyConfig:
    """Small set of direct thresholds used by the three-action policy."""

    split_min_separation: float = 0.5
    merge_min_pair_support: float = 0.0
    merge_min_supporting_edges: int = 2
    recruit_min_target_share: float = 0.5
    recruit_min_margin: float = 0.0
    recruit_min_supporting_edges: int = 2
    max_embedding_worsening: float = 0.0
    require_embedding: bool = True
    eps: float = 1e-12

    def __post_init__(self) -> None:
        for name in (
            "split_min_separation",
            "recruit_min_target_share",
            "recruit_min_margin",
        ):
            numeric = float(getattr(self, name))
            if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                raise ValueError(f"{name} must be between zero and one.")
        pair_support = float(self.merge_min_pair_support)
        if not math.isfinite(pair_support) or pair_support < 0.0:
            raise ValueError(
                "merge_min_pair_support must be finite and non-negative."
            )
        if int(self.merge_min_supporting_edges) < 1:
            raise ValueError("merge_min_supporting_edges must be positive.")
        if int(self.recruit_min_supporting_edges) < 1:
            raise ValueError("recruit_min_supporting_edges must be positive.")
        for name in ("max_embedding_worsening",):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if not math.isfinite(float(self.eps)) or float(self.eps) <= 0.0:
            raise ValueError("eps must be a finite positive number.")


class ActionPolicy:
    """Decide actions using ordered gates rather than one opaque score."""

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()

    def decide(self, evidence: ActionEvidence) -> Decision:
        gates: list[GateResult] = []

        structural = GateResult(
            name="structural",
            status=evidence.structural.status,
            reason=evidence.structural.reason,
        )
        gates.append(structural)
        if structural.status is GateStatus.CONFLICT:
            return _decision(
                evidence,
                gates,
                DecisionStatus.REJECT,
                "structural_invalid",
                structural.reason,
            )
        if structural.status is GateStatus.UNINFORMATIVE:
            return _decision(
                evidence,
                gates,
                DecisionStatus.ABSTAIN,
                "stale_or_unsynchronized",
                structural.reason,
            )

        scg = _scg_gate(evidence)
        gates.append(scg)
        if scg.status is GateStatus.CONFLICT:
            return _decision(
                evidence,
                gates,
                DecisionStatus.REJECT,
                "scg_conflict",
                scg.reason,
            )

        contact = self._contact_gate(evidence)
        gates.append(contact)
        if contact.status is not GateStatus.PASS:
            return _decision(
                evidence,
                gates,
                DecisionStatus.ABSTAIN,
                "contact_insufficient",
                contact.reason,
            )

        embedding = self._embedding_gate(evidence)
        gates.append(embedding)
        if embedding.status is not GateStatus.PASS:
            return _decision(
                evidence,
                gates,
                DecisionStatus.ABSTAIN,
                "embedding_unsupported",
                embedding.reason,
            )

        return _decision(
            evidence,
            gates,
            DecisionStatus.ACCEPT,
            "accepted",
            "All required action gates passed.",
        )

    def _contact_gate(self, evidence: ActionEvidence) -> GateResult:
        contact = evidence.contact
        if contact is None:
            return GateResult(
                "contact",
                GateStatus.UNINFORMATIVE,
                "contact_evidence_unavailable",
            )

        action_type = evidence.proposal.action_type
        if action_type is ActionType.SPLIT:
            threshold = self.config.split_min_separation
            if contact.split_separation is None:
                return GateResult(
                    "contact",
                    GateStatus.UNINFORMATIVE,
                    "split_has_no_informative_contact",
                )
            if contact.split_separation + self.config.eps < threshold:
                return GateResult(
                    "contact",
                    GateStatus.CONFLICT,
                    "split_children_remain_contact_mixed",
                    contact.split_separation,
                )
            return GateResult(
                "contact",
                GateStatus.PASS,
                "split_children_contact_separated",
                contact.split_separation,
            )

        if action_type is ActionType.MERGE:
            threshold = self.config.merge_min_pair_support
            if (
                contact.merge_pair_support is None
                or contact.merge_supporting_edge_count
                < self.config.merge_min_supporting_edges
                or contact.merge_pair_support <= threshold
            ):
                return GateResult(
                    "contact",
                    GateStatus.CONFLICT,
                    "merge_lacks_cross_bin_hyperedge_support",
                    contact.merge_pair_support,
                )
            return GateResult(
                "contact",
                GateStatus.PASS,
                "merge_has_cross_bin_hyperedge_support",
                contact.merge_pair_support,
            )

        single = contact.single_contig
        if single is None:
            return GateResult(
                "contact",
                GateStatus.UNINFORMATIVE,
                "single_contig_contact_evidence_unavailable",
            )

        if action_type is ActionType.RECRUIT:
            threshold = self.config.recruit_min_margin
            target_bin = evidence.proposal.target_bins[0]
            transition = contact.transition(target_bin)
            if single.target_margin is None or transition is None:
                return GateResult(
                    "contact",
                    GateStatus.UNINFORMATIVE,
                    "recruit_target_contact_unavailable",
                )
            if (
                single.target_support <= 0.0
                or single.target_edge_count
                < self.config.recruit_min_supporting_edges
                or single.target_share is None
                or single.target_share + self.config.eps
                < self.config.recruit_min_target_share
                or single.target_margin + self.config.eps < threshold
                or transition.coherence_change
                < -self.config.eps
            ):
                return GateResult(
                    "contact",
                    GateStatus.CONFLICT,
                    "recruit_target_contact_requirements_failed",
                    single.target_margin,
                )
            return GateResult(
                "contact",
                GateStatus.PASS,
                "recruit_target_contact_requirements_passed",
                single.target_margin,
            )

        return GateResult(
            "contact",
            GateStatus.UNINFORMATIVE,
            f"unsupported_action_type:{action_type.value}",
        )

    def _embedding_gate(self, evidence: ActionEvidence) -> GateResult:
        if evidence.proposal.action_type is ActionType.MERGE:
            direct = evidence.bin_pair_embedding
            if (
                direct is None
                or direct.centroid_distance is None
                or direct.combined_radius is None
            ):
                return GateResult(
                    "embedding",
                    (
                        GateStatus.CONFLICT
                        if self.config.require_embedding
                        else GateStatus.UNINFORMATIVE
                    ),
                    "merge_embedding_evidence_unavailable",
                )
            if (
                direct.centroid_distance
                > direct.combined_radius + self.config.eps
            ):
                return GateResult(
                    "embedding",
                    GateStatus.CONFLICT,
                    "merge_hgvae_regions_do_not_overlap",
                    direct.centroid_distance,
                )
            return GateResult(
                "embedding",
                GateStatus.PASS,
                "merge_hgvae_regions_overlap",
                direct.centroid_distance,
            )
        if evidence.proposal.action_type is ActionType.RECRUIT:
            direct = evidence.contig_embedding
            if (
                direct is None
                or direct.distance is None
                or direct.target_radius is None
            ):
                return GateResult(
                    "embedding",
                    (
                        GateStatus.CONFLICT
                        if self.config.require_embedding
                        else GateStatus.UNINFORMATIVE
                    ),
                    "recruit_embedding_evidence_unavailable",
                )
            if direct.distance > direct.target_radius + self.config.eps:
                return GateResult(
                    "embedding",
                    GateStatus.CONFLICT,
                    "recruit_outside_target_embedding_radius",
                    direct.distance,
                )
            return GateResult(
                "embedding",
                GateStatus.PASS,
                "recruit_inside_target_embedding_radius",
                direct.distance,
            )
        return _metric_gate(
            "embedding",
            evidence.embedding,
            max_worsening=self.config.max_embedding_worsening,
            required=self.config.require_embedding,
        )


def _scg_gate(evidence: ActionEvidence) -> GateResult:
    scg = evidence.scg
    if scg is None:
        return GateResult(
            "scg",
            GateStatus.UNINFORMATIVE,
            "scg_evidence_unavailable",
        )
    marker_detail = ",".join(scg.new_duplicate_markers)
    return GateResult(
        name="scg",
        status=scg.status,
        reason=scg.reason,
        value=marker_detail or scg.duplicate_change,
    )


def _metric_gate(
    name: str,
    metric: MetricEvidence | None,
    *,
    max_worsening: float,
    required: bool,
) -> GateResult:
    if metric is None or metric.relative_gain is None:
        return GateResult(
            name,
            GateStatus.CONFLICT if required else GateStatus.UNINFORMATIVE,
            f"{name}_evidence_unavailable",
        )
    if metric.relative_gain < -float(max_worsening):
        return GateResult(
            name,
            GateStatus.CONFLICT,
            f"{name}_worsened",
            metric.relative_gain,
        )
    return GateResult(
        name,
        GateStatus.PASS,
        f"{name}_compatible",
        metric.relative_gain,
    )


def _decision(
    evidence: ActionEvidence,
    gates: list[GateResult],
    status: DecisionStatus,
    reason_code: str,
    reason: str,
) -> Decision:
    return Decision(
        status=status,
        proposal=evidence.proposal,
        evidence=evidence,
        gates=tuple(gates),
        reason_code=reason_code,
        reason=reason,
    )
