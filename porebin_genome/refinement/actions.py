"""Stable action contracts for the replacement refinement engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from porebin_genome.refinement.profiles import ActionEvidenceUpdate
from porebin_genome.refinement.state import ActionDelta, AssignmentChange


class ActionType(str, Enum):
    """Assignment transitions supported by refinement."""

    SPLIT = "split"
    MERGE = "merge"
    RECRUIT = "recruit"


class GateStatus(str, Enum):
    """Result of one interpretable evidence gate."""

    PASS = "pass"
    CONFLICT = "conflict"
    UNINFORMATIVE = "uninformative"


class DecisionStatus(str, Enum):
    """Final policy outcome for one proposal."""

    ACCEPT = "accept"
    REJECT = "reject"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class Proposal:
    """One immutable candidate assignment transition."""

    proposal_id: str
    action_type: ActionType
    changes: tuple[AssignmentChange, ...]
    state_version: int

    def __post_init__(self) -> None:
        proposal_id = str(self.proposal_id).strip()
        if not proposal_id:
            raise ValueError("proposal_id must not be empty.")
        changes = tuple(self.changes)
        if not changes:
            raise ValueError("A proposal must contain at least one change.")
        if len({change.contig_idx for change in changes}) != len(changes):
            raise ValueError("A contig may appear only once in one proposal.")
        object.__setattr__(self, "proposal_id", proposal_id)
        object.__setattr__(self, "action_type", ActionType(self.action_type))
        object.__setattr__(self, "changes", changes)
        object.__setattr__(self, "state_version", int(self.state_version))

    @property
    def contig_indices(self) -> tuple[int, ...]:
        return tuple(change.contig_idx for change in self.changes)

    @property
    def source_bins(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    change.old_bin_idx
                    for change in self.changes
                    if change.old_bin_idx >= 0
                }
            )
        )

    @property
    def target_bins(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                {
                    change.new_bin_idx
                    for change in self.changes
                    if change.new_bin_idx >= 0
                }
            )
        )


@dataclass(frozen=True)
class StructuralEvidence:
    """Whether a proposal has a valid action shape and viable output bins."""

    status: GateStatus
    reason: str
    checked_bins: tuple[int, ...] = ()


@dataclass(frozen=True)
class BinContactTransition:
    """Exact contact-coherence transition for one affected bin."""

    bin_idx: int
    numerator_before: float
    numerator_after: float
    denominator_before: float
    denominator_after: float
    coherence_before: float
    coherence_after: float

    @property
    def coherence_change(self) -> float:
        return self.coherence_after - self.coherence_before


@dataclass(frozen=True)
class ContigContactEvidence:
    """Local hypergraph support for a single-contig action."""

    contig_idx: int
    source_bin_idx: int
    target_bin_idx: int
    source_support: float
    target_support: float
    runner_up_support: float
    source_share: float | None
    target_share: float | None
    target_margin: float | None
    source_edge_count: int
    target_edge_count: int


@dataclass(frozen=True)
class ContigEmbeddingEvidence:
    """Direct HG-VAE compatibility of one recruit with its target bin."""

    contig_idx: int
    target_bin_idx: int
    distance: float | None
    target_radius: float | None


@dataclass(frozen=True)
class BinPairEmbeddingEvidence:
    """Direct HG-VAE region compatibility for one merge pair."""

    source_bin_idx: int
    target_bin_idx: int
    centroid_distance: float | None
    source_radius: float | None
    target_radius: float | None

    @property
    def combined_radius(self) -> float | None:
        if self.source_radius is None or self.target_radius is None:
            return None
        return float(self.source_radius + self.target_radius)


@dataclass(frozen=True)
class ContactEvidence:
    """Raw action-local Pore-C evidence without a combined score."""

    transitions: tuple[BinContactTransition, ...]
    affected_edge_count: int
    single_contig: ContigContactEvidence | None = None
    merge_pair_support: float | None = None
    merge_normalized_support: float | None = None
    merge_supporting_edge_count: int = 0
    split_separation: float | None = None
    split_within_support: float | None = None
    split_cross_support: float | None = None

    def transition(self, bin_idx: int) -> BinContactTransition | None:
        for item in self.transitions:
            if item.bin_idx == int(bin_idx):
                return item
        return None


@dataclass(frozen=True)
class MetricEvidence:
    """Aggregate split transition for lower-is-better HG-VAE dispersion."""

    before: float | None
    after: float | None
    relative_gain: float | None
    observed_before: int
    observed_after: int


@dataclass(frozen=True)
class ScgEvidence:
    """Action-level transition of the fixed canonical SCG panel."""

    status: GateStatus
    reason: str
    duplicate_burden_before: int
    duplicate_burden_after: int
    duplicate_change: int
    new_duplicate_markers: tuple[str, ...]
    resolved_duplicate_markers: tuple[str, ...]
    observed_marker_count: int


@dataclass(frozen=True)
class ActionEvidence:
    """All independent evidence calculated for one proposal."""

    proposal: Proposal
    structural: StructuralEvidence
    contact: ContactEvidence | None
    embedding: MetricEvidence | None
    contig_embedding: ContigEmbeddingEvidence | None
    bin_pair_embedding: BinPairEmbeddingEvidence | None
    scg: ScgEvidence | None
    contact_delta: ActionDelta | None
    profile_update: ActionEvidenceUpdate | None

    @property
    def is_evaluable(self) -> bool:
        return (
            self.structural.status is GateStatus.PASS
            and self.contact_delta is not None
            and self.profile_update is not None
        )


@dataclass(frozen=True)
class GateResult:
    """One ordered policy check recorded for action auditing."""

    name: str
    status: GateStatus
    reason: str
    value: float | int | str | None = None


@dataclass(frozen=True)
class Decision:
    """Policy outcome retaining the complete proposal evidence."""

    status: DecisionStatus
    proposal: Proposal
    evidence: ActionEvidence
    gates: tuple[GateResult, ...]
    reason_code: str
    reason: str
