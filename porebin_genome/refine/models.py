"""Genome-centric data models for refine MVP."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RefineState:
    """Mutable refine state for the current contig-to-bin assignment."""

    contig_lengths: dict[str, int]
    coverage_by_contig: dict[str, float]
    coarse_assignment: dict[str, str]
    current_assignment: dict[str, str]
    assignment_stage: dict[str, str]
    assignment_reason: dict[str, str]
    unbinned_rows: dict[str, "UnbinnedRow"] = field(default_factory=dict)
    next_bin_id: int = 0

    def bin_to_contigs(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for contig_id, bin_id in self.current_assignment.items():
            if not bin_id:
                continue
            out[str(bin_id)].append(contig_id)
        return dict(out)

    def allocate_bin_id(self) -> str:
        value = str(self.next_bin_id)
        self.next_bin_id += 1
        return value

    def assign(self, contig_id: str, bin_id: str, *, stage: str, reason: str) -> None:
        self.current_assignment[str(contig_id)] = str(bin_id)
        self.assignment_stage[str(contig_id)] = str(stage)
        self.assignment_reason[str(contig_id)] = str(reason)
        self.unbinned_rows.pop(str(contig_id), None)

    def unassign(
        self,
        contig_id: str,
        *,
        stage: str,
        reason: str,
        source_bin: str = "",
        note: str = "",
    ) -> None:
        self.current_assignment.pop(str(contig_id), None)
        self.assignment_stage[str(contig_id)] = str(stage)
        self.assignment_reason[str(contig_id)] = str(reason)
        self.unbinned_rows[str(contig_id)] = UnbinnedRow(
            contig_id=str(contig_id),
            stage=str(stage),
            reason=str(reason),
            source_bin=str(source_bin),
            note=str(note),
        )


@dataclass(frozen=True)
class BinFeatureProfile:
    """Feature centroid and dispersion profile for one bin."""

    bin_id: str
    members: tuple[str, ...]
    centroid: "object"
    distance_median: float
    distance_mad: float
    feature_dispersion: float
    median_coverage: Optional[float]
    coverage_mad: Optional[float]


@dataclass(frozen=True)
class BinSnapshot:
    """Bin-level refine metrics used for suspect detection and QC."""

    bin_id: str
    members: tuple[str, ...]
    n_contigs: int
    total_length: int
    median_coverage: Optional[float]
    coverage_dispersion: Optional[float]
    feature_dispersion: float
    contact_coherence: float
    scg_status: str
    scg_duplicate_marker_count: int
    contact_components: int
    low_support_ratio: float
    suspect_flag: bool
    suspect_reasons: tuple[str, ...]
    refine_status: str


@dataclass(frozen=True)
class SuspectBin:
    """A bin flagged for possible split or closer inspection."""

    bin_id: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RefinedAssignmentRow:
    """One row in final/bins.refined.tsv."""

    contig_id: str
    bin_id: str
    assignment_stage: str
    assignment_reason: str


@dataclass(frozen=True)
class UnbinnedRow:
    """One row in final/unbinned.tsv."""

    contig_id: str
    stage: str
    reason: str
    source_bin: str
    note: str


@dataclass(frozen=True)
class BinQcRow:
    """One row in final/bin_qc.tsv."""

    bin_id: str
    n_contigs: int
    total_length: int
    median_coverage: str
    contact_coherence: float
    scg_status: str
    scg_duplicate_marker_count: int
    suspect_flag: bool
    refine_status: str
    notes: str


@dataclass(frozen=True)
class RefineActionRow:
    """One row in final/refine_actions.tsv."""

    action_type: str
    contig_id: str
    bin_id: str
    source_bin: str
    target_bin: str
    reason: str
    accepted: bool
    confidence: float
    delta_contact: float | None
    scg_status: str
    note: str


@dataclass(frozen=True)
class SplitCandidate:
    """A local split proposal for one suspect bin."""

    source_bin: str
    groups: tuple[tuple[str, ...], ...]
    method: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SplitDecision:
    """Accepted or rejected split decision."""

    candidate: SplitCandidate
    accepted: bool
    new_bin_id: str
    reason: str
    confidence: float
    note: str


@dataclass(frozen=True)
class ReassignCandidate:
    """A proposal to move one boundary contig to a target bin."""

    contig_id: str
    source_bin: str
    target_bin: str
    current_bin_support: float
    target_bin_support: float
    runner_up_support: float
    target_support_edges: int
    feature_gate: bool
    feature_note: str


@dataclass(frozen=True)
class ReassignDecision:
    """Accepted or rejected target-bin move."""

    candidate: ReassignCandidate
    accepted: bool
    move_to_unbinned: bool
    reason: str
    confidence: float
    note: str


@dataclass(frozen=True)
class RecruitCandidate:
    """A proposal to recruit one unbinned contig into a target bin."""

    contig_id: str
    target_bin: str
    target_bin_support: float
    runner_up_support: float
    target_support_edges: int
    feature_gate: bool
    feature_note: str


@dataclass(frozen=True)
class RecruitDecision:
    """Accepted or rejected recruitment decision."""

    candidate: RecruitCandidate
    accepted: bool
    reason: str
    confidence: float
    note: str


@dataclass(frozen=True)
class MergeCandidate:
    """A conservative proposal to merge two coarse-split bins."""

    source_bin: str
    target_bin: str
    cross_support: float
    support_edges: int
    feature_compatible: bool
    coverage_compatible: bool


@dataclass(frozen=True)
class MergeDecision:
    """Accepted or rejected merge decision."""

    candidate: MergeCandidate
    accepted: bool
    reason: str
    confidence: float
    note: str
