from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RefineState:
    contig_len: dict[str, int]
    min_contig_len: int
    coarse_raw: dict[str, str]
    contig_to_bin: dict[str, str]
    coverage: Optional[dict[str, float]] = None
    bin_cov_stats: Optional[dict[str, dict[str, float]]] = None
    unbinned_reason: dict[str, str] = field(default_factory=dict)

    def rebuild_bin_to_contigs(self) -> dict[str, list[str]]:
        bin_to_contigs: dict[str, list[str]] = defaultdict(list)
        for contig, bin_id in self.contig_to_bin.items():
            if not bin_id or str(bin_id).strip() == "-1":
                continue
            bin_to_contigs[str(bin_id)].append(contig)
        return dict(bin_to_contigs)


@dataclass(frozen=True)
class ResidualRecord:
    contig_name: str
    reason: str
    stage: str
    coarse_bin_id: str
    refined_status: str
    note: str


@dataclass(frozen=True)
class RefineActionRecord:
    action_id: str
    action_type: str
    target_bin: str
    source_bin: str
    affected_contigs: tuple[str, ...]
    accepted: bool
    accept_reason: str
    reject_reason: str
    qc_before_ref: str
    qc_after_ref: str
    scg_gate_used: bool
    scg_gate_result: str
    scg_gate_reason: str


@dataclass(frozen=True)
class SplitCandidate:
    source_bin: str
    primary_contigs: tuple[str, ...]
    secondary_contigs: tuple[str, ...]
    residual_contigs: tuple[str, ...]
    candidate_path: str
    candidate_reason: str
    trigger_reasons: tuple[str, ...]
    priority_source: str
    evidence_summary: str


@dataclass(frozen=True)
class SplitDecision:
    candidate: SplitCandidate
    accepted: bool
    new_bin_id: str
    accept_reason: str
    reject_reason: str
    qc_before_ref: str
    qc_after_ref: str
    scg_gate_used: bool
    scg_gate_result: str
    scg_gate_reason: str
    residual_rows: tuple[ResidualRecord, ...]


@dataclass(frozen=True)
class ReassignWindow:
    window_id: str
    left_bin: str
    right_bin: str
    window_contigs: tuple[str, ...]


@dataclass(frozen=True)
class ReassignCandidate:
    window_id: str
    contig_name: str
    current_bin: str
    sibling_bin: str
    current_support: float
    sibling_support: float
    support_margin: float
    candidate_action: str
    evidence_summary: str


@dataclass(frozen=True)
class ReassignDecision:
    candidate: ReassignCandidate
    accepted: bool
    final_action: str
    target_bin: str
    accept_reason: str
    reject_reason: str
    qc_before_ref: str
    qc_after_ref: str
    scg_gate_used: bool
    scg_gate_result: str
    scg_gate_reason: str
    residual_row: Optional[ResidualRecord]


@dataclass(frozen=True)
class RecruitCandidate:
    contig_name: str
    current_stage: str
    current_reason: str
    target_bin: str
    target_support: float
    runner_up_bin: str
    runner_up_support: float
    support_margin: float
    evidence_summary: str


@dataclass(frozen=True)
class RecruitDecision:
    candidate: RecruitCandidate
    accepted: bool
    final_action: str
    target_bin: str
    accept_reason: str
    reject_reason: str
    qc_before_ref: str
    qc_after_ref: str
    scg_gate_used: bool
    scg_gate_result: str
    scg_gate_reason: str
    residual_row: Optional[ResidualRecord]
