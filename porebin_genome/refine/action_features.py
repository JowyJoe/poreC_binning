"""Candidate-action feature extraction for refine ML scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Collection

from porebin_genome.io.tables import write_tsv_rows
from porebin_genome.refine.hyperedge import (
    HyperedgeStore,
    ReliabilityFn,
    collect_edge_ids_for_contig,
    default_edge_reliability,
)
from porebin_genome.refine.local_graph import select_edges_for_bin
from porebin_genome.refine.models import (
    MergeCandidate,
    MergeDecision,
    ReassignCandidate,
    ReassignDecision,
    RecruitCandidate,
    RecruitDecision,
    RefineState,
    SplitCandidate,
    SplitDecision,
)


ACTION_FEATURE_COLUMNS = (
    "action_type",
    "contig_id",
    "source_bin",
    "target_bin",
    "rule_reason",
    "rule_accepted",
    "confidence",
    "delta_contact",
    "scg_status",
    "support_edge_count",
    "support_weight_sum",
    "support_reliability_median",
    "support_k_eff_median",
    "current_support",
    "target_support",
    "runner_up_support",
    "cross_support",
    "source_bin_n_contigs",
    "target_bin_n_contigs",
    "contig_length",
    "feature_status",
    "coverage_status",
    "candidate_method",
    "candidate_reasons",
)


@dataclass(frozen=True)
class ActionFeatureRow:
    """One candidate action represented as stable tabular ML features."""

    action_type: str
    contig_id: str
    source_bin: str
    target_bin: str
    rule_reason: str
    rule_accepted: bool
    confidence: float
    delta_contact: float | None
    scg_status: str
    support_edge_count: int
    support_weight_sum: float
    support_reliability_median: float
    support_k_eff_median: float
    current_support: float | None
    target_support: float | None
    runner_up_support: float | None
    cross_support: float | None
    source_bin_n_contigs: int
    target_bin_n_contigs: int
    contig_length: int
    feature_status: str
    coverage_status: str
    candidate_method: str
    candidate_reasons: str


def build_split_action_feature(
    *,
    candidate: SplitCandidate,
    decision: SplitDecision,
    state: RefineState,
    store: HyperedgeStore,
    reliability_fn: ReliabilityFn | None,
    delta_contact: float | None,
    scg_status: str,
) -> ActionFeatureRow:
    reliability_fn = reliability_fn or default_edge_reliability
    bin_members = state.bin_to_contigs()
    edge_ids = select_edges_for_bin(
        store,
        state.current_assignment,
        bin_id=candidate.source_bin,
        reliability_fn=reliability_fn,
    )
    stats = _edge_stats(store, edge_ids, reliability_fn=reliability_fn)
    child_sizes = [len(group) for group in candidate.groups]
    return ActionFeatureRow(
        action_type="split",
        contig_id="",
        source_bin=str(candidate.source_bin),
        target_bin=str(decision.new_bin_id),
        rule_reason=str(decision.reason),
        rule_accepted=bool(decision.accepted),
        confidence=float(decision.confidence),
        delta_contact=delta_contact,
        scg_status=str(scg_status),
        support_edge_count=stats.edge_count,
        support_weight_sum=stats.weight_sum,
        support_reliability_median=stats.reliability_median,
        support_k_eff_median=stats.k_eff_median,
        current_support=None,
        target_support=None,
        runner_up_support=None,
        cross_support=None,
        source_bin_n_contigs=len(bin_members.get(candidate.source_bin, ())),
        target_bin_n_contigs=max(child_sizes, default=0),
        contig_length=0,
        feature_status=_feature_status_from_reason(decision.reason),
        coverage_status=_coverage_status_from_reason(decision.reason),
        candidate_method=str(candidate.method),
        candidate_reasons=";".join(str(value) for value in candidate.reasons),
    )


def build_reassign_action_feature(
    *,
    candidate: ReassignCandidate,
    decision: ReassignDecision,
    state: RefineState,
    store: HyperedgeStore,
    reliability_fn: ReliabilityFn | None,
    delta_contact: float | None,
    scg_status: str,
) -> ActionFeatureRow:
    reliability_fn = reliability_fn or default_edge_reliability
    bin_members = state.bin_to_contigs()
    edge_ids = _support_edges_for_contig_to_bin(
        store,
        state.current_assignment,
        contig_id=candidate.contig_id,
        target_bin=candidate.target_bin,
        reliability_fn=reliability_fn,
    )
    stats = _edge_stats(store, edge_ids, reliability_fn=reliability_fn)
    return ActionFeatureRow(
        action_type="reassign",
        contig_id=str(candidate.contig_id),
        source_bin=str(candidate.source_bin),
        target_bin=str(candidate.target_bin),
        rule_reason=str(decision.reason),
        rule_accepted=bool(decision.accepted),
        confidence=float(decision.confidence),
        delta_contact=delta_contact,
        scg_status=str(scg_status),
        support_edge_count=stats.edge_count,
        support_weight_sum=stats.weight_sum,
        support_reliability_median=stats.reliability_median,
        support_k_eff_median=stats.k_eff_median,
        current_support=float(candidate.current_bin_support),
        target_support=float(candidate.target_bin_support),
        runner_up_support=float(candidate.runner_up_support),
        cross_support=None,
        source_bin_n_contigs=len(bin_members.get(candidate.source_bin, ())),
        target_bin_n_contigs=len(bin_members.get(candidate.target_bin, ())),
        contig_length=int(state.contig_lengths.get(candidate.contig_id, 0)),
        feature_status="pass" if candidate.feature_gate else str(candidate.feature_note),
        coverage_status=_coverage_status_from_reason(decision.reason),
        candidate_method="boundary_contig_move",
        candidate_reasons=str(candidate.feature_note),
    )


def build_merge_action_feature(
    *,
    candidate: MergeCandidate,
    decision: MergeDecision,
    state: RefineState,
    store: HyperedgeStore,
    reliability_fn: ReliabilityFn | None,
    delta_contact: float | None,
    scg_status: str,
) -> ActionFeatureRow:
    reliability_fn = reliability_fn or default_edge_reliability
    bin_members = state.bin_to_contigs()
    edge_ids = _support_edges_between_bins(
        store,
        state.current_assignment,
        source_bin=candidate.source_bin,
        target_bin=candidate.target_bin,
        reliability_fn=reliability_fn,
    )
    stats = _edge_stats(store, edge_ids, reliability_fn=reliability_fn)
    return ActionFeatureRow(
        action_type="merge",
        contig_id="",
        source_bin=str(candidate.source_bin),
        target_bin=str(candidate.target_bin),
        rule_reason=str(decision.reason),
        rule_accepted=bool(decision.accepted),
        confidence=float(decision.confidence),
        delta_contact=delta_contact,
        scg_status=str(scg_status),
        support_edge_count=stats.edge_count,
        support_weight_sum=stats.weight_sum,
        support_reliability_median=stats.reliability_median,
        support_k_eff_median=stats.k_eff_median,
        current_support=None,
        target_support=None,
        runner_up_support=None,
        cross_support=float(candidate.cross_support),
        source_bin_n_contigs=len(bin_members.get(candidate.source_bin, ())),
        target_bin_n_contigs=len(bin_members.get(candidate.target_bin, ())),
        contig_length=0,
        feature_status="pass" if candidate.feature_compatible else "merge_feature_incompatible",
        coverage_status="pass" if candidate.coverage_compatible else "merge_coverage_incompatible",
        candidate_method="cross_bin_hyperedge_support",
        candidate_reasons=f"support_edges={candidate.support_edges}",
    )


def build_recruit_action_feature(
    *,
    candidate: RecruitCandidate,
    decision: RecruitDecision,
    state: RefineState,
    store: HyperedgeStore,
    reliability_fn: ReliabilityFn | None,
    delta_contact: float | None,
    scg_status: str,
) -> ActionFeatureRow:
    reliability_fn = reliability_fn or default_edge_reliability
    bin_members = state.bin_to_contigs()
    edge_ids = _support_edges_for_contig_to_bin(
        store,
        state.current_assignment,
        contig_id=candidate.contig_id,
        target_bin=candidate.target_bin,
        reliability_fn=reliability_fn,
    )
    stats = _edge_stats(store, edge_ids, reliability_fn=reliability_fn)
    return ActionFeatureRow(
        action_type="recruit",
        contig_id=str(candidate.contig_id),
        source_bin="",
        target_bin=str(candidate.target_bin),
        rule_reason=str(decision.reason),
        rule_accepted=bool(decision.accepted),
        confidence=float(decision.confidence),
        delta_contact=delta_contact,
        scg_status=str(scg_status),
        support_edge_count=stats.edge_count,
        support_weight_sum=stats.weight_sum,
        support_reliability_median=stats.reliability_median,
        support_k_eff_median=stats.k_eff_median,
        current_support=None,
        target_support=float(candidate.target_bin_support),
        runner_up_support=float(candidate.runner_up_support),
        cross_support=None,
        source_bin_n_contigs=0,
        target_bin_n_contigs=len(bin_members.get(candidate.target_bin, ())),
        contig_length=int(state.contig_lengths.get(candidate.contig_id, 0)),
        feature_status="pass" if candidate.feature_gate else str(candidate.feature_note),
        coverage_status=_coverage_status_from_reason(decision.reason),
        candidate_method="unbinned_contig_recruitment",
        candidate_reasons=str(candidate.feature_note),
    )


def write_action_features_tsv(*, rows: list[ActionFeatureRow], out_path) -> None:
    """Write ML action features with a stable TSV schema."""
    write_tsv_rows(
        out_path,
        ACTION_FEATURE_COLUMNS,
        (
            (
                row.action_type,
                row.contig_id,
                row.source_bin,
                row.target_bin,
                row.rule_reason,
                row.rule_accepted,
                _format_float(row.confidence),
                _format_optional_float(row.delta_contact),
                row.scg_status,
                row.support_edge_count,
                _format_float(row.support_weight_sum),
                _format_float(row.support_reliability_median),
                _format_float(row.support_k_eff_median),
                _format_optional_float(row.current_support),
                _format_optional_float(row.target_support),
                _format_optional_float(row.runner_up_support),
                _format_optional_float(row.cross_support),
                row.source_bin_n_contigs,
                row.target_bin_n_contigs,
                row.contig_length,
                row.feature_status,
                row.coverage_status,
                row.candidate_method,
                row.candidate_reasons,
            )
            for row in rows
        ),
    )


@dataclass(frozen=True)
class _EdgeStats:
    edge_count: int
    weight_sum: float
    reliability_median: float
    k_eff_median: float


def _edge_stats(
    store: HyperedgeStore,
    edge_ids: Collection[int],
    *,
    reliability_fn: ReliabilityFn,
) -> _EdgeStats:
    reliabilities: list[float] = []
    k_eff_values: list[float] = []
    for edge_id in sorted(set(int(value) for value in edge_ids)):
        edge = store.edges_by_id.get(int(edge_id))
        if edge is None:
            continue
        reliability = float(reliability_fn(edge))
        if reliability <= 0.0:
            continue
        reliabilities.append(reliability)
        k_eff_values.append(float(edge.k_eff))
    return _EdgeStats(
        edge_count=len(reliabilities),
        weight_sum=float(sum(reliabilities)),
        reliability_median=_median(reliabilities),
        k_eff_median=_median(k_eff_values),
    )


def _support_edges_for_contig_to_bin(
    store: HyperedgeStore,
    assignment: dict[str, str],
    *,
    contig_id: str,
    target_bin: str,
    reliability_fn: ReliabilityFn,
) -> tuple[int, ...]:
    out: list[int] = []
    target = str(target_bin)
    contig = str(contig_id)
    for edge_id in collect_edge_ids_for_contig(store, contig):
        edge = store.edges_by_id.get(int(edge_id))
        if edge is None or float(reliability_fn(edge)) <= 0.0:
            continue
        other_mass = 0.0
        for member_id, alpha in zip(edge.members, edge.alpha, strict=True):
            if member_id == contig:
                continue
            if str(assignment.get(member_id, "")).strip() == target:
                other_mass += float(alpha)
        if other_mass > 0.0:
            out.append(int(edge_id))
    return tuple(sorted(set(out)))


def _support_edges_between_bins(
    store: HyperedgeStore,
    assignment: dict[str, str],
    *,
    source_bin: str,
    target_bin: str,
    reliability_fn: ReliabilityFn,
) -> tuple[int, ...]:
    source = str(source_bin)
    target = str(target_bin)
    out: list[int] = []
    for edge in store.edges:
        if float(reliability_fn(edge)) <= 0.0:
            continue
        has_source = False
        has_target = False
        for member_id in edge.members:
            bin_id = str(assignment.get(member_id, "")).strip()
            has_source = has_source or bin_id == source
            has_target = has_target or bin_id == target
        if has_source and has_target:
            out.append(int(edge.edge_id))
    return tuple(sorted(set(out)))


def _feature_status_from_reason(reason: str) -> str:
    return str(reason) if "feature" in str(reason) else "pass_or_not_evaluated"


def _coverage_status_from_reason(reason: str) -> str:
    return str(reason) if "coverage" in str(reason) else "pass_or_not_evaluated"


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    xs = sorted(float(value) for value in values)
    mid = len(xs) // 2
    if len(xs) % 2:
        return float(xs[mid])
    return float(0.5 * (xs[mid - 1] + xs[mid]))


def _format_float(value: float) -> str:
    return f"{float(value):.8g}"


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return _format_float(float(value))
