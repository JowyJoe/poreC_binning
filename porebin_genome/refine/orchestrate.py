"""Genome-centric refine MVP orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from porebin_genome.io.contacts import validate_contacts_parquet_core_schema
from porebin_genome.io.contracts import (
    COARSE_BINS_COLUMNS,
    FINAL_BINS_COLUMNS,
    UNBINNED_COLUMNS,
    build_pipeline_layout,
)
from porebin_genome.io.coverage import validate_coverage_tsv
from porebin_genome.io.fasta import read_contig_lengths
from porebin_genome.io.runtime import ensure_dir, write_json
from porebin_genome.io.tables import read_assignment_tsv, write_tsv_rows
from porebin_genome.refine.actions import write_refine_actions_tsv
from porebin_genome.refine.models import RefineActionRow, RefineState, RefinedAssignmentRow, UnbinnedRow
from porebin_genome.refine.qc import build_bin_qc_rows, write_bin_qc_tsv
from porebin_genome.refine.reassign import (
    apply_reassign_decision,
    evaluate_reassign_candidate,
    filter_low_confidence_assignments,
    generate_reassign_candidates,
)
from porebin_genome.refine.recruit import apply_recruit_decision, evaluate_recruit_candidate, generate_recruit_candidates
from porebin_genome.refine.split import apply_split_decision, evaluate_split_candidate, generate_split_candidates
from porebin_genome.refine.suspect import build_bin_snapshots, detect_suspect_bins
from porebin_genome.refine.support import (
    build_bin_feature_profiles,
    compute_assignment_confidence,
    evaluate_feature_gate,
    load_refine_feature_inputs,
    scan_bin_support,
)


@dataclass(frozen=True)
class RefineRunResult:
    """Outputs emitted by the refine layer."""

    bins_refined_tsv: Path
    unbinned_tsv: Path
    bin_qc_tsv: Path
    refine_actions_tsv: Path
    refine_meta_json: Path
    implemented: bool


def run_refinement(
    *,
    contigs_fasta: Path,
    coarse_bins_tsv: Path,
    contacts_parquet: Path,
    coverage_tsv: Path,
    out_dir: Path,
    logger: Optional[object] = None,
) -> RefineRunResult:
    """Run the genome-centric refine MVP."""
    _ = logger
    layout = build_pipeline_layout(out_dir)
    ensure_dir(layout.final_dir)

    contigs_fasta = contigs_fasta.resolve()
    contacts_parquet = contacts_parquet.resolve()
    coverage_tsv = coverage_tsv.resolve()
    coarse_bins_tsv = coarse_bins_tsv.resolve()

    if not contigs_fasta.exists():
        raise FileNotFoundError(f"Contigs FASTA not found: {contigs_fasta}")
    validate_contacts_parquet_core_schema(contacts_parquet)
    validate_coverage_tsv(coverage_tsv)

    coarse_assignment_all = read_assignment_tsv(coarse_bins_tsv, expected_header=COARSE_BINS_COLUMNS)
    contig_lengths = read_contig_lengths(contigs_fasta)
    feature_matrix, contig_name_to_idx, _idx_to_name, coverage_by_contig = load_refine_feature_inputs(
        contigs_fasta=contigs_fasta,
        coverage_tsv=coverage_tsv,
    )

    current_assignment = {
        contig_id: str(bin_id)
        for contig_id, bin_id in coarse_assignment_all.items()
        if contig_id in contig_lengths and str(bin_id).strip()
    }
    next_bin_id = _next_bin_id(current_assignment.values())
    state = RefineState(
        contig_lengths=contig_lengths,
        coverage_by_contig=coverage_by_contig,
        coarse_assignment={contig_id: str(bin_id) for contig_id, bin_id in coarse_assignment_all.items()},
        current_assignment=current_assignment,
        assignment_stage={contig_id: "coarse_keep" for contig_id in current_assignment},
        assignment_reason={contig_id: "coarse_assignment_retained" for contig_id in current_assignment},
        next_bin_id=next_bin_id,
    )
    for contig_id in sorted(contig_lengths.keys()):
        if contig_id not in state.current_assignment:
            state.unassign(
                contig_id,
                stage="coarse",
                reason="coarse_unassigned",
                source_bin="",
                note="not assigned during coarse candidate-bin discovery",
            )

    n_bins_in = len(state.bin_to_contigs())
    action_rows: list[RefineActionRow] = []

    support_summary = scan_bin_support(contacts_parquet=contacts_parquet, state=state)
    profiles = build_bin_feature_profiles(
        state=state,
        feature_matrix=feature_matrix,
        contig_name_to_idx=contig_name_to_idx,
    )
    snapshots = build_bin_snapshots(state=state, support_summary=support_summary, profiles=profiles)
    suspects = detect_suspect_bins(snapshots=snapshots)

    split_candidates = generate_split_candidates(
        state=state,
        suspects=suspects,
        support_summary=support_summary,
        feature_matrix=feature_matrix,
        contig_name_to_idx=contig_name_to_idx,
    )
    n_split_applied = 0
    for candidate in split_candidates:
        snapshot = snapshots[candidate.source_bin]
        decision = evaluate_split_candidate(
            candidate=candidate,
            state=state,
            snapshot=snapshot,
            support_summary=support_summary,
            feature_matrix=feature_matrix,
            contig_name_to_idx=contig_name_to_idx,
            new_bin_id=state.allocate_bin_id(),
        )
        action_rows.append(
            RefineActionRow(
                action_type="split",
                contig_id="",
                bin_id=candidate.source_bin,
                source_bin=candidate.source_bin,
                target_bin=decision.new_bin_id,
                reason=decision.reason,
                accepted=decision.accepted,
                confidence=float(decision.confidence),
                note=decision.note,
            )
        )
        if decision.accepted:
            apply_split_decision(state=state, decision=decision)
            n_split_applied += 1

    support_summary = scan_bin_support(contacts_parquet=contacts_parquet, state=state)
    profiles = build_bin_feature_profiles(
        state=state,
        feature_matrix=feature_matrix,
        contig_name_to_idx=contig_name_to_idx,
    )

    reassign_candidates = generate_reassign_candidates(
        state=state,
        support_summary=support_summary,
        profiles=profiles,
        feature_matrix=feature_matrix,
        contig_name_to_idx=contig_name_to_idx,
    )
    n_reassigned = 0
    for candidate in reassign_candidates:
        decision = evaluate_reassign_candidate(candidate)
        action_rows.append(
            RefineActionRow(
                action_type="reassign",
                contig_id=candidate.contig_id,
                bin_id="",
                source_bin=candidate.source_bin,
                target_bin=candidate.target_bin,
                reason=decision.reason,
                accepted=decision.accepted,
                confidence=float(decision.confidence),
                note=decision.note,
            )
        )
        if decision.accepted:
            apply_reassign_decision(state=state, decision=decision)
            n_reassigned += 1

    support_summary = scan_bin_support(contacts_parquet=contacts_parquet, state=state)
    profiles = build_bin_feature_profiles(
        state=state,
        feature_matrix=feature_matrix,
        contig_name_to_idx=contig_name_to_idx,
    )
    filtered = filter_low_confidence_assignments(
        state=state,
        support_summary=support_summary,
        profiles=profiles,
        feature_matrix=feature_matrix,
        contig_name_to_idx=contig_name_to_idx,
    )
    for contig_id, source_bin, confidence, note in filtered:
        state.unassign(
            contig_id,
            stage="filter",
            reason="assignment_confidence_below_threshold",
            source_bin=source_bin,
            note=note,
        )
        action_rows.append(
            RefineActionRow(
                action_type="filter",
                contig_id=contig_id,
                bin_id="",
                source_bin=source_bin,
                target_bin="",
                reason="move_to_unbinned",
                accepted=True,
                confidence=float(confidence),
                note=note,
            )
        )

    support_summary = scan_bin_support(contacts_parquet=contacts_parquet, state=state)
    profiles = build_bin_feature_profiles(
        state=state,
        feature_matrix=feature_matrix,
        contig_name_to_idx=contig_name_to_idx,
    )

    recruit_candidates = generate_recruit_candidates(
        state=state,
        support_summary=support_summary,
        profiles=profiles,
        feature_matrix=feature_matrix,
        contig_name_to_idx=contig_name_to_idx,
    )
    n_recruited = 0
    for candidate in recruit_candidates:
        decision = evaluate_recruit_candidate(candidate)
        action_rows.append(
            RefineActionRow(
                action_type="recruit",
                contig_id=candidate.contig_id,
                bin_id="",
                source_bin="",
                target_bin=candidate.target_bin,
                reason=decision.reason,
                accepted=decision.accepted,
                confidence=float(decision.confidence),
                note=decision.note,
            )
        )
        if decision.accepted:
            apply_recruit_decision(state=state, decision=decision)
            n_recruited += 1
        else:
            state.unassign(
                candidate.contig_id,
                stage="recruit",
                reason=decision.reason,
                source_bin="",
                note=decision.note,
            )

    final_support = scan_bin_support(contacts_parquet=contacts_parquet, state=state)
    final_profiles = build_bin_feature_profiles(
        state=state,
        feature_matrix=feature_matrix,
        contig_name_to_idx=contig_name_to_idx,
    )
    final_snapshots = build_bin_snapshots(
        state=state,
        support_summary=final_support,
        profiles=final_profiles,
    )

    refined_rows = _build_refined_assignment_rows(
        state=state,
        support_summary=final_support,
        profiles=final_profiles,
        feature_matrix=feature_matrix,
        contig_name_to_idx=contig_name_to_idx,
    )
    unbinned_rows = _build_unbinned_rows(state=state)
    bin_qc_rows = build_bin_qc_rows(snapshots=final_snapshots)

    write_tsv_rows(
        layout.refined_bins_tsv,
        FINAL_BINS_COLUMNS,
        [
            (
                row.contig_id,
                row.bin_id,
                row.assignment_stage,
                f"{float(row.assignment_confidence):.6g}",
                row.assignment_reason,
            )
            for row in refined_rows
        ],
    )
    write_tsv_rows(
        layout.unbinned_tsv,
        UNBINNED_COLUMNS,
        [
            (
                row.contig_id,
                row.stage,
                row.reason,
                row.source_bin,
                row.note,
            )
            for row in unbinned_rows
        ],
    )
    write_bin_qc_tsv(rows=bin_qc_rows, out_path=layout.bin_qc_tsv)
    write_refine_actions_tsv(rows=action_rows, out_path=layout.refine_actions_tsv)

    refine_meta = {
        "stage": "bin_refinement",
        "implemented": True,
        "inputs": {
            "contigs_fasta": str(contigs_fasta),
            "coarse_bins_tsv": str(coarse_bins_tsv),
            "coarse_run_json": str(layout.coarse_run_json),
            "contacts_parquet": str(contacts_parquet),
            "coverage_tsv": str(coverage_tsv),
        },
        "outputs": {
            "bins_refined_tsv": str(layout.refined_bins_tsv),
            "unbinned_tsv": str(layout.unbinned_tsv),
            "bin_qc_tsv": str(layout.bin_qc_tsv),
            "refine_actions_tsv": str(layout.refine_actions_tsv),
        },
        "n_bins_in": int(n_bins_in),
        "n_bins_out": int(len(final_snapshots)),
        "n_suspect_bins": int(len(suspects)),
        "n_split_candidates": int(len(split_candidates)),
        "n_split_applied": int(n_split_applied),
        "n_reassign_candidates": int(len(reassign_candidates)),
        "n_reassigned": int(n_reassigned),
        "n_recruit_candidates": int(len(recruit_candidates)),
        "n_recruited": int(n_recruited),
        "n_unbinned_final": int(len(unbinned_rows)),
        "n_actions_accepted": int(sum(1 for row in action_rows if row.accepted)),
        "n_actions_rejected": int(sum(1 for row in action_rows if not row.accepted)),
        "notes": {
            "refine_goal": "improve genome-bin purity and control contamination risk",
            "assignment_confidence_formula": "0.45*target_bin_support_share + 0.35*runner_up_margin + 0.20*feature_gate",
            "reassign_semantics": "move_to_target_bin_only",
            "abstention_policy": "retain_low_confidence_or_ambiguous_contigs_as_unbinned",
        },
    }
    write_json(layout.refine_meta_json, refine_meta)

    return RefineRunResult(
        bins_refined_tsv=layout.refined_bins_tsv,
        unbinned_tsv=layout.unbinned_tsv,
        bin_qc_tsv=layout.bin_qc_tsv,
        refine_actions_tsv=layout.refine_actions_tsv,
        refine_meta_json=layout.refine_meta_json,
        implemented=True,
    )


def _build_refined_assignment_rows(
    *,
    state: RefineState,
    support_summary,
    profiles,
    feature_matrix,
    contig_name_to_idx,
) -> list[RefinedAssignmentRow]:
    rows: list[RefinedAssignmentRow] = []
    for contig_id in sorted(state.current_assignment.keys()):
        bin_id = state.current_assignment[contig_id]
        target_support = float(support_summary.support_by_contig_bin.get(contig_id, {}).get(bin_id, 0.0))
        runner_up_support = float(support_summary.runner_up_support.get(contig_id, 0.0))
        feature_gate, _feature_score, _feature_note = evaluate_feature_gate(
            contig_id=contig_id,
            target_bin=bin_id,
            feature_matrix=feature_matrix,
            contig_name_to_idx=contig_name_to_idx,
            profiles=profiles,
            coverage_by_contig=state.coverage_by_contig,
        )
        confidence = compute_assignment_confidence(
            target_bin_support=target_support,
            runner_up_support=runner_up_support,
            feature_gate=feature_gate,
        )
        rows.append(
            RefinedAssignmentRow(
                contig_id=contig_id,
                bin_id=bin_id,
                assignment_stage=state.assignment_stage.get(contig_id, "coarse_keep"),
                assignment_confidence=float(confidence),
                assignment_reason=state.assignment_reason.get(contig_id, "coarse_assignment_retained"),
            )
        )
    return rows


def _build_unbinned_rows(*, state: RefineState) -> list[UnbinnedRow]:
    rows: list[UnbinnedRow] = []
    assigned = set(state.current_assignment.keys())
    for contig_id in sorted(state.contig_lengths.keys()):
        if contig_id in assigned:
            continue
        row = state.unbinned_rows.get(contig_id)
        if row is None:
            row = UnbinnedRow(
                contig_id=contig_id,
                stage="refine",
                reason="unresolved_after_refine",
                source_bin="",
                note="refine left contig unresolved",
            )
        rows.append(row)
    return rows


def _next_bin_id(existing_bin_ids) -> int:
    values: list[int] = []
    for bin_id in existing_bin_ids:
        try:
            values.append(int(str(bin_id)))
        except Exception:
            continue
    return (max(values) + 1) if values else 0
