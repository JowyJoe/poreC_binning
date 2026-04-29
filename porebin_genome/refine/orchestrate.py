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
from porebin_genome.refine.features import (
    build_bin_feature_profiles,
    load_refine_feature_inputs,
)
from porebin_genome.refine.hyperedge import default_edge_reliability, load_hyperedges
from porebin_genome.refine.markers import prepare_scg_profiles
from porebin_genome.refine.merge import apply_merge_decision, evaluate_merge_candidate, generate_merge_candidates
from porebin_genome.refine.models import RefineActionRow, RefineState, RefinedAssignmentRow, UnbinnedRow
from porebin_genome.refine.qc import build_bin_qc_rows, write_bin_qc_tsv
from porebin_genome.refine.reassign import (
    apply_reassign_decision,
    evaluate_reassign_candidate,
    generate_reassign_candidates,
)
from porebin_genome.refine.recruit import apply_recruit_decision, evaluate_recruit_candidate, generate_recruit_candidates
from porebin_genome.refine.split import apply_split_decision, evaluate_split_candidate, generate_split_candidates
from porebin_genome.refine.suspect import build_bin_snapshots, detect_suspect_bins


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
    enable_scg: bool = True,
    scg_hmm_path: Path | None = None,
    prodigal_executable: str = "prodigal",
    hmmsearch_executable: str = "hmmsearch",
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
    hyperedges = load_hyperedges(contacts_parquet)
    reliability_fn = default_edge_reliability
    feature_matrix, contig_name_to_idx, _idx_to_name, coverage_by_contig = load_refine_feature_inputs(
        contigs_fasta=contigs_fasta,
        coverage_tsv=coverage_tsv,
    )
    scg_result = None
    contig_scg_profiles = {}
    if enable_scg:
        scg_result = prepare_scg_profiles(
            contigs_fasta=contigs_fasta,
            out_dir=layout.evidence_dir / "scg",
            scg_hmm_path=scg_hmm_path,
            prodigal_executable=prodigal_executable,
            hmmsearch_executable=hmmsearch_executable,
        )
        contig_scg_profiles = scg_result.contig_profiles

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

    profiles = build_bin_feature_profiles(
        state=state,
        feature_matrix=feature_matrix,
        contig_name_to_idx=contig_name_to_idx,
    )
    snapshots = build_bin_snapshots(
        state=state,
        store=hyperedges,
        profiles=profiles,
        reliability_fn=reliability_fn,
        contig_scg_profiles=contig_scg_profiles if enable_scg else None,
    )
    suspects = detect_suspect_bins(snapshots=snapshots)

    split_candidates = generate_split_candidates(
        state=state,
        suspects=suspects,
        store=hyperedges,
        feature_matrix=feature_matrix,
        contig_name_to_idx=contig_name_to_idx,
        reliability_fn=reliability_fn,
    )
    n_split_applied = 0
    for candidate in split_candidates:
        decision = evaluate_split_candidate(
            candidate=candidate,
            state=state,
            store=hyperedges,
            feature_matrix=feature_matrix,
            contig_name_to_idx=contig_name_to_idx,
            profiles=profiles,
            new_bin_id=state.allocate_bin_id(),
            reliability_fn=reliability_fn,
            contig_scg_profiles=contig_scg_profiles if enable_scg else None,
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
                delta_contact=_extract_delta_contact(decision.note),
                scg_status=_extract_scg_status(
                    reason=decision.reason,
                    note=decision.note,
                    scg_enabled=enable_scg,
                ),
                note=decision.note,
            )
        )
        if decision.accepted:
            apply_split_decision(state=state, decision=decision)
            n_split_applied += 1

    profiles = build_bin_feature_profiles(
        state=state,
        feature_matrix=feature_matrix,
        contig_name_to_idx=contig_name_to_idx,
    )

    reassign_candidates = generate_reassign_candidates(
        state=state,
        store=hyperedges,
        profiles=profiles,
        feature_matrix=feature_matrix,
        contig_name_to_idx=contig_name_to_idx,
        reliability_fn=reliability_fn,
    )
    n_reassigned = 0
    n_reassign_abstained = 0
    for candidate in reassign_candidates:
        decision = evaluate_reassign_candidate(
            candidate=candidate,
            state=state,
            store=hyperedges,
            profiles=profiles,
            feature_matrix=feature_matrix,
            contig_name_to_idx=contig_name_to_idx,
            reliability_fn=reliability_fn,
            contig_scg_profiles=contig_scg_profiles if enable_scg else None,
        )
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
                delta_contact=_extract_delta_contact(decision.note),
                scg_status=_extract_scg_status(
                    reason=decision.reason,
                    note=decision.note,
                    scg_enabled=enable_scg,
                ),
                note=decision.note,
            )
        )
        before_assigned = candidate.contig_id in state.current_assignment
        apply_reassign_decision(state=state, decision=decision)
        after_assigned = candidate.contig_id in state.current_assignment
        if decision.accepted:
            n_reassigned += 1
        elif before_assigned and not after_assigned:
            n_reassign_abstained += 1
            action_rows.append(
                RefineActionRow(
                    action_type="abstain",
                    contig_id=candidate.contig_id,
                    bin_id="",
                    source_bin=candidate.source_bin,
                    target_bin="",
                    reason=decision.reason,
                    accepted=True,
                    confidence=float(decision.confidence),
                    delta_contact=_extract_delta_contact(decision.note),
                    scg_status=_extract_scg_status(
                        reason=decision.reason,
                        note=decision.note,
                        scg_enabled=enable_scg,
                    ),
                    note=decision.note,
                )
            )

    profiles = build_bin_feature_profiles(
        state=state,
        feature_matrix=feature_matrix,
        contig_name_to_idx=contig_name_to_idx,
    )

    merge_candidates = generate_merge_candidates(
        state=state,
        store=hyperedges,
        profiles=profiles,
        reliability_fn=reliability_fn,
    )
    n_merged = 0
    for candidate in merge_candidates:
        decision = evaluate_merge_candidate(
            candidate=candidate,
            state=state,
            store=hyperedges,
            profiles=profiles,
            feature_matrix=feature_matrix,
            contig_name_to_idx=contig_name_to_idx,
            reliability_fn=reliability_fn,
            contig_scg_profiles=contig_scg_profiles if enable_scg else None,
        )
        action_rows.append(
            RefineActionRow(
                action_type="merge",
                contig_id="",
                bin_id=candidate.source_bin,
                source_bin=candidate.target_bin,
                target_bin=candidate.source_bin,
                reason=decision.reason,
                accepted=decision.accepted,
                confidence=float(decision.confidence),
                delta_contact=_extract_delta_contact(decision.note),
                scg_status=_extract_scg_status(
                    reason=decision.reason,
                    note=decision.note,
                    scg_enabled=enable_scg,
                ),
                note=decision.note,
            )
        )
        if decision.accepted:
            apply_merge_decision(state=state, decision=decision)
            n_merged += 1

    profiles = build_bin_feature_profiles(
        state=state,
        feature_matrix=feature_matrix,
        contig_name_to_idx=contig_name_to_idx,
    )

    recruit_candidates = generate_recruit_candidates(
        state=state,
        store=hyperedges,
        profiles=profiles,
        feature_matrix=feature_matrix,
        contig_name_to_idx=contig_name_to_idx,
        reliability_fn=reliability_fn,
    )
    n_recruited = 0
    for candidate in recruit_candidates:
        decision = evaluate_recruit_candidate(
            candidate=candidate,
            state=state,
            store=hyperedges,
            reliability_fn=reliability_fn,
            contig_scg_profiles=contig_scg_profiles if enable_scg else None,
        )
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
                delta_contact=_extract_delta_contact(decision.note),
                scg_status=_extract_scg_status(
                    reason=decision.reason,
                    note=decision.note,
                    scg_enabled=enable_scg,
                ),
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

    final_profiles = build_bin_feature_profiles(
        state=state,
        feature_matrix=feature_matrix,
        contig_name_to_idx=contig_name_to_idx,
    )
    final_snapshots = build_bin_snapshots(
        state=state,
        store=hyperedges,
        profiles=final_profiles,
        reliability_fn=reliability_fn,
        contig_scg_profiles=contig_scg_profiles if enable_scg else None,
    )

    refined_rows = _build_refined_assignment_rows(state=state)
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
            "scg_dir": (str(scg_result.scg_dir) if scg_result is not None else None),
        },
        "n_bins_in": int(n_bins_in),
        "n_bins_out": int(len(final_snapshots)),
        "n_suspect_bins": int(len(suspects)),
        "n_split_candidates": int(len(split_candidates)),
        "n_split_applied": int(n_split_applied),
        "n_reassign_candidates": int(len(reassign_candidates)),
        "n_reassigned": int(n_reassigned),
        "n_reassign_abstained": int(n_reassign_abstained),
        "n_merge_candidates": int(len(merge_candidates)),
        "n_merged": int(n_merged),
        "n_recruit_candidates": int(len(recruit_candidates)),
        "n_recruited": int(n_recruited),
        "n_unbinned_final": int(len(unbinned_rows)),
        "n_scg_profiled_contigs": int(len(contig_scg_profiles)),
        "n_bins_with_duplicate_scg": int(
            sum(1 for snapshot in final_snapshots.values() if int(snapshot.scg_duplicate_marker_count) > 0)
        ),
        "n_scg_veto_actions": int(sum(1 for row in action_rows if row.scg_status == "veto")),
        "n_actions_accepted": int(sum(1 for row in action_rows if row.accepted)),
        "n_actions_rejected": int(sum(1 for row in action_rows if not row.accepted)),
        "notes": {
            "refine_goal": "improve genome-bin purity and control contamination risk",
            "reassign_semantics": "move_to_target_bin_or_abstain_to_unbinned",
            "abstention_policy": "boundary contigs with no positive contact-gain move may remain or move to unbinned",
            "scg_enabled": bool(enable_scg),
            "scg_dependency_semantics": (
                "requires external prodigal and hmmsearch unless refine is run with SCG disabled"
            ),
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
) -> list[RefinedAssignmentRow]:
    rows: list[RefinedAssignmentRow] = []
    for contig_id in sorted(state.current_assignment.keys()):
        bin_id = state.current_assignment[contig_id]
        rows.append(
            RefinedAssignmentRow(
                contig_id=contig_id,
                bin_id=bin_id,
                assignment_stage=state.assignment_stage.get(contig_id, "coarse_keep"),
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


def _extract_delta_contact(note: str) -> float | None:
    for token in str(note).split(";"):
        token = token.strip()
        if token.startswith("delta_contact="):
            value = token.split("=", 1)[1].strip()
            try:
                return float(value)
            except Exception:
                return None
    return None


def _extract_scg_status(*, reason: str, note: str, scg_enabled: bool) -> str:
    if not scg_enabled:
        return "disabled"
    if "_scg_" in str(reason):
        return "veto"
    marker = "scg="
    if marker in str(note):
        suffix = str(note).split(marker, 1)[1]
        status = suffix.split(":", 1)[0].strip()
        if status in {"veto", "support", "abstain"}:
            return status
    return "abstain"
