"""Thin, observable orchestration for the replacement refine engine."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from porebin_genome.coarse.hyperedge_weight import (
    DEFAULT_HYPERGRAPH_WEIGHT_ETA,
)
from porebin_genome.io.contacts import validate_contacts_parquet_core_schema
from porebin_genome.io.contracts import (
    BIN_QC_COLUMNS,
    COARSE_BINS_COLUMNS,
    FINAL_BINS_COLUMNS,
    REFINE_ACTIONS_COLUMNS,
    UNBINNED_COLUMNS,
    build_pipeline_layout,
)
from porebin_genome.io.coverage import validate_coverage_tsv
from porebin_genome.io.fasta import read_contig_lengths
from porebin_genome.io.runtime import ensure_dir, write_json
from porebin_genome.io.tables import read_assignment_tsv, write_tsv_rows
from porebin_genome.evidence.scg.discovery import prepare_scg_profiles
from porebin_genome.refinement.actions import (
    Decision,
    DecisionStatus,
)
from porebin_genome.refinement.contact_index import (
    ContactIndex,
    build_contact_index,
)
from porebin_genome.refinement.evaluator import (
    ActionEvaluator,
    apply_decision,
)
from porebin_genome.refinement.merge import (
    MergeCandidate,
    apply_merge_batch,
    generate_merge_candidates,
    prepare_merge_batch,
)
from porebin_genome.refinement.policy import ActionPolicy
from porebin_genome.refinement.profiles import (
    EvidenceProfileState,
    load_evidence_inputs,
)
from porebin_genome.refinement.recruit import (
    RecruitCandidate,
    RecruitCandidateStatus,
    run_recruitment_pass,
)
from porebin_genome.refinement.split import (
    SplitCandidate,
    SplitCandidateStatus,
    generate_split_candidate,
)
from porebin_genome.refinement.stage_log import RefinementStageLog
from porebin_genome.refinement.state import RefineState


@dataclass(frozen=True)
class RefineRunResult:
    """Public outputs emitted by the replacement refinement engine."""

    bins_refined_tsv: Path
    unbinned_tsv: Path
    bin_qc_tsv: Path
    refine_actions_tsv: Path
    refine_stage_log_jsonl: Path
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
    prodigal_executable: str = "prodigal",
    hmmsearch_executable: str = "hmmsearch",
    embedding_tsv: Path | None = None,
    hypergraph_weight_eta: float = DEFAULT_HYPERGRAPH_WEIGHT_ETA,
    logger: object | None = None,
) -> RefineRunResult:
    """Run SCG-guided split, conservative merge, then unbinned recruitment."""
    layout = build_pipeline_layout(out_dir)
    ensure_dir(layout.final_dir)
    _clear_previous_refine_outputs(layout)
    stage_log = RefinementStageLog(
        layout.refine_stage_log_jsonl,
        logger=logger,
    )
    stage_log.emit(
        "run_start",
        stage="refine",
        message="replacement refinement started",
        checkpoint_enabled=False,
    )

    try:
        with stage_log.stage(
            "validate_inputs",
            message="validating refine input contracts",
        ) as report:
            contigs_fasta = contigs_fasta.resolve()
            coarse_bins_tsv = coarse_bins_tsv.resolve()
            contacts_parquet = contacts_parquet.resolve()
            coverage_tsv = coverage_tsv.resolve()
            embedding_tsv = (
                embedding_tsv.resolve()
                if embedding_tsv is not None
                else None
            )
            if not contigs_fasta.exists():
                raise FileNotFoundError(
                    f"Contigs FASTA not found: {contigs_fasta}"
                )
            if not coarse_bins_tsv.exists():
                raise FileNotFoundError(
                    f"Coarse bins TSV not found: {coarse_bins_tsv}"
                )
            if embedding_tsv is not None and not embedding_tsv.exists():
                raise FileNotFoundError(
                    f"HG-VAE embedding TSV not found: {embedding_tsv}"
                )
            validate_contacts_parquet_core_schema(contacts_parquet)
            validate_coverage_tsv(coverage_tsv)
            contig_lengths_by_name = read_contig_lengths(contigs_fasta)
            if not contig_lengths_by_name:
                raise ValueError("Contigs FASTA contains no records.")
            coarse_assignment = read_assignment_tsv(
                coarse_bins_tsv,
                expected_header=COARSE_BINS_COLUMNS,
            )
            _validate_coarse_assignment(
                coarse_assignment,
                known_contigs=set(contig_lengths_by_name),
            )
            report.update(
                message="refine input contracts validated",
                counts={
                    "contigs": len(contig_lengths_by_name),
                    "coarse_assignments": sum(
                        bool(bin_id)
                        for bin_id in coarse_assignment.values()
                    ),
                },
            )

        scg_result = None
        scg_index_json: Path | None = None
        with stage_log.stage(
            "scg",
            message=(
                "preparing fixed-panel SCG evidence"
                if enable_scg
                else "SCG evidence disabled"
            ),
            enabled=bool(enable_scg),
        ) as report:
            if enable_scg:
                scg_result = prepare_scg_profiles(
                    contigs_fasta=contigs_fasta,
                    out_dir=layout.evidence_dir / "scg",
                    prodigal_executable=prodigal_executable,
                    hmmsearch_executable=hmmsearch_executable,
                )
                scg_index_json = scg_result.contig_index_json
                report.update(
                    message="fixed-panel SCG evidence ready",
                    cache_status=scg_result.cache_status,
                    counts={
                        "profiled_contigs": len(
                            scg_result.contig_profiles
                        ),
                        "canonical_markers": scg_result.panel.marker_count,
                    },
                )
            else:
                report.update(
                    message="SCG evidence skipped by configuration",
                    cache_status="disabled",
                    counts={"profiled_contigs": 0},
                )

        with stage_log.stage(
            "initialize",
            message="building contact index and synchronized evidence state",
        ) as report:
            contig_names = tuple(contig_lengths_by_name)
            contact_index = build_contact_index(
                contacts_parquet,
                contig_names=contig_names,
                eta=float(hypergraph_weight_eta),
            )
            assignment, n_bins = _build_integer_assignment(
                contig_names=contig_names,
                coarse_assignment=coarse_assignment,
            )
            state = RefineState(
                contact_index=contact_index,
                assignment=assignment,
                contig_lengths=np.asarray(
                    [
                        contig_lengths_by_name[name]
                        for name in contig_names
                    ],
                    dtype=np.int64,
                ),
                n_bins=n_bins,
            )
            evidence_inputs = load_evidence_inputs(
                contact_index=contact_index,
                embedding_tsv=embedding_tsv,
                contigs_fasta=contigs_fasta,
                coverage_tsv=coverage_tsv,
                scg_index_json=scg_index_json,
            )
            profiles = EvidenceProfileState(
                refine_state=state,
                inputs=evidence_inputs,
            )
            n_bins_in = _nonempty_bin_count(state)
            n_unbinned_in = _unbinned_count(state)
            assignment_stage = [
                "coarse_keep" if int(bin_idx) >= 0 else "coarse"
                for bin_idx in state.assignment
            ]
            assignment_reason = [
                (
                    "coarse_assignment_retained"
                    if int(bin_idx) >= 0
                    else "coarse_unassigned"
                )
                for bin_idx in state.assignment
            ]
            unbinned_detail = {
                contig_idx: {
                    "stage": "coarse",
                    "reason": "coarse_unassigned",
                    "source_bin": "",
                    "note": "contig was not assigned by coarse clustering",
                }
                for contig_idx, bin_idx in enumerate(state.assignment)
                if int(bin_idx) == -1
            }
            report.update(
                message="contact and evidence state initialized",
                state_version=state.version,
                counts={
                    "contigs": state.n_contigs,
                    "hyperedges": contact_index.n_edges,
                    "incidences": contact_index.n_incidences,
                    "bins": n_bins_in,
                    "unbinned": n_unbinned_in,
                    "embedding_rows": int(
                        np.count_nonzero(evidence_inputs.embedding_present)
                    ),
                },
                contact_counters=_contact_counters(contact_index),
            )

        policy = ActionPolicy()
        action_rows: list[tuple[object, ...]] = []
        stage_counts: dict[str, dict[str, int]] = {}

        split_counter_before = _contact_counters(contact_index)
        with stage_log.stage(
            "split",
            message="evaluating SCG-suspect bins for deterministic split",
            state_version=state.version,
        ) as report:
            suspect_bins = tuple(
                bin_idx
                for bin_idx in range(state.n_bins)
                if state.bin_members(bin_idx)
                and profiles.profile(bin_idx).scg.duplicate_burden > 0
            )
            split_counts = _empty_decision_counts()
            split_counts["suspect"] = len(suspect_bins)
            split_counts["candidates"] = 0
            for source_bin_idx in suspect_bins:
                candidate = generate_split_candidate(
                    source_bin_idx=source_bin_idx,
                    state=state,
                    profiles=profiles,
                )
                if (
                    candidate.status is not SplitCandidateStatus.READY
                    or candidate.proposal is None
                ):
                    split_counts["abstained"] += 1
                    action_rows.append(_split_candidate_row(candidate))
                    continue
                split_counts["candidates"] += 1
                decision = policy.decide(
                    ActionEvaluator(
                        state=state,
                        profiles=profiles,
                    ).evaluate(candidate.proposal)
                )
                _count_decision(split_counts, decision)
                action_rows.append(
                    _decision_row(
                        decision,
                        source_bin=str(candidate.source_bin_idx),
                        target_bin=",".join(
                            str(value)
                            for value in candidate.proposal.target_bins
                        ),
                        extra=_split_details(candidate),
                    )
                )
                if decision.status is DecisionStatus.ACCEPT:
                    apply_decision(
                        state=state,
                        profiles=profiles,
                        decision=decision,
                    )
                    _mark_split_assignment_metadata(
                        candidate,
                        assignment_stage=assignment_stage,
                        assignment_reason=assignment_reason,
                    )
            stage_counts["split"] = split_counts
            report.update(
                message="split stage completed",
                state_version=state.version,
                counts=split_counts,
                contact_counters=_counter_delta(
                    split_counter_before,
                    _contact_counters(contact_index),
                ),
            )

        merge_counter_before = _contact_counters(contact_index)
        with stage_log.stage(
            "merge",
            message="scanning once for conservative mutual-best bin merges",
            state_version=state.version,
        ) as report:
            merge_batch = generate_merge_candidates(
                state=state,
                profiles=profiles,
            )
            evaluator = ActionEvaluator(state=state, profiles=profiles)
            merge_decisions: list[Decision] = []
            merge_counts = _empty_decision_counts()
            merge_counts.update(
                {
                    "eligible_bins": len(
                        merge_batch.eligible_bin_indices
                    ),
                    "supported_pairs": merge_batch.supported_pair_count,
                    "candidates": len(merge_batch.candidates),
                }
            )
            for candidate in merge_batch.candidates:
                decision = policy.decide(
                    evaluator.evaluate(candidate.proposal)
                )
                _count_decision(merge_counts, decision)
                action_rows.append(
                    _decision_row(
                        decision,
                        source_bin=str(candidate.source_bin_idx),
                        target_bin=str(candidate.target_bin_idx),
                        extra=_merge_details(candidate),
                    )
                )
                if decision.status is DecisionStatus.ACCEPT:
                    merge_decisions.append(decision)
            if merge_decisions:
                merge_update = prepare_merge_batch(
                    state=state,
                    profiles=profiles,
                    decisions=tuple(merge_decisions),
                )
                apply_merge_batch(
                    state=state,
                    profiles=profiles,
                    update=merge_update,
                )
                for decision in merge_decisions:
                    for change in decision.proposal.changes:
                        assignment_stage[change.contig_idx] = "merge"
                        assignment_reason[change.contig_idx] = (
                            "contact_supported_compatible_bins_merged"
                        )
            stage_counts["merge"] = merge_counts
            report.update(
                message="merge stage completed",
                state_version=state.version,
                counts=merge_counts,
                contact_counters=_counter_delta(
                    merge_counter_before,
                    _contact_counters(contact_index),
                ),
            )

        recruit_counter_before = _contact_counters(contact_index)
        with stage_log.stage(
            "recruit",
            message="testing unbinned contigs by Pore-C and HG-VAE agreement",
            state_version=state.version,
            counts={"unbinned": _unbinned_count(state)},
        ) as report:
            recruit_result = run_recruitment_pass(
                state=state,
                profiles=profiles,
                policy=policy,
            )
            recruit_counts = _empty_decision_counts()
            recruit_counts.update(
                {
                    "unbinned_considered": len(
                        recruit_result.initial_batch.outcomes
                    ),
                    "candidates": len(
                        recruit_result.initial_batch.candidates
                    ),
                }
            )
            for outcome in recruit_result.initial_batch.outcomes:
                if outcome.status is RecruitCandidateStatus.READY:
                    continue
                recruit_counts["abstained"] += 1
                action_rows.append(
                    _recruit_candidate_row(
                        outcome,
                        contig_name=contact_index.contig_names[
                            outcome.contig_idx
                        ],
                    )
                )
                unbinned_detail[outcome.contig_idx] = {
                    "stage": "recruit",
                    "reason": outcome.reason,
                    "source_bin": "",
                    "note": _compact_json(_recruit_details(outcome)),
                }
            for attempt in recruit_result.attempts:
                current = attempt.evaluated_candidate
                decision = attempt.decision
                if decision is None:
                    recruit_counts["abstained"] += 1
                    action_rows.append(
                        _recruit_candidate_row(
                            current,
                            contig_name=contact_index.contig_names[
                                current.contig_idx
                            ],
                        )
                    )
                    unbinned_detail[current.contig_idx] = {
                        "stage": "recruit",
                        "reason": current.reason,
                        "source_bin": "",
                        "note": _compact_json(
                            _recruit_details(current)
                        ),
                    }
                    continue
                _count_decision(recruit_counts, decision)
                action_rows.append(
                    _decision_row(
                        decision,
                        contig_id=contact_index.contig_names[
                            current.contig_idx
                        ],
                        target_bin=str(
                            current.contact_target_bin_idx
                            if current.contact_target_bin_idx is not None
                            else ""
                        ),
                        extra=_recruit_details(current),
                    )
                )
                if decision.status is DecisionStatus.ACCEPT:
                    assignment_stage[current.contig_idx] = "recruit"
                    assignment_reason[current.contig_idx] = (
                        "porec_hgvae_agreement_scg_safe"
                    )
                    unbinned_detail.pop(current.contig_idx, None)
                else:
                    unbinned_detail[current.contig_idx] = {
                        "stage": "recruit",
                        "reason": decision.reason_code,
                        "source_bin": "",
                        "note": _compact_json(
                            _decision_details(
                                decision,
                                extra=_recruit_details(current),
                            )
                        ),
                    }
            stage_counts["recruit"] = recruit_counts
            report.update(
                message="recruit stage completed",
                state_version=state.version,
                counts={
                    **recruit_counts,
                    "accepted": recruit_result.accepted_count,
                    "unbinned_final": _unbinned_count(state),
                },
                contact_counters=_counter_delta(
                    recruit_counter_before,
                    _contact_counters(contact_index),
                ),
            )

        refine_meta: dict[str, Any]
        with stage_log.stage(
            "finalize",
            message="writing final assignments, QC, actions, and metadata",
            state_version=state.version,
        ) as report:
            refined_rows = _build_refined_rows(
                state=state,
                contact_index=contact_index,
                assignment_stage=assignment_stage,
                assignment_reason=assignment_reason,
            )
            unbinned_rows = _build_unbinned_rows(
                state=state,
                contact_index=contact_index,
                details=unbinned_detail,
            )
            bin_qc_rows = _build_bin_qc_rows(
                state=state,
                profiles=profiles,
            )
            write_tsv_rows(
                layout.refined_bins_tsv,
                FINAL_BINS_COLUMNS,
                refined_rows,
            )
            write_tsv_rows(
                layout.unbinned_tsv,
                UNBINNED_COLUMNS,
                unbinned_rows,
            )
            write_tsv_rows(
                layout.bin_qc_tsv,
                BIN_QC_COLUMNS,
                bin_qc_rows,
            )
            write_tsv_rows(
                layout.refine_actions_tsv,
                REFINE_ACTIONS_COLUMNS,
                action_rows,
            )
            refine_meta = _build_refine_meta(
                contigs_fasta=contigs_fasta,
                coarse_bins_tsv=coarse_bins_tsv,
                contacts_parquet=contacts_parquet,
                coverage_tsv=coverage_tsv,
                embedding_tsv=embedding_tsv,
                layout=layout,
                state=state,
                profiles=profiles,
                contact_index=contact_index,
                stage_counts=stage_counts,
                action_rows=action_rows,
                scg_result=scg_result,
                enable_scg=enable_scg,
                hypergraph_weight_eta=float(hypergraph_weight_eta),
                n_bins_in=n_bins_in,
                n_unbinned_in=n_unbinned_in,
                n_unbinned_final=len(unbinned_rows),
                stage_durations=stage_log.stage_durations,
            )
            write_json(layout.refine_meta_json, refine_meta)
            report.update(
                message="final refine outputs written",
                state_version=state.version,
                counts={
                    "bins": len(bin_qc_rows),
                    "assigned_contigs": len(refined_rows),
                    "unbinned_contigs": len(unbinned_rows),
                    "action_records": len(action_rows),
                },
                contact_counters=_contact_counters(contact_index),
            )

        refine_meta["stage_durations_seconds"] = dict(
            stage_log.stage_durations
        )
        write_json(layout.refine_meta_json, refine_meta)
        stage_log.emit(
            "run_complete",
            stage="refine",
            message="replacement refinement completed",
            state_version=state.version,
            counts={
                "bins": _nonempty_bin_count(state),
                "assigned_contigs": state.n_contigs
                - _unbinned_count(state),
                "unbinned_contigs": _unbinned_count(state),
            },
            contact_counters=_contact_counters(contact_index),
        )
        return RefineRunResult(
            bins_refined_tsv=layout.refined_bins_tsv,
            unbinned_tsv=layout.unbinned_tsv,
            bin_qc_tsv=layout.bin_qc_tsv,
            refine_actions_tsv=layout.refine_actions_tsv,
            refine_stage_log_jsonl=layout.refine_stage_log_jsonl,
            refine_meta_json=layout.refine_meta_json,
            implemented=True,
        )
    except Exception as exc:
        stage_log.emit(
            "run_error",
            stage="refine",
            message=f"replacement refinement failed: {exc}",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        raise
    finally:
        stage_log.close()


def _validate_coarse_assignment(
    assignment: dict[str, str],
    *,
    known_contigs: set[str],
) -> None:
    unknown = sorted(set(assignment) - known_contigs)
    if unknown:
        preview = ", ".join(unknown[:5])
        raise ValueError(
            "Coarse bins contain contigs absent from FASTA: "
            f"{preview}"
        )
    invalid: list[str] = []
    for contig_name, raw_bin in assignment.items():
        if not raw_bin:
            continue
        try:
            bin_idx = int(raw_bin)
        except ValueError:
            invalid.append(f"{contig_name}={raw_bin}")
            continue
        if bin_idx < 0:
            invalid.append(f"{contig_name}={raw_bin}")
    if invalid:
        raise ValueError(
            "Replacement refine requires non-negative integer coarse bin IDs; "
            f"invalid={', '.join(invalid[:5])}"
        )


def _clear_previous_refine_outputs(layout: Any) -> None:
    for path in (
        layout.refined_bins_tsv,
        layout.unbinned_tsv,
        layout.bin_qc_tsv,
        layout.refine_actions_tsv,
        layout.refine_stage_log_jsonl,
        layout.refine_meta_json,
        # Remove stale artifacts written by the retired refine implementation.
        layout.final_dir / "refine_action_features.tsv",
        layout.final_dir / "refine_action_scores.tsv",
        layout.final_dir / "refine_embedding_scores.tsv",
    ):
        path.unlink(missing_ok=True)


def _build_integer_assignment(
    *,
    contig_names: tuple[str, ...],
    coarse_assignment: dict[str, str],
) -> tuple[np.ndarray, int]:
    assignment = np.full(len(contig_names), -1, dtype=np.int32)
    for contig_idx, contig_name in enumerate(contig_names):
        raw_bin = coarse_assignment.get(contig_name, "")
        if raw_bin:
            assignment[contig_idx] = int(raw_bin)
    n_bins = (
        int(assignment[assignment >= 0].max()) + 1
        if bool(np.any(assignment >= 0))
        else 0
    )
    return assignment, n_bins


def _empty_decision_counts() -> dict[str, int]:
    return {"accepted": 0, "rejected": 0, "abstained": 0}


def _count_decision(counts: dict[str, int], decision: Decision) -> None:
    if decision.status is DecisionStatus.ACCEPT:
        counts["accepted"] += 1
    elif decision.status is DecisionStatus.REJECT:
        counts["rejected"] += 1
    else:
        counts["abstained"] += 1


def _mark_split_assignment_metadata(
    candidate: SplitCandidate,
    *,
    assignment_stage: list[str],
    assignment_reason: list[str],
) -> None:
    if not candidate.child_members:
        return
    for contig_idx in candidate.child_members[0]:
        assignment_stage[contig_idx] = "split"
        assignment_reason[contig_idx] = "split_retained_source_child"
    for child in candidate.child_members[1:]:
        for contig_idx in child:
            assignment_stage[contig_idx] = "split"
            assignment_reason[contig_idx] = "split_created_new_child"


def _build_refined_rows(
    *,
    state: RefineState,
    contact_index: ContactIndex,
    assignment_stage: list[str],
    assignment_reason: list[str],
) -> list[tuple[object, ...]]:
    return [
        (
            contact_index.contig_names[contig_idx],
            int(bin_idx),
            assignment_stage[contig_idx],
            assignment_reason[contig_idx],
        )
        for contig_idx, bin_idx in enumerate(state.assignment)
        if int(bin_idx) >= 0
    ]


def _build_unbinned_rows(
    *,
    state: RefineState,
    contact_index: ContactIndex,
    details: dict[int, dict[str, str]],
) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    for contig_idx, bin_idx in enumerate(state.assignment):
        if int(bin_idx) >= 0:
            continue
        detail = details.get(
            contig_idx,
            {
                "stage": "refine",
                "reason": "unresolved_after_refine",
                "source_bin": "",
                "note": "no conservative refine action was accepted",
            },
        )
        rows.append(
            (
                contact_index.contig_names[contig_idx],
                detail["stage"],
                detail["reason"],
                detail["source_bin"],
                detail["note"],
            )
        )
    return rows


def _build_bin_qc_rows(
    *,
    state: RefineState,
    profiles: EvidenceProfileState,
) -> list[tuple[object, ...]]:
    rows: list[tuple[object, ...]] = []
    for bin_idx in range(state.n_bins):
        if not state.bin_members(bin_idx):
            continue
        profile = profiles.profile(bin_idx)
        scg = profile.scg
        if scg.duplicate_burden > 0:
            scg_status = "scg_duplicate_present"
            refine_status = "unresolved_scg_duplication"
            notes = "duplicated_scg_remains_after_conservative_refine"
        elif scg.unique_marker_count > 0:
            scg_status = "scg_clean"
            refine_status = "stable_after_refine"
            notes = "no_duplicated_scg"
        else:
            scg_status = "no_scg"
            refine_status = "stable_after_refine"
            notes = "no_scg_observed"
        median_coverage = (
            ""
            if profile.coverage.log_median is None
            else f"{math.expm1(profile.coverage.log_median):.6g}"
        )
        rows.append(
            (
                bin_idx,
                profile.member_count,
                profile.total_length,
                median_coverage,
                f"{float(state.contact_coherence[bin_idx]):.6g}",
                scg_status,
                len(scg.duplicate_marker_ids),
                int(scg.duplicate_burden > 0),
                refine_status,
                notes,
            )
        )
    return rows


def _split_candidate_row(
    candidate: SplitCandidate,
) -> tuple[object, ...]:
    return (
        "split",
        "",
        candidate.source_bin_idx,
        candidate.source_bin_idx,
        "",
        candidate.reason,
        0,
        "",
        "",
        "uninformative",
        _compact_json(_split_details(candidate)),
    )


def _recruit_candidate_row(
    candidate: RecruitCandidate,
    *,
    contig_name: str,
) -> tuple[object, ...]:
    return (
        "recruit",
        contig_name,
        "",
        "",
        (
            ""
            if candidate.contact_target_bin_idx is None
            else candidate.contact_target_bin_idx
        ),
        candidate.reason,
        0,
        "",
        "",
        "uninformative",
        _compact_json(_recruit_details(candidate)),
    )


def _decision_row(
    decision: Decision,
    *,
    contig_id: str = "",
    source_bin: str = "",
    target_bin: str = "",
    extra: dict[str, Any] | None = None,
) -> tuple[object, ...]:
    scg_status = (
        decision.evidence.scg.status.value
        if decision.evidence.scg is not None
        else "uninformative"
    )
    return (
        decision.proposal.action_type.value,
        contig_id,
        source_bin,
        source_bin,
        target_bin,
        decision.reason_code,
        int(decision.status is DecisionStatus.ACCEPT),
        "",
        "",
        scg_status,
        _compact_json(_decision_details(decision, extra=extra)),
    )


def _decision_details(
    decision: Decision,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "decision": decision.status.value,
        "reason": decision.reason,
        "gates": [
            {
                "name": gate.name,
                "status": gate.status.value,
                "reason": gate.reason,
                "value": gate.value,
            }
            for gate in decision.gates
        ],
    }
    if extra:
        details["candidate"] = extra
    contact = decision.evidence.contact
    if contact is not None:
        details["contact"] = {
            "affected_edge_count": contact.affected_edge_count,
            "split_separation": contact.split_separation,
            "split_within_support": contact.split_within_support,
            "split_cross_support": contact.split_cross_support,
            "merge_pair_support": contact.merge_pair_support,
            "merge_normalized_support": contact.merge_normalized_support,
            "merge_supporting_edge_count": (
                contact.merge_supporting_edge_count
            ),
        }
    scg = decision.evidence.scg
    if scg is not None:
        details["scg"] = {
            "duplicate_burden_before": scg.duplicate_burden_before,
            "duplicate_burden_after": scg.duplicate_burden_after,
            "duplicate_change": scg.duplicate_change,
            "new_duplicate_markers": scg.new_duplicate_markers,
            "resolved_duplicate_markers": scg.resolved_duplicate_markers,
        }
    return details


def _split_details(candidate: SplitCandidate) -> dict[str, Any]:
    return {
        "status": candidate.status.value,
        "reason": candidate.reason,
        "marker_id": candidate.marker_id,
        "seed_contigs": candidate.seed_contig_indices,
        "child_sizes": tuple(len(child) for child in candidate.child_members),
        "inertia": candidate.inertia,
        "iterations": candidate.iterations,
    }


def _merge_details(candidate: MergeCandidate) -> dict[str, Any]:
    return {
        "pair": candidate.pair,
        "raw_support": candidate.raw_support,
        "normalized_support": candidate.normalized_support,
        "supporting_edge_count": candidate.supporting_edge_count,
    }


def _recruit_details(candidate: RecruitCandidate) -> dict[str, Any]:
    return {
        "status": candidate.status.value,
        "reason": candidate.reason,
        "contig_idx": candidate.contig_idx,
        "contact_target_bin": candidate.contact_target_bin_idx,
        "latent_target_bin": candidate.latent_target_bin_idx,
        "target_support": candidate.target_support,
        "runner_up_support": candidate.runner_up_support,
        "target_share": candidate.target_share,
        "target_margin": candidate.target_margin,
        "supporting_edge_count": candidate.target_edge_count,
        "latent_distance": candidate.latent_distance,
        "target_radius": candidate.target_radius,
    }


def _compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )


def _contact_counters(index: ContactIndex) -> dict[str, int]:
    counters = index.counters
    return {
        "full_edge_scans": int(counters.full_edge_scans),
        "full_edges_visited": int(counters.full_edges_visited),
        "local_queries": int(counters.local_queries),
        "local_edges_visited": int(counters.local_edges_visited),
    }


def _counter_delta(
    before: dict[str, int],
    after: dict[str, int],
) -> dict[str, int]:
    return {
        key: int(after[key] - before.get(key, 0))
        for key in after
    }


def _nonempty_bin_count(state: RefineState) -> int:
    return sum(
        bool(state.bin_members(bin_idx))
        for bin_idx in range(state.n_bins)
    )


def _unbinned_count(state: RefineState) -> int:
    return int(np.count_nonzero(state.assignment == -1))


def _build_refine_meta(
    *,
    contigs_fasta: Path,
    coarse_bins_tsv: Path,
    contacts_parquet: Path,
    coverage_tsv: Path,
    embedding_tsv: Path | None,
    layout: Any,
    state: RefineState,
    profiles: EvidenceProfileState,
    contact_index: ContactIndex,
    stage_counts: dict[str, dict[str, int]],
    action_rows: list[tuple[object, ...]],
    scg_result: Any,
    enable_scg: bool,
    hypergraph_weight_eta: float,
    n_bins_in: int,
    n_unbinned_in: int,
    n_unbinned_final: int,
    stage_durations: dict[str, float],
) -> dict[str, Any]:
    final_profiles = [
        profiles.profile(bin_idx)
        for bin_idx in range(state.n_bins)
        if state.bin_members(bin_idx)
    ]
    return {
        "stage": "bin_refinement",
        "engine": "replacement_refinement",
        "implemented": True,
        "checkpoint_enabled": False,
        "inputs": {
            "contigs_fasta": str(contigs_fasta),
            "coarse_bins_tsv": str(coarse_bins_tsv),
            "contacts_parquet": str(contacts_parquet),
            "coverage_tsv": str(coverage_tsv),
            "embedding_tsv": (
                str(embedding_tsv) if embedding_tsv is not None else None
            ),
            "contact_weight_mode": "hypergraph_native",
            "hypergraph_weight_eta": hypergraph_weight_eta,
        },
        "outputs": {
            "bins_refined_tsv": str(layout.refined_bins_tsv),
            "unbinned_tsv": str(layout.unbinned_tsv),
            "bin_qc_tsv": str(layout.bin_qc_tsv),
            "refine_actions_tsv": str(layout.refine_actions_tsv),
            "refine_stage_log_jsonl": str(
                layout.refine_stage_log_jsonl
            ),
            "refine_meta_json": str(layout.refine_meta_json),
            "scg_dir": (
                str(scg_result.scg_dir)
                if scg_result is not None
                else None
            ),
        },
        "stage_order": ["split", "merge", "recruit"],
        "stage_counts": stage_counts,
        "stage_durations_seconds": dict(stage_durations),
        "n_bins_in": int(n_bins_in),
        "n_bins_out": len(final_profiles),
        "n_unbinned_in": int(n_unbinned_in),
        "n_unbinned_final": int(n_unbinned_final),
        "n_actions_recorded": len(action_rows),
        "n_actions_accepted": sum(
            int(row[6]) == 1 for row in action_rows
        ),
        "n_bins_with_duplicate_scg": sum(
            profile.scg.duplicate_burden > 0
            for profile in final_profiles
        ),
        "scg_enabled": bool(enable_scg),
        "scg_cache_status": (
            scg_result.cache_status
            if scg_result is not None
            else "disabled"
        ),
        "scg_panel": (
            scg_result.panel.metadata()
            if scg_result is not None
            else None
        ),
        "contact_index": {
            "n_edges": contact_index.n_edges,
            "n_incidences": contact_index.n_incidences,
            "counters": _contact_counters(contact_index),
        },
        "profile_counters": {
            "bin_profiles_built": profiles.counters.bin_profiles_built,
            "contigs_visited": profiles.counters.contigs_visited,
        },
        "notes": {
            "actions": "split_then_merge_then_recruit",
            "decision_rule": (
                "ordered structural, SCG, Pore-C, and HG-VAE gates"
            ),
            "tnf_coverage_role": (
                "profile and QC evidence; HG-VAE is the action-level "
                "representation"
            ),
            "no_checkpoint": (
                "interrupted refinement is rerun from coarse assignments "
                "and reusable SCG evidence"
            ),
        },
    }
