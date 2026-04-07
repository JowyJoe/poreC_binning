"""
Active refine mainline for the current porebin architecture.

Public role:
  - consume coarse initial bins
  - perform post-binning refinement
  - write final `bins.refined.tsv`
  - write `residual_pool.tsv` for downstream association

This module still contains transitional compatibility helpers and historical
heuristics, but they are not the public method contract. Downstream relation
mining belongs to `porebin.associate`, not to `refine`.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional

from porebin import __version__
from porebin.build_graph import order_norm
from porebin.contact_hypergraph import ContactHypergraphError, iter_canonical_contact_rows
from porebin.export import MIN_BIN_BP
from porebin.refine_qc import (
    annotate_split_check_flags,
    annotate_suspect_flags,
    compute_bin_qc_rows,
    compute_contact_component_qc,
    select_suspect_bins,
    select_split_check_bins,
    write_bin_qc_tsv,
    write_suspect_bins_tsv,
)
from porebin.refine_state import (
    RecruitCandidate,
    RecruitDecision,
    ReassignCandidate,
    ReassignDecision,
    ReassignWindow,
    RefineActionRecord,
    RefineState,
    ResidualRecord,
    SplitCandidate,
    SplitDecision,
)
from porebin.scg import (
    ScgError,
    compute_bin_scg_qc,
    ensure_scg_hits,
    read_scg_hits,
    resolve_scg_resources,
)
from porebin.utils import (
    ensure_dir,
    iter_fasta_records,
    utc_now_iso,
    write_json,
)


class RefineError(RuntimeError):
    pass


@dataclass
class RefineStats:
    contigs_total: int = 0
    bins_total: int = 0
    bins_kept_initial: int = 0
    bins_kept_final: int = 0
    contigs_short: int = 0
    contigs_assigned_initial: int = 0
    contigs_assigned_final: int = 0

    reads_total: int = 0
    reads_kept: int = 0
    reads_informative: int = 0
    reads_skipped_k_lt_2: int = 0
    reads_rejected_anchor_sparse: int = 0
    reads_rejected_anchor_conflict: int = 0
    input_sorted_by_readid: bool = True

    hard_anchors_total: int = 0
    decontam_removed: int = 0
    reassign_moved: int = 0
    reassign_unbinned: int = 0
    recruit_assigned: int = 0
    split_triggered: int = 0
    split_applied: int = 0
    contigs_relation_only: int = 0
    contigs_abstained: int = 0
    contigs_residual_total: int = 0
    contigs_residual_associate_candidate: int = 0


LEGACY_HELPER_ROLES: dict[str, str] = {
    "_split_bins_parquet": "candidate_generator_only_not_applied_in_active_orchestration",
    "_reassign_or_unbin": "compatibility_helper_not_in_active_orchestration",
    "_recruit_gmm": "compatibility_helper_not_in_active_orchestration",
    "_legacy_contig_state_bridge": "compatibility_bridge_for_scores_and_residual_mapping",
}


@dataclass(frozen=True)
class PreCleanupResult:
    cleaned_assignment: dict[str, str]
    anchor_mask: dict[str, bool]
    hard_anchor_mask: dict[str, bool]
    anchor_weight: dict[str, float]
    bin_status: dict[str, str]
    bin_weight: dict[str, float]
    anchor_count: dict[str, int]
    hard_anchor_count: dict[str, int]
    removed_contigs: tuple[str, ...]
    removed_by_bin: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class ReadAnchorEvidence:
    informative: bool
    reject_reason: str
    host_mass: dict[str, float]
    hard_anchor_hits: int
    top1_host: str
    top1_mass: float
    top2_host: str
    top2_mass: float


def _build_refine_state_from_assignment(
    *,
    contig_len: dict[str, int],
    assignment: dict[str, str],
    coverage: Optional[dict[str, float]],
) -> RefineState:
    min_contig_len, _meta = auto_min_contig_len(contig_len)
    contig_to_bin: dict[str, str] = {}
    for contig, bin_id in assignment.items():
        if contig not in contig_len:
            continue
        b = str(bin_id).strip() if bin_id is not None else "-1"
        if not b or b == "-1":
            continue
        contig_to_bin[contig] = b
    state = RefineState(
        contig_len=dict(contig_len),
        min_contig_len=int(min_contig_len),
        coarse_raw=dict(assignment),
        contig_to_bin=contig_to_bin,
        coverage=(dict(coverage) if coverage is not None else None),
        bin_cov_stats=None,
    )
    if coverage is not None:
        state.bin_cov_stats = _bin_coverage_stats(state.rebuild_bin_to_contigs(), coverage)
    return state


def _collect_refine_qc_snapshot(
    *,
    state: RefineState,
    contacts_parquet: Path,
    scg_hits_tsv: Optional[Path],
    scg_expected_markers: Optional[set[str]],
    scg_status: str,
) -> dict[str, Any]:
    intra: dict[str, float] = {}
    other: dict[str, float] = {}
    if state.contig_to_bin:
        intra, other, _affinity, _meta = _scan_contacts_support_and_affinity_parquet(
            contacts_parquet=contacts_parquet,
            contig_len=state.contig_len,
            min_contig_len=state.min_contig_len,
            contig_to_bin=state.contig_to_bin,
        )

    contact_qc = compute_contact_component_qc(
        contacts_parquet=contacts_parquet,
        contig_to_bin=state.contig_to_bin,
        contig_len=state.contig_len,
        min_contig_len=state.min_contig_len,
    )

    scg_hits = read_scg_hits(scg_hits_tsv) if scg_hits_tsv is not None and scg_hits_tsv.exists() else []
    bin_scg_qc = compute_bin_scg_qc(
        contig_to_bin=state.contig_to_bin,
        hits=scg_hits,
        expected_markers=scg_expected_markers,
    )

    rows = compute_bin_qc_rows(
        contig_to_bin=state.contig_to_bin,
        contig_len=state.contig_len,
        intra_support=intra,
        other_support=other,
        bin_cov_stats=state.bin_cov_stats,
        contact_component_qc=contact_qc,
        bin_scg_qc=bin_scg_qc,
        scg_status=scg_status,
        scg_expected_markers_count=(len(scg_expected_markers) if scg_expected_markers is not None else 0),
    )
    suspects = select_suspect_bins(rows)
    rows = annotate_suspect_flags(rows, suspects)
    split_checks = select_split_check_bins(rows)
    rows = annotate_split_check_flags(rows, split_checks)
    return {
        "rows": rows,
        "suspects_records": suspects,
        "split_checks_records": split_checks,
        "bins": int(len(rows)),
        "suspects": int(len(suspects)),
        "split_checks": int(len(split_checks)),
        "scg_status": scg_status,
        "intra_support": intra,
        "other_support": other,
    }


def _write_refine_qc_snapshot(
    *,
    snapshot_name: str,
    state: RefineState,
    contacts_parquet: Path,
    out_dir: Path,
    scg_hits_tsv: Optional[Path],
    scg_expected_markers: Optional[set[str]],
    scg_status: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    out_dir = out_dir.resolve()
    bin_qc_path = out_dir / f"bin_qc.{snapshot_name}.tsv"
    suspect_path = out_dir / f"suspect_bins.{snapshot_name}.tsv"
    meta = _collect_refine_qc_snapshot(
        state=state,
        contacts_parquet=contacts_parquet,
        scg_hits_tsv=scg_hits_tsv,
        scg_expected_markers=scg_expected_markers,
        scg_status=scg_status,
    )
    rows = meta["rows"]
    suspects = meta["suspects_records"]
    write_bin_qc_tsv(bin_qc_path, rows)
    write_suspect_bins_tsv(suspect_path, suspects)
    logger.info("Refine QC snapshot '%s': bins=%s suspects=%s", snapshot_name, len(rows), len(suspects))
    return {
        "bin_qc_tsv": str(bin_qc_path),
        "suspect_bins_tsv": str(suspect_path),
        **meta,
    }


def _preflight_refine_scg_requirements(
    *,
    db_dir: Optional[Path] = None,
    prodigal_exe: str = "prodigal",
    hmmsearch_exe: str = "hmmsearch",
) -> None:
    try:
        resolve_scg_resources(db_dir=db_dir)
    except ScgError as exc:
        raise RefineError(
            "refine requires SCG resources. Missing or invalid manifest/HMM configuration: "
            f"{exc}"
        ) from exc

    missing_tools: list[str] = []
    if shutil.which(prodigal_exe) is None:
        missing_tools.append(prodigal_exe)
    if shutil.which(hmmsearch_exe) is None:
        missing_tools.append(hmmsearch_exe)
    if missing_tools:
        raise RefineError(
            "refine requires SCG tooling before it can run. Missing executables: "
            + ", ".join(missing_tools)
        )


def _legacy_residual_record(
    *,
    contig_name: str,
    coarse_host: str,
    refine_state: str,
    abstain_reason: str,
    top1_host: str,
    top2_host: str,
    informative_reads: int,
    has_contact_support: bool,
    contig_len: dict[str, int],
    min_contig_len: int,
) -> tuple[str, str, str, str]:
    """
    Map legacy contig-state compatibility labels into residual-pool records.

    This helper is transitional glue. It must not be treated as the public refine
    semantics once refine becomes fully operation-native.
    """
    coarse_host = str(coarse_host).strip() if coarse_host is not None else "-1"
    stage = "refine_base"
    reason = "assignment_unresolved"
    refined_status = "residual_unresolved"

    if int(contig_len.get(contig_name, 0)) < int(min_contig_len):
        stage = "input_filter"
        reason = "short_contig"
        refined_status = "residual_filtered"
    elif refine_state == "relation_only":
        reason = "associate_candidate"
        refined_status = "residual_associate_candidate"
    elif abstain_reason == "conflicting_evidence":
        reason = "contact_inconsistency"
        refined_status = "residual_unresolved"
    elif abstain_reason == "unreliable_background":
        reason = "low_contact_support"
        refined_status = "residual_filtered"
    elif abstain_reason == "insufficient_information":
        reason = "assignment_unresolved"
        refined_status = "residual_unresolved"
    elif not has_contact_support:
        reason = "low_contact_support"
        refined_status = "residual_unresolved"

    note = (
        f"legacy_refine_state={refine_state};"
        f"top1_host={top1_host};"
        f"top2_host={top2_host};"
        f"informative_reads={informative_reads}"
    )
    return reason, stage, refined_status, note


def _compat_bridge_row(
    *,
    contig_name: str,
    coarse_host: str,
    bin_status: str,
    top1_host: str,
    top2_host: str,
    top1_score: float,
    top2_score: float,
    margin: float,
    entropy: float,
    effective_hosts: float,
    has_contact_support: bool,
    is_core_like: bool,
    is_ambiguous: bool,
    is_accessory_candidate: bool,
    refine_state: str,
    abstain_reason: str,
    informative_reads: int,
    reject_sparse: int,
    reject_conflict: int,
    is_hard_anchor: bool,
) -> dict[str, Any]:
    """
    Build one row in the legacy contig-host compatibility bridge.

    This structure is retained only for auditability and the temporary handoff to
    downstream association while the new bin-centric refine core is still being
    phased in.
    """
    return {
        "contig_name": contig_name,
        "coarse_host": coarse_host,
        "bin_status": bin_status,
        "top1_host": top1_host,
        "top2_host": top2_host,
        "top1_score": float(top1_score),
        "top2_score": float(top2_score),
        "margin": float(margin),
        "entropy": float(entropy),
        "effective_hosts": float(effective_hosts),
        "has_contact_support": bool(has_contact_support),
        "is_core_like": bool(is_core_like),
        "is_ambiguous": bool(is_ambiguous),
        "is_accessory_candidate": bool(is_accessory_candidate),
        "refine_state": refine_state,
        "abstain_reason": abstain_reason,
        "informative_reads": int(informative_reads),
        "read_reject_anchor_sparse": int(reject_sparse),
        "read_reject_anchor_conflict": int(reject_conflict),
        "is_hard_anchor": bool(is_hard_anchor),
    }


def _write_contig_host_scores_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write the transitional per-contig compatibility bridge for audit/debug use."""
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(
            "contig_name\tcoarse_host\tbin_status\ttop1_host\ttop2_host\t"
            "top1_score\ttop2_score\tmargin\tentropy\teffective_hosts\t"
            "has_contact_support\tis_core_like\tis_ambiguous\tis_accessory_candidate\t"
            "refine_state\tabstain_reason\tinformative_reads\t"
            "read_reject_anchor_sparse\tread_reject_anchor_conflict\tis_hard_anchor\n"
        )
        for row in rows:
            fh.write(
                f"{row['contig_name']}\t{row['coarse_host']}\t{row['bin_status']}\t{row['top1_host']}\t{row['top2_host']}\t"
                f"{float(row['top1_score']):.6g}\t{float(row['top2_score']):.6g}\t{float(row['margin']):.6g}\t"
                f"{float(row['entropy']):.6g}\t{float(row['effective_hosts']):.6g}\t"
                f"{1 if row['has_contact_support'] else 0}\t"
                f"{1 if row['is_core_like'] else 0}\t{1 if row['is_ambiguous'] else 0}\t"
                f"{1 if row['is_accessory_candidate'] else 0}\t{row['refine_state']}\t{row['abstain_reason']}\t"
                f"{int(row['informative_reads'])}\t{int(row['read_reject_anchor_sparse'])}\t"
                f"{int(row['read_reject_anchor_conflict'])}\t{1 if row['is_hard_anchor'] else 0}\n"
            )


def _derive_final_assignment_and_residuals(
    *,
    compat_rows: list[dict[str, Any]],
    cleanup: PreCleanupResult,
    contig_len: dict[str, int],
    min_contig_len: int,
) -> tuple[dict[str, str], list[ResidualRecord], dict[str, int], set[str]]:
    """
    Derive the public refine outputs from the transitional legacy compatibility bridge.

    The public artifacts are:
      - bins.refined.tsv
      - residual_pool.tsv

    The bridge remains internal and temporary.
    """
    final_assignment: dict[str, str] = {}
    residual_rows: list[ResidualRecord] = []
    final_bins_used: set[str] = set()

    counters = {
        "assigned": 0,
        "relation_only_compat": 0,
        "abstained_compat": 0,
        "ambiguous_compat": 0,
        "residual_total": 0,
        "residual_associate_candidate": 0,
    }

    for row in compat_rows:
        contig_name = str(row["contig_name"])
        top1_host = str(row["top1_host"])
        coarse_host = str(row["coarse_host"])
        refine_state = str(row["refine_state"])
        abstain_reason = str(row["abstain_reason"])
        informative_reads = int(row["informative_reads"])
        has_contact_support = bool(row["has_contact_support"])
        top2_host = str(row["top2_host"])
        is_core_like = bool(row["is_core_like"])
        is_ambiguous = bool(row["is_ambiguous"])

        if is_core_like and top1_host != "-1" and float(cleanup.bin_weight.get(top1_host, 0.0)) > 0.0:
            final_assignment[contig_name] = top1_host
            final_bins_used.add(top1_host)
            counters["assigned"] += 1
        else:
            reason, stage, refined_status, note = _legacy_residual_record(
                contig_name=contig_name,
                coarse_host=coarse_host,
                refine_state=refine_state,
                abstain_reason=abstain_reason,
                top1_host=top1_host,
                top2_host=top2_host,
                informative_reads=informative_reads,
                has_contact_support=has_contact_support,
                contig_len=contig_len,
                min_contig_len=min_contig_len,
            )
            residual_rows.append(
                ResidualRecord(
                    contig_name=contig_name,
                    reason=reason,
                    stage=stage,
                    coarse_bin_id=coarse_host,
                    refined_status=refined_status,
                    note=note,
                )
            )
            counters["residual_total"] += 1
            if refined_status == "residual_associate_candidate":
                counters["residual_associate_candidate"] += 1

        if refine_state == "relation_only":
            counters["relation_only_compat"] += 1
        elif not is_core_like:
            counters["abstained_compat"] += 1
        if is_ambiguous:
            counters["ambiguous_compat"] += 1

    return final_assignment, residual_rows, counters, final_bins_used


def _write_bins_refined_tsv(path: Path, assignment: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("contig_name\tbin_id\n")
        for contig_name, bin_id in sorted(assignment.items(), key=lambda kv: kv[0]):
            fh.write(f"{contig_name}\t{bin_id}\n")


def _build_legacy_compatibility_rows(
    *,
    contigs_all: list[str],
    coarse_raw: dict[str, str],
    cleanup: PreCleanupResult,
    support: dict[str, dict[str, float]],
    informative_reads_by_contig: dict[str, int],
    raw_reads_by_contig: dict[str, int],
    rejected_sparse_by_contig: dict[str, int],
    rejected_conflict_by_contig: dict[str, int],
    B: list[str],
    B_set: set[str],
    B_count: int,
    prior_strength: float,
    eps: float,
    assign_top1_score_thresh: float,
    assign_margin_thresh: float,
    assign_eff_hosts_thresh: float,
    min_informative_reads_assign: int,
    min_informative_reads_relation: int,
) -> list[dict[str, Any]]:
    """
    Transitional compatibility bridge retained during the refine refactor.

    This helper keeps legacy contig-state logic out of the public refine control
    flow. It exists only to support:
      - contig_host_scores.tsv
      - compatibility-derived residual mapping

    It must not be treated as the new refine method core.
    """
    compat_rows: list[dict[str, Any]] = []
    for c in contigs_all:
        coarse_host = str(coarse_raw.get(c, "-1")).strip() if c in coarse_raw else "-1"
        bin_status = cleanup.bin_status.get(coarse_host, "none")
        sdict = support.get(c, {})
        informative_reads = int(informative_reads_by_contig.get(c, 0))
        raw_reads = int(raw_reads_by_contig.get(c, 0))
        reject_sparse = int(rejected_sparse_by_contig.get(c, 0))
        reject_conflict = int(rejected_conflict_by_contig.get(c, 0))
        is_hard_anchor = bool(cleanup.hard_anchor_mask.get(c, False))

        sum_support = float(sum(float(v) for v in sdict.values()))
        has_contact_support = bool(sum_support > 0.0)
        coarse_bin_weight = float(cleanup.bin_weight.get(coarse_host, 0.0)) if coarse_host != "-1" else 0.0
        alpha_c = 0.0
        if coarse_host in B_set and (is_hard_anchor or informative_reads > 0):
            alpha_c = float(prior_strength * coarse_bin_weight * (sum_support + eps))

        top1_host = "-1"
        top2_host = "-1"
        top1_score = 0.0
        top2_score = 0.0
        margin = 0.0
        ent = 0.0
        effective_hosts = 0.0

        score_nz: dict[str, float] = {b: float(v) for b, v in sdict.items() if b in B_set and float(v) > 0.0}
        if coarse_host in B_set and alpha_c > 0.0:
            score_nz[coarse_host] = float(score_nz.get(coarse_host, 0.0)) + alpha_c

        sum_score = float(sum_support + (alpha_c if coarse_host in B_set else 0.0))
        denom = float(sum_score + (B_count * eps)) if B_count > 0 else 0.0
        if denom <= 0.0 and B_count > 0:
            denom = float(B_count * eps)

        def theta_of(b: str) -> float:
            if B_count <= 0 or denom <= 0.0:
                return 0.0
            return float((float(score_nz.get(b, 0.0)) + eps) / denom)

        ranked_nz: list[str] = [bb for bb, _v in sorted(score_nz.items(), key=lambda kv: (-kv[1], kv[0]))]
        if ranked_nz:
            ranked: list[str] = list(ranked_nz)
            if len(ranked) < 3:
                for bb in B:
                    if bb in score_nz:
                        continue
                    ranked.append(bb)
                    if len(ranked) >= 3:
                        break
            top1_host = ranked[0]
            top2_host = ranked[1] if B_count >= 2 else ""
            top1_score = theta_of(top1_host)
            top2_score = theta_of(top2_host) if top2_host else 0.0
            margin = float(top1_score - top2_score)

            theta0 = float(eps / denom)
            ent = 0.0
            for _bb, sc in score_nz.items():
                th = float((float(sc) + eps) / denom)
                ent -= th * math.log(th + eps)
            n0 = int(B_count - len(score_nz))
            if n0 > 0:
                ent -= float(n0) * theta0 * math.log(theta0 + eps)
            effective_hosts = float(math.exp(ent))

        legacy_refine_state = "abstain_insufficient_information"
        assignable = bool(coarse_host in B_set and coarse_host != "-1")
        if is_hard_anchor and assignable:
            legacy_refine_state = "assigned_bin"
        elif (
            assignable
            and has_contact_support
            and informative_reads >= min_informative_reads_assign
            and top1_host == coarse_host
            and top1_score >= assign_top1_score_thresh
            and margin >= assign_margin_thresh
            and effective_hosts <= assign_eff_hosts_thresh
        ):
            legacy_refine_state = "assigned_bin"
        elif (
            has_contact_support
            and top1_host != "-1"
            and (informative_reads >= min_informative_reads_relation or reject_conflict > 0)
        ):
            legacy_refine_state = "relation_only"
        elif raw_reads == 0:
            legacy_refine_state = "abstain_insufficient_information"
        elif reject_conflict > max(informative_reads, reject_sparse):
            legacy_refine_state = "abstain_conflicting_evidence"
        elif reject_sparse > 0:
            legacy_refine_state = "abstain_unreliable_background"

        abstain_reason = ""
        if legacy_refine_state.startswith("abstain_"):
            abstain_reason = legacy_refine_state.removeprefix("abstain_")

        is_core_like = bool(legacy_refine_state == "assigned_bin")
        is_accessory = bool(legacy_refine_state == "relation_only")
        is_ambiguous = bool(not is_core_like)
        compat_rows.append(
            _compat_bridge_row(
                contig_name=c,
                coarse_host=coarse_host,
                bin_status=bin_status,
                top1_host=top1_host,
                top2_host=top2_host,
                top1_score=top1_score,
                top2_score=top2_score,
                margin=margin,
                entropy=ent,
                effective_hosts=effective_hosts,
                has_contact_support=has_contact_support,
                is_core_like=is_core_like,
                is_ambiguous=is_ambiguous,
                is_accessory_candidate=is_accessory,
                refine_state=legacy_refine_state,
                abstain_reason=abstain_reason,
                informative_reads=informative_reads,
                reject_sparse=reject_sparse,
                reject_conflict=reject_conflict,
                is_hard_anchor=is_hard_anchor,
            )
        )
    return compat_rows


def _write_residual_pool_tsv(path: Path, residual_rows: list[ResidualRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("contig_name\treason\tstage\tcoarse_bin_id\trefined_status\tnote\n")
        for rec in sorted(residual_rows, key=lambda r: r.contig_name):
            fh.write(
                f"{rec.contig_name}\t{rec.reason}\t{rec.stage}\t{rec.coarse_bin_id}\t{rec.refined_status}\t{rec.note}\n"
            )


def _rows_by_bin_id(rows: list[Any]) -> dict[str, Any]:
    return {str(row.bin_id): row for row in rows}


def _evaluate_decontam_scg_gate(
    *,
    before_row: Optional[Any],
    after_row: Optional[Any],
    scg_state: str,
) -> tuple[bool, str, str]:
    if scg_state == "disabled":
        return False, "not_available", "scg_unavailable"
    if before_row is None or after_row is None:
        return True, "neutral", "scg_bin_qc_missing"
    if before_row.unique_scg is None or after_row.unique_scg is None:
        return False, "not_available", "scg_metrics_unavailable"

    positive: list[str] = []
    negative: list[str] = []

    if after_row.duplicated_scg is not None and before_row.duplicated_scg is not None:
        if after_row.duplicated_scg < before_row.duplicated_scg:
            positive.append("duplicated_scg_decreased")
        elif after_row.duplicated_scg > before_row.duplicated_scg:
            negative.append("duplicated_scg_increased")

    if before_row.contamination_like is not None and after_row.contamination_like is not None:
        if after_row.contamination_like < before_row.contamination_like:
            positive.append("contamination_like_decreased")
        elif after_row.contamination_like > before_row.contamination_like:
            negative.append("contamination_like_increased")

    if after_row.unique_scg < before_row.unique_scg and not positive:
        negative.append("unique_scg_decreased_without_scg_gain")

    if negative:
        return True, "veto", ",".join(negative)
    if positive:
        return True, "support", ",".join(positive)
    return True, "neutral", "no_scg_change"


def _evaluate_split_scg_gate(
    *,
    row: Any,
    scg_state: str,
) -> tuple[bool, str, str]:
    if scg_state == "disabled":
        return False, "not_available", "scg_unavailable"
    if row is None:
        return True, "neutral", "split_row_missing"
    if row.duplicated_scg is None and row.contamination_like is None and row.completeness_like is None:
        return False, "not_available", "scg_metrics_unavailable"

    reasons: list[str] = []
    if row.duplicated_scg is not None and row.duplicated_scg > 0:
        reasons.append("duplicated_scg_present")
    if row.contamination_like is not None and row.contamination_like > 0.1:
        reasons.append("contamination_like_high")
    if row.completeness_like is not None and row.completeness_like < 0.5:
        reasons.append("completeness_like_low")

    if reasons:
        return True, "support", ",".join(reasons)
    return True, "neutral", "no_scg_split_signal"


def _apply_scg_gate_to_cleanup(
    *,
    cleanup: PreCleanupResult,
    coarse_qc_meta: dict[str, Any],
    cleaned_qc_meta: dict[str, Any],
    contig_len: dict[str, int],
    coverage_snapshot: Optional[dict[str, float]],
    contacts_parquet: Path,
    out_dir: Path,
    scg_hits_tsv: Optional[Path],
    scg_expected_markers: Optional[set[str]],
    scg_state: str,
    logger: logging.Logger,
) -> tuple[PreCleanupResult, RefineState, dict[str, Any], dict[str, tuple[bool, str, str]]]:
    coarse_rows = _rows_by_bin_id(coarse_qc_meta.get("rows", []))
    cleaned_rows = _rows_by_bin_id(cleaned_qc_meta.get("rows", []))
    gate_by_bin: dict[str, tuple[bool, str, str]] = {}
    veto_bins: set[str] = set()

    for bin_id in sorted(cleanup.removed_by_bin.keys(), key=str):
        gate = _evaluate_decontam_scg_gate(
            before_row=coarse_rows.get(str(bin_id)),
            after_row=cleaned_rows.get(str(bin_id)),
            scg_state=scg_state,
        )
        gate_by_bin[str(bin_id)] = gate
        _used, result, _reason = gate
        if result == "veto":
            veto_bins.add(str(bin_id))

    if not veto_bins:
        cleaned_state = _build_refine_state_from_assignment(
            contig_len=contig_len,
            assignment=cleanup.cleaned_assignment,
            coverage=coverage_snapshot,
        )
        return cleanup, cleaned_state, cleaned_qc_meta, gate_by_bin

    restored_assignment = dict(cleanup.cleaned_assignment)
    kept_removed_by_bin: dict[str, tuple[str, ...]] = {}
    kept_removed: list[str] = []
    for bin_id, contigs in cleanup.removed_by_bin.items():
        if str(bin_id) in veto_bins:
            for contig in contigs:
                restored_assignment[str(contig)] = str(bin_id)
        else:
            kept_removed_by_bin[str(bin_id)] = tuple(sorted(contigs))
            kept_removed.extend(contigs)

    updated_cleanup = replace(
        cleanup,
        cleaned_assignment=restored_assignment,
        removed_by_bin=kept_removed_by_bin,
        removed_contigs=tuple(sorted(str(c) for c in kept_removed)),
    )
    cleaned_state = _build_refine_state_from_assignment(
        contig_len=contig_len,
        assignment=updated_cleanup.cleaned_assignment,
        coverage=coverage_snapshot,
    )
    updated_cleaned_qc_meta = _write_refine_qc_snapshot(
        snapshot_name="cleaned",
        state=cleaned_state,
        contacts_parquet=contacts_parquet,
        out_dir=out_dir,
        scg_hits_tsv=scg_hits_tsv,
        scg_expected_markers=scg_expected_markers,
        scg_status=scg_state,
        logger=logger,
    )
    return updated_cleanup, cleaned_state, updated_cleaned_qc_meta, gate_by_bin


def _build_refine_action_log(
    *,
    cleanup: PreCleanupResult,
    decontam_scg_gate: dict[str, tuple[bool, str, str]],
    split_decisions: list[SplitDecision],
    reassign_decisions: list[ReassignDecision],
    recruit_decisions: list[RecruitDecision],
) -> list[RefineActionRecord]:
    actions: list[RefineActionRecord] = []
    action_idx = 1

    for bin_id in sorted(cleanup.removed_by_bin.keys(), key=str):
        affected = cleanup.removed_by_bin.get(bin_id, ())
        if not affected:
            continue
        actions.append(
            RefineActionRecord(
                action_id=f"action_{action_idx:04d}",
                action_type="decontam",
                target_bin="-1",
                source_bin=str(bin_id),
                affected_contigs=tuple(sorted(affected)),
                accepted=(decontam_scg_gate.get(str(bin_id), (False, "neutral", ""))[1] != "veto"),
                accept_reason=(
                    "pre_cleanup_decontam_applied"
                    if decontam_scg_gate.get(str(bin_id), (False, "neutral", ""))[1] != "veto"
                    else ""
                ),
                reject_reason=(
                    "scg_gate_veto"
                    if decontam_scg_gate.get(str(bin_id), (False, "neutral", ""))[1] == "veto"
                    else ""
                ),
                qc_before_ref="coarse",
                qc_after_ref="cleaned",
                scg_gate_used=bool(decontam_scg_gate.get(str(bin_id), (False, "neutral", ""))[0]),
                scg_gate_result=str(decontam_scg_gate.get(str(bin_id), (False, "neutral", ""))[1]),
                scg_gate_reason=str(decontam_scg_gate.get(str(bin_id), (False, "neutral", ""))[2]),
            )
        )
        action_idx += 1

    for decision in split_decisions:
        affected = (
            tuple(
                sorted(
                    set(decision.candidate.primary_contigs)
                    | set(decision.candidate.secondary_contigs)
                    | set(decision.candidate.residual_contigs)
                )
            )
            if decision.accepted or decision.reject_reason
            else ()
        )
        actions.append(
            RefineActionRecord(
                action_id=f"action_{action_idx:04d}",
                action_type="split",
                target_bin=(decision.new_bin_id if decision.accepted else decision.candidate.source_bin),
                source_bin=decision.candidate.source_bin,
                affected_contigs=affected,
                accepted=decision.accepted,
                accept_reason=decision.accept_reason,
                reject_reason=decision.reject_reason,
                qc_before_ref=decision.qc_before_ref,
                qc_after_ref=decision.qc_after_ref,
                scg_gate_used=decision.scg_gate_used,
                scg_gate_result=decision.scg_gate_result,
                scg_gate_reason=decision.scg_gate_reason,
            )
        )
        action_idx += 1

    for decision in reassign_decisions:
        target_bin = (
            decision.target_bin
            if decision.accepted
            else (decision.candidate.sibling_bin if decision.candidate.candidate_action == "move_to_sibling" else "-1")
        )
        actions.append(
            RefineActionRecord(
                action_id=f"action_{action_idx:04d}",
                action_type="reassign",
                target_bin=target_bin,
                source_bin=decision.candidate.current_bin,
                affected_contigs=(str(decision.candidate.contig_name),),
                accepted=decision.accepted,
                accept_reason=decision.accept_reason,
                reject_reason=decision.reject_reason,
                qc_before_ref=decision.qc_before_ref,
                qc_after_ref=decision.qc_after_ref,
                scg_gate_used=decision.scg_gate_used,
                scg_gate_result=decision.scg_gate_result,
                scg_gate_reason=decision.scg_gate_reason,
            )
        )
        action_idx += 1

    for decision in recruit_decisions:
        target_bin = decision.target_bin if decision.final_action == "assign_to_bin" else "-1"
        actions.append(
            RefineActionRecord(
                action_id=f"action_{action_idx:04d}",
                action_type="recruit",
                target_bin=target_bin,
                source_bin="-1",
                affected_contigs=(str(decision.candidate.contig_name),),
                accepted=decision.accepted,
                accept_reason=decision.accept_reason,
                reject_reason=decision.reject_reason,
                qc_before_ref=decision.qc_before_ref,
                qc_after_ref=decision.qc_after_ref,
                scg_gate_used=decision.scg_gate_used,
                scg_gate_result=decision.scg_gate_result,
                scg_gate_reason=decision.scg_gate_reason,
            )
        )
        action_idx += 1

    if not actions:
        actions.append(
            RefineActionRecord(
                action_id="action_0001",
                action_type="no_op",
                target_bin="-1",
                source_bin="-1",
                affected_contigs=(),
                accepted=True,
                accept_reason="no_candidate_operations",
                reject_reason="",
                qc_before_ref="cleaned",
                qc_after_ref="refined",
                scg_gate_used=False,
                scg_gate_result="not_applicable",
                scg_gate_reason="no_candidate_operations",
            )
        )
    return actions


def _write_refine_actions_tsv(path: Path, actions: list[RefineActionRecord]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(
            "action_id\taction_type\ttarget_bin\tsource_bin\taffected_contigs\taccepted\t"
            "accept_reason\treject_reason\tqc_before_ref\tqc_after_ref\t"
            "scg_gate_used\tscg_gate_result\tscg_gate_reason\n"
        )
        for rec in actions:
            fh.write(
                f"{rec.action_id}\t{rec.action_type}\t{rec.target_bin}\t{rec.source_bin}\t"
                f"{json.dumps(list(rec.affected_contigs), ensure_ascii=True)}\t"
                f"{1 if rec.accepted else 0}\t{rec.accept_reason}\t{rec.reject_reason}\t"
                f"{rec.qc_before_ref}\t{rec.qc_after_ref}\t"
                f"{1 if rec.scg_gate_used else 0}\t{rec.scg_gate_result}\t{rec.scg_gate_reason}\n"
            )


def _make_split_residual_rows(
    *,
    candidate: SplitCandidate,
    coarse_raw: dict[str, str],
) -> tuple[ResidualRecord, ...]:
    rows: list[ResidualRecord] = []
    for contig_name in candidate.residual_contigs:
        coarse_bin_id = str(coarse_raw.get(contig_name, candidate.source_bin)).strip() or candidate.source_bin
        rows.append(
            ResidualRecord(
                contig_name=contig_name,
                reason="split_residual",
                stage="split",
                coarse_bin_id=coarse_bin_id,
                refined_status="residual_unresolved",
                note=(
                    f"source_bin={candidate.source_bin};"
                    f"candidate_reason={candidate.candidate_reason};"
                    f"evidence={candidate.evidence_summary}"
                ),
            )
        )
    return tuple(sorted(rows, key=lambda rec: rec.contig_name))


def _load_split_feature_matrix(
    *,
    contigs_fasta: Path,
    coverage_tsv: Optional[Path],
    contig_len: dict[str, int],
    cache_dir: Path,
    logger: logging.Logger,
) -> tuple[dict[str, int], "object"]:
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise RefineError("Local split reclustering requires numpy.") from exc

    try:
        from porebin.hypergraph_joint_spectral import (
            load_coverage_feature_optional,
            load_or_build_tnf136_features,
            zscore_features,
        )
    except Exception as exc:  # pragma: no cover
        raise RefineError(
            "Local split reclustering requires the coarse feature-layer helpers to be importable."
        ) from exc

    ordered_contigs = list(contig_len.keys())
    contig_name_to_idx = {name: idx for idx, name in enumerate(ordered_contigs)}
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    X_tnf = load_or_build_tnf136_features(
        graph_dir=cache_dir,
        contigs_fasta=contigs_fasta,
        contig_name_to_idx=contig_name_to_idx,
        logger=logger,
    )
    x_cov, _coverage_missing_count, _coverage_used = load_coverage_feature_optional(
        coverage_tsv,
        contig_name_to_idx,
    )
    X = np.concatenate([X_tnf.astype(np.float32, copy=False), x_cov.reshape(-1, 1)], axis=1)
    X = zscore_features(X)
    return contig_name_to_idx, X


def _build_local_recluster_split_candidate(
    *,
    state: RefineState,
    row: Any,
    split_check: Any,
    contig_name_to_idx: dict[str, int],
    feature_matrix: "object",
) -> Optional[SplitCandidate]:
    try:
        import numpy as np
        from sklearn.cluster import KMeans
    except Exception as exc:  # pragma: no cover
        raise RefineError(
            "Local split reclustering requires scikit-learn (KMeans) and numpy."
        ) from exc

    source_bin = str(row.bin_id)
    contigs = sorted(c for c, b in state.contig_to_bin.items() if str(b) == source_bin)
    if len(contigs) < 4:
        return None

    try:
        idx = np.asarray([int(contig_name_to_idx[c]) for c in contigs], dtype=int)
    except KeyError:
        return None

    X_bin = np.asarray(feature_matrix[idx], dtype=np.float32)
    if X_bin.ndim != 2 or X_bin.shape[0] != len(contigs):
        return None
    if np.allclose(X_bin, X_bin[0], atol=1e-8):
        return None

    km = KMeans(n_clusters=2, random_state=0, n_init=10)
    labels = km.fit_predict(X_bin)
    uniq = sorted(set(int(x) for x in labels.tolist()))
    if len(uniq) != 2:
        return None

    groups: list[tuple[str, ...]] = []
    for lab in uniq:
        group = tuple(sorted(contigs[i] for i, cur in enumerate(labels.tolist()) if int(cur) == lab))
        if group:
            groups.append(group)
    groups = sorted(groups, key=lambda g: (-len(g), tuple(g)))
    if len(groups) != 2:
        return None

    primary = groups[0]
    secondary = groups[1]
    if len(primary) < 2 or len(secondary) < 2:
        return None

    return SplitCandidate(
        source_bin=source_bin,
        primary_contigs=primary,
        secondary_contigs=secondary,
        residual_contigs=(),
        candidate_path="local_recluster",
        candidate_reason="scg_triggered_local_recluster_kmeans2",
        trigger_reasons=tuple(split_check.trigger_reasons + split_check.auxiliary_reasons),
        priority_source=str(split_check.priority_source),
        evidence_summary=(
            f"n_contigs={len(contigs)};"
            f"trigger_reasons={','.join(split_check.trigger_reasons)};"
            f"auxiliary_reasons={','.join(split_check.auxiliary_reasons)};"
            f"coverage_dispersion={getattr(row, 'coverage_dispersion', None)}"
        ),
    )


def _generate_split_candidates(
    *,
    state: RefineState,
    refined_snapshot: dict[str, Any],
    contigs_fasta: Path,
    coverage_tsv: Optional[Path],
    out_dir: Path,
    logger: logging.Logger,
) -> list[SplitCandidate]:
    rows_by_bin = _rows_by_bin_id(refined_snapshot.get("rows", []))
    split_checks = select_split_check_bins(refined_snapshot.get("rows", []))
    if not split_checks:
        return []

    contig_name_to_idx, feature_matrix = _load_split_feature_matrix(
        contigs_fasta=contigs_fasta,
        coverage_tsv=coverage_tsv,
        contig_len=state.contig_len,
        cache_dir=out_dir / "_split_feature_cache",
        logger=logger,
    )
    candidates: list[SplitCandidate] = []
    for split_check in split_checks:
        row = rows_by_bin.get(str(split_check.bin_id))
        if row is None:
            continue
        candidate = _build_local_recluster_split_candidate(
            state=state,
            row=row,
            split_check=split_check,
            contig_name_to_idx=contig_name_to_idx,
            feature_matrix=feature_matrix,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _evaluate_live_split_scg_gate(
    *,
    before_row: Optional[Any],
    primary_row: Optional[Any],
    secondary_row: Optional[Any],
    scg_state: str,
) -> tuple[bool, str, str]:
    if scg_state != "enabled":
        return False, "not_available", "scg_unavailable_for_live_split"
    if before_row is None or primary_row is None or secondary_row is None:
        return True, "neutral", "split_row_missing"
    if before_row.unique_scg is None or primary_row.unique_scg is None or secondary_row.unique_scg is None:
        return False, "not_available", "scg_metrics_unavailable"

    before_dup = int(before_row.duplicated_scg or 0)
    after_dup = int(primary_row.duplicated_scg or 0) + int(secondary_row.duplicated_scg or 0)
    before_contam = before_row.contamination_like
    after_contams = [row.contamination_like for row in (primary_row, secondary_row) if row.contamination_like is not None]
    after_max_contam = max(after_contams) if after_contams else None
    before_comp = before_row.completeness_like
    after_comps = [row.completeness_like for row in (primary_row, secondary_row) if row.completeness_like is not None]
    after_best_comp = max(after_comps) if after_comps else None

    positive: list[str] = []
    negative: list[str] = []

    if after_dup < before_dup:
        positive.append("duplicated_scg_decreased")
    elif after_dup > before_dup:
        negative.append("duplicated_scg_increased")

    if before_contam is not None and after_max_contam is not None:
        if after_max_contam < before_contam:
            positive.append("contamination_like_decreased")
        elif after_max_contam > before_contam and after_dup >= before_dup:
            negative.append("contamination_like_increased_without_dup_gain")

    if (
        before_comp is not None
        and after_best_comp is not None
        and after_best_comp < before_comp
        and after_dup >= before_dup
    ):
        negative.append("completeness_like_decreased_without_dup_gain")

    if negative:
        return True, "veto", ",".join(negative)
    if positive:
        return True, "support", ",".join(positive)
    return True, "neutral", "no_scg_split_change"


def _evaluate_split_candidate(
    *,
    candidate: SplitCandidate,
    current_assignment: dict[str, str],
    current_state: RefineState,
    current_snapshot: dict[str, Any],
    coarse_raw: dict[str, str],
    contacts_parquet: Path,
    scg_hits_tsv: Optional[Path],
    scg_expected_markers: Optional[set[str]],
    scg_state: str,
    new_bin_id: str,
) -> SplitDecision:
    before_row = _rows_by_bin_id(current_snapshot.get("rows", [])).get(candidate.source_bin)
    affected = set(candidate.primary_contigs) | set(candidate.secondary_contigs) | set(candidate.residual_contigs)
    if before_row is None:
        return SplitDecision(
            candidate=candidate,
            accepted=False,
            new_bin_id=new_bin_id,
            accept_reason="",
            reject_reason="source_bin_qc_missing",
            qc_before_ref="refined_pre_split",
            qc_after_ref="",
            scg_gate_used=False,
            scg_gate_result="not_applicable",
            scg_gate_reason="source_bin_qc_missing",
            residual_rows=(),
        )

    if len(candidate.primary_contigs) < 2 or len(candidate.secondary_contigs) < 2:
        return SplitDecision(
            candidate=candidate,
            accepted=False,
            new_bin_id=new_bin_id,
            accept_reason="",
            reject_reason="child_group_too_small",
            qc_before_ref="refined_pre_split",
            qc_after_ref="",
            scg_gate_used=False,
            scg_gate_result="not_applicable",
            scg_gate_reason="child_group_too_small",
            residual_rows=(),
        )

    tentative_assignment = dict(current_assignment)
    for contig_name in affected:
        tentative_assignment.pop(contig_name, None)
    for contig_name in candidate.primary_contigs:
        tentative_assignment[contig_name] = candidate.source_bin
    for contig_name in candidate.secondary_contigs:
        tentative_assignment[contig_name] = new_bin_id

    tentative_state = _build_refine_state_from_assignment(
        contig_len=current_state.contig_len,
        assignment=tentative_assignment,
        coverage=current_state.coverage,
    )
    tentative_snapshot = _collect_refine_qc_snapshot(
        state=tentative_state,
        contacts_parquet=contacts_parquet,
        scg_hits_tsv=scg_hits_tsv,
        scg_expected_markers=scg_expected_markers,
        scg_status=scg_state,
    )
    tentative_rows = _rows_by_bin_id(tentative_snapshot.get("rows", []))
    primary_row = tentative_rows.get(candidate.source_bin)
    secondary_row = tentative_rows.get(new_bin_id)

    if primary_row is None or secondary_row is None:
        return SplitDecision(
            candidate=candidate,
            accepted=False,
            new_bin_id=new_bin_id,
            accept_reason="",
            reject_reason="child_bin_qc_missing",
            qc_before_ref="refined_pre_split",
            qc_after_ref="",
            scg_gate_used=False,
            scg_gate_result="not_applicable",
            scg_gate_reason="child_bin_qc_missing",
            residual_rows=(),
        )

    if candidate.residual_contigs and len(candidate.residual_contigs) >= len(candidate.primary_contigs) + len(candidate.secondary_contigs):
        return SplitDecision(
            candidate=candidate,
            accepted=False,
            new_bin_id=new_bin_id,
            accept_reason="",
            reject_reason="residual_dominates_split",
            qc_before_ref="refined_pre_split",
            qc_after_ref="",
            scg_gate_used=False,
            scg_gate_result="not_applicable",
            scg_gate_reason="residual_dominates_split",
            residual_rows=(),
        )

    before_consistency = float(before_row.contact_consistency or 0.0)
    child_consistencies = [float(primary_row.contact_consistency or 0.0), float(secondary_row.contact_consistency or 0.0)]
    if min(child_consistencies) + 1e-9 < before_consistency:
        return SplitDecision(
            candidate=candidate,
            accepted=False,
            new_bin_id=new_bin_id,
            accept_reason="",
            reject_reason="contact_consistency_worsened",
            qc_before_ref="refined_pre_split",
            qc_after_ref="",
            scg_gate_used=False,
            scg_gate_result="not_applicable",
            scg_gate_reason="contact_consistency_worsened",
            residual_rows=(),
        )
    if max(child_consistencies) <= before_consistency + 1e-9 and candidate.candidate_path != "local_recluster":
        return SplitDecision(
            candidate=candidate,
            accepted=False,
            new_bin_id=new_bin_id,
            accept_reason="",
            reject_reason="contact_consistency_not_improved",
            qc_before_ref="refined_pre_split",
            qc_after_ref="",
            scg_gate_used=False,
            scg_gate_result="not_applicable",
            scg_gate_reason="contact_consistency_not_improved",
            residual_rows=(),
        )

    scg_gate_used, scg_gate_result, scg_gate_reason = _evaluate_live_split_scg_gate(
        before_row=before_row,
        primary_row=primary_row,
        secondary_row=secondary_row,
        scg_state=scg_state,
    )
    if scg_gate_result == "veto":
        return SplitDecision(
            candidate=candidate,
            accepted=False,
            new_bin_id=new_bin_id,
            accept_reason="",
            reject_reason="scg_gate_veto",
            qc_before_ref="refined_pre_split",
            qc_after_ref="",
            scg_gate_used=scg_gate_used,
            scg_gate_result=scg_gate_result,
            scg_gate_reason=scg_gate_reason,
            residual_rows=(),
        )
    if str(candidate.priority_source) == "scg_priority" and scg_gate_result != "support":
        return SplitDecision(
            candidate=candidate,
            accepted=False,
            new_bin_id=new_bin_id,
            accept_reason="",
            reject_reason="scg_priority_without_improvement",
            qc_before_ref="refined_pre_split",
            qc_after_ref="",
            scg_gate_used=scg_gate_used,
            scg_gate_result=scg_gate_result,
            scg_gate_reason=scg_gate_reason,
            residual_rows=(),
        )

    residual_rows = _make_split_residual_rows(candidate=candidate, coarse_raw=coarse_raw)
    accept_reasons = ["contact_consistency_validated"]
    if max(child_consistencies) > before_consistency + 1e-9:
        accept_reasons.append("contact_consistency_improved")
    if candidate.candidate_path == "local_recluster":
        accept_reasons.append("local_recluster_partition")
    if scg_gate_result == "support":
        accept_reasons.append("scg_support")
    if candidate.residual_contigs:
        accept_reasons.append("residualized_edge_components")
    return SplitDecision(
        candidate=candidate,
        accepted=True,
        new_bin_id=new_bin_id,
        accept_reason=",".join(accept_reasons),
        reject_reason="",
        qc_before_ref="refined_pre_split",
        qc_after_ref="refined",
        scg_gate_used=scg_gate_used,
        scg_gate_result=scg_gate_result,
        scg_gate_reason=scg_gate_reason,
        residual_rows=residual_rows,
    )


def _apply_split_decision(
    *,
    decision: SplitDecision,
    final_assignment: dict[str, str],
    residual_rows: list[ResidualRecord],
) -> None:
    if not decision.accepted:
        return
    candidate = decision.candidate
    affected = set(candidate.primary_contigs) | set(candidate.secondary_contigs) | set(candidate.residual_contigs)
    for contig_name in affected:
        final_assignment.pop(contig_name, None)
    for contig_name in candidate.primary_contigs:
        final_assignment[contig_name] = candidate.source_bin
    for contig_name in candidate.secondary_contigs:
        final_assignment[contig_name] = decision.new_bin_id
    residual_rows.extend(decision.residual_rows)


def _build_reassign_windows_from_split_decisions(
    *,
    split_decisions: list[SplitDecision],
    final_assignment: dict[str, str],
) -> list[ReassignWindow]:
    windows: list[ReassignWindow] = []
    window_idx = 1
    for decision in split_decisions:
        if not decision.accepted:
            continue
        left_bin = str(decision.candidate.source_bin)
        right_bin = str(decision.new_bin_id)
        window_contigs = tuple(
            sorted(
                contig_name
                for contig_name in final_assignment.keys()
                if str(final_assignment.get(contig_name, "")) in {left_bin, right_bin}
            )
        )
        if not window_contigs:
            continue
        windows.append(
            ReassignWindow(
                window_id=f"reassign_window_{window_idx:04d}",
                left_bin=left_bin,
                right_bin=right_bin,
                window_contigs=window_contigs,
            )
        )
        window_idx += 1
    return windows


def _scan_reassign_pair_support_parquet(
    *,
    contacts_parquet: Path,
    contig_len: dict[str, int],
    min_contig_len: int,
    current_assignment: dict[str, str],
    windows: list[ReassignWindow],
) -> dict[str, dict[str, tuple[float, float]]]:
    if not windows:
        return {}

    windows_by_id = {window.window_id: window for window in windows}
    contig_to_window: dict[str, str] = {}
    support_by_window: dict[str, dict[str, list[float]]] = {}
    for window in windows:
        support_by_window[window.window_id] = {contig_name: [0.0, 0.0] for contig_name in window.window_contigs}
        for contig_name in window.window_contigs:
            contig_to_window[contig_name] = window.window_id

    batch_size = 100_000
    for contigs_raw, p_raw, row_weight in _iter_contacts_parquet(contacts_parquet, batch_size=batch_size):
        seen: dict[str, int] = {}
        contigs: list[str] = []
        p: list[float] = []
        for contig_name, weight in zip(contigs_raw, p_raw, strict=True):
            if not contig_name:
                continue
            if contig_len.get(contig_name, 0) < min_contig_len:
                continue
            ww = float(weight)
            if ww <= 0.0:
                continue
            if contig_name in seen:
                p[seen[contig_name]] += ww
            else:
                seen[contig_name] = len(contigs)
                contigs.append(contig_name)
                p.append(ww)

        if len(contigs) < 2:
            continue
        total = float(sum(p))
        if not (total > 0.0):
            continue
        p = [float(x) / total for x in p]
        w_contact = order_norm(len(contigs), method="pair") * float(row_weight)

        grouped_members: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
        for contig_name, pc in zip(contigs, p, strict=True):
            window_id = contig_to_window.get(contig_name)
            if window_id is None:
                continue
            current_bin = str(current_assignment.get(contig_name, "")).strip()
            if not current_bin:
                continue
            grouped_members[window_id].append((contig_name, float(pc), current_bin))

        for window_id, members in grouped_members.items():
            if len(members) < 2:
                continue
            window = windows_by_id[window_id]
            left_mass = sum(pc for _c, pc, bin_id in members if bin_id == window.left_bin)
            right_mass = sum(pc for _c, pc, bin_id in members if bin_id == window.right_bin)
            if left_mass <= 0.0 and right_mass <= 0.0:
                continue
            for contig_name, pc, bin_id in members:
                left_eff = max(0.0, left_mass - float(pc)) if bin_id == window.left_bin else float(left_mass)
                right_eff = max(0.0, right_mass - float(pc)) if bin_id == window.right_bin else float(right_mass)
                pair = support_by_window[window_id].get(contig_name)
                if pair is None:
                    continue
                pair[0] += float(w_contact) * float(pc) * float(left_eff)
                pair[1] += float(w_contact) * float(pc) * float(right_eff)

    out: dict[str, dict[str, tuple[float, float]]] = {}
    for window_id, by_contig in support_by_window.items():
        out[window_id] = {
            contig_name: (float(vals[0]), float(vals[1]))
            for contig_name, vals in by_contig.items()
        }
    return out


def _generate_reassign_candidates(
    *,
    windows: list[ReassignWindow],
    pair_support: dict[str, dict[str, tuple[float, float]]],
    current_assignment: dict[str, str],
    support_margin_thresh: float = 0.2,
    eps: float = 1e-12,
) -> list[ReassignCandidate]:
    candidates: list[ReassignCandidate] = []
    for window in windows:
        support_map = pair_support.get(window.window_id, {})
        for contig_name in sorted(window.window_contigs):
            current_bin = str(current_assignment.get(contig_name, "")).strip()
            if current_bin not in {window.left_bin, window.right_bin}:
                continue
            sibling_bin = window.right_bin if current_bin == window.left_bin else window.left_bin
            left_support, right_support = support_map.get(contig_name, (0.0, 0.0))
            current_support = float(left_support if current_bin == window.left_bin else right_support)
            sibling_support = float(right_support if current_bin == window.left_bin else left_support)
            total_support = float(current_support + sibling_support)
            support_margin = abs(float(current_support) - float(sibling_support)) / (total_support + eps)

            if current_support > sibling_support:
                continue

            candidate_action = (
                "move_to_sibling"
                if sibling_support > current_support and support_margin >= support_margin_thresh
                else "residualize"
            )
            candidates.append(
                ReassignCandidate(
                    window_id=window.window_id,
                    contig_name=contig_name,
                    current_bin=current_bin,
                    sibling_bin=sibling_bin,
                    current_support=float(current_support),
                    sibling_support=float(sibling_support),
                    support_margin=float(support_margin),
                    candidate_action=candidate_action,
                    evidence_summary=(
                        f"window={window.window_id};"
                        f"current_support={current_support:.6g};"
                        f"sibling_support={sibling_support:.6g};"
                        f"support_margin={support_margin:.6g}"
                    ),
                )
            )
    return sorted(
        candidates,
        key=lambda cand: (
            0 if cand.candidate_action == "move_to_sibling" else 1,
            -float(cand.support_margin),
            str(cand.contig_name),
        ),
    )


def _make_reassign_residual_row(
    *,
    candidate: ReassignCandidate,
    coarse_raw: dict[str, str],
) -> ResidualRecord:
    coarse_bin_id = str(coarse_raw.get(candidate.contig_name, candidate.current_bin)).strip() or candidate.current_bin
    return ResidualRecord(
        contig_name=candidate.contig_name,
        reason="reassign_unresolved",
        stage="reassign",
        coarse_bin_id=coarse_bin_id,
        refined_status="residual_unresolved",
        note=(
            f"current_bin={candidate.current_bin};"
            f"sibling_bin={candidate.sibling_bin};"
            f"candidate_action={candidate.candidate_action};"
            f"evidence={candidate.evidence_summary}"
        ),
    )


def _evaluate_reassign_scg_gate(
    *,
    before_source_row: Optional[Any],
    before_target_row: Optional[Any],
    after_source_row: Optional[Any],
    after_target_row: Optional[Any],
    scg_state: str,
) -> tuple[bool, str, str]:
    if scg_state != "enabled":
        return False, "not_available", "scg_unavailable_for_reassign"
    if before_source_row is None or after_source_row is None:
        return False, "not_available", "source_bin_qc_missing"

    negative: list[str] = []
    positive: list[str] = []

    def _compare_metric(
        *,
        before_row: Optional[Any],
        after_row: Optional[Any],
        attr: str,
        increase_reason: str,
        decrease_reason: str,
    ) -> None:
        if before_row is None or after_row is None:
            return
        before_value = getattr(before_row, attr, None)
        after_value = getattr(after_row, attr, None)
        if before_value is None or after_value is None:
            return
        if float(after_value) > float(before_value) + 1e-9:
            negative.append(increase_reason)
        elif float(after_value) < float(before_value) - 1e-9:
            positive.append(decrease_reason)

    _compare_metric(
        before_row=before_source_row,
        after_row=after_source_row,
        attr="duplicated_scg",
        increase_reason="source_duplicated_scg_increased",
        decrease_reason="source_duplicated_scg_decreased",
    )
    _compare_metric(
        before_row=before_source_row,
        after_row=after_source_row,
        attr="contamination_like",
        increase_reason="source_contamination_like_increased",
        decrease_reason="source_contamination_like_decreased",
    )
    if (
        getattr(before_source_row, "completeness_like", None) is not None
        and getattr(after_source_row, "completeness_like", None) is not None
        and float(after_source_row.completeness_like) < float(before_source_row.completeness_like) - 1e-9
    ):
        negative.append("source_completeness_like_decreased")

    _compare_metric(
        before_row=before_target_row,
        after_row=after_target_row,
        attr="duplicated_scg",
        increase_reason="target_duplicated_scg_increased",
        decrease_reason="target_duplicated_scg_decreased",
    )
    _compare_metric(
        before_row=before_target_row,
        after_row=after_target_row,
        attr="contamination_like",
        increase_reason="target_contamination_like_increased",
        decrease_reason="target_contamination_like_decreased",
    )

    if negative:
        return True, "veto", ",".join(negative)
    if positive:
        return True, "support", ",".join(positive)
    return True, "neutral", "no_scg_change"


def _evaluate_reassign_action_variant(
    *,
    candidate: ReassignCandidate,
    action: str,
    current_assignment: dict[str, str],
    current_state: RefineState,
    current_snapshot: dict[str, Any],
    coarse_raw: dict[str, str],
    contacts_parquet: Path,
    scg_hits_tsv: Optional[Path],
    scg_expected_markers: Optional[set[str]],
    scg_state: str,
) -> tuple[bool, str, str, str, str, bool, str, str, Optional[ResidualRecord]]:
    before_rows = _rows_by_bin_id(current_snapshot.get("rows", []))
    before_source_row = before_rows.get(candidate.current_bin)
    before_target_row = before_rows.get(candidate.sibling_bin)
    if before_source_row is None:
        return (
            False,
            action,
            "",
            "",
            "source_bin_qc_missing",
            False,
            "not_applicable",
            "source_bin_qc_missing",
            None,
        )

    tentative_assignment = dict(current_assignment)
    residual_row: Optional[ResidualRecord] = None
    target_bin = candidate.current_bin
    if action == "move_to_sibling":
        tentative_assignment[candidate.contig_name] = candidate.sibling_bin
        target_bin = candidate.sibling_bin
    elif action == "residualize":
        tentative_assignment.pop(candidate.contig_name, None)
        target_bin = "-1"
        residual_row = _make_reassign_residual_row(candidate=candidate, coarse_raw=coarse_raw)
    else:
        raise RefineError(f"Unsupported reassign action: {action}")

    tentative_state = _build_refine_state_from_assignment(
        contig_len=current_state.contig_len,
        assignment=tentative_assignment,
        coverage=current_state.coverage,
    )
    tentative_snapshot = _collect_refine_qc_snapshot(
        state=tentative_state,
        contacts_parquet=contacts_parquet,
        scg_hits_tsv=scg_hits_tsv,
        scg_expected_markers=scg_expected_markers,
        scg_status=scg_state,
    )
    tentative_rows = _rows_by_bin_id(tentative_snapshot.get("rows", []))
    after_source_row = tentative_rows.get(candidate.current_bin)
    after_target_row = tentative_rows.get(candidate.sibling_bin)

    if after_source_row is None or int(after_source_row.n_contigs) < 2:
        return (
            False,
            action,
            "",
            "",
            "source_bin_too_small_after_reassign",
            False,
            "not_applicable",
            "source_bin_too_small_after_reassign",
            None,
        )
    if action == "move_to_sibling":
        if after_target_row is None or int(after_target_row.n_contigs) < 2:
            return (
                False,
                action,
                "",
                "",
                "target_bin_too_small_after_reassign",
                False,
                "not_applicable",
                "target_bin_too_small_after_reassign",
                None,
            )

    before_source_consistency = float(before_source_row.contact_consistency or 0.0)
    after_source_consistency = float(after_source_row.contact_consistency or 0.0)
    if after_source_consistency + 1e-9 < before_source_consistency:
        return (
            False,
            action,
            "",
            "",
            "source_contact_consistency_worsened",
            False,
            "not_applicable",
            "source_contact_consistency_worsened",
            None,
        )

    if action == "move_to_sibling":
        before_target_consistency = float(before_target_row.contact_consistency or 0.0) if before_target_row is not None else 0.0
        after_target_consistency = float(after_target_row.contact_consistency or 0.0) if after_target_row is not None else 0.0
        if after_target_consistency + 1e-9 < before_target_consistency:
            return (
                False,
                action,
                "",
                "",
                "target_contact_consistency_worsened",
                False,
                "not_applicable",
                "target_contact_consistency_worsened",
                None,
            )

    scg_gate_used, scg_gate_result, scg_gate_reason = _evaluate_reassign_scg_gate(
        before_source_row=before_source_row,
        before_target_row=before_target_row if action == "move_to_sibling" else None,
        after_source_row=after_source_row,
        after_target_row=after_target_row if action == "move_to_sibling" else None,
        scg_state=scg_state,
    )
    if scg_gate_result == "veto":
        return (
            False,
            action,
            "",
            "",
            "scg_gate_veto",
            scg_gate_used,
            scg_gate_result,
            scg_gate_reason,
            None,
        )

    accept_reasons = ["contact_support_reassigned" if action == "move_to_sibling" else "reassign_residualized"]
    if action == "move_to_sibling":
        accept_reasons.append("sibling_support_dominates")
    else:
        accept_reasons.append("support_ambiguous")
    if scg_gate_result == "support":
        accept_reasons.append("scg_support")

    return (
        True,
        action,
        target_bin,
        ",".join(accept_reasons),
        "",
        scg_gate_used,
        scg_gate_result,
        scg_gate_reason,
        residual_row,
    )


def _evaluate_reassign_candidate(
    *,
    candidate: ReassignCandidate,
    current_assignment: dict[str, str],
    current_state: RefineState,
    current_snapshot: dict[str, Any],
    coarse_raw: dict[str, str],
    contacts_parquet: Path,
    scg_hits_tsv: Optional[Path],
    scg_expected_markers: Optional[set[str]],
    scg_state: str,
) -> ReassignDecision:
    attempted_actions = [candidate.candidate_action]
    if candidate.candidate_action == "move_to_sibling":
        attempted_actions.append("residualize")

    reject_reasons: list[str] = []
    last_scg_gate_used = False
    last_scg_gate_result = "not_applicable"
    last_scg_gate_reason = "candidate_not_applied"
    for action in attempted_actions:
        (
            accepted,
            final_action,
            target_bin,
            accept_reason,
            reject_reason,
            scg_gate_used,
            scg_gate_result,
            scg_gate_reason,
            residual_row,
        ) = _evaluate_reassign_action_variant(
            candidate=candidate,
            action=action,
            current_assignment=current_assignment,
            current_state=current_state,
            current_snapshot=current_snapshot,
            coarse_raw=coarse_raw,
            contacts_parquet=contacts_parquet,
            scg_hits_tsv=scg_hits_tsv,
            scg_expected_markers=scg_expected_markers,
            scg_state=scg_state,
        )
        last_scg_gate_used = scg_gate_used
        last_scg_gate_result = scg_gate_result
        last_scg_gate_reason = scg_gate_reason
        if accepted:
            return ReassignDecision(
                candidate=candidate,
                accepted=True,
                final_action=final_action,
                target_bin=target_bin,
                accept_reason=accept_reason,
                reject_reason="",
                qc_before_ref="refined_post_split",
                qc_after_ref="refined",
                scg_gate_used=scg_gate_used,
                scg_gate_result=scg_gate_result,
                scg_gate_reason=scg_gate_reason,
                residual_row=residual_row,
            )
        reject_reasons.append(str(reject_reason or final_action))

    return ReassignDecision(
        candidate=candidate,
        accepted=False,
        final_action="keep",
        target_bin=candidate.current_bin,
        accept_reason="",
        reject_reason=";".join(reject_reasons) if reject_reasons else "reassign_candidate_rejected",
        qc_before_ref="refined_post_split",
        qc_after_ref="",
        scg_gate_used=last_scg_gate_used,
        scg_gate_result=last_scg_gate_result,
        scg_gate_reason=last_scg_gate_reason,
        residual_row=None,
    )


def _apply_reassign_decision(
    *,
    decision: ReassignDecision,
    final_assignment: dict[str, str],
    residual_rows: list[ResidualRecord],
) -> None:
    if not decision.accepted:
        return
    contig_name = decision.candidate.contig_name
    if decision.final_action == "move_to_sibling":
        final_assignment[contig_name] = decision.target_bin
        return
    if decision.final_action == "residualize":
        final_assignment.pop(contig_name, None)
        if decision.residual_row is not None:
            residual_rows.append(decision.residual_row)


def _generate_recruit_candidates(
    *,
    residual_rows: list[ResidualRecord],
    affinity: dict[str, dict[str, float]],
    keep_bins: set[str],
) -> list[RecruitCandidate]:
    candidates: list[RecruitCandidate] = []
    eps = 1e-12

    for row in residual_rows:
        if str(row.refined_status) != "residual_unresolved":
            continue
        contig_name = str(row.contig_name)
        scores = affinity.get(contig_name, {})
        top = _top_candidates(scores, keep_bins=keep_bins, topk=2)
        top1_bin = top[0][0] if top else ""
        top1_support = float(top[0][1]) if top else 0.0
        top2_bin = top[1][0] if len(top) > 1 else ""
        top2_support = float(top[1][1]) if len(top) > 1 else 0.0
        margin = (
            (top1_support - top2_support) / (top1_support + top2_support + eps)
            if top1_support > 0.0
            else 0.0
        )
        candidates.append(
            RecruitCandidate(
                contig_name=contig_name,
                current_stage=str(row.stage),
                current_reason=str(row.reason),
                target_bin=top1_bin,
                target_support=top1_support,
                runner_up_bin=top2_bin,
                runner_up_support=top2_support,
                support_margin=float(margin),
                evidence_summary=(
                    f"current_stage={row.stage};current_reason={row.reason};"
                    f"top1_bin={top1_bin or '-1'};top1_support={top1_support:.6g};"
                    f"top2_bin={top2_bin or '-1'};top2_support={top2_support:.6g};"
                    f"support_margin={margin:.6g}"
                ),
            )
        )

    candidates.sort(
        key=lambda cand: (
            float(cand.target_support),
            float(cand.support_margin),
            str(cand.contig_name),
        ),
        reverse=True,
    )
    return candidates


def _make_recruit_keep_residual_row(
    *,
    candidate: RecruitCandidate,
    coarse_raw: dict[str, str],
    previous_note: str,
) -> ResidualRecord:
    coarse_bin_id = str(coarse_raw.get(candidate.contig_name, "-1")).strip() or "-1"
    note = (
        f"prev_stage={candidate.current_stage};"
        f"prev_reason={candidate.current_reason};"
        f"{candidate.evidence_summary}"
    )
    if previous_note:
        note += f";previous_note={previous_note}"
    return ResidualRecord(
        contig_name=candidate.contig_name,
        reason="recruit_unresolved",
        stage="recruit",
        coarse_bin_id=coarse_bin_id,
        refined_status="residual_unresolved",
        note=note,
    )


def _evaluate_recruit_scg_gate(
    *,
    before_target_row: Optional[Any],
    after_target_row: Optional[Any],
    scg_state: str,
) -> tuple[bool, str, str]:
    if scg_state != "enabled":
        return False, "not_available", "scg_unavailable_for_recruit"
    if before_target_row is None or after_target_row is None:
        return False, "not_available", "target_bin_qc_missing"

    negative: list[str] = []
    positive: list[str] = []

    def _compare_metric(attr: str, increase_reason: str, decrease_reason: str) -> None:
        before_value = getattr(before_target_row, attr, None)
        after_value = getattr(after_target_row, attr, None)
        if before_value is None or after_value is None:
            return
        if float(after_value) > float(before_value) + 1e-9:
            negative.append(increase_reason)
        elif float(after_value) < float(before_value) - 1e-9:
            positive.append(decrease_reason)

    _compare_metric("duplicated_scg", "target_duplicated_scg_increased", "target_duplicated_scg_decreased")
    _compare_metric(
        "contamination_like",
        "target_contamination_like_increased",
        "target_contamination_like_decreased",
    )

    if negative:
        return True, "veto", ",".join(negative)
    if positive:
        return True, "support", ",".join(positive)
    return True, "neutral", "no_scg_change"


def _evaluate_recruit_candidate(
    *,
    candidate: RecruitCandidate,
    current_assignment: dict[str, str],
    current_state: RefineState,
    current_snapshot: dict[str, Any],
    current_residual_by_contig: dict[str, ResidualRecord],
    coarse_raw: dict[str, str],
    contacts_parquet: Path,
    scg_hits_tsv: Optional[Path],
    scg_expected_markers: Optional[set[str]],
    scg_state: str,
    support_margin_thresh: float,
) -> RecruitDecision:
    before_rows = _rows_by_bin_id(current_snapshot.get("rows", []))
    before_target_row = before_rows.get(candidate.target_bin) if candidate.target_bin else None
    current_residual = current_residual_by_contig.get(candidate.contig_name)
    previous_note = current_residual.note if current_residual is not None else ""

    keep_row = _make_recruit_keep_residual_row(
        candidate=candidate,
        coarse_raw=coarse_raw,
        previous_note=previous_note,
    )

    if not candidate.target_bin or candidate.target_support <= 0.0:
        return RecruitDecision(
            candidate=candidate,
            accepted=True,
            final_action="keep_residual",
            target_bin="-1",
            accept_reason="no_positive_recruit_support",
            reject_reason="",
            qc_before_ref="reassign",
            qc_after_ref="refined",
            scg_gate_used=False,
            scg_gate_result="not_applicable",
            scg_gate_reason="no_positive_recruit_support",
            residual_row=keep_row,
        )

    if float(candidate.support_margin) + 1e-9 < float(support_margin_thresh):
        return RecruitDecision(
            candidate=candidate,
            accepted=True,
            final_action="keep_residual",
            target_bin="-1",
            accept_reason="recruit_support_not_dominant_enough",
            reject_reason="",
            qc_before_ref="reassign",
            qc_after_ref="refined",
            scg_gate_used=False,
            scg_gate_result="not_applicable",
            scg_gate_reason="recruit_support_not_dominant_enough",
            residual_row=keep_row,
        )

    tentative_assignment = dict(current_assignment)
    tentative_assignment[candidate.contig_name] = candidate.target_bin
    tentative_state = _build_refine_state_from_assignment(
        contig_len=current_state.contig_len,
        assignment=tentative_assignment,
        coverage=current_state.coverage,
    )
    tentative_snapshot = _collect_refine_qc_snapshot(
        state=tentative_state,
        contacts_parquet=contacts_parquet,
        scg_hits_tsv=scg_hits_tsv,
        scg_expected_markers=scg_expected_markers,
        scg_status=scg_state,
    )
    tentative_rows = _rows_by_bin_id(tentative_snapshot.get("rows", []))
    after_target_row = tentative_rows.get(candidate.target_bin)

    if after_target_row is None or int(after_target_row.n_contigs) < 2:
        keep_row = replace(
            keep_row,
            note=f"{keep_row.note};attempted_target={candidate.target_bin};recruit_reject=target_bin_too_small_after_recruit",
        )
        return RecruitDecision(
            candidate=candidate,
            accepted=True,
            final_action="keep_residual",
            target_bin="-1",
            accept_reason="kept_residual_after_failed_recruit",
            reject_reason="target_bin_too_small_after_recruit",
            qc_before_ref="reassign",
            qc_after_ref="refined",
            scg_gate_used=False,
            scg_gate_result="not_applicable",
            scg_gate_reason="target_bin_too_small_after_recruit",
            residual_row=keep_row,
        )

    before_target_consistency = float(before_target_row.contact_consistency or 0.0) if before_target_row is not None else 0.0
    after_target_consistency = float(after_target_row.contact_consistency or 0.0) if after_target_row is not None else 0.0
    if after_target_consistency + 1e-9 < before_target_consistency:
        keep_row = replace(
            keep_row,
            note=f"{keep_row.note};attempted_target={candidate.target_bin};recruit_reject=target_contact_consistency_worsened",
        )
        return RecruitDecision(
            candidate=candidate,
            accepted=True,
            final_action="keep_residual",
            target_bin="-1",
            accept_reason="kept_residual_after_failed_recruit",
            reject_reason="target_contact_consistency_worsened",
            qc_before_ref="reassign",
            qc_after_ref="refined",
            scg_gate_used=False,
            scg_gate_result="not_applicable",
            scg_gate_reason="target_contact_consistency_worsened",
            residual_row=keep_row,
        )

    scg_gate_used, scg_gate_result, scg_gate_reason = _evaluate_recruit_scg_gate(
        before_target_row=before_target_row,
        after_target_row=after_target_row,
        scg_state=scg_state,
    )
    if scg_gate_result == "veto":
        keep_row = replace(
            keep_row,
            note=f"{keep_row.note};attempted_target={candidate.target_bin};recruit_reject=scg_gate_veto",
        )
        return RecruitDecision(
            candidate=candidate,
            accepted=True,
            final_action="keep_residual",
            target_bin="-1",
            accept_reason="kept_residual_after_failed_recruit",
            reject_reason="scg_gate_veto",
            qc_before_ref="reassign",
            qc_after_ref="refined",
            scg_gate_used=scg_gate_used,
            scg_gate_result=scg_gate_result,
            scg_gate_reason=scg_gate_reason,
            residual_row=keep_row,
        )

    accept_reasons = ["recruit_assigned_to_existing_bin", "target_support_dominates"]
    if scg_gate_result == "support":
        accept_reasons.append("scg_support")
    return RecruitDecision(
        candidate=candidate,
        accepted=True,
        final_action="assign_to_bin",
        target_bin=candidate.target_bin,
        accept_reason=",".join(accept_reasons),
        reject_reason="",
        qc_before_ref="reassign",
        qc_after_ref="refined",
        scg_gate_used=scg_gate_used,
        scg_gate_result=scg_gate_result,
        scg_gate_reason=scg_gate_reason,
        residual_row=None,
    )


def _apply_recruit_decision(
    *,
    decision: RecruitDecision,
    final_assignment: dict[str, str],
    residual_rows: list[ResidualRecord],
) -> None:
    if not decision.accepted:
        return
    contig_name = decision.candidate.contig_name
    residual_rows[:] = [row for row in residual_rows if str(row.contig_name) != contig_name]
    if decision.final_action == "assign_to_bin":
        final_assignment[contig_name] = decision.target_bin
        return
    if decision.final_action == "keep_residual":
        if decision.residual_row is not None:
            residual_rows.append(decision.residual_row)
        return
    raise RefineError(f"Unsupported recruit action: {decision.final_action}")


def pre_refine_cleanup(
    *,
    contacts_parquet: Path,
    bins_tsv: Path,
    contigs_fasta: Path,
    coverage_tsv: Optional[Path],
    out_dir: Path,
) -> PreCleanupResult:
    """
    Minimal pre-refine cleanup for parquet path (v1):
      - build contig_to_bin/bin_to_contigs from bins.tsv (exclude -1)
      - compute contig_len + min_contig_len
      - scan contacts.parquet support/affinity
      - decontam to remove low-intra-support contigs
      - derive anchor_mask + bin_status (strong/weak/impure)
    """
    contig_len = _read_contig_lengths(contigs_fasta)
    min_contig_len, _meta = auto_min_contig_len(contig_len)

    coarse_raw = _read_bins_tsv(bins_tsv)
    contig_to_bin: dict[str, str] = {}
    bin_to_contigs: dict[str, list[str]] = defaultdict(list)
    bin_counts: dict[str, int] = defaultdict(int)
    for c, b in coarse_raw.items():
        c = str(c).strip()
        b = str(b).strip()
        if not c or c not in contig_len:
            continue
        if not b or b == "-1":
            continue
        contig_to_bin[c] = b
        bin_to_contigs[b].append(c)
        bin_counts[b] += 1

    cov: Optional[dict[str, float]] = None
    bin_cov_stats: Optional[dict[str, dict[str, float]]] = None
    if coverage_tsv is not None and coverage_tsv.exists():
        cov = _coverage_from_tsv(coverage_tsv, contig_len=contig_len, min_contig_len=min_contig_len)
        bin_cov_stats = _bin_coverage_stats(bin_to_contigs, cov)

    intra, other, _affinity, _meta2 = _scan_contacts_support_and_affinity_parquet(
        contacts_parquet=contacts_parquet,
        contig_len=contig_len,
        min_contig_len=min_contig_len,
        contig_to_bin=contig_to_bin,
    )

    out_dir = out_dir.resolve()
    ensure_dir(out_dir)
    decontam_path = out_dir / "pre_refine_decontam.tsv"
    removed = _decontam(
        bin_to_contigs=bin_to_contigs,
        contig_len=contig_len,
        contig_to_bin=contig_to_bin,
        intra_support=intra,
        other_support=other,
        cov=cov,
        bin_cov_stats=bin_cov_stats,
        out_path=decontam_path,
    )
    removed_by_bin: dict[str, list[str]] = defaultdict(list)
    for contig in removed:
        source_bin = str(coarse_raw.get(contig, "-1")).strip() if contig in coarse_raw else "-1"
        if source_bin and source_bin != "-1":
            removed_by_bin[source_bin].append(contig)

    # Anchor rule (v1): kept in cleaned_assignment, length ok, intra >= bin median, intra > other
    anchor_mask: dict[str, bool] = {c: False for c in contig_len.keys()}
    anchor_weight: dict[str, float] = {c: 0.0 for c in contig_len.keys()}
    anchor_count: dict[str, int] = defaultdict(int)
    bin_intra_median: dict[str, float] = {}
    for bin_id, contigs in bin_to_contigs.items():
        vals = [float(intra.get(c, 0.0)) for c in contigs]
        if not vals:
            continue
        med = _median(vals)
        bin_intra_median[bin_id] = float(med)
        for c in contigs:
            if contig_len.get(c, 0) < min_contig_len:
                continue
            if float(intra.get(c, 0.0)) < float(med):
                continue
            if float(intra.get(c, 0.0)) <= float(other.get(c, 0.0)):
                continue
            anchor_mask[c] = True
            anchor_count[bin_id] += 1

    # Bin status (engineering gating)
    bin_status: dict[str, str] = {}
    bin_weight: dict[str, float] = {}
    bin_bp, _ = _bin_sizes_from_assignment(contig_to_bin, contig_len)
    for bin_id, contigs in bin_to_contigs.items():
        removed_in_bin = 0
        orig_n = int(bin_counts.get(bin_id, 0))
        if orig_n > 0:
            removed_in_bin = sum(1 for c in removed if coarse_raw.get(c) == bin_id)
        removed_ratio = (removed_in_bin / orig_n) if orig_n > 0 else 0.0
        if removed_ratio > 0.3:
            bin_status[bin_id] = "impure"
        elif int(anchor_count.get(bin_id, 0)) >= 2:
            bin_status[bin_id] = "strong"
        else:
            bin_status[bin_id] = "weak"
        bin_weight[bin_id] = _bin_weight_from_status(bin_status[bin_id])

    for bin_id, contigs in bin_to_contigs.items():
        bin_med = float(bin_intra_median.get(bin_id, 0.0))
        for c in contigs:
            anchor_weight[c] = _anchor_weight_from_support(
                intra_support=float(intra.get(c, 0.0)),
                other_support=float(other.get(c, 0.0)),
                bin_median_support=bin_med,
                is_anchor=bool(anchor_mask.get(c, False)),
            )

    # Hard anchors are the only contigs allowed to define read-level host direction in conservative binning.
    # Strong bins can contribute all anchors; weak bins contribute only their single best anchor.
    hard_anchor_mask: dict[str, bool] = {c: False for c in contig_len.keys()}
    hard_anchor_count: dict[str, int] = defaultdict(int)
    for bin_id, contigs in bin_to_contigs.items():
        if bin_status.get(bin_id) == "impure":
            continue
        anchor_contigs = [c for c in contigs if bool(anchor_mask.get(c, False))]
        if not anchor_contigs:
            continue
        if bin_status.get(bin_id) == "strong":
            chosen = anchor_contigs
        else:
            chosen = [
                max(
                    anchor_contigs,
                    key=lambda c: (
                        float(anchor_weight.get(c, 0.0)),
                        float(intra.get(c, 0.0)),
                        int(contig_len.get(c, 0)),
                        str(c),
                    ),
                )
            ]
        for c in chosen:
            hard_anchor_mask[c] = True
            hard_anchor_count[bin_id] += 1

    return PreCleanupResult(
        cleaned_assignment=contig_to_bin,
        anchor_mask=anchor_mask,
        hard_anchor_mask=hard_anchor_mask,
        anchor_weight=anchor_weight,
        bin_status=bin_status,
        bin_weight=bin_weight,
        anchor_count=dict(anchor_count),
        hard_anchor_count=dict(hard_anchor_count),
        removed_contigs=tuple(sorted(removed)),
        removed_by_bin={b: tuple(sorted(cs)) for b, cs in removed_by_bin.items()},
    )


def _entropy(probs: list[float]) -> float:
    """Natural entropy (nats)."""
    h = 0.0
    for p in probs:
        p = float(p)
        if p > 0.0:
            h -= p * math.log(p)
    return float(h)


def _effective_hosts(entropy_nats: float) -> float:
    """Effective support count: exp(entropy)."""
    return float(math.exp(float(entropy_nats)))


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, float(x))))


def _bin_weight_from_status(status: str) -> float:
    if status == "strong":
        return 1.0
    if status == "weak":
        return 0.35
    return 0.0


def _anchor_weight_from_support(
    *,
    intra_support: float,
    other_support: float,
    bin_median_support: float,
    is_anchor: bool,
    eps: float = 1e-12,
) -> float:
    intra_support = max(0.0, float(intra_support))
    other_support = max(0.0, float(other_support))
    bin_median_support = max(0.0, float(bin_median_support))

    purity = max(0.0, intra_support - other_support) / (intra_support + other_support + eps)
    if intra_support <= 0.0:
        strength = 0.0
    elif bin_median_support <= 0.0:
        strength = 1.0
    else:
        strength = min(1.0, intra_support / (bin_median_support + eps))

    weight = purity * math.sqrt(max(0.0, strength))
    if is_anchor:
        weight = max(weight, 0.85)
    return _clip01(weight)


def _summarize_read_anchor_evidence(
    *,
    contigs: list[str],
    pi: list[float],
    contig_to_coarse: dict[str, str],
    hard_anchor_mask: dict[str, bool],
    anchor_weight: dict[str, float],
    bin_weight: dict[str, float],
    dominance_ratio: float,
) -> ReadAnchorEvidence:
    host_mass: dict[str, float] = {}
    hard_anchor_hits = 0

    for c, pc in zip(contigs, pi, strict=True):
        if not bool(hard_anchor_mask.get(c, False)):
            continue
        b = contig_to_coarse.get(c, "-1")
        if b == "-1":
            continue
        bw = float(bin_weight.get(b, 0.0))
        aw = float(anchor_weight.get(c, 0.0))
        if bw <= 0.0 or aw <= 0.0:
            continue
        host_mass[b] = host_mass.get(b, 0.0) + (bw * aw * float(pc))
        hard_anchor_hits += 1

    if not host_mass:
        return ReadAnchorEvidence(
            informative=False,
            reject_reason="anchor_sparse",
            host_mass={},
            hard_anchor_hits=int(hard_anchor_hits),
            top1_host="",
            top1_mass=0.0,
            top2_host="",
            top2_mass=0.0,
        )

    ranked = sorted(host_mass.items(), key=lambda kv: (-kv[1], kv[0]))
    top1_host, top1_mass = ranked[0]
    top2_host = ranked[1][0] if len(ranked) > 1 else ""
    top2_mass = float(ranked[1][1]) if len(ranked) > 1 else 0.0

    if top2_mass > 0.0 and float(top1_mass) < float(dominance_ratio) * float(top2_mass):
        return ReadAnchorEvidence(
            informative=False,
            reject_reason="anchor_conflict",
            host_mass=host_mass,
            hard_anchor_hits=int(hard_anchor_hits),
            top1_host=str(top1_host),
            top1_mass=float(top1_mass),
            top2_host=str(top2_host),
            top2_mass=float(top2_mass),
        )

    return ReadAnchorEvidence(
        informative=True,
        reject_reason="",
        host_mass=host_mass,
        hard_anchor_hits=int(hard_anchor_hits),
        top1_host=str(top1_host),
        top1_mass=float(top1_mass),
        top2_host=str(top2_host),
        top2_mass=float(top2_mass),
    )


def refine_bins(
    *,
    contigs_fasta: Path,
    ppl_contacts: Path,
    bins_tsv: Path,
    bam: Optional[Path],
    out_dir: Path,
    threads: int = 1,
    seed: int = 0,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """
    Legacy BAM/PPL refine entrypoint retained for backward compatibility only.

    This delegate is not part of the active `coarse -> refine -> associate`
    mainline. The current main refine entrypoint is `refine_bins_parquet(...)`.
    """
    from porebin.refine_legacy import refine_bins as _legacy_refine_bins

    return _legacy_refine_bins(
        contigs_fasta=contigs_fasta,
        ppl_contacts=ppl_contacts,
        bins_tsv=bins_tsv,
        bam=bam,
        out_dir=out_dir,
        threads=threads,
        seed=seed,
        logger=logger,
    )


def refine_bins_parquet(
    *,
    contigs_fasta: Path,
    contacts_parquet: Path,
    bins_tsv: Path,
    coverage_tsv: Optional[Path],
    out_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """
    Active parquet-based post-binning refine entrypoint.

    User-facing outputs:
      Primary:
        - `bins.refined.tsv`
        - `residual_pool.tsv`

      QC / audit:
        - `bin_qc.refined.tsv`
        - `refine_actions.tsv`
        - `run_refine.json`

      Compatibility / transition:
        - `contig_host_scores.tsv`

    Current internal behavior:
      - consumes coarse bins as candidate host communities
      - applies a bin-centric refinement skeleton
      - retains a legacy contig-state bridge internally only for compatibility
      - does not own downstream relation-mining outputs

    Evidence semantics (must hold):
      - Each Pore-C read r is a hyperedge (one row per contact in contacts.parquet).
      - contacts.parquet.contig_weights are pi_{r,c}: normalized evidence shares from read r to contig c.
        They are NOT posterior probabilities.
      - contacts.parquet.weight is q(r): read-level alignment quality weight in [0,1].

    Refine semantics (this function):
      - Coarse bins are candidate host communities only (not final truth).
      - Refine computes posterior-like host support scores theta_{c,b} for each contig c over candidate hosts b,
        primarily from contact evidence, with a weak coarse-label prior/regularizer (indicator on coarse_host).
      - Non-core contigs are exported to an explicit residual pool instead of
        being treated as a primary relation-mining output owned by refine.

    High-order read treatment:
      - Total read information is governed mainly by q(r).
      - No strong pairwise-style normalization (e.g. 2/[k(k-1)]) is used in this refine inference pass.

    Note:
      - coverage_tsv is currently accepted for API compatibility; this MVP refine path does not directly use it.
    """
    logger = logger or logging.getLogger("porebin")
    out_dir = out_dir.resolve()
    ensure_dir(out_dir)

    if not contigs_fasta.exists():
        raise FileNotFoundError(f"Contigs FASTA not found: {contigs_fasta}")
    if not contacts_parquet.exists():
        raise FileNotFoundError(f"Contacts Parquet not found: {contacts_parquet}")
    if not bins_tsv.exists():
        raise FileNotFoundError(f"Bins TSV not found: {bins_tsv}")

    run_json = out_dir / "run_refine.json"
    record: dict[str, Any] = {
        "porebin_version": __version__,
        "command": "refine",
        "started_at": utc_now_iso(),
        "ended_at": None,
        "status": "running",
        "inputs": {
            "contigs_fasta": str(contigs_fasta),
            "contacts_parquet": str(contacts_parquet),
            "bins_tsv": str(bins_tsv),
            "coverage_tsv": str(coverage_tsv) if coverage_tsv is not None else None,
        },
        "thresholds": {},
        "decisions": {},
        "stats": {},
        "outputs": {},
        "cwd": os.getcwd(),
    }
    write_json(run_json, record)

    stats = RefineStats()
    try:
        # Numerical stability constant used in prior, normalization, and entropy.
        eps = 1e-12

        contig_len = _read_contig_lengths(contigs_fasta)
        if not contig_len:
            raise RefineError(f"No contigs found in FASTA: {contigs_fasta}")
        stats.contigs_total = len(contig_len)
        min_contig_len_output, _min_contig_len_meta = auto_min_contig_len(contig_len)

        coverage_snapshot: Optional[dict[str, float]] = None
        _preflight_refine_scg_requirements()
        scg_result = ensure_scg_hits(contigs_fasta=contigs_fasta, out_dir=out_dir, logger=logger)
        scg_expected_markers = set(scg_result.expected_markers) if scg_result.expected_markers else None
        if coverage_tsv is not None and coverage_tsv.exists():
            min_contig_len_snapshot, _min_meta = auto_min_contig_len(contig_len)
            coverage_snapshot = _coverage_from_tsv(
                coverage_tsv,
                contig_len=contig_len,
                min_contig_len=min_contig_len_snapshot,
            )

        coarse_raw = _read_bins_tsv(bins_tsv)
        coarse_state = _build_refine_state_from_assignment(
            contig_len=contig_len,
            assignment=coarse_raw,
            coverage=coverage_snapshot,
        )
        coarse_qc_meta = _write_refine_qc_snapshot(
            snapshot_name="coarse",
            state=coarse_state,
            contacts_parquet=contacts_parquet,
            out_dir=out_dir,
            scg_hits_tsv=scg_result.hits_tsv,
            scg_expected_markers=scg_expected_markers,
            scg_status=scg_result.state,
            logger=logger,
        )
        cleanup = pre_refine_cleanup(
            contacts_parquet=contacts_parquet,
            bins_tsv=bins_tsv,
            contigs_fasta=contigs_fasta,
            coverage_tsv=coverage_tsv,
            out_dir=out_dir,
        )
        cleaned_state = _build_refine_state_from_assignment(
            contig_len=contig_len,
            assignment=cleanup.cleaned_assignment,
            coverage=coverage_snapshot,
        )
        cleaned_qc_meta = _write_refine_qc_snapshot(
            snapshot_name="cleaned",
            state=cleaned_state,
            contacts_parquet=contacts_parquet,
            out_dir=out_dir,
            scg_hits_tsv=scg_result.hits_tsv,
            scg_expected_markers=scg_expected_markers,
            scg_status=scg_result.state,
            logger=logger,
        )
        cleanup, cleaned_state, cleaned_qc_meta, decontam_scg_gate = _apply_scg_gate_to_cleanup(
            cleanup=cleanup,
            coarse_qc_meta=coarse_qc_meta,
            cleaned_qc_meta=cleaned_qc_meta,
            contig_len=contig_len,
            coverage_snapshot=coverage_snapshot,
            contacts_parquet=contacts_parquet,
            out_dir=out_dir,
            scg_hits_tsv=scg_result.hits_tsv,
            scg_expected_markers=scg_expected_markers,
            scg_state=scg_result.state,
            logger=logger,
        )

        # Candidate host bins set B (exclude impure bins; strong bins dominate via higher bin_weight).
        B: list[str] = sorted({b for b, w in cleanup.bin_weight.items() if float(w) > 0.0})
        B_set = set(B)
        stats.bins_total = int(len(B))
        stats.bins_kept_initial = int(stats.bins_total)

        # Coarse label per contig in the FASTA; use cleaned_assignment for scoring.
        contig_to_coarse: dict[str, str] = {}
        assigned = 0
        for c in contig_len.keys():
            b = cleanup.cleaned_assignment.get(c, "-1")
            b = str(b).strip() if b is not None else "-1"
            if b not in B_set or float(cleanup.bin_weight.get(b, 0.0)) <= 0.0:
                b = "-1"
            else:
                assigned += 1
            contig_to_coarse[c] = b
        stats.contigs_assigned_initial = int(assigned)
        stats.hard_anchors_total = int(sum(1 for v in cleanup.hard_anchor_mask.values() if v))

        # support[c][b] accumulates contact-driven host support mass for contig c towards candidate host bin b.
        # This is a refine-time host-support quantity (not pi_{r,c}).
        support: defaultdict[str, dict[str, float]] = defaultdict(dict)
        raw_reads_by_contig: defaultdict[str, int] = defaultdict(int)
        informative_reads_by_contig: defaultdict[str, int] = defaultdict(int)
        rejected_sparse_by_contig: defaultdict[str, int] = defaultdict(int)
        rejected_conflict_by_contig: defaultdict[str, int] = defaultdict(int)

        reads_total = 0
        reads_skipped_k_lt_2 = 0
        reads_informative = 0
        reads_rejected_anchor_sparse = 0
        reads_rejected_anchor_conflict = 0
        read_anchor_dominance_ratio = 2.0
        for contigs_raw, pi_raw, q in _iter_contacts_parquet(contacts_parquet, batch_size=100_000):
            reads_total += 1
            if not contigs_raw or not pi_raw:
                continue

            contigs: list[str] = []
            pi: list[float] = []
            for c, w in zip(contigs_raw, pi_raw, strict=True):
                c = str(c).strip()
                if not c:
                    continue
                if c not in contig_len:
                    continue
                ww = float(w)
                if ww <= 0.0:
                    continue
                contigs.append(c)
                pi.append(ww)

            if len(contigs) < 2:
                reads_skipped_k_lt_2 += 1
                continue

            s = float(sum(pi))
            if not (s > 0.0):
                continue
            pi = [float(x) / s for x in pi]

            for c in contigs:
                raw_reads_by_contig[c] += 1

            anchor_ev = _summarize_read_anchor_evidence(
                contigs=contigs,
                pi=pi,
                contig_to_coarse=contig_to_coarse,
                hard_anchor_mask=cleanup.hard_anchor_mask,
                anchor_weight=cleanup.anchor_weight,
                bin_weight=cleanup.bin_weight,
                dominance_ratio=read_anchor_dominance_ratio,
            )
            if not anchor_ev.informative:
                if anchor_ev.reject_reason == "anchor_conflict":
                    reads_rejected_anchor_conflict += 1
                    for c in contigs:
                        rejected_conflict_by_contig[c] += 1
                elif anchor_ev.reject_reason == "anchor_sparse":
                    reads_rejected_anchor_sparse += 1
                    for c in contigs:
                        rejected_sparse_by_contig[c] += 1
                    continue

            if anchor_ev.informative:
                reads_informative += 1
            q_read = float(q)
            host_of: list[str] = []
            self_anchor_mass_of: list[float] = []
            for c, pc in zip(contigs, pi, strict=True):
                b = contig_to_coarse.get(c, "-1")
                host_of.append(b)
                if bool(cleanup.hard_anchor_mask.get(c, False)) and b != "-1":
                    anchor_w = float(cleanup.anchor_weight.get(c, 0.0))
                    bin_w = float(cleanup.bin_weight.get(b, 0.0))
                    self_anchor_mass_of.append(bin_w * anchor_w * float(pc))
                else:
                    self_anchor_mass_of.append(0.0)

            # Conservative binning support update:
            #   - read host direction is defined only by hard anchors
            #   - target contigs never define their own read-level host direction
            #   - no informative anchors => read does not vote
            for c, pc, b_c, self_anchor_mass in zip(contigs, pi, host_of, self_anchor_mass_of, strict=True):
                pc = float(pc)
                if pc <= 0.0:
                    continue
                d = support.get(c)
                if d is None:
                    d = {}
                    support[c] = d
                updated = False
                for b, m in anchor_ev.host_mass.items():
                    m_eff = float(m)
                    if b_c == b:
                        m_eff = max(0.0, m_eff - float(self_anchor_mass))
                    if m_eff <= 0.0:
                        continue
                    d[b] = float(d.get(b, 0.0)) + q_read * pc * m_eff
                    updated = True
                if updated and anchor_ev.informative:
                    informative_reads_by_contig[c] += 1

        stats.reads_total = int(reads_total)
        stats.reads_kept = int(reads_informative)
        stats.reads_informative = int(reads_informative)
        stats.reads_skipped_k_lt_2 = int(reads_skipped_k_lt_2)
        stats.reads_rejected_anchor_sparse = int(reads_rejected_anchor_sparse)
        stats.reads_rejected_anchor_conflict = int(reads_rejected_anchor_conflict)
        stats.input_sorted_by_readid = True

        # Fixed MVP constants/thresholds (must be explicit and recorded).
        prior_strength = 0.05

        assign_top1_score_thresh = 0.80
        assign_margin_thresh = 0.50
        assign_eff_hosts_thresh = 1.5
        min_informative_reads_assign = 3
        min_informative_reads_relation = 2

        accessory_eff_hosts_thresh = 2.0
        accessory_entropy_thresh = float(math.log(2.0))
        accessory_top1_lt = 0.85
        reassign_support_margin_thresh = 0.2
        recruit_support_margin_thresh = 0.2

        # Transitional compatibility bridge retained during the Phase-2/3 migration.
        # Public refine outputs are exported below from explicit final-bin and residual
        # data structures; the bridge exists only for auditability and compatibility.
        contig_scores_path = out_dir / "contig_host_scores.tsv"
        residual_pool_path = out_dir / "residual_pool.tsv"
        refined_bins = out_dir / "bins.refined.tsv"
        refine_actions_path = out_dir / "refine_actions.tsv"

        B_count = int(len(B))
        contigs_all = sorted(contig_len.keys())

        compat_rows = _build_legacy_compatibility_rows(
            contigs_all=contigs_all,
            coarse_raw=coarse_raw,
            cleanup=cleanup,
            support=support,
            informative_reads_by_contig=dict(informative_reads_by_contig),
            raw_reads_by_contig=dict(raw_reads_by_contig),
            rejected_sparse_by_contig=dict(rejected_sparse_by_contig),
            rejected_conflict_by_contig=dict(rejected_conflict_by_contig),
            B=B,
            B_set=B_set,
            B_count=B_count,
            prior_strength=prior_strength,
            eps=eps,
            assign_top1_score_thresh=assign_top1_score_thresh,
            assign_margin_thresh=assign_margin_thresh,
            assign_eff_hosts_thresh=assign_eff_hosts_thresh,
            min_informative_reads_assign=min_informative_reads_assign,
            min_informative_reads_relation=min_informative_reads_relation,
        )

        final_assignment, residual_rows, outcome_counts, final_bins_used = _derive_final_assignment_and_residuals(
            compat_rows=compat_rows,
            cleanup=cleanup,
            contig_len=contig_len,
            min_contig_len=min_contig_len_output,
        )

        preliminary_refined_state = _build_refine_state_from_assignment(
            contig_len=contig_len,
            assignment=final_assignment,
            coverage=coverage_snapshot,
        )
        pre_split_refined_qc_meta = _collect_refine_qc_snapshot(
            state=preliminary_refined_state,
            contacts_parquet=contacts_parquet,
            scg_hits_tsv=scg_result.hits_tsv,
            scg_expected_markers=scg_expected_markers,
            scg_status=scg_result.state,
        )
        split_candidates = _generate_split_candidates(
            state=preliminary_refined_state,
            refined_snapshot=pre_split_refined_qc_meta,
            contigs_fasta=contigs_fasta,
            coverage_tsv=coverage_tsv,
            out_dir=out_dir,
            logger=logger,
        )

        split_decisions: list[SplitDecision] = []
        next_split_bin_id = _next_bin_id(final_assignment.values())
        for candidate in split_candidates:
            decision = _evaluate_split_candidate(
                candidate=candidate,
                current_assignment=final_assignment,
                current_state=preliminary_refined_state,
                current_snapshot=pre_split_refined_qc_meta,
                coarse_raw=coarse_raw,
                contacts_parquet=contacts_parquet,
                scg_hits_tsv=scg_result.hits_tsv,
                scg_expected_markers=scg_expected_markers,
                scg_state=scg_result.state,
                new_bin_id=str(next_split_bin_id),
            )
            split_decisions.append(decision)
            if decision.accepted:
                _apply_split_decision(
                    decision=decision,
                    final_assignment=final_assignment,
                    residual_rows=residual_rows,
                )
                next_split_bin_id += 1

        reassign_windows = _build_reassign_windows_from_split_decisions(
            split_decisions=split_decisions,
            final_assignment=final_assignment,
        )
        reassign_candidates: list[ReassignCandidate] = []
        reassign_decisions: list[ReassignDecision] = []
        if reassign_windows:
            post_split_state = _build_refine_state_from_assignment(
                contig_len=contig_len,
                assignment=final_assignment,
                coverage=coverage_snapshot,
            )
            post_split_snapshot = _collect_refine_qc_snapshot(
                state=post_split_state,
                contacts_parquet=contacts_parquet,
                scg_hits_tsv=scg_result.hits_tsv,
                scg_expected_markers=scg_expected_markers,
                scg_status=scg_result.state,
            )
            pair_support = _scan_reassign_pair_support_parquet(
                contacts_parquet=contacts_parquet,
                contig_len=contig_len,
                min_contig_len=post_split_state.min_contig_len,
                current_assignment=final_assignment,
                windows=reassign_windows,
            )
            reassign_candidates = _generate_reassign_candidates(
                windows=reassign_windows,
                pair_support=pair_support,
                current_assignment=final_assignment,
                support_margin_thresh=reassign_support_margin_thresh,
            )
            reassign_state = post_split_state
            reassign_snapshot = post_split_snapshot
            for candidate in reassign_candidates:
                decision = _evaluate_reassign_candidate(
                    candidate=candidate,
                    current_assignment=final_assignment,
                    current_state=reassign_state,
                    current_snapshot=reassign_snapshot,
                    coarse_raw=coarse_raw,
                    contacts_parquet=contacts_parquet,
                    scg_hits_tsv=scg_result.hits_tsv,
                    scg_expected_markers=scg_expected_markers,
                    scg_state=scg_result.state,
                )
                reassign_decisions.append(decision)
                if decision.accepted:
                    _apply_reassign_decision(
                        decision=decision,
                        final_assignment=final_assignment,
                        residual_rows=residual_rows,
                    )
                    reassign_state = _build_refine_state_from_assignment(
                        contig_len=contig_len,
                        assignment=final_assignment,
                        coverage=coverage_snapshot,
                    )
                    reassign_snapshot = _collect_refine_qc_snapshot(
                        state=reassign_state,
                        contacts_parquet=contacts_parquet,
                        scg_hits_tsv=scg_result.hits_tsv,
                        scg_expected_markers=scg_expected_markers,
                        scg_status=scg_result.state,
                    )

        recruit_candidates: list[RecruitCandidate] = []
        recruit_decisions: list[RecruitDecision] = []
        recruit_state = _build_refine_state_from_assignment(
            contig_len=contig_len,
            assignment=final_assignment,
            coverage=coverage_snapshot,
        )
        recruit_snapshot = _collect_refine_qc_snapshot(
            state=recruit_state,
            contacts_parquet=contacts_parquet,
            scg_hits_tsv=scg_result.hits_tsv,
            scg_expected_markers=scg_expected_markers,
            scg_status=scg_result.state,
        )
        keep_bins = set(final_assignment.values())
        _intra_recruit, _other_recruit, recruit_affinity, _recruit_meta = _scan_contacts_support_and_affinity_parquet(
            contacts_parquet=contacts_parquet,
            contig_len=contig_len,
            min_contig_len=recruit_state.min_contig_len,
            contig_to_bin=final_assignment,
        )
        recruit_candidates = _generate_recruit_candidates(
            residual_rows=residual_rows,
            affinity=recruit_affinity,
            keep_bins=keep_bins,
        )
        for candidate in recruit_candidates:
            current_residual_by_contig = {str(row.contig_name): row for row in residual_rows}
            decision = _evaluate_recruit_candidate(
                candidate=candidate,
                current_assignment=final_assignment,
                current_state=recruit_state,
                current_snapshot=recruit_snapshot,
                current_residual_by_contig=current_residual_by_contig,
                coarse_raw=coarse_raw,
                contacts_parquet=contacts_parquet,
                scg_hits_tsv=scg_result.hits_tsv,
                scg_expected_markers=scg_expected_markers,
                scg_state=scg_result.state,
                support_margin_thresh=recruit_support_margin_thresh,
            )
            recruit_decisions.append(decision)
            if decision.accepted:
                _apply_recruit_decision(
                    decision=decision,
                    final_assignment=final_assignment,
                    residual_rows=residual_rows,
                )
                recruit_state = _build_refine_state_from_assignment(
                    contig_len=contig_len,
                    assignment=final_assignment,
                    coverage=coverage_snapshot,
                )
                recruit_snapshot = _collect_refine_qc_snapshot(
                    state=recruit_state,
                    contacts_parquet=contacts_parquet,
                    scg_hits_tsv=scg_result.hits_tsv,
                    scg_expected_markers=scg_expected_markers,
                    scg_status=scg_result.state,
                )

        final_bins_used = set(final_assignment.values())
        outcome_counts["assigned"] = int(len(final_assignment))
        outcome_counts["residual_total"] = int(len(residual_rows))
        outcome_counts["residual_associate_candidate"] = int(
            sum(1 for rec in residual_rows if rec.refined_status == "residual_associate_candidate")
        )
        action_log = _build_refine_action_log(
            cleanup=cleanup,
            decontam_scg_gate=decontam_scg_gate,
            split_decisions=split_decisions,
            reassign_decisions=reassign_decisions,
            recruit_decisions=recruit_decisions,
        )

        _write_contig_host_scores_tsv(contig_scores_path, compat_rows)
        _write_bins_refined_tsv(refined_bins, final_assignment)
        _write_residual_pool_tsv(residual_pool_path, residual_rows)
        _write_refine_actions_tsv(refine_actions_path, action_log)

        stats.contigs_assigned_final = int(outcome_counts["assigned"])
        stats.contigs_relation_only = int(outcome_counts["relation_only_compat"])
        stats.contigs_abstained = int(outcome_counts["abstained_compat"])
        stats.bins_kept_final = int(len(final_bins_used))
        stats.contigs_residual_total = int(outcome_counts["residual_total"])
        stats.contigs_residual_associate_candidate = int(outcome_counts["residual_associate_candidate"])
        stats.decontam_removed = int(len(cleanup.removed_contigs))
        stats.split_triggered = int(len(split_candidates))
        stats.split_applied = int(sum(1 for decision in split_decisions if decision.accepted))
        stats.reassign_moved = int(
            sum(1 for decision in reassign_decisions if decision.accepted and decision.final_action == "move_to_sibling")
        )
        stats.reassign_unbinned = int(
            sum(1 for decision in reassign_decisions if decision.accepted and decision.final_action == "residualize")
        )
        stats.recruit_assigned = int(
            sum(1 for decision in recruit_decisions if decision.accepted and decision.final_action == "assign_to_bin")
        )

        refined_state = _build_refine_state_from_assignment(
            contig_len=contig_len,
            assignment=final_assignment,
            coverage=coverage_snapshot,
        )
        refined_qc_meta = _write_refine_qc_snapshot(
            snapshot_name="refined",
            state=refined_state,
            contacts_parquet=contacts_parquet,
            out_dir=out_dir,
            scg_hits_tsv=scg_result.hits_tsv,
            scg_expected_markers=scg_expected_markers,
            scg_status=scg_result.state,
            logger=logger,
        )

        record["thresholds"] = {
            "eps": eps,
            "bin_weight_status_map": {"strong": 1.0, "weak": 0.35, "impure": 0.0},
            "conservative_binning_thresholds": {
                "top1_score_min": assign_top1_score_thresh,
                "margin_min": assign_margin_thresh,
                "effective_hosts_max": assign_eff_hosts_thresh,
                "min_informative_reads": min_informative_reads_assign,
            },
            "compatibility_residual_bridge_thresholds": {
                "min_informative_reads": min_informative_reads_relation,
                "effective_hosts_min": accessory_eff_hosts_thresh,
                "entropy_min": accessory_entropy_thresh,
                "top1_score_lt_if_entropy_trigger": accessory_top1_lt,
            },
            "reassign_thresholds": {
                "support_margin_min_for_move": reassign_support_margin_thresh,
                "support_margin_min_for_stable_keep": reassign_support_margin_thresh,
            },
            "recruit_thresholds": {
                "support_margin_min_for_assign": recruit_support_margin_thresh,
            },
            "read_anchor_thresholds": {
                "dominance_ratio_min": read_anchor_dominance_ratio,
            },
            "ambiguous_rule": "not assigned_bin",
        }
        record["decisions"] = {
            "coarse_bins_semantics": "candidate_host_communities_for_anchor_discovery",
            "contig_weights_semantics": "normalized_evidence_share",
            "refine_method": "bin_centric_refinement_with_live_split_and_compat_bridge",
            "refine_orchestration_model": "bin_centric_live_split_reassign_recruit",
            "pipeline_stages": [
                "load_coarse_bins",
                "qc_snapshot_coarse",
                "candidate_generation_precleanup",
                "candidate_evaluation_precleanup",
                "apply_accepted_operations",
                "qc_snapshot_cleaned",
                "compatibility_bridge_generation_legacy",
                "generate_split_candidates",
                "evaluate_split_candidates",
                "apply_accepted_split_operations",
                "build_reassign_windows",
                "generate_reassign_candidates",
                "evaluate_reassign_candidates",
                "apply_accepted_reassign_operations",
                "generate_recruit_candidates",
                "evaluate_recruit_candidates",
                "apply_accepted_recruit_operations",
                "export_final_bins",
                "export_residual_pool",
                "export_refine_actions",
                "qc_snapshot_refined",
            ],
            "high_order_read_philosophy": "q_controls_total_information",
            "eta_k": 1.0,
            "soft_gating_enabled": True,
            "candidate_bin_rule": "non_impure_bins",
            "anchor_discovery_role": "host_anchor_discovery",
            "conservative_binning_role": "conservative_host_binning",
            "residual_pool_role": "explicit_handoff_to_associate",
            "associate_module_expected": True,
            "legacy_contig_host_scores_retained": True,
            "legacy_contig_host_scores_role": "compatibility_audit_bridge",
            "legacy_contig_host_scores_scope": "pre_split_compatibility_bridge",
            "legacy_accessory_output_removed_from_refine": True,
            "legacy_refine_state_labels_present": True,
            "legacy_contig_state_role": "compatibility_bridge_only_not_public_refine_semantics",
            "legacy_helper_roles": {**LEGACY_HELPER_ROLES},
            "split_live_operation_enabled": True,
            "split_candidate_source": "scg_triggered_local_recluster_on_preliminary_final_bins",
            "split_scg_required": True,
            "split_scg_no_resource_behavior": "fail_fast_with_user_actionable_error",
            "reassign_live_operation_enabled": True,
            "reassign_scope": "accepted_split_sibling_pairs_only",
            "reassign_allowed_actions": ["keep", "move_to_sibling", "residualize"],
            "reassign_contact_role": "primary_local_cleanup_signal",
            "reassign_scg_role": "non_worsening_gate_only",
            "recruit_live_operation_enabled": True,
            "recruit_input_scope": "residual_unresolved_only",
            "recruit_allowed_actions": ["assign_to_bin", "keep_residual"],
            "recruit_target_scope": "any_existing_final_bin",
            "recruit_contact_role": "primary_global_residual_recovery_signal",
            "recruit_scg_role": "non_worsening_gate_only",
            "bin_weight_scheme": {
                "type": "status_piecewise_constant",
                "strong": 1.0,
                "weak": 0.35,
                "impure": 0.0,
            },
            "hard_anchor_rule": {
                "strong_bins": "all anchors are hard anchors",
                "weak_bins": "best anchor only",
                "impure_bins": "no hard anchors",
            },
            "anchor_weight_scheme": {
                "type": "support_purity_times_sqrt_strength_with_anchor_floor",
                "purity": "max(0, intra_support-other_support)/(intra_support+other_support+eps)",
                "strength": "min(1, intra_support/(bin_median_support+eps))",
                "anchor_floor_if_mask": 0.85,
            },
            "read_evidence_rule": "only_hard_anchors_define_host_direction",
            "read_reject_rules": {
                "anchor_sparse": "no hard-anchor host mass",
                "anchor_conflict": "top1 anchor host mass < dominance_ratio_min * top2",
            },
            "support_formula": "support[c,b] += q(r) * pi[r,c] * max(0, A_r(b) - self_anchor_term)",
            "support_host_mass_formula": "A_r(b) = sum_{u in hard_anchors_of_b} bin_weight(b) * anchor_weight(u) * pi[r,u]",
            "coarse_prior_used": True,
            "coarse_prior_strength": prior_strength,
            "coarse_prior_bin_weighted": True,
            "coarse_prior_requires_support_or_hard_anchor": True,
            "direct_feature_prior_used": False,
            "scg_enabled": bool(scg_result.enabled),
            "scg_state": scg_result.state,
            "scg_reason": scg_result.reason,
            "scg_cache_reused": bool(scg_result.reused_cache),
            "scg_resources_loaded": bool(scg_result.resources is not None),
            "scg_expected_markers_count": int(len(scg_result.expected_markers)),
            "scg_marker_set_id": (
                scg_result.resources.marker_set_id if scg_result.resources is not None else None
            ),
            "scg_db_version": (
                scg_result.resources.db_version if scg_result.resources is not None else None
            ),
        }
        record["stats"] = {
            **stats.__dict__,
            "candidate_bins_count": int(B_count),
            "contigs_core_like": int(outcome_counts["assigned"]),
            "contigs_ambiguous": int(outcome_counts["ambiguous_compat"]),
            "contigs_accessory_candidates_compat": int(outcome_counts["relation_only_compat"]),
            "contigs_relation_only": int(outcome_counts["relation_only_compat"]),
            "contigs_abstained": int(outcome_counts["abstained_compat"]),
            "contigs_residual_total": int(outcome_counts["residual_total"]),
            "contigs_residual_associate_candidate": int(outcome_counts["residual_associate_candidate"]),
            "split_candidates_total": int(len(split_candidates)),
            "split_candidates_accepted": int(sum(1 for decision in split_decisions if decision.accepted)),
            "split_candidates_rejected": int(sum(1 for decision in split_decisions if not decision.accepted)),
            "split_candidates_local_recluster": int(
                sum(1 for candidate in split_candidates if candidate.candidate_path == "local_recluster")
            ),
            "split_candidates_scg_priority": int(
                sum(1 for candidate in split_candidates if candidate.priority_source == "scg_priority")
            ),
            "reassign_windows_total": int(len(reassign_windows)),
            "reassign_candidates_total": int(len(reassign_candidates)),
            "reassign_candidates_move_to_sibling": int(
                sum(1 for candidate in reassign_candidates if candidate.candidate_action == "move_to_sibling")
            ),
            "reassign_candidates_residualize": int(
                sum(1 for candidate in reassign_candidates if candidate.candidate_action == "residualize")
            ),
            "reassign_decisions_accepted": int(sum(1 for decision in reassign_decisions if decision.accepted)),
            "reassign_decisions_moved": int(
                sum(1 for decision in reassign_decisions if decision.accepted and decision.final_action == "move_to_sibling")
            ),
            "reassign_decisions_residualized": int(
                sum(1 for decision in reassign_decisions if decision.accepted and decision.final_action == "residualize")
            ),
            "recruit_candidates_total": int(len(recruit_candidates)),
            "recruit_candidates_assign_to_bin": int(sum(1 for candidate in recruit_candidates if candidate.target_bin)),
            "recruit_decisions_accepted": int(sum(1 for decision in recruit_decisions if decision.accepted)),
            "recruit_decisions_assigned": int(
                sum(1 for decision in recruit_decisions if decision.accepted and decision.final_action == "assign_to_bin")
            ),
            "recruit_decisions_kept_residual": int(
                sum(1 for decision in recruit_decisions if decision.accepted and decision.final_action == "keep_residual")
            ),
            "split_check_bins_refined": int(refined_qc_meta.get("split_checks", 0)),
            "split_check_bins_refined_scg_priority": int(
                sum(
                    1
                    for rec in refined_qc_meta.get("split_checks_records", [])
                    if str(rec.priority_source) == "scg_priority"
                )
            ),
            "split_residual_contigs": int(sum(len(decision.residual_rows) for decision in split_decisions if decision.accepted)),
            "refine_actions_total": int(len(action_log)),
            "refine_actions_accepted": int(sum(1 for rec in action_log if rec.accepted)),
            "refine_actions_rejected": int(sum(1 for rec in action_log if not rec.accepted)),
            "refine_actions_scg_evaluated": int(sum(1 for rec in action_log if rec.scg_gate_used)),
            "refine_actions_scg_vetoed": int(sum(1 for rec in action_log if rec.scg_gate_result == "veto")),
            "refine_actions_scg_supported": int(sum(1 for rec in action_log if rec.scg_gate_result == "support")),
            "coarse_qc_bins": int(coarse_qc_meta["bins"]),
            "coarse_qc_suspects": int(coarse_qc_meta["suspects"]),
            "cleaned_qc_bins": int(cleaned_qc_meta["bins"]),
            "cleaned_qc_suspects": int(cleaned_qc_meta["suspects"]),
            "refined_qc_bins": int(refined_qc_meta["bins"]),
            "refined_qc_suspects": int(refined_qc_meta["suspects"]),
        }
        record["outputs"] = {
            "bins_refined_tsv": str(refined_bins),
            "residual_pool_tsv": str(residual_pool_path),
            "refine_actions_tsv": str(refine_actions_path),
            "contig_host_scores_tsv": str(contig_scores_path),
            "bin_qc_refined_tsv": str(refined_qc_meta["bin_qc_tsv"]),
            "suspect_bins_refined_tsv": str(refined_qc_meta["suspect_bins_tsv"]),
            "scg_hits_tsv": (str(scg_result.hits_tsv) if scg_result.hits_tsv is not None else None),
            "bin_qc_coarse_tsv": str(coarse_qc_meta["bin_qc_tsv"]),
            "suspect_bins_coarse_tsv": str(coarse_qc_meta["suspect_bins_tsv"]),
            "bin_qc_cleaned_tsv": str(cleaned_qc_meta["bin_qc_tsv"]),
            "suspect_bins_cleaned_tsv": str(cleaned_qc_meta["suspect_bins_tsv"]),
            "deprecated_relation_outputs": {"accessory_associations_tsv": None},
        }
        record["status"] = "ok"
        return refined_bins
    except Exception as exc:
        record["status"] = "error"
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        record["ended_at"] = utc_now_iso()
        write_json(run_json, record)


def auto_min_contig_len(contig_len: dict[str, int]) -> tuple[int, dict[str, Any]]:
    """
    SemiBin-like heuristic: choose MIN_CONTIG_LEN=1000 or 2500 depending on how much
    sequence mass sits in the 1000-2500bp range.
    """
    total_bp = sum(contig_len.values())
    bp_1000_2500 = sum(L for L in contig_len.values() if 1000 <= L <= 2500)
    ratio = (bp_1000_2500 / total_bp) if total_bp else 0.0
    min_len = 1000 if ratio >= 0.05 else 2500
    meta = {
        "type": "semiBin_ratio_1000_2500",
        "ratio_1000_2500_bp": ratio,
        "ratio_threshold": 0.05,
        "rule": "min_contig_len=1000 if ratio>=0.05 else 2500",
    }
    return min_len, meta


def _read_contig_lengths(contigs_fasta: Path) -> dict[str, int]:
    contig_len: dict[str, int] = {}
    for name, _header, seq in iter_fasta_records(contigs_fasta):
        contig_len[name] = len(seq)
    return contig_len


def _read_bins_tsv(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if row[0] in {"contig_name", "contig"}:
                continue
            if len(row) < 2:
                raise RefineError(f"Invalid bins row in {path}: {row}")
            mapping[row[0]] = row[1]
    return mapping


def _bin_sizes(
    coarse: dict[str, str],
    contig_len: dict[str, int],
    *,
    min_contig_len: int,
) -> tuple[Counter[str], Counter[str]]:
    bp: Counter[str] = Counter()
    n: Counter[str] = Counter()
    for contig, bin_id in coarse.items():
        L = contig_len.get(contig)
        if L is None or L < min_contig_len:
            continue
        bp[bin_id] += L
        n[bin_id] += 1
    return bp, n


def _bin_sizes_from_assignment(
    contig_to_bin: dict[str, str],
    contig_len: dict[str, int],
) -> tuple[Counter[str], Counter[str]]:
    bp: Counter[str] = Counter()
    n: Counter[str] = Counter()
    for contig, bin_id in contig_to_bin.items():
        L = contig_len.get(contig)
        if L is None:
            continue
        bp[bin_id] += L
        n[bin_id] += 1
    return bp, n


def _drop_bins_below_min(
    bin_to_contigs: dict[str, list[str]],
    contig_to_bin: dict[str, str],
    keep_bins: set[str],
    unbinned_reason: dict[str, str],
) -> None:
    for bin_id in list(bin_to_contigs.keys()):
        if bin_id in keep_bins:
            continue
        for c in bin_to_contigs[bin_id]:
            contig_to_bin.pop(c, None)
            unbinned_reason[c] = "tiny_bin"
        bin_to_contigs.pop(bin_id, None)


def _coverage_from_bam(
    bam_path: Path,
    contig_len: dict[str, int],
    *,
    min_contig_len: int,
    logger: logging.Logger,
) -> dict[str, float]:
    """
    Coverage proxy (fast, no per-base depth): aligned_bases_sum / contig_length.
    """
    try:
        import pysam
    except Exception as exc:  # pragma: no cover
        raise RefineError("Coverage requested but 'pysam' is not installed. Install it or omit --bam.") from exc

    logger.info("Refine: computing coverage from BAM (aligned_bases_sum/contig_length)")
    bam = pysam.AlignmentFile(str(bam_path), "rb")
    ref_names = list(bam.references)
    aligned_bases: defaultdict[str, int] = defaultdict(int)
    for aln in bam.fetch(until_eof=True):
        if aln.is_unmapped:
            continue
        rid = aln.reference_id
        if rid is None or rid < 0 or rid >= len(ref_names):
            continue
        ref = ref_names[rid]
        aln_len = aln.query_alignment_length
        if not aln_len and aln.reference_end is not None and aln.reference_start is not None:
            aln_len = aln.reference_end - aln.reference_start
        if not aln_len:
            continue
        aligned_bases[ref] += int(aln_len)
    bam.close()

    cov: dict[str, float] = {}
    for contig, L in contig_len.items():
        if L < min_contig_len:
            continue
        cov[contig] = aligned_bases.get(contig, 0) / float(L) if L else 0.0
    return cov


def _coverage_from_tsv(
    path: Path,
    *,
    contig_len: dict[str, int],
    min_contig_len: int,
) -> dict[str, float]:
    cov: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if row[0] in {"contig_name", "contig"}:
                continue
            if len(row) < 2:
                continue
            name = row[0].strip()
            if contig_len.get(name, 0) < min_contig_len:
                continue
            try:
                cov[name] = float(row[1])
            except ValueError:
                continue
    return cov


def _bin_coverage_stats(
    bin_to_contigs: dict[str, list[str]],
    cov: dict[str, float],
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for bin_id, contigs in bin_to_contigs.items():
        vals = [cov.get(c, 0.0) for c in contigs]
        if not vals:
            continue
        med = _median(vals)
        mad = _mad(vals, center=med)
        stats[bin_id] = {"median": med, "mad": mad}
    return stats


def _iter_contacts_parquet(
    contacts_parquet: Path,
    *,
    batch_size: int,
) -> Iterable[tuple[list[str], list[float], float]]:
    """
    Stream contacts.parquet rows.

    Expected columns:
      - contigs: list[str]
      - contig_weights: list[float]  (pi_{r,c} normalized evidence shares; should sum to 1 per read)
      - weight: float (q(r), read quality weight; default 1.0 if missing)
    """
    try:
        for row in iter_canonical_contact_rows(
            contacts_parquet,
            parquet_batch_size=int(batch_size),
            require_contig_weights=True,
        ):
            yield list(row.contigs), list(row.contig_weights or []), float(row.weight)
    except ContactHypergraphError as exc:
        raise RefineError(str(exc)) from exc


def _scan_contacts_support_and_affinity_parquet(
    *,
    contacts_parquet: Path,
    contig_len: dict[str, int],
    min_contig_len: int,
    contig_to_bin: dict[str, str],
) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, float]], dict[str, Any]]:
    """
    Streaming pass over BAM-derived contacts.parquet (one row per read/contact).

    This implements the soft-hyperedge support in docs/BAM_HYPERGRAPH_SPECTRAL_PIPELINE.md:
      - P_{r,c} is contig_weights (renormalized after filtering)
      - w(r) = OrderNorm(k) * q(r) where q(r) is contact weight
      - Support towards a bin uses S(v,b)=sum_r w(r) P_{r,v} * (sum_{u in b} P_{r,u})
    """
    intra: defaultdict[str, float] = defaultdict(float)
    other: defaultdict[str, float] = defaultdict(float)
    affinity: dict[str, dict[str, float]] = {}

    reads_total = 0
    reads_kept = 0
    reads_skipped = 0

    batch_size = 100_000

    for contigs_raw, p_raw, R in _iter_contacts_parquet(contacts_parquet, batch_size=batch_size):
        reads_total += 1

        # Filter contigs by length and merge duplicates by summing P.
        seen: dict[str, int] = {}
        contigs: list[str] = []
        p: list[float] = []
        for c, w in zip(contigs_raw, p_raw, strict=True):
            if not c:
                continue
            if contig_len.get(c, 0) < min_contig_len:
                continue
            ww = float(w)
            if ww <= 0.0:
                continue
            if c in seen:
                p[seen[c]] += ww
            else:
                seen[c] = len(contigs)
                contigs.append(c)
                p.append(ww)

        if len(contigs) < 2:
            reads_skipped += 1
            continue

        s = float(sum(p))
        if not (s > 0.0):
            reads_skipped += 1
            continue
        p = [x / s for x in p]

        k = len(contigs)
        reads_kept += 1
        w_contact = order_norm(k, method="pair") * float(R)

        bin_mass: dict[str, float] = {}
        for c, pc in zip(contigs, p, strict=True):
            b = contig_to_bin.get(c)
            if b is None:
                continue
            bin_mass[b] = bin_mass.get(b, 0.0) + float(pc)
        if not bin_mass:
            continue

        # Track top-2 bin masses for "best other" without per-contig dicts.
        max1_bin = None
        max1 = -1.0
        max2_bin = None
        max2 = -1.0
        for bb, m in bin_mass.items():
            m = float(m)
            if m > max1:
                max2, max2_bin = max1, max1_bin
                max1, max1_bin = m, bb
            elif m > max2:
                max2, max2_bin = m, bb

        for c, pc in zip(contigs, p, strict=True):
            b0 = contig_to_bin.get(c)
            if b0 is not None:
                m0 = float(bin_mass.get(b0, 0.0))
                intra[c] += float(w_contact) * float(pc) * max(0.0, m0 - float(pc))
                if len(bin_mass) > 1:
                    if max1_bin != b0:
                        other[c] += float(w_contact) * float(pc) * float(max1)
                    elif max2_bin is not None and max2 > 0.0:
                        other[c] += float(w_contact) * float(pc) * float(max2)
            else:
                top = affinity.get(c)
                if top is None:
                    top = {}
                    affinity[c] = top
                for bb, m in bin_mass.items():
                    top[bb] = top.get(bb, 0.0) + float(w_contact) * float(pc) * float(m)
                if len(top) > 32:
                    _prune_topk(top, k=3)

    return dict(intra), dict(other), affinity, {
        "reads_total": reads_total,
        "reads_kept": reads_kept,
        "reads_skipped_k_lt_2": reads_skipped,
        "input_sorted_by_readid": True,
        "source": "contacts_parquet",
    }


def _fit_2gmm_params(xs: list[float]) -> dict[str, float]:
    """
    Fit a 1D 2-component Gaussian mixture with EM.
    Returns parameters: w1, mu1, var1, mu2, var2.
    """
    if not xs:
        return {"w1": 0.5, "mu1": 0.0, "var1": 1.0, "mu2": 0.0, "var2": 1.0}
    n = len(xs)
    xs_sorted = sorted(xs)
    mu1 = float(xs_sorted[n // 4])
    mu2 = float(xs_sorted[(3 * n) // 4])
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / n
    var1 = var2 = max(float(var), 1e-3)
    w1 = 0.5

    for _ in range(50):
        r1_sum = 0.0
        r2_sum = 0.0
        mu1_num = 0.0
        mu2_num = 0.0
        var1_num = 0.0
        var2_num = 0.0

        # E-step: responsibilities
        for x in xs:
            l1 = math.log(w1) + _log_norm_pdf(x, mu1, var1)
            l2 = math.log(1.0 - w1) + _log_norm_pdf(x, mu2, var2)
            m = max(l1, l2)
            d = math.exp(l1 - m) + math.exp(l2 - m)
            r1 = math.exp(l1 - m) / d
            r2 = 1.0 - r1
            r1_sum += r1
            r2_sum += r2
            mu1_num += r1 * x
            mu2_num += r2 * x

        if r1_sum <= 1e-9 or r2_sum <= 1e-9:
            break

        # M-step
        w1 = max(1e-3, min(1.0 - 1e-3, r1_sum / n))
        mu1 = mu1_num / r1_sum
        mu2 = mu2_num / r2_sum

        for x in xs:
            l1 = math.log(w1) + _log_norm_pdf(x, mu1, var1)
            l2 = math.log(1.0 - w1) + _log_norm_pdf(x, mu2, var2)
            m = max(l1, l2)
            d = math.exp(l1 - m) + math.exp(l2 - m)
            r1 = math.exp(l1 - m) / d
            r2 = 1.0 - r1
            var1_num += r1 * (x - mu1) ** 2
            var2_num += r2 * (x - mu2) ** 2

        var1 = max(var1_num / r1_sum, 1e-6)
        var2 = max(var2_num / r2_sum, 1e-6)

    return {"w1": float(w1), "mu1": float(mu1), "var1": float(var1), "mu2": float(mu2), "var2": float(var2)}


def _gmm_post_comp1(x: float, *, w1: float, mu1: float, var1: float, mu2: float, var2: float) -> float:
    l1 = math.log(float(w1)) + _log_norm_pdf(float(x), float(mu1), float(var1))
    l2 = math.log(1.0 - float(w1)) + _log_norm_pdf(float(x), float(mu2), float(var2))
    m = max(l1, l2)
    d = math.exp(l1 - m) + math.exp(l2 - m)
    return float(math.exp(l1 - m) / d)


def _reassign_or_unbin(
    *,
    contacts_parquet: Path,
    contig_len: dict[str, int],
    min_contig_len: int,
    contig_to_bin: dict[str, str],
    bin_to_contigs: dict[str, list[str]],
    intra_support: dict[str, float],
    other_support: dict[str, float],
    cov: Optional[dict[str, float]],
    bin_cov_stats: Optional[dict[str, dict[str, float]]],
    out_path: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    """
    Legacy refine helper retained as a compatibility tool, not active mainline logic.

    Historical behavior:
      - reassign or unbin contigs based on a data-driven split of
        LLR=log(intra)-log(other) via 1v2-GMM BIC

    Phase-5 role:
      - compatibility helper
      - possible future fallback
      - not part of the active bin-centric orchestration

    If 2-component model is preferred, contigs in the lower-mean component are
    treated as "suspects". For suspects, we find the best alternative bin by
    rescanning contacts.parquet and accumulating S(v,b).
    """
    eps = 1e-12
    llr_by_contig: dict[str, float] = {}
    xs: list[float] = []
    for c, b in contig_to_bin.items():
        if contig_len.get(c, 0) < min_contig_len:
            continue
        s_intra = float(intra_support.get(c, 0.0))
        s_other = float(other_support.get(c, 0.0))
        if s_other <= 0.0:
            continue
        llr = math.log(s_intra + eps) - math.log(s_other + eps)
        llr_by_contig[c] = float(llr)
        xs.append(float(llr))

    meta: dict[str, Any] = {
        "method": "llr_gmm_bic_1v2",
        "llr": "LLR(v)=log(intra_support+eps)-log(best_other_support+eps)",
        "n": int(len(xs)),
        "bic1": None,
        "bic2": None,
        "enabled": False,
        "moved": 0,
        "unbinned": 0,
    }

    # Default: no changes, but still write the file for traceability.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(
            "contig_name\told_bin\tnew_bin\taction\tllr\tintra_support\tother_support\tbest_bin_support\tcov\tcov_z\treason\n"
        )

        if len(xs) < 200:
            meta["reason"] = "skip_small_n"
            return meta

        bic1 = _bic(xs, k=1)
        bic2 = _bic(xs, k=2)
        meta["bic1"] = float(bic1)
        meta["bic2"] = float(bic2)
        if bic2 >= bic1:
            meta["reason"] = "bic_prefers_1_component"
            return meta

        params = _fit_2gmm_params(xs)
        # Identify low-mean component.
        mu1 = float(params["mu1"])
        mu2 = float(params["mu2"])
        low_is_1 = mu1 <= mu2

        suspects: set[str] = set()
        for c, llr in llr_by_contig.items():
            r1 = _gmm_post_comp1(llr, **params)
            r_low = r1 if low_is_1 else (1.0 - r1)
            if r_low > 0.5:
                suspects.add(c)

        meta["enabled"] = True
        meta["params"] = params
        meta["low_component"] = 1 if low_is_1 else 2
        meta["suspects"] = int(len(suspects))

        if not suspects:
            meta["reason"] = "no_suspects"
            return meta

        # Rescan contacts.parquet to find best alternative bin for suspects.
        best_scores: dict[str, dict[str, float]] = {c: {} for c in suspects}
        for contigs_raw, p_raw, R in _iter_contacts_parquet(contacts_parquet, batch_size=100_000):
            # Filter/normalize (same as scan pass)
            seen: dict[str, int] = {}
            contigs: list[str] = []
            p: list[float] = []
            for cc, ww in zip(contigs_raw, p_raw, strict=True):
                if not cc:
                    continue
                if contig_len.get(cc, 0) < min_contig_len:
                    continue
                wv = float(ww)
                if wv <= 0.0:
                    continue
                if cc in seen:
                    p[seen[cc]] += wv
                else:
                    seen[cc] = len(contigs)
                    contigs.append(cc)
                    p.append(wv)
            if len(contigs) < 2:
                continue
            s = float(sum(p))
            if not (s > 0.0):
                continue
            p = [x / s for x in p]
            k = len(contigs)
            w_contact = order_norm(k, method="pair") * float(R)

            bin_mass: dict[str, float] = {}
            for cc, pc in zip(contigs, p, strict=True):
                bb = contig_to_bin.get(cc)
                if bb is None:
                    continue
                bin_mass[bb] = bin_mass.get(bb, 0.0) + float(pc)
            if not bin_mass:
                continue

            for cc, pc in zip(contigs, p, strict=True):
                if cc not in suspects:
                    continue
                old_b = contig_to_bin.get(cc)
                if old_b is None:
                    continue
                top = best_scores.get(cc)
                if top is None:
                    continue
                for bb, m in bin_mass.items():
                    if bb == old_b:
                        continue
                    top[bb] = top.get(bb, 0.0) + float(w_contact) * float(pc) * float(m)
                if len(top) > 16:
                    _prune_topk(top, k=3)

        # Apply moves
        moved = 0
        unbinned = 0
        new_assign: dict[str, str] = dict(contig_to_bin)
        for c in suspects:
            old_b = contig_to_bin.get(c)
            if old_b is None:
                continue
            cand = best_scores.get(c, {})
            best_b = None
            best_s = 0.0
            for bb, ss in cand.items():
                if float(ss) > float(best_s):
                    best_b = bb
                    best_s = float(ss)

            s_intra = float(intra_support.get(c, 0.0))
            s_other = float(other_support.get(c, 0.0))
            llr = float(llr_by_contig.get(c, 0.0))

            cov_val = cov.get(c) if cov is not None else None
            cov_z = None
            if cov_val is not None and bin_cov_stats is not None and old_b in bin_cov_stats:
                cov_med = float(bin_cov_stats[old_b].get("median", 0.0))
                cov_mad = float(bin_cov_stats[old_b].get("mad", 0.0))
                denom = cov_mad if cov_mad > 0 else 1e-12
                cov_z = abs(float(cov_val) - cov_med) / denom

            if best_b is not None and best_s > 0.0:
                new_assign[c] = str(best_b)
                moved += 1
                fh.write(
                    f"{c}\t{old_b}\t{best_b}\tmoved\t{llr:.6g}\t{s_intra:.6g}\t{s_other:.6g}\t{best_s:.6g}\t"
                    f"{'' if cov_val is None else f'{cov_val:.6g}'}\t"
                    f"{'' if cov_z is None else f'{cov_z:.6g}'}\tllr_low_component\n"
                )
            else:
                new_assign.pop(c, None)
                unbinned += 1
                fh.write(
                    f"{c}\t{old_b}\t\tunbinned\t{llr:.6g}\t{s_intra:.6g}\t{s_other:.6g}\t\t"
                    f"{'' if cov_val is None else f'{cov_val:.6g}'}\t"
                    f"{'' if cov_z is None else f'{cov_z:.6g}'}\tllr_low_component_no_target\n"
                )

        contig_to_bin.clear()
        contig_to_bin.update(new_assign)
        bin_to_contigs.clear()
        for c, b in contig_to_bin.items():
            bin_to_contigs.setdefault(b, []).append(c)

        meta["moved"] = int(moved)
        meta["unbinned"] = int(unbinned)
        meta["reason"] = "applied"
        return meta


def _recruit_gmm(
    *,
    affinity: dict[str, dict[str, float]],
    contig_len: dict[str, int],
    min_contig_len: int,
    contig_to_bin: dict[str, str],
    bin_to_contigs: dict[str, list[str]],
    keep_bins: set[str],
    cov: Optional[dict[str, float]],
    bin_cov_stats: Optional[dict[str, dict[str, float]]],
    out_path: Path,
) -> dict[str, Any]:
    """
    Legacy recruit helper retained as a compatibility/fallback tool.

    Phase-5 role:
      - compatibility helper
      - possible future fallback
      - not part of the active bin-centric orchestration

    Historical behavior:
      - recruit unbinned contigs using a data-driven split of
        LLR=log(top1)-log(top2) via 1v2-GMM BIC
    """
    eps = 1e-12
    candidates: dict[str, tuple[str, float, float, float]] = {}
    xs: list[float] = []
    for c, scores in affinity.items():
        if c in contig_to_bin or contig_len.get(c, 0) < min_contig_len:
            continue
        best = _top_candidates(scores, keep_bins=keep_bins, topk=2)
        if not best:
            continue
        b1, s1 = best[0]
        s2 = best[1][1] if len(best) > 1 else 0.0
        llr = math.log(float(s1) + eps) - math.log(float(s2) + eps)
        candidates[c] = (b1, float(s1), float(s2), float(llr))
        xs.append(float(llr))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("contig_name\tbin_id\tscore1\tscore2\tllr\taction\tcov\tcov_z\treason\n")

        meta: dict[str, Any] = {"method": "llr_gmm_bic_1v2", "n": int(len(xs)), "bic1": None, "bic2": None}
        if len(xs) < 200:
            meta["reason"] = "skip_small_n"
            return {"assigned": [], **meta}

        bic1 = _bic(xs, k=1)
        bic2 = _bic(xs, k=2)
        meta["bic1"] = float(bic1)
        meta["bic2"] = float(bic2)
        if bic2 >= bic1:
            meta["reason"] = "bic_prefers_1_component"
            return {"assigned": [], **meta}

        params = _fit_2gmm_params(xs)
        mu1 = float(params["mu1"])
        mu2 = float(params["mu2"])
        high_is_1 = mu1 >= mu2

        assigned: list[str] = []
        for c, (b1, s1, s2, llr) in candidates.items():
            r1 = _gmm_post_comp1(llr, **params)
            r_high = r1 if high_is_1 else (1.0 - r1)
            if r_high <= 0.5:
                continue

            cov_val = cov.get(c) if cov is not None else None
            cov_z = None
            if cov_val is not None and bin_cov_stats is not None and b1 in bin_cov_stats:
                cov_med = float(bin_cov_stats[b1].get("median", 0.0))
                cov_mad = float(bin_cov_stats[b1].get("mad", 0.0))
                denom = cov_mad if cov_mad > 0 else 1e-12
                cov_z = abs(float(cov_val) - cov_med) / denom

            contig_to_bin[c] = b1
            bin_to_contigs.setdefault(b1, []).append(c)
            assigned.append(c)
            fh.write(
                f"{c}\t{b1}\t{s1:.6g}\t{s2:.6g}\t{llr:.6g}\tassigned\t"
                f"{'' if cov_val is None else f'{cov_val:.6g}'}\t"
                f"{'' if cov_z is None else f'{cov_z:.6g}'}\tllr_high_component\n"
            )

        meta["params"] = params
        meta["high_component"] = 1 if high_is_1 else 2
        meta["assigned"] = int(len(assigned))
        meta["reason"] = "applied"
        return {"assigned": assigned, **meta}


def _split_bins_parquet(
    *,
    contacts_parquet: Path,
    contig_len: dict[str, int],
    min_contig_len: int,
    contig_to_bin: dict[str, str],
    bin_to_contigs: dict[str, list[str]],
    cov: Optional[dict[str, float]],
    out_path: Path,
    seed: int,
) -> dict[str, Any]:
    """
    Legacy split-candidate helper implemented on contacts.parquet hyperedges.

    Phase-5 role:
      - candidate generator only
      - current coverage-GMM trigger is deprecated as a primary split trigger
      - not part of the active bin-centric orchestration

    Historical behavior:
      - optional refinement split step with a coverage BIC trigger

    Trigger:
      - If cov is provided, for each bin: compare BIC(1-Gauss) vs BIC(2-GMM) on log1p(cov).
    Partition:
      - Build an induced bipartite graph (contig-contact) inside triggered bins using weighted incidences.
      - Run Leiden modularity partition and keep sub-bins >= MIN_BIN_BP.
    """
    import leidenalg

    meta = {
        "enabled": cov is not None,
        "trigger": {"type": "gmm_bic_1v2_on_log1p_cov", "rule": "trigger if BIC2 < BIC1"},
        "partition": {"type": "leiden_modularity_on_induced_bipartite", "seed": seed},
        "min_subbin_bp": MIN_BIN_BP,
    }

    if cov is None:
        with out_path.open("w", encoding="utf-8", newline="") as fh:
            fh.write("contig_name\told_bin\tnew_bin\taction\treason\n")
        write_json(out_path.with_suffix(".json"), {"meta": meta, "bic": {}, "bins_triggered": []})
        return {
            "bins_triggered": 0,
            "bins_split": 0,
            "contigs_unbinned": set(),
            "meta": meta,
            "bins_triggered_list": [],
        }

    bic_table: dict[str, dict[str, float]] = {}
    bins_triggered: list[str] = []
    for bin_id, contigs in bin_to_contigs.items():
        vals = [cov.get(c, 0.0) for c in contigs if contig_len.get(c, 0) >= min_contig_len]
        if len(vals) < 20:
            continue
        xs = [math.log1p(v) for v in vals]
        bic1 = _bic(xs, k=1)
        bic2 = _bic(xs, k=2)
        bic_table[bin_id] = {"bic1": bic1, "bic2": bic2, "delta": bic1 - bic2}
        if bic2 < bic1:
            bins_triggered.append(bin_id)

    if not bins_triggered:
        with out_path.open("w", encoding="utf-8", newline="") as fh:
            fh.write("contig_name\told_bin\tnew_bin\taction\treason\n")
        write_json(out_path.with_suffix(".json"), {"meta": meta, "bic": bic_table, "bins_triggered": []})
        return {
            "bins_triggered": 0,
            "bins_split": 0,
            "contigs_unbinned": set(),
            "meta": meta,
            "bins_triggered_list": [],
        }

    builders: dict[str, _InducedBuilder] = {}
    for bin_id in bins_triggered:
        contigs = [c for c in bin_to_contigs.get(bin_id, []) if contig_len.get(c, 0) >= min_contig_len]
        if len(contigs) >= 2:
            builders[bin_id] = _InducedBuilder(bin_id, contigs)

    contig_to_trigger_bin: dict[str, str] = {}
    for bin_id, bld in builders.items():
        for c in bld.contigs:
            contig_to_trigger_bin[c] = bin_id

    # Stream contacts.parquet and add within-bin incidences.
    for contigs_raw, p_raw, R in _iter_contacts_parquet(contacts_parquet, batch_size=100_000):
        # filter / normalize
        seen: dict[str, int] = {}
        contigs: list[str] = []
        p: list[float] = []
        for c, w in zip(contigs_raw, p_raw, strict=True):
            if not c:
                continue
            if contig_len.get(c, 0) < min_contig_len:
                continue
            ww = float(w)
            if ww <= 0.0:
                continue
            if c in seen:
                p[seen[c]] += ww
            else:
                seen[c] = len(contigs)
                contigs.append(c)
                p.append(ww)
        if len(contigs) < 2:
            continue
        s = float(sum(p))
        if not (s > 0.0):
            continue
        p = [x / s for x in p]
        k = len(contigs)
        w_contact = order_norm(k, method="pair") * float(R)

        by_bin: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for c, pc in zip(contigs, p, strict=True):
            b = contig_to_trigger_bin.get(c)
            if b is not None:
                by_bin[b].append((c, float(w_contact) * float(pc)))
        for b, items in by_bin.items():
            if len(items) >= 2:
                builders[b].add_contact_weighted(items)

    next_bin_id = _next_bin_id(contig_to_bin.values())
    contigs_unbinned: set[str] = set()
    bins_split = 0

    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("contig_name\told_bin\tnew_bin\taction\treason\n")
        for bin_id, bld in builders.items():
            if bld.n_contacts < 2:
                continue
            g = bld.to_igraph()
            partition = leidenalg.find_partition(
                g,
                leidenalg.ModularityVertexPartition,
                weights="weight",
                seed=seed,
            )
            membership = partition.membership[: bld.n_contigs]
            comm_to_contigs: dict[int, list[str]] = defaultdict(list)
            for c, m in zip(bld.contigs, membership, strict=True):
                comm_to_contigs[int(m)].append(c)

            comm_bp: dict[int, int] = {
                comm: sum(contig_len.get(c, 0) for c in cs) for comm, cs in comm_to_contigs.items()
            }
            kept_comms = [comm for comm, bp in comm_bp.items() if bp >= MIN_BIN_BP]
            if len(kept_comms) < 2:
                for c in bld.contigs:
                    fh.write(f"{c}\t{bin_id}\t\tkept\tno_split\n")
                continue

            bins_split += 1
            new_bins: dict[int, str] = {}
            for comm in sorted(kept_comms, key=lambda x: comm_bp[x], reverse=True):
                new_bins[comm] = str(next_bin_id)
                next_bin_id += 1

            # Remove old bin and replace with new bins.
            for c in list(bin_to_contigs.get(bin_id, [])):
                contig_to_bin.pop(c, None)
            bin_to_contigs.pop(bin_id, None)

            for comm, cs in comm_to_contigs.items():
                if comm in new_bins:
                    nb = new_bins[comm]
                    for c in cs:
                        contig_to_bin[c] = nb
                        bin_to_contigs[nb].append(c)
                        fh.write(f"{c}\t{bin_id}\t{nb}\tsplit\tcommunity\n")
                else:
                    for c in cs:
                        contigs_unbinned.add(c)
                        fh.write(f"{c}\t{bin_id}\t\tunbinned\tsplit_tiny\n")

    write_json(out_path.with_suffix(".json"), {"meta": meta, "bic": bic_table, "bins_triggered": bins_triggered})
    return {
        "bins_triggered": len(bins_triggered),
        "bins_split": bins_split,
        "contigs_unbinned": contigs_unbinned,
        "meta": meta,
        "bins_triggered_list": bins_triggered,
    }

def _decontam(
    *,
    bin_to_contigs: dict[str, list[str]],
    contig_len: dict[str, int],
    contig_to_bin: dict[str, str],
    intra_support: dict[str, float],
    other_support: dict[str, float],
    cov: Optional[dict[str, float]],
    bin_cov_stats: Optional[dict[str, dict[str, float]]],
    out_path: Path,
) -> set[str]:
    """
    Remove contigs with low intra_support using per-bin median/MAD rule.
    Coverage (if provided) is used as a veto: cov outlier + weak support => remove.
    """
    removed: set[str] = set()
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("contig_name\tbin_id\tlength\tintra_support\tthreshold\tcov\tcov_z\treason\n")
        for bin_id, contigs in list(bin_to_contigs.items()):
            vals = [float(intra_support.get(c, 0.0)) for c in contigs]
            if not vals:
                continue
            med = _median(vals)
            mad = _mad(vals, center=med)
            thr = max(0.0, med - 3.0 * mad)

            cov_med = cov_mad = None
            if cov is not None and bin_cov_stats is not None and bin_id in bin_cov_stats:
                cov_med = float(bin_cov_stats[bin_id].get("median", 0.0))
                cov_mad = float(bin_cov_stats[bin_id].get("mad", 0.0))

            keep: list[str] = []
            for c in contigs:
                s_intra = float(intra_support.get(c, 0.0))
                remove = s_intra < thr
                reason = "low_intra_support" if remove else ""

                cov_z = None
                cov_val = cov.get(c) if cov is not None else None
                if cov_val is not None and cov_med is not None and cov_mad is not None:
                    denom = cov_mad if cov_mad > 0 else 1e-12
                    cov_z = abs(float(cov_val) - cov_med) / denom
                    if cov_z > 3.0 and s_intra < med:
                        remove = True
                        reason = reason + "|cov_outlier" if reason else "cov_outlier"

                if remove:
                    removed.add(c)
                    contig_to_bin.pop(c, None)
                    fh.write(
                        f"{c}\t{bin_id}\t{contig_len.get(c, 0)}\t{s_intra:.6g}\t{thr:.6g}\t"
                        f"{'' if cov_val is None else f'{cov_val:.6g}'}\t"
                        f"{'' if cov_z is None else f'{cov_z:.6g}'}\t{reason}\n"
                    )
                else:
                    keep.append(c)
            bin_to_contigs[bin_id] = keep
    return removed


class _InducedBuilder:
    def __init__(self, bin_id: str, contigs: list[str]):
        self.bin_id = bin_id
        self.contigs = contigs
        self.contig_to_idx = {c: i for i, c in enumerate(contigs)}
        self.n_contigs = len(contigs)
        self.edges: list[tuple[int, int]] = []
        self.weights: list[float] = []
        self.n_contacts = 0

    def add_contact(self, contigs_in_bin: list[str], w: float) -> None:
        contact_node = self.n_contigs + self.n_contacts
        self.n_contacts += 1
        for c in contigs_in_bin:
            idx = self.contig_to_idx.get(c)
            if idx is None:
                continue
            self.edges.append((idx, contact_node))
            self.weights.append(w)

    def add_contact_weighted(self, items: list[tuple[str, float]]) -> None:
        """
        Add a contact node where each incidence can have its own weight.

        This is used by the BAM/parquet refine split step, where we build an induced bipartite
        graph inside one coarse bin using weighted incidences:
          edge_weight(c,r) = OrderNorm(k(r)) * q(r) * P_{r,c}
        (see docs/BAM_HYPERGRAPH_SPECTRAL_PIPELINE.md).
        """
        contact_node = self.n_contigs + self.n_contacts
        self.n_contacts += 1
        for c, w in items:
            idx = self.contig_to_idx.get(c)
            if idx is None:
                continue
            ww = float(w)
            if ww <= 0.0:
                continue
            self.edges.append((idx, contact_node))
            self.weights.append(ww)

    def to_igraph(self):
        import igraph as ig

        g = ig.Graph(n=self.n_contigs + self.n_contacts, edges=self.edges, directed=False)
        g.es["weight"] = self.weights
        g.vs["type"] = [False] * self.n_contigs + [True] * self.n_contacts
        return g


def _next_bin_id(existing: Iterable[str]) -> int:
    mx = -1
    for b in existing:
        try:
            mx = max(mx, int(str(b)))
        except Exception:
            continue
    return mx + 1


def _top_candidates(
    scores: dict[str, float],
    *,
    keep_bins: set[str],
    topk: int,
) -> list[tuple[str, float]]:
    items = [(b, float(s)) for b, s in scores.items() if b in keep_bins and s > 0]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:topk]


def _prune_topk(scores: dict[str, float], *, k: int) -> None:
    best = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    scores.clear()
    scores.update(best)


def _auto_threshold_otsu(values: list[float]) -> tuple[float, dict[str, Any]]:
    if not values:
        return 0.0, {"type": "none", "note": "no candidates"}
    xs = [math.log1p(v) for v in values if v > 0]
    if len(xs) < 50:
        return math.expm1(_median(xs)), {"type": "fallback_median_log1p", "n": len(xs)}
    thr_log = _otsu_threshold(xs, bins=256)
    return math.expm1(thr_log), {"type": "otsu_log1p", "bins": 256, "n": len(xs)}


def _otsu_threshold(values: list[float], *, bins: int = 256) -> float:
    vmin = min(values)
    vmax = max(values)
    if vmin == vmax:
        return vmin
    step = (vmax - vmin) / bins
    hist = [0] * bins
    for v in values:
        idx = int((v - vmin) / step)
        if idx >= bins:
            idx = bins - 1
        hist[idx] += 1

    total = sum(hist)
    if total <= 0:
        return vmin
    prob = [h / total for h in hist]

    omega = [0.0] * bins
    mu = [0.0] * bins
    omega[0] = prob[0]
    mu[0] = prob[0] * 0.0
    for i in range(1, bins):
        omega[i] = omega[i - 1] + prob[i]
        mu[i] = mu[i - 1] + prob[i] * i

    mu_t = mu[-1]
    best_i = 0
    best_sigma = -1.0
    for i in range(bins - 1):
        if omega[i] <= 0.0 or omega[i] >= 1.0:
            continue
        mu0 = mu[i] / omega[i]
        mu1 = (mu_t - mu[i]) / (1.0 - omega[i])
        sigma = omega[i] * (1.0 - omega[i]) * (mu0 - mu1) ** 2
        if sigma > best_sigma:
            best_sigma = sigma
            best_i = i
    return vmin + (best_i + 0.5) * step


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return float(xs[mid])
    return 0.5 * (xs[mid - 1] + xs[mid])


def _mad(values: list[float], *, center: Optional[float] = None) -> float:
    if not values:
        return 0.0
    c = float(center) if center is not None else _median(values)
    return _median([abs(x - c) for x in values])


def _bic(xs: list[float], *, k: int) -> float:
    if not xs:
        return 0.0
    if k == 1:
        ll = _loglik_1gauss(xs)
        p = 2
    elif k == 2:
        ll = _loglik_2gmm(xs)
        p = 5
    else:
        raise ValueError("k must be 1 or 2")
    n = len(xs)
    return -2.0 * ll + p * math.log(n)


def _loglik_1gauss(xs: list[float]) -> float:
    n = len(xs)
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / n
    var = max(var, 1e-6)
    return sum(_log_norm_pdf(x, mu, var) for x in xs)


def _loglik_2gmm(xs: list[float]) -> float:
    n = len(xs)
    xs_sorted = sorted(xs)
    mu1 = xs_sorted[n // 4]
    mu2 = xs_sorted[(3 * n) // 4]
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / n
    var1 = var2 = max(var, 1e-3)
    w1 = 0.5

    for _ in range(50):
        r1_sum = 0.0
        r2_sum = 0.0
        mu1_num = 0.0
        mu2_num = 0.0
        var1_num = 0.0
        var2_num = 0.0

        for x in xs:
            l1 = math.log(w1) + _log_norm_pdf(x, mu1, var1)
            l2 = math.log(1.0 - w1) + _log_norm_pdf(x, mu2, var2)
            m = max(l1, l2)
            d = math.exp(l1 - m) + math.exp(l2 - m)
            r1 = math.exp(l1 - m) / d
            r2 = 1.0 - r1
            r1_sum += r1
            r2_sum += r2
            mu1_num += r1 * x
            mu2_num += r2 * x
            # var numerator uses previous mu; update after mu update in next iteration

        if r1_sum <= 1e-9 or r2_sum <= 1e-9:
            break

        w1 = max(1e-3, min(1.0 - 1e-3, r1_sum / n))
        mu1 = mu1_num / r1_sum
        mu2 = mu2_num / r2_sum

        for x in xs:
            l1 = math.log(w1) + _log_norm_pdf(x, mu1, var1)
            l2 = math.log(1.0 - w1) + _log_norm_pdf(x, mu2, var2)
            m = max(l1, l2)
            d = math.exp(l1 - m) + math.exp(l2 - m)
            r1 = math.exp(l1 - m) / d
            r2 = 1.0 - r1
            var1_num += r1 * (x - mu1) ** 2
            var2_num += r2 * (x - mu2) ** 2

        var1 = max(var1_num / r1_sum, 1e-6)
        var2 = max(var2_num / r2_sum, 1e-6)

    ll = 0.0
    for x in xs:
        l1 = math.log(w1) + _log_norm_pdf(x, mu1, var1)
        l2 = math.log(1.0 - w1) + _log_norm_pdf(x, mu2, var2)
        m = max(l1, l2)
        ll += m + math.log(math.exp(l1 - m) + math.exp(l2 - m))
    return ll


def _log_norm_pdf(x: float, mu: float, var: float) -> float:
    return -0.5 * (math.log(2.0 * math.pi * var) + ((x - mu) ** 2) / var)
