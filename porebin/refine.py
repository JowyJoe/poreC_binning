from __future__ import annotations

import csv
import gzip
import logging
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from porebin import __version__
from porebin.build_graph import order_norm
from porebin.export import MIN_BIN_BP
from porebin.pairwise_baseline import parse_contacts_stream
from porebin.utils import (
    dedupe_preserve_order,
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
    reads_skipped_k_lt_2: int = 0
    input_sorted_by_readid: bool = True

    decontam_removed: int = 0
    recruit_assigned: int = 0
    split_triggered: int = 0
    split_applied: int = 0


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
    Refine coarse bins using automatic/default rules (no user thresholds).

    Outputs:
      - out_dir/bins.refined.tsv
      - out_dir/recruit_assign.tsv
      - out_dir/decontam_removed.tsv
      - out_dir/split_map.tsv
      - out_dir/run_refine.json
    """
    logger = logger or logging.getLogger("porebin")
    out_dir = out_dir.resolve()
    ensure_dir(out_dir)

    if not contigs_fasta.exists():
        raise FileNotFoundError(f"Contigs FASTA not found: {contigs_fasta}")
    if not ppl_contacts.exists():
        raise FileNotFoundError(f"PPL contacts not found: {ppl_contacts}")
    if not bins_tsv.exists():
        raise FileNotFoundError(f"Bins TSV not found: {bins_tsv}")
    if bam is not None and not bam.exists():
        raise FileNotFoundError(f"BAM not found: {bam}")

    run_json = out_dir / "run_refine.json"
    record: dict[str, Any] = {
        "porebin_version": __version__,
        "command": "refine",
        "started_at": utc_now_iso(),
        "ended_at": None,
        "status": "running",
        "inputs": {
            "contigs_fasta": str(contigs_fasta),
            "ppl_contacts": str(ppl_contacts),
            "bins_tsv": str(bins_tsv),
            "bam": str(bam) if bam is not None else None,
        },
        "seed": seed,
        "threads": threads,
        "thresholds": {},
        "decisions": {},
        "stats": {},
        "outputs": {},
        "cwd": os.getcwd(),
    }
    write_json(run_json, record)

    stats = RefineStats()
    try:
        contig_len = _read_contig_lengths(contigs_fasta)
        if not contig_len:
            raise RefineError(f"No contigs found in FASTA: {contigs_fasta}")
        stats.contigs_total = len(contig_len)

        min_contig_len, min_contig_len_meta = auto_min_contig_len(contig_len)

        coarse = _read_bins_tsv(bins_tsv)
        stats.bins_total = len(set(coarse.values()))

        # keep_bins is defined the same way as export: total_bp >= 200kb
        coarse_bin_bp, _coarse_bin_n = _bin_sizes(coarse, contig_len, min_contig_len=min_contig_len)
        keep_bins = {b for b, bp in coarse_bin_bp.items() if bp >= MIN_BIN_BP}
        stats.bins_kept_initial = len(keep_bins)

        contig_to_bin: dict[str, str] = {}
        bin_to_contigs: dict[str, list[str]] = defaultdict(list)
        unbinned_reason: dict[str, str] = {}
        for contig, L in contig_len.items():
            if L < min_contig_len:
                stats.contigs_short += 1
                unbinned_reason[contig] = "short_contig"
                continue
            b = coarse.get(contig)
            if b is None:
                unbinned_reason[contig] = "unassigned"
                continue
            if b not in keep_bins:
                unbinned_reason[contig] = "tiny_bin"
                continue
            contig_to_bin[contig] = b
            bin_to_contigs[b].append(contig)
        stats.contigs_assigned_initial = len(contig_to_bin)

        cov: Optional[dict[str, float]] = None
        bin_cov_stats: Optional[dict[str, dict[str, float]]] = None
        coverage_source_meta: Optional[dict[str, Any]] = None
        if bam is not None:
            cov_tsv = bins_tsv.parent / "coverage" / "coverage.tsv"
            if cov_tsv.exists():
                logger.info(f"Refine: using cached coverage TSV: {cov_tsv}")
                cov = _coverage_from_tsv(cov_tsv, contig_len=contig_len, min_contig_len=min_contig_len)
                coverage_source_meta = {"type": "coverage_tsv", "path": str(cov_tsv)}
            else:
                cov = _coverage_from_bam(bam, contig_len, min_contig_len=min_contig_len, logger=logger)
                coverage_source_meta = {"type": "bam_scan", "bam": str(bam)}
            bin_cov_stats = _bin_coverage_stats(bin_to_contigs, cov)

        intra, other, affinity, contact_meta = _scan_contacts_support_and_affinity(
            ppl_contacts=ppl_contacts,
            contig_len=contig_len,
            min_contig_len=min_contig_len,
            contig_to_bin=contig_to_bin,
        )
        stats.reads_total = int(contact_meta["reads_total"])
        stats.reads_kept = int(contact_meta["reads_kept"])
        stats.reads_skipped_k_lt_2 = int(contact_meta["reads_skipped_k_lt_2"])
        stats.input_sorted_by_readid = bool(contact_meta["input_sorted_by_readid"])

        decontam_path = out_dir / "decontam_removed.tsv"
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
        stats.decontam_removed = len(removed)
        for c in removed:
            unbinned_reason[c] = "removed_by_refine"

        # Recompute keep bins after decontam (align with export)
        bin_bp2, _ = _bin_sizes_from_assignment(contig_to_bin, contig_len)
        keep_bins2 = {b for b, bp in bin_bp2.items() if bp >= MIN_BIN_BP}
        _drop_bins_below_min(bin_to_contigs, contig_to_bin, keep_bins2, unbinned_reason)

        recruit_path = out_dir / "recruit_assign.tsv"
        recruit_result = _recruit(
            affinity=affinity,
            contig_len=contig_len,
            min_contig_len=min_contig_len,
            contig_to_bin=contig_to_bin,
            bin_to_contigs=bin_to_contigs,
            keep_bins=keep_bins2,
            cov=cov,
            bin_cov_stats=bin_cov_stats,
            out_path=recruit_path,
        )
        recruited = recruit_result["assigned"]
        stats.recruit_assigned = len(recruited)
        for c in recruited:
            unbinned_reason.pop(c, None)

        split_map = out_dir / "split_map.tsv"
        split_meta = _split_bins(
            ppl_contacts=ppl_contacts,
            contig_len=contig_len,
            min_contig_len=min_contig_len,
            contig_to_bin=contig_to_bin,
            bin_to_contigs=bin_to_contigs,
            cov=cov,
            out_path=split_map,
            seed=seed,
        )
        stats.split_triggered = int(split_meta["bins_triggered"])
        stats.split_applied = int(split_meta["bins_split"])
        for c in split_meta["contigs_unbinned"]:
            unbinned_reason[c] = "removed_by_refine"

        # Final keep bins (align with export)
        final_bp, _ = _bin_sizes_from_assignment(contig_to_bin, contig_len)
        keep_bins_final = {b for b, bp in final_bp.items() if bp >= MIN_BIN_BP}
        stats.bins_kept_final = len(keep_bins_final)

        refined_bins = out_dir / "bins.refined.tsv"
        with refined_bins.open("w", encoding="utf-8", newline="") as fh:
            fh.write("contig_name\tbin_id\n")
            for contig, b in contig_to_bin.items():
                if b in keep_bins_final:
                    fh.write(f"{contig}\t{b}\n")
        stats.contigs_assigned_final = sum(1 for _c, b in contig_to_bin.items() if b in keep_bins_final)

        record["thresholds"] = {
            "MIN_BIN_BP": MIN_BIN_BP,
            "MIN_CONTIG_LEN": min_contig_len,
            "MIN_CONTIG_LEN_method": min_contig_len_meta,
        }
        record["decisions"] = {
            "status_filter": "passed",
            "order_norm": "2/(k*(k-1))  # == 1/C(k,2)",
            "coverage_source": coverage_source_meta,
            "keep_bins_rule": "total_bp >= 200kb",
            "recruit": {
                "topK": 3,
                "threshold": float(recruit_result["threshold"]),
                "threshold_method": recruit_result["threshold_method"],
                "ratio_threshold": 3.0,
                "coverage_veto": {
                    "enabled": bam is not None,
                    "rule": "abs(cov_u - bin_median) <= 3*MAD",
                    "k": 3.0,
                },
            },
            "decontam_rule": "per-bin: remove if intra_support < max(0, median - 3*MAD)",
            "split": {
                **split_meta["meta"],
                "bins_triggered": split_meta.get("bins_triggered_list", []),
                "bic_top": split_meta.get("bic_top", []),
            },
        }
        record["stats"] = stats.__dict__
        record["outputs"] = {
            "bins_refined_tsv": str(refined_bins),
            "recruit_assign_tsv": str(recruit_path),
            "decontam_removed_tsv": str(decontam_path),
            "split_map_tsv": str(split_map),
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


def _scan_contacts_support_and_affinity(
    *,
    ppl_contacts: Path,
    contig_len: dict[str, int],
    min_contig_len: int,
    contig_to_bin: dict[str, str],
) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, float]], dict[str, Any]]:
    """
    Streaming pass over PPL contacts (segment-level TSV, passed-only) grouped by readID.
    Computes:
      - intra_support for contigs already assigned to keep bins
      - other_support (best other-bin support) for the same contigs
      - affinity scores for unbinned contigs towards keep bins
    """
    intra: defaultdict[str, float] = defaultdict(float)
    other: defaultdict[str, float] = defaultdict(float)
    affinity: dict[str, dict[str, float]] = {}

    reads_total = 0
    reads_kept = 0
    reads_skipped = 0
    input_sorted = True

    current_read: Optional[str] = None
    current_contigs: list[str] = []
    prev_read: Optional[str] = None

    def flush() -> None:
        nonlocal reads_kept, reads_skipped
        if current_read is None:
            return
        contigs = dedupe_preserve_order(current_contigs)
        contigs = [c for c in contigs if contig_len.get(c, 0) >= min_contig_len]
        k = len(contigs)
        if k < 2:
            reads_skipped += 1
            return
        reads_kept += 1
        w = order_norm(k, method="pair")

        bin_counts: dict[str, int] = {}
        for c in contigs:
            b = contig_to_bin.get(c)
            if b is None:
                continue
            bin_counts[b] = bin_counts.get(b, 0) + 1
        if not bin_counts:
            return

        best_other: dict[str, int] = {}
        for c in contigs:
            b = contig_to_bin.get(c)
            if b is not None:
                same = max(0, bin_counts.get(b, 0) - 1)
                if same:
                    intra[c] += w * same
                if len(bin_counts) > 1:
                    if b not in best_other:
                        best_other[b] = max((cnt for bb, cnt in bin_counts.items() if bb != b), default=0)
                    if best_other[b]:
                        other[c] += w * best_other[b]
            else:
                top = affinity.get(c)
                if top is None:
                    top = {}
                    affinity[c] = top
                for bb, cnt in bin_counts.items():
                    top[bb] = top.get(bb, 0.0) + w * cnt
                if len(top) > 32:
                    _prune_topk(top, k=3)

    for contig, read_id, status in parse_contacts_stream(ppl_contacts):
        if str(status).strip().lower() != "passed":
            continue
        if prev_read is not None and read_id < prev_read:
            input_sorted = False
        prev_read = read_id

        if current_read is None:
            current_read = read_id
            reads_total += 1
        elif read_id != current_read:
            flush()
            current_read = read_id
            current_contigs = []
            reads_total += 1
        current_contigs.append(contig)
    flush()

    return dict(intra), dict(other), affinity, {
        "reads_total": reads_total,
        "reads_kept": reads_kept,
        "reads_skipped_k_lt_2": reads_skipped,
        "input_sorted_by_readid": input_sorted,
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


def _recruit(
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
    Recruit unbinned contigs to kept bins.
      - keep topK=3 candidates
      - threshold T from Otsu on log1p(top1_score)
      - ratio threshold fixed at 3.0 (recorded in run_refine.json)
      - optional coverage veto: abs(cov_u - bin_median) <= 3*MAD
    """
    top1_scores = []
    for c, scores in affinity.items():
        if c in contig_to_bin or contig_len.get(c, 0) < min_contig_len:
            continue
        best = _top_candidates(scores, keep_bins=keep_bins, topk=1)
        if best:
            top1_scores.append(best[0][1])

    threshold, thr_meta = _auto_threshold_otsu(top1_scores)
    ratio_threshold = 3.0

    assigned: set[str] = set()
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(
            "contig_name\tassigned_bin\ttop1_bin\ttop1_score\ttop2_score\tratio\tthreshold\tcov\tveto_reason\n"
        )
        for c, scores in affinity.items():
            if c in contig_to_bin or contig_len.get(c, 0) < min_contig_len:
                continue
            cand = _top_candidates(scores, keep_bins=keep_bins, topk=3)
            if not cand:
                continue
            b1, s1 = cand[0]
            s2 = cand[1][1] if len(cand) > 1 else 0.0
            ratio = (s1 / s2) if s2 > 0 else float("inf")

            veto_reason = ""
            cov_val = cov.get(c) if cov is not None else None
            if cov_val is not None and bin_cov_stats is not None and b1 in bin_cov_stats:
                bmed = float(bin_cov_stats[b1].get("median", 0.0))
                bmad = float(bin_cov_stats[b1].get("mad", 0.0))
                denom = bmad if bmad > 0 else 1e-12
                if abs(float(cov_val) - bmed) > 3.0 * denom:
                    veto_reason = "cov_veto"

            ok = (s1 >= threshold) and (ratio >= ratio_threshold) and not veto_reason
            fh.write(
                f"{c}\t{b1 if ok else ''}\t{b1}\t{s1:.6g}\t{s2:.6g}\t{ratio:.6g}\t{threshold:.6g}\t"
                f"{'' if cov_val is None else f'{cov_val:.6g}'}\t{veto_reason}\n"
            )
            if ok:
                contig_to_bin[c] = b1
                bin_to_contigs[b1].append(c)
                assigned.add(c)

    return {"assigned": assigned, "threshold": threshold, "threshold_method": thr_meta}


def _split_bins(
    *,
    ppl_contacts: Path,
    contig_len: dict[str, int],
    min_contig_len: int,
    contig_to_bin: dict[str, str],
    bin_to_contigs: dict[str, list[str]],
    cov: Optional[dict[str, float]],
    out_path: Path,
    seed: int,
) -> dict[str, Any]:
    """
    Split bins based on coverage multi-modality (BIC: 1-Gaussian vs 2-GMM on log1p(cov)).
    If triggered, build an induced contig-contact bipartite graph and partition it with Leiden modularity
    (no resolution parameter).

    Rule: apply split only if >=2 sub-bins each >= MIN_BIN_BP; otherwise revoke.
    """
    meta: dict[str, Any] = {"enabled": False}
    if cov is None:
        with out_path.open("w", encoding="utf-8", newline="") as fh:
            fh.write("contig_name\told_bin\tnew_bin\taction\treason\n")
        return {
            "bins_triggered": 0,
            "bins_split": 0,
            "contigs_unbinned": set(),
            "meta": meta,
        }

    try:
        import igraph as ig  # noqa: F401
        import leidenalg
    except Exception as exc:  # pragma: no cover
        raise RefineError(
            "split requires 'python-igraph' and 'leidenalg' (already required by porebin)."
        ) from exc

    meta = {
        "enabled": True,
        "trigger": {"type": "gmm_bic_1v2_on_log1p_cov", "rule": "trigger if BIC2 < BIC1"},
        "partition": {"type": "leiden_modularity", "seed": seed},
        "min_subbin_bp": MIN_BIN_BP,
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
            "bic_top": [],
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

    current_read: Optional[str] = None
    current_contigs: list[str] = []

    def flush_contact() -> None:
        if current_read is None:
            return
        contigs = dedupe_preserve_order(current_contigs)
        contigs = [c for c in contigs if contig_len.get(c, 0) >= min_contig_len]
        if len(contigs) < 2:
            return
        by_bin: dict[str, list[str]] = defaultdict(list)
        for c in contigs:
            b = contig_to_trigger_bin.get(c)
            if b is not None:
                by_bin[b].append(c)
        for b, cs in by_bin.items():
            if len(cs) >= 2:
                builders[b].add_contact(cs, order_norm(len(cs), method="pair"))

    for contig, read_id, status in parse_contacts_stream(ppl_contacts):
        if str(status).strip().lower() != "passed":
            continue
        if current_read is None:
            current_read = read_id
        elif read_id != current_read:
            flush_contact()
            current_read = read_id
            current_contigs = []
        current_contigs.append(contig)
    flush_contact()

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
        "bic_top": [
            {"bin_id": b, **m}
            for b, m in sorted(bic_table.items(), key=lambda kv: kv[1]["delta"], reverse=True)[:50]
        ],
    }


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
