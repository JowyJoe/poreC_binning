"""Low-risk BAM -> canonical contact evidence conversion for the new tool."""

from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from porebin_genome.io.fasta import iter_fasta_records
from porebin_genome.io.runtime import ensure_dir, write_json


class BamEvidenceError(RuntimeError):
    """Raised when BAM evidence construction fails."""


@dataclass
class BamEvidenceStats:
    """Execution statistics for BAM-derived contact evidence."""

    reads_total: int = 0
    reads_kept: int = 0
    reads_skipped_k_lt_2: int = 0
    reads_skipped_no_evidence: int = 0

    alignments_total: int = 0
    alignments_kept: int = 0
    alignments_skipped_unmapped: int = 0
    alignments_skipped_secondary: int = 0
    alignments_supplementary_used: int = 0
    alignments_skipped_missing_ref: int = 0
    alignments_skipped_len_missing: int = 0

    mapq_missing_count: int = 0
    nm_missing_count: int = 0
    len_missing_count: int = 0

    overlap_blocks_created_total: int = 0
    overlap_segments_merged_total: int = 0

    k_counter: Counter[int] = field(default_factory=Counter)


def bam_to_contact_evidence(
    *,
    bam: Path,
    contigs_fasta: Path,
    out_dir: Path,
    parquet_batch_size: int = 10_000,
    write_internal_coverage: bool = True,
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    """Convert a queryname-sorted BAM into canonical contact evidence."""
    logger = logger or logging.getLogger("porebin_genome")
    bam = bam.resolve()
    contigs_fasta = contigs_fasta.resolve()
    out_dir = out_dir.resolve()

    if not bam.exists():
        raise FileNotFoundError(f"BAM not found: {bam}")
    if not contigs_fasta.exists():
        raise FileNotFoundError(f"Contigs FASTA not found: {contigs_fasta}")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover
        raise BamEvidenceError("BAM evidence construction requires pyarrow.") from exc

    try:
        import pysam
    except Exception as exc:  # pragma: no cover
        raise BamEvidenceError("BAM evidence construction requires pysam.") from exc

    evidence_dir = out_dir / "evidence"
    ensure_dir(evidence_dir)
    out_parquet = evidence_dir / "contacts.parquet"
    out_coverage = evidence_dir / "coverage.tsv"
    out_qc = evidence_dir / "evidence_qc.json"

    if out_parquet.exists():
        out_parquet.unlink()

    contig_len: dict[str, int] = {}
    for name, _header, seq in iter_fasta_records(contigs_fasta):
        contig_len[name] = len(seq)
    if not contig_len:
        raise BamEvidenceError(f"No contigs found in FASTA: {contigs_fasta}")

    schema = pa.schema(
        [
            ("contact_id", pa.int64()),
            ("contigs", pa.list_(pa.string())),
            ("contig_weights", pa.list_(pa.float64())),
            ("k", pa.int32()),
            ("k_eff", pa.float64()),
            ("weight", pa.float64()),
            ("n_segments", pa.int32()),
            ("aligned_len_sum", pa.int64()),
            ("deoverlap_query_union_len_sum", pa.int64()),
            ("mapq_missing_count", pa.int32()),
            ("nm_missing_count", pa.int32()),
        ]
    )
    writer = pq.ParquetWriter(str(out_parquet), schema=schema, compression="zstd")
    buffer: dict[str, list[Any]] = {name: [] for name in schema.names}
    stats = BamEvidenceStats()
    aligned_bases_weighted: defaultdict[str, float] = defaultdict(float)

    def flush_buffer() -> None:
        if not buffer["contact_id"]:
            return
        writer.write_table(pa.table(buffer, schema=schema))
        for name in schema.names:
            buffer[name].clear()

    def emit_contact(
        *,
        contact_id: int,
        contigs: list[str],
        contig_weights: list[float],
        k: int,
        k_eff: float,
        weight: float,
        n_segments: int,
        aligned_len_sum: int,
        deoverlap_query_union_len_sum: int,
        mapq_missing_count: int,
        nm_missing_count: int,
    ) -> None:
        buffer["contact_id"].append(int(contact_id))
        buffer["contigs"].append(list(contigs))
        buffer["contig_weights"].append([float(x) for x in contig_weights])
        buffer["k"].append(int(k))
        buffer["k_eff"].append(float(k_eff))
        buffer["weight"].append(float(weight))
        buffer["n_segments"].append(int(n_segments))
        buffer["aligned_len_sum"].append(int(aligned_len_sum))
        buffer["deoverlap_query_union_len_sum"].append(int(deoverlap_query_union_len_sum))
        buffer["mapq_missing_count"].append(int(mapq_missing_count))
        buffer["nm_missing_count"].append(int(nm_missing_count))
        if len(buffer["contact_id"]) >= int(parquet_batch_size):
            flush_buffer()

    def mapq_to_p_ok(mapq_raw: Any) -> tuple[float, bool]:
        try:
            value = int(mapq_raw)
        except Exception:
            return 0.5, True
        if value == 255 or value < 0 or value > 255:
            return 0.5, True
        return 1.0 - (10.0 ** (-value / 10.0)), False

    bam_fh = pysam.AlignmentFile(str(bam), "rb")
    header_sort = (bam_fh.header.to_dict().get("HD") or {}).get("SO")
    if header_sort is not None and str(header_sort).lower() not in {"queryname", "unknown"}:
        logger.warning("BAM header sort order is %r; queryname grouping is expected.", header_sort)

    current_qname: Optional[str] = None
    enforce_lex_monotone = header_sort is None or str(header_sort).lower() not in {"queryname", "unknown"}
    prev_qname: Optional[str] = None

    segments_by_contig: dict[str, list[tuple[int, int, float, float]]] = {}
    seg_count = 0
    q_num = 0.0
    len_sum = 0
    mapq_missing = 0
    nm_missing = 0
    contact_id = 0

    def flush_read() -> None:
        nonlocal segments_by_contig, seg_count, q_num, len_sum, mapq_missing, nm_missing, contact_id
        if current_qname is None:
            return

        stats.reads_total += 1
        if not segments_by_contig:
            stats.reads_skipped_no_evidence += 1
            segments_by_contig = {}
            seg_count = 0
            q_num = 0.0
            len_sum = 0
            mapq_missing = 0
            nm_missing = 0
            return

        evidence_by_contig: dict[str, float] = {}
        deoverlap_union_len_sum = 0
        blocks_created = 0
        segments_merged = 0
        for contig_name, segs in segments_by_contig.items():
            if not segs:
                continue
            segs_sorted = sorted(segs, key=lambda item: (item[0], -item[1]))
            num_original = len(segs_sorted)
            blocks: list[tuple[int, int, list[tuple[int, int, float, float]]]] = []
            block_start: Optional[int] = None
            block_end: Optional[int] = None
            block_segments: list[tuple[int, int, float, float]] = []

            def close_block() -> None:
                nonlocal block_start, block_end, block_segments
                if block_start is None or block_end is None:
                    return
                blocks.append((int(block_start), int(block_end), block_segments))
                block_start = None
                block_end = None
                block_segments = []

            for qstart, qend, p_ok, id_est in segs_sorted:
                if block_start is None:
                    block_start = int(qstart)
                    block_end = int(qend)
                    block_segments = [(qstart, qend, p_ok, id_est)]
                    continue
                assert block_end is not None
                if int(qstart) < int(block_end) and int(qend) > int(block_start):
                    block_end = max(int(block_end), int(qend))
                    block_segments.append((qstart, qend, p_ok, id_est))
                else:
                    close_block()
                    block_start = int(qstart)
                    block_end = int(qend)
                    block_segments = [(qstart, qend, p_ok, id_est)]
            close_block()

            num_blocks = len(blocks)
            blocks_created += num_blocks
            if num_original > num_blocks:
                segments_merged += num_original - num_blocks

            evidence_sum = 0.0
            for start, end, block_items in blocks:
                block_len = int(end) - int(start)
                if block_len <= 0:
                    continue
                denom = 0.0
                p_num = 0.0
                id_num = 0.0
                for qstart, qend, p_ok, id_est in block_items:
                    overlap = max(0, min(int(qend), int(end)) - max(int(qstart), int(start)))
                    if overlap <= 0:
                        continue
                    denom += float(overlap)
                    p_num += float(overlap) * float(p_ok)
                    id_num += float(overlap) * float(id_est)
                if denom <= 0.0:
                    continue
                block_p_ok = p_num / denom
                block_id = id_num / denom
                evidence_sum += float(block_p_ok) * float(block_id) * float(block_len)
                deoverlap_union_len_sum += int(block_len)
            if evidence_sum > 0.0:
                evidence_by_contig[contig_name] = float(evidence_sum)

        total_evidence = float(sum(evidence_by_contig.values()))
        if not (total_evidence > 0.0):
            stats.reads_skipped_no_evidence += 1
            segments_by_contig = {}
            seg_count = 0
            q_num = 0.0
            len_sum = 0
            mapq_missing = 0
            nm_missing = 0
            return

        contigs = list(evidence_by_contig.keys())
        contig_weights = [float(evidence_by_contig[name]) / total_evidence for name in contigs]
        k = len(contigs)
        if k < 2:
            stats.reads_skipped_k_lt_2 += 1

        concentration = float(sum(value * value for value in contig_weights))
        if concentration <= 0.0:
            stats.reads_skipped_no_evidence += 1
            segments_by_contig = {}
            seg_count = 0
            q_num = 0.0
            len_sum = 0
            mapq_missing = 0
            nm_missing = 0
            return

        k_eff = 1.0 / concentration
        q_read = (q_num / float(len_sum)) if len_sum > 0 else 0.0
        weight = float(max(0.0, min(1.0, q_read)))

        order = sorted(range(k), key=lambda idx: (-contig_weights[idx], contigs[idx]))
        ordered_contigs = [contigs[idx] for idx in order]
        ordered_weights = [contig_weights[idx] for idx in order]

        emit_contact(
            contact_id=contact_id,
            contigs=ordered_contigs,
            contig_weights=ordered_weights,
            k=k,
            k_eff=k_eff,
            weight=weight,
            n_segments=seg_count,
            aligned_len_sum=len_sum,
            deoverlap_query_union_len_sum=deoverlap_union_len_sum,
            mapq_missing_count=mapq_missing,
            nm_missing_count=nm_missing,
        )
        stats.reads_kept += 1
        stats.k_counter[k] += 1
        stats.overlap_blocks_created_total += int(blocks_created)
        stats.overlap_segments_merged_total += int(segments_merged)
        contact_id += 1

        segments_by_contig = {}
        seg_count = 0
        q_num = 0.0
        len_sum = 0
        mapq_missing = 0
        nm_missing = 0

    for aln in bam_fh.fetch(until_eof=True):
        stats.alignments_total += 1

        flag = int(getattr(aln, "flag", 0) or 0)
        is_unmapped = bool(getattr(aln, "is_unmapped", False)) or ((flag & 0x4) != 0)
        if is_unmapped:
            stats.alignments_skipped_unmapped += 1
            continue
        is_secondary = bool(getattr(aln, "is_secondary", False)) or ((flag & 0x100) != 0)
        if is_secondary:
            stats.alignments_skipped_secondary += 1
            continue
        is_supplementary = bool(getattr(aln, "is_supplementary", False)) or ((flag & 0x800) != 0)
        if is_supplementary:
            stats.alignments_supplementary_used += 1

        qname = getattr(aln, "query_name", None)
        if not qname:
            continue

        if enforce_lex_monotone:
            if prev_qname is not None and qname < prev_qname:
                raise BamEvidenceError(
                    "BAM must be queryname-grouped (samtools sort -n) for streaming evidence construction."
                )
            prev_qname = qname

        if current_qname is None:
            current_qname = qname
        elif qname != current_qname:
            flush_read()
            current_qname = qname

        ref_name = getattr(aln, "reference_name", None)
        if not ref_name or ref_name not in contig_len:
            stats.alignments_skipped_missing_ref += 1
            continue

        aln_len = int(aln.query_alignment_length or 0)
        if aln_len <= 0 and aln.reference_start is not None and aln.reference_end is not None:
            aln_len = int(aln.reference_end - aln.reference_start)
        if aln_len <= 0:
            stats.alignments_skipped_len_missing += 1
            stats.len_missing_count += 1
            continue

        qstart_raw = getattr(aln, "query_alignment_start", None)
        qend_raw = getattr(aln, "query_alignment_end", None)
        if qstart_raw is None or qend_raw is None:
            stats.alignments_skipped_len_missing += 1
            stats.len_missing_count += 1
            continue
        qstart = int(qstart_raw)
        qend = int(qend_raw)
        if qend <= qstart:
            stats.alignments_skipped_len_missing += 1
            stats.len_missing_count += 1
            continue

        p_ok, mapq_is_missing = mapq_to_p_ok(getattr(aln, "mapping_quality", None))
        if mapq_is_missing:
            stats.mapq_missing_count += 1
            mapq_missing += 1

        try:
            nm_value = int(aln.get_tag("NM"))
        except Exception:
            nm_value = None
        if nm_value is None:
            stats.nm_missing_count += 1
            nm_missing += 1
            id_est = 1.0
        else:
            id_est = max(0.0, 1.0 - (float(nm_value) / float(aln_len)))

        evidence = float(p_ok) * float(id_est) * float(aln_len)
        q_num += evidence
        len_sum += int(aln_len)
        stats.alignments_kept += 1
        seg_count += 1
        segments_by_contig.setdefault(str(ref_name), []).append((qstart, qend, float(p_ok), float(id_est)))
        aligned_bases_weighted[str(ref_name)] += float(p_ok) * float(aln_len)

    flush_read()
    flush_buffer()
    writer.close()
    bam_fh.close()

    coverage_written = False
    if write_internal_coverage:
        with out_coverage.open("w", encoding="utf-8", newline="") as fh:
            fh.write("contig_name\tcoverage\n")
            for contig_name, length in contig_len.items():
                if length <= 0:
                    continue
                coverage = float(aligned_bases_weighted.get(contig_name, 0.0)) / float(length)
                fh.write(f"{contig_name}\t{coverage:.12g}\n")
        coverage_written = True

    qc = {
        "tool": "porebin_genome",
        "inputs": {"bam": str(bam), "contigs_fasta": str(contigs_fasta)},
        "outputs": {
            "contacts_parquet": str(out_parquet),
            "coverage_tsv": str(out_coverage) if coverage_written else None,
        },
        "stats": {
            "reads_total": stats.reads_total,
            "reads_kept": stats.reads_kept,
            "reads_skipped_k_lt_2": stats.reads_skipped_k_lt_2,
            "reads_skipped_no_evidence": stats.reads_skipped_no_evidence,
            "alignments_total": stats.alignments_total,
            "alignments_kept": stats.alignments_kept,
            "alignments_skipped_unmapped": stats.alignments_skipped_unmapped,
            "alignments_skipped_secondary": stats.alignments_skipped_secondary,
            "alignments_supplementary_used": stats.alignments_supplementary_used,
            "alignments_skipped_missing_ref": stats.alignments_skipped_missing_ref,
            "alignments_skipped_len_missing": stats.alignments_skipped_len_missing,
            "mapq_missing_count": stats.mapq_missing_count,
            "nm_missing_count": stats.nm_missing_count,
            "len_missing_count": stats.len_missing_count,
            "overlap_blocks_created_total": stats.overlap_blocks_created_total,
            "overlap_segments_merged_total": stats.overlap_segments_merged_total,
            "k_distribution": {str(k): count for k, count in stats.k_counter.items()},
        },
        "notes": {
            "bam_entry_role": "evidence_construction_only",
            "contact_semantics": "one_read_name_equals_one_contact",
            "contig_weights_semantics": "normalized_evidence_shares",
            "weight_semantics": "read_quality_weight_without_pairwise_penalty",
            "internal_coverage_semantics": (
                "MAPQ-weighted aligned bases divided by contig length"
                if coverage_written
                else "not written"
            ),
        },
    }
    write_json(out_qc, qc)

    return {
        "contacts_parquet": out_parquet,
        "coverage_tsv": out_coverage if coverage_written else None,
        "qc_json": out_qc,
        "stats": qc["stats"],
    }
