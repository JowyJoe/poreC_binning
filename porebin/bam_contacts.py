from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from porebin import __version__
from porebin.utils import ensure_dir, iter_fasta_records, write_json


class BamContactsError(RuntimeError):
    pass


@dataclass
class BamContactsStats:
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

    # Per-alignment missingness counters (across the whole BAM).
    # These are also recorded per-contact in contacts.parquet.
    mapq_missing_count: int = 0  # MAPQ==255 ("unknown") or missing/invalid MAPQ
    nm_missing_count: int = 0  # NM tag missing
    len_missing_count: int = 0  # alignment length missing (both query len and ref span unavailable)

    # de-overlap (per read+contig) statistics
    overlap_blocks_created_total: int = 0
    overlap_segments_merged_total: int = 0

    k_counter: Counter[int] = None  # set in __post_init__
    weight_min: Optional[float] = None
    weight_max: Optional[float] = None
    weight_sum: float = 0.0
    weight_count: int = 0

    def __post_init__(self) -> None:
        if self.k_counter is None:
            self.k_counter = Counter()

    def add_weight(self, w: float) -> None:
        self.weight_sum += float(w)
        self.weight_count += 1
        self.weight_min = w if self.weight_min is None else min(self.weight_min, w)
        self.weight_max = w if self.weight_max is None else max(self.weight_max, w)

    @property
    def weight_mean(self) -> Optional[float]:
        if self.weight_count <= 0:
            return None
        return self.weight_sum / self.weight_count


def bam_to_contacts_parquet(
    *,
    bam: Path,
    contigs_fasta: Path,
    out_dir: Path,
    parquet_batch_size: int = 10_000,
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    """
    Convert a *queryname-sorted* BAM into our internal hyperedge/contact format (contacts.parquet).

    Mathematical definition (see docs/BAM_HYPERGRAPH_SPECTRAL_PIPELINE.md):
      - Each read r (QNAME) forms one hyperedge/contact.
      - For each alignment a in r, define evidence:
          aligned_len(a) = query_alignment_length (preferred);
                           else: reference_end - reference_start;
                           else: skip this alignment and count len_missing_count.
          p_ok(a) = 1 - 10^(-MAPQ(a)/10) for MAPQ in normal range
                   = 0.5               for MAPQ==255 ("unknown") or missing/invalid MAPQ
          id(a)   = max(0, 1 - NM(a)/aligned_len(a)) if NM tag present
                   = 1                  if NM tag missing (count nm_missing_count)
          e(a)    = p_ok(a) * id(a) * aligned_len(a)
      - Aggregate per contig: E_{r,c} = sum_{a:ref=c} e(a)
      - Normalized evidence share (soft incidence weight):
                               pi_{r,c} = E_{r,c} / sum_{c'} E_{r,c'}
        NOTE: pi_{r,c} are evidence shares, not posterior probabilities.
      - Concentration (QC only): C(r) = sum_c pi_{r,c}^2
      - Effective order (QC only): k_eff(r) = 1/C(r)
      - Read quality weight (no multi-way penalty):
          q(r) = (sum_a p_ok(a)*id(a)*aligned_len(a)) / (sum_a aligned_len(a))
          weight = clamp(q(r), 0, 1)
      - We output:
          contigs = [c...]
          contig_weights = [pi_{r,c}...]
          k = len(contigs)
          k_eff = 1/C(r)
          weight = q(r)
        Downstream graph uses incidence weight:
          edge_weight(c,r) = OrderNorm(k) * q(r) * pi_{r,c}
    """
    logger = logger or logging.getLogger("porebin")
    out_dir = out_dir.resolve()
    bam = bam.resolve()
    contigs_fasta = contigs_fasta.resolve()

    if not bam.exists():
        raise FileNotFoundError(f"BAM not found: {bam}")
    if not contigs_fasta.exists():
        raise FileNotFoundError(f"Contigs FASTA not found: {contigs_fasta}")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover
        raise BamContactsError(
            "bam2contacts requires 'pyarrow' for Parquet output. Install it, e.g. pip install pyarrow."
        ) from exc

    try:
        import pysam
    except Exception as exc:  # pragma: no cover
        raise BamContactsError(
            "bam2contacts requires 'pysam'. Install it (pip install 'porebin[bam]' or conda install pysam)."
        ) from exc

    out_contacts_dir = out_dir / "contacts"
    out_coverage_dir = out_dir / "coverage"
    ensure_dir(out_contacts_dir)
    ensure_dir(out_coverage_dir)

    out_parquet = out_contacts_dir / "contacts.parquet"
    out_qc = out_contacts_dir / "qc_bam2contacts.json"
    out_cov = out_coverage_dir / "coverage.tsv"

    if out_parquet.exists():
        out_parquet.unlink()

    # contig lengths (for coverage and consistency checks)
    contig_len: dict[str, int] = {}
    for name, _header, seq in iter_fasta_records(contigs_fasta):
        contig_len[name] = len(seq)
    if not contig_len:
        raise BamContactsError(f"No contigs found in FASTA: {contigs_fasta}")

    schema = pa.schema(
        [
            ("contact_id", pa.int64()),
            ("contigs", pa.list_(pa.string())),
            ("contig_weights", pa.list_(pa.float64())),  # pi_{r,c} (normalized evidence shares)
            ("k", pa.int32()),
            ("k_eff", pa.float64()),
            ("weight", pa.float64()),  # q(r) in [0,1]
            ("support_count", pa.int32()),
            ("n_segments", pa.int32()),
            ("mapq_min", pa.int32()),
            ("p_ok_mean", pa.float64()),
            ("aligned_len_sum", pa.int64()),
            ("deoverlap_query_union_len_sum", pa.int64()),
            ("nm_sum", pa.int64()),
            ("mapq_missing_count", pa.int32()),
            ("nm_missing_count", pa.int32()),
            ("len_missing_count", pa.int32()),
        ]
    )

    writer = pq.ParquetWriter(str(out_parquet), schema=schema, compression="zstd")
    buffer: dict[str, list[Any]] = {name: [] for name in schema.names}
    stats = BamContactsStats()

    aligned_bases_weighted: defaultdict[str, float] = defaultdict(float)

    def flush_buffer() -> None:
        if not buffer["contact_id"]:
            return
        writer.write_table(pa.table(buffer, schema=schema))
        for k in buffer:
            buffer[k].clear()

    def mapq_to_p_ok(mapq_raw: Any) -> tuple[float, bool, int]:
        """
        Convert MAPQ to p_ok(a) with a missing-value strategy.

        Code-level definition (must match docs/ and tests):
          - If MAPQ is missing/invalid (None, <0, non-int) OR MAPQ==255 ("unknown"):
              p_ok(a) = 0.5, and mapq_missing_count is incremented.
          - Otherwise:
              p_ok(a) = 1 - 10^(-MAPQ/10).

        Returns (p_ok, is_missing, mapq_for_qc), where mapq_for_qc is 255 for missing/unknown.
        """
        try:
            m = int(mapq_raw)
        except Exception:
            return 0.5, True, 255
        if m == 255 or m < 0 or m > 255:
            return 0.5, True, 255
        return 1.0 - (10.0 ** (-m / 10.0)), False, m

    def emit_contact(
        *,
        contact_id: int,
        contigs: list[str],
        p_weights: list[float],
        k: int,
        k_eff: float,
        weight: float,
        n_segments: int,
        mapq_min: int,
        p_ok_mean: float,
        aligned_len_sum: int,
        deoverlap_query_union_len_sum: int,
        nm_sum: int,
        mapq_missing_count: int,
        nm_missing_count: int,
        len_missing_count: int,
    ) -> None:
        buffer["contact_id"].append(int(contact_id))
        buffer["contigs"].append(contigs)
        buffer["contig_weights"].append([float(x) for x in p_weights])
        buffer["k"].append(int(k))
        buffer["k_eff"].append(float(k_eff))
        buffer["weight"].append(float(weight))
        buffer["support_count"].append(1)
        buffer["n_segments"].append(int(n_segments))
        buffer["mapq_min"].append(int(mapq_min))
        buffer["p_ok_mean"].append(float(p_ok_mean))
        buffer["aligned_len_sum"].append(int(aligned_len_sum))
        buffer["deoverlap_query_union_len_sum"].append(int(deoverlap_query_union_len_sum))
        buffer["nm_sum"].append(int(nm_sum))
        buffer["mapq_missing_count"].append(int(mapq_missing_count))
        buffer["nm_missing_count"].append(int(nm_missing_count))
        buffer["len_missing_count"].append(int(len_missing_count))
        if len(buffer["contact_id"]) >= int(parquet_batch_size):
            flush_buffer()

    logger.info("BAM→contacts: reading name-sorted BAM: %s", bam)
    logger.info("BAM→contacts: writing contacts.parquet: %s", out_parquet)

    bam_fh = pysam.AlignmentFile(str(bam), "rb")
    header_sort = (bam_fh.header.to_dict().get("HD") or {}).get("SO")
    if header_sort is not None and str(header_sort).lower() not in {"queryname", "unknown"}:
        logger.warning(
            "BAM header sort order is %r (expected 'queryname'). If this BAM is not name-sorted, results will be wrong.",
            header_sort,
        )

    current_qname: Optional[str] = None
    # NOTE: We only require QNAME *grouping* (all records of a read are contiguous),
    # not a lexicographic monotone order. samtools may advertise "queryname:natural"
    # ordering, which is not compatible with Python's string comparison.
    enforce_lex_monotone = header_sort is None or str(header_sort).lower() not in {"queryname", "unknown"}
    prev_qname: Optional[str] = None if enforce_lex_monotone else None

    # Per-read accumulators
    # We store per-contig segments and perform de-overlap in flush_read() to avoid double-counting
    # query overlaps from split/supplementary alignments (minimap2).
    segments_by_contig: dict[str, list[tuple[int, int, float, float]]] = {}
    seg_count = 0
    mapq_min: Optional[int] = None
    mapq_sum_lenw = 0.0
    q_num = 0.0  # sum_a p_ok(a)*id(a)*aligned_len(a)
    len_sum = 0
    nm_sum = 0
    mapq_missing = 0
    nm_missing = 0
    len_missing = 0

    contact_id = 0

    def flush_read() -> None:
        nonlocal contact_id
        nonlocal segments_by_contig, seg_count, mapq_min, mapq_sum_lenw, q_num, len_sum, nm_sum
        nonlocal mapq_missing, nm_missing, len_missing

        if current_qname is None:
            return

        stats.reads_total += 1

        if not segments_by_contig:
            stats.reads_skipped_no_evidence += 1
            segments_by_contig = {}
            seg_count = 0
            mapq_min = None
            mapq_sum_lenw = 0.0
            q_num = 0.0
            len_sum = 0
            nm_sum = 0
            mapq_missing = 0
            nm_missing = 0
            len_missing = 0
            return

        # De-overlap within each contig group using query coordinate blocks. Only merge if overlap > 0
        # (do NOT merge merely adjacent segments) to avoid over-merging.
        E_rc: dict[str, float] = {}
        deoverlap_union_len_sum = 0
        blocks_created = 0
        segments_merged = 0

        for contig, segs in segments_by_contig.items():
            if not segs:
                continue
            segs_sorted = sorted(segs, key=lambda s: (s[0], -s[1]))  # qstart asc, qend desc
            num_original = len(segs_sorted)

            blocks: list[tuple[int, int, list[tuple[int, int, float, float]]]] = []
            b_start: Optional[int] = None
            b_end: Optional[int] = None
            b_segs: list[tuple[int, int, float, float]] = []

            def close_block() -> None:
                nonlocal b_start, b_end, b_segs
                if b_start is None or b_end is None:
                    return
                blocks.append((int(b_start), int(b_end), b_segs))
                b_start = None
                b_end = None
                b_segs = []

            for qstart, qend, p_ok, id_est in segs_sorted:
                if b_start is None:
                    b_start = int(qstart)
                    b_end = int(qend)
                    b_segs = [(qstart, qend, p_ok, id_est)]
                    continue

                # Merge only if overlap > 0 (not merely adjacent).
                assert b_end is not None
                if int(qstart) < int(b_end) and int(qend) > int(b_start):
                    b_end = max(int(b_end), int(qend))
                    b_segs.append((qstart, qend, p_ok, id_est))
                else:
                    close_block()
                    b_start = int(qstart)
                    b_end = int(qend)
                    b_segs = [(qstart, qend, p_ok, id_est)]

            close_block()

            num_blocks = len(blocks)
            blocks_created += num_blocks
            if num_original > num_blocks:
                segments_merged += (num_original - num_blocks)

            e_sum = 0.0
            for bs, be, seg_list in blocks:
                L_block = int(be) - int(bs)
                if L_block <= 0:
                    continue
                denom = 0.0
                p_num = 0.0
                id_num = 0.0
                for qs, qe, pp, ii in seg_list:
                    cov_len = max(0, min(int(qe), int(be)) - max(int(qs), int(bs)))
                    if cov_len <= 0:
                        continue
                    denom += float(cov_len)
                    p_num += float(cov_len) * float(pp)
                    id_num += float(cov_len) * float(ii)
                if denom <= 0:
                    continue
                p_ok_block = float(p_num / denom)
                id_block = float(id_num / denom)
                e_sum += float(p_ok_block) * float(id_block) * float(L_block)
                deoverlap_union_len_sum += int(L_block)

            if e_sum > 0.0:
                E_rc[str(contig)] = float(e_sum)

        # Normalize to P_{r,c}
        total_e = float(sum(E_rc.values()))
        if not (total_e > 0.0):
            stats.reads_skipped_no_evidence += 1
            segments_by_contig = {}
            seg_count = 0
            mapq_min = None
            mapq_sum_lenw = 0.0
            q_num = 0.0
            len_sum = 0
            nm_sum = 0
            mapq_missing = 0
            nm_missing = 0
            len_missing = 0
            return

        contigs = list(E_rc.keys())
        p = [float(E_rc[c]) / total_e for c in contigs]

        # hard order
        k = len(contigs)
        if k < 2:
            # Keep the record in contacts.parquet for audit/debugging, but note it's not a usable
            # hyperedge for graph building (build_graph skips k<2).
            stats.reads_skipped_k_lt_2 += 1

        # Effective order and concentration
        c_simpson = float(sum(x * x for x in p))
        if c_simpson <= 0.0:
            stats.reads_skipped_no_evidence += 1
            segments_by_contig = {}
            seg_count = 0
            mapq_min = None
            mapq_sum_lenw = 0.0
            q_num = 0.0
            len_sum = 0
            nm_sum = 0
            mapq_missing = 0
            nm_missing = 0
            len_missing = 0
            return

        k_eff = 1.0 / c_simpson
        p_ok_mean = (mapq_sum_lenw / float(len_sum)) if len_sum > 0 else 0.0

        # Read quality weight q(r) (no concentration penalty; multi-way is signal).
        q = (q_num / float(len_sum)) if len_sum > 0 else 0.0
        weight = float(max(0.0, min(1.0, q)))

        # Deterministic order: sort by descending P, tie-break by contig name.
        order = sorted(range(k), key=lambda i: (-p[i], contigs[i]))
        contigs = [contigs[i] for i in order]
        p = [p[i] for i in order]

        emit_contact(
            contact_id=contact_id,
            contigs=contigs,
            p_weights=p,
            k=k,
            k_eff=k_eff,
            weight=weight,
            n_segments=seg_count,
            mapq_min=int(mapq_min) if mapq_min is not None else 255,
            p_ok_mean=float(p_ok_mean),
            aligned_len_sum=int(len_sum),
            deoverlap_query_union_len_sum=int(deoverlap_union_len_sum),
            nm_sum=int(nm_sum),
            mapq_missing_count=int(mapq_missing),
            nm_missing_count=int(nm_missing),
            len_missing_count=int(len_missing),
        )

        stats.reads_kept += 1
        stats.k_counter[k] += 1
        stats.add_weight(weight)
        stats.overlap_blocks_created_total += int(blocks_created)
        stats.overlap_segments_merged_total += int(segments_merged)
        contact_id += 1

        segments_by_contig = {}
        seg_count = 0
        mapq_min = None
        mapq_sum_lenw = 0.0
        q_num = 0.0
        len_sum = 0
        nm_sum = 0
        mapq_missing = 0
        nm_missing = 0
        len_missing = 0

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
            # minimap2: supplementary (0x800) encodes split/chimeric alignment segments; for Pore-C this
            # is required to preserve multi-way contacts, so we keep these records.
            stats.alignments_supplementary_used += 1

        qname = aln.query_name
        if not qname:
            continue

        # Sortedness sanity check:
        # - If header says SO:queryname (possibly SS:queryname:natural), we trust it and do NOT
        #   enforce Python-lexicographic monotonicity.
        # - If header is missing/odd, fall back to a strict monotone check to catch obvious mis-sorts.
        if enforce_lex_monotone:
            if prev_qname is not None and qname < prev_qname:
                raise BamContactsError(
                    "BAM must be queryname-grouped (samtools sort -n) so we can stream-group alignments by QNAME.\n"
                    f"Detected non-monotonic QNAME (lex order): {qname!r} < {prev_qname!r}.\n"
                    "If your BAM header reports SO:queryname with SS:queryname:natural, update porebin to a version "
                    "that supports natural queryname ordering, or re-sort with samtools sort -n."
                )
            prev_qname = qname

        if current_qname is None:
            current_qname = qname
        elif qname != current_qname:
            flush_read()
            current_qname = qname

        ref = aln.reference_name
        if not ref or ref not in contig_len:
            stats.alignments_skipped_missing_ref += 1
            continue

        # aligned length ℓ(a) (bp)
        aln_len = int(aln.query_alignment_length or 0)
        if aln_len <= 0 and aln.reference_start is not None and aln.reference_end is not None:
            aln_len = int(aln.reference_end - aln.reference_start)
        if aln_len <= 0:
            stats.alignments_skipped_len_missing += 1
            stats.len_missing_count += 1
            len_missing += 1
            continue

        # query interval [qstart, qend) is required for de-overlap
        qstart_raw = getattr(aln, "query_alignment_start", None)
        qend_raw = getattr(aln, "query_alignment_end", None)
        if qstart_raw is None or qend_raw is None:
            stats.alignments_skipped_len_missing += 1
            stats.len_missing_count += 1
            len_missing += 1
            continue
        qstart = int(qstart_raw)
        qend = int(qend_raw)
        if qend <= qstart:
            stats.alignments_skipped_len_missing += 1
            stats.len_missing_count += 1
            len_missing += 1
            continue

        # MAPQ -> p_ok(a)
        p_ok, mapq_is_missing, mapq_for_qc = mapq_to_p_ok(getattr(aln, "mapping_quality", None))
        if mapq_is_missing:
            stats.mapq_missing_count += 1
            mapq_missing += 1

        # NM -> id(a)
        nm_val: Optional[int] = None
        try:
            nm_val = int(aln.get_tag("NM"))
        except Exception:
            nm_val = None

        if nm_val is None:
            stats.nm_missing_count += 1
            nm_missing += 1
            id_est = 1.0
            nm_for_sum = 0
        else:
            id_est = max(0.0, 1.0 - (float(nm_val) / float(aln_len)))
            nm_for_sum = int(nm_val)

        # evidence e(a) and read-quality numerator
        e = float(p_ok * id_est * float(aln_len))
        # NOTE: E_{r,c} is computed in flush_read() after de-overlap; here we only accumulate
        # the read-level quality numerator/denominator and store segments for later merging.
        q_num += float(e)

        stats.alignments_kept += 1
        seg_count += 1
        mapq_min = mapq_for_qc if mapq_min is None else min(mapq_min, mapq_for_qc)
        mapq_sum_lenw += float(p_ok) * float(aln_len)
        len_sum += int(aln_len)
        nm_sum += int(nm_for_sum)

        segments_by_contig.setdefault(str(ref), []).append((qstart, qend, float(p_ok), float(id_est)))

        # coverage accumulation (continuous, no hard MAPQ threshold)
        aligned_bases_weighted[ref] += float(p_ok) * float(aln_len)

    flush_read()
    flush_buffer()
    writer.close()
    bam_fh.close()

    # Coverage TSV
    with out_cov.open("w", encoding="utf-8", newline="") as fh:
        fh.write("contig_name\tcoverage\n")
        for name, L in contig_len.items():
            if L <= 0:
                continue
            cov = float(aligned_bases_weighted.get(name, 0.0)) / float(L)
            fh.write(f"{name}\t{cov:.12g}\n")

    qc = {
        "porebin_version": __version__,
        "input_bam": str(bam),
        "input_contigs_fasta": str(contigs_fasta),
        "out_contacts_parquet": str(out_parquet),
        "out_coverage_tsv": str(out_cov),
        "stats": {
            "reads_total": stats.reads_total,
            "reads_kept": stats.reads_kept,
            "reads_skipped_k_lt_2": stats.reads_skipped_k_lt_2,
            "reads_skipped_no_evidence": stats.reads_skipped_no_evidence,
            "alignments_total": stats.alignments_total,
            "alignments_kept": stats.alignments_kept,
            "alignments_skipped_unmapped": stats.alignments_skipped_unmapped,
            "alignments_skipped_secondary": stats.alignments_skipped_secondary,
            "alignments_skipped_missing_ref": stats.alignments_skipped_missing_ref,
            "alignments_skipped_len_missing": stats.alignments_skipped_len_missing,
            "alignments_supplementary_used": stats.alignments_supplementary_used,
            "mapq_missing_count": stats.mapq_missing_count,
            "nm_missing_count": stats.nm_missing_count,
            "len_missing_count": stats.len_missing_count,
            # UX-friendly aliases (requested; keep schema/backwards-compat fields above too):
            "unmapped_skipped": stats.alignments_skipped_unmapped,
            "secondary_skipped": stats.alignments_skipped_secondary,
            "supplementary_used": stats.alignments_supplementary_used,
            "deoverlap_enabled": True,
            "overlap_blocks_created_total": stats.overlap_blocks_created_total,
            "overlap_segments_merged_total": stats.overlap_segments_merged_total,
            "k_distribution": {str(k): v for k, v in stats.k_counter.items()},
            "weight_summary": {
                "count": stats.weight_count,
                "min": stats.weight_min,
                "max": stats.weight_max,
                "mean": stats.weight_mean,
            },
        },
        "notes": {
            "sortedness": "Streaming grouping requires queryname-sorted BAM (samtools sort -n).",
            "flags": "Skip unmapped (0x4), skip secondary (0x100), keep primary+supplementary (0x800) to preserve Pore-C multi-way contacts.",
            "mapq": "MAPQ converted to p_ok=1-10^(-MAPQ/10). MAPQ==255 or missing/invalid treated as unknown -> p_ok=0.5 (counted).",
            "nm": "If NM tag is missing: id=1 and nm_missing_count is incremented.",
            "coverage": "coverage bases are accumulated as sum(p_ok * aligned_len) per contig.",
            "weight": "weight=q(r)=sum(p_ok*id*len)/sum(len), clamped to [0,1] (no concentration penalty).",
            "deoverlap": {
                "enabled": True,
                "scope": "within each read+contig",
                "merge_rule": "merge segments only if query-interval overlap > 0; do not merge adjacent",
                "block_stats": "p_ok/id are cov_len-weighted within each block; ℓ_block is query union length",
            },
        },
    }
    write_json(out_qc, qc)

    logger.info(
        "BAM→contacts done: reads_kept=%s/%s contacts=%s coverage=%s",
        f"{stats.reads_kept:,}",
        f"{stats.reads_total:,}",
        out_parquet,
        out_cov,
    )

    logger.info(
        "BAM flags: alignments_skipped_unmapped=%s alignments_skipped_secondary=%s alignments_supplementary_used=%s",
        f"{stats.alignments_skipped_unmapped:,}",
        f"{stats.alignments_skipped_secondary:,}",
        f"{stats.alignments_supplementary_used:,}",
    )

    return {"contacts_parquet": out_parquet, "coverage_tsv": out_cov, "qc_json": out_qc, "stats": qc["stats"]}
