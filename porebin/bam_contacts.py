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
    alignments_skipped_missing_ref: int = 0

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
          p_ok(a) = 1 - 10^(-MAPQ(a)/10)   (MAPQ=255 treated as unknown -> 0)
          id(a)   = max(0, 1 - NM(a)/aligned_len(a))   (if NM missing: id(a)=1)
          e(a)    = p_ok(a) * id(a) * aligned_len(a)
      - Aggregate per contig: E_{r,c} = sum_{a:ref=c} e(a)
      - Soft assignment:        P_{r,c} = E_{r,c} / sum_{c'} E_{r,c'}
      - Concentration:          C(r) = sum_c P_{r,c}^2
      - Read reliability:       R(r) = mean_len_weighted(p_ok(a)) * C(r)
      - We output:
          contigs = [c...]
          contig_weights = [P_{r,c}...]
          k = len(contigs)
          weight = R(r)
        Downstream graph uses incidence weight:
          edge_weight(c,r) = OrderNorm(k) * weight * contig_weight
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
            ("contig_weights", pa.list_(pa.float64())),  # P_{r,c}
            ("k", pa.int32()),
            ("k_eff", pa.float64()),
            ("weight", pa.float64()),  # R(r) in (0,1]
            ("support_count", pa.int32()),
            ("n_segments", pa.int32()),
            ("mapq_min", pa.int32()),
            ("p_ok_mean", pa.float64()),
            ("aligned_len_sum", pa.int64()),
            ("nm_sum", pa.int64()),
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

    def p_ok_from_mapq(mapq: int) -> float:
        # MAPQ=255 means "mapping quality not available" (SAM spec) -> treat as unknown/low confidence.
        m = int(mapq)
        if m < 0:
            m = 0
        if m >= 255:
            m = 0
        return 1.0 - (10.0 ** (-m / 10.0))

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
        nm_sum: int,
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
        buffer["nm_sum"].append(int(nm_sum))
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
    prev_qname: Optional[str] = None

    # Per-read accumulators
    E_rc: dict[str, float] = {}
    seg_count = 0
    mapq_min: Optional[int] = None
    mapq_sum_lenw = 0.0
    len_sum = 0
    nm_sum = 0

    contact_id = 0

    def flush_read() -> None:
        nonlocal contact_id
        nonlocal E_rc, seg_count, mapq_min, mapq_sum_lenw, len_sum, nm_sum

        if current_qname is None:
            return

        stats.reads_total += 1

        if not E_rc:
            stats.reads_skipped_no_evidence += 1
            E_rc = {}
            seg_count = 0
            mapq_min = None
            mapq_sum_lenw = 0.0
            len_sum = 0
            nm_sum = 0
            return

        # Normalize to P_{r,c}
        total_e = float(sum(E_rc.values()))
        if not (total_e > 0.0):
            stats.reads_skipped_no_evidence += 1
            E_rc = {}
            seg_count = 0
            mapq_min = None
            mapq_sum_lenw = 0.0
            len_sum = 0
            nm_sum = 0
            return

        contigs = list(E_rc.keys())
        p = [float(E_rc[c]) / total_e for c in contigs]

        # hard order
        k = len(contigs)
        if k < 2:
            stats.reads_skipped_k_lt_2 += 1
            E_rc = {}
            seg_count = 0
            mapq_min = None
            mapq_sum_lenw = 0.0
            len_sum = 0
            nm_sum = 0
            return

        # Effective order and concentration
        c_simpson = float(sum(x * x for x in p))
        if c_simpson <= 0.0:
            stats.reads_skipped_no_evidence += 1
            E_rc = {}
            seg_count = 0
            mapq_min = None
            mapq_sum_lenw = 0.0
            len_sum = 0
            nm_sum = 0
            return

        k_eff = 1.0 / c_simpson
        p_ok_mean = (mapq_sum_lenw / float(len_sum)) if len_sum > 0 else 0.0

        # Read reliability factor R(r) = mean(p_ok) * C(r)
        weight = float(p_ok_mean * c_simpson)

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
            mapq_min=int(mapq_min) if mapq_min is not None else 0,
            p_ok_mean=float(p_ok_mean),
            aligned_len_sum=int(len_sum),
            nm_sum=int(nm_sum),
        )

        stats.reads_kept += 1
        stats.k_counter[k] += 1
        stats.add_weight(weight)
        contact_id += 1

        E_rc = {}
        seg_count = 0
        mapq_min = None
        mapq_sum_lenw = 0.0
        len_sum = 0
        nm_sum = 0

    for aln in bam_fh.fetch(until_eof=True):
        stats.alignments_total += 1

        if aln.is_unmapped:
            stats.alignments_skipped_unmapped += 1
            continue
        if aln.is_secondary:
            stats.alignments_skipped_secondary += 1
            continue

        qname = aln.query_name
        if not qname:
            continue

        # Queryname sortedness check: required for streaming correctness.
        if prev_qname is not None and qname < prev_qname:
            raise BamContactsError(
                "BAM must be queryname-sorted (samtools sort -n) so we can stream-group alignments by QNAME.\n"
                f"Detected non-monotonic QNAME: {qname!r} < {prev_qname!r}."
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

        aln_len = int(aln.query_alignment_length or 0)
        if aln_len <= 0:
            # fallback to reference span if available
            if aln.reference_start is not None and aln.reference_end is not None:
                aln_len = int(aln.reference_end - aln.reference_start)
        if aln_len <= 0:
            continue

        mapq = int(aln.mapping_quality or 0)
        p_ok = p_ok_from_mapq(mapq)

        nm = 0
        try:
            nm = int(aln.get_tag("NM"))
        except Exception:
            nm = 0
        id_est = 1.0
        if aln_len > 0 and nm > 0:
            id_est = max(0.0, 1.0 - (float(nm) / float(aln_len)))

        e = float(p_ok * id_est * float(aln_len))
        if e > 0.0:
            E_rc[ref] = float(E_rc.get(ref, 0.0) + e)

        stats.alignments_kept += 1
        seg_count += 1
        mapq_min = mapq if mapq_min is None else min(mapq_min, mapq)
        mapq_sum_lenw += float(p_ok) * float(aln_len)
        len_sum += int(aln_len)
        nm_sum += int(nm)

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
            "secondary": "Secondary alignments are skipped to reduce multi-mapping noise.",
            "mapq": "MAPQ converted to p_ok=1-10^(-MAPQ/10). MAPQ=255 treated as unknown -> 0.",
            "coverage": "coverage bases are accumulated as sum(p_ok * aligned_len) per contig.",
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

    return {"contacts_parquet": out_parquet, "coverage_tsv": out_cov, "qc_json": out_qc, "stats": qc["stats"]}
