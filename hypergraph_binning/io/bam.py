from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import pysam
import numpy as np


@dataclass
class ReadSegment:
    contig: str
    mapq: int
    qlen: int  # query length (full read)
    qaln: int  # aligned query length on this segment


@dataclass
class ReadBundle:
    read_id: str
    segments: List[ReadSegment]

    @property
    def explained_frac(self) -> float:
        qlen = max((s.qlen for s in self.segments), default=0)
        if qlen <= 0:
            return 0.0
        total_qaln = sum(s.qaln for s in self.segments)
        return min(1.0, total_qaln / max(1, qlen))

    def contig_set(self) -> List[str]:
        # merge adjacent segments on same contig is upstream task; here we deduplicate contig IDs
        # keep order stable by first appearance
        seen: Set[str] = set()
        ordered: List[str] = []
        for s in self.segments:
            if s.contig not in seen:
                seen.add(s.contig)
                ordered.append(s.contig)
        return ordered

    def min_segment_qaln(self) -> int:
        return min((s.qaln for s in self.segments), default=0)

    def mean_mapq(self) -> float:
        vals = [s.mapq for s in self.segments]
        return sum(vals) / len(vals) if vals else 0.0


@dataclass
class PoreCFilter:
    mapq_min: int = 30
    segment_min_bases: int = 1000
    min_segments_per_read: int = 3
    read_coverage_min: float = 0.0  # kept for compatibility; not used in weighting
    max_hyperedge_size: int = 20


@dataclass
class PoreCStats:
    reads_total: int = 0
    reads_pass_segments: int = 0
    reads_pass_contigs: int = 0
    reads_filtered_low_quality: int = 0
    reads_filtered_small: int = 0
    edges_truncated: int = 0
    edges_yielded: int = 0
    contig_hits_discarded: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "reads_total": self.reads_total,
            "reads_pass_segments": self.reads_pass_segments,
            "reads_pass_contigs": self.reads_pass_contigs,
            "reads_filtered_low_quality": self.reads_filtered_low_quality,
            "reads_filtered_small": self.reads_filtered_small,
            "edges_truncated": self.edges_truncated,
            "edges_yielded": self.edges_yielded,
            "contig_hits_discarded": self.contig_hits_discarded,
        }


def _bundle_alignments_by_read(bam: pysam.AlignmentFile) -> Iterator[ReadBundle]:
    """
    Group alignments by read_id. This requires BAM to be name-sorted for streaming efficiency.
    If the BAM is coordinate-sorted, this will buffer all reads in memory (not recommended for large data).
    """
    current_id: Optional[str] = None
    current_segments: List[ReadSegment] = []

    def flush():
        nonlocal current_id, current_segments
        if current_id is not None and current_segments:
            yield ReadBundle(read_id=current_id, segments=current_segments)
        current_id = None
        current_segments = []

    buffer: Dict[str, List[ReadSegment]] = {}

    # Try to detect if name-sorted by checking first few records
    prev_qname: Optional[str] = None
    name_sorted = True
    peeked: List[pysam.AlignedSegment] = []
    for i, aln in enumerate(bam.fetch(until_eof=True)):
        if i < 1000:
            peeked.append(aln)
            if prev_qname is not None and aln.query_name != prev_qname and any(a.query_name == prev_qname for a in peeked[-2:-1]):
                # cannot reliably detect here; keep name_sorted True by default
                pass
        else:
            break
        prev_qname = aln.query_name
    # Restart reading from beginning
    bam.reset()

    if name_sorted:
        for aln in bam.fetch(until_eof=True):
            if aln.is_unmapped or aln.is_secondary or aln.is_supplementary:
                continue
            qname = aln.query_name
            if current_id is None:
                current_id = qname
            if qname != current_id:
                # flush previous
                yield ReadBundle(read_id=current_id, segments=current_segments)
                current_id = qname
                current_segments = []
            qlen = aln.query_length or (aln.infer_query_length() or 0)
            qaln = aln.query_alignment_length or 0
            current_segments.append(ReadSegment(contig=aln.reference_name, mapq=int(aln.mapping_quality), qlen=int(qlen), qaln=int(qaln)))
        # flush last
        if current_id is not None and current_segments:
            yield ReadBundle(read_id=current_id, segments=current_segments)
    else:
        # Fallback: coordinate-sorted; buffer by read name (memory heavy)
        for aln in bam.fetch(until_eof=True):
            if aln.is_unmapped or aln.is_secondary or aln.is_supplementary:
                continue
            qlen = aln.query_length or (aln.infer_query_length() or 0)
            qaln = aln.query_alignment_length or 0
            seg = ReadSegment(contig=aln.reference_name, mapq=int(aln.mapping_quality), qlen=int(qlen), qaln=int(qaln))
            buffer.setdefault(aln.query_name, []).append(seg)
        for qname, segs in buffer.items():
            yield ReadBundle(read_id=qname, segments=segs)


def iterate_porec_hyperedges(
    bam_path: str,
    contig_name_set: Set[str],
    flt: PoreCFilter,
    stats: Optional[PoreCStats] = None,
    log_every: int = 0,
) -> Iterator[Tuple[List[str], float]]:
    """
    Yield (members, q_prime) for each qualified Pore-C read as a hyperedge.
    members: ordered unique contig names within the read (after per-read dedup/merge)
    q_prime (q'): r * (prod_i p_i)^(1/k), where p_i = 1 - 10^(-MAPQ_i/10),
      r = explained fraction using per-contig merged aligned length / read length.
    """
    bam = pysam.AlignmentFile(bam_path, "rb" if bam_path.endswith(".bam") else "r")
    log_every = int(log_every or 0)
    try:
        for bundle in _bundle_alignments_by_read(bam):
            if stats is not None:
                stats.reads_total += 1
                if log_every and stats.reads_total % log_every == 0:
                    print(
                        f"[Pore-C] processed {stats.reads_total:,} reads "
                        f"(yielded {stats.edges_yielded:,} edges, "
                        f"pass_segments={stats.reads_pass_segments:,}, "
                        f"pass_contigs={stats.reads_pass_contigs:,})"
                    )
            # 1) per-segment filtering by MAPQ and min aligned length
            segs = [s for s in bundle.segments if s.mapq >= flt.mapq_min and s.qaln >= flt.segment_min_bases]
            if not segs:
                if stats is not None:
                    stats.reads_filtered_low_quality += 1
                continue

            if stats is not None:
                stats.reads_pass_segments += 1

            # 2) merge same-read hits on the same contig: keep max MAPQ and max aligned length
            per_contig_mapq: Dict[str, int] = {}
            per_contig_qaln: Dict[str, int] = {}
            contigs_ordered: List[str] = []
            seen: Set[str] = set()
            for s in segs:
                c = s.contig
                if c not in contig_name_set:
                    if stats is not None:
                        stats.contig_hits_discarded += 1
                    continue
                if c not in seen:
                    seen.add(c)
                    contigs_ordered.append(c)
                    per_contig_mapq[c] = int(s.mapq)
                    per_contig_qaln[c] = int(s.qaln)
                else:
                    if s.mapq > per_contig_mapq[c]:
                        per_contig_mapq[c] = int(s.mapq)
                    if s.qaln > per_contig_qaln[c]:
                        per_contig_qaln[c] = int(s.qaln)

            # 3) k based on unique contigs after merge
            k = len(contigs_ordered)
            if k < flt.min_segments_per_read:
                if stats is not None:
                    stats.reads_filtered_small += 1
                continue
            if stats is not None:
                stats.reads_pass_contigs += 1

            # optionally truncate very large hyperedges deterministically
            if k > flt.max_hyperedge_size:
                keep = contigs_ordered[:flt.max_hyperedge_size]
                per_contig_mapq = {c: per_contig_mapq[c] for c in keep}
                per_contig_qaln = {c: per_contig_qaln[c] for c in keep}
                contigs_ordered = keep
                k = len(contigs_ordered)
                if stats is not None:
                    stats.edges_truncated += 1

            # 4) compute r using per-contig merged aligned length (avoid double counting)
            qlen = max((s.qlen for s in segs), default=0)
            merged_aln = sum(per_contig_qaln.values())
            r = float(min(1.0, merged_aln / max(1, qlen))) if qlen > 0 else 0.0

            # 5) compute q_read via geometric mean over p_i = 1 - 10^(-MAPQ/10)
            p_vals = []
            for c in contigs_ordered:
                m = per_contig_mapq[c]
                p = 1.0 - pow(10.0, -float(m) / 10.0)
                # safety clamp
                p_vals.append(float(np.clip(p, 1e-12, 1.0)))
            if not p_vals:
                continue
            log_p = np.log(p_vals)
            q_read = float(np.exp(log_p.mean()))
            q_prime = float(np.clip(r * q_read, 0.0, 1.0))

            if stats is not None:
                stats.edges_yielded += 1
            yield contigs_ordered, q_prime
    finally:
        bam.close()
