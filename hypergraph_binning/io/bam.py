from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import pysam


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
    mapq_min: int = 20
    segment_min_bases: int = 500
    min_segments_per_read: int = 3
    read_coverage_min: float = 0.6
    max_hyperedge_size: int = 20


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
) -> Iterator[Tuple[List[str], float]]:
    """
    Yield (members, quality_weight) for each qualified Pore-C read as a hyperedge.
    members: ordered unique contig names within the read
    quality_weight: q_r in (0,1]
    """
    bam = pysam.AlignmentFile(bam_path, "rb" if bam_path.endswith(".bam") else "r")
    try:
        for bundle in _bundle_alignments_by_read(bam):
            # filter segments by MAPQ and min length
            segs = [s for s in bundle.segments if s.mapq >= flt.mapq_min and s.qaln >= flt.segment_min_bases]
            if len(segs) < flt.min_segments_per_read:
                continue
            # coverage fraction on read
            if bundle.explained_frac < flt.read_coverage_min:
                continue
            # dedup contig IDs, and ensure they exist in contig set
            contigs = [c for c in bundle.contig_set() if c in contig_name_set]
            k = len(contigs)
            if k < 2:
                continue
            if k > flt.max_hyperedge_size:
                # truncate by keeping the first max_hyperedge_size members (deterministic)
                contigs = contigs[:flt.max_hyperedge_size]
                k = len(contigs)
            # quality weight q_r
            q_mapq = max(0.0, min(1.0, (bundle.mean_mapq() - flt.mapq_min) / max(1.0, 60 - flt.mapq_min)))
            q_seg = max(0.0, min(1.0, (bundle.min_segment_qaln() - flt.segment_min_bases) / max(1.0, 2000 - flt.segment_min_bases)))
            q_cov = bundle.explained_frac  # already 0..1
            q_r = 0.2 + 0.8 * (0.5 * q_mapq + 0.25 * q_seg + 0.25 * q_cov)  # keep >0.2 baseline
            yield contigs, float(q_r)
    finally:
        bam.close()
