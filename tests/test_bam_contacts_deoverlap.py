from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_bam2contacts_deoverlap_union_len_and_qc(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for contacts.parquet")
    pysam = pytest.importorskip("pysam", reason="bam2contacts optional dependency not installed")

    from porebin.cli import bam2contacts

    contigs = tmp_path / "contigs.fasta"
    contigs.write_text(">contigA\n" + ("A" * 1000) + "\n", encoding="utf-8")

    bam = tmp_path / "reads.bam"
    header = {"HD": {"VN": "1.6", "SO": "unknown"}, "SQ": [{"SN": "contigA", "LN": 1000}]}

    def seg(*, cigar: str, flag: int) -> "pysam.AlignedSegment":
        a = pysam.AlignedSegment()
        a.query_name = "read1"
        a.query_sequence = "T" * 150
        a.flag = int(flag)
        a.reference_id = 0
        a.reference_start = 0
        a.mapping_quality = 60
        a.cigarstring = cigar
        a.set_tag("NM", 0)
        return a

    with pysam.AlignmentFile(str(bam), "wb", header=header) as out:
        # Two overlapping query intervals on the same contig:
        # seg1: [0,100) via 100M50S
        # seg2: [50,150) via 50S100M (supplementary)
        out.write(seg(cigar="100M50S", flag=0x0))
        out.write(seg(cigar="50S100M", flag=0x800))

    out_dir = tmp_path / "out"
    bam2contacts(bam=bam, contigs=contigs, out=out_dir, parquet_batch_size=10_000)

    run = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    qc = json.loads(Path(run["outputs"]["qc_json"]).read_text(encoding="utf-8"))

    stats = qc["stats"]
    assert stats["deoverlap_enabled"] is True
    assert stats["overlap_blocks_created_total"] == 1
    assert stats["overlap_segments_merged_total"] == 1

    import pyarrow.parquet as pq

    table = pq.read_table(run["outputs"]["contacts_parquet"])
    d = table.to_pydict()

    assert len(d["contact_id"]) == 1
    assert d["contigs"][0] == ["contigA"]
    assert d["contig_weights"][0] == [1.0]

    # Key assertion: union length is 150 (not 200), written as a debug column.
    assert d["deoverlap_query_union_len_sum"][0] == 150

