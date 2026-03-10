from __future__ import annotations

from pathlib import Path

import pytest


def _write_fasta(path: Path) -> None:
    path.write_text(
        ">A\n" + ("A" * 1000) + "\n>B\n" + ("C" * 1000) + "\n",
        encoding="utf-8",
    )


def _write_name_sorted_bam(path: Path) -> None:
    pysam = pytest.importorskip("pysam", reason="bam2contacts optional dependency not installed")

    header = {
        "HD": {"VN": "1.6", "SO": "queryname"},
        "SQ": [
            {"SN": "A", "LN": 1000},
            {"SN": "B", "LN": 1000},
        ],
    }

    def seg(*, qname: str, rname: str, mapq: int, nm: int | None) -> "pysam.AlignedSegment":
        a = pysam.AlignedSegment()
        a.query_name = qname
        a.query_sequence = "G" * 100
        a.flag = 0
        a.reference_id = 0 if rname == "A" else 1
        a.reference_start = 0
        a.mapping_quality = int(mapq)
        a.cigarstring = "100M"
        if nm is not None:
            a.set_tag("NM", int(nm))
        return a

    with pysam.AlignmentFile(str(path), "wb", header=header) as out_bam:
        # Read 1: uniform soft assignment, includes MAPQ=255 and a missing NM tag.
        out_bam.write(seg(qname="r_uniform", rname="A", mapq=255, nm=None))  # MAPQ unknown, NM missing
        out_bam.write(seg(qname="r_uniform", rname="B", mapq=255, nm=0))

        # Read 2: same q(r) as r_uniform but different concentration C(r) via skewed E_rc.
        # Segment1: p_ok=0.9 (MAPQ=10), id=1 -> e=90
        # Segment2: p_ok=0.5 (MAPQ=255 treated as missing), id=0.2 (NM=80, len=100) -> e=10
        # => q(r)=(90+10)/200=0.5, but P=(0.9,0.1) so C(r)=0.82 and k_eff=1/0.82.
        out_bam.write(seg(qname="r_skew", rname="A", mapq=10, nm=0))
        out_bam.write(seg(qname="r_skew", rname="B", mapq=255, nm=80))


def test_bam2contacts_weight_mapq255_and_no_concentration_penalty(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for contacts.parquet")
    pytest.importorskip("pysam", reason="bam2contacts optional dependency not installed")

    from porebin.bam_contacts import bam_to_contacts_parquet

    contigs = tmp_path / "contigs.fasta"
    _write_fasta(contigs)

    bam = tmp_path / "reads.namesorted.bam"
    _write_name_sorted_bam(bam)

    out_dir = tmp_path / "out"
    meta = bam_to_contacts_parquet(bam=bam, contigs_fasta=contigs, out_dir=out_dir, parquet_batch_size=10_000)

    import pyarrow.parquet as pq

    table = pq.read_table(meta["contacts_parquet"])
    d = table.to_pydict()

    assert d["k"] == [2, 2]
    assert all(abs(sum(ws) - 1.0) < 1e-12 for ws in d["contig_weights"])

    # MAPQ=255 is treated as unknown/missing, but p_ok=0.5 (not 0), so weight stays > 0.
    w0, w1 = d["weight"]
    assert abs(w0 - 0.5) < 1e-12
    assert abs(w1 - 0.5) < 1e-12

    # Different concentration C(r) (via k_eff), but same weight=q(r): weight must not depend on C(r).
    keff0, keff1 = d["k_eff"]
    assert abs(keff0 - 2.0) < 1e-12  # P=(0.5,0.5) -> C=0.5 -> k_eff=2
    assert abs(keff1 - (1.0 / (0.9**2 + 0.1**2))) < 1e-12

    assert d["mapq_missing_count"] == [2, 1]
    assert d["nm_missing_count"] == [1, 0]

    # p_ok_mean is length-weighted mean of p_ok(a) (id not included).
    assert abs(d["p_ok_mean"][0] - 0.5) < 1e-12
    assert abs(d["p_ok_mean"][1] - 0.7) < 1e-12

