from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_contigs(path: Path) -> None:
    path.write_text(
        ">contigA\n" + ("A" * 1000) + "\n>contigB\n" + ("C" * 1000) + "\n>contigC\n" + ("G" * 1000) + "\n",
        encoding="utf-8",
    )


def test_bam2contacts_keeps_supplementary_drops_secondary_skips_unmapped(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for contacts.parquet")
    pysam = pytest.importorskip("pysam", reason="bam2contacts optional dependency not installed")

    from porebin.cli import bam2contacts

    contigs = tmp_path / "contigs.fasta"
    _write_contigs(contigs)

    bam = tmp_path / "reads.namesorted.bam"
    header = {
        "HD": {"VN": "1.6", "SO": "queryname"},
        "SQ": [
            {"SN": "contigA", "LN": 1000},
            {"SN": "contigB", "LN": 1000},
            {"SN": "contigC", "LN": 1000},
        ],
    }

    def seg(*, flag: int, rname: str | None) -> "pysam.AlignedSegment":
        a = pysam.AlignedSegment()
        a.query_name = "read1"
        a.query_sequence = "T" * 100
        a.flag = int(flag)
        if rname is None:
            a.reference_id = -1
            a.reference_start = -1
            a.mapping_quality = 0
            a.cigarstring = "*"
        else:
            a.reference_id = {"contigA": 0, "contigB": 1, "contigC": 2}[rname]
            a.reference_start = 0
            a.mapping_quality = 10
            a.cigarstring = "100M"
            a.set_tag("NM", 0)
        return a

    with pysam.AlignmentFile(str(bam), "wb", header=header) as out:
        # primary mapped to contigA
        out.write(seg(flag=0x0, rname="contigA"))
        # supplementary mapped to contigB (must be kept)
        out.write(seg(flag=0x800, rname="contigB"))
        # secondary mapped to contigC (must be dropped)
        out.write(seg(flag=0x100, rname="contigC"))
        # unmapped (must be skipped)
        out.write(seg(flag=0x4, rname=None))

    out_dir = tmp_path / "out"
    bam2contacts(bam=bam, contigs=contigs, out=out_dir, parquet_batch_size=10_000)

    run = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    stats = run.get("stats") or {}
    for key in ("alignments_skipped_unmapped", "alignments_skipped_secondary", "alignments_supplementary_used"):
        assert key in stats
    assert stats["alignments_skipped_unmapped"] == 1
    assert stats["alignments_skipped_secondary"] == 1
    assert stats["alignments_supplementary_used"] == 1

    import pyarrow.parquet as pq

    table = pq.read_table(run["outputs"]["contacts_parquet"])
    d = table.to_pydict()
    assert len(d["contact_id"]) == 1
    contigs_out = set(d["contigs"][0])
    assert "contigA" in contigs_out
    assert "contigB" in contigs_out
    assert "contigC" not in contigs_out

    qc = json.loads(Path(run["outputs"]["qc_json"]).read_text(encoding="utf-8"))
    assert qc["stats"]["unmapped_skipped"] == 1
    assert qc["stats"]["secondary_skipped"] == 1
    assert qc["stats"]["supplementary_used"] == 1
