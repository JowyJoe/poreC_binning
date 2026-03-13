from __future__ import annotations

import gzip
from porebin.pairwise_baseline import expand_to_pairs
from pathlib import Path

import pytest
from typer.testing import CliRunner


def test_expand_to_pairs_k2() -> None:
    edges = list(expand_to_pairs(["A", "B"]))
    assert edges == [("A", "B", 1.0)]


def test_expand_to_pairs_k5_weights_sum_to_one() -> None:
    edges = list(expand_to_pairs(["A", "B", "C", "D", "E"]))
    assert len(edges) == 10
    weights = [w for _, _, w in edges]
    assert abs(sum(weights) - 1.0) < 1e-12
    assert all(abs(w - 0.1) < 1e-12 for w in weights)


def test_expand_to_pairs_dedup_and_no_self_loops() -> None:
    edges = list(expand_to_pairs(["A", "A", "B", "B", "C"]))
    pairs = {(a, b) for a, b, _ in edges}
    assert pairs == {("A", "B"), ("A", "C"), ("B", "C")}
    assert all(a < b for a, b, _ in edges)
    assert all(a != b for a, b, _ in edges)
    assert all(abs(w - (1.0 / 3.0)) < 1e-12 for _, _, w in edges)


def test_cli_help_smoke() -> None:
    from porebin.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "run-bam" in result.stdout


def test_build_pairwise_edges_from_bam(tmp_path: Path) -> None:
    pysam = pytest.importorskip("pysam", reason="pairwise BAM baseline optional dependency not installed")

    from porebin.pairwise_baseline import build_pairwise_edges_from_bam

    contigs = tmp_path / "contigs.fasta"
    contigs.write_text(
        ">A\n" + ("A" * 100) + "\n>B\n" + ("C" * 100) + "\n>C\n" + ("G" * 100) + "\n",
        encoding="utf-8",
    )

    bam = tmp_path / "reads.namesorted.bam"
    header = {
        "HD": {"VN": "1.6", "SO": "queryname"},
        "SQ": [
            {"SN": "A", "LN": 100},
            {"SN": "B", "LN": 100},
            {"SN": "C", "LN": 100},
        ],
    }

    def seg(*, qname: str, ref_id: int, flag: int = 0) -> "pysam.AlignedSegment":
        a = pysam.AlignedSegment()
        a.query_name = qname
        a.query_sequence = "T" * 50
        a.flag = int(flag)
        a.reference_id = int(ref_id)
        a.reference_start = 0
        a.mapping_quality = 30
        a.cigarstring = "50M"
        return a

    with pysam.AlignmentFile(str(bam), "wb", header=header) as out:
        out.write(seg(qname="read1", ref_id=0))
        out.write(seg(qname="read1", ref_id=1))
        out.write(seg(qname="read1", ref_id=1, flag=0x800))
        out.write(seg(qname="read1", ref_id=2, flag=0x100))
        out.write(seg(qname="read2", ref_id=2))

    out_edges = tmp_path / "pairwise_edges.tsv.gz"
    stats = build_pairwise_edges_from_bam(
        bam=bam,
        contigs_fasta=contigs,
        out_edges_path=out_edges,
        tmp_dir=tmp_path / "tmp",
        chunk_lines=10,
    )

    assert stats.alignments_total == 5
    assert stats.alignments_kept == 3
    assert stats.reads_total == 2
    assert stats.reads_skipped_k_lt_2 == 1
    assert stats.raw_pairs_written == 1
    assert stats.unique_edges == 1
    assert stats.input_sorted_by_qname is True

    with gzip.open(out_edges, "rt", encoding="utf-8") as fh:
        lines = [line.rstrip("\n") for line in fh if line.strip()]
    assert lines == ["A\tB\t1"]
