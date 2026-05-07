from __future__ import annotations

from pathlib import Path

import pytest


def test_evidence_preflight_collects_multiple_input_errors(tmp_path: Path) -> None:
    from porebin_genome.evidence.preflight import run_evidence_preflight

    report = run_evidence_preflight(
        bam=tmp_path / "missing.bam",
        contigs_fasta=tmp_path / "missing.fasta",
        coverage_method="coverm",
        coverage_bam=None,
        coverage_tsv=None,
        parquet_batch_size=0,
        coverage_threads=0,
    )

    assert not report.passed
    joined = "\n".join(report.errors)
    assert "--parquet-batch-size" in joined
    assert "--coverage-threads" in joined
    assert "Contigs FASTA not found" in joined
    assert "--bam not found" in joined
    assert "--coverage-bam is required" in joined


def test_evidence_preflight_rejects_coordinate_sorted_contact_bam(
    tmp_path: Path,
) -> None:
    pysam = pytest.importorskip("pysam")
    from porebin_genome.evidence.preflight import run_evidence_preflight

    contigs = _write_fasta(tmp_path / "contigs.fasta", ["c1"])
    bam = _write_empty_bam(
        pysam=pysam,
        path=tmp_path / "contact.bam",
        sort_order="coordinate",
        references=["c1"],
    )
    coverage = tmp_path / "coverage.tsv"
    coverage.write_text("contig_name\tcoverage\nc1\t10\n", encoding="utf-8")

    report = run_evidence_preflight(
        bam=bam,
        contigs_fasta=contigs,
        coverage_method="coverm",
        coverage_bam=None,
        coverage_tsv=coverage,
        parquet_batch_size=100,
        coverage_threads=1,
    )

    assert any("coordinate-sorted" in error for error in report.errors)


def test_evidence_preflight_accepts_coverage_tsv_with_warning_for_missing_contigs(
    tmp_path: Path,
) -> None:
    pysam = pytest.importorskip("pysam")
    from porebin_genome.evidence.preflight import run_evidence_preflight

    contigs = _write_fasta(tmp_path / "contigs.fasta", ["c1", "c2"])
    bam = _write_empty_bam(
        pysam=pysam,
        path=tmp_path / "contact.bam",
        sort_order="queryname",
        references=["c1", "c2"],
    )
    coverage = tmp_path / "coverage.tsv"
    coverage.write_text("contig_name\tcoverage\nc1\t10\n", encoding="utf-8")

    report = run_evidence_preflight(
        bam=bam,
        contigs_fasta=contigs,
        coverage_method="coverm",
        coverage_bam=None,
        coverage_tsv=coverage,
        parquet_batch_size=100,
        coverage_threads=1,
    )

    assert report.passed
    assert report.checks["coverage_source"] == "tsv"
    assert any("missing coverage for 1 FASTA contigs" in warning for warning in report.warnings)


def test_evidence_preflight_coverm_rejects_queryname_coverage_bam(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pysam = pytest.importorskip("pysam")
    from porebin_genome.evidence import preflight

    monkeypatch.setattr(preflight.shutil, "which", lambda _name: "/usr/bin/coverm")

    contigs = _write_fasta(tmp_path / "contigs.fasta", ["c1"])
    contact_bam = _write_empty_bam(
        pysam=pysam,
        path=tmp_path / "contact.bam",
        sort_order="queryname",
        references=["c1"],
    )
    coverage_bam = _write_empty_bam(
        pysam=pysam,
        path=tmp_path / "coverage.bam",
        sort_order="queryname",
        references=["c1"],
    )

    report = preflight.run_evidence_preflight(
        bam=contact_bam,
        contigs_fasta=contigs,
        coverage_method="coverm",
        coverage_bam=coverage_bam,
        coverage_tsv=None,
        parquet_batch_size=100,
        coverage_threads=1,
    )

    joined = "\n".join(report.errors)
    assert "--coverage-bam appears to be queryname-sorted" in joined
    assert "has no .bai or .csi index" in joined


def _write_fasta(path: Path, names: list[str]) -> Path:
    path.write_text("".join(f">{name}\nACGTACGT\n" for name in names), encoding="utf-8")
    return path


def _write_empty_bam(
    *,
    pysam: object,
    path: Path,
    sort_order: str,
    references: list[str],
) -> Path:
    header = {
        "HD": {"VN": "1.6", "SO": sort_order},
        "SQ": [{"SN": name, "LN": 1000} for name in references],
    }
    with pysam.AlignmentFile(str(path), "wb", header=header):
        pass
    return path
