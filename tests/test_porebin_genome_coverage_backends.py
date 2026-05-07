from __future__ import annotations

from pathlib import Path


def test_parse_coverm_mean_depth_tsv(tmp_path: Path) -> None:
    from porebin_genome.evidence.coverage_backends import parse_coverm_mean_depth_tsv

    coverm = tmp_path / "coverm.contig.tsv"
    coverm.write_text(
        "Contig\tsample.bam Mean\n"
        "c1\t12.5\n"
        "c2\t0\n",
        encoding="utf-8",
    )

    assert parse_coverm_mean_depth_tsv(coverm) == {"c1": 12.5, "c2": 0.0}


def test_adopt_coverage_tsv_copies_public_contract(tmp_path: Path) -> None:
    from porebin_genome.evidence.coverage_backends import adopt_coverage_tsv
    from porebin_genome.io.coverage import read_coverage_tsv

    source = tmp_path / "input_coverage.tsv"
    out = tmp_path / "evidence" / "coverage.tsv"
    source.write_text("contig_name\tcoverage\nc1\t7\nc2\t9.5\n", encoding="utf-8")

    result = adopt_coverage_tsv(source_tsv=source, out_tsv=out)

    assert result.source == "tsv"
    assert result.coverage_tsv == out.resolve()
    assert read_coverage_tsv(out) == {"c1": 7.0, "c2": 9.5}
