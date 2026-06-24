from __future__ import annotations

from typer.testing import CliRunner


def test_porebin_genome_cli_help_smoke() -> None:
    from porebin_genome.cli import app

    runner = CliRunner()

    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "genome-centric" in result.stdout
    assert "evidence" in result.stdout
    assert "bin" in result.stdout
    assert "export" in result.stdout
    assert "spectral" in result.stdout
    assert "HG-VAE" in result.stdout
    assert "host" not in result.stdout.lower()

    evidence_help = runner.invoke(app, ["evidence", "--help"])
    assert evidence_help.exit_code == 0
    assert "evidence construction" in evidence_help.stdout.lower()
    assert "--bam" in evidence_help.stdout
    assert "--coverage-bam" in evidence_help.stdout
    assert "--coverage-tsv" in evidence_help.stdout

    bin_help = runner.invoke(app, ["bin", "--help"])
    assert bin_help.exit_code == 0
    assert "--contacts" in bin_help.stdout
    assert "--coverage-tsv" in bin_help.stdout
    assert "--knn-k" in bin_help.stdout
    assert "hyperedge" in bin_help.stdout.lower()
    assert "HG-VAE" in bin_help.stdout
    assert "adaptive" in bin_help.stdout
    assert "stable" in bin_help.stdout
    assert "recommended" in bin_help.stdout
    assert "experimental" in bin_help.stdout
    assert "similarity" in bin_help.stdout
    assert "candidate genome-bin discovery" in bin_help.stdout.lower() or "refinement" in bin_help.stdout.lower()
    assert "placeholder" not in bin_help.stdout.lower()
    assert "host" not in bin_help.stdout.lower()

    export_help = runner.invoke(app, ["export", "--help"])
    assert export_help.exit_code == 0
    assert "--bins-refined-tsv" in export_help.stdout
    assert "--unbinned-tsv" in export_help.stdout
