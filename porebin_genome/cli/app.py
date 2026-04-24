"""Top-level CLI app for the genome-centric mainline."""

from __future__ import annotations

import typer

from porebin_genome.cli.binning import bin_command
from porebin_genome.cli.evidence import evidence_command
from porebin_genome.cli.export import export_command
from porebin_genome.io.runtime import setup_logging

app = typer.Typer(
    add_completion=False,
    help=(
        "porebin_genome: genome-centric Pore-C metagenomic binning. "
        "Mainline: evidence construction -> candidate genome-bin discovery -> "
        "refinement -> QC/export."
    ),
)


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Configure logging for CLI commands."""
    setup_logging(verbose=verbose)


app.command(
    "evidence",
    help="Evidence construction from reads.namesorted.bam into canonical contact evidence and coverage.",
)(evidence_command)
app.command(
    "bin",
    help="Run candidate genome-bin discovery followed by genome-bin refinement.",
)(bin_command)
app.command(
    "export",
    help="Export final genome bins and unresolved contigs as FASTA files.",
)(export_command)


if __name__ == "__main__":  # pragma: no cover
    app()
