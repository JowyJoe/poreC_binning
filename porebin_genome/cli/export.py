"""CLI command for FASTA export of final bins and unresolved contigs."""

from __future__ import annotations

from pathlib import Path

import typer

from porebin_genome.export.orchestrate import export_final_results
from porebin_genome.io.runtime import record_run


def export_command(
    contigs: Path = typer.Option(..., "--contigs", help="Contigs FASTA file."),
    bins_refined_tsv: Path = typer.Option(..., "--bins-refined-tsv", help="Final refined genome-bin table."),
    unbinned_tsv: Path = typer.Option(..., "--unbinned-tsv", help="Unresolved-contig table from the binning mainline."),
    out: Path = typer.Option(..., "--out", help="Output directory."),
) -> None:
    """Materialize FASTA files for final bins and unresolved contigs."""
    out = out.resolve()
    params = {
        "contigs": str(contigs),
        "bins_refined_tsv": str(bins_refined_tsv),
        "unbinned_tsv": str(unbinned_tsv),
        "out": str(out),
    }
    with record_run(out, command="export", params=params) as run_record:
        result = export_final_results(
            contigs_fasta=contigs,
            bins_refined_tsv=bins_refined_tsv,
            unbinned_tsv=unbinned_tsv,
            out_dir=out,
        )
        run_record["outputs"] = {
            "bins_fasta_dir": str(result.bins_fasta_dir),
            "unresolved_fasta": str(result.unresolved_fasta),
        }
        run_record["stats"] = {
            "bins_exported": result.bins_exported,
            "unresolved_contigs": result.unresolved_contigs,
        }
