from __future__ import annotations
import json
from pathlib import Path
import typer
from typing import Optional

from .pipeline.pipeline import run_pipeline, self_test_pipeline
from .pipeline.export_bins import export_bins_fasta

app = typer.Typer(add_completion=False, help="Hypergraph spectral binning (single-source Pore-C)")


@app.command()
def pipeline(
    config: Path = typer.Argument(..., exists=True, help="Path to YAML config"),
    k: Optional[int] = typer.Option(None, help="Override spectral k (clusters/embedding dim)"),
):
    """Run end-to-end pipeline: BAM/SAM -> hypergraph -> spectral -> bins.tsv"""
    run_pipeline(config_path=config, override_k=k)


@app.command("self-test")
def self_test():
    """Run a tiny synthetic hypergraph example (no external data)."""
    self_test_pipeline()


if __name__ == "__main__":
    app()


@app.command("export-bins")
def export_bins(
    contigs: Path = typer.Argument(..., exists=True, help="Contigs FASTA path"),
    bins: Path = typer.Argument(..., exists=True, help="bins.tsv (contig\tbin)"),
    out_dir: Path = typer.Argument(..., help="Output directory to write per-bin FASTA files"),
):
    """Export per-bin FASTA files for downstream tools (e.g., CheckM2)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    export_bins_fasta(contigs, bins, out_dir)
    typer.echo(f"Per-bin FASTA written to {out_dir}")