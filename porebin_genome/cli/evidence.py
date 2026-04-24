"""CLI command for evidence construction."""

from __future__ import annotations

from pathlib import Path

import typer

from porebin_genome.evidence.orchestrate import run_evidence_from_bam
from porebin_genome.io.runtime import record_run


def evidence_command(
    bam: Path = typer.Option(..., "--bam", help="Queryname-sorted Pore-C BAM used only for evidence construction."),
    contigs: Path = typer.Option(..., "--contigs", help="Contigs FASTA file."),
    out: Path = typer.Option(..., "--out", help="Output directory."),
    parquet_batch_size: int = typer.Option(10_000, "--parquet-batch-size", help="Parquet batch size."),
) -> None:
    """Build canonical contact evidence and coverage from a name-sorted BAM."""
    out = out.resolve()
    params = {
        "bam": str(bam),
        "contigs": str(contigs),
        "out": str(out),
        "parquet_batch_size": parquet_batch_size,
    }
    with record_run(out, command="evidence", params=params) as run_record:
        result = run_evidence_from_bam(
            bam=bam,
            contigs_fasta=contigs,
            out_dir=out,
            parquet_batch_size=parquet_batch_size,
        )
        run_record["outputs"] = {
            "contacts_parquet": str(result.contacts_parquet),
            "coverage_tsv": str(result.coverage_tsv),
            "evidence_qc_json": str(result.qc_json),
        }
        run_record["stats"] = {"reads_kept": result.reads_kept}
