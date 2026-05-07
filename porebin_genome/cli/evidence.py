"""CLI command for evidence construction."""

from __future__ import annotations

from pathlib import Path

import typer

from porebin_genome.evidence.orchestrate import run_evidence_from_bam
from porebin_genome.io.runtime import record_run


def evidence_command(
    bam: Path = typer.Option(..., "--bam", help="Queryname-sorted Pore-C BAM used for contact evidence."),
    contigs: Path = typer.Option(..., "--contigs", help="Contigs FASTA file."),
    out: Path = typer.Option(..., "--out", help="Output directory."),
    parquet_batch_size: int = typer.Option(10_000, "--parquet-batch-size", help="Parquet batch size."),
    coverage_method: str = typer.Option(
        "coverm",
        "--coverage-method",
        help="Coverage backend: coverm or internal. Ignored when --coverage-tsv is provided.",
    ),
    coverage_bam: Path | None = typer.Option(
        None,
        "--coverage-bam",
        help=(
            "Reference-sorted contig-aligned BAM for CoverM mean-depth; "
            "required unless --coverage-tsv or --coverage-method internal is used."
        ),
    ),
    coverage_tsv: Path | None = typer.Option(
        None,
        "--coverage-tsv",
        help="Existing porebin-compatible coverage table to adopt instead of recomputing coverage.",
    ),
    coverage_threads: int = typer.Option(1, "--coverage-threads", help="Threads passed to CoverM."),
) -> None:
    """Build canonical contact evidence and a contig mean-depth coverage table."""
    out = out.resolve()
    params = {
        "bam": str(bam),
        "contigs": str(contigs),
        "out": str(out),
        "parquet_batch_size": parquet_batch_size,
        "coverage_method": coverage_method,
        "coverage_bam": str(coverage_bam) if coverage_bam is not None else None,
        "coverage_tsv": str(coverage_tsv) if coverage_tsv is not None else None,
        "coverage_threads": coverage_threads,
    }
    with record_run(out, command="evidence", params=params) as run_record:
        result = run_evidence_from_bam(
            bam=bam,
            contigs_fasta=contigs,
            out_dir=out,
            parquet_batch_size=parquet_batch_size,
            coverage_method=coverage_method,
            coverage_bam=coverage_bam,
            coverage_tsv=coverage_tsv,
            coverage_threads=coverage_threads,
        )
        run_record["outputs"] = {
            "contacts_parquet": str(result.contacts_parquet),
            "coverage_tsv": str(result.coverage_tsv),
            "evidence_qc_json": str(result.qc_json),
        }
        run_record["stats"] = {
            "reads_kept": result.reads_kept,
            "coverage_source": result.coverage_source,
        }
