"""Evidence-layer orchestration for the genome-centric workflow."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from porebin_genome.evidence.bam import bam_to_contact_evidence
from porebin_genome.evidence.coverage_backends import (
    CoverageBuildResult,
    adopt_coverage_tsv,
    compute_coverm_coverage_tsv,
)
from porebin_genome.evidence.preflight import run_evidence_preflight
from porebin_genome.io.contacts import validate_contacts_parquet_core_schema
from porebin_genome.io.contracts import build_pipeline_layout
from porebin_genome.io.coverage import validate_coverage_tsv
from porebin_genome.io.runtime import write_json


@dataclass(frozen=True)
class EvidenceRunResult:
    """Outputs produced by evidence construction."""

    contacts_parquet: Path
    coverage_tsv: Path
    qc_json: Path
    reads_kept: int
    coverage_source: str


def run_evidence_from_bam(
    *,
    bam: Path,
    contigs_fasta: Path,
    out_dir: Path,
    parquet_batch_size: int = 10_000,
    coverage_method: str = "coverm",
    coverage_bam: Optional[Path] = None,
    coverage_tsv: Optional[Path] = None,
    coverage_threads: int = 1,
    logger: Optional[object] = None,
) -> EvidenceRunResult:
    """Run BAM-derived evidence construction and validate the resulting contracts."""
    logger = logger or logging.getLogger("porebin_genome")
    layout = build_pipeline_layout(out_dir)
    coverage_method = str(coverage_method).strip().lower()
    if coverage_tsv is not None:
        coverage_source = "tsv"
    elif coverage_method in {"coverm", "internal"}:
        coverage_source = coverage_method
    else:
        raise ValueError("coverage_method must be one of: coverm, internal")

    preflight = run_evidence_preflight(
        bam=bam,
        contigs_fasta=contigs_fasta,
        coverage_method=coverage_method,
        coverage_bam=coverage_bam,
        coverage_tsv=coverage_tsv,
        parquet_batch_size=parquet_batch_size,
        coverage_threads=coverage_threads,
    )
    for warning in preflight.warnings:
        logger.warning(warning)
    preflight.raise_for_errors()

    meta = bam_to_contact_evidence(
        bam=bam,
        contigs_fasta=contigs_fasta,
        out_dir=layout.root,
        parquet_batch_size=parquet_batch_size,
        write_internal_coverage=coverage_source == "internal",
    )
    validate_contacts_parquet_core_schema(Path(meta["contacts_parquet"]))
    coverage_result = _build_coverage(
        coverage_source=coverage_source,
        coverage_bam=coverage_bam,
        coverage_tsv=coverage_tsv,
        contigs_fasta=contigs_fasta,
        out_tsv=layout.coverage_tsv,
        raw_coverm_tsv=layout.evidence_dir / "coverm.contig.tsv",
        threads=coverage_threads,
        internal_meta=meta,
    )
    validate_coverage_tsv(coverage_result.coverage_tsv)
    _annotate_evidence_qc(
        qc_json=Path(meta["qc_json"]),
        coverage_result=coverage_result,
        coverage_method=coverage_method,
        coverage_bam=coverage_bam,
        coverage_tsv=coverage_tsv,
    )
    return EvidenceRunResult(
        contacts_parquet=Path(meta["contacts_parquet"]).resolve(),
        coverage_tsv=coverage_result.coverage_tsv.resolve(),
        qc_json=Path(meta["qc_json"]).resolve(),
        reads_kept=int(meta["stats"]["reads_kept"]),
        coverage_source=coverage_result.source,
    )


def _build_coverage(
    *,
    coverage_source: str,
    coverage_bam: Optional[Path],
    coverage_tsv: Optional[Path],
    contigs_fasta: Path,
    out_tsv: Path,
    raw_coverm_tsv: Path,
    threads: int,
    internal_meta: dict,
) -> CoverageBuildResult:
    if coverage_source == "tsv":
        assert coverage_tsv is not None
        return adopt_coverage_tsv(source_tsv=coverage_tsv, out_tsv=out_tsv)
    if coverage_source == "internal":
        internal_coverage = internal_meta.get("coverage_tsv")
        if internal_coverage is None:
            raise RuntimeError("Internal coverage was requested but no internal coverage table was written.")
        return CoverageBuildResult(
            coverage_tsv=Path(internal_coverage).resolve(),
            source="internal",
            raw_output_tsv=None,
            n_contigs=0,
            n_missing_contigs=0,
        )
    if coverage_bam is None:
        raise ValueError(
            "CoverM coverage requires --coverage-bam with a reference-sorted contig-aligned BAM. "
            "Use --coverage-tsv to provide precomputed contig mean depth instead."
        )
    return compute_coverm_coverage_tsv(
        coverage_bam=coverage_bam,
        contigs_fasta=contigs_fasta,
        out_tsv=out_tsv,
        raw_output_tsv=raw_coverm_tsv,
        threads=threads,
    )


def _annotate_evidence_qc(
    *,
    qc_json: Path,
    coverage_result: CoverageBuildResult,
    coverage_method: str,
    coverage_bam: Optional[Path],
    coverage_tsv: Optional[Path],
) -> None:
    payload = json.loads(qc_json.read_text(encoding="utf-8"))
    payload.setdefault("outputs", {})["coverage_tsv"] = str(coverage_result.coverage_tsv)
    payload["coverage"] = {
        "source": coverage_result.source,
        "requested_method": coverage_method,
        "semantic": "contig mean read depth for binning abundance consistency",
        "coverage_tsv": str(coverage_result.coverage_tsv),
        "coverage_bam": str(coverage_bam.resolve()) if coverage_bam is not None else None,
        "input_coverage_tsv": str(coverage_tsv.resolve()) if coverage_tsv is not None else None,
        "raw_output_tsv": (
            str(coverage_result.raw_output_tsv)
            if coverage_result.raw_output_tsv is not None
            else None
        ),
        "n_contigs": int(coverage_result.n_contigs),
        "n_missing_contigs": int(coverage_result.n_missing_contigs),
    }
    write_json(qc_json, payload)
