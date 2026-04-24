"""Evidence-layer orchestration for the genome-centric workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from porebin_genome.evidence.bam import bam_to_contact_evidence
from porebin_genome.io.contacts import validate_contacts_parquet_core_schema
from porebin_genome.io.contracts import build_pipeline_layout
from porebin_genome.io.coverage import validate_coverage_tsv


@dataclass(frozen=True)
class EvidenceRunResult:
    """Outputs produced by evidence construction."""

    contacts_parquet: Path
    coverage_tsv: Path
    qc_json: Path
    reads_kept: int


def run_evidence_from_bam(
    *,
    bam: Path,
    contigs_fasta: Path,
    out_dir: Path,
    parquet_batch_size: int = 10_000,
    logger: Optional[object] = None,
) -> EvidenceRunResult:
    """Run BAM-derived evidence construction and validate the resulting contracts."""
    _ = logger
    layout = build_pipeline_layout(out_dir)
    meta = bam_to_contact_evidence(
        bam=bam,
        contigs_fasta=contigs_fasta,
        out_dir=layout.root,
        parquet_batch_size=parquet_batch_size,
    )
    validate_contacts_parquet_core_schema(Path(meta["contacts_parquet"]))
    validate_coverage_tsv(Path(meta["coverage_tsv"]))
    return EvidenceRunResult(
        contacts_parquet=Path(meta["contacts_parquet"]).resolve(),
        coverage_tsv=Path(meta["coverage_tsv"]).resolve(),
        qc_json=Path(meta["qc_json"]).resolve(),
        reads_kept=int(meta["stats"]["reads_kept"]),
    )
