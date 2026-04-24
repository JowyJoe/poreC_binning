"""Public table contracts and default output layout for the new tool."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


CONTACTS_PARQUET_CORE_FIELDS = (
    "contact_id",
    "contigs",
    "contig_weights",
    "k",
    "k_eff",
    "weight",
)

COVERAGE_TSV_COLUMNS = (
    "contig_name",
    "coverage",
)

COARSE_BINS_COLUMNS = (
    "contig_name",
    "bin_id",
)

FINAL_BINS_COLUMNS = (
    "contig_id",
    "bin_id",
    "assignment_stage",
    "assignment_confidence",
    "assignment_reason",
)

UNBINNED_COLUMNS = (
    "contig_id",
    "stage",
    "reason",
    "source_bin",
    "note",
)

BIN_QC_COLUMNS = (
    "bin_id",
    "n_contigs",
    "total_length",
    "median_coverage",
    "contact_consistency",
    "suspect_flag",
    "refine_status",
    "notes",
)

REFINE_ACTIONS_COLUMNS = (
    "action_type",
    "contig_id",
    "bin_id",
    "source_bin",
    "target_bin",
    "reason",
    "accepted",
    "confidence",
    "note",
)


@dataclass(frozen=True)
class PipelineLayout:
    """Default directory and file layout for the genome-centric mainline."""

    root: Path
    evidence_dir: Path
    coarse_dir: Path
    final_dir: Path
    export_dir: Path
    contacts_parquet: Path
    coverage_tsv: Path
    evidence_qc_json: Path
    coarse_bins_tsv: Path
    coarse_run_json: Path
    refined_bins_tsv: Path
    unbinned_tsv: Path
    bin_qc_tsv: Path
    refine_actions_tsv: Path
    refine_meta_json: Path
    export_meta_json: Path
    bins_fasta_dir: Path
    unresolved_fasta: Path


def build_pipeline_layout(out_dir: Path) -> PipelineLayout:
    """Construct the default path layout under an output directory."""
    root = out_dir.resolve()
    evidence_dir = root / "evidence"
    coarse_dir = root / "coarse"
    final_dir = root / "final"
    export_dir = root / "export"
    return PipelineLayout(
        root=root,
        evidence_dir=evidence_dir,
        coarse_dir=coarse_dir,
        final_dir=final_dir,
        export_dir=export_dir,
        contacts_parquet=evidence_dir / "contacts.parquet",
        coverage_tsv=evidence_dir / "coverage.tsv",
        evidence_qc_json=evidence_dir / "evidence_qc.json",
        coarse_bins_tsv=coarse_dir / "bins.tsv",
        coarse_run_json=coarse_dir / "run.json",
        refined_bins_tsv=final_dir / "bins.refined.tsv",
        unbinned_tsv=final_dir / "unbinned.tsv",
        bin_qc_tsv=final_dir / "bin_qc.tsv",
        refine_actions_tsv=final_dir / "refine_actions.tsv",
        refine_meta_json=final_dir / "refine_meta.json",
        export_meta_json=export_dir / "export_meta.json",
        bins_fasta_dir=export_dir / "bins_fasta",
        unresolved_fasta=export_dir / "unresolved.fasta",
    )
