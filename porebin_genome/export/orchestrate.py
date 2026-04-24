"""FASTA export for final genome bins and unresolved contigs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from porebin_genome.io.contracts import FINAL_BINS_COLUMNS, UNBINNED_COLUMNS, build_pipeline_layout
from porebin_genome.io.fasta import iter_fasta_records
from porebin_genome.io.runtime import ensure_dir, write_json
from porebin_genome.io.tables import read_assignment_tsv, read_tsv_dict_rows


@dataclass(frozen=True)
class ExportRunResult:
    """Outputs emitted by the export layer."""

    bins_fasta_dir: Path
    unresolved_fasta: Path
    bins_exported: int
    unresolved_contigs: int


def export_final_results(
    *,
    contigs_fasta: Path,
    bins_refined_tsv: Path,
    unbinned_tsv: Path,
    out_dir: Path,
    logger: Optional[object] = None,
) -> ExportRunResult:
    """Export final bins as FASTA files and unresolved contigs as one FASTA."""
    _ = logger
    layout = build_pipeline_layout(out_dir)
    ensure_dir(layout.export_dir)
    ensure_dir(layout.bins_fasta_dir)

    if not contigs_fasta.exists():
        raise FileNotFoundError(f"Contigs FASTA not found: {contigs_fasta}")

    assignment = read_assignment_tsv(bins_refined_tsv, expected_header=FINAL_BINS_COLUMNS)
    unresolved_rows = read_tsv_dict_rows(unbinned_tsv, required_columns=UNBINNED_COLUMNS)
    unresolved_set = {str(row["contig_id"]).strip() for row in unresolved_rows}

    for existing in layout.bins_fasta_dir.glob("bin_*.fasta"):
        existing.unlink()
    if layout.unresolved_fasta.exists():
        layout.unresolved_fasta.unlink()

    bins_written: set[str] = set()
    unresolved_count = 0
    with layout.unresolved_fasta.open("w", encoding="utf-8", newline="") as unresolved_fh:
        for name, header, seq in iter_fasta_records(contigs_fasta):
            bin_id = assignment.get(name)
            if bin_id:
                out_path = layout.bins_fasta_dir / f"bin_{bin_id}.fasta"
                with out_path.open("a", encoding="utf-8", newline="") as fh:
                    _write_fasta_record(fh, header, seq)
                bins_written.add(bin_id)
                continue
            if name in unresolved_set:
                _write_fasta_record(unresolved_fh, header, seq)
                unresolved_count += 1

    write_json(
        layout.export_meta_json,
        {
            "stage": "export",
            "mode": "fasta_materialization",
            "inputs": {
                "contigs_fasta": str(contigs_fasta),
                "bins_refined_tsv": str(bins_refined_tsv),
                "unbinned_tsv": str(unbinned_tsv),
            },
            "outputs": {
                "bins_fasta_dir": str(layout.bins_fasta_dir),
                "unresolved_fasta": str(layout.unresolved_fasta),
            },
            "counts": {
                "bins_exported": len(bins_written),
                "unresolved_contigs": unresolved_count,
            },
        },
    )
    return ExportRunResult(
        bins_fasta_dir=layout.bins_fasta_dir,
        unresolved_fasta=layout.unresolved_fasta,
        bins_exported=len(bins_written),
        unresolved_contigs=unresolved_count,
    )


def _write_fasta_record(fh, header: str, seq: str, *, wrap: int = 80) -> None:
    fh.write(f">{header}\n")
    for start in range(0, len(seq), wrap):
        fh.write(seq[start : start + wrap] + "\n")
