"""Bin-level QC export for genome-centric refine MVP."""

from __future__ import annotations

from pathlib import Path

from porebin_genome.io.contracts import BIN_QC_COLUMNS
from porebin_genome.io.tables import write_tsv_rows
from porebin_genome.refine.models import BinQcRow, BinSnapshot


def build_bin_qc_rows(*, snapshots: dict[str, BinSnapshot]) -> list[BinQcRow]:
    """Convert bin snapshots into final/bin_qc.tsv rows."""
    rows: list[BinQcRow] = []
    for bin_id in sorted(snapshots.keys(), key=str):
        snapshot = snapshots[bin_id]
        rows.append(
            BinQcRow(
                bin_id=bin_id,
                n_contigs=int(snapshot.n_contigs),
                total_length=int(snapshot.total_length),
                median_coverage=(
                    "NA"
                    if snapshot.median_coverage is None
                    else f"{float(snapshot.median_coverage):.6g}"
                ),
                contact_coherence=float(snapshot.contact_coherence),
                scg_status=str(snapshot.scg_status),
                scg_duplicate_marker_count=int(snapshot.scg_duplicate_marker_count),
                suspect_flag=bool(snapshot.suspect_flag),
                refine_status=str(snapshot.refine_status),
                notes=",".join(snapshot.suspect_reasons) if snapshot.suspect_reasons else "stable_after_refine",
            )
        )
    return rows


def write_bin_qc_tsv(*, rows: list[BinQcRow], out_path: Path) -> None:
    """Write final/bin_qc.tsv."""
    write_tsv_rows(
        out_path,
        BIN_QC_COLUMNS,
        [
            (
                row.bin_id,
                row.n_contigs,
                row.total_length,
                row.median_coverage,
                f"{float(row.contact_coherence):.6g}",
                row.scg_status,
                row.scg_duplicate_marker_count,
                int(row.suspect_flag),
                row.refine_status,
                row.notes,
            )
            for row in rows
        ],
    )
