"""Minimal QC writer retained as a lightweight public helper."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from porebin_genome.io.contracts import BIN_QC_COLUMNS
from porebin_genome.io.tables import write_tsv_rows


@dataclass(frozen=True)
class BinQcRow:
    """One minimal bin-level QC summary row."""

    bin_id: str
    n_contigs: int
    total_length: int
    median_coverage: str
    contact_consistency: str
    suspect_flag: bool
    refine_status: str
    notes: str


def write_bin_qc_summary(
    *,
    assignment: dict[str, str],
    contig_len: dict[str, int],
    out_path: Path,
) -> list[BinQcRow]:
    """Write a minimal genome-centric `bin_qc.tsv` summary."""
    total_length: Counter[str] = Counter()
    n_contigs: Counter[str] = Counter()
    for contig_id, bin_id in assignment.items():
        if not bin_id:
            continue
        total_length[str(bin_id)] += int(contig_len.get(contig_id, 0))
        n_contigs[str(bin_id)] += 1

    rows: list[BinQcRow] = []
    for bin_id in sorted(total_length.keys(), key=str):
        rows.append(
            BinQcRow(
                bin_id=bin_id,
                n_contigs=int(n_contigs[bin_id]),
                total_length=int(total_length[bin_id]),
                median_coverage="NA",
                contact_consistency="NA",
                suspect_flag=False,
                refine_status="not_evaluated",
                notes="minimal_qc_summary",
            )
        )

    write_tsv_rows(
        out_path,
        BIN_QC_COLUMNS,
        [
            (
                row.bin_id,
                row.n_contigs,
                row.total_length,
                row.median_coverage,
                row.contact_consistency,
                int(row.suspect_flag),
                row.refine_status,
                row.notes,
            )
            for row in rows
        ],
    )
    return rows
