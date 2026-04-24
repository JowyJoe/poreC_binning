"""Refine action export helpers."""

from __future__ import annotations

from pathlib import Path

from porebin_genome.io.contracts import REFINE_ACTIONS_COLUMNS
from porebin_genome.io.tables import write_tsv_rows
from porebin_genome.refine.models import RefineActionRow


def write_refine_actions_tsv(*, rows: list[RefineActionRow], out_path: Path) -> None:
    """Write final/refine_actions.tsv."""
    write_tsv_rows(
        out_path,
        REFINE_ACTIONS_COLUMNS,
        [
            (
                row.action_type,
                row.contig_id,
                row.bin_id,
                row.source_bin,
                row.target_bin,
                row.reason,
                int(row.accepted),
                f"{float(row.confidence):.6g}",
                row.note,
            )
            for row in rows
        ],
    )
