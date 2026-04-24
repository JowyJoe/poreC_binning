"""Coverage-table readers and validators."""

from __future__ import annotations

from pathlib import Path

from porebin_genome.io.contracts import COVERAGE_TSV_COLUMNS
from porebin_genome.io.tables import read_tsv_dict_rows


def read_coverage_tsv(path: Path) -> dict[str, float]:
    """Read `contig_name -> coverage` from a public coverage table."""
    rows = read_tsv_dict_rows(path, required_columns=COVERAGE_TSV_COLUMNS)
    out: dict[str, float] = {}
    for row in rows:
        name = str(row["contig_name"]).strip()
        try:
            cov = float(row["coverage"])
        except ValueError:
            continue
        out[name] = cov
    return out


def validate_coverage_tsv(path: Path) -> None:
    """Validate the public coverage-table contract."""
    _ = read_tsv_dict_rows(path, required_columns=COVERAGE_TSV_COLUMNS)
