"""TSV contract readers and writers for public result tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence


class TableContractError(RuntimeError):
    """Raised when a public TSV does not match the declared contract."""


def write_tsv_rows(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    """Write a TSV file with a fixed header and row sequence."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(["" if value is None else str(value) for value in row])


def read_assignment_tsv(path: Path, expected_header: Sequence[str]) -> dict[str, str]:
    """Read a `contig_name -> bin_id` style TSV and validate its header."""
    rows = read_tsv_dict_rows(path, required_columns=expected_header)
    mapping: dict[str, str] = {}
    left = expected_header[0]
    right = expected_header[1]
    for row in rows:
        mapping[str(row[left]).strip()] = str(row[right]).strip()
    return mapping


def read_tsv_dict_rows(path: Path, required_columns: Sequence[str]) -> list[dict[str, str]]:
    """Read a TSV into dictionaries and require a specific public header subset."""
    path = path.resolve()
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if reader.fieldnames is None:
            raise TableContractError(f"Missing TSV header in {path}")
        fieldnames = tuple(str(name) for name in reader.fieldnames if name is not None)
        missing = [column for column in required_columns if column not in fieldnames]
        if missing:
            raise TableContractError(
                f"TSV schema mismatch in {path}. Missing columns: {', '.join(missing)}"
            )
        rows: list[dict[str, str]] = []
        for row in reader:
            if not row:
                continue
            rows.append({str(k): str(v).strip() for k, v in row.items() if k is not None})
        return rows


def validate_tsv_header(path: Path, expected_header: Sequence[str]) -> None:
    """Validate that a TSV file contains the expected header columns."""
    _ = read_tsv_dict_rows(path, required_columns=expected_header)
