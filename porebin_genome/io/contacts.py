"""Parquet schema checks for canonical Pore-C contact evidence."""

from __future__ import annotations

from pathlib import Path

from porebin_genome.io.contracts import CONTACTS_PARQUET_CORE_FIELDS


class ContactContractError(RuntimeError):
    """Raised when a contacts.parquet file does not match the public contract."""


def validate_contacts_parquet_core_schema(path: Path) -> None:
    """Require the new tool's minimal public contact-evidence schema."""
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Contacts Parquet not found: {path}")
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover
        raise ContactContractError("Validating contacts.parquet requires pyarrow.") from exc

    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    columns = set(schema.names)
    missing = [field for field in CONTACTS_PARQUET_CORE_FIELDS if field not in columns]
    if missing:
        raise ContactContractError(
            f"contacts.parquet schema mismatch in {path}. Missing columns: {', '.join(missing)}"
        )
