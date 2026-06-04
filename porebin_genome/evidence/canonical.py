"""Canonical contact-row semantics for public Pore-C evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


class ContactEvidenceError(RuntimeError):
    """Raised when canonical contact evidence is malformed."""


@dataclass(frozen=True)
class CanonicalContact:
    """One canonicalized Pore-C contact row."""

    contact_id: int | None
    contigs: list[str]
    contig_weights: Optional[list[float]]
    k_input: int
    k_valid: int
    weight: float


def canonicalize_contact(
    *,
    contigs_raw: object,
    contig_weights_raw: object = None,
    contact_id_raw: object = None,
    k_raw: object = None,
    weight_raw: object = None,
    row_number: Optional[int] = None,
) -> CanonicalContact:
    """Canonicalize one raw contact row into a stable public representation."""
    label = f"row {row_number}" if row_number is not None else "contact row"

    if not isinstance(contigs_raw, list):
        raise ContactEvidenceError(f"Invalid contigs type in {label}: expected list, got {type(contigs_raw)!r}")

    nonempty_contigs = [str(c) for c in contigs_raw if str(c)]
    contigs: list[str] = []
    contig_weights: Optional[list[float]] = None

    if contig_weights_raw is not None:
        if not isinstance(contig_weights_raw, list):
            raise ContactEvidenceError(
                f"Invalid contig_weights type in {label}: expected list, got {type(contig_weights_raw)!r}"
            )
        if len(contig_weights_raw) != len(contigs_raw):
            raise ContactEvidenceError(
                f"Mismatched contigs/contig_weights lengths in {label}: "
                f"len(contigs)={len(contigs_raw)} len(contig_weights)={len(contig_weights_raw)}"
            )

        seen: dict[str, int] = {}
        uniq_contigs: list[str] = []
        uniq_w: list[float] = []
        for contig_name, raw_weight in zip(contigs_raw, contig_weights_raw, strict=True):
            name = str(contig_name)
            if not name:
                continue
            try:
                weight = float(raw_weight)
            except Exception as exc:
                raise ContactEvidenceError(
                    f"Invalid contig_weight in {label}: contig={name!r} value={raw_weight!r}"
                ) from exc
            if weight < 0.0:
                raise ContactEvidenceError(f"Negative contig_weight in {label}: contig={name!r} w={weight}")
            if weight == 0.0:
                continue
            if name in seen:
                uniq_w[seen[name]] += weight
            else:
                seen[name] = len(uniq_contigs)
                uniq_contigs.append(name)
                uniq_w.append(weight)

        total = float(sum(uniq_w))
        if total > 0.0:
            contigs = uniq_contigs
            contig_weights = [float(value) / total for value in uniq_w]
        else:
            contigs = []
            contig_weights = []
    else:
        seen_names: set[str] = set()
        for name in nonempty_contigs:
            if name in seen_names:
                continue
            seen_names.add(name)
            contigs.append(name)

    try:
        k_input = int(k_raw) if k_raw is not None else len(nonempty_contigs)
    except Exception as exc:
        raise ContactEvidenceError(f"Invalid k value in {label}: {k_raw!r}") from exc

    try:
        weight = float(weight_raw) if weight_raw is not None else 1.0
    except Exception as exc:
        raise ContactEvidenceError(f"Invalid weight value in {label}: {weight_raw!r}") from exc

    contact_id: int | None
    if contact_id_raw is None:
        contact_id = (int(row_number) - 1) if row_number is not None else None
    else:
        try:
            contact_id = int(contact_id_raw)
        except Exception as exc:
            raise ContactEvidenceError(f"Invalid contact_id value in {label}: {contact_id_raw!r}") from exc

    return CanonicalContact(
        contact_id=contact_id,
        contigs=contigs,
        contig_weights=contig_weights,
        k_input=k_input,
        k_valid=len(contigs),
        weight=weight,
    )


def iter_canonical_contacts(
    contacts_path: Path,
    *,
    parquet_batch_size: int = 200_000,
    require_contig_weights: bool = False,
) -> Iterator[CanonicalContact]:
    """Stream canonical contact rows from `contacts.parquet`."""
    contacts_path = contacts_path.resolve()
    if not contacts_path.exists():
        raise FileNotFoundError(f"Contacts Parquet not found: {contacts_path}")

    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover
        raise ContactEvidenceError("Reading contacts.parquet requires pyarrow.") from exc

    parquet = pq.ParquetFile(contacts_path)
    schema = parquet.schema_arrow
    columns = set(schema.names)
    if "contigs" not in columns:
        raise ContactEvidenceError(f"contacts.parquet missing required column 'contigs'. Found: {schema.names}")
    if require_contig_weights and "contig_weights" not in columns:
        raise ContactEvidenceError(
            f"contacts.parquet missing required column 'contig_weights'. Found: {schema.names}"
        )

    has_contig_weights = "contig_weights" in columns
    has_contact_id = "contact_id" in columns
    has_k = "k" in columns
    has_weight = "weight" in columns
    read_columns = (
        ["contigs"]
        + (["contact_id"] if has_contact_id else [])
        + (["contig_weights"] if has_contig_weights else [])
        + (["k"] if has_k else [])
        + (["weight"] if has_weight else [])
    )

    row_number = 0
    for batch in parquet.iter_batches(batch_size=int(parquet_batch_size), columns=read_columns):
        data = batch.to_pydict()
        contigs_list = data["contigs"]
        contact_id_list = data.get("contact_id")
        contig_weights_list = data.get("contig_weights")
        k_list = data.get("k")
        weight_list = data.get("weight")
        for idx in range(len(contigs_list)):
            row_number += 1
            yield canonicalize_contact(
                contigs_raw=contigs_list[idx] or [],
                contact_id_raw=(contact_id_list[idx] if contact_id_list is not None else None),
                contig_weights_raw=(contig_weights_list[idx] or []) if contig_weights_list is not None else None,
                k_raw=(k_list[idx] if k_list is not None else None),
                weight_raw=(weight_list[idx] if weight_list is not None else None),
                row_number=row_number,
            )
