"""Contact-hypergraph construction for genome-centric coarse bin discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from porebin_genome.evidence.canonical import ContactEvidenceError, iter_canonical_contacts


class CoarseContactError(RuntimeError):
    """Raised when contact incidence construction fails."""


@dataclass(frozen=True)
class ContactIncidence:
    """Sparse contact incidence matrix and its diagonal terms."""

    H_csr: "object"
    W: "object"
    De: "object"
    Dv: "object"
    contact_hyperedge_count: int
    dropped_singleton_contacts: int


def build_contact_incidence_from_parquet(
    contacts_path: Path,
    contig_name_to_idx: dict[str, int],
    *,
    parquet_batch_size: int = 200_000,
    logger: Optional[object] = None,
) -> ContactIncidence:
    """Build the contact hypergraph incidence from canonical contacts.parquet."""
    _ = logger
    contacts_path = contacts_path.resolve()
    if not contacts_path.exists():
        raise FileNotFoundError(f"Contacts Parquet not found: {contacts_path}")

    try:
        import numpy as np
        import scipy.sparse as sp
    except Exception as exc:  # pragma: no cover
        raise CoarseContactError("Contact incidence construction requires numpy and scipy.") from exc

    n_contigs = len(contig_name_to_idx)
    dropped = 0
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    weights: list[float] = []
    hyperedge_degree: list[float] = []
    edge_idx = 0

    try:
        for row in iter_canonical_contacts(
            contacts_path,
            parquet_batch_size=int(parquet_batch_size),
            require_contig_weights=True,
        ):
            if row.k_valid < 2:
                dropped += 1
                continue
            if row.contig_weights is None:
                raise CoarseContactError("Canonical contact row is missing contig_weights.")

            edge_weight = float(row.weight) / float(row.k_valid - 1)
            degree = 0.0
            edge_rows: list[int] = []
            edge_data: list[float] = []
            for contig_name, incidence_value in zip(row.contigs, row.contig_weights, strict=True):
                idx = contig_name_to_idx.get(contig_name)
                if idx is None:
                    raise CoarseContactError(
                        f"Contig {contig_name!r} in contacts.parquet is not present in contigs.fasta."
                    )
                value = float(incidence_value)
                edge_rows.append(int(idx))
                edge_data.append(value)
                degree += value

            if degree <= 0.0:
                dropped += 1
                continue

            rows.extend(edge_rows)
            cols.extend([edge_idx] * len(edge_rows))
            data.extend(edge_data)
            weights.append(float(edge_weight))
            hyperedge_degree.append(float(degree))
            edge_idx += 1
    except ContactEvidenceError as exc:
        raise CoarseContactError(str(exc)) from exc

    if not weights:
        raise CoarseContactError("No usable contact hyperedges were found in contacts.parquet.")

    H = sp.coo_matrix(
        (
            np.asarray(data, dtype=float),
            (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)),
        ),
        shape=(n_contigs, len(weights)),
    ).tocsr()
    W = np.asarray(weights, dtype=float)
    De = np.asarray(hyperedge_degree, dtype=float)
    Dv = np.asarray(H @ W).ravel()
    return ContactIncidence(
        H_csr=H,
        W=W,
        De=De,
        Dv=Dv,
        contact_hyperedge_count=int(len(weights)),
        dropped_singleton_contacts=int(dropped),
    )
