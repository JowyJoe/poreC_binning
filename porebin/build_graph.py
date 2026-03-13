from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from porebin import __version__
from porebin.contact_hypergraph import ContactHypergraphError, iter_canonical_contact_rows
from porebin.utils import ensure_dir, iter_fasta_names


class GraphBuildError(RuntimeError):
    pass


def order_norm(k: int, *, method: str = "pair") -> float:
    if k < 2:
        raise ValueError("k must be >= 2 for OrderNorm")
    if method == "pair":
        return 2.0 / (k * (k - 1))
    if method == "star":
        return 1.0 / (k - 1)
    raise ValueError(f"Unknown OrderNorm method: {method!r} (use 'pair' or 'star')")


@dataclass
class BuildStats:
    contacts_total: int = 0
    contacts_kept: int = 0
    contacts_skipped_k_lt_2: int = 0
    edges_written: int = 0
    k_counter: Counter[int] = field(default_factory=Counter)
    weight_total: float = 0.0
    weight_count: int = 0
    weight_min: Optional[float] = None
    weight_max: Optional[float] = None

    def add_weight(self, w: float) -> None:
        self.weight_total += w
        self.weight_count += 1
        self.weight_min = w if self.weight_min is None else min(self.weight_min, w)
        self.weight_max = w if self.weight_max is None else max(self.weight_max, w)

    @property
    def weight_mean(self) -> Optional[float]:
        if self.weight_count == 0:
            return None
        return self.weight_total / self.weight_count


def build_graph(
    *,
    contigs_fasta: Path,
    contacts_parquet: Path,
    out_dir: Path,
    order_norm_method: str = "pair",
    parquet_batch_size: int = 100_000,
    logger: Optional[logging.Logger] = None,
) -> Path:
    logger = logger or logging.getLogger("porebin")

    if not contigs_fasta.exists():
        raise FileNotFoundError(f"Contigs FASTA not found: {contigs_fasta}")
    if not contacts_parquet.exists():
        raise FileNotFoundError(f"Contacts Parquet not found: {contacts_parquet}")

    out_graph_dir = out_dir / "graph"
    ensure_dir(out_graph_dir)

    contig_index_path = out_graph_dir / "contig_index.tsv"
    contigs_tsv_path = out_graph_dir / "contigs.tsv"
    edges_path = out_graph_dir / "edges.tsv"
    contacts_meta_path = out_graph_dir / "contacts_meta.tsv"
    graph_meta_path = out_graph_dir / "graph_meta.json"

    contig_to_idx: dict[str, int] = {}
    contig_names: list[str] = []
    try:
        for name in iter_fasta_names(contigs_fasta):
            if not name:
                continue
            if name in contig_to_idx:
                raise GraphBuildError(f"Duplicate contig name in FASTA: {name}")
            contig_to_idx[name] = len(contig_names)
            contig_names.append(name)
    except ValueError as exc:
        raise GraphBuildError(str(exc)) from exc

    if not contig_names:
        raise GraphBuildError(f"No contigs found in FASTA: {contigs_fasta}")

    logger.info(f"Building graph from contacts: {contacts_parquet}")
    logger.info(f"Contigs in FASTA: {len(contig_names)}")

    stats = BuildStats()
    contact_idx = 0

    with edges_path.open("w", encoding="utf-8", newline="") as edges_fh, contacts_meta_path.open(
        "w", encoding="utf-8", newline=""
    ) as meta_fh:
        edges_fh.write("contig_idx\tcontact_idx\tedge_weight\n")
        meta_fh.write("contact_idx\tk_input\tk_valid\tweight\n")

        try:
            for row in iter_canonical_contact_rows(contacts_parquet, parquet_batch_size=parquet_batch_size):
                stats.contacts_total += 1
                if row.k_valid < 2:
                    stats.contacts_skipped_k_lt_2 += 1
                    continue

                try:
                    onorm = order_norm(row.k_valid, method=order_norm_method)
                except ValueError as exc:
                    raise GraphBuildError(f"Invalid k_valid={row.k_valid} in contacts.parquet: {exc}") from exc
                edge_weight_base = onorm * row.weight

                meta_fh.write(f"{contact_idx}\t{row.k_input}\t{row.k_valid}\t{row.weight:.10g}\n")
                stats.k_counter[row.k_valid] += 1
                stats.add_weight(row.weight)
                stats.contacts_kept += 1

                for j, contig in enumerate(row.contigs):
                    idx = contig_to_idx.get(contig)
                    if idx is None:
                        raise GraphBuildError(
                            f"Contig '{contig}' in contacts.parquet not found in FASTA '{contigs_fasta}'."
                        )
                    edge_weight = edge_weight_base
                    if row.contig_weights is not None:
                        edge_weight *= float(row.contig_weights[j])
                    edges_fh.write(f"{idx}\t{contact_idx}\t{edge_weight:.10g}\n")
                    stats.edges_written += 1

                contact_idx += 1
        except ContactHypergraphError as exc:
            raise GraphBuildError(str(exc)) from exc

    if stats.contacts_kept == 0:
        raise GraphBuildError("No usable contacts after filtering (k<2).")

    with contig_index_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("contig_name\tcontig_idx\n")
        for idx, name in enumerate(contig_names):
            fh.write(f"{name}\t{idx}\n")

    # v2 spectral clustering uses a stable contig index file without a header.
    # Format (required): contig_idx<TAB>contig_name
    with contigs_tsv_path.open("w", encoding="utf-8", newline="") as fh:
        for idx, name in enumerate(contig_names):
            fh.write(f"{idx}\t{name}\n")

    meta: dict[str, Any] = {
        "porebin_version": __version__,
        "input_contigs_fasta": str(contigs_fasta),
        "input_contacts_parquet": str(contacts_parquet),
        "order_norm_method": order_norm_method,
        "order_norm_formula": _order_norm_formula(order_norm_method),
        "contact_order_semantics": "k_valid_after_canonicalization",
        "contacts_meta_columns": ["contact_idx", "k_input", "k_valid", "weight"],
        "edge_weight_formula": (
            "edge_weight = OrderNorm(k_valid) * contact_weight"
            " * contig_weight (if contig_weights present in contacts.parquet)"
        ),
        "num_contigs": len(contig_names),
        "num_contacts": stats.contacts_kept,
        "num_edges": stats.edges_written,
        "contacts_total": stats.contacts_total,
        "contacts_skipped_k_lt_2": stats.contacts_skipped_k_lt_2,
        "k_distribution": {str(k): v for k, v in stats.k_counter.items()},
        "k_summary": _k_summary(stats.k_counter),
        "contact_weight_summary": {
            "count": stats.weight_count,
            "min": stats.weight_min,
            "max": stats.weight_max,
            "mean": stats.weight_mean,
        },
    }
    graph_meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    logger.info(
        f"Wrote graph: contigs={meta['num_contigs']}, contacts={meta['num_contacts']}, edges={meta['num_edges']}"
    )
    return out_graph_dir


def _k_summary(k_counter: Counter[int]) -> dict[str, Any]:
    if not k_counter:
        return {"min": None, "max": None, "mean": None}
    min_order = min(k_counter)
    max_order = max(k_counter)
    total = sum(k_counter.values())
    mean_order = sum(k * c for k, c in k_counter.items()) / total
    return {"min": min_order, "max": max_order, "mean": mean_order}


def _order_norm_formula(method: str) -> str:
    if method == "pair":
        return "2/(k*(k-1))  # == 1/C(k,2)"
    if method == "star":
        return "1/(k-1)"
    return "unknown"
