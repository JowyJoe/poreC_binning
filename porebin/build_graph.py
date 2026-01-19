from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from porebin import __version__
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

    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover
        raise GraphBuildError(
            "build requires 'pyarrow' to read contacts.parquet. Install it, e.g. pip install pyarrow."
        ) from exc

    stats = BuildStats()
    contact_idx = 0

    with edges_path.open("w", encoding="utf-8", newline="") as edges_fh, contacts_meta_path.open(
        "w", encoding="utf-8", newline=""
    ) as meta_fh:
        edges_fh.write("contig_idx\tcontact_idx\tedge_weight\n")
        meta_fh.write("contact_idx\tk\tweight\n")

        parquet = pq.ParquetFile(contacts_parquet)
        schema = parquet.schema_arrow
        cols = set(schema.names)
        if "contigs" not in cols:
            raise GraphBuildError(f"contacts.parquet missing required column 'contigs'. Found: {schema.names}")
        has_k = "k" in cols
        has_weight = "weight" in cols
        read_cols = ["contigs"] + (["k"] if has_k else []) + (["weight"] if has_weight else [])

        for batch in parquet.iter_batches(batch_size=parquet_batch_size, columns=read_cols):
            data = batch.to_pydict()
            contigs_list = data["contigs"]
            k_list = data.get("k")
            w_list = data.get("weight")
            n = len(contigs_list)
            for i in range(n):
                stats.contacts_total += 1
                contigs = contigs_list[i] or []
                if not isinstance(contigs, list):
                    raise GraphBuildError(
                        f"Invalid contigs type in contacts.parquet (expected list) at row {stats.contacts_total}"
                    )
                contigs = [str(c) for c in contigs if str(c)]
                contigs = _dedupe(contigs)
                k = int(k_list[i]) if k_list is not None and k_list[i] is not None else len(contigs)
                weight = float(w_list[i]) if w_list is not None and w_list[i] is not None else 1.0

                if k < 2 or len(contigs) < 2:
                    stats.contacts_skipped_k_lt_2 += 1
                    continue

                try:
                    onorm = order_norm(k, method=order_norm_method)
                except ValueError as exc:
                    raise GraphBuildError(f"Invalid k={k} in contacts.parquet: {exc}") from exc
                edge_weight = onorm * weight

                meta_fh.write(f"{contact_idx}\t{k}\t{weight:.10g}\n")
                stats.k_counter[k] += 1
                stats.add_weight(weight)
                stats.contacts_kept += 1

                for contig in contigs:
                    idx = contig_to_idx.get(contig)
                    if idx is None:
                        raise GraphBuildError(
                            f"Contig '{contig}' in contacts.parquet not found in FASTA '{contigs_fasta}'."
                        )
                    edges_fh.write(f"{idx}\t{contact_idx}\t{edge_weight:.10g}\n")
                    stats.edges_written += 1

                contact_idx += 1

    if stats.contacts_kept == 0:
        raise GraphBuildError("No usable contacts after filtering (k<2).")

    with contig_index_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("contig_name\tcontig_idx\n")
        for idx, name in enumerate(contig_names):
            fh.write(f"{name}\t{idx}\n")

    meta: dict[str, Any] = {
        "porebin_version": __version__,
        "input_contigs_fasta": str(contigs_fasta),
        "input_contacts_parquet": str(contacts_parquet),
        "order_norm_method": order_norm_method,
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


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _k_summary(k_counter: Counter[int]) -> dict[str, Any]:
    if not k_counter:
        return {"min": None, "max": None, "mean": None}
    min_k = min(k_counter)
    max_k = max(k_counter)
    total = sum(k_counter.values())
    mean_k = sum(k * c for k, c in k_counter.items()) / total
    return {"min": min_k, "max": max_k, "mean": mean_k}
