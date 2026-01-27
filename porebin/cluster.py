from __future__ import annotations

import csv
import gzip
import json
import logging
from pathlib import Path
from typing import Optional


class GraphClusterError(RuntimeError):
    pass


def cluster_leiden(
    *,
    graph_dir: Path,
    out_bins_tsv: Path,
    resolution: float = 1.0,
    seed: int = 0,
    logger: Optional[logging.Logger] = None,
) -> None:
    logger = logger or logging.getLogger("porebin")
    graph_dir = graph_dir.resolve()
    out_bins_tsv = out_bins_tsv.resolve()

    contig_names = _read_contig_index(graph_dir / "contig_index.tsv")
    if not contig_names:
        raise GraphClusterError(f"No contigs found in {graph_dir / 'contig_index.tsv'}")

    meta = _read_graph_meta(graph_dir / "graph_meta.json")
    num_contigs = len(contig_names)
    num_contacts = int(meta.get("num_contacts", 0))
    if num_contacts <= 0:
        raise GraphClusterError(f"Invalid num_contacts in {graph_dir / 'graph_meta.json'}: {num_contacts}")

    edges, weights = _read_edges(graph_dir / "edges.tsv", contig_offset=num_contigs)
    if not edges:
        raise GraphClusterError(f"No edges found in {graph_dir / 'edges.tsv'}")

    logger.info(
        f"Clustering bipartite graph: contigs={num_contigs}, contacts={num_contacts}, edges={len(edges)}"
    )

    try:
        import igraph as ig
        import leidenalg
    except Exception as exc:  # pragma: no cover
        raise GraphClusterError(
            "cluster requires 'python-igraph' and 'leidenalg'. Install them, e.g. pip install python-igraph leidenalg."
        ) from exc

    g = ig.Graph(n=num_contigs + num_contacts, edges=edges, directed=False)
    g.es["weight"] = weights
    g.vs["type"] = [False] * num_contigs + [True] * num_contacts

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
    )

    membership = partition.membership[:num_contigs]
    out_bins_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_bins_tsv.open("w", encoding="utf-8", newline="") as fh:
        fh.write("contig_name\tbin_id\n")
        for name, bin_id in zip(contig_names, membership, strict=True):
            fh.write(f"{name}\t{bin_id}\n")

    logger.info(f"Wrote bins: {out_bins_tsv}")


def cluster_leiden_pairwise(
    *,
    contig_index_tsv: Path,
    pairwise_edges_tsv_gz: Path,
    out_bins_tsv: Path,
    resolution: float = 1.0,
    seed: int = 0,
    logger: Optional[logging.Logger] = None,
) -> None:
    logger = logger or logging.getLogger("porebin")
    contig_index_tsv = contig_index_tsv.resolve()
    pairwise_edges_tsv_gz = pairwise_edges_tsv_gz.resolve()
    out_bins_tsv = out_bins_tsv.resolve()

    contig_names = _read_contig_index(contig_index_tsv)
    if not contig_names:
        raise GraphClusterError(f"No contigs found in {contig_index_tsv}")
    contig_to_idx = {name: i for i, name in enumerate(contig_names)}

    if not pairwise_edges_tsv_gz.exists():
        raise FileNotFoundError(f"Missing pairwise edges file: {pairwise_edges_tsv_gz}")

    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    with gzip.open(pairwise_edges_tsv_gz, "rt", encoding="utf-8", newline="") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = line.rstrip("\n").split("\t")
            if len(row) < 3:
                raise GraphClusterError(f"Invalid edge row in {pairwise_edges_tsv_gz}:{line_no}: {row}")
            a, b, w_s = row[0], row[1], row[2]
            if a == b:
                continue
            try:
                u = contig_to_idx[a]
            except KeyError as exc:
                raise GraphClusterError(
                    f"Contig '{a}' in {pairwise_edges_tsv_gz} not found in {contig_index_tsv}."
                ) from exc
            try:
                v = contig_to_idx[b]
            except KeyError as exc:
                raise GraphClusterError(
                    f"Contig '{b}' in {pairwise_edges_tsv_gz} not found in {contig_index_tsv}."
                ) from exc
            try:
                w = float(w_s)
            except ValueError as exc:
                raise GraphClusterError(f"Invalid edge weight in {pairwise_edges_tsv_gz}:{line_no}: {w_s!r}") from exc
            edges.append((u, v))
            weights.append(w)

    if not edges:
        raise GraphClusterError(f"No edges found in {pairwise_edges_tsv_gz}")

    logger.info(f"Clustering pairwise graph: contigs={len(contig_names)}, edges={len(edges)}")

    try:
        import igraph as ig
        import leidenalg
    except Exception as exc:  # pragma: no cover
        raise GraphClusterError(
            "cluster requires 'python-igraph' and 'leidenalg'. Install them, e.g. pip install python-igraph leidenalg."
        ) from exc

    g = ig.Graph(n=len(contig_names), edges=edges, directed=False)
    g.es["weight"] = weights

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=seed,
    )

    out_bins_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_bins_tsv.open("w", encoding="utf-8", newline="") as fh:
        fh.write("contig_name\tbin_id\n")
        for name, bin_id in zip(contig_names, partition.membership, strict=True):
            fh.write(f"{name}\t{bin_id}\n")

    logger.info(f"Wrote bins: {out_bins_tsv}")


def _read_graph_meta(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing graph meta file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_contig_index(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing contig index file: {path}")

    names_by_idx: dict[int, str] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if row[0] == "contig_name":
                continue
            if len(row) < 2:
                raise GraphClusterError(f"Invalid contig_index row in {path}: {row}")
            name, idx_s = row[0], row[1]
            try:
                idx = int(idx_s)
            except ValueError as exc:
                raise GraphClusterError(f"Invalid contig_idx in {path}: {idx_s!r}") from exc
            names_by_idx[idx] = name

    if not names_by_idx:
        return []
    max_idx = max(names_by_idx)
    contigs: list[str] = [""] * (max_idx + 1)
    for idx, name in names_by_idx.items():
        contigs[idx] = name
    if any(not n for n in contigs):
        raise GraphClusterError(f"Non-contiguous contig_idx values in {path}")
    return contigs


def _read_edges(path: Path, *, contig_offset: int) -> tuple[list[tuple[int, int]], list[float]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing edges file: {path}")

    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if row[0] == "contig_idx":
                continue
            if len(row) < 3:
                raise GraphClusterError(f"Invalid edge row in {path}: {row}")
            try:
                c_idx = int(row[0])
                contact_idx = int(row[1])
                w = float(row[2])
            except ValueError as exc:
                raise GraphClusterError(f"Invalid edge row in {path}: {row}") from exc
            edges.append((c_idx, contig_offset + contact_idx))
            weights.append(w)
    return edges, weights
