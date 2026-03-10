from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_many_bins_not_limited(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for contacts.parquet")
    pytest.importorskip("scipy", reason="spectral optional deps not installed")
    pytest.importorskip("sklearn", reason="spectral v2 features require scikit-learn")
    pytest.importorskip("hdbscan", reason="spectral v2 clustering requires hdbscan")

    import pyarrow as pa
    import pyarrow.parquet as pq

    from porebin.cluster import cluster_spectral_hypergraph

    num_clusters = 100
    cluster_size = 5
    V = num_clusters * cluster_size

    contigs = [f"c{i}" for i in range(V)]

    graph_dir = tmp_path / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    (graph_dir / "contigs.tsv").write_text(
        "".join([f"{i}\t{name}\n" for i, name in enumerate(contigs)]),
        encoding="utf-8",
    )

    def _motif_for_cluster(g: int, *, length: int = 16) -> str:
        # Deterministic pseudo-random motif per cluster to avoid distance ties in kNN.
        x = (g + 1) * 2654435761  # Knuth multiplicative hash seed
        bases = "ACGT"
        out = []
        for _ in range(length):
            x = (1103515245 * x + 12345) & 0x7FFFFFFF
            out.append(bases[(x >> 16) & 3])
        return "".join(out)

    contigs_fasta = tmp_path / "contigs.fasta"
    with contigs_fasta.open("w", encoding="utf-8", newline="") as fh:
        for g in range(num_clusters):
            motif = _motif_for_cluster(g)
            seq = motif * 400  # enough windows for stable freq
            for j in range(cluster_size):
                idx = g * cluster_size + j
                fh.write(f">{contigs[idx]}\n{seq}\n")

    # contacts.parquet: per cluster generate 10 reads, each picks 3 contigs from that cluster.
    rows_contigs: list[list[str]] = []
    rows_p: list[list[float]] = []
    rows_k: list[int] = []
    rows_q: list[float] = []
    for g in range(num_clusters):
        base = g * cluster_size
        for t in range(10):
            i0 = base + (t % cluster_size)
            i1 = base + ((t + 1) % cluster_size)
            i2 = base + ((t + 2) % cluster_size)
            rows_contigs.append([contigs[i0], contigs[i1], contigs[i2]])
            rows_p.append([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
            rows_k.append(3)
            rows_q.append(1.0)

    contacts_parquet = tmp_path / "contacts.parquet"
    pq.write_table(
        pa.Table.from_pydict(
            {"contigs": rows_contigs, "contig_weights": rows_p, "k": rows_k, "weight": rows_q}
        ),
        contacts_parquet,
    )

    (graph_dir / "graph_meta.json").write_text(
        json.dumps(
            {"num_contigs": V, "input_contigs_fasta": str(contigs_fasta), "input_contacts_parquet": str(contacts_parquet)},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    out_bins = tmp_path / "bins.tsv"
    meta = cluster_spectral_hypergraph(graph_dir=graph_dir, out_bins_tsv=out_bins, seed=0, threads=1)

    assert meta["num_bins"] >= 80
    assert meta["unbinned_count"] <= 20
