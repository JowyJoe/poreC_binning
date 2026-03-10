from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_graph_dir(tmp_path: Path) -> Path:
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)

    contigs_fasta = tmp_path / "contigs.fasta"
    # Two clusters with >= min_cluster_size (5) contigs each.
    # Use distinct composition to help feature hypergraph as well.
    seq_a = ("ACGTACGTACGTACGA" * 300)
    seq_c = ("TTGCGGATTTGCGGAA" * 300)
    contigs_fasta.write_text(
        "".join([f">A{i}\n{seq_a}\n" for i in range(5)] + [f">C{i}\n{seq_c}\n" for i in range(5)]),
        encoding="utf-8",
    )

    # v2 spectral prefers graph/contigs.tsv (no header): contig_idx<TAB>contig_name
    contigs = [f"A{i}" for i in range(5)] + [f"C{i}" for i in range(5)]
    (graph_dir / "contigs.tsv").write_text(
        "".join([f"{i}\t{name}\n" for i, name in enumerate(contigs)]),
        encoding="utf-8",
    )

    pytest.importorskip("pyarrow", reason="pyarrow required for contacts.parquet")
    import pyarrow.parquet as pq
    import pyarrow as pa

    # Synthetic contacts.parquet: contacts only within each 5-contig cluster (k=3 reads).
    rows_contigs = []
    rows_p = []
    rows_k = []
    rows_q = []
    for base in (0, 5):
        for t in range(12):
            i0 = base + (t % 5)
            i1 = base + ((t + 1) % 5)
            i2 = base + ((t + 2) % 5)
            rows_contigs.append([contigs[i0], contigs[i1], contigs[i2]])
            rows_p.append([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
            rows_k.append(3)
            rows_q.append(1.0)

    contacts_parquet = tmp_path / "contacts.parquet"
    table = pa.Table.from_pydict(
        {"contigs": rows_contigs, "contig_weights": rows_p, "k": rows_k, "weight": rows_q}
    )
    pq.write_table(table, contacts_parquet)

    meta = {
        "num_contigs": 10,
        "input_contigs_fasta": str(contigs_fasta),
        "input_contacts_parquet": str(contacts_parquet),
    }
    (graph_dir / "graph_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return graph_dir


def test_cluster_spectral_hypergraph_two_components(tmp_path: Path) -> None:
    pytest.importorskip("scipy", reason="spectral optional deps not installed")
    pytest.importorskip("hdbscan", reason="spectral v2 clustering requires hdbscan")
    pytest.importorskip("sklearn", reason="spectral v2 clustering requires scikit-learn")

    from porebin.cluster import cluster_spectral_hypergraph

    graph_dir = _write_graph_dir(tmp_path)
    out_bins = tmp_path / "bins.tsv"

    cluster_spectral_hypergraph(graph_dir=graph_dir, out_bins_tsv=out_bins, seed=0)

    mapping = {}
    for line in out_bins.read_text(encoding="utf-8").splitlines()[1:]:
        contig, bin_id = line.split("\t")
        mapping[contig] = bin_id

    # Cluster labels should separate A* and C* groups.
    a_label = {mapping[f"A{i}"] for i in range(5)}
    c_label = {mapping[f"C{i}"] for i in range(5)}
    assert len(a_label) == 1
    assert len(c_label) == 1
    assert next(iter(a_label)) != next(iter(c_label))
