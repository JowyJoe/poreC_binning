from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_graph_dir(tmp_path: Path) -> Path:
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)

    (graph_dir / "contig_index.tsv").write_text(
        "contig_name\tcontig_idx\nA\t0\nB\t1\nC\t2\nD\t3\n",
        encoding="utf-8",
    )

    # 4 contacts: two connect (A,B), two connect (C,D)
    (graph_dir / "edges.tsv").write_text(
        "contig_idx\tcontact_idx\tedge_weight\n"
        "0\t0\t1.0\n"
        "1\t0\t1.0\n"
        "0\t1\t1.0\n"
        "1\t1\t1.0\n"
        "2\t2\t1.0\n"
        "3\t2\t1.0\n"
        "2\t3\t1.0\n"
        "3\t3\t1.0\n",
        encoding="utf-8",
    )

    meta = {"num_contigs": 4, "num_contacts": 4, "num_edges": 8}
    (graph_dir / "graph_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return graph_dir


def test_cluster_spectral_hypergraph_two_components(tmp_path: Path) -> None:
    pytest.importorskip("scipy", reason="spectral optional deps not installed")
    pytest.importorskip("sklearn", reason="spectral optional deps not installed")

    from porebin.cluster import cluster_spectral_hypergraph

    graph_dir = _write_graph_dir(tmp_path)
    out_bins = tmp_path / "bins.tsv"

    cluster_spectral_hypergraph(graph_dir=graph_dir, out_bins_tsv=out_bins, seed=0)

    mapping = {}
    for line in out_bins.read_text(encoding="utf-8").splitlines()[1:]:
        contig, bin_id = line.split("\t")
        mapping[contig] = bin_id

    assert mapping["A"] == mapping["B"]
    assert mapping["C"] == mapping["D"]
    assert mapping["A"] != mapping["C"]
