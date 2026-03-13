from __future__ import annotations

from pathlib import Path

import pytest


def test_canonicalize_contact_row_merges_duplicates_and_uses_k_valid() -> None:
    from porebin.contact_hypergraph import canonicalize_contact_row

    row = canonicalize_contact_row(
        contigs_raw=["c0", "c1", "c1", "", "c2"],
        contig_weights_raw=[0.2, 0.3, 0.1, 0.4, 0.0],
        k_raw=5,
        weight_raw=0.8,
        row_number=1,
    )

    assert row.contigs == ["c0", "c1"]
    assert row.contig_weights is not None
    assert row.contig_weights == pytest.approx([1.0 / 3.0, 2.0 / 3.0])
    assert row.k_input == 5
    assert row.k_valid == 2
    assert row.weight == pytest.approx(0.8)


def test_build_graph_and_spectral_share_canonical_contact_rows(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for contacts.parquet")
    pytest.importorskip("scipy", reason="spectral optional deps not installed")

    import pyarrow as pa
    import pyarrow.parquet as pq

    from porebin.build_graph import build_graph
    from porebin.hypergraph_joint_spectral import build_contact_incidence_from_parquet

    contigs_fasta = tmp_path / "contigs.fasta"
    contigs_fasta.write_text(">c0\nAAAA\n>c1\nCCCC\n>c2\nGGGG\n", encoding="utf-8")

    contacts_parquet = tmp_path / "contacts.parquet"
    pq.write_table(
        pa.Table.from_pydict(
            {
                "contigs": [["c0", "c1", "c1", "", "c2"]],
                "contig_weights": [[0.2, 0.3, 0.1, 0.4, 0.0]],
                "k": [5],
                "weight": [0.8],
            }
        ),
        contacts_parquet,
    )

    out_dir = tmp_path / "out"
    graph_dir = build_graph(contigs_fasta=contigs_fasta, contacts_parquet=contacts_parquet, out_dir=out_dir)

    meta_lines = (graph_dir / "contacts_meta.tsv").read_text(encoding="utf-8").splitlines()
    assert meta_lines[0] == "contact_idx\tk_input\tk_valid\tweight"
    assert meta_lines[1] == "0\t5\t2\t0.8"

    edges = {}
    for line in (graph_dir / "edges.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        c_idx, contact_idx, weight = line.split("\t")
        edges[(int(c_idx), int(contact_idx))] = float(weight)
    # pair norm with k_valid=2 gives OrderNorm=1, so edge weight is q * pi.
    assert edges == {
        (0, 0): pytest.approx(0.8 * (1.0 / 3.0)),
        (1, 0): pytest.approx(0.8 * (2.0 / 3.0)),
    }

    contig_name_to_idx = {"c0": 0, "c1": 1, "c2": 2}
    contact = build_contact_incidence_from_parquet(contacts_parquet, contig_name_to_idx)
    assert contact.dropped_edges_singleton_contact == 0
    assert contact.H_csr.shape == (3, 1)
    assert contact.W.tolist() == pytest.approx([0.8])
    assert contact.De.tolist() == pytest.approx([1.0])
    assert contact.H_csr[:, 0].toarray().ravel().tolist() == pytest.approx([1.0 / 3.0, 2.0 / 3.0, 0.0])
