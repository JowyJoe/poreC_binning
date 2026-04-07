from __future__ import annotations

from pathlib import Path

import pytest


def test_compute_bin_scg_qc_counts_duplicates() -> None:
    from porebin.scg import ScgHit, compute_bin_scg_qc

    hits = [
        ScgHit("c1", "g1", "SCG_A", 100.0, 1e-20, 1, 10, 1, 10, 1, 10, 10, 100, "+"),
        ScgHit("c2", "g2", "SCG_A", 99.0, 1e-19, 1, 10, 1, 10, 1, 10, 20, 120, "+"),
        ScgHit("c2", "g3", "SCG_B", 98.0, 1e-18, 1, 10, 1, 10, 1, 10, 130, 240, "+"),
        ScgHit("c3", "g4", "SCG_A", 97.0, 1e-17, 1, 10, 1, 10, 1, 10, 30, 160, "+"),
    ]
    qc = compute_bin_scg_qc(
        contig_to_bin={"c1": "0", "c2": "0", "c3": "1"},
        hits=hits,
        expected_markers={"SCG_A", "SCG_B", "SCG_C", "SCG_D"},
    )

    assert qc["0"].unique_markers == 2
    assert qc["0"].duplicated_markers == 1
    assert qc["0"].completeness_like == pytest.approx(0.5)
    assert qc["0"].contamination_like == pytest.approx(0.25)
    assert set(qc["0"].implicated_contigs) == {"c1", "c2"}
    assert qc["1"].unique_markers == 1
    assert qc["1"].duplicated_markers == 0


def test_ensure_scg_hits_reuses_cache_and_keeps_expected_markers(tmp_path: Path) -> None:
    from porebin.scg import ensure_scg_hits

    contigs_fasta = tmp_path / "contigs.fasta"
    contigs_fasta.write_text(">c1\nACGTACGT\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    scg_dir = out_dir / "scg"
    scg_dir.mkdir(parents=True, exist_ok=True)
    (scg_dir / "scg_hits.tsv").write_text(
        "contig_name\tgene_id\tmarker_id\tbitscore\tevalue\thmm_from\thmm_to\tali_from\tali_to\t"
        "env_from\tenv_to\torf_start\torf_end\tstrand\n",
        encoding="utf-8",
    )

    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    (db_dir / "core_bacterial_scg.hmm").write_text("HMMER3/f\nNAME  SCG_A\n//\n", encoding="utf-8")
    (db_dir / "manifest.json").write_text(
        (
            '{"marker_set_id":"test_scg","db_version":"0.1","marker_hmm":"core_bacterial_scg.hmm",'
            '"expected_markers":["SCG_A","SCG_B"]}'
        ),
        encoding="utf-8",
    )

    result = ensure_scg_hits(contigs_fasta=contigs_fasta, out_dir=out_dir, db_dir=db_dir)
    assert result.enabled is True
    assert result.reused_cache is True
    assert result.state == "enabled"
    assert result.expected_markers == ("SCG_A", "SCG_B")
    assert result.resources is not None
    assert result.resources.marker_hmm.name == "core_bacterial_scg.hmm"


def test_ensure_scg_hits_cached_without_resources_is_partial(tmp_path: Path) -> None:
    from porebin.scg import ensure_scg_hits

    contigs_fasta = tmp_path / "contigs.fasta"
    contigs_fasta.write_text(">c1\nACGTACGT\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    scg_dir = out_dir / "scg"
    scg_dir.mkdir(parents=True, exist_ok=True)
    (scg_dir / "scg_hits.tsv").write_text(
        "contig_name\tgene_id\tmarker_id\tbitscore\tevalue\thmm_from\thmm_to\tali_from\tali_to\t"
        "env_from\tenv_to\torf_start\torf_end\tstrand\n",
        encoding="utf-8",
    )

    result = ensure_scg_hits(
        contigs_fasta=contigs_fasta,
        out_dir=out_dir,
        db_dir=tmp_path / "missing_db",
    )
    assert result.enabled is True
    assert result.reused_cache is True
    assert result.state == "partial"
    assert result.expected_markers == ()
    assert "cached_hits_without_resources" in result.reason


def test_compute_contact_qc_and_suspects(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for contacts.parquet")
    import pyarrow as pa
    import pyarrow.parquet as pq

    from porebin.refine_qc import compute_bin_qc_rows, compute_contact_component_qc, select_suspect_bins
    from porebin.scg import BinScgQc

    contacts_parquet = tmp_path / "contacts.parquet"
    pq.write_table(
        pa.table(
            {
                "contigs": pa.array(
                    [["a1", "a2"], ["a3", "a4"], ["b1", "b2"]],
                    type=pa.list_(pa.string()),
                ),
                "contig_weights": pa.array(
                    [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
                    type=pa.list_(pa.float64()),
                ),
                "weight": pa.array([1.0, 1.0, 1.0], type=pa.float64()),
            }
        ),
        contacts_parquet,
    )

    contig_to_bin = {"a1": "0", "a2": "0", "a3": "0", "a4": "0", "b1": "1", "b2": "1"}
    contig_len = {c: 5000 for c in contig_to_bin}
    contact_qc = compute_contact_component_qc(
        contacts_parquet=contacts_parquet,
        contig_to_bin=contig_to_bin,
        contig_len=contig_len,
        min_contig_len=1000,
    )
    rows = compute_bin_qc_rows(
        contig_to_bin=contig_to_bin,
        contig_len=contig_len,
        intra_support={"a1": 1.0, "a2": 1.0, "a3": 0.1, "a4": 0.1, "b1": 1.0, "b2": 1.0},
        other_support={"a1": 0.0, "a2": 0.0, "a3": 0.2, "a4": 0.2, "b1": 0.0, "b2": 0.0},
        bin_cov_stats=None,
        contact_component_qc=contact_qc,
        bin_scg_qc={
            "0": BinScgQc("0", 4, 2, 1, 3, 0.5, 0.25, ("a1", "a3")),
            "1": BinScgQc("1", 4, 2, 0, 2, 0.5, 0.0, ()),
        },
        scg_status="enabled",
        scg_expected_markers_count=4,
    )
    suspects = select_suspect_bins(rows)

    suspect_ids = {rec.bin_id for rec in suspects}
    assert suspect_ids == {"0"}
