from __future__ import annotations

from pathlib import Path

import pytest

from porebin_genome.evidence.scg.discovery import (
    ScgHit,
    collapse_scg_hits_to_contigs,
    ensure_scg_toolchain_available,
    parse_scg_hits,
    prepare_scg_profiles,
)


def test_scg_toolchain_dependency_error_is_explicit() -> None:
    with pytest.raises(RuntimeError, match="requires external dependencies"):
        ensure_scg_toolchain_available(
            prodigal_executable="__missing_prodigal__",
            hmmsearch_executable="__missing_hmmsearch__",
        )


def test_collapse_scg_hits_deduplicates_same_marker_per_contig() -> None:
    profiles = collapse_scg_hits_to_contigs(
        [
            ScgHit(
                orf_id="orf1",
                contig_id="c1",
                marker_id="Ribosomal_L23",
                domain="bacteria",
                bitscore=100.0,
                evalue=1e-20,
                hmm_coverage=0.90,
            ),
            ScgHit(
                orf_id="orf2",
                contig_id="c1",
                marker_id="Ribosomal_L23",
                domain="bacteria",
                bitscore=90.0,
                evalue=1e-10,
                hmm_coverage=0.85,
            ),
            ScgHit(
                orf_id="orf3",
                contig_id="c1",
                marker_id="Ribosomal_S9",
                domain="bacteria",
                bitscore=95.0,
                evalue=1e-18,
                hmm_coverage=0.88,
            ),
        ]
    )
    assert set(profiles["c1"].marker_ids) == {"Ribosomal_L23", "Ribosomal_S9"}
    assert profiles["c1"].n_markers == 2


def test_collapse_scg_hits_canonicalizes_aliases_before_deduplication() -> None:
    profiles = collapse_scg_hits_to_contigs(
        [
            ScgHit(
                orf_id="orf1",
                contig_id="c1",
                marker_id="TIGR00388",
                domain="bacteria",
                bitscore=100.0,
                evalue=1e-20,
                hmm_coverage=0.90,
            ),
            ScgHit(
                orf_id="orf2",
                contig_id="c1",
                marker_id="TIGR00389",
                domain="bacteria",
                bitscore=90.0,
                evalue=1e-10,
                hmm_coverage=0.85,
            ),
        ]
    )

    assert profiles["c1"].marker_ids == ("TIGR00389",)
    assert profiles["c1"].n_markers == 1
    assert profiles["c1"].marker_orf_counts == {"TIGR00389": 2}


def test_parse_scg_hits_maps_prodigal_faa_orf_id_to_gff_contig(
    tmp_path: Path,
) -> None:
    prodigal_gff = tmp_path / "orfs.gff"
    domtblout = tmp_path / "raw_hits.domtblout"
    prodigal_gff.write_text(
        "\n".join(
            [
                "##gff-version 3",
                (
                    "s228.ctg000409l\tProdigal_v2.6.3\tCDS\t259779\t260963"
                    "\t.\t-\t0\tID=409_287;partial=00"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    domtblout.write_text(
        _domtblout_line(
            orf_id="s228.ctg000409l_287",
            marker_id="PGK",
        ),
        encoding="utf-8",
    )

    hits = parse_scg_hits(
        domtblout_tsv=domtblout,
        prodigal_gff=prodigal_gff,
        domain="bacteria",
    )

    assert len(hits) == 1
    assert hits[0].orf_id == "s228.ctg000409l_287"
    assert hits[0].contig_id == "s228.ctg000409l"
    assert hits[0].marker_id == "PGK"


def test_scg_cache_reuses_orfs_hmmsearch_and_parser(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from porebin_genome.evidence.scg import discovery

    fasta = tmp_path / "contigs.fasta"
    fasta.write_text(">c1\nACGTACGT\n", encoding="utf-8")
    calls = {"prodigal": 0, "hmmsearch": 0}

    monkeypatch.setattr(
        discovery,
        "_resolve_required_executable",
        lambda executable, purpose: executable,
    )

    def fake_prodigal(**kwargs) -> None:
        calls["prodigal"] += 1
        kwargs["proteins_faa"].write_text(">orf1\nMPEPTIDE\n", encoding="utf-8")
        kwargs["prodigal_gff"].write_text(
            "c1\tProdigal\tCDS\t1\t8\t.\t+\t0\tID=orf1\n",
            encoding="utf-8",
        )

    def fake_hmmsearch(**kwargs) -> None:
        calls["hmmsearch"] += 1
        kwargs["domtblout_tsv"].write_text(
            _domtblout_line(orf_id="orf1", marker_id="TIGR00388"),
            encoding="utf-8",
        )
        kwargs["hmmsearch_txt"].write_text("ok\n", encoding="utf-8")

    monkeypatch.setattr(discovery, "_run_prodigal", fake_prodigal)
    monkeypatch.setattr(discovery, "_run_hmmsearch", fake_hmmsearch)

    first = prepare_scg_profiles(contigs_fasta=fasta, out_dir=tmp_path / "scg")
    second = prepare_scg_profiles(contigs_fasta=fasta, out_dir=tmp_path / "scg")

    assert first.cache_status == "full_rebuild"
    assert second.cache_status == "complete_cache_hit"
    assert calls == {"prodigal": 1, "hmmsearch": 1}
    assert second.contig_profiles["c1"].marker_ids == ("TIGR00389",)

    monkeypatch.setattr(
        discovery,
        "SCG_PARSER_SCHEMA_VERSION",
        discovery.SCG_PARSER_SCHEMA_VERSION + 1,
    )
    third = prepare_scg_profiles(contigs_fasta=fasta, out_dir=tmp_path / "scg")

    assert third.cache_status == "parser_rebuilt"
    assert calls == {"prodigal": 1, "hmmsearch": 1}


def _domtblout_line(*, orf_id: str, marker_id: str) -> str:
    fields = [
        orf_id,
        "-",
        "100",
        marker_id,
        "-",
        "100",
        "1e-20",
        "50",
        "0",
        "1",
        "1",
        "1e-20",
        "1e-20",
        "50",
        "0",
        "1",
        "90",
        "1",
        "90",
        "1",
        "90",
        "0.9",
        "test",
    ]
    return " ".join(fields) + "\n"
