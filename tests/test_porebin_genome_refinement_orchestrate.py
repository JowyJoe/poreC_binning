from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.porebin_genome_testkit import (
    read_json,
    read_tsv_rows,
    write_noop_refine_fixture,
    write_recruit_refine_fixture,
    write_split_refine_fixture,
)


def _write_embedding(
    path: Path,
    rows: dict[str, tuple[float, float]],
) -> Path:
    lines = ["contig_id\tz0\tz1\tsupport_weight"]
    lines.extend(
        f"{name}\t{vector[0]}\t{vector[1]}\t1"
        for name, vector in rows.items()
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_replacement_orchestrator_writes_stage_log_without_checkpoints(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    from porebin_genome.refinement.orchestrate import run_refinement

    fixture = write_noop_refine_fixture(tmp_path)
    embedding = _write_embedding(
        tmp_path / "embedding.tsv",
        {
            "a1": (1.0, 0.0),
            "a2": (0.99, 0.01),
            "b1": (0.0, 1.0),
            "b2": (0.01, 0.99),
        },
    )

    result = run_refinement(
        contigs_fasta=fixture["contigs"],
        coarse_bins_tsv=fixture["coarse_bins"],
        contacts_parquet=fixture["contacts"],
        coverage_tsv=fixture["coverage"],
        embedding_tsv=embedding,
        enable_scg=False,
        out_dir=tmp_path / "out",
    )

    events = _read_jsonl(result.refine_stage_log_jsonl)
    completed_stages = [
        event["stage"]
        for event in events
        if event["event"] == "stage_complete"
    ]
    assert completed_stages == [
        "validate_inputs",
        "scg",
        "initialize",
        "split",
        "merge",
        "recruit",
        "finalize",
    ]
    assert events[0]["event"] == "run_start"
    assert events[-1]["event"] == "run_complete"
    assert events[0]["checkpoint_enabled"] is False

    merge_event = next(
        event
        for event in events
        if event["event"] == "stage_complete"
        and event["stage"] == "merge"
    )
    assert merge_event["contact_counters"]["full_edge_scans"] == 1
    split_event = next(
        event
        for event in events
        if event["event"] == "stage_complete"
        and event["stage"] == "split"
    )
    recruit_event = next(
        event
        for event in events
        if event["event"] == "stage_complete"
        and event["stage"] == "recruit"
    )
    assert split_event["contact_counters"]["full_edge_scans"] == 0
    assert recruit_event["contact_counters"]["full_edge_scans"] == 0

    meta = read_json(result.refine_meta_json)
    assert meta["engine"] == "replacement_refinement"
    assert meta["checkpoint_enabled"] is False
    assert meta["stage_order"] == ["split", "merge", "recruit"]
    assert meta["inputs"]["refine_embedding_source"] == "hgvae"
    assert meta["inputs"]["embedding_role"] == "hgvae_similarity_evidence"
    assert meta["contact_index"]["counters"]["full_edge_scans"] == 2
    assert not list((tmp_path / "out").rglob("*checkpoint*"))


def test_replacement_orchestrator_applies_scg_guided_split_sequentially(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pyarrow")
    from porebin_genome.refinement import orchestrate

    fixture = write_split_refine_fixture(tmp_path)
    embedding = _write_embedding(
        tmp_path / "embedding.tsv",
        {
            "a": (1.0, 0.0),
            "b": (0.99, 0.01),
            "c": (0.0, 1.0),
            "d": (0.01, 0.99),
        },
    )
    scg_dir = tmp_path / "out" / "evidence" / "scg"
    scg_dir.mkdir(parents=True)
    scg_index = scg_dir / "contig_scg_index.json"
    scg_index.write_text(
        json.dumps(
            {
                "profiles": {
                    "a": {
                        "marker_ids": ["GrpE"],
                        "marker_orf_counts": {"GrpE": 1},
                    },
                    "c": {
                        "marker_ids": ["GrpE"],
                        "marker_orf_counts": {"GrpE": 1},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    fake_panel = SimpleNamespace(
        marker_count=107,
        metadata=lambda: {"canonical_marker_count": 107},
    )
    monkeypatch.setattr(
        orchestrate,
        "prepare_scg_profiles",
        lambda **_kwargs: SimpleNamespace(
            contig_index_json=scg_index,
            contig_profiles={"a": object(), "c": object()},
            cache_status="test_cache",
            panel=fake_panel,
            scg_dir=scg_dir,
        ),
    )

    result = orchestrate.run_refinement(
        contigs_fasta=fixture["contigs"],
        coarse_bins_tsv=fixture["coarse_bins"],
        contacts_parquet=fixture["contacts"],
        coverage_tsv=fixture["coverage"],
        embedding_tsv=embedding,
        enable_scg=True,
        out_dir=tmp_path / "out",
    )

    assignment = {
        row["contig_id"]: row["bin_id"]
        for row in read_tsv_rows(result.bins_refined_tsv)
    }
    assert assignment["a"] == assignment["b"] == "0"
    assert assignment["c"] == assignment["d"] == "1"

    actions = read_tsv_rows(result.refine_actions_tsv)
    split_rows = [
        row for row in actions if row["action_type"] == "split"
    ]
    assert len(split_rows) == 1
    assert split_rows[0]["accepted"] == "1"
    assert split_rows[0]["reason"] == "accepted"

    meta = read_json(result.refine_meta_json)
    assert meta["stage_counts"]["split"]["suspect"] == 1
    assert meta["stage_counts"]["split"]["accepted"] == 1
    assert meta["contact_index"]["counters"]["full_edge_scans"] == 2


def test_recruit_action_note_records_porec_hgvae_agreement(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    from porebin_genome.refinement.orchestrate import run_refinement

    fixture = write_recruit_refine_fixture(tmp_path)
    embedding = _write_embedding(
        tmp_path / "embedding.tsv",
        {
            "a": (1.0, 0.0),
            "b": (1.0, 0.0),
            "h": (1.0, 0.0),
        },
    )

    result = run_refinement(
        contigs_fasta=fixture["contigs"],
        coarse_bins_tsv=fixture["coarse_bins"],
        contacts_parquet=fixture["contacts"],
        coverage_tsv=fixture["coverage"],
        embedding_tsv=embedding,
        enable_scg=False,
        out_dir=tmp_path / "out",
    )

    recruit_rows = [
        row
        for row in read_tsv_rows(result.refine_actions_tsv)
        if row["action_type"] == "recruit"
    ]
    assert len(recruit_rows) == 1
    assert recruit_rows[0]["accepted"] == "1"
    note = json.loads(recruit_rows[0]["note"])
    candidate = note["candidate"]

    assert candidate["note_schema_version"] == 2
    assert candidate["selector"] == (
        "porec_best_target_and_hgvae_min_radius_normalized_score"
    )
    assert candidate["target_agreement"] is True
    assert candidate["porec_target_bin"] == 0
    assert candidate["hgvae_target_bin"] == 0
    assert candidate["hgvae_score_formula"] == (
        "latent_distance/(target_radius+eps)"
    )
    assert candidate["hgvae_target_score"] == pytest.approx(0.0)
    assert candidate["porec_supporting_edges"] >= 2

    assert note["embedding"]["type"] == "recruit_contig_to_bin"
    assert note["embedding"]["rule"] == "distance <= target_radius"
    assert note["embedding"]["inside_radius"] is True

    meta = read_json(result.refine_meta_json)
    assert meta["embedding_evidence"]["enabled"] is True
    assert meta["embedding_evidence"]["source"] == "hgvae"
    assert meta["embedding_evidence"]["action_note_schema_version"] == 2
    assert "radius-normalized" in meta["embedding_evidence"]["recruit_selector"]


def test_stage_log_records_input_failure_without_partial_results(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    from porebin_genome.refinement.orchestrate import run_refinement

    out_dir = tmp_path / "out"
    with pytest.raises(FileNotFoundError):
        run_refinement(
            contigs_fasta=tmp_path / "missing.fasta",
            coarse_bins_tsv=tmp_path / "missing.tsv",
            contacts_parquet=tmp_path / "missing.parquet",
            coverage_tsv=tmp_path / "missing_coverage.tsv",
            enable_scg=False,
            out_dir=out_dir,
        )

    events = _read_jsonl(
        out_dir / "final" / "refine_stage_log.jsonl"
    )
    assert [event["event"] for event in events][-2:] == [
        "stage_error",
        "run_error",
    ]
    assert not (out_dir / "final" / "bins.refined.tsv").exists()
