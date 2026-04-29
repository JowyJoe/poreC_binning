from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tests.porebin_genome_testkit import read_json, read_tsv_rows, write_dual_community_fixture


def test_porebin_genome_bin_runs_coarse_and_refine_mvp(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for genome-centric e2e")
    pytest.importorskip("hdbscan", reason="hdbscan required for genome-centric e2e")

    from porebin_genome.cli import app

    fixture = write_dual_community_fixture(tmp_path, community_size=4)
    out_dir = tmp_path / "out"
    export_dir = tmp_path / "export_out"

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "bin",
            "--contigs",
            str(fixture["contigs"]),
            "--contacts",
            str(fixture["contacts"]),
            "--coverage-tsv",
            str(fixture["coverage"]),
            "--disable-scg",
            "--out",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.stdout

    coarse_bins = out_dir / "coarse" / "bins.tsv"
    coarse_run = out_dir / "coarse" / "run.json"
    refined_bins = out_dir / "final" / "bins.refined.tsv"
    unbinned = out_dir / "final" / "unbinned.tsv"
    bin_qc = out_dir / "final" / "bin_qc.tsv"
    refine_actions = out_dir / "final" / "refine_actions.tsv"
    refine_meta = out_dir / "final" / "refine_meta.json"

    assert coarse_bins.exists()
    assert coarse_run.exists()
    assert refined_bins.exists()
    assert unbinned.exists()
    assert bin_qc.exists()
    assert refine_actions.exists()
    assert refine_meta.exists()

    coarse_rows = read_tsv_rows(coarse_bins)
    assert len(coarse_rows) >= 2
    assert "host" not in coarse_bins.read_text(encoding="utf-8").lower()

    refined_rows = read_tsv_rows(refined_bins)
    assert refined_rows
    assert set(refined_rows[0].keys()) == {
        "contig_id",
        "bin_id",
        "assignment_stage",
        "assignment_reason",
    }
    assert "host" not in refined_bins.read_text(encoding="utf-8").lower()

    unbinned_rows = read_tsv_rows(unbinned)
    assert set(unbinned_rows[0].keys()) == {"contig_id", "stage", "reason", "source_bin", "note"} if unbinned_rows else True

    qc_rows = read_tsv_rows(bin_qc)
    assert qc_rows
    assert set(qc_rows[0].keys()) == {
        "bin_id",
        "n_contigs",
        "total_length",
        "median_coverage",
        "contact_coherence",
        "scg_status",
        "scg_duplicate_marker_count",
        "suspect_flag",
        "refine_status",
        "notes",
    }

    action_rows = read_tsv_rows(refine_actions)
    assert refine_actions.read_text(encoding="utf-8").splitlines()[0] == (
        "action_type\tcontig_id\tbin_id\tsource_bin\ttarget_bin\treason\taccepted\tconfidence\tdelta_contact\tscg_status\tnote"
    )
    assert "host" not in refine_actions.read_text(encoding="utf-8").lower()

    refine_meta_payload = read_json(refine_meta)
    assert refine_meta_payload["implemented"] is True
    assert refine_meta_payload["n_bins_out"] >= 1
    assert refine_meta_payload["notes"]["reassign_semantics"] == "move_to_target_bin_or_abstain_to_unbinned"

    export_result = runner.invoke(
        app,
        [
            "export",
            "--contigs",
            str(fixture["contigs"]),
            "--bins-refined-tsv",
            str(refined_bins),
            "--unbinned-tsv",
            str(unbinned),
            "--out",
            str(export_dir),
        ],
    )
    assert export_result.exit_code == 0, export_result.stdout

    export_meta = export_dir / "export" / "export_meta.json"
    assert export_meta.exists()
