from __future__ import annotations

from pathlib import Path

import pytest

from tests.porebin_genome_testkit import (
    read_json,
    read_tsv_rows,
    write_ambiguous_refine_fixture,
    write_noop_refine_fixture,
    write_reassign_refine_fixture,
    write_recruit_refine_fixture,
    write_split_refine_fixture,
)


def test_refine_split_applies_on_obvious_two_group_coarse_bin(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for refine integration test")

    from porebin_genome.refine.orchestrate import run_refinement

    fixture = write_split_refine_fixture(tmp_path)
    out_dir = tmp_path / "out"
    result = run_refinement(
        contigs_fasta=fixture["contigs"],
        coarse_bins_tsv=fixture["coarse_bins"],
        contacts_parquet=fixture["contacts"],
        coverage_tsv=fixture["coverage"],
        out_dir=out_dir,
    )

    rows = read_tsv_rows(result.bins_refined_tsv)
    assignment = {row["contig_id"]: row["bin_id"] for row in rows}
    assert assignment["a"] == assignment["b"]
    assert assignment["c"] == assignment["d"]
    assert assignment["a"] != assignment["c"]

    meta = read_json(result.refine_meta_json)
    assert meta["n_split_candidates"] >= 1
    assert meta["n_split_applied"] >= 1


def test_refine_reassign_moves_contig_to_target_bin(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for refine integration test")

    from porebin_genome.refine.orchestrate import run_refinement

    fixture = write_reassign_refine_fixture(tmp_path)
    out_dir = tmp_path / "out"
    result = run_refinement(
        contigs_fasta=fixture["contigs"],
        coarse_bins_tsv=fixture["coarse_bins"],
        contacts_parquet=fixture["contacts"],
        coverage_tsv=fixture["coverage"],
        out_dir=out_dir,
    )

    rows = read_tsv_rows(result.bins_refined_tsv)
    assignment = {row["contig_id"]: row["bin_id"] for row in rows}
    assert assignment["e"] == assignment["c"]

    actions = read_tsv_rows(result.refine_actions_tsv)
    reassign_rows = [row for row in actions if row["action_type"] == "reassign" and row["contig_id"] == "e"]
    assert reassign_rows
    assert reassign_rows[0]["accepted"] == "1"
    assert reassign_rows[0]["reason"] == "move_to_target_bin"


def test_refine_keeps_low_confidence_contig_unbinned(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for refine integration test")

    from porebin_genome.refine.orchestrate import run_refinement

    fixture = write_ambiguous_refine_fixture(tmp_path)
    out_dir = tmp_path / "out"
    result = run_refinement(
        contigs_fasta=fixture["contigs"],
        coarse_bins_tsv=fixture["coarse_bins"],
        contacts_parquet=fixture["contacts"],
        coverage_tsv=fixture["coverage"],
        out_dir=out_dir,
    )

    rows = read_tsv_rows(result.bins_refined_tsv)
    assignment = {row["contig_id"]: row["bin_id"] for row in rows}
    assert "x" not in assignment

    unbinned = read_tsv_rows(result.unbinned_tsv)
    by_contig = {row["contig_id"]: row for row in unbinned}
    assert by_contig["x"]["reason"] in {
        "assignment_confidence_below_threshold",
        "target_bin_margin_too_small",
        "target_bin_feature_gate_failed",
    }


def test_refine_recruits_high_confidence_unbinned_contig(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for refine integration test")

    from porebin_genome.refine.orchestrate import run_refinement

    fixture = write_recruit_refine_fixture(tmp_path)
    out_dir = tmp_path / "out"
    result = run_refinement(
        contigs_fasta=fixture["contigs"],
        coarse_bins_tsv=fixture["coarse_bins"],
        contacts_parquet=fixture["contacts"],
        coverage_tsv=fixture["coverage"],
        out_dir=out_dir,
    )

    rows = read_tsv_rows(result.bins_refined_tsv)
    assignment = {row["contig_id"]: row["bin_id"] for row in rows}
    assert assignment["h"] == "0"

    meta = read_json(result.refine_meta_json)
    assert meta["n_recruit_candidates"] >= 1
    assert meta["n_recruited"] >= 1


def test_refine_noop_regression_keeps_clean_coarse_result_stable(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for refine integration test")

    from porebin_genome.refine.orchestrate import run_refinement

    fixture = write_noop_refine_fixture(tmp_path)
    out_dir = tmp_path / "out"
    result = run_refinement(
        contigs_fasta=fixture["contigs"],
        coarse_bins_tsv=fixture["coarse_bins"],
        contacts_parquet=fixture["contacts"],
        coverage_tsv=fixture["coverage"],
        out_dir=out_dir,
    )

    refined = read_tsv_rows(result.bins_refined_tsv)
    coarse = read_tsv_rows(fixture["coarse_bins"])
    refined_assignment = {row["contig_id"]: row["bin_id"] for row in refined}
    coarse_assignment = {row["contig_name"]: row["bin_id"] for row in coarse}
    assert refined_assignment == coarse_assignment

    meta = read_json(result.refine_meta_json)
    assert meta["n_split_applied"] == 0
    assert meta["n_reassigned"] == 0
    assert meta["n_recruited"] == 0
