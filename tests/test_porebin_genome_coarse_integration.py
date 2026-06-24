from __future__ import annotations

from pathlib import Path

import pytest

from tests.porebin_genome_testkit import read_json, read_tsv_rows, write_dual_community_fixture


def test_dual_community_synthetic_produces_multiple_candidate_bins(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for coarse integration test")
    pytest.importorskip("hdbscan", reason="hdbscan required for coarse integration test")

    from porebin_genome.coarse.orchestrate import run_coarse_discovery

    fixture = write_dual_community_fixture(tmp_path, community_size=4)
    out_dir = tmp_path / "out"
    result = run_coarse_discovery(
        contigs_fasta=fixture["contigs"],
        contacts_parquet=fixture["contacts"],
        coverage_tsv=fixture["coverage"],
        out_dir=out_dir,
        feature_knn_k=2,
    )

    coarse_rows = read_tsv_rows(result.bins_tsv)
    coarse_run = read_json(result.run_json)

    assert result.implemented is True
    assert len(coarse_rows) > 0
    assert len({row["bin_id"] for row in coarse_rows}) >= 2
    assert coarse_run["n_bins"] >= 2
    assert coarse_run["n_contigs_clustered"] >= 2
    assert coarse_run["lambda_contact"] == 0.5
    assert coarse_run["collapse_warning"] is False
    assert "host" not in result.bins_tsv.read_text(encoding="utf-8").lower()
    assert "host" not in result.run_json.read_text(encoding="utf-8").lower()


def test_coarse_run_json_contains_required_collapse_audit_fields(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for coarse audit test")
    pytest.importorskip("hdbscan", reason="hdbscan required for coarse audit test")

    from porebin_genome.coarse.orchestrate import run_coarse_discovery

    fixture = write_dual_community_fixture(tmp_path, community_size=3)
    out_dir = tmp_path / "out"
    result = run_coarse_discovery(
        contigs_fasta=fixture["contigs"],
        contacts_parquet=fixture["contacts"],
        coverage_tsv=fixture["coverage"],
        out_dir=out_dir,
    )
    coarse_run = read_json(result.run_json)

    for key in (
        "n_contigs_total",
        "n_contigs_clustered",
        "n_contigs_unbinned",
        "n_bins",
        "largest_bin_fraction",
        "hdbscan_noise_fraction",
        "bin_size_quantiles",
        "embedding_dim",
        "feature_mode",
        "contact_hyperedge_count",
        "feature_knn_k",
        "coarse_embedding_source",
        "refine_embedding_source",
        "hgvae_role",
    ):
        assert key in coarse_run


def test_default_knn_mode_is_adaptive_and_writes_report_and_meta(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for adaptive coarse integration test")
    pytest.importorskip("hdbscan", reason="hdbscan required for adaptive coarse integration test")

    from porebin_genome.coarse.orchestrate import run_coarse_discovery

    fixture = write_dual_community_fixture(tmp_path, community_size=4)
    out_dir = tmp_path / "out"
    result = run_coarse_discovery(
        contigs_fasta=fixture["contigs"],
        contacts_parquet=fixture["contacts"],
        coverage_tsv=fixture["coverage"],
        out_dir=out_dir,
    )
    coarse_run = read_json(result.run_json)
    adaptive_report = read_tsv_rows(out_dir / "coarse" / "adaptive_k_report.tsv")
    adaptive_meta = read_json(out_dir / "coarse" / "adaptive_k_meta.json")

    assert adaptive_report
    assert len(adaptive_report) == len(fixture["all_names"])
    assert {
        "contig_id",
        "selected_k_i",
        "top_score",
        "cutoff_score",
        "cutoff_rank",
        "cutoff_reason",
    }.issubset(adaptive_report[0])
    assert adaptive_meta["mode"] == "adaptive"
    assert adaptive_meta["min_k"] == 5
    assert adaptive_meta["default_k"] == 15
    assert adaptive_meta["k_max"] == 30
    assert adaptive_meta["drop_ratio"] == 0.15
    assert adaptive_meta["mutual_knn"] is True
    assert adaptive_meta["fallback_used"] is False
    assert coarse_run["feature_knn_mode"] == "adaptive"
    assert "adaptive_k_report_tsv" in coarse_run["outputs"]


def test_fixed_knn_mode_still_uses_manual_integer_k(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for fixed coarse integration test")
    pytest.importorskip("hdbscan", reason="hdbscan required for fixed coarse integration test")

    from porebin_genome.coarse.orchestrate import run_coarse_discovery

    fixture = write_dual_community_fixture(tmp_path, community_size=4)
    out_dir = tmp_path / "out"
    result = run_coarse_discovery(
        contigs_fasta=fixture["contigs"],
        contacts_parquet=fixture["contacts"],
        coverage_tsv=fixture["coverage"],
        out_dir=out_dir,
        feature_knn_k=7,
    )
    coarse_run = read_json(result.run_json)

    assert coarse_run["feature_knn_mode"] == "fixed"
    assert coarse_run["feature_knn_k"] == 7
    assert "adaptive_k_report_tsv" not in coarse_run["outputs"]


def test_coarse_method_normalization_accepts_hgvae_aliases() -> None:
    from porebin_genome.coarse.orchestrate import normalize_coarse_method

    assert normalize_coarse_method("spectral") == "spectral"
    assert normalize_coarse_method("hgvae") == "hgvae"
    assert normalize_coarse_method("hg-vae") == "hgvae"


def test_collapse_warning_is_emitted_for_overcollapsed_labels() -> None:
    import numpy as np

    from porebin_genome.coarse.metadata import build_coarse_run_record

    run_record = build_coarse_run_record(
        contigs_fasta="contigs.fasta",
        contacts_parquet="contacts.parquet",
        coverage_tsv="coverage.tsv",
        bins_tsv="coarse/bins.tsv",
        labels=np.zeros((8,), dtype=int),
        feature_mode="tnf_plus_cov",
        embedding_dim=4,
        lambda_contact=0.5,
        contact_hyperedge_count=12,
        feature_knn_k=7,
        dropped_singleton_contacts=0,
        coverage_used=True,
        coverage_missing_count=0,
        hdbscan_meta={"impl": "hdbscan"},
    )

    assert run_record["largest_bin_fraction"] == 1.0
    assert run_record["collapse_warning"] is True
    assert run_record["warnings"]
