from __future__ import annotations

from pathlib import Path

import pytest


def test_porebin_genome_contract_validators(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow", reason="pyarrow required for contacts.parquet contract test")
    import pyarrow as pa
    import pyarrow.parquet as pq

    from porebin_genome.io.contacts import validate_contacts_parquet_core_schema
    from porebin_genome.io.contracts import (
        BIN_QC_COLUMNS,
        COARSE_BINS_COLUMNS,
        COVERAGE_TSV_COLUMNS,
        FINAL_BINS_COLUMNS,
        REFINE_ACTIONS_COLUMNS,
        UNBINNED_COLUMNS,
    )
    from porebin_genome.io.coverage import read_coverage_tsv, validate_coverage_tsv
    from porebin_genome.io.tables import read_assignment_tsv, validate_tsv_header, write_tsv_rows

    contacts = tmp_path / "contacts.parquet"
    pq.write_table(
        pa.table(
            {
                "contact_id": pa.array([0], type=pa.int64()),
                "contigs": pa.array([["c1", "c2"]], type=pa.list_(pa.string())),
                "contig_weights": pa.array([[0.5, 0.5]], type=pa.list_(pa.float64())),
                "k": pa.array([2], type=pa.int32()),
                "k_eff": pa.array([2.0], type=pa.float64()),
                "weight": pa.array([1.0], type=pa.float64()),
            }
        ),
        contacts,
    )
    validate_contacts_parquet_core_schema(contacts)

    coverage = tmp_path / "coverage.tsv"
    write_tsv_rows(coverage, COVERAGE_TSV_COLUMNS, [("c1", 10.0), ("c2", 12.5)])
    validate_coverage_tsv(coverage)
    assert read_coverage_tsv(coverage) == {"c1": 10.0, "c2": 12.5}

    coarse_bins = tmp_path / "bins.tsv"
    write_tsv_rows(coarse_bins, COARSE_BINS_COLUMNS, [("c1", "0"), ("c2", "1")])
    validate_tsv_header(coarse_bins, COARSE_BINS_COLUMNS)
    assert read_assignment_tsv(coarse_bins, COARSE_BINS_COLUMNS) == {"c1": "0", "c2": "1"}
    assert all("host" not in column.lower() for column in COARSE_BINS_COLUMNS)

    refined_bins = tmp_path / "bins.refined.tsv"
    write_tsv_rows(refined_bins, FINAL_BINS_COLUMNS, [("c1", "0", "coarse_keep", "coarse_assignment_retained")])
    validate_tsv_header(refined_bins, FINAL_BINS_COLUMNS)
    assert all("host" not in column.lower() for column in FINAL_BINS_COLUMNS)

    unbinned = tmp_path / "unbinned.tsv"
    write_tsv_rows(
        unbinned,
        UNBINNED_COLUMNS,
        [("c2", "recruit", "target_bin_margin_too_small", "", "kept_unbinned")],
    )
    validate_tsv_header(unbinned, UNBINNED_COLUMNS)
    assert all("host" not in column.lower() for column in UNBINNED_COLUMNS)

    bin_qc = tmp_path / "bin_qc.tsv"
    write_tsv_rows(bin_qc, BIN_QC_COLUMNS, [("0", 1, 1000, "10", "0.95", "scg_clean", 0, 0, "stable", "stable_after_refine")])
    validate_tsv_header(bin_qc, BIN_QC_COLUMNS)
    assert all("host" not in column.lower() for column in BIN_QC_COLUMNS)

    refine_actions = tmp_path / "refine_actions.tsv"
    write_tsv_rows(
        refine_actions,
        REFINE_ACTIONS_COLUMNS,
        [
            (
                "recruit",
                "c1",
                "",
                "",
                "1",
                "accepted",
                1,
                0.9,
                0.2,
                "pass",
                "accepted",
            )
        ],
    )
    validate_tsv_header(refine_actions, REFINE_ACTIONS_COLUMNS)
    assert all("host" not in column.lower() for column in REFINE_ACTIONS_COLUMNS)
