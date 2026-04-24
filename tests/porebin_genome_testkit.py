from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_dual_community_fixture(tmp_path: Path, *, community_size: int = 4) -> dict[str, object]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    contigs = tmp_path / "contigs.fasta"
    contacts = tmp_path / "contacts.parquet"
    coverage = tmp_path / "coverage.tsv"

    a_names = [f"a{i}" for i in range(1, community_size + 1)]
    b_names = [f"b{i}" for i in range(1, community_size + 1)]
    seqs: dict[str, str] = {}
    for idx, name in enumerate(a_names, start=1):
        seqs[name] = ("AAAACAAA" + ("AT" * idx) + "AAAAGAAA") * 12
    for idx, name in enumerate(b_names, start=1):
        seqs[name] = ("CCCCGCCC" + ("CG" * idx) + "CCCCTCCC") * 12

    fasta_lines: list[str] = []
    for name in a_names + b_names:
        fasta_lines.append(f">{name}")
        fasta_lines.append(seqs[name])
    contigs.write_text("\n".join(fasta_lines) + "\n", encoding="utf-8")

    coverage_lines = ["contig_name\tcoverage"]
    for idx, name in enumerate(a_names, start=1):
        coverage_lines.append(f"{name}\t{8 + idx}")
    for idx, name in enumerate(b_names, start=1):
        coverage_lines.append(f"{name}\t{35 + idx}")
    coverage.write_text("\n".join(coverage_lines) + "\n", encoding="utf-8")

    contact_rows: list[dict[str, object]] = []
    contact_id = 0
    for members in (a_names, b_names):
        for left, right in combinations(members, 2):
            contact_rows.append(
                {
                    "contact_id": contact_id,
                    "contigs": [left, right],
                    "contig_weights": [0.5, 0.5],
                    "k": 2,
                    "k_eff": 2.0,
                    "weight": 1.0,
                }
            )
            contact_id += 1

    pq.write_table(
        pa.table(
            {
                "contact_id": pa.array([row["contact_id"] for row in contact_rows], type=pa.int64()),
                "contigs": pa.array([row["contigs"] for row in contact_rows], type=pa.list_(pa.string())),
                "contig_weights": pa.array(
                    [row["contig_weights"] for row in contact_rows],
                    type=pa.list_(pa.float64()),
                ),
                "k": pa.array([row["k"] for row in contact_rows], type=pa.int32()),
                "k_eff": pa.array([row["k_eff"] for row in contact_rows], type=pa.float64()),
                "weight": pa.array([row["weight"] for row in contact_rows], type=pa.float64()),
            }
        ),
        contacts,
    )

    return {
        "contigs": contigs,
        "contacts": contacts,
        "coverage": coverage,
        "community_a": a_names,
        "community_b": b_names,
        "all_names": a_names + b_names,
    }


def write_collapsed_fixture(tmp_path: Path, *, n_contigs: int = 6) -> dict[str, object]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    contigs = tmp_path / "contigs.fasta"
    contacts = tmp_path / "contacts.parquet"
    coverage = tmp_path / "coverage.tsv"

    names = [f"c{i}" for i in range(1, n_contigs + 1)]
    contigs.write_text(
        "\n".join([line for name in names for line in (f">{name}", ("ACGT" * 40))]) + "\n",
        encoding="utf-8",
    )
    coverage.write_text(
        "contig_name\tcoverage\n" + "\n".join(f"{name}\t20" for name in names) + "\n",
        encoding="utf-8",
    )

    contact_rows: list[dict[str, object]] = []
    contact_id = 0
    for left, right in combinations(names, 2):
        contact_rows.append(
            {
                "contact_id": contact_id,
                "contigs": [left, right],
                "contig_weights": [0.5, 0.5],
                "k": 2,
                "k_eff": 2.0,
                "weight": 1.0,
            }
        )
        contact_id += 1

    pq.write_table(
        pa.table(
            {
                "contact_id": pa.array([row["contact_id"] for row in contact_rows], type=pa.int64()),
                "contigs": pa.array([row["contigs"] for row in contact_rows], type=pa.list_(pa.string())),
                "contig_weights": pa.array(
                    [row["contig_weights"] for row in contact_rows],
                    type=pa.list_(pa.float64()),
                ),
                "k": pa.array([row["k"] for row in contact_rows], type=pa.int32()),
                "k_eff": pa.array([row["k_eff"] for row in contact_rows], type=pa.float64()),
                "weight": pa.array([row["weight"] for row in contact_rows], type=pa.float64()),
            }
        ),
        contacts,
    )

    return {
        "contigs": contigs,
        "contacts": contacts,
        "coverage": coverage,
        "all_names": names,
    }


def write_generic_refine_fixture(
    tmp_path: Path,
    *,
    contig_sequences: dict[str, str],
    coverage_by_contig: dict[str, float],
    coarse_assignment: dict[str, str],
    contacts_rows: list[dict[str, object]],
) -> dict[str, Path]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    contigs = tmp_path / "contigs.fasta"
    contacts = tmp_path / "contacts.parquet"
    coverage = tmp_path / "coverage.tsv"
    coarse_bins = tmp_path / "bins.tsv"

    fasta_lines: list[str] = []
    for contig_id, sequence in contig_sequences.items():
        fasta_lines.append(f">{contig_id}")
        fasta_lines.append(sequence)
    contigs.write_text("\n".join(fasta_lines) + "\n", encoding="utf-8")

    coverage_lines = ["contig_name\tcoverage"]
    for contig_id, value in coverage_by_contig.items():
        coverage_lines.append(f"{contig_id}\t{value}")
    coverage.write_text("\n".join(coverage_lines) + "\n", encoding="utf-8")

    coarse_lines = ["contig_name\tbin_id"]
    for contig_id, bin_id in coarse_assignment.items():
        coarse_lines.append(f"{contig_id}\t{bin_id}")
    coarse_bins.write_text("\n".join(coarse_lines) + "\n", encoding="utf-8")

    pq.write_table(
        pa.table(
            {
                "contact_id": pa.array([row["contact_id"] for row in contacts_rows], type=pa.int64()),
                "contigs": pa.array([row["contigs"] for row in contacts_rows], type=pa.list_(pa.string())),
                "contig_weights": pa.array(
                    [row["contig_weights"] for row in contacts_rows],
                    type=pa.list_(pa.float64()),
                ),
                "k": pa.array([row["k"] for row in contacts_rows], type=pa.int32()),
                "k_eff": pa.array([row["k_eff"] for row in contacts_rows], type=pa.float64()),
                "weight": pa.array([row["weight"] for row in contacts_rows], type=pa.float64()),
            }
        ),
        contacts,
    )

    return {
        "contigs": contigs,
        "contacts": contacts,
        "coverage": coverage,
        "coarse_bins": coarse_bins,
    }


def write_noop_refine_fixture(tmp_path: Path) -> dict[str, Path]:
    seq_a = ("AAAACCCCGGGGTTTT" * 150)
    seq_b = ("ATATATATCGCGCGCG" * 150)
    contigs = {
        "a1": seq_a,
        "a2": seq_a,
        "b1": seq_b,
        "b2": seq_b,
    }
    coverage = {"a1": 20.0, "a2": 22.0, "b1": 60.0, "b2": 58.0}
    coarse = {"a1": "0", "a2": "0", "b1": "1", "b2": "1"}
    rows: list[dict[str, object]] = []
    contact_id = 0
    for _ in range(60):
        rows.append(
            {
                "contact_id": contact_id,
                "contigs": ["a1", "a2"],
                "contig_weights": [0.5, 0.5],
                "k": 2,
                "k_eff": 2.0,
                "weight": 1.0,
            }
        )
        contact_id += 1
    for _ in range(60):
        rows.append(
            {
                "contact_id": contact_id,
                "contigs": ["b1", "b2"],
                "contig_weights": [0.5, 0.5],
                "k": 2,
                "k_eff": 2.0,
                "weight": 1.0,
            }
        )
        contact_id += 1
    return write_generic_refine_fixture(
        tmp_path,
        contig_sequences=contigs,
        coverage_by_contig=coverage,
        coarse_assignment=coarse,
        contacts_rows=rows,
    )


def write_split_refine_fixture(tmp_path: Path) -> dict[str, Path]:
    seq_left = ("AAAACCCCGGGGTTTT" * 150)
    seq_right = ("ATATATATCGCGCGCG" * 150)
    contigs = {
        "a": seq_left,
        "b": seq_left,
        "c": seq_right,
        "d": seq_right,
    }
    coverage = {"a": 30.0, "b": 31.0, "c": 9.0, "d": 10.0}
    coarse = {"a": "0", "b": "0", "c": "0", "d": "0"}
    rows: list[dict[str, object]] = []
    contact_id = 0
    for _ in range(80):
        rows.append(
            {
                "contact_id": contact_id,
                "contigs": ["a", "b"],
                "contig_weights": [0.5, 0.5],
                "k": 2,
                "k_eff": 2.0,
                "weight": 1.0,
            }
        )
        contact_id += 1
    for _ in range(80):
        rows.append(
            {
                "contact_id": contact_id,
                "contigs": ["c", "d"],
                "contig_weights": [0.5, 0.5],
                "k": 2,
                "k_eff": 2.0,
                "weight": 1.0,
            }
        )
        contact_id += 1
    return write_generic_refine_fixture(
        tmp_path,
        contig_sequences=contigs,
        coverage_by_contig=coverage,
        coarse_assignment=coarse,
        contacts_rows=rows,
    )


def write_reassign_refine_fixture(tmp_path: Path) -> dict[str, Path]:
    seq_zero = ("AAAACCCCGGGGTTTT" * 150)
    seq_one = ("ATATATATCGCGCGCG" * 150)
    contigs = {
        "a": seq_zero,
        "b": seq_zero,
        "c": seq_one,
        "d": seq_one,
        "e": seq_one,
    }
    coverage = {"a": 35.0, "b": 33.0, "c": 11.0, "d": 10.0, "e": 9.5}
    coarse = {"a": "0", "b": "0", "c": "1", "d": "1", "e": "0"}
    rows: list[dict[str, object]] = []
    contact_id = 0
    for _ in range(50):
        rows.append(
            {
                "contact_id": contact_id,
                "contigs": ["a", "b"],
                "contig_weights": [0.5, 0.5],
                "k": 2,
                "k_eff": 2.0,
                "weight": 1.0,
            }
        )
        contact_id += 1
    for _ in range(50):
        rows.append(
            {
                "contact_id": contact_id,
                "contigs": ["c", "d"],
                "contig_weights": [0.5, 0.5],
                "k": 2,
                "k_eff": 2.0,
                "weight": 1.0,
            }
        )
        contact_id += 1
    for _ in range(35):
        rows.append(
            {
                "contact_id": contact_id,
                "contigs": ["e", "c"],
                "contig_weights": [0.5, 0.5],
                "k": 2,
                "k_eff": 2.0,
                "weight": 1.0,
            }
        )
        contact_id += 1
    for _ in range(35):
        rows.append(
            {
                "contact_id": contact_id,
                "contigs": ["e", "d"],
                "contig_weights": [0.5, 0.5],
                "k": 2,
                "k_eff": 2.0,
                "weight": 1.0,
            }
        )
        contact_id += 1
    for _ in range(6):
        rows.append(
            {
                "contact_id": contact_id,
                "contigs": ["e", "a"],
                "contig_weights": [0.5, 0.5],
                "k": 2,
                "k_eff": 2.0,
                "weight": 1.0,
            }
        )
        contact_id += 1
    return write_generic_refine_fixture(
        tmp_path,
        contig_sequences=contigs,
        coverage_by_contig=coverage,
        coarse_assignment=coarse,
        contacts_rows=rows,
    )


def write_ambiguous_refine_fixture(tmp_path: Path) -> dict[str, Path]:
    seq_zero = ("AAAACCCCGGGGTTTT" * 150)
    seq_one = ("ATATATATCGCGCGCG" * 150)
    seq_ambiguous = ("GGGGAAAATTTTCCCC" * 150)
    contigs = {
        "a": seq_zero,
        "b": seq_zero,
        "c": seq_one,
        "d": seq_one,
        "x": seq_ambiguous,
    }
    coverage = {"a": 35.0, "b": 34.0, "c": 12.0, "d": 11.0, "x": 24.0}
    coarse = {"a": "0", "b": "0", "c": "1", "d": "1", "x": "0"}
    rows: list[dict[str, object]] = []
    contact_id = 0
    for _ in range(40):
        rows.append(
            {
                "contact_id": contact_id,
                "contigs": ["a", "b"],
                "contig_weights": [0.5, 0.5],
                "k": 2,
                "k_eff": 2.0,
                "weight": 1.0,
            }
        )
        contact_id += 1
    for _ in range(40):
        rows.append(
            {
                "contact_id": contact_id,
                "contigs": ["c", "d"],
                "contig_weights": [0.5, 0.5],
                "k": 2,
                "k_eff": 2.0,
                "weight": 1.0,
            }
        )
        contact_id += 1
    for _ in range(10):
        rows.append(
            {
                "contact_id": contact_id,
                "contigs": ["x", "a"],
                "contig_weights": [0.5, 0.5],
                "k": 2,
                "k_eff": 2.0,
                "weight": 1.0,
            }
        )
        contact_id += 1
    for _ in range(10):
        rows.append(
            {
                "contact_id": contact_id,
                "contigs": ["x", "c"],
                "contig_weights": [0.5, 0.5],
                "k": 2,
                "k_eff": 2.0,
                "weight": 1.0,
            }
        )
        contact_id += 1
    return write_generic_refine_fixture(
        tmp_path,
        contig_sequences=contigs,
        coverage_by_contig=coverage,
        coarse_assignment=coarse,
        contacts_rows=rows,
    )


def write_recruit_refine_fixture(tmp_path: Path) -> dict[str, Path]:
    seq_zero = ("AAAACCCCGGGGTTTT" * 150)
    contigs = {
        "a": seq_zero,
        "b": seq_zero,
        "h": seq_zero,
    }
    coverage = {"a": 40.0, "b": 38.0, "h": 39.0}
    coarse = {"a": "0", "b": "0"}
    rows: list[dict[str, object]] = []
    contact_id = 0
    for _ in range(60):
        rows.append(
            {
                "contact_id": contact_id,
                "contigs": ["a", "b"],
                "contig_weights": [0.5, 0.5],
                "k": 2,
                "k_eff": 2.0,
                "weight": 1.0,
            }
        )
        contact_id += 1
    for _ in range(18):
        rows.append(
            {
                "contact_id": contact_id,
                "contigs": ["h", "a"],
                "contig_weights": [0.5, 0.5],
                "k": 2,
                "k_eff": 2.0,
                "weight": 1.0,
            }
        )
        contact_id += 1
    for _ in range(18):
        rows.append(
            {
                "contact_id": contact_id,
                "contigs": ["h", "b"],
                "contig_weights": [0.5, 0.5],
                "k": 2,
                "k_eff": 2.0,
                "weight": 1.0,
            }
        )
        contact_id += 1
    return write_generic_refine_fixture(
        tmp_path,
        contig_sequences=contigs,
        coverage_by_contig=coverage,
        coarse_assignment=coarse,
        contacts_rows=rows,
    )
