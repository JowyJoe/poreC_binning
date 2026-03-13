from __future__ import annotations

from pathlib import Path


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        for name, seq in records:
            fh.write(f">{name}\n")
            fh.write(f"{seq}\n")


def test_refine_mvp_outputs_and_core_only_bins(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from porebin.refine import refine_bins_parquet

    contigs_fasta = tmp_path / "contigs.fasta"
    contacts_parquet = tmp_path / "contacts.parquet"
    bins_tsv = tmp_path / "bins.tsv"
    out_dir = tmp_path / "refined"

    # 3 candidate host bins: 0,1,2.
    # - X/Y are coarse -1 but have contact support (accessory-like association head).
    # - Z is coarse -1 with no contact support (unresolved_no_contact).
    contigs = [
        ("A", "ACGT" * 800),
        ("B", "ACGT" * 800),
        ("C", "TGCA" * 800),
        ("D", "TGCA" * 800),
        ("E", "AAAA" * 800),
        ("F", "TTTT" * 800),
        ("X", "CCCC" * 800),
        ("Y", "CCCC" * 800),
        ("Z", "CCCC" * 800),
    ]
    _write_fasta(contigs_fasta, contigs)

    with bins_tsv.open("w", encoding="utf-8", newline="") as fh:
        fh.write("contig_name\tbin_id\n")
        fh.write("A\t0\n")
        fh.write("B\t0\n")
        fh.write("C\t1\n")
        fh.write("D\t1\n")
        fh.write("E\t2\n")
        fh.write("F\t2\n")
        fh.write("X\t-1\n")
        fh.write("Y\t-1\n")
        fh.write("Z\t-1\n")
        # This contig is not in the FASTA and must not pollute candidate host set B.
        fh.write("G\t999\n")

    # Contacts:
    # - Within-bin evidence makes A..F core-like.
    # - X co-occurs with one contig from each bin, producing broad support (accessory-like).
    rows_contigs: list[list[str]] = []
    rows_pi: list[list[float]] = []
    rows_q: list[float] = []

    def add_read(cs: list[str], ws: list[float], q: float = 1.0, n: int = 1) -> None:
        for _ in range(int(n)):
            rows_contigs.append(list(cs))
            rows_pi.append(list(ws))
            rows_q.append(float(q))

    add_read(["A", "B"], [0.5, 0.5], n=50)
    add_read(["C", "D"], [0.5, 0.5], n=50)
    add_read(["E", "F"], [0.5, 0.5], n=50)
    add_read(["X", "A", "C", "E"], [0.7, 0.1, 0.1, 0.1], n=10)
    # Y has contact support to exactly two candidate hosts (0 and 1), so top_hosts should be 2 (no zero-padding).
    add_read(["Y", "A", "C"], [0.8, 0.1, 0.1], n=10)

    table = pa.table(
        {
            "contigs": pa.array(rows_contigs, type=pa.list_(pa.string())),
            "contig_weights": pa.array(rows_pi, type=pa.list_(pa.float64())),
            "weight": pa.array(rows_q, type=pa.float64()),
        }
    )
    pq.write_table(table, contacts_parquet)

    refined_bins = refine_bins_parquet(
        contigs_fasta=contigs_fasta,
        contacts_parquet=contacts_parquet,
        bins_tsv=bins_tsv,
        coverage_tsv=None,
        out_dir=out_dir,
    )
    assert refined_bins.name == "bins.refined.tsv"

    scores_path = out_dir / "contig_host_scores.tsv"
    assoc_path = out_dir / "accessory_associations.tsv"

    assert scores_path.exists()
    assert assoc_path.exists()
    assert (out_dir / "run_refine.json").exists()

    # contig_host_scores.tsv: must include required columns, for all contigs.
    scores_lines = scores_path.read_text(encoding="utf-8").splitlines()
    header = scores_lines[0].split("\t")
    assert header == [
        "contig_name",
        "coarse_host",
        "bin_status",
        "top1_host",
        "top2_host",
        "top1_score",
        "top2_score",
        "margin",
        "entropy",
        "effective_hosts",
        "has_contact_support",
        "is_core_like",
        "is_ambiguous",
        "is_accessory_candidate",
    ]
    by_contig: dict[str, dict[str, str]] = {}
    for line in scores_lines[1:]:
        row = line.split("\t")
        by_contig[row[0]] = {header[i]: row[i] for i in range(len(header))}
    assert set(by_contig.keys()) == {name for name, _seq in contigs}

    core_like = {c for c, d in by_contig.items() if d["is_core_like"] == "1"}
    accessory = {c for c, d in by_contig.items() if d["is_accessory_candidate"] == "1"}

    assert {"X", "Y"}.issubset(accessory)
    assert "Z" not in accessory
    assert {"X", "Y", "Z"}.isdisjoint(core_like)
    assert {"A", "B", "C", "D", "E", "F"}.issubset(core_like)

    # Z: no contact support and coarse -1 => unresolved_no_contact behavior (not accessory).
    assert by_contig["Z"]["has_contact_support"] == "0"
    assert by_contig["Z"]["top1_host"] == "-1"
    assert by_contig["Z"]["top2_host"] == "-1"
    assert by_contig["Z"]["top1_score"] == "0"
    assert by_contig["Z"]["top2_score"] == "0"
    assert by_contig["Z"]["is_accessory_candidate"] == "0"
    assert by_contig["Z"]["is_ambiguous"] == "1"

    # bins.refined.tsv must contain only core-like contigs.
    bins_lines = (out_dir / "bins.refined.tsv").read_text(encoding="utf-8").splitlines()
    bins_contigs = {ln.split("\t")[0] for ln in bins_lines[1:] if ln.strip()}
    assert bins_contigs.issubset(core_like)

    # accessory_associations.tsv columns must exist; contigs listed must be accessory candidates.
    assoc_lines = assoc_path.read_text(encoding="utf-8").splitlines()
    assoc_header = assoc_lines[0].split("\t")
    assert assoc_header == [
        "contig_name",
        "top_hosts",
        "host_weights",
        "host_entropy",
        "effective_hosts",
        "association_confidence",
        "single_host_like",
        "broad_host_like",
    ]
    assoc_contigs = {ln.split("\t")[0] for ln in assoc_lines[1:] if ln.strip()}
    assert assoc_contigs == accessory

    by_assoc: dict[str, dict[str, str]] = {}
    for line in assoc_lines[1:]:
        row = line.split("\t")
        by_assoc[row[0]] = {assoc_header[i]: row[i] for i in range(len(assoc_header))}
    assert "Z" not in by_assoc

    # Y should not be padded with a zero-score host.
    y_hosts = [h for h in by_assoc["Y"]["top_hosts"].split(",") if h]
    assert len(y_hosts) == 2
    assert "999" not in y_hosts
    assert "2" not in y_hosts

    # run_refine.json: candidate host set excludes bins from contigs not in FASTA; coarse prior is not mislabeled as feature.
    import json

    run_refine = json.loads((out_dir / "run_refine.json").read_text(encoding="utf-8"))
    assert run_refine["stats"]["candidate_bins_count"] == 3
    decisions = run_refine["decisions"]
    assert "feature_prior_used" not in decisions
    assert "feature_prior_strength" not in decisions
    assert decisions["coarse_prior_used"] is True
    assert abs(float(decisions["coarse_prior_strength"]) - 0.05) < 1e-12
    assert decisions["direct_feature_prior_used"] is False


def test_refine_soft_gating_keeps_weak_bin_as_candidate_host(tmp_path: Path) -> None:
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    from porebin.refine import refine_bins_parquet

    contigs_fasta = tmp_path / "contigs.fasta"
    contacts_parquet = tmp_path / "contacts.parquet"
    bins_tsv = tmp_path / "bins.tsv"
    out_dir = tmp_path / "refined"

    contigs = [
        ("A", "ACGT" * 800),
        ("B", "ACGT" * 800),
        ("C", "TGCA" * 800),
        ("D", "TGCA" * 800),
        ("X", "CCCC" * 800),
    ]
    _write_fasta(contigs_fasta, contigs)

    with bins_tsv.open("w", encoding="utf-8", newline="") as fh:
        fh.write("contig_name\tbin_id\n")
        fh.write("A\t0\n")
        fh.write("B\t0\n")
        fh.write("C\t1\n")
        fh.write("D\t1\n")
        fh.write("X\t-1\n")

    rows_contigs: list[list[str]] = []
    rows_pi: list[list[float]] = []
    rows_q: list[float] = []

    def add_read(cs: list[str], ws: list[float], q: float = 1.0, n: int = 1) -> None:
        for _ in range(int(n)):
            rows_contigs.append(list(cs))
            rows_pi.append(list(ws))
            rows_q.append(float(q))

    # Bin 0 becomes strong through clean within-bin support.
    add_read(["A", "B"], [0.5, 0.5], n=50)
    # Bin 1 becomes weak: C and D have some within-bin support, but D is strongly cross-linked to bin 0.
    add_read(["C", "D"], [0.5, 0.5], n=20)
    add_read(["D", "A"], [0.5, 0.5], n=50)
    # Unbinned X only links to C from the weak bin, so soft gating should still expose host 1.
    add_read(["X", "C"], [0.8, 0.2], n=20)

    table = pa.table(
        {
            "contigs": pa.array(rows_contigs, type=pa.list_(pa.string())),
            "contig_weights": pa.array(rows_pi, type=pa.list_(pa.float64())),
            "weight": pa.array(rows_q, type=pa.float64()),
        }
    )
    pq.write_table(table, contacts_parquet)

    refine_bins_parquet(
        contigs_fasta=contigs_fasta,
        contacts_parquet=contacts_parquet,
        bins_tsv=bins_tsv,
        coverage_tsv=None,
        out_dir=out_dir,
    )

    scores_lines = (out_dir / "contig_host_scores.tsv").read_text(encoding="utf-8").splitlines()
    header = scores_lines[0].split("\t")
    by_contig: dict[str, dict[str, str]] = {}
    for line in scores_lines[1:]:
        row = line.split("\t")
        by_contig[row[0]] = {header[i]: row[i] for i in range(len(header))}

    assert by_contig["X"]["has_contact_support"] == "1"
    assert by_contig["X"]["top1_host"] == "1"

    run_refine = json.loads((out_dir / "run_refine.json").read_text(encoding="utf-8"))
    assert run_refine["stats"]["candidate_bins_count"] == 2
    decisions = run_refine["decisions"]
    assert decisions["soft_gating_enabled"] is True
    assert decisions["candidate_bin_rule"] == "non_impure_bins"
    assert decisions["bin_weight_scheme"]["weak"] == 0.35
