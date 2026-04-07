from __future__ import annotations

from pathlib import Path


def _write_fasta(path: Path, records: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        for name, seq in records:
            fh.write(f">{name}\n")
            fh.write(f"{seq}\n")


def _write_scg_hits(path: Path, rows: list[tuple[str, str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(
            "contig_name\tgene_id\tmarker_id\tbitscore\tevalue\thmm_from\thmm_to\tali_from\tali_to\t"
            "env_from\tenv_to\torf_start\torf_end\tstrand\n"
        )
        for idx, (contig_name, gene_id, marker_id) in enumerate(rows, start=1):
            start = 100 * idx
            end = start + 90
            fh.write(
                f"{contig_name}\t{gene_id}\t{marker_id}\t100\t1e-20\t1\t10\t1\t10\t1\t10\t{start}\t{end}\t+\n"
            )


def _patch_enabled_scg(monkeypatch, tmp_path: Path, rows: list[tuple[str, str, str]], expected_markers: list[str]) -> None:
    from porebin import refine as refine_mod
    from porebin.scg import ScgResources, ScgRunResult

    monkeypatch.setattr(refine_mod, "_preflight_refine_scg_requirements", lambda **_kwargs: None)

    db_dir = tmp_path / "scg_db"
    db_dir.mkdir(parents=True, exist_ok=True)
    marker_hmm = db_dir / "marker.hmm"
    manifest_json = db_dir / "manifest.json"
    marker_hmm.write_text("HMMER3/f\nNAME  TEST\n//\n", encoding="utf-8")
    manifest_json.write_text(
        '{"marker_set_id":"test_scg","db_version":"0.1","expected_markers":["'
        + '","'.join(expected_markers)
        + '"]}',
        encoding="utf-8",
    )

    hits_tsv = tmp_path / "scg_hits.tsv"
    _write_scg_hits(hits_tsv, rows)
    resources = ScgResources(
        db_dir=db_dir,
        marker_hmm=marker_hmm,
        manifest_json=manifest_json,
        marker_set_id="test_scg",
        db_version="0.1",
        expected_markers=tuple(expected_markers),
    )
    result = ScgRunResult(
        state="enabled",
        enabled=True,
        hits_tsv=hits_tsv,
        proteins_faa=None,
        domtblout=None,
        resources=resources,
        expected_markers=tuple(expected_markers),
        reused_cache=True,
        reason="test_enabled_scg",
    )
    monkeypatch.setattr(refine_mod, "ensure_scg_hits", lambda **_kwargs: result)


def test_legacy_refine_entrypoint_validates_inputs(tmp_path: Path) -> None:
    import pytest

    from porebin.refine import refine_bins

    contigs_fasta = tmp_path / "contigs.fasta"
    bins_tsv = tmp_path / "bins.tsv"

    _write_fasta(contigs_fasta, [("A", "ACGT" * 10)])
    bins_tsv.write_text("contig_name\tbin_id\nA\t0\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="PPL contacts not found"):
        refine_bins(
            contigs_fasta=contigs_fasta,
            ppl_contacts=tmp_path / "missing.contacts",
            bins_tsv=bins_tsv,
            bam=None,
            out_dir=tmp_path / "refined",
        )


def test_refine_mvp_outputs_and_core_only_bins(tmp_path: Path, monkeypatch) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from porebin.associate import associate_residuals
    from porebin.refine import refine_bins_parquet

    contigs_fasta = tmp_path / "contigs.fasta"
    contacts_parquet = tmp_path / "contacts.parquet"
    bins_tsv = tmp_path / "bins.tsv"
    out_dir = tmp_path / "refined"

    # 3 candidate host bins: 0,1,2.
    # - X/Y are coarse -1 but have contact support and should be exported to the residual pool
    #   as downstream association candidates.
    # - Z is coarse -1 with no contact support and should remain residual-unresolved.
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

    _patch_enabled_scg(
        monkeypatch,
        tmp_path,
        rows=[],
        expected_markers=["SCG_A", "SCG_B", "SCG_C", "SCG_D"],
    )

    refined_bins = refine_bins_parquet(
        contigs_fasta=contigs_fasta,
        contacts_parquet=contacts_parquet,
        bins_tsv=bins_tsv,
        coverage_tsv=None,
        out_dir=out_dir,
    )
    assert refined_bins.name == "bins.refined.tsv"

    scores_path = out_dir / "contig_host_scores.tsv"
    residual_pool_path = out_dir / "residual_pool.tsv"
    refine_actions_path = out_dir / "refine_actions.tsv"
    bin_qc_refined_path = out_dir / "bin_qc.refined.tsv"

    assert scores_path.exists()
    assert residual_pool_path.exists()
    assert refine_actions_path.exists()
    assert (out_dir / "run_refine.json").exists()
    assert bin_qc_refined_path.exists()
    assert not (out_dir / "accessory_associations.tsv").exists()

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
        "refine_state",
        "abstain_reason",
        "informative_reads",
        "read_reject_anchor_sparse",
        "read_reject_anchor_conflict",
        "is_hard_anchor",
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
    assert by_contig["X"]["refine_state"] == "relation_only"
    assert by_contig["Y"]["refine_state"] == "relation_only"
    assert by_contig["Z"]["refine_state"] == "abstain_insufficient_information"

    # Z: no contact support and coarse -1 => unresolved_no_contact behavior (not accessory).
    assert by_contig["Z"]["has_contact_support"] == "0"
    assert by_contig["Z"]["top1_host"] == "-1"
    assert by_contig["Z"]["top2_host"] == "-1"
    assert by_contig["Z"]["top1_score"] == "0"
    assert by_contig["Z"]["top2_score"] == "0"
    assert by_contig["Z"]["is_accessory_candidate"] == "0"
    assert by_contig["Z"]["is_ambiguous"] == "1"
    assert by_contig["Z"]["abstain_reason"] == "insufficient_information"

    # bins.refined.tsv must contain only core-like contigs.
    bins_lines = (out_dir / "bins.refined.tsv").read_text(encoding="utf-8").splitlines()
    bins_contigs = {ln.split("\t")[0] for ln in bins_lines[1:] if ln.strip()}
    assert bins_contigs.issubset(core_like)

    # residual_pool.tsv is the formal refine handoff to downstream association.
    residual_lines = residual_pool_path.read_text(encoding="utf-8").splitlines()
    residual_header = residual_lines[0].split("\t")
    assert residual_header == [
        "contig_name",
        "reason",
        "stage",
        "coarse_bin_id",
        "refined_status",
        "note",
    ]
    residual_by_contig: dict[str, dict[str, str]] = {}
    for line in residual_lines[1:]:
        row = line.split("\t")
        residual_by_contig[row[0]] = {residual_header[i]: row[i] for i in range(len(residual_header))}
    assert set(residual_by_contig.keys()) == {"X", "Y", "Z"}
    assert residual_by_contig["X"]["refined_status"] == "residual_associate_candidate"
    assert residual_by_contig["Y"]["refined_status"] == "residual_associate_candidate"
    assert residual_by_contig["X"]["stage"] == "refine_base"
    assert residual_by_contig["Y"]["stage"] == "refine_base"
    assert residual_by_contig["X"]["reason"] == "associate_candidate"
    assert residual_by_contig["Y"]["reason"] == "associate_candidate"
    assert residual_by_contig["Z"]["refined_status"] == "residual_unresolved"
    assert residual_by_contig["Z"]["stage"] == "recruit"
    assert residual_by_contig["Z"]["reason"] == "recruit_unresolved"

    qc_lines = bin_qc_refined_path.read_text(encoding="utf-8").splitlines()
    qc_header = qc_lines[0].split("\t")
    assert qc_header[:12] == [
        "bin_id",
        "total_bp",
        "n_contigs",
        "contact_consistency",
        "coverage_dispersion",
        "tnf_dispersion",
        "scg_status",
        "unique_scg",
        "duplicated_scg",
        "completeness_like",
        "contamination_like",
        "suspect_flag",
    ]
    assert qc_header[12:16] == [
        "suspect_reasons",
        "split_check_flag",
        "split_check_reasons",
        "split_priority_source",
    ]
    qc_first = qc_lines[1].split("\t")
    assert qc_first[6] == "enabled"
    assert qc_first[7] == "0"
    assert qc_first[8] == "0"
    assert qc_first[9] == "0"
    assert qc_first[10] == "0"
    assert qc_first[13] == "0"
    assert qc_first[14] == ""
    assert qc_first[15] == ""

    action_lines = refine_actions_path.read_text(encoding="utf-8").splitlines()
    action_header = action_lines[0].split("\t")
    assert action_header == [
        "action_id",
        "action_type",
        "target_bin",
        "source_bin",
        "affected_contigs",
        "accepted",
        "accept_reason",
        "reject_reason",
        "qc_before_ref",
        "qc_after_ref",
        "scg_gate_used",
        "scg_gate_result",
        "scg_gate_reason",
    ]
    assert len(action_lines) >= 2
    recruit_rows = [line.split("\t") for line in action_lines[1:] if line.split("\t")[1] == "recruit"]
    assert len(recruit_rows) == 1
    assert recruit_rows[0][2] == "-1"
    assert recruit_rows[0][5] == "1"
    assert recruit_rows[0][6] == "no_positive_recruit_support"
    assert recruit_rows[0][10] == "0"
    assert recruit_rows[0][11] == "not_applicable"

    # run_refine.json: candidate host set excludes bins from contigs not in FASTA; coarse prior is not mislabeled as feature.
    import json

    run_refine = json.loads((out_dir / "run_refine.json").read_text(encoding="utf-8"))
    assert run_refine["stats"]["candidate_bins_count"] == 3
    assert run_refine["stats"]["refine_actions_scg_evaluated"] == 0
    assert run_refine["stats"]["refine_actions_scg_vetoed"] == 0
    assert run_refine["stats"]["refine_actions_scg_supported"] == 0
    assert run_refine["stats"]["split_check_bins_refined"] == 0
    assert run_refine["stats"]["split_check_bins_refined_scg_priority"] == 0
    assert run_refine["stats"]["reassign_windows_total"] == 0
    assert run_refine["stats"]["reassign_candidates_total"] == 0
    assert run_refine["stats"]["reassign_decisions_accepted"] == 0
    assert run_refine["stats"]["recruit_candidates_total"] == 1
    assert run_refine["stats"]["recruit_candidates_assign_to_bin"] == 0
    assert run_refine["stats"]["recruit_decisions_accepted"] == 1
    assert run_refine["stats"]["recruit_decisions_assigned"] == 0
    assert run_refine["stats"]["recruit_decisions_kept_residual"] == 1
    decisions = run_refine["decisions"]
    assert "feature_prior_used" not in decisions
    assert "feature_prior_strength" not in decisions
    assert decisions["coarse_prior_used"] is True
    assert abs(float(decisions["coarse_prior_strength"]) - 0.05) < 1e-12
    assert decisions["direct_feature_prior_used"] is False
    assert decisions["conservative_binning_role"] == "conservative_host_binning"
    assert decisions["refine_orchestration_model"] == "bin_centric_live_split_reassign_recruit"
    assert decisions["residual_pool_role"] == "explicit_handoff_to_associate"
    assert decisions["associate_module_expected"] is True
    assert decisions["legacy_accessory_output_removed_from_refine"] is True
    assert decisions["legacy_contig_state_role"] == "compatibility_bridge_only_not_public_refine_semantics"
    assert decisions["legacy_helper_roles"] == {
        "_split_bins_parquet": "candidate_generator_only_not_applied_in_active_orchestration",
        "_reassign_or_unbin": "compatibility_helper_not_in_active_orchestration",
        "_recruit_gmm": "compatibility_helper_not_in_active_orchestration",
        "_legacy_contig_state_bridge": "compatibility_bridge_for_scores_and_residual_mapping",
    }
    assert decisions["scg_enabled"] is True
    assert decisions["scg_state"] == "enabled"
    assert decisions["scg_resources_loaded"] is True
    assert decisions["scg_expected_markers_count"] == 4
    assert decisions["split_live_operation_enabled"] is True
    assert decisions["recruit_live_operation_enabled"] is True

    assoc_out = tmp_path / "associate"
    assoc_path = associate_residuals(
        bins_refined_tsv=out_dir / "bins.refined.tsv",
        residual_pool_tsv=residual_pool_path,
        contacts_parquet=contacts_parquet,
        out_dir=assoc_out,
        contig_scores_tsv=scores_path,
    )
    assoc_lines = assoc_path.read_text(encoding="utf-8").splitlines()
    assoc_header = assoc_lines[0].split("\t")
    assert assoc_header == [
        "contig_name",
        "association_type",
        "host_bin_ids",
        "primary_host_bin_id",
        "support_strength",
        "support_class",
        "uncertainty",
        "evidence_summary",
    ]
    assoc_by_contig: dict[str, dict[str, str]] = {}
    for line in assoc_lines[1:]:
        row = line.split("\t")
        assoc_by_contig[row[0]] = {assoc_header[i]: row[i] for i in range(len(assoc_header))}
    assert set(assoc_by_contig.keys()) == {"X", "Y"}
    assert "Z" not in assoc_by_contig
    y_hosts = [h for h in assoc_by_contig["Y"]["host_bin_ids"].split(",") if h]
    assert len(y_hosts) == 2
    assert "999" not in y_hosts
    assert "2" not in y_hosts
    assert assoc_by_contig["X"]["evidence_summary"] == "compat_from_contig_host_scores"


def test_refine_live_split_accepts_and_updates_outputs(tmp_path: Path, monkeypatch) -> None:
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq

    from porebin.refine import refine_bins_parquet

    contigs_fasta = tmp_path / "contigs.fasta"
    contacts_parquet = tmp_path / "contacts.parquet"
    bins_tsv = tmp_path / "bins.tsv"
    out_dir = tmp_path / "refined"

    contigs = [
        ("A", "AAAACCCCGGGGTTTT" * 200),
        ("B", "AAAACCCCGGGGTTTT" * 200),
        ("C", "ATATATATCGCGCGCG" * 200),
        ("D", "ATATATATCGCGCGCG" * 200),
    ]
    _write_fasta(contigs_fasta, contigs)
    bins_tsv.write_text(
        "contig_name\tbin_id\nA\t0\nB\t0\nC\t0\nD\t0\n",
        encoding="utf-8",
    )

    rows_contigs: list[list[str]] = []
    rows_pi: list[list[float]] = []
    rows_q: list[float] = []

    def add_read(cs: list[str], ws: list[float], n: int) -> None:
        for _ in range(n):
            rows_contigs.append(list(cs))
            rows_pi.append(list(ws))
            rows_q.append(1.0)

    add_read(["A", "B"], [0.5, 0.5], 40)
    add_read(["C", "D"], [0.5, 0.5], 40)
    pq.write_table(
        pa.table(
            {
                "contigs": pa.array(rows_contigs, type=pa.list_(pa.string())),
                "contig_weights": pa.array(rows_pi, type=pa.list_(pa.float64())),
                "weight": pa.array(rows_q, type=pa.float64()),
            }
        ),
        contacts_parquet,
    )

    _patch_enabled_scg(
        monkeypatch,
        tmp_path,
        rows=[
            ("A", "g1", "SCG_A"),
            ("B", "g2", "SCG_B"),
            ("C", "g3", "SCG_A"),
            ("D", "g4", "SCG_B"),
        ],
        expected_markers=["SCG_A", "SCG_B", "SCG_C", "SCG_D"],
    )

    refine_bins_parquet(
        contigs_fasta=contigs_fasta,
        contacts_parquet=contacts_parquet,
        bins_tsv=bins_tsv,
        coverage_tsv=None,
        out_dir=out_dir,
    )

    bins_lines = (out_dir / "bins.refined.tsv").read_text(encoding="utf-8").splitlines()[1:]
    assignment = dict(line.split("\t") for line in bins_lines if line.strip())
    assert assignment["A"] == "0"
    assert assignment["B"] == "0"
    assert assignment["C"] != "0"
    assert assignment["D"] == assignment["C"]

    residual_lines = (out_dir / "residual_pool.tsv").read_text(encoding="utf-8").splitlines()
    residual_by_contig = {line.split("\t")[0]: line.split("\t") for line in residual_lines[1:] if line.strip()}
    assert residual_by_contig == {}

    action_lines = (out_dir / "refine_actions.tsv").read_text(encoding="utf-8").splitlines()
    split_rows = [line.split("\t") for line in action_lines[1:] if line.split("\t")[1] == "split"]
    assert len(split_rows) == 1
    assert split_rows[0][5] == "1"
    assert "local_recluster_partition" in split_rows[0][6]
    assert split_rows[0][11] == "support"

    run_refine = json.loads((out_dir / "run_refine.json").read_text(encoding="utf-8"))
    assert run_refine["stats"]["split_candidates_total"] == 1
    assert run_refine["stats"]["split_candidates_accepted"] == 1
    assert run_refine["stats"]["split_candidates_rejected"] == 0
    assert run_refine["stats"]["split_candidates_local_recluster"] == 1
    assert run_refine["stats"]["split_candidates_scg_priority"] == 1
    assert run_refine["stats"]["split_residual_contigs"] == 0
    assert run_refine["stats"]["reassign_windows_total"] == 1
    assert run_refine["stats"]["reassign_candidates_total"] == 0
    assert run_refine["stats"]["reassign_decisions_accepted"] == 0
    assert run_refine["stats"]["recruit_candidates_total"] == 0
    assert run_refine["stats"]["recruit_decisions_assigned"] == 0


def test_refine_reassign_moves_boundary_contig_to_sibling(tmp_path: Path, monkeypatch) -> None:
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq

    from porebin.refine import refine_bins_parquet

    contigs_fasta = tmp_path / "contigs.fasta"
    contacts_parquet = tmp_path / "contacts.parquet"
    bins_tsv = tmp_path / "bins.tsv"
    coverage_tsv = tmp_path / "coverage.tsv"
    out_dir = tmp_path / "refined"

    contigs = [
        ("A", "AAAACCCCGGGGTTTT" * 200),
        ("B", "AAAACCCCGGGGTTTT" * 200),
        ("C", "ATATATATCGCGCGCG" * 200),
        ("D", "ATATATATCGCGCGCG" * 200),
        ("E", "AAAACCCCGGGGTTTT" * 200),
    ]
    _write_fasta(contigs_fasta, contigs)
    bins_tsv.write_text(
        "contig_name\tbin_id\nA\t0\nB\t0\nC\t0\nD\t0\nE\t0\n",
        encoding="utf-8",
    )
    coverage_tsv.write_text(
        "contig_name\tcoverage\nA\t40\nB\t30\nC\t10\nD\t10\nE\t30\n",
        encoding="utf-8",
    )

    rows_contigs: list[list[str]] = []
    rows_pi: list[list[float]] = []
    rows_q: list[float] = []

    def add_read(cs: list[str], ws: list[float], n: int) -> None:
        for _ in range(n):
            rows_contigs.append(list(cs))
            rows_pi.append(list(ws))
            rows_q.append(1.0)

    add_read(["A", "B"], [0.5, 0.5], 100)
    add_read(["C", "D"], [0.5, 0.5], 40)
    add_read(["E", "C"], [0.5, 0.5], 30)
    add_read(["E", "D"], [0.5, 0.5], 30)
    pq.write_table(
        pa.table(
            {
                "contigs": pa.array(rows_contigs, type=pa.list_(pa.string())),
                "contig_weights": pa.array(rows_pi, type=pa.list_(pa.float64())),
                "weight": pa.array(rows_q, type=pa.float64()),
            }
        ),
        contacts_parquet,
    )

    _patch_enabled_scg(
        monkeypatch,
        tmp_path,
        rows=[
            ("A", "g1", "SCG_A"),
            ("B", "g2", "SCG_B"),
            ("C", "g3", "SCG_A"),
            ("D", "g4", "SCG_C"),
        ],
        expected_markers=["SCG_A", "SCG_B", "SCG_C", "SCG_D"],
    )

    refine_bins_parquet(
        contigs_fasta=contigs_fasta,
        contacts_parquet=contacts_parquet,
        bins_tsv=bins_tsv,
        coverage_tsv=coverage_tsv,
        out_dir=out_dir,
    )

    bins_lines = (out_dir / "bins.refined.tsv").read_text(encoding="utf-8").splitlines()[1:]
    assignment = dict(line.split("\t") for line in bins_lines if line.strip())
    assert assignment["A"] == "0"
    assert assignment["B"] == "0"
    assert assignment["C"] != "0"
    assert assignment["D"] == assignment["C"]
    assert assignment["E"] == assignment["C"]

    residual_lines = (out_dir / "residual_pool.tsv").read_text(encoding="utf-8").splitlines()
    assert len(residual_lines) == 1

    action_lines = (out_dir / "refine_actions.tsv").read_text(encoding="utf-8").splitlines()
    reassign_rows = [line.split("\t") for line in action_lines[1:] if line.split("\t")[1] == "reassign"]
    assert len(reassign_rows) == 1
    assert reassign_rows[0][3] == "0"
    assert reassign_rows[0][5] == "1"
    assert reassign_rows[0][6].startswith("contact_support_reassigned")
    assert reassign_rows[0][11] in {"neutral", "support"}

    run_refine = json.loads((out_dir / "run_refine.json").read_text(encoding="utf-8"))
    assert run_refine["stats"]["reassign_windows_total"] == 1
    assert run_refine["stats"]["reassign_candidates_total"] == 1
    assert run_refine["stats"]["reassign_candidates_move_to_sibling"] == 1
    assert run_refine["stats"]["reassign_candidates_residualize"] == 0
    assert run_refine["stats"]["reassign_decisions_accepted"] == 1
    assert run_refine["stats"]["reassign_decisions_moved"] == 1
    assert run_refine["stats"]["reassign_decisions_residualized"] == 0


def test_refine_reassign_residualizes_ambiguous_contig(tmp_path: Path, monkeypatch) -> None:
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq

    from porebin.refine import refine_bins_parquet

    contigs_fasta = tmp_path / "contigs.fasta"
    contacts_parquet = tmp_path / "contacts.parquet"
    bins_tsv = tmp_path / "bins.tsv"
    coverage_tsv = tmp_path / "coverage.tsv"
    out_dir = tmp_path / "refined"

    contigs = [
        ("A", "AAAACCCCGGGGTTTT" * 200),
        ("B", "AAAACCCCGGGGTTTT" * 200),
        ("C", "ATATATATCGCGCGCG" * 200),
        ("D", "ATATATATCGCGCGCG" * 200),
        ("E", "AAAACCCCGGGGTTTT" * 200),
    ]
    _write_fasta(contigs_fasta, contigs)
    bins_tsv.write_text(
        "contig_name\tbin_id\nA\t0\nB\t0\nC\t0\nD\t0\nE\t0\n",
        encoding="utf-8",
    )
    coverage_tsv.write_text(
        "contig_name\tcoverage\nA\t40\nB\t30\nC\t10\nD\t10\nE\t30\n",
        encoding="utf-8",
    )

    rows_contigs: list[list[str]] = []
    rows_pi: list[list[float]] = []
    rows_q: list[float] = []

    def add_read(cs: list[str], ws: list[float], n: int) -> None:
        for _ in range(n):
            rows_contigs.append(list(cs))
            rows_pi.append(list(ws))
            rows_q.append(1.0)

    add_read(["A", "B"], [0.5, 0.5], 40)
    add_read(["C", "D"], [0.5, 0.5], 40)
    add_read(["E", "A"], [0.5, 0.5], 10)
    add_read(["E", "C"], [0.5, 0.5], 10)
    pq.write_table(
        pa.table(
            {
                "contigs": pa.array(rows_contigs, type=pa.list_(pa.string())),
                "contig_weights": pa.array(rows_pi, type=pa.list_(pa.float64())),
                "weight": pa.array(rows_q, type=pa.float64()),
            }
        ),
        contacts_parquet,
    )

    _patch_enabled_scg(
        monkeypatch,
        tmp_path,
        rows=[
            ("A", "g1", "SCG_A"),
            ("B", "g2", "SCG_B"),
            ("C", "g3", "SCG_A"),
            ("D", "g4", "SCG_C"),
        ],
        expected_markers=["SCG_A", "SCG_B", "SCG_C", "SCG_D"],
    )

    refine_bins_parquet(
        contigs_fasta=contigs_fasta,
        contacts_parquet=contacts_parquet,
        bins_tsv=bins_tsv,
        coverage_tsv=coverage_tsv,
        out_dir=out_dir,
    )

    bins_lines = (out_dir / "bins.refined.tsv").read_text(encoding="utf-8").splitlines()[1:]
    assignment = dict(line.split("\t") for line in bins_lines if line.strip())
    assert "E" not in assignment

    residual_lines = (out_dir / "residual_pool.tsv").read_text(encoding="utf-8").splitlines()
    residual_by_contig = {line.split("\t")[0]: line.split("\t") for line in residual_lines[1:] if line.strip()}
    assert "E" in residual_by_contig
    assert residual_by_contig["E"][1] == "recruit_unresolved"
    assert residual_by_contig["E"][2] == "recruit"

    action_lines = (out_dir / "refine_actions.tsv").read_text(encoding="utf-8").splitlines()
    reassign_rows = [line.split("\t") for line in action_lines[1:] if line.split("\t")[1] == "reassign"]
    assert len(reassign_rows) == 1
    assert reassign_rows[0][5] == "1"
    assert reassign_rows[0][2] == "-1"
    assert reassign_rows[0][6].startswith("reassign_residualized")
    recruit_rows = [line.split("\t") for line in action_lines[1:] if line.split("\t")[1] == "recruit"]
    assert len(recruit_rows) == 1
    assert recruit_rows[0][5] == "1"
    assert recruit_rows[0][2] == "-1"
    assert recruit_rows[0][6] in {"no_positive_recruit_support", "recruit_support_not_dominant_enough"}

    run_refine = json.loads((out_dir / "run_refine.json").read_text(encoding="utf-8"))
    assert run_refine["stats"]["reassign_candidates_total"] == 1
    assert run_refine["stats"]["reassign_candidates_move_to_sibling"] == 0
    assert run_refine["stats"]["reassign_candidates_residualize"] == 1
    assert run_refine["stats"]["reassign_decisions_accepted"] == 1
    assert run_refine["stats"]["reassign_decisions_moved"] == 0
    assert run_refine["stats"]["reassign_decisions_residualized"] == 1
    assert run_refine["stats"]["recruit_candidates_total"] == 1
    assert run_refine["stats"]["recruit_decisions_assigned"] == 0
    assert run_refine["stats"]["recruit_decisions_kept_residual"] == 1


def test_refine_recruit_assigns_unresolved_residual_to_existing_bin(tmp_path: Path, monkeypatch) -> None:
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq

    from porebin.refine import refine_bins_parquet

    contigs_fasta = tmp_path / "contigs.fasta"
    contacts_parquet = tmp_path / "contacts.parquet"
    bins_tsv = tmp_path / "bins.tsv"
    coverage_tsv = tmp_path / "coverage.tsv"
    out_dir = tmp_path / "refined"

    contigs = [
        ("A", "AAAACCCCGGGGTTTT" * 200),
        ("B", "AAAACCCCGGGGTTTT" * 200),
        ("H", "GGGGAAAATTTTCCCC" * 200),
    ]
    _write_fasta(contigs_fasta, contigs)
    bins_tsv.write_text(
        "contig_name\tbin_id\nA\t0\nB\t0\nH\t-1\n",
        encoding="utf-8",
    )
    coverage_tsv.write_text("contig_name\tcoverage\nA\t40\nB\t35\nH\t30\n", encoding="utf-8")

    rows_contigs: list[list[str]] = []
    rows_pi: list[list[float]] = []
    rows_q: list[float] = []

    def add_read(cs: list[str], ws: list[float], n: int) -> None:
        for _ in range(n):
            rows_contigs.append(list(cs))
            rows_pi.append(list(ws))
            rows_q.append(1.0)

    add_read(["A", "B"], [0.5, 0.5], 60)
    # H only has one informative read, so base refine should leave it residual_unresolved
    # rather than assign it directly; recruit can then recover it into bin 0.
    add_read(["H", "A"], [0.5, 0.5], 1)
    pq.write_table(
        pa.table(
            {
                "contigs": pa.array(rows_contigs, type=pa.list_(pa.string())),
                "contig_weights": pa.array(rows_pi, type=pa.list_(pa.float64())),
                "weight": pa.array(rows_q, type=pa.float64()),
            }
        ),
        contacts_parquet,
    )

    _patch_enabled_scg(
        monkeypatch,
        tmp_path,
        rows=[
            ("A", "g1", "SCG_A"),
            ("B", "g2", "SCG_B"),
        ],
        expected_markers=["SCG_A", "SCG_B", "SCG_C", "SCG_D"],
    )

    refine_bins_parquet(
        contigs_fasta=contigs_fasta,
        contacts_parquet=contacts_parquet,
        bins_tsv=bins_tsv,
        coverage_tsv=coverage_tsv,
        out_dir=out_dir,
    )

    bins_lines = (out_dir / "bins.refined.tsv").read_text(encoding="utf-8").splitlines()[1:]
    assignment = dict(line.split("\t") for line in bins_lines if line.strip())
    assert assignment["A"] == "0"
    assert assignment["B"] == "0"
    assert assignment["H"] == "0"

    residual_lines = (out_dir / "residual_pool.tsv").read_text(encoding="utf-8").splitlines()
    residual_by_contig = {line.split("\t")[0]: line.split("\t") for line in residual_lines[1:] if line.strip()}
    assert "H" not in residual_by_contig

    action_lines = (out_dir / "refine_actions.tsv").read_text(encoding="utf-8").splitlines()
    recruit_rows = [line.split("\t") for line in action_lines[1:] if line.split("\t")[1] == "recruit"]
    assert len(recruit_rows) == 1
    assert recruit_rows[0][2] == "0"
    assert recruit_rows[0][5] == "1"
    assert recruit_rows[0][6].startswith("recruit_assigned_to_existing_bin")

    run_refine = json.loads((out_dir / "run_refine.json").read_text(encoding="utf-8"))
    assert run_refine["stats"]["recruit_candidates_total"] == 1
    assert run_refine["stats"]["recruit_candidates_assign_to_bin"] == 1
    assert run_refine["stats"]["recruit_decisions_assigned"] == 1
    assert run_refine["stats"]["recruit_decisions_kept_residual"] == 0


def test_refine_live_split_rejected_by_scg_gate(tmp_path: Path, monkeypatch) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    from porebin.refine import refine_bins_parquet

    contigs_fasta = tmp_path / "contigs.fasta"
    contacts_parquet = tmp_path / "contacts.parquet"
    bins_tsv = tmp_path / "bins.tsv"
    out_dir = tmp_path / "refined"

    contigs = [
        ("A", "AAAACCCCGGGGTTTT" * 200),
        ("B", "AAAACCCCGGGGTTTT" * 200),
        ("C", "ATATATATCGCGCGCG" * 200),
        ("D", "ATATATATCGCGCGCG" * 200),
    ]
    _write_fasta(contigs_fasta, contigs)
    bins_tsv.write_text(
        "contig_name\tbin_id\nA\t0\nB\t0\nC\t0\nD\t0\n",
        encoding="utf-8",
    )

    rows_contigs = [["A", "B"]] * 30 + [["C", "D"]] * 30
    rows_pi = [[0.5, 0.5]] * 60
    rows_q = [1.0] * 60
    pq.write_table(
        pa.table(
            {
                "contigs": pa.array(rows_contigs, type=pa.list_(pa.string())),
                "contig_weights": pa.array(rows_pi, type=pa.list_(pa.float64())),
                "weight": pa.array(rows_q, type=pa.float64()),
            }
        ),
        contacts_parquet,
    )

    _patch_enabled_scg(
        monkeypatch,
        tmp_path,
        rows=[
            ("A", "g1", "SCG_A"),
            ("B", "g2", "SCG_A"),
            ("C", "g3", "SCG_B"),
            ("D", "g4", "SCG_C"),
        ],
        expected_markers=["SCG_A", "SCG_B", "SCG_C", "SCG_D"],
    )

    refine_bins_parquet(
        contigs_fasta=contigs_fasta,
        contacts_parquet=contacts_parquet,
        bins_tsv=bins_tsv,
        coverage_tsv=None,
        out_dir=out_dir,
    )

    bins_lines = (out_dir / "bins.refined.tsv").read_text(encoding="utf-8").splitlines()[1:]
    assignment = dict(line.split("\t") for line in bins_lines if line.strip())
    assert set(assignment.keys()) == {"A", "B", "C", "D"}
    assert {assignment["A"], assignment["B"], assignment["C"], assignment["D"]} == {"0"}

    action_lines = (out_dir / "refine_actions.tsv").read_text(encoding="utf-8").splitlines()
    split_rows = [line.split("\t") for line in action_lines[1:] if line.split("\t")[1] == "split"]
    assert len(split_rows) == 1
    assert split_rows[0][5] == "0"
    assert split_rows[0][7] == "scg_gate_veto"
    assert split_rows[0][11] == "veto"


def test_refine_live_split_requires_scg_resources(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pytest

    from porebin import refine as refine_mod
    from porebin.refine import RefineError, refine_bins_parquet

    contigs_fasta = tmp_path / "contigs.fasta"
    contacts_parquet = tmp_path / "contacts.parquet"
    bins_tsv = tmp_path / "bins.tsv"
    out_dir = tmp_path / "refined"

    contigs = [(name, "ACGT" * 800) for name in ["A", "B", "C", "D"]]
    _write_fasta(contigs_fasta, contigs)
    bins_tsv.write_text(
        "contig_name\tbin_id\nA\t0\nB\t0\nC\t0\nD\t0\n",
        encoding="utf-8",
    )
    pq.write_table(
        pa.table(
            {
                "contigs": pa.array([["A", "B"], ["C", "D"]] * 20, type=pa.list_(pa.string())),
                "contig_weights": pa.array([[0.5, 0.5]] * 40, type=pa.list_(pa.float64())),
                "weight": pa.array([1.0] * 40, type=pa.float64()),
            }
        ),
        contacts_parquet,
    )

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(refine_mod.shutil, "which", lambda _name: None)
    try:
        with pytest.raises(RefineError, match="refine requires SCG tooling before it can run"):
            refine_bins_parquet(
                contigs_fasta=contigs_fasta,
                contacts_parquet=contacts_parquet,
                bins_tsv=bins_tsv,
                coverage_tsv=None,
                out_dir=out_dir,
            )
    finally:
        monkeypatch.undo()


def test_refine_soft_gating_keeps_weak_bin_as_candidate_host(tmp_path: Path, monkeypatch) -> None:
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

    _patch_enabled_scg(
        monkeypatch,
        tmp_path,
        rows=[],
        expected_markers=["SCG_A", "SCG_B", "SCG_C", "SCG_D"],
    )

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
    assert by_contig["X"]["refine_state"] == "relation_only"

    run_refine = json.loads((out_dir / "run_refine.json").read_text(encoding="utf-8"))
    assert run_refine["stats"]["candidate_bins_count"] == 2
    decisions = run_refine["decisions"]
    assert decisions["soft_gating_enabled"] is True
    assert decisions["candidate_bin_rule"] == "non_impure_bins"
    assert decisions["bin_weight_scheme"]["weak"] == 0.35
    assert decisions["read_evidence_rule"] == "only_hard_anchors_define_host_direction"
