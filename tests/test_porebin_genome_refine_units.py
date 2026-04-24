from __future__ import annotations

from pathlib import Path

from porebin_genome.io.contracts import BIN_QC_COLUMNS, REFINE_ACTIONS_COLUMNS, UNBINNED_COLUMNS
from porebin_genome.io.tables import validate_tsv_header, write_tsv_rows
from porebin_genome.refine.actions import write_refine_actions_tsv
from porebin_genome.refine.models import (
    BinFeatureProfile,
    BinSnapshot,
    ReassignCandidate,
    RefineActionRow,
    RefineState,
    RecruitCandidate,
    SupportSummary,
)
from porebin_genome.refine.qc import build_bin_qc_rows, write_bin_qc_tsv
from porebin_genome.refine.reassign import evaluate_reassign_candidate
from porebin_genome.refine.recruit import evaluate_recruit_candidate
from porebin_genome.refine.split import evaluate_split_candidate
from porebin_genome.refine.support import compute_assignment_confidence


def test_refine_action_and_table_contracts(tmp_path: Path) -> None:
    actions_tsv = tmp_path / "refine_actions.tsv"
    write_refine_actions_tsv(
        rows=[
            RefineActionRow(
                action_type="reassign",
                contig_id="c1",
                bin_id="",
                source_bin="0",
                target_bin="1",
                reason="move_to_target_bin",
                accepted=True,
                confidence=0.9,
                note="accepted",
            )
        ],
        out_path=actions_tsv,
    )
    validate_tsv_header(actions_tsv, REFINE_ACTIONS_COLUMNS)

    unbinned_tsv = tmp_path / "unbinned.tsv"
    write_tsv_rows(
        unbinned_tsv,
        UNBINNED_COLUMNS,
        [("c2", "recruit", "target_bin_margin_too_small", "", "kept_unbinned")],
    )
    validate_tsv_header(unbinned_tsv, UNBINNED_COLUMNS)

    qc_rows = build_bin_qc_rows(
        snapshots={
            "0": BinSnapshot(
                bin_id="0",
                members=("c1", "c2"),
                n_contigs=2,
                total_length=4000,
                median_coverage=20.0,
                coverage_dispersion=1.0,
                feature_dispersion=0.2,
                contact_consistency=0.95,
                contact_components=1,
                low_support_ratio=0.0,
                suspect_flag=False,
                suspect_reasons=(),
                refine_status="stable",
            )
        }
    )
    qc_tsv = tmp_path / "bin_qc.tsv"
    write_bin_qc_tsv(rows=qc_rows, out_path=qc_tsv)
    validate_tsv_header(qc_tsv, BIN_QC_COLUMNS)


def test_assignment_confidence_rule_uses_support_margin_and_feature_gate() -> None:
    high = compute_assignment_confidence(target_bin_support=10.0, runner_up_support=1.0, feature_gate=True)
    low = compute_assignment_confidence(target_bin_support=10.0, runner_up_support=9.0, feature_gate=False)
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high > low


def test_split_candidate_requires_consistency_improvement() -> None:
    import numpy as np

    state = RefineState(
        contig_lengths={"a": 1500, "b": 1500, "c": 1500, "d": 1500},
        coverage_by_contig={"a": 30.0, "b": 31.0, "c": 10.0, "d": 9.0},
        coarse_assignment={"a": "0", "b": "0", "c": "0", "d": "0"},
        current_assignment={"a": "0", "b": "0", "c": "0", "d": "0"},
        assignment_stage={},
        assignment_reason={},
        next_bin_id=1,
    )
    support_summary = SupportSummary(
        support_by_contig_bin={},
        own_support={},
        total_support={},
        runner_up_bin={},
        runner_up_support={},
        pair_support_by_bin={"0": {("a", "b"): 2.0, ("c", "d"): 2.0}},
        contact_components_by_bin={"0": (("a", "b"), ("c", "d"))},
    )
    feature_matrix = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [3.0, 3.0],
            [3.1, 3.0],
        ],
        dtype=float,
    )
    snapshot = BinSnapshot(
        bin_id="0",
        members=("a", "b", "c", "d"),
        n_contigs=4,
        total_length=6000,
        median_coverage=20.0,
        coverage_dispersion=8.0,
        feature_dispersion=2.0,
        contact_consistency=0.45,
        contact_components=2,
        low_support_ratio=0.5,
        suspect_flag=True,
        suspect_reasons=("multiple_contact_components",),
        refine_status="suspect",
    )
    from porebin_genome.refine.models import SplitCandidate

    accepted = evaluate_split_candidate(
        candidate=SplitCandidate(
            source_bin="0",
            groups=(("a", "b"), ("c", "d")),
            method="contact_components",
            reasons=("multiple_contact_components",),
        ),
        state=state,
        snapshot=snapshot,
        support_summary=support_summary,
        feature_matrix=feature_matrix,
        contig_name_to_idx={"a": 0, "b": 1, "c": 2, "d": 3},
        new_bin_id="1",
    )
    assert accepted.accepted is True


def test_reassign_and_recruit_candidate_rules() -> None:
    reassign_accept = evaluate_reassign_candidate(
        ReassignCandidate(
            contig_id="e",
            source_bin="0",
            target_bin="1",
            current_bin_support=1.0,
            target_bin_support=8.0,
            runner_up_support=1.0,
            runner_up_margin=0.7,
            feature_gate=True,
            non_worsening_gate=True,
            feature_note="feature_agreement_pass",
        )
    )
    assert reassign_accept.accepted is True
    assert reassign_accept.reason == "move_to_target_bin"

    recruit_reject = evaluate_recruit_candidate(
        RecruitCandidate(
            contig_id="x",
            target_bin="1",
            target_bin_support=2.0,
            runner_up_support=1.9,
            runner_up_margin=0.02,
            feature_gate=True,
            feature_note="feature_agreement_pass",
        )
    )
    assert recruit_reject.accepted is False
    assert recruit_reject.reason == "target_bin_margin_too_small"
