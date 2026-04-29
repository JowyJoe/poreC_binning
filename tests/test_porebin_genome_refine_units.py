from __future__ import annotations

from pathlib import Path

import pytest

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
)
from porebin_genome.refine.qc import build_bin_qc_rows, write_bin_qc_tsv
from porebin_genome.refine.reassign import evaluate_reassign_candidate
from porebin_genome.refine.recruit import evaluate_recruit_candidate
from porebin_genome.refine.split import evaluate_split_candidate, generate_split_candidates


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
                delta_contact=0.2,
                scg_status="abstain",
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
                contact_coherence=0.95,
                scg_status="scg_clean",
                scg_duplicate_marker_count=0,
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


def test_split_candidate_requires_consistency_improvement() -> None:
    import numpy as np

    from porebin_genome.refine.hyperedge import HyperedgeRecord, HyperedgeStore
    state = RefineState(
        contig_lengths={"a": 1500, "b": 1500, "c": 1500, "d": 1500},
        coverage_by_contig={"a": 30.0, "b": 31.0, "c": 10.0, "d": 9.0},
        coarse_assignment={"a": "0", "b": "0", "c": "0", "d": "0"},
        current_assignment={"a": "0", "b": "0", "c": "0", "d": "0"},
        assignment_stage={},
        assignment_reason={},
        next_bin_id=1,
    )
    edge0 = HyperedgeRecord(edge_id=0, members=("a", "b"), alpha=(0.5, 0.5), read_weight=1.0, k=2, k_eff=2.0)
    edge1 = HyperedgeRecord(edge_id=1, members=("c", "d"), alpha=(0.5, 0.5), read_weight=1.0, k=2, k_eff=2.0)
    store = HyperedgeStore(
        edges=(edge0, edge1),
        edges_by_id={0: edge0, 1: edge1},
        edge_ids_by_contig={"a": (0,), "b": (0,), "c": (1,), "d": (1,)},
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
    from porebin_genome.refine.models import SplitCandidate

    accepted = evaluate_split_candidate(
        candidate=SplitCandidate(
            source_bin="0",
            groups=(("a", "b"), ("c", "d")),
            method="local_graph_components",
            reasons=("multiple_contact_components",),
        ),
        state=state,
        store=store,
        feature_matrix=feature_matrix,
        contig_name_to_idx={"a": 0, "b": 1, "c": 2, "d": 3},
        profiles={
            "0": BinFeatureProfile(
                bin_id="0",
                members=("a", "b", "c", "d"),
                centroid=np.asarray([1.55, 1.5], dtype=float),
                distance_median=2.0862646045025066,
                distance_mad=0.03535533905932753,
                feature_dispersion=2.0862646045025066,
                median_coverage=20.0,
                coverage_mad=10.5,
            )
        },
        new_bin_id="1",
    )
    assert accepted.accepted is True


def test_generate_split_candidates_prefers_leiden_when_available() -> None:
    pytest.importorskip("leidenalg", reason="leidenalg required for Leiden split test")
    import numpy as np

    from porebin_genome.refine.hyperedge import HyperedgeRecord, HyperedgeStore
    from porebin_genome.refine.models import SuspectBin

    state = RefineState(
        contig_lengths={"a": 1500, "b": 1500, "c": 1500, "d": 1500},
        coverage_by_contig={"a": 30.0, "b": 31.0, "c": 10.0, "d": 9.0},
        coarse_assignment={"a": "0", "b": "0", "c": "0", "d": "0"},
        current_assignment={"a": "0", "b": "0", "c": "0", "d": "0"},
        assignment_stage={},
        assignment_reason={},
        next_bin_id=1,
    )
    edge0 = HyperedgeRecord(edge_id=0, members=("a", "b"), alpha=(0.5, 0.5), read_weight=1.0, k=2, k_eff=2.0)
    edge1 = HyperedgeRecord(edge_id=1, members=("c", "d"), alpha=(0.5, 0.5), read_weight=1.0, k=2, k_eff=2.0)
    store = HyperedgeStore(
        edges=(edge0, edge1),
        edges_by_id={0: edge0, 1: edge1},
        edge_ids_by_contig={"a": (0,), "b": (0,), "c": (1,), "d": (1,)},
    )
    candidates = generate_split_candidates(
        state=state,
        suspects=[SuspectBin(bin_id="0", reasons=("multiple_contact_components",))],
        store=store,
        feature_matrix=np.asarray(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [3.0, 3.0],
                [3.1, 3.0],
            ],
            dtype=float,
        ),
        contig_name_to_idx={"a": 0, "b": 1, "c": 2, "d": 3},
    )

    assert candidates
    assert candidates[0].method.startswith("leiden_two_way")


def test_reassign_and_recruit_candidate_rules() -> None:
    import numpy as np

    from porebin_genome.refine.hyperedge import HyperedgeRecord, HyperedgeStore

    state = RefineState(
        contig_lengths={"a": 1500, "c": 1500, "d": 1500, "e": 1500},
        coverage_by_contig={"a": 35.0, "c": 11.0, "d": 10.0, "e": 9.5},
        coarse_assignment={"a": "0", "c": "1", "d": "1", "e": "0"},
        current_assignment={"a": "0", "c": "1", "d": "1", "e": "0"},
        assignment_stage={},
        assignment_reason={},
        next_bin_id=2,
    )
    edge0 = HyperedgeRecord(edge_id=0, members=("e", "c"), alpha=(0.5, 0.5), read_weight=1.0, k=2, k_eff=2.0)
    edge1 = HyperedgeRecord(edge_id=1, members=("e", "d"), alpha=(0.5, 0.5), read_weight=1.0, k=2, k_eff=2.0)
    edge2 = HyperedgeRecord(edge_id=2, members=("e", "a"), alpha=(0.5, 0.5), read_weight=0.2, k=2, k_eff=2.0)
    store = HyperedgeStore(
        edges=(edge0, edge1, edge2),
        edges_by_id={0: edge0, 1: edge1, 2: edge2},
        edge_ids_by_contig={"e": (0, 1, 2), "c": (0,), "d": (1,), "a": (2,)},
    )
    profiles = {
        "1": BinFeatureProfile(
            bin_id="1",
            members=("c", "d"),
            centroid=np.asarray([3.05, 3.0], dtype=float),
            distance_median=0.05,
            distance_mad=0.0,
            feature_dispersion=0.05,
            median_coverage=10.5,
            coverage_mad=0.5,
        )
    }
    feature_matrix = np.asarray(
        [
            [0.0, 0.0],  # a
            [3.0, 3.0],  # c
            [3.1, 3.0],  # d
            [3.05, 3.0],  # e
        ],
        dtype=float,
    )

    reassign_accept = evaluate_reassign_candidate(
        candidate=ReassignCandidate(
            contig_id="e",
            source_bin="0",
            target_bin="1",
            current_bin_support=0.05,
            target_bin_support=0.5,
            runner_up_support=0.05,
            target_support_edges=2,
            feature_gate=True,
            feature_note="feature_agreement_pass",
        ),
        state=state,
        store=store,
        profiles=profiles,
        feature_matrix=feature_matrix,
        contig_name_to_idx={"a": 0, "c": 1, "d": 2, "e": 3},
    )
    assert reassign_accept.accepted is True
    assert reassign_accept.reason == "move_to_target_bin"

    recruit_state = RefineState(
        contig_lengths={"a": 1500, "b": 1500, "x": 1500},
        coverage_by_contig={"a": 20.0, "b": 21.0, "x": 20.5},
        coarse_assignment={"a": "0", "b": "0"},
        current_assignment={"a": "0", "b": "0"},
        assignment_stage={},
        assignment_reason={},
        next_bin_id=1,
    )
    recruit_edge = HyperedgeRecord(edge_id=0, members=("x", "a"), alpha=(0.5, 0.5), read_weight=1.0, k=2, k_eff=2.0)
    recruit_store = HyperedgeStore(
        edges=(recruit_edge,),
        edges_by_id={0: recruit_edge},
        edge_ids_by_contig={"x": (0,), "a": (0,)},
    )
    recruit_reject = evaluate_recruit_candidate(
        candidate=RecruitCandidate(
            contig_id="x",
            target_bin="0",
            target_bin_support=0.25,
            runner_up_support=0.0,
            target_support_edges=1,
            feature_gate=True,
            feature_note="feature_agreement_pass",
        ),
        state=recruit_state,
        store=recruit_store,
    )
    assert recruit_reject.accepted is False
    assert recruit_reject.reason == "recruit_requires_multiple_independent_hyperedges"
