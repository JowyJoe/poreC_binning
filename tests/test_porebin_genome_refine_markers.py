from __future__ import annotations

from pathlib import Path

import pytest

from porebin_genome.refine.markers import (
    ContigScgProfile,
    ScgHit,
    collapse_scg_hits_to_contigs,
    ensure_scg_toolchain_available,
    evaluate_merge_scg_transition,
    evaluate_reassign_scg_transition,
    evaluate_recruit_scg_transition,
    evaluate_split_scg_transition,
)
from porebin_genome.refine.models import RefineState


def test_scg_toolchain_dependency_error_is_explicit() -> None:
    with pytest.raises(RuntimeError, match="requires external dependencies"):
        ensure_scg_toolchain_available(
            prodigal_executable="__missing_prodigal__",
            hmmsearch_executable="__missing_hmmsearch__",
        )


def test_collapse_scg_hits_deduplicates_same_marker_per_contig() -> None:
    profiles = collapse_scg_hits_to_contigs(
        [
            ScgHit(
                orf_id="orf1",
                contig_id="c1",
                marker_id="Ribosomal_L23",
                domain="bacteria",
                bitscore=100.0,
                evalue=1e-20,
                hmm_coverage=0.90,
            ),
            ScgHit(
                orf_id="orf2",
                contig_id="c1",
                marker_id="Ribosomal_L23",
                domain="bacteria",
                bitscore=90.0,
                evalue=1e-10,
                hmm_coverage=0.85,
            ),
            ScgHit(
                orf_id="orf3",
                contig_id="c1",
                marker_id="Ribosomal_S9",
                domain="bacteria",
                bitscore=95.0,
                evalue=1e-18,
                hmm_coverage=0.88,
            ),
        ]
    )
    assert set(profiles["c1"].marker_ids) == {"Ribosomal_L23", "Ribosomal_S9"}
    assert profiles["c1"].n_markers == 2


def test_merge_scg_transition_vetoes_new_duplicate_marker() -> None:
    state = RefineState(
        contig_lengths={"a": 1000, "b": 1000},
        coverage_by_contig={},
        coarse_assignment={"a": "0", "b": "1"},
        current_assignment={"a": "0", "b": "1"},
        assignment_stage={},
        assignment_reason={},
    )
    profiles = {
        "a": _profile("a", "Ribosomal_L23"),
        "b": _profile("b", "Ribosomal_L23"),
    }
    decision = evaluate_merge_scg_transition(
        source_bin="0",
        target_bin="1",
        state=state,
        contig_profiles=profiles,
    )
    assert decision.status == "veto"
    assert decision.reason == "merge_scg_duplicate_conflict"


def test_reassign_scg_transition_supports_duplicate_resolution() -> None:
    state = RefineState(
        contig_lengths={"x": 1000, "a": 1000, "b": 1000},
        coverage_by_contig={},
        coarse_assignment={"x": "0", "a": "0", "b": "1"},
        current_assignment={"x": "0", "a": "0", "b": "1"},
        assignment_stage={},
        assignment_reason={},
    )
    profiles = {
        "x": _profile("x", "Ribosomal_L23"),
        "a": _profile("a", "Ribosomal_L23"),
        "b": _profile("b", "Ribosomal_S9"),
    }
    decision = evaluate_reassign_scg_transition(
        contig_id="x",
        source_bin="0",
        target_bin="1",
        state=state,
        contig_profiles=profiles,
    )
    assert decision.status == "support"
    assert decision.reason == "reassign_resolves_source_scg_duplication"


def test_recruit_scg_transition_vetoes_new_duplicate_marker() -> None:
    state = RefineState(
        contig_lengths={"a": 1000, "x": 1000},
        coverage_by_contig={},
        coarse_assignment={"a": "0"},
        current_assignment={"a": "0"},
        assignment_stage={},
        assignment_reason={},
    )
    profiles = {
        "a": _profile("a", "Ribosomal_L23"),
        "x": _profile("x", "Ribosomal_L23"),
    }
    decision = evaluate_recruit_scg_transition(
        contig_id="x",
        target_bin="0",
        state=state,
        contig_profiles=profiles,
    )
    assert decision.status == "veto"
    assert decision.reason == "recruit_scg_duplicate_conflict"


def test_split_scg_transition_supports_duplicate_resolution() -> None:
    state = RefineState(
        contig_lengths={"a": 1000, "b": 1000, "c": 1000, "d": 1000},
        coverage_by_contig={},
        coarse_assignment={"a": "0", "b": "0", "c": "0", "d": "0"},
        current_assignment={"a": "0", "b": "0", "c": "0", "d": "0"},
        assignment_stage={},
        assignment_reason={},
        next_bin_id=1,
    )
    profiles = {
        "a": _profile("a", "Ribosomal_L23"),
        "c": _profile("c", "Ribosomal_L23"),
    }
    decision = evaluate_split_scg_transition(
        source_bin="0",
        child_groups=(("a", "b"), ("c", "d")),
        state=state,
        contig_profiles=profiles,
        new_bin_id="1",
    )
    assert decision.status == "support"
    assert decision.reason == "split_resolves_scg_duplication"


def _profile(contig_id: str, *marker_ids: str) -> ContigScgProfile:
    return ContigScgProfile(
        contig_id=contig_id,
        marker_ids=tuple(sorted(marker_ids)),
        marker_to_domain={marker_id: "bacteria" for marker_id in marker_ids},
        n_markers=len(marker_ids),
        domain_counts={"bacteria": len(marker_ids)} if marker_ids else {},
    )
