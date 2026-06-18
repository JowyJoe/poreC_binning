from __future__ import annotations

import numpy as np

from porebin_genome.evidence.canonical import CanonicalContact
from porebin_genome.refinement import (
    ActionEvaluator,
    ActionPolicy,
    DecisionStatus,
    SplitCandidateStatus,
    generate_split_candidate,
    generate_split_candidates,
)
from porebin_genome.refinement.contact_index import (
    build_contact_index_from_contacts,
)
from porebin_genome.refinement.profiles import (
    EvidenceInputs,
    EvidenceProfileState,
)
from porebin_genome.refinement.state import RefineState


def _split_fixture(
    *,
    missing_embedding: bool = False,
    identical_seeds: bool = False,
    duplicated_scg: bool = True,
):
    index = build_contact_index_from_contacts(
        [
            CanonicalContact(0, ["a", "b"], [0.5, 0.5], 2, 2, 1.0),
            CanonicalContact(1, ["a", "c"], [0.5, 0.5], 2, 2, 1.0),
            CanonicalContact(2, ["b", "c"], [0.5, 0.5], 2, 2, 1.0),
            CanonicalContact(3, ["d", "e"], [0.5, 0.5], 2, 2, 1.0),
            CanonicalContact(4, ["d", "f"], [0.5, 0.5], 2, 2, 1.0),
            CanonicalContact(5, ["e", "f"], [0.5, 0.5], 2, 2, 1.0),
            CanonicalContact(6, ["c", "d"], [0.5, 0.5], 2, 2, 0.1),
        ],
        contig_names=("a", "b", "c", "d", "e", "f"),
    )
    state = RefineState(
        contact_index=index,
        assignment=np.zeros(6, dtype=np.int32),
        contig_lengths=np.asarray(
            [3000, 2000, 1000, 1000, 1000, 1000],
            dtype=np.int64,
        ),
        n_bins=1,
    )
    embedding = np.asarray(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.98, 0.02],
            [1.0, 0.0] if identical_seeds else [0.0, 1.0],
            [0.01, 0.99],
            [0.02, 0.98],
        ],
        dtype=np.float64,
    )
    if missing_embedding:
        embedding[5] = np.nan
    markers = {"a": ("GrpE",)}
    if duplicated_scg:
        markers["d"] = ("GrpE",)
    inputs = EvidenceInputs.from_arrays(
        contact_index=index,
        embedding=embedding,
        coverage=np.ones(6, dtype=np.float64),
        scg_markers_by_contig=markers,
    )
    profiles = EvidenceProfileState(refine_state=state, inputs=inputs)
    return index, state, profiles


def test_primary_duplicate_marker_uses_count_then_canonical_id() -> None:
    index, state, _profiles = _split_fixture()
    inputs = EvidenceInputs.from_arrays(
        contact_index=index,
        embedding=np.eye(6, dtype=np.float64),
        scg_markers_by_contig={
            "a": ("GrpE",),
            "d": ("GrpE",),
            "b": ("PGK",),
            "e": ("PGK",),
        },
    )
    profiles = EvidenceProfileState(refine_state=state, inputs=inputs)

    duplicate = profiles.primary_duplicate_marker(0)

    assert duplicate is not None
    assert duplicate.marker_id == "GrpE"
    assert duplicate.contig_indices == (0, 3)
    assert duplicate.copy_count == 2


def test_scg_seeded_hgvae_split_is_deterministic_and_auditable() -> None:
    index, state, profiles = _split_fixture()
    assignment_before = state.assignment.copy()
    index.counters.reset()
    profiles.counters.reset()

    first = generate_split_candidate(
        source_bin_idx=0,
        state=state,
        profiles=profiles,
    )
    second = generate_split_candidate(
        source_bin_idx=0,
        state=state,
        profiles=profiles,
    )

    assert first == second
    assert first.status is SplitCandidateStatus.READY
    assert first.reason == "scg_seeded_hgvae_partition"
    assert first.marker_id == "GrpE"
    assert first.seed_contig_indices == (0, 3)
    assert first.child_members == ((0, 1, 2), (3, 4, 5))
    assert first.n_children == 2
    assert first.proposal is not None
    assert tuple(
        (
            change.contig_idx,
            change.old_bin_idx,
            change.new_bin_idx,
        )
        for change in first.proposal.changes
    ) == (
        (3, 0, 1),
        (4, 0, 1),
        (5, 0, 1),
    )
    assert first.inertia is not None
    assert first.iterations is not None
    assert np.array_equal(state.assignment, assignment_before)
    assert index.counters.full_edge_scans == 0
    assert index.counters.local_queries == 0
    assert profiles.counters.bin_profiles_built == 0


def test_generated_split_passes_existing_local_evaluator_and_policy() -> None:
    index, state, profiles = _split_fixture()
    candidate = generate_split_candidate(
        source_bin_idx=0,
        state=state,
        profiles=profiles,
    )
    assert candidate.proposal is not None
    index.counters.reset()
    profiles.counters.reset()

    evidence = ActionEvaluator(
        state=state,
        profiles=profiles,
    ).evaluate(candidate.proposal)
    decision = ActionPolicy().decide(evidence)

    assert evidence.scg is not None
    assert evidence.scg.duplicate_change == -1
    assert evidence.contact is not None
    assert evidence.contact.split_separation is not None
    assert evidence.embedding is not None
    assert evidence.embedding.relative_gain is not None
    assert decision.status is DecisionStatus.ACCEPT
    assert index.counters.full_edge_scans == 0
    assert index.counters.local_queries == 3
    assert index.counters.local_edges_visited == 7
    assert profiles.counters.bin_profiles_built == 3
    assert profiles.counters.contigs_visited == 6


def test_split_stage_generates_only_scg_suspect_bins() -> None:
    _index, state, profiles = _split_fixture()

    candidates = generate_split_candidates(
        state=state,
        profiles=profiles,
    )

    assert len(candidates) == 1
    assert candidates[0].source_bin_idx == 0
    assert candidates[0].status is SplitCandidateStatus.READY


def test_three_scg_copies_generate_a_stable_three_way_split() -> None:
    index = build_contact_index_from_contacts(
        [
            CanonicalContact(0, ["a", "b"], [0.5, 0.5], 2, 2, 1.0),
            CanonicalContact(1, ["c", "d"], [0.5, 0.5], 2, 2, 1.0),
            CanonicalContact(2, ["e", "f"], [0.5, 0.5], 2, 2, 1.0),
        ],
        contig_names=("a", "b", "c", "d", "e", "f"),
    )
    state = RefineState(
        contact_index=index,
        assignment=np.zeros(6, dtype=np.int32),
        contig_lengths=np.asarray(
            [3000, 2000, 1000, 1000, 1500, 1000],
            dtype=np.int64,
        ),
        n_bins=1,
    )
    inputs = EvidenceInputs.from_arrays(
        contact_index=index,
        embedding=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.99, 0.01, 0.0],
                [0.0, 1.0, 0.0],
                [0.01, 0.99, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.01, 0.99],
            ]
        ),
        scg_markers_by_contig={
            "a": ("GrpE",),
            "c": ("GrpE",),
            "e": ("GrpE",),
        },
    )
    profiles = EvidenceProfileState(refine_state=state, inputs=inputs)

    candidate = generate_split_candidate(
        source_bin_idx=0,
        state=state,
        profiles=profiles,
    )

    assert candidate.status is SplitCandidateStatus.READY
    assert candidate.seed_contig_indices == (0, 2, 4)
    assert candidate.child_members == ((0, 1), (2, 3), (4, 5))
    assert candidate.proposal is not None
    assert tuple(
        (change.contig_idx, change.new_bin_idx)
        for change in candidate.proposal.changes
    ) == ((2, 1), (3, 1), (4, 2), (5, 2))


def test_non_suspect_bin_is_reported_without_clustering() -> None:
    index, state, profiles = _split_fixture(duplicated_scg=False)
    index.counters.reset()

    candidate = generate_split_candidate(
        source_bin_idx=0,
        state=state,
        profiles=profiles,
    )

    assert candidate.status is SplitCandidateStatus.NOT_SUSPECT
    assert candidate.reason == "no_duplicated_scg"
    assert candidate.proposal is None
    assert index.counters.full_edge_scans == 0
    assert index.counters.local_queries == 0


def test_missing_embedding_causes_explicit_split_abstention() -> None:
    index, state, profiles = _split_fixture(missing_embedding=True)
    index.counters.reset()

    candidate = generate_split_candidate(
        source_bin_idx=0,
        state=state,
        profiles=profiles,
    )

    assert candidate.status is SplitCandidateStatus.ABSTAIN
    assert candidate.reason == "bin_contains_contig_without_hgvae_embedding"
    assert candidate.marker_id == "GrpE"
    assert candidate.proposal is None
    assert index.counters.full_edge_scans == 0
    assert index.counters.local_queries == 0


def test_identical_marker_seed_embeddings_cause_abstention() -> None:
    _index, state, profiles = _split_fixture(identical_seeds=True)

    candidate = generate_split_candidate(
        source_bin_idx=0,
        state=state,
        profiles=profiles,
    )

    assert candidate.status is SplitCandidateStatus.ABSTAIN
    assert candidate.reason == (
        "duplicated_scg_seeds_have_identical_hgvae_embeddings"
    )
    assert candidate.proposal is None
