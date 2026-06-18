from __future__ import annotations

import numpy as np
import pytest

from porebin_genome.evidence.canonical import CanonicalContact
from porebin_genome.refinement import (
    ActionEvaluator,
    ActionPolicy,
    DecisionStatus,
    apply_merge_batch,
    generate_merge_candidates,
    prepare_merge_batch,
    rollback_merge_batch,
)
from porebin_genome.refinement.contact_index import (
    build_contact_index_from_contacts,
)
from porebin_genome.refinement.profiles import (
    EvidenceInputs,
    EvidenceProfileState,
)
from porebin_genome.refinement.state import RefineState


def _merge_fixture(
    *,
    far_first_pair: bool = False,
    duplicate_scg_bin: int | None = None,
):
    index = build_contact_index_from_contacts(
        [
            CanonicalContact(0, ["a", "c"], [0.5, 0.5], 2, 2, 1.0),
            CanonicalContact(1, ["b", "d"], [0.5, 0.5], 2, 2, 1.0),
            CanonicalContact(2, ["e", "g"], [0.5, 0.5], 2, 2, 0.8),
            CanonicalContact(3, ["f", "h"], [0.5, 0.5], 2, 2, 0.8),
            CanonicalContact(4, ["a", "e"], [0.5, 0.5], 2, 2, 0.1),
            CanonicalContact(5, ["b", "f"], [0.5, 0.5], 2, 2, 0.1),
        ],
        contig_names=("a", "b", "c", "d", "e", "f", "g", "h"),
    )
    state = RefineState(
        contact_index=index,
        assignment=np.asarray(
            [0, 0, 1, 1, 2, 2, 3, 3],
            dtype=np.int32,
        ),
        contig_lengths=np.asarray(
            [3000, 2000, 1000, 1000, 2500, 1500, 1200, 1200],
            dtype=np.int64,
        ),
        n_bins=4,
    )
    first_target = (
        [[0.0, 1.0], [0.0, 1.0]]
        if far_first_pair
        else [[1.0, 0.0], [0.98, 0.02]]
    )
    embedding = np.asarray(
        [
            [1.0, 0.0],
            [0.98, 0.02],
            *first_target,
            [0.0, 1.0],
            [0.02, 0.98],
            [0.0, 1.0],
            [0.02, 0.98],
        ],
        dtype=np.float64,
    )
    markers: dict[str, tuple[str, ...]] = {}
    if duplicate_scg_bin is not None:
        bin_members = {
            0: ("a", "b"),
            1: ("c", "d"),
            2: ("e", "f"),
            3: ("g", "h"),
        }[duplicate_scg_bin]
        markers = {
            bin_members[0]: ("GrpE",),
            bin_members[1]: ("GrpE",),
        }
    inputs = EvidenceInputs.from_arrays(
        contact_index=index,
        embedding=embedding,
        coverage=np.ones(8, dtype=np.float64),
        scg_markers_by_contig=markers,
    )
    profiles = EvidenceProfileState(refine_state=state, inputs=inputs)
    return index, state, profiles


def test_merge_generation_uses_one_scan_and_mutual_best_pairs() -> None:
    index, state, profiles = _merge_fixture()
    assignment_before = state.assignment.copy()
    index.counters.reset()
    profiles.counters.reset()

    batch = generate_merge_candidates(
        state=state,
        profiles=profiles,
    )

    assert batch.state_version == 0
    assert batch.eligible_bin_indices == (0, 1, 2, 3)
    assert batch.supported_pair_count == 3
    assert tuple(candidate.pair for candidate in batch.candidates) == (
        (0, 1),
        (2, 3),
    )
    assert {
        candidate.source_bin_idx
        for candidate in batch.candidates
    } == {1, 3}
    assert {
        candidate.target_bin_idx
        for candidate in batch.candidates
    } == {0, 2}
    assert batch.candidates[0].raw_support == pytest.approx(0.5)
    assert batch.candidates[0].supporting_edge_count == 2
    assert batch.candidates[0].normalized_support > 0.0
    assert tuple(
        (change.contig_idx, change.old_bin_idx, change.new_bin_idx)
        for change in batch.candidates[0].proposal.changes
    ) == ((2, 1, 0), (3, 1, 0))
    assert np.array_equal(state.assignment, assignment_before)
    assert index.counters.full_edge_scans == 1
    assert index.counters.full_edges_visited == index.n_edges
    assert index.counters.local_queries == 0
    assert profiles.counters.bin_profiles_built == 0


def test_merge_generation_is_deterministic() -> None:
    index, state, profiles = _merge_fixture()

    first = generate_merge_candidates(state=state, profiles=profiles)
    index.counters.reset()
    second = generate_merge_candidates(state=state, profiles=profiles)

    assert first == second
    assert index.counters.full_edge_scans == 1
    assert index.counters.full_edges_visited == index.n_edges


def test_scg_suspect_bin_is_not_a_merge_object() -> None:
    index, state, profiles = _merge_fixture(duplicate_scg_bin=2)
    index.counters.reset()

    batch = generate_merge_candidates(state=state, profiles=profiles)

    assert batch.eligible_bin_indices == (0, 1, 3)
    assert tuple(candidate.pair for candidate in batch.candidates) == ((0, 1),)
    assert all(2 not in candidate.pair for candidate in batch.candidates)
    assert index.counters.full_edge_scans == 1


def test_merge_evaluator_reuses_local_edges_and_matches_candidate_support() -> None:
    index, state, profiles = _merge_fixture()
    candidate = generate_merge_candidates(
        state=state,
        profiles=profiles,
    ).candidates[0]
    index.counters.reset()
    profiles.counters.reset()

    evidence = ActionEvaluator(
        state=state,
        profiles=profiles,
    ).evaluate(candidate.proposal)
    decision = ActionPolicy().decide(evidence)

    assert evidence.contact is not None
    assert evidence.contact.merge_pair_support == pytest.approx(
        candidate.raw_support
    )
    assert evidence.contact.merge_normalized_support == pytest.approx(
        candidate.normalized_support
    )
    assert evidence.contact.merge_supporting_edge_count == 2
    assert evidence.embedding is None
    assert evidence.bin_pair_embedding is not None
    assert evidence.bin_pair_embedding.centroid_distance == pytest.approx(0.0)
    assert decision.status is DecisionStatus.ACCEPT
    assert decision.gates[-1].reason == "merge_hgvae_regions_overlap"
    assert index.counters.full_edge_scans == 0
    assert index.counters.local_queries == 2
    assert index.counters.local_edges_visited == 2
    assert profiles.counters.bin_profiles_built == 2
    assert profiles.counters.contigs_visited == 4


def test_strong_contact_does_not_override_incompatible_hgvae_regions() -> None:
    _index, state, profiles = _merge_fixture(far_first_pair=True)
    candidate = generate_merge_candidates(
        state=state,
        profiles=profiles,
    ).candidates[0]

    evidence = ActionEvaluator(
        state=state,
        profiles=profiles,
    ).evaluate(candidate.proposal)
    decision = ActionPolicy().decide(evidence)

    assert evidence.contact is not None
    assert evidence.contact.merge_supporting_edge_count == 2
    assert evidence.bin_pair_embedding is not None
    assert evidence.bin_pair_embedding.centroid_distance is not None
    assert evidence.bin_pair_embedding.centroid_distance > 1.0
    assert evidence.bin_pair_embedding.combined_radius is not None
    assert (
        evidence.bin_pair_embedding.centroid_distance
        > evidence.bin_pair_embedding.combined_radius
    )
    assert decision.status is DecisionStatus.ABSTAIN
    assert decision.reason_code == "embedding_unsupported"
    assert decision.gates[-1].reason == (
        "merge_hgvae_regions_do_not_overlap"
    )


def test_one_supporting_edge_is_not_a_merge_candidate() -> None:
    index = build_contact_index_from_contacts(
        [CanonicalContact(0, ["a", "b"], [0.5, 0.5], 2, 2, 1.0)],
        contig_names=("a", "b"),
    )
    state = RefineState(
        contact_index=index,
        assignment=np.asarray([0, 1], dtype=np.int32),
        contig_lengths=np.asarray([1000, 1000], dtype=np.int64),
        n_bins=2,
    )
    inputs = EvidenceInputs.from_arrays(
        contact_index=index,
        embedding=np.asarray([[1.0, 0.0], [1.0, 0.0]]),
    )
    profiles = EvidenceProfileState(refine_state=state, inputs=inputs)

    batch = generate_merge_candidates(
        state=state,
        profiles=profiles,
    )

    assert batch.supported_pair_count == 0
    assert batch.candidates == ()


def test_disjoint_accepted_merges_apply_in_one_versioned_batch() -> None:
    _index, state, profiles = _merge_fixture()
    assignment_before = state.assignment.copy()
    candidates = generate_merge_candidates(
        state=state,
        profiles=profiles,
    ).candidates
    evaluator = ActionEvaluator(state=state, profiles=profiles)
    policy = ActionPolicy()
    decisions = tuple(
        policy.decide(evaluator.evaluate(candidate.proposal))
        for candidate in candidates
    )
    assert all(
        decision.status is DecisionStatus.ACCEPT
        for decision in decisions
    )

    update = prepare_merge_batch(
        state=state,
        profiles=profiles,
        decisions=decisions,
    )
    apply_merge_batch(state=state, profiles=profiles, update=update)

    assert state.version == profiles.version == 1
    assert state.bin_members(0) == frozenset({0, 1, 2, 3})
    assert state.bin_members(1) == frozenset()
    assert state.bin_members(2) == frozenset({4, 5, 6, 7})
    assert state.bin_members(3) == frozenset()

    rollback_merge_batch(state=state, profiles=profiles, update=update)
    assert state.version == profiles.version == 0
    assert np.array_equal(state.assignment, assignment_before)
