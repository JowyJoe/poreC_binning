from __future__ import annotations

import numpy as np

from porebin_genome.evidence.canonical import CanonicalContact
from porebin_genome.refinement import (
    ActionEvaluator,
    ActionPolicy,
    DecisionStatus,
    RecruitCandidateStatus,
    RecruitGeneratorConfig,
    generate_recruit_candidate,
    generate_recruit_candidates,
    run_recruitment_pass,
)
from porebin_genome.refinement.contact_index import (
    build_contact_index_from_contacts,
)
from porebin_genome.refinement.profiles import (
    EvidenceInputs,
    EvidenceProfileState,
)
from porebin_genome.refinement.state import RefineState


def _base_fixture(
    *,
    unbinned_embedding: np.ndarray | None = None,
    markers: dict[str, tuple[str, ...]] | None = None,
):
    names = ("a", "b", "c", "d", "u", "v", "w", "x", "y")
    index = build_contact_index_from_contacts(
        [
            CanonicalContact(0, ["u", "a"], [0.5, 0.5], 2, 2, 1.0),
            CanonicalContact(1, ["u", "b"], [0.5, 0.5], 2, 2, 1.0),
            CanonicalContact(2, ["v", "a"], [0.5, 0.5], 2, 2, 1.0),
            CanonicalContact(3, ["v", "b"], [0.5, 0.5], 2, 2, 1.0),
            CanonicalContact(4, ["w", "a"], [0.5, 0.5], 2, 2, 1.0),
            CanonicalContact(5, ["w", "b"], [0.5, 0.5], 2, 2, 1.0),
            CanonicalContact(6, ["w", "c"], [0.5, 0.5], 2, 2, 1.0),
            CanonicalContact(7, ["w", "d"], [0.5, 0.5], 2, 2, 1.0),
            CanonicalContact(8, ["x", "a"], [0.5, 0.5], 2, 2, 1.0),
            CanonicalContact(9, ["x", "b"], [0.5, 0.5], 2, 2, 1.0),
        ],
        contig_names=names,
    )
    state = RefineState(
        contact_index=index,
        assignment=np.asarray(
            [0, 0, 1, 1, -1, -1, -1, -1, -1],
            dtype=np.int32,
        ),
        contig_lengths=np.asarray(
            [2000, 2000, 2000, 2000, 1000, 1000, 1000, 1000, 1000],
            dtype=np.int64,
        ),
        n_bins=2,
    )
    if unbinned_embedding is None:
        unbinned_embedding = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [np.nan, np.nan],
                [1.0, 0.0],
            ],
            dtype=np.float64,
        )
    embedding = np.vstack(
        [
            np.asarray(
                [
                    [1.0, 0.0],
                    [0.98, 0.02],
                    [0.0, 1.0],
                    [0.02, 0.98],
                ],
                dtype=np.float64,
            ),
            unbinned_embedding,
        ]
    )
    inputs = EvidenceInputs.from_arrays(
        contact_index=index,
        embedding=embedding,
        coverage=np.ones(len(names), dtype=np.float64),
        scg_markers_by_contig=markers or {},
    )
    profiles = EvidenceProfileState(refine_state=state, inputs=inputs)
    return index, state, profiles


def test_recruit_batch_is_local_deterministic_and_explainable() -> None:
    index, state, profiles = _base_fixture()
    assignment_before = state.assignment.copy()
    index.counters.reset()
    profiles.counters.reset()

    batch = generate_recruit_candidates(
        state=state,
        profiles=profiles,
    )

    outcomes = {outcome.contig_idx: outcome for outcome in batch.outcomes}
    ready = outcomes[4]
    assert batch.state_version == 0
    assert batch.eligible_bin_indices == (0, 1)
    assert tuple(candidate.contig_idx for candidate in batch.candidates) == (4,)
    assert ready.status is RecruitCandidateStatus.READY
    assert ready.reason == "porec_hgvae_target_agreement"
    assert ready.contact_target_bin_idx == 0
    assert ready.latent_target_bin_idx == 0
    assert ready.target_edge_count == 2
    assert ready.target_share is not None
    assert ready.target_share > 0.99
    assert ready.latent_distance is not None
    assert ready.target_radius is not None
    assert ready.proposal is not None
    assert outcomes[5].reason == "contact_latent_target_disagree"
    assert outcomes[6].reason == "contact_best_target_tied"
    assert outcomes[7].reason == "hgvae_embedding_unavailable"
    assert outcomes[8].reason == "no_assigned_bin_contact_support"
    assert np.array_equal(state.assignment, assignment_before)
    assert index.counters.full_edge_scans == 0
    assert index.counters.local_queries == 4
    assert index.counters.local_edges_visited == 8
    assert profiles.counters.bin_profiles_built == 0

    second = generate_recruit_candidates(
        state=state,
        profiles=profiles,
    )
    assert batch == second


def test_latent_tie_abstains_without_bin_id_tie_breaking() -> None:
    unbinned = np.asarray(
        [
            [1.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [np.nan, np.nan],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    _index, state, profiles = _base_fixture(
        unbinned_embedding=unbinned,
    )

    candidate = generate_recruit_candidate(
        contig_idx=4,
        state=state,
        profiles=profiles,
    )

    assert candidate.status is RecruitCandidateStatus.ABSTAIN
    assert candidate.reason == "latent_best_target_tied"
    assert candidate.contact_target_bin_idx == 0
    assert candidate.proposal is None


def test_ready_candidate_matches_shared_evidence_and_is_accepted() -> None:
    index, state, profiles = _base_fixture()
    candidate = generate_recruit_candidate(
        contig_idx=4,
        state=state,
        profiles=profiles,
    )
    assert candidate.proposal is not None
    index.counters.reset()

    evidence = ActionEvaluator(
        state=state,
        profiles=profiles,
    ).evaluate(candidate.proposal)
    decision = ActionPolicy().decide(evidence)

    assert evidence.contact is not None
    assert evidence.contact.single_contig is not None
    assert evidence.contact.single_contig.target_support == (
        candidate.target_support
    )
    assert evidence.contact.single_contig.target_share == (
        candidate.target_share
    )
    assert evidence.contact.single_contig.target_edge_count == (
        candidate.target_edge_count
    )
    assert evidence.contig_embedding is not None
    assert evidence.contig_embedding.distance == candidate.latent_distance
    assert evidence.contig_embedding.target_radius == candidate.target_radius
    assert decision.status is DecisionStatus.ACCEPT
    assert index.counters.full_edge_scans == 0


def test_radius_is_left_to_the_shared_policy_gate() -> None:
    unbinned = np.asarray(
        [
            [0.8, 0.2],
            [0.0, 1.0],
            [1.0, 1.0],
            [np.nan, np.nan],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    _index, state, profiles = _base_fixture(
        unbinned_embedding=unbinned,
    )
    candidate = generate_recruit_candidate(
        contig_idx=4,
        state=state,
        profiles=profiles,
    )
    assert candidate.status is RecruitCandidateStatus.READY
    assert candidate.proposal is not None

    evidence = ActionEvaluator(
        state=state,
        profiles=profiles,
    ).evaluate(candidate.proposal)
    decision = ActionPolicy().decide(evidence)

    assert candidate.latent_distance is not None
    assert candidate.target_radius is not None
    assert candidate.latent_distance > candidate.target_radius
    assert decision.status is DecisionStatus.ABSTAIN
    assert decision.reason_code == "embedding_unsupported"
    assert decision.gates[-1].reason == (
        "recruit_outside_target_embedding_radius"
    )


def test_scg_conflict_is_rejected_by_shared_evaluator() -> None:
    _index, state, profiles = _base_fixture(
        markers={
            "a": ("GrpE",),
            "u": ("GrpE",),
        }
    )
    candidate = generate_recruit_candidate(
        contig_idx=4,
        state=state,
        profiles=profiles,
    )
    assert candidate.proposal is not None

    decision = ActionPolicy().decide(
        ActionEvaluator(
            state=state,
            profiles=profiles,
        ).evaluate(candidate.proposal)
    )

    assert decision.status is DecisionStatus.REJECT
    assert decision.reason_code == "scg_conflict"
    assert decision.gates[-1].reason == "target_bin_gained_scg_duplication"


def test_scg_suspect_contact_target_is_not_recruitable() -> None:
    _index, state, profiles = _base_fixture(
        markers={
            "a": ("GrpE",),
            "b": ("GrpE",),
        }
    )

    candidate = generate_recruit_candidate(
        contig_idx=4,
        state=state,
        profiles=profiles,
    )

    assert candidate.status is RecruitCandidateStatus.ABSTAIN
    assert candidate.reason == "contact_best_target_not_stable"
    assert candidate.contact_target_bin_idx == 0
    assert candidate.proposal is None


def test_one_contact_edge_is_insufficient_for_recruitment() -> None:
    index = build_contact_index_from_contacts(
        [CanonicalContact(0, ["u", "a"], [0.5, 0.5], 2, 2, 1.0)],
        contig_names=("a", "u"),
    )
    state = RefineState(
        contact_index=index,
        assignment=np.asarray([0, -1], dtype=np.int32),
        contig_lengths=np.asarray([2000, 1000]),
        n_bins=1,
    )
    inputs = EvidenceInputs.from_arrays(
        contact_index=index,
        embedding=np.asarray([[1.0, 0.0], [1.0, 0.0]]),
    )
    profiles = EvidenceProfileState(refine_state=state, inputs=inputs)

    candidate = generate_recruit_candidate(
        contig_idx=1,
        state=state,
        profiles=profiles,
    )

    assert candidate.status is RecruitCandidateStatus.ABSTAIN
    assert candidate.reason == "contact_supporting_edges_insufficient"
    assert candidate.target_edge_count == 1


def test_unique_contact_target_below_required_share_abstains() -> None:
    index = build_contact_index_from_contacts(
        [
            CanonicalContact(0, ["u", "a"], [0.5, 0.5], 2, 2, 1.0),
            CanonicalContact(1, ["u", "b"], [0.5, 0.5], 2, 2, 1.0),
            CanonicalContact(2, ["u", "c"], [0.5, 0.5], 2, 2, 1.0),
        ],
        contig_names=("a", "b", "c", "u"),
    )
    state = RefineState(
        contact_index=index,
        assignment=np.asarray([0, 0, 1, -1], dtype=np.int32),
        contig_lengths=np.asarray([2000, 2000, 2000, 1000]),
        n_bins=2,
    )
    inputs = EvidenceInputs.from_arrays(
        contact_index=index,
        embedding=np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        ),
    )
    profiles = EvidenceProfileState(refine_state=state, inputs=inputs)

    candidate = generate_recruit_candidate(
        contig_idx=3,
        state=state,
        profiles=profiles,
        config=RecruitGeneratorConfig(min_target_share=0.75),
    )

    assert candidate.status is RecruitCandidateStatus.ABSTAIN
    assert candidate.reason == "contact_target_share_below_minimum"
    assert candidate.contact_target_bin_idx == 0
    assert candidate.target_edge_count == 2
    assert candidate.target_share is not None
    assert 0.6 < candidate.target_share < 0.7


def test_recruitment_pass_refreshes_after_acceptance_before_next_apply() -> None:
    names = ("a", "b", "u1", "u2")
    index = build_contact_index_from_contacts(
        [
            CanonicalContact(0, ["u1", "a"], [0.5, 0.5], 2, 2, 1.0),
            CanonicalContact(1, ["u1", "b"], [0.5, 0.5], 2, 2, 1.0),
            CanonicalContact(2, ["u2", "a"], [0.5, 0.5], 2, 2, 1.0),
            CanonicalContact(3, ["u2", "b"], [0.5, 0.5], 2, 2, 1.0),
        ],
        contig_names=names,
    )
    state = RefineState(
        contact_index=index,
        assignment=np.asarray([0, 0, -1, -1], dtype=np.int32),
        contig_lengths=np.asarray([2000, 2000, 1000, 1000]),
        n_bins=1,
    )
    inputs = EvidenceInputs.from_arrays(
        contact_index=index,
        embedding=np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        ),
        scg_markers_by_contig={
            "u1": ("GrpE",),
            "u2": ("GrpE",),
        },
    )
    profiles = EvidenceProfileState(refine_state=state, inputs=inputs)
    index.counters.reset()

    result = run_recruitment_pass(
        state=state,
        profiles=profiles,
    )

    assert tuple(
        candidate.contig_idx
        for candidate in result.initial_batch.candidates
    ) == (2, 3)
    assert result.accepted_count == 1
    assert result.final_state_version == 1
    assert int(state.assignment[2]) == 0
    assert int(state.assignment[3]) == -1
    assert result.attempts[0].decision is not None
    assert result.attempts[0].decision.status is DecisionStatus.ACCEPT
    assert result.attempts[1].evaluated_candidate.proposal is not None
    assert result.attempts[1].evaluated_candidate.proposal.state_version == 1
    assert result.attempts[1].decision is not None
    assert result.attempts[1].decision.status is DecisionStatus.REJECT
    assert result.attempts[1].decision.reason_code == "scg_conflict"
    assert index.counters.full_edge_scans == 0
