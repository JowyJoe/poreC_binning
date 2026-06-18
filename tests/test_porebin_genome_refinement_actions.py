from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from porebin_genome.evidence.canonical import CanonicalContact
from porebin_genome.refinement import (
    ActionEvaluator,
    ActionPolicy,
    ActionType,
    DecisionStatus,
    EvaluationConfig,
    GateStatus,
    PolicyConfig,
    Proposal,
    apply_decision,
    rollback_decision,
)
from porebin_genome.refinement.contact_index import (
    build_contact_index_from_contacts,
)
from porebin_genome.refinement.profiles import (
    EvidenceInputs,
    EvidenceProfileState,
)
from porebin_genome.refinement.state import AssignmentChange, RefineState


def _fixture():
    index = build_contact_index_from_contacts(
        [
            CanonicalContact(0, ["a", "b"], [0.5, 0.5], 2, 2, 1.0),
            CanonicalContact(1, ["c", "d"], [0.5, 0.5], 2, 2, 1.0),
            CanonicalContact(2, ["b", "c"], [0.5, 0.5], 2, 2, 0.8),
            CanonicalContact(3, ["e", "c"], [0.5, 0.5], 2, 2, 0.9),
            CanonicalContact(4, ["e", "d"], [0.5, 0.5], 2, 2, 0.9),
            CanonicalContact(5, ["f", "a"], [0.5, 0.5], 2, 2, 0.9),
            CanonicalContact(6, ["f", "b"], [0.5, 0.5], 2, 2, 0.9),
        ],
        contig_names=("a", "b", "c", "d", "e", "f"),
    )
    state = RefineState(
        contact_index=index,
        assignment=np.asarray([0, 0, 1, 1, -1, -1], dtype=np.int32),
        contig_lengths=np.asarray(
            [1000, 1100, 1200, 1300, 900, 950],
            dtype=np.int64,
        ),
        n_bins=2,
    )
    inputs = EvidenceInputs.from_arrays(
        contact_index=index,
        embedding=np.asarray(
            [
                [1.0, 0.0],
                [0.98, 0.02],
                [0.0, 1.0],
                [0.02, 0.98],
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        ),
        tnf=np.asarray(
            [
                [1.0, 0.0],
                [0.95, 0.05],
                [0.0, 1.0],
                [0.05, 0.95],
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        ),
        coverage=np.asarray([10.0, 11.0, 100.0, 101.0, 100.0, 10.0]),
        scg_markers_by_contig={
            "a": ("GrpE",),
            "b": ("GrpE",),
            "c": ("GrpE", "PGK"),
            "e": ("PGK",),
        },
    )
    profiles = EvidenceProfileState(refine_state=state, inputs=inputs)
    evaluator = ActionEvaluator(
        state=state,
        profiles=profiles,
        config=EvaluationConfig(min_bin_contigs=1, min_bin_length=1),
    )
    return index, state, profiles, evaluator


def _proposal(
    state: RefineState,
    action_type: ActionType,
    *changes: AssignmentChange,
    proposal_id: str | None = None,
) -> Proposal:
    return Proposal(
        proposal_id=proposal_id or action_type.value,
        action_type=action_type,
        changes=tuple(changes),
        state_version=state.version,
    )


@pytest.mark.parametrize(
    ("action_type", "changes"),
    [
        (
            ActionType.SPLIT,
            (AssignmentChange(1, 0, 2),),
        ),
        (
            ActionType.MERGE,
            (
                AssignmentChange(2, 1, 0),
                AssignmentChange(3, 1, 0),
            ),
        ),
        (
            ActionType.RECRUIT,
            (AssignmentChange(4, -1, 1),),
        ),
    ],
)
def test_three_action_shapes_use_one_evaluator_without_mutating_state(
    action_type: ActionType,
    changes: tuple[AssignmentChange, ...],
) -> None:
    index, state, profiles, evaluator = _fixture()
    assignment_before = state.assignment.copy()
    index.counters.reset()
    profiles.counters.reset()

    evidence = evaluator.evaluate(
        _proposal(state, action_type, *changes)
    )

    assert evidence.structural.status is GateStatus.PASS
    assert evidence.contact_delta is not None
    assert evidence.profile_update is not None
    assert evidence.contact is not None
    assert np.array_equal(state.assignment, assignment_before)
    assert state.version == profiles.version == 0
    assert index.counters.full_edge_scans == 0
    assert profiles.counters.bin_profiles_built <= 3


def test_split_reports_direct_scg_resolution_and_hypergraph_separation() -> None:
    _index, state, _profiles, evaluator = _fixture()

    evidence = evaluator.evaluate(
        _proposal(
            state,
            ActionType.SPLIT,
            AssignmentChange(1, 0, 2),
        )
    )

    assert evidence.scg is not None
    assert evidence.scg.status is GateStatus.PASS
    assert evidence.scg.duplicate_burden_before == 1
    assert evidence.scg.duplicate_burden_after == 0
    assert evidence.scg.duplicate_change == -1
    assert evidence.scg.resolved_duplicate_markers == ("GrpE",)
    assert evidence.contact is not None
    assert evidence.contact.split_separation == pytest.approx(
        0.925 / (0.925 + 0.5)
    )
    assert evidence.contact.split_within_support == pytest.approx(0.925)
    assert evidence.contact.split_cross_support == pytest.approx(0.25)


def test_split_may_reduce_burden_without_resolving_every_duplicate() -> None:
    index = build_contact_index_from_contacts(
        [
            CanonicalContact(0, ["a", "b"], [0.5, 0.5], 2, 2, 1.0),
            CanonicalContact(1, ["c", "d"], [0.5, 0.5], 2, 2, 1.0),
        ],
        contig_names=("a", "b", "c", "d"),
    )
    state = RefineState(
        contact_index=index,
        assignment=np.asarray([0, 0, 0, 0], dtype=np.int32),
        contig_lengths=np.asarray([1000, 1000, 1000, 1000]),
        n_bins=1,
    )
    inputs = EvidenceInputs.from_arrays(
        contact_index=index,
        embedding=np.asarray(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        ),
        scg_markers_by_contig={
            name: ("GrpE",)
            for name in index.contig_names
        },
    )
    profiles = EvidenceProfileState(refine_state=state, inputs=inputs)
    evaluator = ActionEvaluator(state=state, profiles=profiles)

    evidence = evaluator.evaluate(
        _proposal(
            state,
            ActionType.SPLIT,
            AssignmentChange(2, 0, 1),
            AssignmentChange(3, 0, 1),
        )
    )

    assert evidence.scg is not None
    assert evidence.scg.duplicate_burden_before == 3
    assert evidence.scg.duplicate_burden_after == 2
    assert evidence.scg.duplicate_change == -1
    assert evidence.scg.new_duplicate_markers == ("GrpE",)
    assert evidence.scg.status is GateStatus.PASS


def test_split_without_scg_improvement_is_rejected() -> None:
    _index, state, _profiles, evaluator = _fixture()

    evidence = evaluator.evaluate(
        _proposal(
            state,
            ActionType.SPLIT,
            AssignmentChange(3, 1, 2),
        )
    )
    decision = ActionPolicy().decide(evidence)

    assert evidence.scg is not None
    assert evidence.scg.duplicate_change == 0
    assert evidence.scg.status is GateStatus.CONFLICT
    assert evidence.scg.reason == "split_did_not_reduce_scg_duplication"
    assert decision.status is DecisionStatus.REJECT
    assert decision.reason_code == "scg_conflict"


def test_new_scg_duplication_is_a_hard_veto_before_ml_evidence() -> None:
    _index, state, _profiles, evaluator = _fixture()
    evidence = evaluator.evaluate(
        _proposal(
            state,
            ActionType.MERGE,
            AssignmentChange(2, 1, 0),
            AssignmentChange(3, 1, 0),
        )
    )

    assert evidence.scg is not None
    assert evidence.scg.status is GateStatus.CONFLICT
    assert evidence.scg.duplicate_change == 1
    decision = ActionPolicy().decide(evidence)
    assert decision.status is DecisionStatus.REJECT
    assert decision.reason_code == "scg_conflict"
    assert [gate.name for gate in decision.gates] == ["structural", "scg"]


def test_changed_contig_without_scg_is_uninformative_not_rejected() -> None:
    _index, state, _profiles, evaluator = _fixture()
    evidence = evaluator.evaluate(
        _proposal(
            state,
            ActionType.RECRUIT,
            AssignmentChange(5, -1, 0),
        )
    )

    assert evidence.scg is not None
    assert evidence.scg.status is GateStatus.UNINFORMATIVE
    assert evidence.scg.reason == "changed_contig_has_no_scg"

    policy = ActionPolicy()
    decision = policy.decide(evidence)
    assert decision.status is DecisionStatus.ACCEPT


def test_recruit_uses_direct_hgvae_radius_after_contact_gate() -> None:
    _index, state, _profiles, evaluator = _fixture()
    evidence = evaluator.evaluate(
        _proposal(
            state,
            ActionType.RECRUIT,
            AssignmentChange(5, -1, 0),
        )
    )

    decision = ActionPolicy().decide(evidence)
    assert decision.status is DecisionStatus.ACCEPT
    assert evidence.contig_embedding is not None
    assert evidence.contig_embedding.distance is not None
    assert evidence.contig_embedding.target_radius is not None
    assert (
        evidence.contig_embedding.distance
        <= evidence.contig_embedding.target_radius
    )

    outside = replace(
        evidence,
        contig_embedding=replace(
            evidence.contig_embedding,
            distance=evidence.contig_embedding.target_radius + 1.0,
        ),
    )
    outside_decision = ActionPolicy().decide(outside)
    assert outside_decision.status is DecisionStatus.ABSTAIN
    assert outside_decision.reason_code == "embedding_unsupported"
    assert outside_decision.gates[-1].reason == (
        "recruit_outside_target_embedding_radius"
    )


def test_malformed_and_stale_proposals_have_distinct_outcomes() -> None:
    _index, state, profiles, evaluator = _fixture()
    malformed = evaluator.evaluate(
        _proposal(
            state,
            ActionType.MERGE,
            AssignmentChange(2, 1, 0),
        )
    )
    malformed_decision = ActionPolicy().decide(malformed)
    assert malformed_decision.status is DecisionStatus.REJECT
    assert malformed_decision.reason_code == "structural_invalid"
    assert malformed.contact_delta is None

    accepted_evidence = evaluator.evaluate(
        _proposal(
            state,
            ActionType.RECRUIT,
            AssignmentChange(5, -1, 0),
            proposal_id="apply-first",
        )
    )
    accepted = ActionPolicy(
        PolicyConfig(
            recruit_min_margin=0.5,
            recruit_min_supporting_edges=2,
        )
    ).decide(accepted_evidence)
    apply_decision(state=state, profiles=profiles, decision=accepted)

    stale = evaluator.evaluate(
        Proposal(
            proposal_id="stale",
            action_type=ActionType.RECRUIT,
            changes=(AssignmentChange(4, -1, 1),),
            state_version=0,
        )
    )
    stale_decision = ActionPolicy().decide(stale)
    assert stale.structural.status is GateStatus.UNINFORMATIVE
    assert stale_decision.status is DecisionStatus.ABSTAIN
    assert stale_decision.reason_code == "stale_or_unsynchronized"


def test_evaluator_rejects_profile_cache_from_another_state() -> None:
    _index, state, _profiles, _evaluator = _fixture()
    _other_index, _other_state, other_profiles, _other_evaluator = _fixture()

    with pytest.raises(ValueError, match="same RefineState"):
        ActionEvaluator(state=state, profiles=other_profiles)


def test_accepted_decision_apply_and_rollback_keep_states_synchronized() -> None:
    _index, state, profiles, evaluator = _fixture()
    before_assignment = state.assignment.copy()
    before_bin = profiles.profile(0)
    evidence = evaluator.evaluate(
        _proposal(
            state,
            ActionType.RECRUIT,
            AssignmentChange(5, -1, 0),
        )
    )
    decision = ActionPolicy(
        PolicyConfig(
            recruit_min_margin=0.5,
            recruit_min_supporting_edges=2,
        )
    ).decide(evidence)
    assert decision.status is DecisionStatus.ACCEPT

    apply_decision(state=state, profiles=profiles, decision=decision)
    assert state.version == profiles.version == 1
    assert int(state.assignment[5]) == 0
    assert profiles.profile(0).member_count == 3

    rollback_decision(state=state, profiles=profiles, decision=decision)
    assert state.version == profiles.version == 0
    assert np.array_equal(state.assignment, before_assignment)
    assert profiles.profile(0) == before_bin


def test_split_policy_uses_explicit_separation_threshold() -> None:
    _index, state, _profiles, evaluator = _fixture()
    evidence = evaluator.evaluate(
        _proposal(
            state,
            ActionType.SPLIT,
            AssignmentChange(1, 0, 2),
        )
    )
    policy = ActionPolicy(
        PolicyConfig(
            split_min_separation=0.6,
            max_embedding_worsening=0.0,
        )
    )

    decision = policy.decide(evidence)

    assert decision.status is DecisionStatus.ACCEPT
    assert decision.gates[2].name == "contact"
    assert decision.gates[2].value == pytest.approx(
        0.925 / (0.925 + 0.5)
    )
