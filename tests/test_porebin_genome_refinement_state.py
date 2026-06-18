from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from porebin_genome.evidence.canonical import CanonicalContact
from porebin_genome.refinement.contact_index import (
    ContactIndex,
    build_contact_index_from_contacts,
)
from porebin_genome.refinement.state import (
    ActionDelta,
    AssignmentChange,
    RefineState,
    RefineStateError,
)


def _index() -> ContactIndex:
    contacts = [
        CanonicalContact(0, ["a", "b", "c"], [0.4, 0.3, 0.3], 3, 3, 1.0),
        CanonicalContact(1, ["b", "c", "d"], [0.2, 0.5, 0.3], 3, 3, 0.8),
        CanonicalContact(2, ["d", "e"], [0.6, 0.4], 2, 2, 0.7),
        CanonicalContact(3, ["a", "e"], [0.5, 0.5], 2, 2, 0.9),
        CanonicalContact(4, ["b", "d"], [0.5, 0.5], 2, 2, 0.6),
    ]
    return build_contact_index_from_contacts(
        contacts,
        contig_names=("a", "b", "c", "d", "e", "isolated"),
    )


def _state() -> RefineState:
    return RefineState(
        contact_index=_index(),
        assignment=np.asarray([0, 0, 1, 1, -1, -1], dtype=np.int32),
        contig_lengths=np.asarray(
            [1000, 2000, 3000, 4000, 5000, 6000],
            dtype=np.int64,
        ),
        n_bins=2,
    )


@dataclass(frozen=True)
class _Snapshot:
    assignment: np.ndarray
    numerator: np.ndarray
    denominator: np.ndarray
    coherence: np.ndarray
    total_length: np.ndarray
    members: tuple[frozenset[int], ...]
    edge_masses: tuple[dict[int, float], ...]
    n_bins: int
    version: int


def _snapshot(state: RefineState) -> _Snapshot:
    return _Snapshot(
        assignment=state.assignment.copy(),
        numerator=state.contact_numerator.copy(),
        denominator=state.contact_denominator.copy(),
        coherence=state.contact_coherence.copy(),
        total_length=state.bin_total_length.copy(),
        members=tuple(state.bin_members(idx) for idx in range(state.n_bins)),
        edge_masses=tuple(
            state.edge_masses(edge_idx)
            for edge_idx in range(state.contact_index.n_edges)
        ),
        n_bins=state.n_bins,
        version=state.version,
    )


def _assert_snapshot_equal(left: _Snapshot, right: _Snapshot) -> None:
    assert np.array_equal(left.assignment, right.assignment)
    assert np.array_equal(left.numerator, right.numerator)
    assert np.array_equal(left.denominator, right.denominator)
    assert np.array_equal(left.coherence, right.coherence)
    assert np.array_equal(left.total_length, right.total_length)
    assert left.members == right.members
    assert left.edge_masses == right.edge_masses
    assert left.n_bins == right.n_bins
    assert left.version == right.version


def _after_assignment(state: RefineState, delta: ActionDelta) -> np.ndarray:
    labels = state.assignment.copy()
    for change in delta.changes:
        labels[change.contig_idx] = change.new_bin_idx
    return labels


def _assert_delta_matches_brute_force(
    state: RefineState,
    delta: ActionDelta,
) -> None:
    after_labels = _after_assignment(state, delta)
    brute = state.contact_index.build_coherence_cache(
        after_labels,
        n_bins=delta.n_bins_after,
    )
    for offset, bin_idx in enumerate(delta.affected_bins):
        assert delta.numerator_after[offset] == pytest.approx(
            brute.numerator[bin_idx],
            abs=1e-12,
        )
        assert delta.denominator_after[offset] == pytest.approx(
            brute.denominator[bin_idx],
            abs=1e-12,
        )
        assert delta.coherence_after[offset] == pytest.approx(
            brute.coherence[bin_idx],
            abs=1e-12,
        )


@pytest.mark.parametrize(
    ("action_name", "changes"),
    [
        (
            "recruit",
            (AssignmentChange(contig_idx=4, old_bin_idx=-1, new_bin_idx=1),),
        ),
        (
            "merge",
            (
                AssignmentChange(contig_idx=2, old_bin_idx=1, new_bin_idx=0),
                AssignmentChange(contig_idx=3, old_bin_idx=1, new_bin_idx=0),
            ),
        ),
        (
            "split",
            (AssignmentChange(contig_idx=1, old_bin_idx=0, new_bin_idx=2),),
        ),
    ],
)
def test_three_refine_action_shapes_match_brute_force_and_rollback(
    action_name: str,
    changes: tuple[AssignmentChange, ...],
) -> None:
    _ = action_name
    state = _state()
    before = _snapshot(state)
    delta = state.evaluate(changes)

    _assert_delta_matches_brute_force(state, delta)
    state.apply(delta)

    brute_after = state.contact_index.build_coherence_cache(
        state.assignment,
        n_bins=state.n_bins,
    )
    assert state.contact_numerator == pytest.approx(brute_after.numerator)
    assert state.contact_denominator == pytest.approx(brute_after.denominator)
    assert state.contact_coherence == pytest.approx(brute_after.coherence)

    state.rollback(delta)
    _assert_snapshot_equal(_snapshot(state), before)


def test_simultaneous_changes_on_same_edge_are_counted_once() -> None:
    state = _state()
    changes = (
        AssignmentChange(contig_idx=0, old_bin_idx=0, new_bin_idx=1),
        AssignmentChange(contig_idx=2, old_bin_idx=1, new_bin_idx=0),
    )

    state.contact_index.counters.reset()
    delta = state.evaluate(changes)

    assert delta.affected_edges.tolist() == [0, 1, 3]
    assert len(delta.edge_updates) == 3
    assert state.contact_index.counters.full_edge_scans == 0
    assert state.contact_index.counters.local_queries == 2
    assert state.contact_index.counters.local_edges_visited == 4
    _assert_delta_matches_brute_force(state, delta)


def test_evaluate_does_not_mutate_state_or_scan_all_edges() -> None:
    state = _state()
    before = _snapshot(state)
    state.contact_index.counters.reset()

    delta = state.evaluate(
        [AssignmentChange(contig_idx=0, old_bin_idx=0, new_bin_idx=1)]
    )

    _assert_snapshot_equal(_snapshot(state), before)
    assert delta.state_version == 0
    assert state.contact_index.counters.full_edge_scans == 0
    assert state.contact_index.counters.local_edges_visited == 2


def test_sequential_actions_use_updated_sparse_edge_masses() -> None:
    state = _state()
    first = state.evaluate(
        [AssignmentChange(contig_idx=0, old_bin_idx=0, new_bin_idx=1)]
    )
    state.apply(first)
    second = state.evaluate(
        [AssignmentChange(contig_idx=2, old_bin_idx=1, new_bin_idx=0)]
    )

    _assert_delta_matches_brute_force(state, second)
    state.apply(second)
    brute = state.contact_index.build_coherence_cache(
        state.assignment,
        n_bins=state.n_bins,
    )
    assert state.contact_numerator == pytest.approx(brute.numerator)
    assert state.contact_denominator == pytest.approx(brute.denominator)

    state.rollback(second)
    state.rollback(first)
    assert state.assignment.tolist() == [0, 0, 1, 1, -1, -1]


def test_new_bins_must_be_contiguous_and_are_removed_on_rollback() -> None:
    state = _state()

    with pytest.raises(RefineStateError, match="contiguous suffix"):
        state.evaluate(
            [AssignmentChange(contig_idx=1, old_bin_idx=0, new_bin_idx=3)]
        )

    delta = state.evaluate(
        [
            AssignmentChange(contig_idx=0, old_bin_idx=0, new_bin_idx=2),
            AssignmentChange(contig_idx=1, old_bin_idx=0, new_bin_idx=3),
        ]
    )
    state.apply(delta)
    assert state.n_bins == 4
    assert state.bin_members(2) == frozenset({0})
    assert state.bin_members(3) == frozenset({1})
    assert state.bin_total_length.tolist() == [0, 7000, 1000, 2000]

    state.rollback(delta)
    assert state.n_bins == 2
    assert state.bin_total_length.tolist() == [3000, 7000]


def test_isolated_contig_action_changes_membership_but_not_contact_values() -> None:
    state = _state()
    before_n = state.contact_numerator.copy()
    before_d = state.contact_denominator.copy()
    delta = state.evaluate(
        [AssignmentChange(contig_idx=5, old_bin_idx=-1, new_bin_idx=0)]
    )

    assert delta.affected_edges.tolist() == []
    assert delta.delta_numerator == pytest.approx((0.0,))
    assert delta.delta_denominator == pytest.approx((0.0,))
    state.apply(delta)
    assert state.bin_members(0) == frozenset({0, 1, 5})
    assert state.bin_total_length[0] == 9000
    assert np.array_equal(state.contact_numerator, before_n)
    assert np.array_equal(state.contact_denominator, before_d)


def test_invalid_and_stale_changes_are_rejected() -> None:
    state = _state()
    with pytest.raises(RefineStateError, match="at least one"):
        state.evaluate([])
    with pytest.raises(RefineStateError, match="more than once"):
        state.evaluate(
            [
                AssignmentChange(0, 0, 1),
                AssignmentChange(0, 0, -1),
            ]
        )
    with pytest.raises(RefineStateError, match="Stale old bin"):
        state.evaluate([AssignmentChange(0, 1, -1)])
    with pytest.raises(RefineStateError, match="No-op"):
        state.evaluate([AssignmentChange(0, 0, 0)])

    first = state.evaluate([AssignmentChange(0, 0, 1)])
    stale = state.evaluate([AssignmentChange(1, 0, 1)])
    state.apply(first)
    with pytest.raises(RefineStateError, match="Stale ActionDelta"):
        state.apply(stale)


def test_delta_cannot_cross_states_or_rollback_out_of_order() -> None:
    first_state = _state()
    second_state = _state()
    foreign = first_state.evaluate([AssignmentChange(0, 0, 1)])
    with pytest.raises(RefineStateError, match="different RefineState"):
        second_state.apply(foreign)

    first_state.apply(foreign)
    nested = first_state.evaluate([AssignmentChange(1, 0, 1)])
    first_state.apply(nested)
    with pytest.raises(RefineStateError, match="most recently"):
        first_state.rollback(foreign)


def test_state_arrays_are_read_only_to_callers() -> None:
    state = _state()

    with pytest.raises(ValueError):
        state.assignment[0] = 1
    with pytest.raises(ValueError):
        state.contact_numerator[0] = 0.0
    with pytest.raises(ValueError):
        state.bin_total_length[0] = 0
