"""Incremental assignment state for the replacement refine engine."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from porebin_genome.refinement.contact_index import (
    ContactCoherenceCache,
    ContactIndex,
)


IntArray = NDArray[np.integer]
FloatArray = NDArray[np.floating]
EdgeMasses = tuple[tuple[int, float], ...]


class RefineStateError(RuntimeError):
    """Raised when an incremental state transition is invalid or stale."""


@dataclass(frozen=True)
class AssignmentChange:
    """One simultaneous contig assignment change; -1 denotes unassigned."""

    contig_idx: int
    old_bin_idx: int
    new_bin_idx: int


@dataclass(frozen=True)
class EdgeMassUpdate:
    """Before/after bin mass for one hyperedge touched by an action."""

    edge_idx: int
    before: EdgeMasses
    after: EdgeMasses
    prior_override: EdgeMasses | None = field(repr=False)


@dataclass(frozen=True)
class ActionDelta:
    """Exact local state transition calculated without mutating RefineState."""

    changes: tuple[AssignmentChange, ...]
    affected_edges: IntArray
    edge_updates: tuple[EdgeMassUpdate, ...]
    affected_bins: tuple[int, ...]
    numerator_before: tuple[float, ...]
    numerator_after: tuple[float, ...]
    denominator_before: tuple[float, ...]
    denominator_after: tuple[float, ...]
    coherence_before: tuple[float, ...]
    coherence_after: tuple[float, ...]
    n_bins_before: int
    n_bins_after: int
    state_version: int
    _state_token: object = field(repr=False, compare=False)

    @property
    def delta_numerator(self) -> tuple[float, ...]:
        return tuple(
            after - before
            for before, after in zip(
                self.numerator_before,
                self.numerator_after,
                strict=True,
            )
        )

    @property
    def delta_denominator(self) -> tuple[float, ...]:
        return tuple(
            after - before
            for before, after in zip(
                self.denominator_before,
                self.denominator_after,
                strict=True,
            )
        )


class RefineState:
    """Mutable assignment plus exact local contact-coherence caches."""

    def __init__(
        self,
        *,
        contact_index: ContactIndex,
        assignment: Sequence[int] | IntArray,
        contig_lengths: Sequence[int] | IntArray,
        n_bins: int | None = None,
        eps: float = 1e-12,
    ) -> None:
        self.contact_index = contact_index
        self._assignment = _validated_assignment(
            assignment,
            n_contigs=contact_index.n_contigs,
        )
        self._contig_lengths = _validated_lengths(
            contig_lengths,
            n_contigs=contact_index.n_contigs,
        )
        inferred_bins = (
            int(self._assignment[self._assignment >= 0].max()) + 1
            if bool(np.any(self._assignment >= 0))
            else 0
        )
        resolved_n_bins = inferred_bins if n_bins is None else int(n_bins)
        if resolved_n_bins < inferred_bins or resolved_n_bins < 0:
            raise ValueError(
                f"n_bins={resolved_n_bins} cannot represent assignment labels "
                f"requiring {inferred_bins} bins."
            )
        if not math.isfinite(float(eps)) or float(eps) <= 0.0:
            raise ValueError("eps must be a finite positive number.")

        initial_cache = contact_index.build_coherence_cache(
            self._assignment,
            n_bins=resolved_n_bins,
            eps=float(eps),
        )
        self._base_contact_cache = initial_cache
        self._edge_mass_overrides: dict[int, EdgeMasses] = {}
        self._numerator = np.asarray(
            initial_cache.numerator,
            dtype=np.float64,
        ).copy()
        self._denominator = np.asarray(
            initial_cache.denominator,
            dtype=np.float64,
        ).copy()
        self._coherence = np.asarray(
            initial_cache.coherence,
            dtype=np.float64,
        ).copy()
        self._bin_members = [set() for _ in range(resolved_n_bins)]
        self._bin_total_length = np.zeros(resolved_n_bins, dtype=np.int64)
        for contig_idx, bin_idx_raw in enumerate(self._assignment):
            bin_idx = int(bin_idx_raw)
            if bin_idx < 0:
                continue
            self._bin_members[bin_idx].add(contig_idx)
            self._bin_total_length[bin_idx] += int(
                self._contig_lengths[contig_idx]
            )

        self._eps = float(eps)
        self._version = 0
        self._state_token = object()
        self._applied_stack: list[ActionDelta] = []

    @property
    def n_contigs(self) -> int:
        return self.contact_index.n_contigs

    @property
    def n_bins(self) -> int:
        return len(self._bin_members)

    @property
    def version(self) -> int:
        return self._version

    @property
    def assignment(self) -> IntArray:
        return _readonly_view(self._assignment)

    @property
    def contig_lengths(self) -> IntArray:
        return _readonly_view(self._contig_lengths)

    @property
    def contact_numerator(self) -> FloatArray:
        return _readonly_view(self._numerator)

    @property
    def contact_denominator(self) -> FloatArray:
        return _readonly_view(self._denominator)

    @property
    def contact_coherence(self) -> FloatArray:
        return _readonly_view(self._coherence)

    @property
    def bin_total_length(self) -> IntArray:
        return _readonly_view(self._bin_total_length)

    def bin_members(self, bin_idx: int) -> frozenset[int]:
        self._require_existing_bin(bin_idx)
        return frozenset(self._bin_members[int(bin_idx)])

    def next_bin_indices(self, count: int) -> tuple[int, ...]:
        """Return the contiguous new IDs an action may introduce."""
        if int(count) < 1:
            raise ValueError("count must be positive.")
        return tuple(range(self.n_bins, self.n_bins + int(count)))

    def edge_masses(self, edge_idx: int) -> dict[int, float]:
        """Return the current cached m[e,b] values for one hyperedge."""
        edge_idx = int(edge_idx)
        current = self._edge_mass_overrides.get(edge_idx)
        if current is None:
            return self._base_contact_cache.edge_masses(edge_idx)
        return dict(current)

    def evaluate(
        self,
        changes: Iterable[AssignmentChange],
    ) -> ActionDelta:
        """Calculate an exact local transition without mutating state."""
        normalized = self._validate_changes(changes)
        changed_by_contig = {
            change.contig_idx: change
            for change in normalized
        }
        affected_edges = self.contact_index.affected_edges(
            change.contig_idx for change in normalized
        )

        delta_n: dict[int, float] = {}
        delta_d: dict[int, float] = {}
        edge_updates: list[EdgeMassUpdate] = []
        for edge_idx_raw in affected_edges:
            edge_idx = int(edge_idx_raw)
            before = _canonical_masses(self.edge_masses(edge_idx))
            before_map = dict(before)
            after_map = dict(before_map)
            member_slice = self.contact_index.edge_member_slice(edge_idx)
            for position in range(member_slice.start, member_slice.stop):
                contig_idx = int(self.contact_index.edge_members[position])
                change = changed_by_contig.get(contig_idx)
                if change is None:
                    continue
                alpha = float(self.contact_index.edge_alpha[position])
                if change.old_bin_idx >= 0:
                    _add_mass(after_map, change.old_bin_idx, -alpha)
                if change.new_bin_idx >= 0:
                    _add_mass(after_map, change.new_bin_idx, alpha)
            after = _canonical_masses(after_map)
            after_values = dict(after)

            reliability = float(self.contact_index.edge_reliability[edge_idx])
            for bin_idx in set(before_map) | set(after_values):
                old_mass = float(before_map.get(bin_idx, 0.0))
                new_mass = float(after_values.get(bin_idx, 0.0))
                denominator_change = reliability * (new_mass - old_mass)
                numerator_change = reliability * (
                    new_mass * new_mass - old_mass * old_mass
                )
                if denominator_change != 0.0:
                    delta_d[bin_idx] = float(
                        delta_d.get(bin_idx, 0.0) + denominator_change
                    )
                if numerator_change != 0.0:
                    delta_n[bin_idx] = float(
                        delta_n.get(bin_idx, 0.0) + numerator_change
                    )

            edge_updates.append(
                EdgeMassUpdate(
                    edge_idx=edge_idx,
                    before=before,
                    after=after,
                    prior_override=self._edge_mass_overrides.get(edge_idx),
                )
            )

        changed_bins = {
            bin_idx
            for change in normalized
            for bin_idx in (change.old_bin_idx, change.new_bin_idx)
            if bin_idx >= 0
        }
        affected_bins = tuple(sorted(changed_bins | set(delta_n) | set(delta_d)))
        n_bins_after = max(
            self.n_bins,
            max((change.new_bin_idx + 1 for change in normalized), default=0),
        )

        numerator_before: list[float] = []
        numerator_after: list[float] = []
        denominator_before: list[float] = []
        denominator_after: list[float] = []
        coherence_before: list[float] = []
        coherence_after: list[float] = []
        for bin_idx in affected_bins:
            before_n = (
                float(self._numerator[bin_idx])
                if bin_idx < self.n_bins
                else 0.0
            )
            before_d = (
                float(self._denominator[bin_idx])
                if bin_idx < self.n_bins
                else 0.0
            )
            after_n = _nonnegative(
                before_n + float(delta_n.get(bin_idx, 0.0)),
                label=f"contact numerator for bin {bin_idx}",
            )
            after_d = _nonnegative(
                before_d + float(delta_d.get(bin_idx, 0.0)),
                label=f"contact denominator for bin {bin_idx}",
            )
            numerator_before.append(before_n)
            numerator_after.append(after_n)
            denominator_before.append(before_d)
            denominator_after.append(after_d)
            coherence_before.append(before_n / (before_d + self._eps))
            coherence_after.append(after_n / (after_d + self._eps))

        return ActionDelta(
            changes=normalized,
            affected_edges=affected_edges,
            edge_updates=tuple(edge_updates),
            affected_bins=affected_bins,
            numerator_before=tuple(numerator_before),
            numerator_after=tuple(numerator_after),
            denominator_before=tuple(denominator_before),
            denominator_after=tuple(denominator_after),
            coherence_before=tuple(coherence_before),
            coherence_after=tuple(coherence_after),
            n_bins_before=self.n_bins,
            n_bins_after=n_bins_after,
            state_version=self._version,
            _state_token=self._state_token,
        )

    def apply(self, delta: ActionDelta) -> None:
        """Apply a previously evaluated transition if it is still current."""
        self._validate_delta_for_apply(delta)
        self._expand_bins(delta.n_bins_after)

        for change in delta.changes:
            contig_idx = change.contig_idx
            length = int(self._contig_lengths[contig_idx])
            if change.old_bin_idx >= 0:
                self._bin_members[change.old_bin_idx].remove(contig_idx)
                self._bin_total_length[change.old_bin_idx] -= length
            if change.new_bin_idx >= 0:
                self._bin_members[change.new_bin_idx].add(contig_idx)
                self._bin_total_length[change.new_bin_idx] += length
            self._assignment[contig_idx] = change.new_bin_idx

        for update in delta.edge_updates:
            self._set_edge_override(update.edge_idx, update.after)
        self._set_contact_values(
            delta.affected_bins,
            delta.numerator_after,
            delta.denominator_after,
            delta.coherence_after,
        )
        self._version += 1
        self._applied_stack.append(delta)

    def rollback(self, delta: ActionDelta) -> None:
        """Rollback the most recently applied transition exactly."""
        if delta._state_token is not self._state_token:
            raise RefineStateError("ActionDelta belongs to a different RefineState.")
        if not self._applied_stack or self._applied_stack[-1] is not delta:
            raise RefineStateError(
                "Only the most recently applied ActionDelta can be rolled back."
            )
        if self._version != delta.state_version + 1:
            raise RefineStateError("RefineState version does not match rollback delta.")

        for change in reversed(delta.changes):
            contig_idx = change.contig_idx
            length = int(self._contig_lengths[contig_idx])
            if change.new_bin_idx >= 0:
                self._bin_members[change.new_bin_idx].remove(contig_idx)
                self._bin_total_length[change.new_bin_idx] -= length
            if change.old_bin_idx >= 0:
                self._bin_members[change.old_bin_idx].add(contig_idx)
                self._bin_total_length[change.old_bin_idx] += length
            self._assignment[contig_idx] = change.old_bin_idx

        for update in delta.edge_updates:
            if update.prior_override is None:
                self._edge_mass_overrides.pop(update.edge_idx, None)
            else:
                self._edge_mass_overrides[update.edge_idx] = update.prior_override
        self._set_contact_values(
            delta.affected_bins,
            delta.numerator_before,
            delta.denominator_before,
            delta.coherence_before,
        )
        self._shrink_bins(delta.n_bins_before)
        self._version = delta.state_version
        self._applied_stack.pop()

    def _validate_changes(
        self,
        changes: Iterable[AssignmentChange],
    ) -> tuple[AssignmentChange, ...]:
        normalized = tuple(
            AssignmentChange(
                contig_idx=int(change.contig_idx),
                old_bin_idx=int(change.old_bin_idx),
                new_bin_idx=int(change.new_bin_idx),
            )
            for change in changes
        )
        if not normalized:
            raise RefineStateError("An action must contain at least one change.")

        seen_contigs: set[int] = set()
        introduced_bins: set[int] = set()
        for change in normalized:
            if change.contig_idx < 0 or change.contig_idx >= self.n_contigs:
                raise RefineStateError(
                    f"Contig index out of range: {change.contig_idx}"
                )
            if change.contig_idx in seen_contigs:
                raise RefineStateError(
                    f"Contig appears more than once in one action: "
                    f"{change.contig_idx}"
                )
            seen_contigs.add(change.contig_idx)
            if change.old_bin_idx < -1 or change.new_bin_idx < -1:
                raise RefineStateError("Bin indices must be -1 or non-negative.")
            if max(change.old_bin_idx, change.new_bin_idx) > np.iinfo(np.int32).max:
                raise RefineStateError("Bin index exceeds the int32 state contract.")
            current_bin = int(self._assignment[change.contig_idx])
            if current_bin != change.old_bin_idx:
                raise RefineStateError(
                    f"Stale old bin for contig {change.contig_idx}: "
                    f"expected={current_bin}, observed={change.old_bin_idx}"
                )
            if change.old_bin_idx == change.new_bin_idx:
                raise RefineStateError(
                    f"No-op assignment change for contig {change.contig_idx}."
                )
            if change.new_bin_idx >= self.n_bins:
                introduced_bins.add(change.new_bin_idx)

        if introduced_bins:
            expected = set(range(self.n_bins, max(introduced_bins) + 1))
            if introduced_bins != expected:
                raise RefineStateError(
                    "New bin IDs must form a contiguous suffix beginning at "
                    f"{self.n_bins}; observed={sorted(introduced_bins)}"
                )
        return normalized

    def _validate_delta_for_apply(self, delta: ActionDelta) -> None:
        if delta._state_token is not self._state_token:
            raise RefineStateError("ActionDelta belongs to a different RefineState.")
        if delta.state_version != self._version:
            raise RefineStateError(
                f"Stale ActionDelta: evaluated_at={delta.state_version}, "
                f"current={self._version}"
            )
        if delta.n_bins_before != self.n_bins:
            raise RefineStateError("ActionDelta bin count is stale.")
        for change in delta.changes:
            if int(self._assignment[change.contig_idx]) != change.old_bin_idx:
                raise RefineStateError(
                    f"Assignment changed after evaluation for contig "
                    f"{change.contig_idx}."
                )

    def _set_edge_override(
        self,
        edge_idx: int,
        masses: EdgeMasses,
    ) -> None:
        base = _canonical_masses(
            self._base_contact_cache.edge_masses(int(edge_idx))
        )
        if _masses_equal(masses, base):
            self._edge_mass_overrides.pop(int(edge_idx), None)
        else:
            self._edge_mass_overrides[int(edge_idx)] = masses

    def _set_contact_values(
        self,
        bins: tuple[int, ...],
        numerator: tuple[float, ...],
        denominator: tuple[float, ...],
        coherence: tuple[float, ...],
    ) -> None:
        for bin_idx, value_n, value_d, value_c in zip(
            bins,
            numerator,
            denominator,
            coherence,
            strict=True,
        ):
            self._numerator[bin_idx] = value_n
            self._denominator[bin_idx] = value_d
            self._coherence[bin_idx] = value_c

    def _expand_bins(self, size: int) -> None:
        if int(size) <= self.n_bins:
            return
        extension = int(size) - self.n_bins
        self._bin_members.extend(set() for _ in range(extension))
        self._bin_total_length = np.pad(
            self._bin_total_length,
            (0, extension),
            constant_values=0,
        )
        self._numerator = np.pad(
            self._numerator,
            (0, extension),
            constant_values=0.0,
        )
        self._denominator = np.pad(
            self._denominator,
            (0, extension),
            constant_values=0.0,
        )
        self._coherence = np.pad(
            self._coherence,
            (0, extension),
            constant_values=0.0,
        )

    def _shrink_bins(self, size: int) -> None:
        if int(size) > self.n_bins:
            raise RefineStateError("Cannot rollback to a larger bin count.")
        for bin_idx in range(int(size), self.n_bins):
            if self._bin_members[bin_idx] or int(self._bin_total_length[bin_idx]):
                raise RefineStateError(
                    f"Cannot remove non-empty rollback bin {bin_idx}."
                )
        del self._bin_members[int(size):]
        self._bin_total_length = self._bin_total_length[: int(size)].copy()
        self._numerator = self._numerator[: int(size)].copy()
        self._denominator = self._denominator[: int(size)].copy()
        self._coherence = self._coherence[: int(size)].copy()

    def _require_existing_bin(self, bin_idx: int) -> None:
        if int(bin_idx) < 0 or int(bin_idx) >= self.n_bins:
            raise IndexError(f"Bin index out of range: {bin_idx}")


def _validated_assignment(
    assignment: Sequence[int] | IntArray,
    *,
    n_contigs: int,
) -> NDArray[np.int32]:
    labels = np.asarray(assignment)
    if labels.ndim != 1 or len(labels) != int(n_contigs):
        raise ValueError(
            "assignment must have one label per contig: "
            f"expected={n_contigs}, observed_shape={labels.shape}"
        )
    if labels.dtype.kind not in {"i", "u"}:
        raise TypeError("assignment labels must use an integer dtype.")
    if labels.dtype.kind == "u" and bool(np.any(labels > np.iinfo(np.int32).max)):
        raise ValueError("assignment contains a bin ID outside int32 range.")
    out = labels.astype(np.int32, copy=True)
    if bool(np.any(out < -1)):
        raise ValueError("assignment labels must be -1 or non-negative.")
    return out


def _validated_lengths(
    contig_lengths: Sequence[int] | IntArray,
    *,
    n_contigs: int,
) -> NDArray[np.int64]:
    lengths = np.asarray(contig_lengths)
    if lengths.ndim != 1 or len(lengths) != int(n_contigs):
        raise ValueError(
            "contig_lengths must contain one value per contig: "
            f"expected={n_contigs}, observed_shape={lengths.shape}"
        )
    if lengths.dtype.kind not in {"i", "u"}:
        raise TypeError("contig_lengths must use an integer dtype.")
    if lengths.dtype.kind == "u" and bool(
        np.any(lengths > np.iinfo(np.int64).max)
    ):
        raise ValueError("contig length exceeds the int64 state contract.")
    out = lengths.astype(np.int64, copy=True)
    if bool(np.any(out < 0)):
        raise ValueError("contig lengths must be non-negative.")
    return out


def _add_mass(masses: dict[int, float], bin_idx: int, delta: float) -> None:
    updated = float(masses.get(int(bin_idx), 0.0) + float(delta))
    if abs(updated) <= 1e-12:
        masses.pop(int(bin_idx), None)
        return
    if updated < 0.0:
        raise RefineStateError(
            f"Hyperedge mass became negative for bin {bin_idx}: {updated}"
        )
    masses[int(bin_idx)] = updated


def _canonical_masses(masses: dict[int, float] | EdgeMasses) -> EdgeMasses:
    items = masses.items() if isinstance(masses, dict) else masses
    return tuple(
        (int(bin_idx), float(mass))
        for bin_idx, mass in sorted(items)
        if float(mass) > 1e-12
    )


def _masses_equal(left: EdgeMasses, right: EdgeMasses) -> bool:
    if len(left) != len(right):
        return False
    return all(
        left_bin == right_bin
        and math.isclose(left_mass, right_mass, rel_tol=0.0, abs_tol=1e-12)
        for (left_bin, left_mass), (right_bin, right_mass) in zip(
            left,
            right,
            strict=True,
        )
    )


def _nonnegative(value: float, *, label: str) -> float:
    if value >= 0.0:
        return float(value)
    if value >= -1e-10:
        return 0.0
    raise RefineStateError(f"{label} became negative: {value}")


def _readonly_view(value: NDArray) -> NDArray:
    view = value.view()
    view.flags.writeable = False
    return view
