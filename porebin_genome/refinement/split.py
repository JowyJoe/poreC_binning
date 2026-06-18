"""Deterministic SCG-guided HG-VAE split candidate generation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils.fixes import threadpool_limits

from porebin_genome.refinement.actions import ActionType, Proposal
from porebin_genome.refinement.profiles import EvidenceProfileState
from porebin_genome.refinement.state import AssignmentChange, RefineState


class SplitCandidateStatus(str, Enum):
    """Outcome of proposing one split without evaluating its evidence."""

    READY = "ready"
    NOT_SUSPECT = "not_suspect"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class SplitGeneratorConfig:
    """Numerical convergence settings for deterministic seeded k-means."""

    max_iterations: int = 100
    tolerance: float = 1e-4

    def __post_init__(self) -> None:
        if int(self.max_iterations) < 1:
            raise ValueError("max_iterations must be positive.")
        tolerance = float(self.tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        object.__setattr__(self, "max_iterations", int(self.max_iterations))
        object.__setattr__(self, "tolerance", tolerance)


@dataclass(frozen=True)
class SplitCandidate:
    """One auditable split proposal or an explicit reason to abstain."""

    source_bin_idx: int
    status: SplitCandidateStatus
    reason: str
    marker_id: str | None = None
    seed_contig_indices: tuple[int, ...] = ()
    child_members: tuple[tuple[int, ...], ...] = ()
    proposal: Proposal | None = None
    inertia: float | None = None
    iterations: int | None = None

    @property
    def n_children(self) -> int:
        return len(self.child_members)


def generate_split_candidates(
    *,
    state: RefineState,
    profiles: EvidenceProfileState,
    config: SplitGeneratorConfig | None = None,
) -> tuple[SplitCandidate, ...]:
    """Generate at most one candidate for each SCG-suspect bin."""
    _validate_shared_state(state, profiles)
    resolved = config or SplitGeneratorConfig()
    return tuple(
        generate_split_candidate(
            source_bin_idx=bin_idx,
            state=state,
            profiles=profiles,
            config=resolved,
        )
        for bin_idx in range(state.n_bins)
        if profiles.profile(bin_idx).scg.duplicate_burden > 0
    )


def generate_split_candidate(
    *,
    source_bin_idx: int,
    state: RefineState,
    profiles: EvidenceProfileState,
    config: SplitGeneratorConfig | None = None,
) -> SplitCandidate:
    """Propose one seeded HG-VAE partition without changing refine state."""
    _validate_shared_state(state, profiles)
    resolved = config or SplitGeneratorConfig()
    source_bin_idx = int(source_bin_idx)
    members = tuple(sorted(state.bin_members(source_bin_idx)))
    duplicate = profiles.primary_duplicate_marker(source_bin_idx)
    if duplicate is None:
        return SplitCandidate(
            source_bin_idx=source_bin_idx,
            status=SplitCandidateStatus.NOT_SUSPECT,
            reason="no_duplicated_scg",
        )

    inputs = profiles.inputs
    if inputs.embedding is None:
        return _abstain(
            source_bin_idx,
            "hgvae_embedding_unavailable",
            marker_id=duplicate.marker_id,
            seeds=duplicate.contig_indices,
        )
    missing = tuple(
        contig_idx
        for contig_idx in members
        if not bool(inputs.embedding_present[contig_idx])
    )
    if missing:
        return _abstain(
            source_bin_idx,
            "bin_contains_contig_without_hgvae_embedding",
            marker_id=duplicate.marker_id,
            seeds=duplicate.contig_indices,
        )

    n_clusters = duplicate.copy_count
    if n_clusters < 2 or n_clusters > len(members):
        return _abstain(
            source_bin_idx,
            "invalid_scg_seed_count",
            marker_id=duplicate.marker_id,
            seeds=duplicate.contig_indices,
        )

    member_rows = np.asarray(members, dtype=np.int64)
    seed_rows = np.asarray(duplicate.contig_indices, dtype=np.int64)
    member_embedding = np.asarray(
        inputs.embedding[member_rows],
        dtype=np.float64,
    )
    seed_embedding = np.asarray(
        inputs.embedding[seed_rows],
        dtype=np.float64,
    )
    if np.unique(seed_embedding, axis=0).shape[0] != n_clusters:
        return _abstain(
            source_bin_idx,
            "duplicated_scg_seeds_have_identical_hgvae_embeddings",
            marker_id=duplicate.marker_id,
            seeds=duplicate.contig_indices,
        )

    model = KMeans(
        n_clusters=n_clusters,
        init=seed_embedding,
        n_init=1,
        max_iter=resolved.max_iterations,
        tol=resolved.tolerance,
        algorithm="lloyd",
        random_state=0,
    )
    with threadpool_limits(limits=1):
        labels = np.asarray(
            model.fit_predict(member_embedding),
            dtype=np.int32,
        )
    observed_labels = tuple(sorted(int(value) for value in np.unique(labels)))
    if len(observed_labels) != n_clusters:
        return _abstain(
            source_bin_idx,
            "seeded_kmeans_produced_empty_child",
            marker_id=duplicate.marker_id,
            seeds=duplicate.contig_indices,
            inertia=float(model.inertia_),
            iterations=int(model.n_iter_),
        )

    label_by_contig = {
        contig_idx: int(labels[offset])
        for offset, contig_idx in enumerate(members)
    }
    seed_labels = tuple(
        label_by_contig[contig_idx]
        for contig_idx in duplicate.contig_indices
    )
    if len(set(seed_labels)) != n_clusters:
        return _abstain(
            source_bin_idx,
            "duplicated_scg_seeds_not_separated",
            marker_id=duplicate.marker_id,
            seeds=duplicate.contig_indices,
            inertia=float(model.inertia_),
            iterations=int(model.n_iter_),
        )

    raw_children = tuple(
        tuple(
            contig_idx
            for contig_idx in members
            if label_by_contig[contig_idx] == label
        )
        for label in observed_labels
    )
    retained_child = max(
        raw_children,
        key=lambda child: (
            _total_length(child, state),
            -min(child),
        ),
    )
    new_children = tuple(
        sorted(
            (
                child
                for child in raw_children
                if child != retained_child
            ),
            key=lambda child: min(child),
        )
    )
    child_members = (retained_child, *new_children)
    new_bin_indices = state.next_bin_indices(len(new_children))
    changes = tuple(
        AssignmentChange(
            contig_idx=contig_idx,
            old_bin_idx=source_bin_idx,
            new_bin_idx=new_bin_idx,
        )
        for child, new_bin_idx in zip(
            new_children,
            new_bin_indices,
            strict=True,
        )
        for contig_idx in child
    )
    proposal = Proposal(
        proposal_id=(
            f"split:v{state.version}:bin{source_bin_idx}:"
            f"marker:{duplicate.marker_id}"
        ),
        action_type=ActionType.SPLIT,
        changes=changes,
        state_version=state.version,
    )
    return SplitCandidate(
        source_bin_idx=source_bin_idx,
        status=SplitCandidateStatus.READY,
        reason="scg_seeded_hgvae_partition",
        marker_id=duplicate.marker_id,
        seed_contig_indices=duplicate.contig_indices,
        child_members=child_members,
        proposal=proposal,
        inertia=float(model.inertia_),
        iterations=int(model.n_iter_),
    )


def _validate_shared_state(
    state: RefineState,
    profiles: EvidenceProfileState,
) -> None:
    if profiles.refine_state is not state:
        raise ValueError(
            "Split generator and EvidenceProfileState must share one "
            "RefineState instance."
        )
    if profiles.version != state.version:
        raise ValueError(
            "Split generator requires synchronized assignment and profiles."
        )


def _total_length(
    members: tuple[int, ...],
    state: RefineState,
) -> int:
    return sum(int(state.contig_lengths[contig_idx]) for contig_idx in members)


def _abstain(
    source_bin_idx: int,
    reason: str,
    *,
    marker_id: str,
    seeds: tuple[int, ...],
    inertia: float | None = None,
    iterations: int | None = None,
) -> SplitCandidate:
    return SplitCandidate(
        source_bin_idx=source_bin_idx,
        status=SplitCandidateStatus.ABSTAIN,
        reason=reason,
        marker_id=marker_id,
        seed_contig_indices=seeds,
        inertia=inertia,
        iterations=iterations,
    )
