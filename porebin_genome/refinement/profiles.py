"""Independent HG-VAE, TNF, coverage, and SCG profiles for refinement."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from porebin_genome.coarse.hyperedge_embedding import (
    load_hyperedge_embedding_tsv,
)
from porebin_genome.evidence.tnf import compute_tnf136_features
from porebin_genome.io.coverage import read_coverage_tsv
from porebin_genome.refinement.contact_index import ContactIndex
from porebin_genome.evidence.scg.panel import (
    DEFAULT_SCG_MARKERS,
    canonicalize_marker_id,
)
from porebin_genome.refinement.state import ActionDelta, RefineState


FloatArray = NDArray[np.floating]


class EvidenceProfileError(RuntimeError):
    """Raised when feature evidence cannot satisfy the refine contract."""


@dataclass(frozen=True)
class EmbeddingProfile:
    """HG-VAE centroid and robust Euclidean distance summary."""

    observed_count: int
    centroid: FloatArray | None
    dispersion: float | None
    distance_mad: float | None

    @property
    def radius(self) -> float | None:
        """Robust membership radius: median distance plus three MADs."""
        if self.dispersion is None or self.distance_mad is None:
            return None
        return float(self.dispersion + 3.0 * self.distance_mad)


@dataclass(frozen=True)
class TnfProfile:
    """TNF centroid plus median and MAD cosine-distance summaries."""

    observed_count: int
    centroid: FloatArray | None
    distance_median: float | None
    distance_mad: float | None


@dataclass(frozen=True)
class CoverageProfile:
    """Median and MAD of log1p transformed coverage."""

    observed_count: int
    log_median: float | None
    log_mad: float | None


@dataclass(frozen=True)
class ScgProfile:
    """Fixed-panel SCG state using distinct marker-bearing contigs."""

    observed_contig_count: int
    unique_marker_count: int
    completeness_proxy: float
    duplicate_burden: int
    duplicate_marker_ids: tuple[str, ...]
    marker_contig_counts: tuple[int, ...] = field(repr=False)
    marker_orf_counts: tuple[int, ...] = field(repr=False)


@dataclass(frozen=True)
class DuplicateMarkerGroup:
    """Primary duplicated SCG and its distinct carrier contigs."""

    marker_idx: int
    marker_id: str
    contig_indices: tuple[int, ...]

    @property
    def copy_count(self) -> int:
        return len(self.contig_indices)


@dataclass(frozen=True)
class BinEvidenceProfile:
    """Four independent evidence profiles for one bin membership set."""

    bin_idx: int
    member_count: int
    total_length: int
    embedding: EmbeddingProfile
    tnf: TnfProfile
    coverage: CoverageProfile
    scg: ScgProfile


@dataclass(frozen=True)
class BinEvidenceTransition:
    """Before and after evidence for one bin affected by an action."""

    bin_idx: int
    before: BinEvidenceProfile
    after: BinEvidenceProfile
    embedding_gain: float | None
    tnf_dispersion_change: float | None
    coverage_mad_change: float | None
    scg_duplicate_change: int


@dataclass(frozen=True)
class ContigBinMetrics:
    """Direct feature compatibility of one contig with one bin profile."""

    contig_idx: int
    bin_idx: int
    embedding_distance: float | None
    tnf_distance: float | None
    coverage_distance: float | None
    conflicting_scg_markers: tuple[str, ...]


@dataclass(frozen=True)
class ActionEvidenceUpdate:
    """Local profile transition corresponding to one contact ActionDelta."""

    contact_delta: ActionDelta
    transitions: tuple[BinEvidenceTransition, ...]
    state_version: int
    _profile_token: object = field(repr=False, compare=False)
    _refine_state: RefineState = field(repr=False, compare=False)


@dataclass
class EvidenceProfileCounters:
    """Counts profile work so tests can enforce action-local recomputation."""

    bin_profiles_built: int = 0
    contigs_visited: int = 0

    def reset(self) -> None:
        self.bin_profiles_built = 0
        self.contigs_visited = 0


@dataclass(frozen=True)
class EvidenceInputs:
    """Contig-aligned independent evidence used by the replacement refine."""

    embedding: FloatArray | None
    embedding_present: NDArray[np.bool_]
    tnf: FloatArray | None
    tnf_present: NDArray[np.bool_]
    coverage_log1p: FloatArray
    scg_markers: tuple[tuple[int, ...], ...]
    scg_orf_counts: tuple[tuple[tuple[int, int], ...], ...]
    marker_ids: tuple[str, ...] = DEFAULT_SCG_MARKERS

    def __post_init__(self) -> None:
        n_contigs = self.n_contigs
        if self.marker_ids != DEFAULT_SCG_MARKERS:
            raise EvidenceProfileError(
                "EvidenceInputs must use the fixed embedded 107-marker order."
            )
        if len(self.embedding_present) != n_contigs:
            raise EvidenceProfileError("Embedding presence mask length mismatch.")
        if self.embedding is not None and self.embedding.shape[0] != n_contigs:
            raise EvidenceProfileError("Embedding row count mismatch.")
        if len(self.tnf_present) != n_contigs:
            raise EvidenceProfileError("TNF presence mask length mismatch.")
        if self.tnf is not None and self.tnf.shape[0] != n_contigs:
            raise EvidenceProfileError("TNF row count mismatch.")
        if len(self.scg_markers) != n_contigs:
            raise EvidenceProfileError("SCG marker rows do not match contigs.")
        if len(self.scg_orf_counts) != n_contigs:
            raise EvidenceProfileError("SCG ORF rows do not match contigs.")
        marker_count = len(self.marker_ids)
        for marker_indices, orf_counts in zip(
            self.scg_markers,
            self.scg_orf_counts,
            strict=True,
        ):
            if any(
                marker_idx < 0 or marker_idx >= marker_count
                for marker_idx in marker_indices
            ):
                raise EvidenceProfileError("SCG marker index is out of range.")
            if len(set(marker_indices)) != len(marker_indices):
                raise EvidenceProfileError(
                    "One contig contains duplicate canonical SCG indices."
                )
            if any(
                marker_idx < 0
                or marker_idx >= marker_count
                or count < 0
                for marker_idx, count in orf_counts
            ):
                raise EvidenceProfileError("SCG ORF count row is invalid.")

    @property
    def n_contigs(self) -> int:
        return len(self.coverage_log1p)

    @classmethod
    def from_arrays(
        cls,
        *,
        contact_index: ContactIndex,
        embedding: object | None = None,
        tnf: object | None = None,
        coverage: Sequence[float] | FloatArray | None = None,
        scg_markers_by_contig: Mapping[str, Iterable[str]] | None = None,
        scg_orf_counts_by_contig: (
            Mapping[str, Mapping[str, int]] | None
        ) = None,
    ) -> "EvidenceInputs":
        n_contigs = contact_index.n_contigs
        normalized_embedding, embedding_present = _normalized_embedding(
            embedding,
            n_contigs=n_contigs,
        )
        validated_tnf, tnf_present = _validated_tnf(
            tnf,
            n_contigs=n_contigs,
        )
        coverage_log1p = _coverage_log1p(
            coverage,
            n_contigs=n_contigs,
        )
        scg_markers, scg_orf_counts = _encode_scg(
            contact_index=contact_index,
            markers_by_contig=scg_markers_by_contig or {},
            orf_counts_by_contig=scg_orf_counts_by_contig or {},
        )
        return cls(
            embedding=normalized_embedding,
            embedding_present=embedding_present,
            tnf=validated_tnf,
            tnf_present=tnf_present,
            coverage_log1p=coverage_log1p,
            scg_markers=scg_markers,
            scg_orf_counts=scg_orf_counts,
        )


class EvidenceProfileState:
    """Versioned bin-profile cache synchronized with one RefineState."""

    def __init__(
        self,
        *,
        refine_state: RefineState,
        inputs: EvidenceInputs,
        eps: float = 1e-12,
    ) -> None:
        if inputs.n_contigs != refine_state.n_contigs:
            raise EvidenceProfileError(
                "EvidenceInputs and RefineState contig counts differ."
            )
        if not math.isfinite(float(eps)) or float(eps) <= 0.0:
            raise ValueError("eps must be a finite positive number.")
        self.inputs = inputs
        self._refine_state = refine_state
        self._eps = float(eps)
        self._version = refine_state.version
        self._profile_token = object()
        self._applied_stack: list[ActionEvidenceUpdate] = []
        self.counters = EvidenceProfileCounters()
        self._profiles = {
            bin_idx: self._build_profile(
                bin_idx,
                refine_state.bin_members(bin_idx),
            )
            for bin_idx in range(refine_state.n_bins)
        }

    @property
    def version(self) -> int:
        return self._version

    @property
    def refine_state(self) -> RefineState:
        """The assignment state that owns this profile cache."""
        return self._refine_state

    def profile(self, bin_idx: int) -> BinEvidenceProfile:
        bin_idx = int(bin_idx)
        if bin_idx < 0:
            raise IndexError(f"Bin index out of range: {bin_idx}")
        profile = self._profiles.get(bin_idx)
        if profile is not None:
            return profile
        return self._build_profile(bin_idx, ())

    def profile_for_members(
        self,
        bin_idx: int,
        members: Iterable[int],
    ) -> BinEvidenceProfile:
        """Build a local profile without changing the cached state."""
        return self._build_profile(int(bin_idx), members)

    def primary_duplicate_marker(
        self,
        bin_idx: int,
    ) -> DuplicateMarkerGroup | None:
        """Return the most repeated SCG, with canonical-ID tie breaking."""
        bin_idx = int(bin_idx)
        profile = self.profile(bin_idx)
        candidate_indices = [
            marker_idx
            for marker_idx, count in enumerate(
                profile.scg.marker_contig_counts
            )
            if count > 1
        ]
        if not candidate_indices:
            return None
        marker_idx = min(
            candidate_indices,
            key=lambda idx: (
                -profile.scg.marker_contig_counts[idx],
                self.inputs.marker_ids[idx],
            ),
        )
        carriers = tuple(
            sorted(
                contig_idx
                for contig_idx in self._refine_state.bin_members(bin_idx)
                if marker_idx in self.inputs.scg_markers[contig_idx]
            )
        )
        expected_count = profile.scg.marker_contig_counts[marker_idx]
        if len(carriers) != expected_count:
            raise EvidenceProfileError(
                "Cached SCG counts disagree with bin membership for "
                f"bin={bin_idx}, marker={self.inputs.marker_ids[marker_idx]}."
            )
        return DuplicateMarkerGroup(
            marker_idx=marker_idx,
            marker_id=self.inputs.marker_ids[marker_idx],
            contig_indices=carriers,
        )

    def contig_bin_metrics(
        self,
        contig_idx: int,
        bin_idx: int,
        *,
        exclude_contig: bool = False,
    ) -> ContigBinMetrics:
        """Return direct, unthresholded compatibility distances."""
        contig_idx = int(contig_idx)
        if contig_idx < 0 or contig_idx >= self.inputs.n_contigs:
            raise IndexError(f"Contig index out of range: {contig_idx}")
        bin_idx = int(bin_idx)
        profile = self.profile(bin_idx)
        current_members = self._refine_state.bin_members(bin_idx)
        if exclude_contig and contig_idx in current_members:
            profile = self.profile_for_members(
                bin_idx,
                (
                    member_idx
                    for member_idx in current_members
                    if member_idx != contig_idx
                ),
            )

        embedding_distance = None
        if (
            profile.embedding.centroid is not None
            and bool(self.inputs.embedding_present[contig_idx])
            and self.inputs.embedding is not None
        ):
            embedding_distance = float(
                np.linalg.norm(
                    self.inputs.embedding[contig_idx]
                    - profile.embedding.centroid
                )
            )

        tnf_distance = None
        if (
            profile.tnf.centroid is not None
            and bool(self.inputs.tnf_present[contig_idx])
            and self.inputs.tnf is not None
        ):
            tnf_distance = _cosine_distance(
                self.inputs.tnf[contig_idx],
                profile.tnf.centroid,
            )

        coverage_distance = None
        coverage_value = float(self.inputs.coverage_log1p[contig_idx])
        if (
            profile.coverage.log_median is not None
            and math.isfinite(coverage_value)
        ):
            coverage_distance = abs(
                coverage_value - profile.coverage.log_median
            )

        present_markers = {
            marker_idx
            for marker_idx, count in enumerate(
                profile.scg.marker_contig_counts
            )
            if count > 0
        }
        conflicting = tuple(
            self.inputs.marker_ids[marker_idx]
            for marker_idx in self.inputs.scg_markers[contig_idx]
            if marker_idx in present_markers
        )
        return ContigBinMetrics(
            contig_idx=contig_idx,
            bin_idx=bin_idx,
            embedding_distance=embedding_distance,
            tnf_distance=tnf_distance,
            coverage_distance=coverage_distance,
            conflicting_scg_markers=conflicting,
        )

    def evaluate(
        self,
        delta: ActionDelta,
    ) -> ActionEvidenceUpdate:
        """Build before/after profiles only for bins touched by an action."""
        if self._version != self._refine_state.version:
            raise EvidenceProfileError(
                "EvidenceProfileState is not synchronized with RefineState."
            )
        if delta.state_version != self._version:
            raise EvidenceProfileError(
                f"Stale ActionDelta for evidence profiles: "
                f"evaluated_at={delta.state_version}, current={self._version}"
            )

        before_members: dict[int, set[int]] = {
            bin_idx: (
                set(self._refine_state.bin_members(bin_idx))
                if bin_idx < self._refine_state.n_bins
                else set()
            )
            for bin_idx in delta.affected_bins
        }
        after_members = {
            bin_idx: set(members)
            for bin_idx, members in before_members.items()
        }
        for change in delta.changes:
            if change.old_bin_idx >= 0:
                after_members.setdefault(change.old_bin_idx, set()).remove(
                    change.contig_idx
                )
            if change.new_bin_idx >= 0:
                after_members.setdefault(change.new_bin_idx, set()).add(
                    change.contig_idx
                )

        transitions: list[BinEvidenceTransition] = []
        for bin_idx in delta.affected_bins:
            before = self._profiles.get(bin_idx)
            if before is None:
                before = self._build_profile(
                    bin_idx,
                    before_members.get(bin_idx, ()),
                )
            after = self._build_profile(
                bin_idx,
                after_members.get(bin_idx, ()),
            )
            transitions.append(
                BinEvidenceTransition(
                    bin_idx=bin_idx,
                    before=before,
                    after=after,
                    embedding_gain=_relative_improvement(
                        before.embedding.dispersion,
                        after.embedding.dispersion,
                        eps=self._eps,
                    ),
                    tnf_dispersion_change=_difference(
                        after.tnf.distance_median,
                        before.tnf.distance_median,
                    ),
                    coverage_mad_change=_difference(
                        after.coverage.log_mad,
                        before.coverage.log_mad,
                    ),
                    scg_duplicate_change=(
                        after.scg.duplicate_burden
                        - before.scg.duplicate_burden
                    ),
                )
            )
        return ActionEvidenceUpdate(
            contact_delta=delta,
            transitions=tuple(transitions),
            state_version=self._version,
            _profile_token=self._profile_token,
            _refine_state=self._refine_state,
        )

    def apply(self, update: ActionEvidenceUpdate) -> None:
        """Commit profiles after the corresponding contact delta is applied."""
        self._validate_update(update)
        if self._refine_state.version != update.state_version + 1:
            raise EvidenceProfileError(
                "Apply the contact ActionDelta before its evidence update."
            )
        for transition in update.transitions:
            self._profiles[transition.bin_idx] = transition.after
        self._version += 1
        self._applied_stack.append(update)

    def rollback(self, update: ActionEvidenceUpdate) -> None:
        """Restore profiles before rolling back the corresponding contact delta."""
        if update._profile_token is not self._profile_token:
            raise EvidenceProfileError(
                "ActionEvidenceUpdate belongs to a different profile state."
            )
        if not self._applied_stack or self._applied_stack[-1] is not update:
            raise EvidenceProfileError(
                "Only the most recent evidence update can be rolled back."
            )
        if self._refine_state.version != update.state_version + 1:
            raise EvidenceProfileError(
                "Rollback the evidence update before its contact ActionDelta."
            )
        for transition in update.transitions:
            if transition.bin_idx >= self._refine_state.n_bins:
                self._profiles.pop(transition.bin_idx, None)
            else:
                self._profiles[transition.bin_idx] = transition.before
        self._version = update.state_version
        self._applied_stack.pop()

    def _validate_update(self, update: ActionEvidenceUpdate) -> None:
        if update._profile_token is not self._profile_token:
            raise EvidenceProfileError(
                "ActionEvidenceUpdate belongs to a different profile state."
            )
        if update._refine_state is not self._refine_state:
            raise EvidenceProfileError(
                "ActionEvidenceUpdate refers to a different RefineState."
            )
        if update.state_version != self._version:
            raise EvidenceProfileError(
                f"Stale evidence update: evaluated_at={update.state_version}, "
                f"current={self._version}"
            )

    def _build_profile(
        self,
        bin_idx: int,
        members: Iterable[int],
    ) -> BinEvidenceProfile:
        member_indices = tuple(sorted({int(member) for member in members}))
        if any(
            member < 0 or member >= self.inputs.n_contigs
            for member in member_indices
        ):
            raise EvidenceProfileError("Bin profile contains invalid contig ID.")
        self.counters.bin_profiles_built += 1
        self.counters.contigs_visited += len(member_indices)
        contig_lengths = self._refine_state.contig_lengths
        total_length = sum(
            int(contig_lengths[member])
            for member in member_indices
        )
        return BinEvidenceProfile(
            bin_idx=int(bin_idx),
            member_count=len(member_indices),
            total_length=int(total_length),
            embedding=_embedding_profile(self.inputs, member_indices),
            tnf=_tnf_profile(self.inputs, member_indices),
            coverage=_coverage_profile(self.inputs, member_indices),
            scg=_scg_profile(self.inputs, member_indices),
        )


def load_evidence_inputs(
    *,
    contact_index: ContactIndex,
    embedding_tsv: Path | None,
    contigs_fasta: Path,
    coverage_tsv: Path,
    scg_index_json: Path | None,
) -> EvidenceInputs:
    """Load public evidence files and align every row to ContactIndex IDs."""
    embedding = None
    if embedding_tsv is not None:
        embedding_names, embedding_matrix, _support = (
            load_hyperedge_embedding_tsv(embedding_tsv)
        )
        embedding = _align_matrix(
            contact_index,
            embedding_names,
            embedding_matrix,
        )

    tnf = compute_tnf136_features(
        contigs_fasta,
        dict(contact_index.contig_name_to_idx),
    )
    coverage_by_name = read_coverage_tsv(coverage_tsv)
    coverage = np.full(contact_index.n_contigs, np.nan, dtype=np.float64)
    for name, value in coverage_by_name.items():
        contig_idx = contact_index.contig_name_to_idx.get(name)
        if contig_idx is not None:
            coverage[int(contig_idx)] = float(value)

    scg_markers: dict[str, tuple[str, ...]] = {}
    scg_orf_counts: dict[str, dict[str, int]] = {}
    if scg_index_json is not None and not scg_index_json.exists():
        raise FileNotFoundError(f"SCG contig index not found: {scg_index_json}")
    if scg_index_json is not None:
        payload = json.loads(scg_index_json.read_text(encoding="utf-8"))
        profiles = payload.get("profiles", {})
        if not isinstance(profiles, dict):
            raise EvidenceProfileError(
                f"Invalid SCG contig index: {scg_index_json}"
            )
        for contig_name, raw_profile in profiles.items():
            if not isinstance(raw_profile, dict):
                continue
            scg_markers[str(contig_name)] = tuple(
                str(marker)
                for marker in raw_profile.get("marker_ids", ())
            )
            scg_orf_counts[str(contig_name)] = {
                str(marker): int(count)
                for marker, count in dict(
                    raw_profile.get("marker_orf_counts", {})
                ).items()
            }
    return EvidenceInputs.from_arrays(
        contact_index=contact_index,
        embedding=embedding,
        tnf=tnf,
        coverage=coverage,
        scg_markers_by_contig=scg_markers,
        scg_orf_counts_by_contig=scg_orf_counts,
    )


def _normalized_embedding(
    embedding: object | None,
    *,
    n_contigs: int,
) -> tuple[FloatArray | None, NDArray[np.bool_]]:
    if embedding is None:
        present = np.zeros(n_contigs, dtype=np.bool_)
        present.flags.writeable = False
        return None, present
    matrix = np.asarray(embedding, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != n_contigs:
        raise EvidenceProfileError(
            "Embedding matrix must be two-dimensional with one row per contig."
        )
    finite = np.all(np.isfinite(matrix), axis=1)
    norms = np.linalg.norm(np.where(finite[:, None], matrix, 0.0), axis=1)
    present = finite & (norms > 0.0)
    normalized = np.full(matrix.shape, np.nan, dtype=np.float64)
    normalized[present] = matrix[present] / norms[present, None]
    return _readonly(normalized), _readonly(present)


def _validated_tnf(
    tnf: object | None,
    *,
    n_contigs: int,
) -> tuple[FloatArray | None, NDArray[np.bool_]]:
    if tnf is None:
        present = np.zeros(n_contigs, dtype=np.bool_)
        present.flags.writeable = False
        return None, present
    matrix = np.asarray(tnf, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != n_contigs:
        raise EvidenceProfileError(
            "TNF matrix must be two-dimensional with one row per contig."
        )
    finite = np.all(np.isfinite(matrix), axis=1)
    norms = np.linalg.norm(np.where(finite[:, None], matrix, 0.0), axis=1)
    present = finite & (norms > 0.0)
    validated = matrix.copy()
    validated[~present] = np.nan
    return _readonly(validated), _readonly(present)


def _coverage_log1p(
    coverage: Sequence[float] | FloatArray | None,
    *,
    n_contigs: int,
) -> FloatArray:
    if coverage is None:
        return _readonly(np.full(n_contigs, np.nan, dtype=np.float64))
    values = np.asarray(coverage, dtype=np.float64)
    if values.ndim != 1 or len(values) != n_contigs:
        raise EvidenceProfileError(
            "Coverage must contain one value per contig."
        )
    finite = np.isfinite(values)
    if bool(np.any(values[finite] < 0.0)):
        raise EvidenceProfileError("Coverage values must be non-negative.")
    out = np.full(n_contigs, np.nan, dtype=np.float64)
    out[finite] = np.log1p(values[finite])
    return _readonly(out)


def _encode_scg(
    *,
    contact_index: ContactIndex,
    markers_by_contig: Mapping[str, Iterable[str]],
    orf_counts_by_contig: Mapping[str, Mapping[str, int]],
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[tuple[tuple[int, int], ...], ...],
]:
    marker_to_idx = {
        marker_id: idx
        for idx, marker_id in enumerate(DEFAULT_SCG_MARKERS)
    }
    markers_out: list[tuple[int, ...]] = []
    orfs_out: list[tuple[tuple[int, int], ...]] = []
    for contig_name in contact_index.contig_names:
        marker_indices: set[int] = set()
        for raw_marker in markers_by_contig.get(contig_name, ()):
            marker = canonicalize_marker_id(str(raw_marker))
            if marker not in marker_to_idx:
                raise EvidenceProfileError(
                    f"Unknown SCG marker for {contig_name}: {raw_marker}"
                )
            marker_indices.add(marker_to_idx[marker])

        canonical_orf_counts: dict[int, int] = {}
        for raw_marker, raw_count in orf_counts_by_contig.get(
            contig_name,
            {},
        ).items():
            marker = canonicalize_marker_id(str(raw_marker))
            if marker not in marker_to_idx:
                raise EvidenceProfileError(
                    f"Unknown SCG ORF marker for {contig_name}: {raw_marker}"
                )
            count = int(raw_count)
            if count < 0:
                raise EvidenceProfileError(
                    f"Negative SCG ORF count for {contig_name}: {raw_marker}"
                )
            marker_idx = marker_to_idx[marker]
            canonical_orf_counts[marker_idx] = (
                canonical_orf_counts.get(marker_idx, 0) + count
            )
            if count > 0:
                marker_indices.add(marker_idx)
        markers_out.append(tuple(sorted(marker_indices)))
        orfs_out.append(tuple(sorted(canonical_orf_counts.items())))
    return tuple(markers_out), tuple(orfs_out)


def _embedding_profile(
    inputs: EvidenceInputs,
    members: tuple[int, ...],
) -> EmbeddingProfile:
    if inputs.embedding is None:
        return EmbeddingProfile(0, None, None, None)
    observed = [
        member
        for member in members
        if bool(inputs.embedding_present[member])
    ]
    if not observed:
        return EmbeddingProfile(0, None, None, None)
    matrix = inputs.embedding[np.asarray(observed, dtype=np.int64)]
    centroid = np.mean(matrix, axis=0, dtype=np.float64)
    distances = np.linalg.norm(matrix - centroid, axis=1)
    distance_median = float(np.median(distances))
    return EmbeddingProfile(
        observed_count=len(observed),
        centroid=_readonly(np.asarray(centroid, dtype=np.float64)),
        dispersion=distance_median,
        distance_mad=float(np.median(np.abs(distances - distance_median))),
    )


def _tnf_profile(
    inputs: EvidenceInputs,
    members: tuple[int, ...],
) -> TnfProfile:
    if inputs.tnf is None:
        return TnfProfile(0, None, None, None)
    observed = [
        member for member in members if bool(inputs.tnf_present[member])
    ]
    if not observed:
        return TnfProfile(0, None, None, None)
    matrix = inputs.tnf[np.asarray(observed, dtype=np.int64)]
    centroid = np.mean(matrix, axis=0, dtype=np.float64)
    distances = np.asarray(
        [_cosine_distance(row, centroid) for row in matrix],
        dtype=np.float64,
    )
    median = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median)))
    return TnfProfile(
        observed_count=len(observed),
        centroid=_readonly(np.asarray(centroid, dtype=np.float64)),
        distance_median=median,
        distance_mad=mad,
    )


def _coverage_profile(
    inputs: EvidenceInputs,
    members: tuple[int, ...],
) -> CoverageProfile:
    values = np.asarray(
        [
            float(inputs.coverage_log1p[member])
            for member in members
            if math.isfinite(float(inputs.coverage_log1p[member]))
        ],
        dtype=np.float64,
    )
    if values.size == 0:
        return CoverageProfile(0, None, None)
    center = float(np.median(values))
    return CoverageProfile(
        observed_count=int(values.size),
        log_median=center,
        log_mad=float(np.median(np.abs(values - center))),
    )


def _scg_profile(
    inputs: EvidenceInputs,
    members: tuple[int, ...],
) -> ScgProfile:
    marker_contig_counts = np.zeros(
        len(inputs.marker_ids),
        dtype=np.int32,
    )
    marker_orf_counts = np.zeros(
        len(inputs.marker_ids),
        dtype=np.int32,
    )
    observed_contigs = 0
    for member in members:
        marker_indices = inputs.scg_markers[member]
        if marker_indices:
            observed_contigs += 1
        for marker_idx in marker_indices:
            marker_contig_counts[marker_idx] += 1
        for marker_idx, count in inputs.scg_orf_counts[member]:
            marker_orf_counts[marker_idx] += int(count)
    unique_marker_count = int(np.count_nonzero(marker_contig_counts))
    duplicate_indices = np.flatnonzero(marker_contig_counts > 1)
    duplicate_burden = int(
        np.maximum(marker_contig_counts - 1, 0).sum()
    )
    return ScgProfile(
        observed_contig_count=observed_contigs,
        unique_marker_count=unique_marker_count,
        completeness_proxy=float(
            unique_marker_count / max(len(inputs.marker_ids), 1)
        ),
        duplicate_burden=duplicate_burden,
        duplicate_marker_ids=tuple(
            inputs.marker_ids[int(marker_idx)]
            for marker_idx in duplicate_indices
        ),
        marker_contig_counts=tuple(
            int(value) for value in marker_contig_counts
        ),
        marker_orf_counts=tuple(
            int(value) for value in marker_orf_counts
        ),
    )


def _cosine_distance(left: FloatArray, right: FloatArray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return 0.0
    similarity = float(np.dot(left, right) / denominator)
    return float(1.0 - min(1.0, max(-1.0, similarity)))


def _relative_improvement(
    before: float | None,
    after: float | None,
    *,
    eps: float,
) -> float | None:
    if before is None or after is None:
        return None
    return float((before - after) / (before + eps))


def _difference(
    after: float | None,
    before: float | None,
) -> float | None:
    if before is None or after is None:
        return None
    return float(after - before)


def _align_matrix(
    contact_index: ContactIndex,
    row_names: Sequence[str],
    matrix: object,
) -> FloatArray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != len(row_names):
        raise EvidenceProfileError(
            "Named matrix rows do not match the supplied row names."
        )
    if len(set(str(name) for name in row_names)) != len(row_names):
        raise EvidenceProfileError("Named matrix contains duplicate contigs.")
    aligned = np.full(
        (contact_index.n_contigs, values.shape[1]),
        np.nan,
        dtype=np.float64,
    )
    for row_idx, name in enumerate(row_names):
        contig_idx = contact_index.contig_name_to_idx.get(str(name))
        if contig_idx is not None:
            aligned[int(contig_idx)] = values[row_idx]
    return aligned


def _readonly(value: NDArray) -> NDArray:
    out = np.asarray(value)
    out.flags.writeable = False
    return out
