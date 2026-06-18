"""Compact, hypergraph-native contact index for the rewritten refine engine."""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from porebin_genome.coarse.hyperedge_weight import (
    DEFAULT_HYPERGRAPH_WEIGHT_ETA,
    effective_order,
    hypergraph_native_weight,
)
from porebin_genome.evidence.canonical import CanonicalContact, iter_canonical_contacts


IntArray = NDArray[np.integer]
FloatArray = NDArray[np.floating]


class ContactIndexError(RuntimeError):
    """Raised when contact evidence cannot form a valid refinement index."""


@dataclass
class ContactIndexCounters:
    """Observable scan counts used to guard the refinement complexity contract."""

    full_edge_scans: int = 0
    full_edges_visited: int = 0
    local_queries: int = 0
    local_edges_visited: int = 0

    def reset(self) -> None:
        self.full_edge_scans = 0
        self.full_edges_visited = 0
        self.local_queries = 0
        self.local_edges_visited = 0


@dataclass(frozen=True)
class BinContactSupport:
    """Raw hypergraph support and independent edge count for one target bin."""

    bin_idx: int
    support: float
    edge_count: int


@dataclass(frozen=True)
class ContactCoherenceCache:
    """Initial edge-to-bin masses and bin-level coherence components."""

    numerator: FloatArray
    denominator: FloatArray
    coherence: FloatArray
    edge_bin_offsets: IntArray
    edge_bin_ids: IntArray
    edge_bin_mass: FloatArray
    eps: float

    @property
    def n_bins(self) -> int:
        return int(self.coherence.shape[0])

    def edge_mass_slice(self, edge_idx: int) -> slice:
        _require_index(edge_idx, len(self.edge_bin_offsets) - 1, "edge")
        return slice(
            int(self.edge_bin_offsets[edge_idx]),
            int(self.edge_bin_offsets[edge_idx + 1]),
        )

    def edge_masses(self, edge_idx: int) -> dict[int, float]:
        member_slice = self.edge_mass_slice(edge_idx)
        return {
            int(bin_idx): float(mass)
            for bin_idx, mass in zip(
                self.edge_bin_ids[member_slice],
                self.edge_bin_mass[member_slice],
                strict=True,
            )
        }


@dataclass(frozen=True)
class ContactIndex:
    """Immutable CSR representation of Pore-C hyperedges and reverse incidence."""

    contig_names: tuple[str, ...]
    contig_name_to_idx: Mapping[str, int]
    edge_ids: IntArray
    edge_offsets: IntArray
    edge_members: IntArray
    edge_alpha: FloatArray
    edge_reliability: FloatArray
    edge_order: IntArray
    edge_effective_order: FloatArray
    contig_edge_offsets: IntArray
    contig_edge_ids: IntArray
    contig_edge_positions: IntArray
    eta: float
    counters: ContactIndexCounters = field(
        default_factory=ContactIndexCounters,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if len(self.contig_name_to_idx) != self.n_contigs:
            raise ContactIndexError("Contig name index is incomplete.")
        if any(
            int(self.contig_name_to_idx.get(name, -1)) != idx
            for idx, name in enumerate(self.contig_names)
        ):
            raise ContactIndexError("Contig names and integer IDs are inconsistent.")
        if any(
            len(values) != self.n_edges
            for values in (
                self.edge_reliability,
                self.edge_order,
                self.edge_effective_order,
            )
        ):
            raise ContactIndexError("Per-edge arrays must all have length n_edges.")
        if len(self.edge_offsets) != self.n_edges + 1:
            raise ContactIndexError("edge_offsets length must equal n_edges + 1.")
        if int(self.edge_offsets[0]) != 0:
            raise ContactIndexError("edge_offsets must start at zero.")
        if len(self.edge_members) != len(self.edge_alpha):
            raise ContactIndexError("edge_members and edge_alpha lengths differ.")
        if int(self.edge_offsets[-1]) != len(self.edge_members):
            raise ContactIndexError("edge_offsets does not terminate at incidence count.")
        if len(self.contig_edge_offsets) != self.n_contigs + 1:
            raise ContactIndexError(
                "contig_edge_offsets length must equal n_contigs + 1."
            )
        if int(self.contig_edge_offsets[0]) != 0:
            raise ContactIndexError("contig_edge_offsets must start at zero.")
        if len(self.contig_edge_ids) != len(self.edge_members):
            raise ContactIndexError("Reverse incidence is incomplete.")
        if len(self.contig_edge_positions) != len(self.contig_edge_ids):
            raise ContactIndexError("Reverse incidence positions are incomplete.")
        if int(self.contig_edge_offsets[-1]) != len(self.contig_edge_ids):
            raise ContactIndexError(
                "contig_edge_offsets does not terminate at reverse incidence count."
            )

    @property
    def n_contigs(self) -> int:
        return len(self.contig_names)

    @property
    def n_edges(self) -> int:
        return int(self.edge_ids.shape[0])

    @property
    def n_incidences(self) -> int:
        return int(self.edge_members.shape[0])

    def contig_index(self, contig_name: str) -> int:
        try:
            return int(self.contig_name_to_idx[str(contig_name)])
        except KeyError as exc:
            raise KeyError(f"Unknown contig: {contig_name}") from exc

    def edge_member_slice(self, edge_idx: int) -> slice:
        _require_index(edge_idx, self.n_edges, "edge")
        return slice(
            int(self.edge_offsets[edge_idx]),
            int(self.edge_offsets[edge_idx + 1]),
        )

    def edge_members_view(self, edge_idx: int) -> tuple[IntArray, FloatArray]:
        member_slice = self.edge_member_slice(edge_idx)
        return self.edge_members[member_slice], self.edge_alpha[member_slice]

    def incident_edges(self, contig_idx: int) -> IntArray:
        _require_index(contig_idx, self.n_contigs, "contig")
        start = int(self.contig_edge_offsets[contig_idx])
        end = int(self.contig_edge_offsets[contig_idx + 1])
        return self.contig_edge_ids[start:end]

    def scan_edge_indices(self) -> Iterator[int]:
        """Yield every edge once while recording one intentional full scan."""
        self.counters.full_edge_scans += 1
        for edge_idx in range(self.n_edges):
            self.counters.full_edges_visited += 1
            yield edge_idx

    def edge_bin_masses(
        self,
        edge_idx: int,
        assignment: Sequence[int] | IntArray,
    ) -> dict[int, float]:
        labels = self._validate_assignment(assignment)
        return self._edge_bin_masses_unchecked(edge_idx, labels)

    def _edge_bin_masses_unchecked(
        self,
        edge_idx: int,
        labels: IntArray,
    ) -> dict[int, float]:
        masses: dict[int, float] = {}
        member_slice = self.edge_member_slice(edge_idx)
        for member_idx, alpha in zip(
            self.edge_members[member_slice],
            self.edge_alpha[member_slice],
            strict=True,
        ):
            bin_idx = int(labels[int(member_idx)])
            if bin_idx < 0:
                continue
            masses[bin_idx] = float(masses.get(bin_idx, 0.0) + float(alpha))
        return masses

    def build_coherence_cache(
        self,
        assignment: Sequence[int] | IntArray,
        *,
        n_bins: int | None = None,
        eps: float = 1e-12,
    ) -> ContactCoherenceCache:
        """Build exact initial m[e,b], N[b], D[b], and C[b] in one edge scan."""
        labels = self._validate_assignment(assignment)
        if eps <= 0.0 or not math.isfinite(float(eps)):
            raise ValueError("eps must be a finite positive number.")

        inferred_bins = (
            int(labels[labels >= 0].max()) + 1
            if bool(np.any(labels >= 0))
            else 0
        )
        resolved_n_bins = inferred_bins if n_bins is None else int(n_bins)
        if resolved_n_bins < inferred_bins or resolved_n_bins < 0:
            raise ValueError(
                f"n_bins={resolved_n_bins} cannot represent assignment labels "
                f"requiring {inferred_bins} bins."
            )

        numerator = np.zeros(resolved_n_bins, dtype=np.float64)
        denominator = np.zeros(resolved_n_bins, dtype=np.float64)
        mass_offsets = array("Q", [0])
        mass_bins = array("I")
        mass_values = array("d")

        self.counters.full_edge_scans += 1
        self.counters.full_edges_visited += self.n_edges
        for edge_idx in range(self.n_edges):
            masses = self._edge_bin_masses_unchecked(edge_idx, labels)
            reliability = float(self.edge_reliability[edge_idx])
            for bin_idx in sorted(masses):
                mass = float(masses[bin_idx])
                mass_bins.append(bin_idx)
                mass_values.append(mass)
                if reliability > 0.0:
                    denominator[bin_idx] += reliability * mass
                    numerator[bin_idx] += reliability * mass * mass
            mass_offsets.append(len(mass_bins))

        coherence = numerator / (denominator + float(eps))
        edge_bin_offsets = _readonly_array(mass_offsets, np.uint64)
        edge_bin_ids = _readonly_array(mass_bins, np.uint32)
        edge_bin_mass = _readonly_array(mass_values, np.float64)
        _make_readonly(numerator, denominator, coherence)
        return ContactCoherenceCache(
            numerator=numerator,
            denominator=denominator,
            coherence=coherence,
            edge_bin_offsets=edge_bin_offsets,
            edge_bin_ids=edge_bin_ids,
            edge_bin_mass=edge_bin_mass,
            eps=float(eps),
        )

    def contig_bin_support(
        self,
        contig_idx: int,
        assignment: Sequence[int] | IntArray,
    ) -> dict[int, BinContactSupport]:
        """Compute S(i,b) only from hyperedges incident to contig i."""
        labels = self._validate_assignment(assignment)
        _require_index(contig_idx, self.n_contigs, "contig")
        start = int(self.contig_edge_offsets[contig_idx])
        end = int(self.contig_edge_offsets[contig_idx + 1])
        incident_edge_ids = self.contig_edge_ids[start:end]
        incident_positions = self.contig_edge_positions[start:end]

        self.counters.local_queries += 1
        self.counters.local_edges_visited += len(incident_edge_ids)
        support_by_bin: dict[int, float] = {}
        edge_count_by_bin: dict[int, int] = {}

        for edge_idx_raw, self_position_raw in zip(
            incident_edge_ids,
            incident_positions,
            strict=True,
        ):
            edge_idx = int(edge_idx_raw)
            reliability = float(self.edge_reliability[edge_idx])
            if reliability <= 0.0:
                continue
            alpha_self = float(self.edge_alpha[int(self_position_raw)])
            if alpha_self <= 0.0:
                continue

            other_mass: dict[int, float] = {}
            member_slice = self.edge_member_slice(edge_idx)
            for position in range(member_slice.start, member_slice.stop):
                member_idx = int(self.edge_members[position])
                if member_idx == contig_idx:
                    continue
                bin_idx = int(labels[member_idx])
                if bin_idx < 0:
                    continue
                other_mass[bin_idx] = float(
                    other_mass.get(bin_idx, 0.0) + float(self.edge_alpha[position])
                )

            for bin_idx, mass in other_mass.items():
                if mass <= 0.0:
                    continue
                support_by_bin[bin_idx] = float(
                    support_by_bin.get(bin_idx, 0.0)
                    + reliability * alpha_self * mass
                )
                edge_count_by_bin[bin_idx] = edge_count_by_bin.get(bin_idx, 0) + 1

        return {
            bin_idx: BinContactSupport(
                bin_idx=bin_idx,
                support=float(support_by_bin[bin_idx]),
                edge_count=int(edge_count_by_bin[bin_idx]),
            )
            for bin_idx in sorted(support_by_bin)
        }

    def affected_edges(self, contig_indices: Iterable[int]) -> IntArray:
        """Return sorted unique edge indices touching any changed contig."""
        edge_ids: set[int] = set()
        query_count = 0
        visited = 0
        for contig_idx_raw in contig_indices:
            contig_idx = int(contig_idx_raw)
            incident = self.incident_edges(contig_idx)
            query_count += 1
            visited += len(incident)
            edge_ids.update(int(edge_idx) for edge_idx in incident)
        self.counters.local_queries += query_count
        self.counters.local_edges_visited += visited
        out = np.fromiter(sorted(edge_ids), dtype=np.uint64, count=len(edge_ids))
        out.flags.writeable = False
        return out

    def _validate_assignment(
        self,
        assignment: Sequence[int] | IntArray,
    ) -> IntArray:
        labels = np.asarray(assignment)
        if labels.ndim != 1 or len(labels) != self.n_contigs:
            raise ValueError(
                "assignment must be one-dimensional with one label per contig: "
                f"expected={self.n_contigs}, observed_shape={labels.shape}"
            )
        if labels.dtype.kind not in {"i", "u"}:
            raise TypeError("assignment labels must use an integer dtype.")
        if labels.dtype.kind == "u":
            return labels
        if bool(np.any(labels < -1)):
            raise ValueError("assignment labels must be -1 (unassigned) or non-negative.")
        return labels


def build_contact_index(
    contacts_parquet: Path,
    *,
    contig_names: Iterable[str] | None = None,
    min_k: int = 2,
    eta: float = DEFAULT_HYPERGRAPH_WEIGHT_ETA,
    parquet_batch_size: int = 200_000,
) -> ContactIndex:
    """Stream contacts.parquet into the second-generation contact index."""
    contacts = iter_canonical_contacts(
        contacts_parquet,
        parquet_batch_size=int(parquet_batch_size),
        require_contig_weights=True,
    )
    return build_contact_index_from_contacts(
        contacts,
        contig_names=contig_names,
        min_k=min_k,
        eta=eta,
    )


def build_contact_index_from_contacts(
    contacts: Iterable[CanonicalContact],
    *,
    contig_names: Iterable[str] | None = None,
    min_k: int = 2,
    eta: float = DEFAULT_HYPERGRAPH_WEIGHT_ETA,
) -> ContactIndex:
    """Build a compact index from canonical contacts without legacy objects."""
    if int(min_k) < 2:
        raise ValueError("min_k must be at least 2.")
    if float(eta) < 0.0 or not math.isfinite(float(eta)):
        raise ValueError("eta must be a finite non-negative number.")

    names: list[str] = []
    name_to_idx: dict[str, int] = {}
    reverse_edge_buckets: list[array] = []
    reverse_position_buckets: list[array] = []
    fixed_contig_set = contig_names is not None
    if contig_names is not None:
        for raw_name in contig_names:
            name = str(raw_name)
            if not name:
                raise ContactIndexError("Contig names must be non-empty.")
            if name in name_to_idx:
                raise ContactIndexError(f"Duplicate contig name: {name}")
            name_to_idx[name] = len(names)
            names.append(name)
            reverse_edge_buckets.append(array("Q"))
            reverse_position_buckets.append(array("Q"))

    edge_ids_buffer = array("q")
    edge_offsets_buffer = array("Q", [0])
    edge_members_buffer = array("I")
    edge_alpha_buffer = array("d")
    edge_reliability_buffer = array("d")
    edge_order_buffer = array("I")
    edge_effective_order_buffer = array("d")

    for source_idx, contact in enumerate(contacts):
        if int(contact.k_valid) < int(min_k):
            continue
        if contact.contig_weights is None:
            raise ContactIndexError(
                "ContactIndex requires canonical per-contig alpha weights."
            )
        if len(contact.contigs) != len(contact.contig_weights):
            raise ContactIndexError(
                f"Contact {contact.contact_id!r} has mismatched members and alpha."
            )
        if len(set(contact.contigs)) != len(contact.contigs):
            raise ContactIndexError(
                f"Contact {contact.contact_id!r} contains duplicate contigs."
            )

        alpha_values = tuple(float(value) for value in contact.contig_weights)
        if not alpha_values or any(
            (not math.isfinite(value)) or value <= 0.0 for value in alpha_values
        ):
            raise ContactIndexError(
                f"Contact {contact.contact_id!r} has invalid alpha values."
            )
        alpha_total = float(sum(alpha_values))
        if not math.isclose(alpha_total, 1.0, rel_tol=1e-9, abs_tol=1e-7):
            raise ContactIndexError(
                f"Contact {contact.contact_id!r} alpha must sum to one; "
                f"observed={alpha_total:.12g}"
            )
        if not math.isfinite(float(contact.weight)):
            raise ContactIndexError(
                f"Contact {contact.contact_id!r} has a non-finite read weight."
            )

        edge_idx = len(edge_ids_buffer)
        for member_name_raw, alpha in zip(
            contact.contigs,
            alpha_values,
            strict=True,
        ):
            member_name = str(member_name_raw)
            member_idx = name_to_idx.get(member_name)
            if member_idx is None:
                if fixed_contig_set:
                    raise ContactIndexError(
                        f"Contact {contact.contact_id!r} references contig absent "
                        f"from the FASTA contig order: {member_name}"
                    )
                member_idx = len(names)
                name_to_idx[member_name] = member_idx
                names.append(member_name)
                reverse_edge_buckets.append(array("Q"))
                reverse_position_buckets.append(array("Q"))

            incidence_position = len(edge_members_buffer)
            edge_members_buffer.append(member_idx)
            edge_alpha_buffer.append(alpha)
            reverse_edge_buckets[member_idx].append(edge_idx)
            reverse_position_buckets[member_idx].append(incidence_position)

        k_eff = float(effective_order(alpha_values))
        reliability = float(
            hypergraph_native_weight(
                read_weight=float(contact.weight),
                alpha_values=alpha_values,
                eta=float(eta),
            )
        )
        edge_ids_buffer.append(
            int(contact.contact_id) if contact.contact_id is not None else source_idx
        )
        edge_offsets_buffer.append(len(edge_members_buffer))
        edge_reliability_buffer.append(reliability)
        edge_order_buffer.append(int(contact.k_valid))
        edge_effective_order_buffer.append(k_eff)

    contig_edge_offsets_buffer = array("Q", [0])
    contig_edge_ids_buffer = array("Q")
    contig_edge_positions_buffer = array("Q")
    for edge_bucket, position_bucket in zip(
        reverse_edge_buckets,
        reverse_position_buckets,
        strict=True,
    ):
        contig_edge_ids_buffer.extend(edge_bucket)
        contig_edge_positions_buffer.extend(position_bucket)
        contig_edge_offsets_buffer.append(len(contig_edge_ids_buffer))

    index = ContactIndex(
        contig_names=tuple(names),
        contig_name_to_idx=MappingProxyType(dict(name_to_idx)),
        edge_ids=_readonly_array(edge_ids_buffer, np.int64),
        edge_offsets=_readonly_array(edge_offsets_buffer, np.uint64),
        edge_members=_readonly_array(edge_members_buffer, np.uint32),
        edge_alpha=_readonly_array(edge_alpha_buffer, np.float64),
        edge_reliability=_readonly_array(edge_reliability_buffer, np.float64),
        edge_order=_readonly_array(edge_order_buffer, np.uint32),
        edge_effective_order=_readonly_array(
            edge_effective_order_buffer,
            np.float64,
        ),
        contig_edge_offsets=_readonly_array(
            contig_edge_offsets_buffer,
            np.uint64,
        ),
        contig_edge_ids=_readonly_array(contig_edge_ids_buffer, np.uint64),
        contig_edge_positions=_readonly_array(
            contig_edge_positions_buffer,
            np.uint64,
        ),
        eta=float(eta),
    )
    return index


def encode_assignment(
    index: ContactIndex,
    assignment: Mapping[str, str],
) -> tuple[IntArray, tuple[str, ...]]:
    """Encode public bin names as stable non-negative integer labels."""
    bin_names = tuple(
        sorted(
            {
                str(bin_name).strip()
                for bin_name in assignment.values()
                if str(bin_name).strip()
            },
            key=str,
        )
    )
    bin_name_to_idx = {name: idx for idx, name in enumerate(bin_names)}
    labels = np.full(index.n_contigs, -1, dtype=np.int32)
    for contig_name, bin_name_raw in assignment.items():
        bin_name = str(bin_name_raw).strip()
        if not bin_name:
            continue
        contig_idx = index.contig_name_to_idx.get(str(contig_name))
        if contig_idx is None:
            continue
        labels[int(contig_idx)] = int(bin_name_to_idx[bin_name])
    return labels, bin_names


def _readonly_array(buffer: array, dtype: np.dtype[object]) -> NDArray:
    out = np.frombuffer(buffer, dtype=dtype).copy()
    out.flags.writeable = False
    return out


def _make_readonly(*arrays: NDArray) -> None:
    for value in arrays:
        value.flags.writeable = False


def _require_index(value: int, size: int, label: str) -> None:
    if int(value) < 0 or int(value) >= int(size):
        raise IndexError(f"{label} index out of range: {value}")
