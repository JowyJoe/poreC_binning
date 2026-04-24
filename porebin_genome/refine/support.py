"""Support aggregation, feature profiles, and confidence scoring for refine MVP."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

from porebin_genome.coarse.features import build_feature_matrix
from porebin_genome.evidence.canonical import iter_canonical_contacts
from porebin_genome.io.coverage import read_coverage_tsv
from porebin_genome.io.fasta import iter_fasta_names
from porebin_genome.refine.models import BinFeatureProfile, RefineState, SupportSummary


def median(values: list[float]) -> float:
    """Return the median of a list or 0 for empty input."""
    if not values:
        return 0.0
    xs = sorted(float(value) for value in values)
    mid = len(xs) // 2
    if len(xs) % 2 == 1:
        return float(xs[mid])
    return 0.5 * float(xs[mid - 1] + xs[mid])


def mad(values: list[float], *, center: Optional[float] = None) -> float:
    """Return the median absolute deviation or 0 for empty input."""
    if not values:
        return 0.0
    reference = float(center) if center is not None else median(values)
    return median([abs(float(value) - reference) for value in values])


def clip01(value: float) -> float:
    """Clamp a float to the closed interval [0, 1]."""
    return float(max(0.0, min(1.0, float(value))))


def build_contig_index(contigs_fasta: Path) -> tuple[dict[str, int], list[str]]:
    """Build contig index mappings from FASTA order."""
    idx_to_name = list(iter_fasta_names(contigs_fasta))
    if len(set(idx_to_name)) != len(idx_to_name):
        raise RuntimeError("contigs.fasta contains duplicate contig identifiers.")
    return {name: idx for idx, name in enumerate(idx_to_name)}, idx_to_name


def load_refine_feature_inputs(
    *,
    contigs_fasta: Path,
    coverage_tsv: Path,
) -> tuple["object", dict[str, int], list[str], dict[str, float]]:
    """Load the feature matrix, contig index, and coverage lookup for refine."""
    contig_name_to_idx, idx_to_name = build_contig_index(contigs_fasta)
    feature_matrix = build_feature_matrix(
        contigs_fasta=contigs_fasta,
        coverage_tsv=coverage_tsv,
        contig_name_to_idx=contig_name_to_idx,
        feature_mode="tnf_plus_cov",
    )
    coverage_by_contig = read_coverage_tsv(coverage_tsv)
    return feature_matrix.X, contig_name_to_idx, idx_to_name, coverage_by_contig


def scan_bin_support(
    *,
    contacts_parquet: Path,
    state: RefineState,
) -> SupportSummary:
    """Aggregate contig-to-bin support and within-bin pair support from contacts.parquet."""
    support_by_contig_bin: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    pair_support_by_bin: dict[str, dict[tuple[str, str], float]] = defaultdict(lambda: defaultdict(float))

    for row in iter_canonical_contacts(contacts_parquet, require_contig_weights=True):
        if row.k_valid < 2 or row.contig_weights is None:
            continue
        members = [(str(name), float(weight)) for name, weight in zip(row.contigs, row.contig_weights, strict=True)]
        for idx, (left_contig, left_weight) in enumerate(members):
            for jdx in range(idx + 1, len(members)):
                right_contig, right_weight = members[jdx]
                contribution = float(row.weight) * float(left_weight) * float(right_weight)
                if contribution <= 0.0:
                    continue
                left_bin = state.current_assignment.get(left_contig, "")
                right_bin = state.current_assignment.get(right_contig, "")
                if right_bin:
                    support_by_contig_bin[left_contig][right_bin] += contribution
                if left_bin:
                    support_by_contig_bin[right_contig][left_bin] += contribution
                if left_bin and left_bin == right_bin:
                    pair_key = tuple(sorted((left_contig, right_contig)))
                    pair_support_by_bin[left_bin][pair_key] += contribution

    own_support: dict[str, float] = {}
    total_support: dict[str, float] = {}
    runner_up_bin: dict[str, str] = {}
    runner_up_support: dict[str, float] = {}
    for contig_id in state.contig_lengths:
        scores = dict(support_by_contig_bin.get(contig_id, {}))
        total = float(sum(scores.values()))
        total_support[contig_id] = total
        current_bin = state.current_assignment.get(contig_id, "")
        own_support[contig_id] = float(scores.get(current_bin, 0.0)) if current_bin else 0.0
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if current_bin:
            ordered = [(bin_id, score) for bin_id, score in ordered if bin_id != current_bin]
        if ordered:
            runner_up_bin[contig_id] = str(ordered[0][0])
            runner_up_support[contig_id] = float(ordered[0][1])
        else:
            runner_up_bin[contig_id] = ""
            runner_up_support[contig_id] = 0.0

    return SupportSummary(
        support_by_contig_bin={key: dict(value) for key, value in support_by_contig_bin.items()},
        own_support=own_support,
        total_support=total_support,
        runner_up_bin=runner_up_bin,
        runner_up_support=runner_up_support,
        pair_support_by_bin={key: dict(value) for key, value in pair_support_by_bin.items()},
        contact_components_by_bin=_compute_contact_components(
            state=state,
            pair_support_by_bin={key: dict(value) for key, value in pair_support_by_bin.items()},
        ),
    )


def _compute_contact_components(
    *,
    state: RefineState,
    pair_support_by_bin: dict[str, dict[tuple[str, str], float]],
) -> dict[str, tuple[tuple[str, ...], ...]]:
    out: dict[str, tuple[tuple[str, ...], ...]] = {}
    for bin_id, members in state.bin_to_contigs().items():
        if not members:
            out[bin_id] = ()
            continue
        parent = {contig_id: contig_id for contig_id in members}

        def find(contig_id: str) -> str:
            while parent[contig_id] != contig_id:
                parent[contig_id] = parent[parent[contig_id]]
                contig_id = parent[contig_id]
            return contig_id

        def union(left: str, right: str) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for (left, right), weight in pair_support_by_bin.get(bin_id, {}).items():
            if weight > 0.0:
                union(left, right)

        groups: dict[str, list[str]] = defaultdict(list)
        for contig_id in members:
            groups[find(contig_id)].append(contig_id)
        ordered = sorted(
            (tuple(sorted(group)) for group in groups.values()),
            key=lambda group: (-len(group), tuple(group)),
        )
        out[bin_id] = tuple(ordered)
    return out


def build_bin_feature_profiles(
    *,
    state: RefineState,
    feature_matrix: "object",
    contig_name_to_idx: dict[str, int],
) -> dict[str, BinFeatureProfile]:
    """Compute bin centroids and feature dispersions for the current assignment."""
    import numpy as np

    profiles: dict[str, BinFeatureProfile] = {}
    for bin_id, members in state.bin_to_contigs().items():
        member_idx = [contig_name_to_idx[contig_id] for contig_id in members if contig_id in contig_name_to_idx]
        if not member_idx:
            continue
        X = np.asarray(feature_matrix[member_idx, :], dtype=float)
        centroid = X.mean(axis=0)
        distances = np.linalg.norm(X - centroid, axis=1)
        coverage_values = [
            float(state.coverage_by_contig[contig_id])
            for contig_id in members
            if contig_id in state.coverage_by_contig
        ]
        profiles[str(bin_id)] = BinFeatureProfile(
            bin_id=str(bin_id),
            members=tuple(sorted(members)),
            centroid=centroid,
            distance_median=float(median(distances.tolist())),
            distance_mad=float(mad(distances.tolist())),
            feature_dispersion=float(median(distances.tolist())),
            median_coverage=(
                float(median(coverage_values)) if coverage_values else None
            ),
            coverage_mad=(
                float(mad(coverage_values)) if coverage_values else None
            ),
        )
    return profiles


def evaluate_feature_gate(
    *,
    contig_id: str,
    target_bin: str,
    feature_matrix: "object",
    contig_name_to_idx: dict[str, int],
    profiles: dict[str, BinFeatureProfile],
    coverage_by_contig: dict[str, float],
) -> tuple[bool, float, str]:
    """Evaluate whether a contig agrees with a target bin's feature profile."""
    import numpy as np

    profile = profiles.get(str(target_bin))
    if profile is None or contig_id not in contig_name_to_idx:
        return True, 1.0, "feature_profile_unavailable"

    contig_vec = np.asarray(feature_matrix[contig_name_to_idx[contig_id], :], dtype=float)
    distance = float(np.linalg.norm(contig_vec - profile.centroid))
    distance_limit = float(profile.distance_median + max(0.5, 3.0 * profile.distance_mad))
    distance_pass = distance <= max(distance_limit, 0.5)

    coverage_pass = True
    if profile.median_coverage is not None and profile.coverage_mad is not None and contig_id in coverage_by_contig:
        coverage_value = float(coverage_by_contig[contig_id])
        if profile.coverage_mad > 0.0:
            coverage_z = abs(coverage_value - profile.median_coverage) / profile.coverage_mad
            coverage_pass = coverage_z <= 3.0
        else:
            coverage_pass = abs(coverage_value - profile.median_coverage) <= max(1.0, 0.25 * profile.median_coverage)

    if distance_pass and coverage_pass:
        return True, 1.0, "feature_agreement_pass"
    if distance_pass:
        return False, 0.0, "coverage_gate_failed"
    if coverage_pass:
        return False, 0.0, "feature_distance_too_large"
    return False, 0.0, "feature_and_coverage_gate_failed"


def compute_assignment_confidence(
    *,
    target_bin_support: float,
    runner_up_support: float,
    feature_gate: bool,
) -> float:
    """Compute an interpretable 0..1 assignment confidence score."""
    numerator = float(max(0.0, target_bin_support))
    denominator = float(max(0.0, target_bin_support) + max(0.0, runner_up_support))
    support_share = numerator / denominator if denominator > 0.0 else 0.0
    margin = (
        (float(target_bin_support) - float(runner_up_support)) / denominator
        if denominator > 0.0
        else 0.0
    )
    margin = clip01(max(0.0, margin))
    gate_value = 1.0 if bool(feature_gate) else 0.0
    return clip01((0.45 * support_share) + (0.35 * margin) + (0.20 * gate_value))


def compute_contact_consistency(
    *,
    bin_id: str,
    members: list[str],
    support_summary: SupportSummary,
) -> float:
    """Compute bin-level contact consistency as the median own-support ratio."""
    ratios: list[float] = []
    for contig_id in members:
        total = float(support_summary.total_support.get(contig_id, 0.0))
        own = float(support_summary.support_by_contig_bin.get(contig_id, {}).get(bin_id, 0.0))
        if total <= 0.0:
            ratios.append(0.0)
            continue
        ratios.append(clip01(own / total))
    return float(median(ratios))


def runner_up_margin(*, target_bin_support: float, runner_up_support: float) -> float:
    """Compute a normalized runner-up margin in [0, 1]."""
    denominator = float(max(0.0, target_bin_support) + max(0.0, runner_up_support))
    if denominator <= 0.0:
        return 0.0
    return clip01(max(0.0, (float(target_bin_support) - float(runner_up_support)) / denominator))
