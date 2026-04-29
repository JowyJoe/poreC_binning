"""Feature- and coverage-based consistency utilities for refinement."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from porebin_genome.coarse.features import build_feature_matrix
from porebin_genome.io.coverage import read_coverage_tsv
from porebin_genome.io.fasta import iter_fasta_names
from porebin_genome.refine.models import BinFeatureProfile, RefineState


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
