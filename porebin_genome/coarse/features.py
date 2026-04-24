"""Sequence-composition and coverage features for candidate genome-bin discovery."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from porebin_genome.evidence.tnf import compute_tnf136_features
from porebin_genome.io.coverage import read_coverage_tsv


class CoarseFeatureError(RuntimeError):
    """Raised when feature construction fails."""


@dataclass(frozen=True)
class FeatureMatrix:
    """Feature matrix and coverage bookkeeping for coarse discovery."""

    X: "object"
    feature_mode: str
    coverage_used: bool
    coverage_missing_count: int


def load_coverage_feature_optional(
    coverage_tsv: Optional[Path],
    contig_name_to_idx: dict[str, int],
) -> tuple["object", int, bool]:
    """Load log1p coverage as a one-dimensional feature, filling 0 for missing contigs."""
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise CoarseFeatureError("Coverage feature loading requires numpy.") from exc

    n_contigs = len(contig_name_to_idx)
    coverage_vector = np.zeros((n_contigs,), dtype=np.float32)
    if coverage_tsv is None or not coverage_tsv.exists():
        return coverage_vector, n_contigs, False

    coverage_by_contig = read_coverage_tsv(coverage_tsv)
    present = 0
    for contig_name, idx in contig_name_to_idx.items():
        raw_coverage = coverage_by_contig.get(contig_name)
        if raw_coverage is None:
            continue
        coverage_vector[int(idx)] = float(math.log1p(max(0.0, float(raw_coverage))))
        present += 1
    return coverage_vector, int(n_contigs - present), True


def zscore_features(X: "object") -> "object":
    """Z-score each feature dimension and zero out zero-variance columns."""
    import numpy as np

    X = np.asarray(X, dtype=np.float32)
    mean = X.mean(axis=0, dtype=np.float64)
    var = ((X.astype(np.float64) - mean) ** 2).mean(axis=0)
    std = np.sqrt(var)
    out = np.zeros_like(X, dtype=np.float32)
    mask = std > 0
    out[:, mask] = ((X[:, mask].astype(np.float64) - mean[mask]) / std[mask]).astype(np.float32)
    return out


def build_feature_matrix(
    *,
    contigs_fasta: Path,
    coverage_tsv: Optional[Path],
    contig_name_to_idx: dict[str, int],
    feature_mode: str = "tnf_plus_cov",
    logger: Optional[object] = None,
) -> FeatureMatrix:
    """Build the genome-centric feature matrix used for coarse discovery."""
    _ = logger
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise CoarseFeatureError("Feature construction requires numpy.") from exc

    feature_mode = str(feature_mode).strip().lower()
    if feature_mode not in {"tnf_plus_cov", "tnf_only"}:
        raise CoarseFeatureError(
            f"Unknown feature_mode {feature_mode!r}. Use 'tnf_plus_cov' or 'tnf_only'."
        )

    composition = compute_tnf136_features(contigs_fasta, contig_name_to_idx, logger=logger)
    coverage_vector, coverage_missing_count, coverage_used = load_coverage_feature_optional(
        coverage_tsv, contig_name_to_idx
    )

    if feature_mode == "tnf_only":
        X = composition.astype(np.float32, copy=False)
    else:
        X = np.concatenate(
            [composition.astype(np.float32, copy=False), coverage_vector.reshape(-1, 1)],
            axis=1,
        )
    return FeatureMatrix(
        X=zscore_features(X),
        feature_mode=feature_mode,
        coverage_used=bool(coverage_used),
        coverage_missing_count=int(coverage_missing_count),
    )
