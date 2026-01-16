"""
Confidence calculation and quality filtering for multiplex spectral clustering.

This module implements post-clustering quality assurance WITHOUT using
downstream evaluation tools (like CheckM2). Instead, it uses upstream
data features and computed metrics.

Three components:
1. Contig Confidence: How confident is the assignment of each contig?
2. Low Confidence Filtering: Mark uncertain contigs as "unassigned"
3. Bin Quality Check: Internal consistency within each bin

Key Metrics (all computed from upstream data):
- Silhouette Score per contig: Cluster membership quality
- Pore-C Contact Ratio: Physical contact within bin vs outside
- TNF Consistency: Tetranucleotide frequency similarity within bin
- Coverage Consistency: Sequencing depth similarity within bin

Design Philosophy:
- Similar to MetaBAT2's "adaptive binning" vision
- No external tool dependencies for quality assessment
- Leverage Pore-C's unique physical contact information
- "Better to miss than to misclassify" → prioritize low contamination

References:
- Rousseeuw (1987). Silhouettes. J Comput Appl Math.
- Kang et al. (2019). MetaBAT 2. PeerJ.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import silhouette_samples


@dataclass
class ContigConfidence:
    """Confidence metrics for a single contig."""
    contig_idx: int
    bin_id: int
    silhouette: float          # [-1, 1], higher = better fit
    contact_ratio: float       # [0, 1], higher = more intra-bin contacts
    combined_score: float      # Weighted combination
    is_confident: bool         # Above threshold?


@dataclass
class BinQuality:
    """Quality metrics for a single bin."""
    bin_id: int
    n_contigs: int
    mean_silhouette: float
    mean_contact_ratio: float
    tnf_consistency: float     # [0, 1], higher = more consistent
    coverage_consistency: float  # [0, 1], higher = more consistent
    pore_c_connectivity: float  # [0, 1], fraction of contigs with contacts
    overall_quality: float     # Combined quality score


@dataclass
class QualityResult:
    """Result of quality assessment and filtering."""
    original_labels: np.ndarray
    filtered_labels: np.ndarray  # -1 for unassigned
    contig_confidences: List[ContigConfidence]
    bin_qualities: List[BinQuality]
    n_unassigned: int
    n_bins_original: int
    n_bins_filtered: int
    method_details: Dict = field(default_factory=dict)


# =============================================================================
# Contig-level Confidence
# =============================================================================

def compute_silhouette_per_contig(
    X: np.ndarray,
    labels: np.ndarray
) -> np.ndarray:
    """
    Compute Silhouette Score for each contig.

    Silhouette measures how similar a contig is to its own cluster
    compared to other clusters. Range: [-1, 1].

    - s(i) ≈ 1: contig is well-matched to its cluster
    - s(i) ≈ 0: contig is on the boundary
    - s(i) < 0: contig might be in the wrong cluster

    Parameters
    ----------
    X : (N, d) embedding matrix
    labels : (N,) cluster assignments

    Returns
    -------
    silhouettes : (N,) Silhouette Score per contig
    """
    unique_labels = np.unique(labels[labels >= 0])

    if len(unique_labels) < 2:
        # Can't compute silhouette with < 2 clusters
        return np.zeros(len(labels))

    # Mask for assigned contigs
    mask = labels >= 0

    silhouettes = np.zeros(len(labels))
    if mask.sum() > 0:
        silhouettes[mask] = silhouette_samples(X[mask], labels[mask])

    return silhouettes


def compute_contact_ratio(
    contact_matrix: csr_matrix,
    labels: np.ndarray,
    contig_idx: int
) -> float:
    """
    Compute Pore-C contact ratio for a contig.

    Contact Ratio = (contacts within bin) / (total contacts)

    This leverages Pore-C's unique physical contact information:
    - High ratio: contig has strong physical connections within its bin
    - Low ratio: contig might be misassigned

    Parameters
    ----------
    contact_matrix : (N, N) Pore-C contact matrix
    labels : (N,) cluster assignments
    contig_idx : Index of the contig

    Returns
    -------
    ratio : Contact ratio [0, 1]
    """
    bin_id = labels[contig_idx]
    if bin_id < 0:
        return 0.0

    # Get contacts for this contig
    row = contact_matrix.getrow(contig_idx).toarray().flatten()

    # Total contacts (excluding self)
    row[contig_idx] = 0
    total_contacts = row.sum()

    if total_contacts == 0:
        return 0.0

    # Contacts within the same bin
    same_bin_mask = labels == bin_id
    same_bin_mask[contig_idx] = False  # Exclude self
    intra_contacts = row[same_bin_mask].sum()

    return float(intra_contacts / total_contacts)


def compute_contig_confidences(
    X: np.ndarray,
    labels: np.ndarray,
    contact_matrix: Optional[csr_matrix] = None,
    silhouette_weight: float = 0.5,
    contact_weight: float = 0.5,
    confidence_threshold: float = 0.3
) -> List[ContigConfidence]:
    """
    Compute confidence scores for all contigs.

    Combined Score = w1 * normalized_silhouette + w2 * contact_ratio

    Where normalized_silhouette = (silhouette + 1) / 2 to map [-1,1] to [0,1]

    Parameters
    ----------
    X : (N, d) embedding matrix
    labels : (N,) cluster assignments
    contact_matrix : (N, N) Pore-C contact matrix (optional)
    silhouette_weight : Weight for Silhouette component
    contact_weight : Weight for contact ratio component
    confidence_threshold : Threshold for "confident" assignment

    Returns
    -------
    confidences : List of ContigConfidence for each contig
    """
    n = len(labels)

    # Compute Silhouette per contig
    silhouettes = compute_silhouette_per_contig(X, labels)

    # Normalize silhouette to [0, 1]
    silhouettes_norm = (silhouettes + 1) / 2

    # Compute contact ratios if contact matrix provided
    if contact_matrix is not None:
        contact_ratios = np.array([
            compute_contact_ratio(contact_matrix, labels, i)
            for i in range(n)
        ])
    else:
        contact_ratios = np.ones(n) * 0.5  # Neutral if no contact info
        contact_weight = 0.0
        silhouette_weight = 1.0

    # Normalize weights
    total_weight = silhouette_weight + contact_weight
    w1 = silhouette_weight / total_weight
    w2 = contact_weight / total_weight

    # Compute combined scores
    combined = w1 * silhouettes_norm + w2 * contact_ratios

    # Build result
    confidences = []
    for i in range(n):
        conf = ContigConfidence(
            contig_idx=i,
            bin_id=int(labels[i]),
            silhouette=float(silhouettes[i]),
            contact_ratio=float(contact_ratios[i]),
            combined_score=float(combined[i]),
            is_confident=combined[i] >= confidence_threshold
        )
        confidences.append(conf)

    return confidences


# =============================================================================
# Low Confidence Filtering
# =============================================================================

def filter_low_confidence(
    labels: np.ndarray,
    confidences: List[ContigConfidence],
    min_confidence: float = 0.3
) -> np.ndarray:
    """
    Filter out low-confidence assignments.

    Philosophy: "Better to miss than to misclassify"
    - Low confidence contigs → mark as unassigned (-1)
    - This reduces contamination at the cost of some completeness

    Parameters
    ----------
    labels : (N,) original cluster assignments
    confidences : List of ContigConfidence
    min_confidence : Minimum confidence threshold

    Returns
    -------
    filtered_labels : (N,) with low-confidence contigs set to -1
    """
    filtered = labels.copy()

    for conf in confidences:
        if conf.combined_score < min_confidence:
            filtered[conf.contig_idx] = -1

    return filtered


# =============================================================================
# Bin-level Quality
# =============================================================================

def compute_tnf_consistency(
    tnf_matrix: np.ndarray,
    labels: np.ndarray,
    bin_id: int
) -> float:
    """
    Compute TNF consistency within a bin.

    TNF (Tetranucleotide Frequency) is a genomic signature.
    Contigs from the same genome should have similar TNF.

    Consistency = 1 - mean(pairwise_distances) / max_distance

    Parameters
    ----------
    tnf_matrix : (N, 136) TNF feature matrix
    labels : (N,) cluster assignments
    bin_id : Bin to evaluate

    Returns
    -------
    consistency : [0, 1], higher = more consistent
    """
    mask = labels == bin_id
    n_in_bin = mask.sum()

    if n_in_bin < 2:
        return 1.0  # Single contig is perfectly consistent

    tnf_bin = tnf_matrix[mask]

    # Compute pairwise distances
    from scipy.spatial.distance import pdist
    distances = pdist(tnf_bin, metric='euclidean')

    if len(distances) == 0:
        return 1.0

    mean_dist = distances.mean()
    max_dist = distances.max() if distances.max() > 0 else 1.0

    # Normalize to [0, 1]
    consistency = 1.0 - (mean_dist / max_dist)
    return float(max(0.0, consistency))


def compute_coverage_consistency(
    coverage: np.ndarray,
    labels: np.ndarray,
    bin_id: int
) -> float:
    """
    Compute coverage consistency within a bin.

    Contigs from the same genome should have similar sequencing depth.

    Consistency = 1 - CV (coefficient of variation)
    where CV = std / mean

    Parameters
    ----------
    coverage : (N,) coverage values
    labels : (N,) cluster assignments
    bin_id : Bin to evaluate

    Returns
    -------
    consistency : [0, 1], higher = more consistent
    """
    mask = labels == bin_id
    cov_bin = coverage[mask]

    if len(cov_bin) < 2:
        return 1.0

    mean_cov = cov_bin.mean()
    if mean_cov == 0:
        return 0.0

    cv = cov_bin.std() / mean_cov

    # Cap CV at 1 for normalization
    consistency = 1.0 - min(cv, 1.0)
    return float(max(0.0, consistency))


def compute_pore_c_connectivity(
    contact_matrix: csr_matrix,
    labels: np.ndarray,
    bin_id: int
) -> float:
    """
    Compute Pore-C connectivity within a bin.

    Connectivity = fraction of contigs that have at least one
    physical contact with another contig in the same bin.

    This is unique to Pore-C data and helps identify:
    - Well-connected bins (high connectivity)
    - Fragmented bins (low connectivity)

    Parameters
    ----------
    contact_matrix : (N, N) Pore-C contact matrix
    labels : (N,) cluster assignments
    bin_id : Bin to evaluate

    Returns
    -------
    connectivity : [0, 1], higher = more connected
    """
    mask = labels == bin_id
    indices = np.where(mask)[0]
    n_in_bin = len(indices)

    if n_in_bin < 2:
        return 1.0

    n_connected = 0
    for idx in indices:
        row = contact_matrix.getrow(idx).toarray().flatten()
        # Check if has contact with any other contig in bin
        row[idx] = 0  # Exclude self
        intra_contacts = row[mask].sum()
        if intra_contacts > 0:
            n_connected += 1

    return float(n_connected / n_in_bin)


def compute_bin_qualities(
    labels: np.ndarray,
    confidences: List[ContigConfidence],
    tnf_matrix: Optional[np.ndarray] = None,
    coverage: Optional[np.ndarray] = None,
    contact_matrix: Optional[csr_matrix] = None
) -> List[BinQuality]:
    """
    Compute quality metrics for all bins.

    Parameters
    ----------
    labels : (N,) cluster assignments
    confidences : List of ContigConfidence
    tnf_matrix : (N, 136) TNF feature matrix (optional)
    coverage : (N,) coverage values (optional)
    contact_matrix : (N, N) Pore-C contact matrix (optional)

    Returns
    -------
    qualities : List of BinQuality for each bin
    """
    unique_bins = np.unique(labels[labels >= 0])
    qualities = []

    for bin_id in unique_bins:
        mask = labels == bin_id
        n_contigs = mask.sum()

        # Aggregate contig confidences
        bin_confidences = [c for c in confidences if c.bin_id == bin_id]
        mean_silhouette = np.mean([c.silhouette for c in bin_confidences])
        mean_contact_ratio = np.mean([c.contact_ratio for c in bin_confidences])

        # TNF consistency
        if tnf_matrix is not None:
            tnf_consistency = compute_tnf_consistency(tnf_matrix, labels, bin_id)
        else:
            tnf_consistency = 0.5  # Neutral

        # Coverage consistency
        if coverage is not None:
            coverage_consistency = compute_coverage_consistency(coverage, labels, bin_id)
        else:
            coverage_consistency = 0.5  # Neutral

        # Pore-C connectivity
        if contact_matrix is not None:
            pore_c_connectivity = compute_pore_c_connectivity(
                contact_matrix, labels, bin_id
            )
        else:
            pore_c_connectivity = 0.5  # Neutral

        # Overall quality (weighted average)
        overall = (
            0.3 * (mean_silhouette + 1) / 2 +  # Normalize to [0,1]
            0.3 * mean_contact_ratio +
            0.2 * tnf_consistency +
            0.1 * coverage_consistency +
            0.1 * pore_c_connectivity
        )

        quality = BinQuality(
            bin_id=int(bin_id),
            n_contigs=int(n_contigs),
            mean_silhouette=float(mean_silhouette),
            mean_contact_ratio=float(mean_contact_ratio),
            tnf_consistency=float(tnf_consistency),
            coverage_consistency=float(coverage_consistency),
            pore_c_connectivity=float(pore_c_connectivity),
            overall_quality=float(overall)
        )
        qualities.append(quality)

    return qualities


# =============================================================================
# Main Entry Point
# =============================================================================

def assess_and_filter(
    X: np.ndarray,
    labels: np.ndarray,
    contact_matrix: Optional[csr_matrix] = None,
    tnf_matrix: Optional[np.ndarray] = None,
    coverage: Optional[np.ndarray] = None,
    confidence_threshold: float = 0.3,
    silhouette_weight: float = 0.5,
    contact_weight: float = 0.5,
    verbose: bool = True
) -> QualityResult:
    """
    Assess clustering quality and filter low-confidence assignments.

    This is the main entry point for post-clustering quality assurance.

    Workflow:
    1. Compute confidence for each contig (Silhouette + Contact Ratio)
    2. Filter low-confidence contigs (mark as unassigned)
    3. Compute quality metrics for each bin

    Parameters
    ----------
    X : (N, d) embedding matrix used for clustering
    labels : (N,) cluster assignments
    contact_matrix : (N, N) Pore-C contact matrix (optional but recommended)
    tnf_matrix : (N, 136) TNF feature matrix (optional)
    coverage : (N,) coverage values (optional)
    confidence_threshold : Minimum confidence for assignment
    silhouette_weight : Weight for Silhouette in confidence
    contact_weight : Weight for contact ratio in confidence
    verbose : Print progress

    Returns
    -------
    QualityResult with filtered labels and quality metrics
    """
    n = len(labels)
    n_bins_original = len(np.unique(labels[labels >= 0]))

    if verbose:
        print("=" * 60)
        print("[Quality] Post-clustering quality assessment")
        print("=" * 60)
        print(f"  Contigs: {n}")
        print(f"  Original bins: {n_bins_original}")
        print(f"  Confidence threshold: {confidence_threshold}")
        print()

    # Step 1: Compute contig confidences
    if verbose:
        print("[Step 1] Computing contig confidences...")

    confidences = compute_contig_confidences(
        X, labels, contact_matrix,
        silhouette_weight=silhouette_weight,
        contact_weight=contact_weight,
        confidence_threshold=confidence_threshold
    )

    n_confident = sum(1 for c in confidences if c.is_confident)
    if verbose:
        print(f"  Confident contigs: {n_confident}/{n} ({100*n_confident/n:.1f}%)")

    # Step 2: Filter low-confidence
    if verbose:
        print("[Step 2] Filtering low-confidence assignments...")

    filtered_labels = filter_low_confidence(
        labels, confidences, min_confidence=confidence_threshold
    )

    n_unassigned = (filtered_labels < 0).sum()
    n_bins_filtered = len(np.unique(filtered_labels[filtered_labels >= 0]))

    if verbose:
        print(f"  Unassigned contigs: {n_unassigned} ({100*n_unassigned/n:.1f}%)")
        print(f"  Remaining bins: {n_bins_filtered}")

    # Step 3: Compute bin qualities
    if verbose:
        print("[Step 3] Computing bin quality metrics...")

    bin_qualities = compute_bin_qualities(
        filtered_labels, confidences,
        tnf_matrix=tnf_matrix,
        coverage=coverage,
        contact_matrix=contact_matrix
    )

    if verbose and bin_qualities:
        mean_quality = np.mean([q.overall_quality for q in bin_qualities])
        print(f"  Mean bin quality: {mean_quality:.3f}")

    # Summary
    if verbose:
        print()
        print("=" * 60)
        print("[Result] Quality assessment complete")
        print(f"  Original: {n_bins_original} bins, {n} contigs")
        print(f"  Filtered: {n_bins_filtered} bins, {n - n_unassigned} assigned")
        print(f"  Unassigned: {n_unassigned} contigs")
        print("=" * 60)

    return QualityResult(
        original_labels=labels,
        filtered_labels=filtered_labels,
        contig_confidences=confidences,
        bin_qualities=bin_qualities,
        n_unassigned=int(n_unassigned),
        n_bins_original=int(n_bins_original),
        n_bins_filtered=int(n_bins_filtered),
        method_details={
            "confidence_threshold": confidence_threshold,
            "silhouette_weight": silhouette_weight,
            "contact_weight": contact_weight,
            "metrics_used": {
                "silhouette": True,
                "contact_ratio": contact_matrix is not None,
                "tnf_consistency": tnf_matrix is not None,
                "coverage_consistency": coverage is not None,
            }
        }
    )
