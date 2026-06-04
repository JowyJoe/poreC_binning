"""Metadata and collapse-audit output for coarse candidate genome bins."""

from __future__ import annotations


COLLAPSE_WARNING_THRESHOLD = 0.80


def _bin_size_quantiles(labels: "object") -> dict[str, float]:
    import numpy as np

    labels = np.asarray(labels, dtype=int)
    sizes = [
        int(np.sum(labels == label))
        for label in sorted({int(label) for label in labels.tolist() if int(label) != -1})
    ]
    if not sizes:
        return {"p00": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p100": 0.0}

    quantiles = np.quantile(np.asarray(sizes, dtype=float), [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "p00": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p100": float(quantiles[4]),
    }


def build_coarse_run_record(
    *,
    contigs_fasta: str,
    contacts_parquet: str,
    coverage_tsv: str,
    bins_tsv: str,
    labels: "object",
    feature_mode: str,
    embedding_dim: int,
    lambda_contact: float | None,
    contact_hyperedge_count: int,
    feature_knn_k: int,
    dropped_singleton_contacts: int,
    coverage_used: bool,
    coverage_missing_count: int,
    hdbscan_meta: dict,
) -> dict:
    """Build the coarse-stage run.json payload with anti-collapse audit metrics."""
    import numpy as np

    labels = np.asarray(labels, dtype=int)
    n_contigs_total = int(labels.shape[0])
    n_contigs_clustered = int(np.sum(labels != -1))
    n_contigs_unbinned = int(np.sum(labels == -1))
    unique_labels = sorted({int(label) for label in labels.tolist() if int(label) != -1})
    n_bins = int(len(unique_labels))
    largest_bin_size = max((int(np.sum(labels == label)) for label in unique_labels), default=0)
    largest_bin_fraction = float(largest_bin_size / max(1, n_contigs_total))
    hdbscan_noise_fraction = float(n_contigs_unbinned / max(1, n_contigs_total))
    collapse_warning = largest_bin_fraction > float(COLLAPSE_WARNING_THRESHOLD)
    warnings: list[str] = []
    if collapse_warning:
        warnings.append(
            "largest_bin_fraction exceeds collapse threshold; coarse candidate bins may be over-collapsed"
        )

    return {
        "stage": "coarse_genome_bin_discovery",
        "implemented": True,
        "notes": {
            "candidate_bin_semantics": "coarse outputs are candidate genome bins only",
            "postprocess_policy": "no component-majority reassignment and no noise promotion are applied",
        },
        "inputs": {
            "contigs_fasta": contigs_fasta,
            "contacts_parquet": contacts_parquet,
            "coverage_tsv": coverage_tsv,
        },
        "outputs": {
            "bins_tsv": bins_tsv,
        },
        "n_contigs_total": n_contigs_total,
        "n_contigs_clustered": n_contigs_clustered,
        "n_contigs_unbinned": n_contigs_unbinned,
        "n_bins": n_bins,
        "largest_bin_fraction": largest_bin_fraction,
        "hdbscan_noise_fraction": hdbscan_noise_fraction,
        "bin_size_quantiles": _bin_size_quantiles(labels),
        "embedding_dim": int(embedding_dim),
        "feature_mode": feature_mode,
        "contact_hyperedge_count": int(contact_hyperedge_count),
        "feature_knn_k": int(feature_knn_k),
        "lambda_contact": (None if lambda_contact is None else float(lambda_contact)),
        "dropped_singleton_contacts": int(dropped_singleton_contacts),
        "coverage_used": bool(coverage_used),
        "coverage_missing_count": int(coverage_missing_count),
        "hdbscan": dict(hdbscan_meta),
        "collapse_warning": bool(collapse_warning),
        "collapse_warning_threshold": float(COLLAPSE_WARNING_THRESHOLD),
        "warnings": warnings,
    }
