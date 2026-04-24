"""Embedding clustering and coarse bins.tsv materialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from porebin_genome.io.contracts import COARSE_BINS_COLUMNS
from porebin_genome.io.tables import write_tsv_rows


class CoarseClusterError(RuntimeError):
    """Raised when HDBSCAN clustering or bins writing fails."""


@dataclass(frozen=True)
class ClusterResult:
    """Coarse clustering outputs and audit counts."""

    labels: "object"
    n_bins: int
    n_contigs_clustered: int
    n_contigs_unbinned: int
    hdbscan_meta: dict


def hdbscan_cluster(
    Z: "object",
    *,
    min_cluster_size: int,
    min_samples: Optional[int],
    selection_method: str,
    threads: int,
) -> tuple["object", dict]:
    """Cluster the embedding with HDBSCAN and return labels plus metadata."""
    import numpy as np

    try:
        import hdbscan
    except Exception as exc:  # pragma: no cover
        raise CoarseClusterError("Coarse clustering requires the hdbscan package.") from exc

    Z = np.asarray(Z, dtype=np.float32)
    if Z.ndim != 2:
        raise CoarseClusterError("Embedding must be 2-dimensional.")

    selection_method = str(selection_method).strip().lower()
    if selection_method not in {"leaf", "eom"}:
        raise CoarseClusterError(
            f"HDBSCAN selection_method must be 'leaf' or 'eom', got {selection_method!r}."
        )

    kwargs = {
        "min_cluster_size": int(max(2, min_cluster_size)),
        "min_samples": (None if min_samples is None else int(min_samples)),
        "metric": "euclidean",
        "cluster_selection_method": selection_method,
    }
    if int(threads) > 1:
        kwargs["core_dist_n_jobs"] = int(threads)

    clusterer = hdbscan.HDBSCAN(**kwargs)
    labels = np.asarray(clusterer.fit_predict(Z), dtype=int)
    return labels, {
        "impl": "hdbscan",
        "min_cluster_size": int(kwargs["min_cluster_size"]),
        "min_samples": kwargs["min_samples"],
        "metric": kwargs["metric"],
        "selection_method": selection_method,
    }


def summarize_labels(labels: "object") -> tuple[int, int, int]:
    """Return bin count, clustered contigs, and unbinned contigs."""
    import numpy as np

    labels = np.asarray(labels, dtype=int)
    n_contigs_clustered = int(np.sum(labels != -1))
    n_contigs_unbinned = int(np.sum(labels == -1))
    n_bins = int(len({int(label) for label in labels.tolist() if int(label) != -1}))
    return n_bins, n_contigs_clustered, n_contigs_unbinned


def write_coarse_bins_tsv(
    *,
    out_path: Path,
    idx_to_name: list[str],
    labels: "object",
) -> ClusterResult:
    """Write candidate genome bins without promoting noise or reassigning components."""
    import numpy as np

    labels = np.asarray(labels, dtype=int)
    if labels.shape[0] != len(idx_to_name):
        raise CoarseClusterError("labels length does not match contig index.")

    unique_labels = sorted({int(label) for label in labels.tolist() if int(label) != -1})
    remap = {old_label: str(new_label) for new_label, old_label in enumerate(unique_labels)}
    rows = [
        (idx_to_name[idx], remap[int(label)])
        for idx, label in enumerate(labels.tolist())
        if int(label) != -1
    ]
    write_tsv_rows(out_path, COARSE_BINS_COLUMNS, rows)

    n_bins, n_contigs_clustered, n_contigs_unbinned = summarize_labels(labels)
    return ClusterResult(
        labels=labels,
        n_bins=n_bins,
        n_contigs_clustered=n_contigs_clustered,
        n_contigs_unbinned=n_contigs_unbinned,
        hdbscan_meta={},
    )
