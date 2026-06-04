"""Pairwise Leiden baseline for clique-expanded Pore-C contacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from porebin_genome.coarse.cluster import write_coarse_bins_tsv
from porebin_genome.coarse.metadata import COLLAPSE_WARNING_THRESHOLD
from porebin_genome.io.runtime import write_json
from porebin_genome.io.tables import write_tsv_rows


DEFAULT_BASELINE_RESOLUTIONS = (0.25, 0.5, 1.0, 2.0, 4.0)
PAIRWISE_LEIDEN_SWEEP_COLUMNS = (
    "resolution",
    "quality",
    "n_bins",
    "largest_bin_size",
    "largest_bin_fraction",
    "collapse_warning",
    "selected",
)


class PairwiseBaselineError(RuntimeError):
    """Raised when the pairwise baseline cannot be produced."""


@dataclass(frozen=True)
class PairwiseBaselineResult:
    """Outputs from the selected pairwise baseline run."""

    bins_tsv: Path
    sweep_tsv: Path
    meta_json: Path
    selected_resolution: float
    n_bins: int
    largest_bin_fraction: float
    collapse_warning: bool


def run_pairwise_leiden_baseline(
    *,
    pairwise_normalized_contacts_path: Path,
    idx_to_name: list[str],
    bins_tsv: Path,
    sweep_tsv: Path,
    meta_json: Path,
    resolutions: tuple[float, ...] | list[float] = DEFAULT_BASELINE_RESOLUTIONS,
) -> PairwiseBaselineResult:
    """Run a resolution sweep over a clique-expanded normalized pairwise graph."""
    try:
        import igraph as ig
        import leidenalg
        import numpy as np
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover
        raise PairwiseBaselineError(
            "Pairwise Leiden baseline requires igraph, leidenalg, numpy, and pyarrow."
        ) from exc

    pairwise_normalized_contacts_path = pairwise_normalized_contacts_path.resolve()
    bins_tsv = bins_tsv.resolve()
    sweep_tsv = sweep_tsv.resolve()
    meta_json = meta_json.resolve()

    graph, n_positive_edges = _load_weighted_graph(
        pairwise_normalized_contacts_path=pairwise_normalized_contacts_path,
        idx_to_name=idx_to_name,
        ig=ig,
        pq=pq,
    )
    if n_positive_edges == 0:
        raise PairwiseBaselineError("Pairwise normalized graph has no positive-weight edges.")

    labels_by_resolution: dict[float, object] = {}
    sweep_rows: list[dict[str, object]] = []
    for resolution in tuple(float(value) for value in resolutions):
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=float(resolution),
            seed=0,
        )
        labels = np.asarray(partition.membership, dtype=int)
        summary = _summarize_labels(labels)
        row = {
            "resolution": float(resolution),
            "quality": float(partition.quality()),
            "n_bins": int(summary["n_bins"]),
            "largest_bin_size": int(summary["largest_bin_size"]),
            "largest_bin_fraction": float(summary["largest_bin_fraction"]),
            "collapse_warning": bool(summary["collapse_warning"]),
            "selected": False,
        }
        labels_by_resolution[float(resolution)] = labels
        sweep_rows.append(row)

    selected_row = _select_resolution(sweep_rows)
    selected_resolution = float(selected_row["resolution"])
    selected_row["selected"] = True
    selected_labels = labels_by_resolution[selected_resolution]
    cluster_result = write_coarse_bins_tsv(
        out_path=bins_tsv,
        idx_to_name=idx_to_name,
        labels=selected_labels,
    )
    write_tsv_rows(
        sweep_tsv,
        PAIRWISE_LEIDEN_SWEEP_COLUMNS,
        (
            (
                row["resolution"],
                f"{float(row['quality']):.8g}",
                row["n_bins"],
                row["largest_bin_size"],
                f"{float(row['largest_bin_fraction']):.8g}",
                row["collapse_warning"],
                row["selected"],
            )
            for row in sweep_rows
        ),
    )

    collapse_warning = bool(selected_row["collapse_warning"])
    write_json(
        meta_json,
        {
            "stage": "pairwise_leiden_baseline",
            "method": "clique_expansion_length_corrected_ice_leiden",
            "inputs": {"pairwise_normalized_contacts_parquet": str(pairwise_normalized_contacts_path)},
            "outputs": {
                "bins_tsv": str(bins_tsv),
                "sweep_tsv": str(sweep_tsv),
            },
            "n_contigs_total": int(len(idx_to_name)),
            "n_positive_pair_edges": int(n_positive_edges),
            "resolutions": [float(value) for value in resolutions],
            "selected_resolution": float(selected_resolution),
            "n_bins": int(cluster_result.n_bins),
            "largest_bin_fraction": float(selected_row["largest_bin_fraction"]),
            "collapse_warning": collapse_warning,
            "collapse_warning_threshold": float(COLLAPSE_WARNING_THRESHOLD),
            "selection_rule": (
                "Choose the highest-quality non-collapsed resolution when available; "
                "otherwise choose the highest-quality collapsed resolution and flag collapse_warning."
            ),
        },
    )
    return PairwiseBaselineResult(
        bins_tsv=bins_tsv,
        sweep_tsv=sweep_tsv,
        meta_json=meta_json,
        selected_resolution=float(selected_resolution),
        n_bins=int(cluster_result.n_bins),
        largest_bin_fraction=float(selected_row["largest_bin_fraction"]),
        collapse_warning=collapse_warning,
    )


def _load_weighted_graph(
    *,
    pairwise_normalized_contacts_path: Path,
    idx_to_name: list[str],
    ig,
    pq,
):
    table = pq.read_table(
        pairwise_normalized_contacts_path,
        columns=["left_idx", "right_idx", "weight"],
    )
    data = table.to_pydict()
    graph = ig.Graph()
    graph.add_vertices(list(idx_to_name))

    edge_pairs: list[tuple[int, int]] = []
    edge_weights: list[float] = []
    n_contigs = len(idx_to_name)
    for left_idx, right_idx, weight in zip(
        data["left_idx"],
        data["right_idx"],
        data["weight"],
        strict=True,
    ):
        w = float(weight)
        if w <= 0.0:
            continue
        left = int(left_idx)
        right = int(right_idx)
        if not (0 <= left < n_contigs and 0 <= right < n_contigs and left != right):
            continue
        edge_pairs.append((left, right))
        edge_weights.append(w)
    if edge_pairs:
        graph.add_edges(edge_pairs)
        graph.es["weight"] = edge_weights
    return graph, len(edge_pairs)


def _summarize_labels(labels) -> dict[str, object]:
    import numpy as np

    labels = np.asarray(labels, dtype=int)
    unique = sorted({int(value) for value in labels.tolist()})
    sizes = [int(np.sum(labels == value)) for value in unique]
    largest = max(sizes, default=0)
    largest_fraction = float(largest / max(1, labels.shape[0]))
    return {
        "n_bins": int(len(unique)),
        "largest_bin_size": int(largest),
        "largest_bin_fraction": float(largest_fraction),
        "collapse_warning": bool(largest_fraction > float(COLLAPSE_WARNING_THRESHOLD)),
    }


def _select_resolution(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise PairwiseBaselineError("No Leiden resolution rows were produced.")
    non_collapsed = [row for row in rows if not bool(row["collapse_warning"])]
    candidates = non_collapsed if non_collapsed else rows
    return max(
        candidates,
        key=lambda row: (
            float(row["quality"]),
            int(row["n_bins"]),
            -float(row["largest_bin_fraction"]),
            -float(row["resolution"]),
        ),
    )
