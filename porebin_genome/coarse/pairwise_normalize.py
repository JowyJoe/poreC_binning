"""Length correction and ICE-style balancing for clique-expanded pairwise contacts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from porebin_genome.io.runtime import write_json


DEFAULT_LENGTH_EPS = 1.0
DEFAULT_ICE_MAX_ITER = 200
DEFAULT_ICE_TOL = 1.0e-3
DEFAULT_DROP_QUANTILE = 0.05


class PairwiseNormalizationError(RuntimeError):
    """Raised when pairwise normalization fails."""


@dataclass(frozen=True)
class PairwiseNormalizationResult:
    """Outputs and audit metrics from pairwise length correction plus balancing."""

    normalized_contacts_parquet: Path
    meta_json: Path
    n_pair_edges_in: int
    n_pair_edges_positive_weight: int
    total_observed_contacts: int
    n_ice_iterations: int
    ice_converged: bool
    drop_quantile: float
    drop_threshold: float


def normalize_pairwise_contacts(
    *,
    pairwise_contacts_path: Path,
    contig_lengths: dict[str, int],
    coverage_by_contig: dict[str, float],
    out_parquet: Path,
    meta_json: Path,
    length_eps: float = DEFAULT_LENGTH_EPS,
    max_iter: int = DEFAULT_ICE_MAX_ITER,
    tol: float = DEFAULT_ICE_TOL,
    drop_quantile: float = DEFAULT_DROP_QUANTILE,
) -> PairwiseNormalizationResult:
    """Normalize raw clique-expanded pair counts with length correction and ICE balancing.

    Coverage is accepted for API symmetry with other coarse routines but is not used
    here: this pairwise route is a deliberately simple contact-map baseline.
    """
    _ = coverage_by_contig
    if float(length_eps) <= 0.0:
        raise PairwiseNormalizationError("length_eps must be positive.")
    if int(max_iter) < 1:
        raise PairwiseNormalizationError("max_iter must be positive.")
    if float(tol) <= 0.0:
        raise PairwiseNormalizationError("tol must be positive.")
    if not (0.0 <= float(drop_quantile) < 1.0):
        raise PairwiseNormalizationError("drop_quantile must be in [0, 1).")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover
        raise PairwiseNormalizationError("Pairwise normalization requires pyarrow.") from exc

    pairwise_contacts_path = pairwise_contacts_path.resolve()
    out_parquet = out_parquet.resolve()
    meta_json = meta_json.resolve()
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(pairwise_contacts_path)
    data = table.to_pydict()
    required = {"left_idx", "right_idx", "left_contig", "right_contig", "n_contacts", "q_sum"}
    missing = sorted(required.difference(data.keys()))
    if missing:
        raise PairwiseNormalizationError(
            f"Pairwise contacts table missing columns: {', '.join(missing)}"
        )

    left_idx_values = [int(value) for value in data["left_idx"]]
    right_idx_values = [int(value) for value in data["right_idx"]]
    left_names = [str(value) for value in data["left_contig"]]
    right_names = [str(value) for value in data["right_contig"]]
    n_values = [int(value) for value in data["n_contacts"]]
    q_values = [float(value) for value in data["q_sum"]]
    total_observed = int(sum(n_values))

    corrected_values: list[float] = []
    edge_pairs: list[tuple[int, int]] = []
    for left_idx, right_idx, left_name, right_name, q_sum, n_contacts in zip(
        left_idx_values,
        right_idx_values,
        left_names,
        right_names,
        q_values,
        n_values,
        strict=True,
    ):
        left_length = float(max(float(contig_lengths.get(left_name, 0)), float(length_eps)))
        right_length = float(max(float(contig_lengths.get(right_name, 0)), float(length_eps)))
        signal = float(q_sum if q_sum > 0.0 else n_contacts)
        corrected = float(signal / math.sqrt(left_length * right_length))
        corrected_values.append(max(0.0, corrected))
        edge_pairs.append((int(left_idx), int(right_idx)))

    balanced_values, ice_iters, ice_converged, row_imbalance = _ice_balance_sparse(
        edge_pairs=edge_pairs,
        weights=corrected_values,
        max_iter=int(max_iter),
        tol=float(tol),
    )
    positive_balanced = [float(value) for value in balanced_values if float(value) > 0.0]
    drop_threshold = _percentile(positive_balanced, float(drop_quantile)) if positive_balanced else 0.0

    reliability_values = _empirical_reliability(balanced_values)
    final_weight_values: list[float] = []
    positive_weight_count = 0
    for balanced, reliability in zip(balanced_values, reliability_values, strict=True):
        if float(balanced) <= 0.0 or float(balanced) < float(drop_threshold):
            final_weight_values.append(0.0)
            continue
        positive_weight_count += 1
        final_weight_values.append(float(balanced * max(0.0, reliability)))

    normalized = pa.table(
        {
            "left_idx": pa.array(left_idx_values, type=pa.int32()),
            "right_idx": pa.array(right_idx_values, type=pa.int32()),
            "left_contig": pa.array(left_names, type=pa.string()),
            "right_contig": pa.array(right_names, type=pa.string()),
            "n_contacts": pa.array(n_values, type=pa.int64()),
            "q_sum": pa.array(q_values, type=pa.float64()),
            "length_corrected": pa.array(corrected_values, type=pa.float64()),
            "balanced_weight": pa.array(balanced_values, type=pa.float64()),
            "reliability": pa.array(reliability_values, type=pa.float64()),
            "weight": pa.array(final_weight_values, type=pa.float64()),
        }
    )
    pq.write_table(normalized, out_parquet, compression="zstd")

    result = PairwiseNormalizationResult(
        normalized_contacts_parquet=out_parquet,
        meta_json=meta_json,
        n_pair_edges_in=int(len(n_values)),
        n_pair_edges_positive_weight=int(positive_weight_count),
        total_observed_contacts=int(total_observed),
        n_ice_iterations=int(ice_iters),
        ice_converged=bool(ice_converged),
        drop_quantile=float(drop_quantile),
        drop_threshold=float(drop_threshold),
    )
    write_json(
        meta_json,
        {
            "stage": "pairwise_contact_map_normalization",
            "method": "length_correction_plus_ice_style_balancing",
            "inputs": {"pairwise_clique_contacts_parquet": str(pairwise_contacts_path)},
            "outputs": {"pairwise_normalized_contacts_parquet": str(out_parquet)},
            "formula": {
                "length_correction": "A_ij = Q_ij / sqrt(L_i * L_j)",
                "balancing": "Iteratively scale rows/columns so non-isolated contact-map row sums approach 1.",
                "filter": "Drop the bottom drop_quantile of positive balanced weights as weak pairwise contacts.",
                "weight": "W_ij = balanced_weight * empirical_reliability among retained positive pair edges.",
            },
            "length_eps": float(length_eps),
            "ice_max_iter": int(max_iter),
            "ice_tol": float(tol),
            "ice_iterations": int(ice_iters),
            "ice_converged": bool(ice_converged),
            "final_row_imbalance": float(row_imbalance),
            "drop_quantile": float(drop_quantile),
            "drop_threshold": float(drop_threshold),
            "total_observed_contacts": int(total_observed),
            "n_pair_edges_in": int(len(n_values)),
            "n_pair_edges_positive_weight": int(positive_weight_count),
            "notes": {
                "baseline_semantics": (
                    "This intentionally simple pairwise baseline follows common contact-map "
                    "practice: length correction, ICE-style matrix balancing, and weak-edge filtering."
                ),
                "coverage_policy": "Coverage is not used in this baseline normalization.",
            },
        },
    )
    return result


def _ice_balance_sparse(
    *,
    edge_pairs: list[tuple[int, int]],
    weights: list[float],
    max_iter: int,
    tol: float,
) -> tuple[list[float], int, bool, float]:
    balanced = [float(max(0.0, value)) for value in weights]
    if not balanced or not any(value > 0.0 for value in balanced):
        return balanced, 0, True, 0.0

    nodes = sorted({idx for pair in edge_pairs for idx in pair})
    converged = False
    last_imbalance = float("inf")
    iteration = 0
    for iteration in range(1, int(max_iter) + 1):
        row_sums = _row_sums(edge_pairs=edge_pairs, weights=balanced, nodes=nodes)
        positive = [value for value in row_sums.values() if value > 0.0]
        if not positive:
            converged = True
            last_imbalance = 0.0
            break
        target = 1.0
        last_imbalance = max(abs(float(value) / target - 1.0) for value in positive)
        if last_imbalance <= float(tol):
            converged = True
            break
        scale = {
            node: (math.sqrt(target / row_sums[node]) if row_sums.get(node, 0.0) > 0.0 else 1.0)
            for node in nodes
        }
        balanced = [
            float(weight) * float(scale.get(left, 1.0)) * float(scale.get(right, 1.0))
            for (left, right), weight in zip(edge_pairs, balanced, strict=True)
        ]

    row_sums = _row_sums(edge_pairs=edge_pairs, weights=balanced, nodes=nodes)
    positive = [value for value in row_sums.values() if value > 0.0]
    final_imbalance = max((abs(float(value) - 1.0) for value in positive), default=0.0)
    return balanced, int(iteration), bool(converged), float(final_imbalance)


def _row_sums(
    *,
    edge_pairs: list[tuple[int, int]],
    weights: list[float],
    nodes: list[int],
) -> dict[int, float]:
    row_sums = {int(node): 0.0 for node in nodes}
    for (left, right), weight in zip(edge_pairs, weights, strict=True):
        value = float(max(0.0, weight))
        row_sums[int(left)] = float(row_sums.get(int(left), 0.0) + value)
        row_sums[int(right)] = float(row_sums.get(int(right), 0.0) + value)
    return row_sums


def _empirical_reliability(values: list[float]) -> list[float]:
    positives = sorted(float(value) for value in values if float(value) > 0.0)
    if not positives:
        return [0.0 for _ in values]
    out: list[float] = []
    for value in values:
        if float(value) <= 0.0:
            out.append(0.0)
            continue
        rank = _upper_bound(positives, float(value))
        out.append(float(rank / len(positives)))
    return out


def _upper_bound(xs: list[float], value: float) -> int:
    lo = 0
    hi = len(xs)
    while lo < hi:
        mid = (lo + hi) // 2
        if xs[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(value) for value in values)
    if len(xs) == 1:
        return float(xs[0])
    pos = float(q) * float(len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return float((1.0 - frac) * xs[lo] + frac * xs[hi])
