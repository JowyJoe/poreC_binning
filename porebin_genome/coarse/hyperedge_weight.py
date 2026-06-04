"""Hypergraph-native Pore-C hyperedge weighting."""

from __future__ import annotations

from typing import Iterable


DEFAULT_HYPERGRAPH_WEIGHT_ETA = 0.5


class HyperedgeWeightError(RuntimeError):
    """Raised when hyperedge weight parameters are invalid."""


def normalize_contact_weight_mode(value: str) -> str:
    """Normalize public contact-weight mode strings."""
    mode = str(value).strip().lower().replace("-", "_")
    if mode == "legacy":
        return "original"
    if mode == "order_aware":
        return "hypergraph_native"
    if mode not in {"original", "hypergraph_native"}:
        raise ValueError("--contact-weight-mode must be 'original' or 'hypergraph-native'.")
    return mode


def effective_order(alpha_values: Iterable[float]) -> float:
    """Return k_eff = 1 / sum(alpha_i^2) for normalized or unnormalized alpha."""
    values = [max(0.0, float(value)) for value in alpha_values]
    total = float(sum(values))
    if total <= 0.0:
        return 0.0
    normalized = [value / total for value in values]
    concentration = float(sum(value * value for value in normalized))
    if concentration <= 0.0:
        return 0.0
    return float(1.0 / concentration)


def hypergraph_native_weight(
    *,
    read_weight: float,
    alpha_values: Iterable[float],
    eta: float = DEFAULT_HYPERGRAPH_WEIGHT_ETA,
) -> float:
    """Return a hyperedge-native evidence budget from q_e, alpha, and k_eff."""
    if float(eta) < 0.0:
        raise HyperedgeWeightError("eta must be non-negative.")
    q = float(max(0.0, float(read_weight)))
    k_eff = effective_order(alpha_values)
    if k_eff <= 0.0 or q <= 0.0:
        return 0.0
    denominator = max(float(k_eff) - 1.0, 1.0) ** float(eta)
    return float(q / denominator)
