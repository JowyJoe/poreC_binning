"""Clique-expanded pairwise contact baseline construction."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from statistics import mean

from porebin_genome.evidence.canonical import ContactEvidenceError, iter_canonical_contacts
from porebin_genome.io.runtime import write_json


DEFAULT_PAIRWISE_ALPHA_MIN = 0.05


class PairwiseContactError(RuntimeError):
    """Raised when pairwise contact construction fails."""


@dataclass(frozen=True)
class PairwiseCliqueResult:
    """Outputs and audit metrics from raw clique expansion."""

    contacts_parquet: Path
    meta_json: Path
    n_contacts_in: int
    n_contacts_used: int
    n_contacts_dropped: int
    n_pair_edges: int
    mean_clique_inflation: float
    p95_clique_inflation: float
    max_clique_inflation: int
    alpha_min: float


def build_clique_pairwise_contacts(
    *,
    contacts_path: Path,
    contig_name_to_idx: dict[str, int],
    out_parquet: Path,
    meta_json: Path,
    alpha_min: float = DEFAULT_PAIRWISE_ALPHA_MIN,
    min_k: int = 2,
    parquet_batch_size: int = 200_000,
) -> PairwiseCliqueResult:
    """Expand each Pore-C hyperedge into a raw clique-expanded pairwise graph.

    This baseline deliberately treats one high-order Pore-C read as all of its
    pairwise contacts, so the emitted metadata records the resulting inflation.
    The main hypergraph method should not use this representation directly.
    """
    if float(alpha_min) < 0.0:
        raise PairwiseContactError("alpha_min must be non-negative.")
    if int(min_k) < 2:
        raise PairwiseContactError("min_k must be at least 2.")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover
        raise PairwiseContactError("Pairwise contact construction requires pyarrow.") from exc

    contacts_path = contacts_path.resolve()
    out_parquet = out_parquet.resolve()
    meta_json = meta_json.resolve()
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    pair_counts: dict[tuple[int, int], tuple[int, float]] = {}
    inflation_values: list[int] = []
    n_contacts_in = 0
    n_contacts_used = 0
    n_contacts_dropped = 0

    try:
        for row in iter_canonical_contacts(
            contacts_path,
            parquet_batch_size=int(parquet_batch_size),
            require_contig_weights=True,
        ):
            n_contacts_in += 1
            if row.k_valid < int(min_k) or row.contig_weights is None:
                n_contacts_dropped += 1
                continue

            active_indices: list[int] = []
            for contig_name, alpha in zip(row.contigs, row.contig_weights, strict=True):
                if float(alpha) < float(alpha_min):
                    continue
                contig_idx = contig_name_to_idx.get(str(contig_name))
                if contig_idx is None:
                    raise PairwiseContactError(
                        f"Contig {contig_name!r} in contacts.parquet is not present in contigs.fasta."
                    )
                active_indices.append(int(contig_idx))

            active_indices = sorted(set(active_indices))
            if len(active_indices) < int(min_k):
                n_contacts_dropped += 1
                continue

            pair_count = int(len(active_indices) * (len(active_indices) - 1) // 2)
            inflation_values.append(pair_count)
            n_contacts_used += 1
            q = float(row.weight)
            for left_idx, right_idx in combinations(active_indices, 2):
                key = (int(left_idx), int(right_idx))
                count, q_sum = pair_counts.get(key, (0, 0.0))
                pair_counts[key] = (int(count + 1), float(q_sum + q))
    except ContactEvidenceError as exc:
        raise PairwiseContactError(str(exc)) from exc

    idx_to_name = _invert_contig_index(contig_name_to_idx)
    left_idx_values: list[int] = []
    right_idx_values: list[int] = []
    left_names: list[str] = []
    right_names: list[str] = []
    n_values: list[int] = []
    q_values: list[float] = []
    for (left_idx, right_idx), (count, q_sum) in sorted(pair_counts.items()):
        left_idx_values.append(int(left_idx))
        right_idx_values.append(int(right_idx))
        left_names.append(idx_to_name[int(left_idx)])
        right_names.append(idx_to_name[int(right_idx)])
        n_values.append(int(count))
        q_values.append(float(q_sum))

    table = pa.table(
        {
            "left_idx": pa.array(left_idx_values, type=pa.int32()),
            "right_idx": pa.array(right_idx_values, type=pa.int32()),
            "left_contig": pa.array(left_names, type=pa.string()),
            "right_contig": pa.array(right_names, type=pa.string()),
            "n_contacts": pa.array(n_values, type=pa.int64()),
            "q_sum": pa.array(q_values, type=pa.float64()),
        }
    )
    pq.write_table(table, out_parquet, compression="zstd")

    p95 = _percentile(inflation_values, 0.95)
    result = PairwiseCliqueResult(
        contacts_parquet=out_parquet,
        meta_json=meta_json,
        n_contacts_in=int(n_contacts_in),
        n_contacts_used=int(n_contacts_used),
        n_contacts_dropped=int(n_contacts_dropped),
        n_pair_edges=int(len(pair_counts)),
        mean_clique_inflation=float(mean(inflation_values) if inflation_values else 0.0),
        p95_clique_inflation=float(p95),
        max_clique_inflation=int(max(inflation_values, default=0)),
        alpha_min=float(alpha_min),
    )
    write_json(
        meta_json,
        {
            "stage": "pairwise_clique_expansion",
            "method": "raw_clique_expansion",
            "contacts_parquet": str(contacts_path),
            "outputs": {"pairwise_clique_contacts_parquet": str(out_parquet)},
            "alpha_min": float(alpha_min),
            "min_k": int(min_k),
            "n_contacts_in": int(n_contacts_in),
            "n_contacts_used": int(n_contacts_used),
            "n_contacts_dropped": int(n_contacts_dropped),
            "n_pair_edges": int(len(pair_counts)),
            "mean_clique_inflation": float(result.mean_clique_inflation),
            "p95_clique_inflation": float(result.p95_clique_inflation),
            "max_clique_inflation": int(result.max_clique_inflation),
            "notes": {
                "baseline_semantics": (
                    "One high-order Pore-C contact is deliberately expanded into all pairwise "
                    "clique edges for a dimensionality-loss baseline."
                )
            },
        },
    )
    return result


def _invert_contig_index(contig_name_to_idx: dict[str, int]) -> dict[int, str]:
    out = {int(idx): str(name) for name, idx in contig_name_to_idx.items()}
    if len(out) != len(contig_name_to_idx):
        raise PairwiseContactError("contig_name_to_idx contains duplicate index values.")
    return out


def _percentile(values: list[int], q: float) -> float:
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
