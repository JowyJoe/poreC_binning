"""Feature incidence and joint-operator primitives for coarse discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from porebin_genome.io.tables import write_tsv_rows


DEFAULT_ADAPTIVE_MIN_K = 5
DEFAULT_ADAPTIVE_K = 15
DEFAULT_ADAPTIVE_K_MAX = 30
DEFAULT_ADAPTIVE_DROP_RATIO = 0.15
DEFAULT_ADAPTIVE_MUTUAL_KNN = True
ADAPTIVE_KNN_METHOD_REFERENCE_NOTE = (
    "Adaptive local kNN graph inspired by aKNNO-style adaptive nearest-neighbor graph "
    "clustering, self-tuning spectral clustering local scale selection, and reciprocal "
    "or shared-neighborhood support from natural-neighbor, SNN, and mutual-kNN graphs."
)
ADAPTIVE_K_REPORT_COLUMNS = (
    "contig_id",
    "selected_k_i",
    "top_score",
    "cutoff_score",
    "cutoff_rank",
    "cutoff_reason",
    "retained_neighbor_count",
)


class CoarseOperatorError(RuntimeError):
    """Raised when coarse operator construction fails."""


@dataclass(frozen=True)
class FeatureIncidence:
    """Sparse feature incidence matrix and its diagonal terms."""

    H_csr: "object"
    W: "object"
    De: "object"
    Dv: "object"
    feature_knn_k: int


@dataclass(frozen=True)
class AdaptiveKnnReportRow:
    """Per-contig adaptive-k selection audit row."""

    contig_id: str
    selected_k_i: int
    top_score: float | None
    cutoff_score: float | None
    cutoff_rank: int
    cutoff_reason: str
    retained_neighbor_count: int


@dataclass(frozen=True)
class AdaptiveKnnResult:
    """Adaptive-k neighbor matrix plus reproducibility metadata."""

    neighbors: "object"
    report_rows: tuple[AdaptiveKnnReportRow, ...]
    meta: dict[str, object]


@dataclass(frozen=True)
class ThetaOperator:
    """Linear operator representation of a hypergraph diffusion kernel."""

    op: "object"
    isolated_mask: "object"


def build_feature_knn_edges(X: "object", knn_k: int) -> "object":
    """Compute k nearest neighbors for the feature matrix, excluding self."""
    try:
        import numpy as np
        from sklearn.neighbors import NearestNeighbors
    except Exception as exc:  # pragma: no cover
        raise CoarseOperatorError("Feature kNN construction requires scikit-learn.") from exc

    X = np.asarray(X, dtype=np.float32)
    n_contigs = int(X.shape[0])
    if n_contigs <= 1:
        return np.zeros((n_contigs, 0), dtype=np.int32)

    k_eff = int(min(max(1, int(knn_k)), n_contigs - 1))
    model = NearestNeighbors(metric="euclidean", n_neighbors=int(k_eff) + 1)
    model.fit(X)
    return np.asarray(model.kneighbors(X, return_distance=False)[:, 1:], dtype=np.int32)


def _as_contig_ids(contig_ids: "object", n_contigs: int) -> list[str]:
    if contig_ids is None:
        return [str(idx) for idx in range(n_contigs)]
    names = [str(name) for name in contig_ids]
    if len(names) != n_contigs:
        raise CoarseOperatorError("contig_ids length does not match feature matrix row count.")
    return names


def _score_from_distance(distance: float) -> float:
    return float(1.0 / (1.0 + max(0.0, float(distance))))


def _select_local_k(
    distances: "object",
    *,
    min_k: int,
    default_k: int,
    drop_ratio: float,
) -> tuple[int, str, list[float]]:
    import numpy as np

    d = np.asarray(distances, dtype=float)
    if d.size == 0:
        return 0, "no_candidate_neighbors", []

    scores = [_score_from_distance(float(value)) for value in d.tolist()]
    eps = 1.0e-12
    for idx in range(int(min_k), int(d.size)):
        prev_score = float(scores[idx - 1])
        curr_score = float(scores[idx])
        score_drop = (prev_score - curr_score) / max(abs(prev_score), eps)
        if score_drop >= float(drop_ratio):
            selected = int(max(min_k, idx))
            return selected, f"score_drop_after_rank_{selected}:relative_drop={score_drop:.6g}", scores

        prev_distance = float(d[idx - 1])
        curr_distance = float(d[idx])
        distance_jump = (curr_distance - prev_distance) / max(abs(prev_distance), eps)
        if curr_distance > prev_distance and distance_jump >= float(drop_ratio):
            selected = int(max(min_k, idx))
            return selected, f"distance_jump_after_rank_{selected}:relative_increase={distance_jump:.6g}", scores

    selected = int(min(max(min_k, default_k), d.size))
    return selected, "no_clear_drop_default_k", scores


def _pad_neighbor_lists(neighbor_lists: list[list[int]], n_contigs: int) -> "object":
    import numpy as np

    width = max((len(row) for row in neighbor_lists), default=0)
    neighbors = np.full((n_contigs, width), -1, dtype=np.int32)
    for row_idx, row in enumerate(neighbor_lists):
        if row:
            neighbors[row_idx, : len(row)] = np.asarray(row, dtype=np.int32)
    return neighbors


def build_adaptive_feature_knn_edges(
    X: "object",
    *,
    contig_ids: "object" = None,
    min_k: int = DEFAULT_ADAPTIVE_MIN_K,
    default_k: int = DEFAULT_ADAPTIVE_K,
    k_max: int = DEFAULT_ADAPTIVE_K_MAX,
    drop_ratio: float = DEFAULT_ADAPTIVE_DROP_RATIO,
    mutual_knn: bool = DEFAULT_ADAPTIVE_MUTUAL_KNN,
) -> AdaptiveKnnResult:
    """Build a local adaptive-k feature graph using one nearest-neighbor pass."""
    try:
        import numpy as np
        from sklearn.neighbors import NearestNeighbors
    except Exception as exc:  # pragma: no cover
        raise CoarseOperatorError("Adaptive feature kNN construction requires scikit-learn.") from exc

    X = np.asarray(X, dtype=np.float32)
    n_contigs = int(X.shape[0])
    names = _as_contig_ids(contig_ids, n_contigs)
    if n_contigs <= 1:
        rows = tuple(
            AdaptiveKnnReportRow(
                contig_id=name,
                selected_k_i=0,
                top_score=None,
                cutoff_score=None,
                cutoff_rank=0,
                cutoff_reason="too_few_contigs",
                retained_neighbor_count=0,
            )
            for name in names
        )
        return AdaptiveKnnResult(
            neighbors=np.zeros((n_contigs, 0), dtype=np.int32),
            report_rows=rows,
            meta=build_adaptive_knn_meta(
                min_k=min_k,
                default_k=default_k,
                k_max=k_max,
                drop_ratio=drop_ratio,
                mutual_knn=mutual_knn,
                fallback_used=False,
                fallback_reason=None,
            ),
        )

    if not (0.0 < float(drop_ratio) < 1.0):
        raise CoarseOperatorError("drop_ratio must be greater than 0 and less than 1.")
    k_max_eff = int(min(max(1, int(k_max)), n_contigs - 1))
    min_k_eff = int(min(max(1, int(min_k)), k_max_eff))
    default_k_eff = int(min(max(min_k_eff, int(default_k)), k_max_eff))

    model = NearestNeighbors(metric="euclidean", n_neighbors=k_max_eff + 1)
    model.fit(X)
    distances, indices = model.kneighbors(X, return_distance=True)
    candidate_distances = np.asarray(distances[:, 1:], dtype=float)
    candidate_indices = np.asarray(indices[:, 1:], dtype=np.int32)

    selected_counts: list[int] = []
    selected_neighbor_lists: list[list[int]] = []
    reasons: list[str] = []
    score_rows: list[list[float]] = []
    for contig_idx in range(n_contigs):
        selected, reason, scores = _select_local_k(
            candidate_distances[contig_idx],
            min_k=min_k_eff,
            default_k=default_k_eff,
            drop_ratio=float(drop_ratio),
        )
        selected = int(min(max(min_k_eff, selected), k_max_eff))
        selected_counts.append(selected)
        selected_neighbor_lists.append(
            [int(value) for value in candidate_indices[contig_idx, :selected].tolist()]
        )
        reasons.append(reason)
        score_rows.append(scores)

    final_neighbor_lists = selected_neighbor_lists
    if bool(mutual_knn):
        selected_sets = [set(row) for row in selected_neighbor_lists]
        final_neighbor_lists = []
        for contig_idx, row in enumerate(selected_neighbor_lists):
            final_neighbor_lists.append(
                [
                    int(neighbor_idx)
                    for neighbor_idx in row
                    if contig_idx in selected_sets[int(neighbor_idx)]
                ]
            )

    report_rows: list[AdaptiveKnnReportRow] = []
    for contig_idx, selected in enumerate(selected_counts):
        scores = score_rows[contig_idx]
        top_score = scores[0] if scores else None
        cutoff_score = scores[selected - 1] if selected > 0 and scores else None
        report_rows.append(
            AdaptiveKnnReportRow(
                contig_id=names[contig_idx],
                selected_k_i=int(selected),
                top_score=top_score,
                cutoff_score=cutoff_score,
                cutoff_rank=int(selected),
                cutoff_reason=reasons[contig_idx],
                retained_neighbor_count=int(len(final_neighbor_lists[contig_idx])),
            )
        )

    neighbors = _pad_neighbor_lists(final_neighbor_lists, n_contigs)
    meta = build_adaptive_knn_meta(
        min_k=min_k,
        default_k=default_k,
        k_max=k_max,
        drop_ratio=drop_ratio,
        mutual_knn=mutual_knn,
        fallback_used=False,
        fallback_reason=None,
        n_contigs=n_contigs,
        effective_min_k=min_k_eff,
        effective_default_k=default_k_eff,
        effective_k_max=k_max_eff,
        selected_k_min=min(selected_counts) if selected_counts else 0,
        selected_k_median=float(np.median(np.asarray(selected_counts, dtype=float))) if selected_counts else 0.0,
        selected_k_max=max(selected_counts) if selected_counts else 0,
        retained_directed_edges=int(sum(len(row) for row in final_neighbor_lists)),
    )
    return AdaptiveKnnResult(neighbors=neighbors, report_rows=tuple(report_rows), meta=meta)


def build_adaptive_knn_meta(
    *,
    min_k: int = DEFAULT_ADAPTIVE_MIN_K,
    default_k: int = DEFAULT_ADAPTIVE_K,
    k_max: int = DEFAULT_ADAPTIVE_K_MAX,
    drop_ratio: float = DEFAULT_ADAPTIVE_DROP_RATIO,
    mutual_knn: bool = DEFAULT_ADAPTIVE_MUTUAL_KNN,
    fallback_used: bool = False,
    fallback_reason: str | None = None,
    **extra: object,
) -> dict[str, object]:
    meta: dict[str, object] = {
        "mode": "adaptive",
        "min_k": int(min_k),
        "default_k": int(default_k),
        "k_max": int(k_max),
        "drop_ratio": float(drop_ratio),
        "mutual_knn": bool(mutual_knn),
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
        "method_reference_note": ADAPTIVE_KNN_METHOD_REFERENCE_NOTE,
        "score_definition": "score = 1 / (1 + euclidean_distance)",
        "cutoff_rule": (
            "For each contig, truncate before the first candidate rank after min_k where "
            "the relative score drop or distance increase is at least drop_ratio; otherwise "
            "retain default_k candidates, capped by k_max."
        ),
    }
    meta.update(extra)
    return meta


def build_adaptive_knn_fallback_report_rows(
    *,
    contig_ids: "object",
    fallback_k: int = DEFAULT_ADAPTIVE_K,
    n_contigs: int | None = None,
) -> tuple[AdaptiveKnnReportRow, ...]:
    names = [str(name) for name in contig_ids]
    total = len(names) if n_contigs is None else int(n_contigs)
    k_eff = int(min(max(0, fallback_k), max(0, total - 1)))
    return tuple(
        AdaptiveKnnReportRow(
            contig_id=name,
            selected_k_i=k_eff,
            top_score=None,
            cutoff_score=None,
            cutoff_rank=k_eff,
            cutoff_reason=f"fallback_fixed_k_{k_eff}",
            retained_neighbor_count=k_eff,
        )
        for name in names
    )


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.8g}"


def write_adaptive_knn_report(path: Path, rows: tuple[AdaptiveKnnReportRow, ...] | list[AdaptiveKnnReportRow]) -> None:
    write_tsv_rows(
        path,
        ADAPTIVE_K_REPORT_COLUMNS,
        (
            (
                row.contig_id,
                row.selected_k_i,
                _format_optional_float(row.top_score),
                _format_optional_float(row.cutoff_score),
                row.cutoff_rank,
                row.cutoff_reason,
                row.retained_neighbor_count,
            )
            for row in rows
        ),
    )


def build_feature_incidence(neighbors: "object") -> FeatureIncidence:
    """Build one feature hyperedge per contig using its kNN neighborhood."""
    try:
        import numpy as np
        import scipy.sparse as sp
    except Exception as exc:  # pragma: no cover
        raise CoarseOperatorError("Feature incidence construction requires numpy and scipy.") from exc

    neighbors = np.asarray(neighbors, dtype=np.int32)
    n_contigs = int(neighbors.shape[0])
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for contig_idx in range(n_contigs):
        members = {int(contig_idx)}
        for neighbor_idx in neighbors[contig_idx].tolist():
            if 0 <= int(neighbor_idx) < n_contigs and int(neighbor_idx) != contig_idx:
                members.add(int(neighbor_idx))
        for member_idx in members:
            rows.append(int(member_idx))
            cols.append(int(contig_idx))
            data.append(1.0)

    H = sp.coo_matrix(
        (
            np.asarray(data, dtype=float),
            (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)),
        ),
        shape=(n_contigs, n_contigs),
    ).tocsr()
    W = np.ones((n_contigs,), dtype=float)
    De = np.asarray(H.sum(axis=0)).ravel()
    Dv = np.asarray(H.sum(axis=1)).ravel()
    return FeatureIncidence(
        H_csr=H,
        W=W,
        De=De,
        Dv=Dv,
        feature_knn_k=int(neighbors.shape[1] if neighbors.ndim == 2 else 0),
    )


def make_theta_operator(H_csr: "object", W: "object", De: "object", Dv: "object") -> ThetaOperator:
    """Construct Theta = Dv^-1/2 H W De^-1 H^T Dv^-1/2 as a LinearOperator."""
    import numpy as np
    import scipy.sparse.linalg as spla

    H = H_csr
    W = np.asarray(W, dtype=float)
    De = np.asarray(De, dtype=float)
    Dv = np.asarray(Dv, dtype=float)

    if H.shape[1] != W.shape[0] or H.shape[1] != De.shape[0]:
        raise CoarseOperatorError("Hypergraph incidence shape does not match W/De lengths.")
    if H.shape[0] != Dv.shape[0]:
        raise CoarseOperatorError("Hypergraph incidence shape does not match Dv length.")

    Dv_inv_sqrt = np.zeros_like(Dv, dtype=float)
    mask_vertex = Dv > 0
    Dv_inv_sqrt[mask_vertex] = 1.0 / np.sqrt(Dv[mask_vertex])

    De_inv = np.zeros_like(De, dtype=float)
    mask_edge = De > 0
    De_inv[mask_edge] = 1.0 / De[mask_edge]
    scale = W * De_inv

    def matvec(x):
        x = np.asarray(x, dtype=float)
        y = Dv_inv_sqrt * x
        z = H.T @ y
        z = z * scale
        y2 = H @ z
        return Dv_inv_sqrt * np.asarray(y2).ravel()

    op = spla.LinearOperator((H.shape[0], H.shape[0]), matvec=matvec, dtype=float)
    return ThetaOperator(op=op, isolated_mask=np.asarray(~mask_vertex, dtype=bool))
