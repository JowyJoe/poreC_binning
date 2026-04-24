"""Feature incidence and joint-operator primitives for coarse discovery."""

from __future__ import annotations

from dataclasses import dataclass


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
