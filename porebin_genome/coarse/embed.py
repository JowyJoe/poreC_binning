"""Spectral embedding for genome-centric candidate genome-bin discovery."""

from __future__ import annotations

import math


class CoarseEmbeddingError(RuntimeError):
    """Raised when spectral embedding fails."""


def auto_embedding_dim(n_contigs: int) -> int:
    """Choose a conservative embedding dimension for coarse discovery."""
    if n_contigs <= 1:
        return 1
    d = min(128, max(32, int(math.floor(math.log2(float(n_contigs)))) * 4))
    d = min(d, max(1, n_contigs - 2))
    return int(d)


def _materialize_joint_matrix(
    *,
    contact_op: "object",
    feature_op: "object",
    lambda_contact: float,
) -> "object":
    import numpy as np

    n_contigs = int(contact_op.shape[0])
    basis = np.eye(n_contigs, dtype=float)
    dense = np.column_stack(
        [
            lambda_contact * (contact_op @ basis[:, idx])
            + (1.0 - lambda_contact) * (feature_op @ basis[:, idx])
            for idx in range(n_contigs)
        ]
    )
    return 0.5 * (dense + dense.T)


def spectral_embed_joint(
    *,
    contact_op: "object",
    feature_op: "object",
    lambda_contact: float,
    d: int,
    seed: int,
) -> "object":
    """Build a joint spectral embedding without any post-clustering relabeling."""
    try:
        import numpy as np
        import scipy.sparse.linalg as spla
    except Exception as exc:  # pragma: no cover
        raise CoarseEmbeddingError("Spectral embedding requires numpy and scipy.") from exc

    n_contigs = int(contact_op.shape[0])
    if int(feature_op.shape[0]) != n_contigs:
        raise CoarseEmbeddingError("Contact and feature operators must have matching shapes.")

    lambda_contact = float(lambda_contact)
    if not (0.0 <= lambda_contact <= 1.0):
        raise CoarseEmbeddingError(f"lambda_contact must be in [0, 1], got {lambda_contact}.")

    d = max(1, min(int(d), max(1, n_contigs - 1)))

    if n_contigs <= 64:
        dense = _materialize_joint_matrix(
            contact_op=contact_op,
            feature_op=feature_op,
            lambda_contact=lambda_contact,
        )
        evals, evecs = np.linalg.eigh(dense)
        order = np.argsort(evals)[::-1]
        evecs = evecs[:, order]
    else:
        k = int(min(d + 1, n_contigs - 1))
        if k < 2:
            raise CoarseEmbeddingError(f"Too few contigs for spectral embedding (n={n_contigs}).")

        def matvec(x):
            return lambda_contact * (contact_op @ x) + (1.0 - lambda_contact) * (feature_op @ x)

        joint = spla.LinearOperator((n_contigs, n_contigs), matvec=matvec, dtype=float)
        v0 = np.random.default_rng(int(seed)).standard_normal(n_contigs)
        try:
            evals, evecs = spla.eigsh(joint, k=k, which="LA", v0=v0)
        except Exception:
            dense = _materialize_joint_matrix(
                contact_op=contact_op,
                feature_op=feature_op,
                lambda_contact=lambda_contact,
            )
            evals, evecs = np.linalg.eigh(dense)
        order = np.argsort(evals)[::-1]
        evecs = evecs[:, order]

    d_eff = min(int(d), max(1, evecs.shape[1] - 1))
    Z = evecs[:, 1 : 1 + d_eff].astype(np.float32, copy=False)
    norms = np.linalg.norm(Z, axis=1)
    mask = norms > 0
    Z[mask] = (Z[mask].T / norms[mask]).T
    return Z
