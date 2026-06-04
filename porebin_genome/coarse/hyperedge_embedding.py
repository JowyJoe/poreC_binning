"""Feature-anchored hypergraph VAE embeddings for Pore-C contacts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from porebin_genome.coarse.hyperedge_weight import (
    DEFAULT_HYPERGRAPH_WEIGHT_ETA,
    hypergraph_native_weight,
    normalize_contact_weight_mode,
)
from porebin_genome.evidence.canonical import ContactEvidenceError, iter_canonical_contacts
from porebin_genome.io.runtime import write_json
from porebin_genome.io.tables import write_tsv_rows


DEFAULT_HYPEREDGE_EMBEDDING_DIM = 64
DEFAULT_HYPEREDGE_EMBEDDING_EPOCHS = 100
DEFAULT_HYPEREDGE_EMBEDDING_LR = 1.0e-3
DEFAULT_HYPEREDGE_EMBEDDING_PATIENCE = 12
DEFAULT_HYPEREDGE_FEATURE_GUARD = True
DEFAULT_HYPEREDGE_VAE_BETA = 0.01
DEFAULT_HYPEREDGE_VAE_LAMBDA = 1.0
DEFAULT_HYPEREDGE_VAE_BATCH_SIZE = 2048


class HyperedgeEmbeddingError(RuntimeError):
    """Raised when feature-anchored hypergraph VAE embedding cannot be trained."""


@dataclass(frozen=True)
class HyperedgeEmbeddingResult:
    """Outputs and diagnostics from feature-anchored hypergraph VAE training."""

    embedding_tsv: Path
    meta_json: Path
    n_contigs: int
    n_training_edges: int
    n_supported_contigs: int
    embedding_dim: int
    epochs_completed: int
    final_loss: float | None
    reconstruction_loss: float | None
    kl_loss: float | None
    hypergraph_loss: float | None
    feature_guard_enabled: bool
    feature_compatibility_scale: float | None
    mean_feature_compatibility: float | None


@dataclass(frozen=True)
class _TrainingEdge:
    contact_id: int
    members: object
    alpha: object
    weight: float
    base_weight: float
    feature_dispersion: float | None
    feature_compatibility: float


@dataclass(frozen=True)
class _TrainingEdgeCandidate:
    contact_id: int
    members: object
    alpha: object
    base_weight: float
    feature_dispersion: float | None


def train_hyperedge_embedding(
    *,
    contacts_path: Path,
    contig_name_to_idx: Mapping[str, int],
    idx_to_name: list[str],
    out_tsv: Path,
    meta_json: Path,
    contact_weight_mode: str = "hypergraph_native",
    hypergraph_weight_eta: float = DEFAULT_HYPERGRAPH_WEIGHT_ETA,
    alpha_min: float = 0.05,
    embedding_dim: int = DEFAULT_HYPEREDGE_EMBEDDING_DIM,
    epochs: int = DEFAULT_HYPEREDGE_EMBEDDING_EPOCHS,
    learning_rate: float = DEFAULT_HYPEREDGE_EMBEDDING_LR,
    patience: int = DEFAULT_HYPEREDGE_EMBEDDING_PATIENCE,
    feature_matrix: object | None = None,
    feature_guard: bool = DEFAULT_HYPEREDGE_FEATURE_GUARD,
    coverage_feature_present: bool = True,
    beta_kl: float = DEFAULT_HYPEREDGE_VAE_BETA,
    lambda_hypergraph: float = DEFAULT_HYPEREDGE_VAE_LAMBDA,
    hyperedge_batch_size: int = DEFAULT_HYPEREDGE_VAE_BATCH_SIZE,
    seed: int = 0,
) -> HyperedgeEmbeddingResult:
    """Train an unsupervised feature-anchored hypergraph VAE.

    The encoder learns latent contig vectors from TNF/coverage features. The
    decoder reconstructs those same features, while reliable Pore-C hyperedges
    regularize latent vectors toward their alpha-weighted hyperedge centroids.
    """
    if feature_matrix is None:
        raise HyperedgeEmbeddingError("HG-VAE embedding requires feature_matrix with TNF/coverage features.")
    if float(alpha_min) < 0.0:
        raise HyperedgeEmbeddingError("alpha_min must be non-negative.")
    if int(embedding_dim) < 1:
        raise HyperedgeEmbeddingError("embedding_dim must be positive.")
    if int(epochs) < 1:
        raise HyperedgeEmbeddingError("epochs must be positive.")
    if float(learning_rate) <= 0.0:
        raise HyperedgeEmbeddingError("learning_rate must be positive.")
    if int(patience) < 1:
        raise HyperedgeEmbeddingError("patience must be positive.")
    if float(beta_kl) < 0.0:
        raise HyperedgeEmbeddingError("beta_kl must be non-negative.")
    if float(lambda_hypergraph) < 0.0:
        raise HyperedgeEmbeddingError("lambda_hypergraph must be non-negative.")

    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise HyperedgeEmbeddingError("HG-VAE embedding requires numpy.") from exc
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover
        raise HyperedgeEmbeddingError(
            "HG-VAE embedding requires PyTorch. Recreate the environment from environment.yml "
            "or reinstall porebin so core dependencies are present."
        ) from exc

    contacts_path = contacts_path.resolve()
    out_tsv = out_tsv.resolve()
    meta_json = meta_json.resolve()
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    n_contigs = len(idx_to_name)
    if n_contigs != len(contig_name_to_idx):
        raise HyperedgeEmbeddingError("idx_to_name and contig_name_to_idx have inconsistent sizes.")

    X_np = np.asarray(feature_matrix, dtype=np.float32)
    if X_np.ndim != 2:
        raise HyperedgeEmbeddingError("feature_matrix must be two-dimensional.")
    if int(X_np.shape[0]) != n_contigs:
        raise HyperedgeEmbeddingError("feature_matrix row count must match idx_to_name.")
    X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    weight_mode = normalize_contact_weight_mode(contact_weight_mode)
    edge_candidates = _load_training_edge_candidates(
        contacts_path=contacts_path,
        contig_name_to_idx=contig_name_to_idx,
        contact_weight_mode=weight_mode,
        hypergraph_weight_eta=float(hypergraph_weight_eta),
        alpha_min=float(alpha_min),
        feature_matrix=X_np,
        coverage_feature_present=bool(coverage_feature_present),
    )
    edges, hyperedge_support, compatibility_scale = _finalize_training_edges(
        edge_candidates=edge_candidates,
        n_contigs=n_contigs,
        feature_guard_enabled=bool(feature_guard),
    )
    mean_feature_compatibility = _mean([edge.feature_compatibility for edge in edges]) if edges else None

    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    input_dim = int(X_np.shape[1])
    latent_dim = int(embedding_dim)
    hidden_dim = _default_hidden_dim(input_dim=input_dim, latent_dim=latent_dim)
    model = _build_hypergraph_vae(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    X = torch.as_tensor(X_np, dtype=torch.float32)

    losses: list[dict[str, float | int]] = []
    best_state: dict[str, object] | None = None
    best_loss = math.inf
    stale_epochs = 0
    epochs_completed = 0

    for epoch_idx in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        reconstruction, mean, logvar = model(X)
        reconstruction_loss = F.mse_loss(reconstruction, X, reduction="mean")
        kl_loss = -0.5 * torch.mean(torch.sum(1.0 + logvar - mean.pow(2) - logvar.exp(), dim=1))
        hg_loss = _torch_hypergraph_loss(
            z=mean,
            edges=edges,
            batch_size=int(hyperedge_batch_size),
            rng=rng,
        )
        total_loss = (
            reconstruction_loss
            + float(beta_kl) * kl_loss
            + float(lambda_hypergraph) * hg_loss
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        row = {
            "epoch": int(epoch_idx + 1),
            "loss": float(total_loss.detach().cpu()),
            "reconstruction_loss": float(reconstruction_loss.detach().cpu()),
            "kl_loss": float(kl_loss.detach().cpu()),
            "hypergraph_loss": float(hg_loss.detach().cpu()),
            "learning_rate": float(learning_rate),
        }
        losses.append(row)
        epochs_completed = int(epoch_idx + 1)
        if float(row["loss"]) + 1.0e-6 < best_loss:
            best_loss = float(row["loss"])
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(patience):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        mean, _logvar = model.encode(X)
    embedding = np.asarray(mean.detach().cpu().numpy(), dtype=np.float32)
    embedding = _normalize_rows(embedding)

    support = np.ones(n_contigs, dtype=np.float64) + np.asarray(hyperedge_support, dtype=np.float64)
    _write_embedding_tsv(out_tsv, idx_to_name=idx_to_name, embedding=embedding, support=support)

    last = losses[-1] if losses else {}
    result = HyperedgeEmbeddingResult(
        embedding_tsv=out_tsv,
        meta_json=meta_json,
        n_contigs=n_contigs,
        n_training_edges=len(edges),
        n_supported_contigs=int((support > 0.0).sum()),
        embedding_dim=latent_dim,
        epochs_completed=epochs_completed,
        final_loss=(float(last["loss"]) if "loss" in last else None),
        reconstruction_loss=(float(last["reconstruction_loss"]) if "reconstruction_loss" in last else None),
        kl_loss=(float(last["kl_loss"]) if "kl_loss" in last else None),
        hypergraph_loss=(float(last["hypergraph_loss"]) if "hypergraph_loss" in last else None),
        feature_guard_enabled=bool(feature_guard),
        feature_compatibility_scale=compatibility_scale,
        mean_feature_compatibility=mean_feature_compatibility,
    )
    _write_meta(
        meta_json,
        result=result,
        contacts_path=contacts_path,
        contact_weight_mode=weight_mode,
        hypergraph_weight_eta=float(hypergraph_weight_eta),
        alpha_min=float(alpha_min),
        epochs=int(epochs),
        learning_rate=float(learning_rate),
        patience=int(patience),
        beta_kl=float(beta_kl),
        lambda_hypergraph=float(lambda_hypergraph),
        hyperedge_batch_size=int(hyperedge_batch_size),
        hidden_dim=hidden_dim,
        input_dim=input_dim,
        feature_guard_enabled=result.feature_guard_enabled,
        feature_compatibility_scale=result.feature_compatibility_scale,
        mean_feature_compatibility=result.mean_feature_compatibility,
        seed=int(seed),
        losses=losses,
        warning=None if edges else "no hyperedges with at least two active known contigs; trained feature-only VAE",
    )
    return result


def load_hyperedge_embedding_tsv(path: Path) -> tuple[list[str], object, dict[str, float]]:
    """Load a hyperedge embedding TSV emitted by :func:`train_hyperedge_embedding`."""
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise HyperedgeEmbeddingError("Loading hyperedge embeddings requires numpy.") from exc

    path = path.resolve()
    with path.open("r", encoding="utf-8", newline="") as fh:
        header = fh.readline().rstrip("\r\n").split("\t")
        if len(header) < 3 or header[0] != "contig_id" or header[-1] != "support_weight":
            raise HyperedgeEmbeddingError(f"Invalid hyperedge embedding TSV header: {path}")
        dim_columns = header[1:-1]
        if not dim_columns or any(not name.startswith("z") for name in dim_columns):
            raise HyperedgeEmbeddingError(f"Invalid embedding dimension columns in {path}")
        contigs: list[str] = []
        vectors: list[list[float]] = []
        support: dict[str, float] = {}
        for line in fh:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != len(header):
                raise HyperedgeEmbeddingError(f"Malformed embedding row in {path}: {line[:80]!r}")
            contig_id = str(fields[0])
            contigs.append(contig_id)
            vectors.append([float(value) for value in fields[1:-1]])
            support[contig_id] = float(fields[-1])
    return contigs, np.asarray(vectors, dtype=np.float32), support


def _build_hypergraph_vae(*, input_dim: int, hidden_dim: int, latent_dim: int):
    import torch
    import torch.nn as nn

    class HypergraphVAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(int(input_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.ReLU(),
            )
            self.mean = nn.Linear(int(hidden_dim), int(latent_dim))
            self.logvar = nn.Linear(int(hidden_dim), int(latent_dim))
            self.decoder = nn.Sequential(
                nn.Linear(int(latent_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), int(hidden_dim)),
                nn.ReLU(),
                nn.Linear(int(hidden_dim), int(input_dim)),
            )

        def encode(self, x):
            hidden = self.encoder(x)
            logvar = self.logvar(hidden).clamp(min=-8.0, max=8.0)
            return self.mean(hidden), logvar

        def forward(self, x):
            mean, logvar = self.encode(x)
            if self.training:
                eps = torch.randn_like(mean)
                z = mean + eps * torch.exp(0.5 * logvar)
            else:
                z = mean
            return self.decoder(z), mean, logvar

    return HypergraphVAE()


def _default_hidden_dim(*, input_dim: int, latent_dim: int) -> int:
    return int(max(64, min(512, max(int(input_dim), int(latent_dim)) * 2)))


def _load_training_edge_candidates(
    *,
    contacts_path: Path,
    contig_name_to_idx: Mapping[str, int],
    contact_weight_mode: str,
    hypergraph_weight_eta: float,
    alpha_min: float,
    feature_matrix: object,
    coverage_feature_present: bool,
) -> list[_TrainingEdgeCandidate]:
    import numpy as np

    X = np.asarray(feature_matrix, dtype=np.float32)
    edges: list[_TrainingEdgeCandidate] = []
    try:
        for row_idx, row in enumerate(iter_canonical_contacts(contacts_path, require_contig_weights=True)):
            if row.contig_weights is None or row.k_valid < 2:
                continue
            active: list[tuple[int, float]] = []
            for contig_id, alpha in zip(row.contigs, row.contig_weights, strict=True):
                idx = contig_name_to_idx.get(str(contig_id))
                if idx is None or float(alpha) < float(alpha_min):
                    continue
                active.append((int(idx), float(alpha)))
            if len(active) < 2:
                continue
            contact_id = int(row.contact_id if row.contact_id is not None else row_idx)
            if str(contact_weight_mode) == "hypergraph_native":
                weight = hypergraph_native_weight(
                    read_weight=float(row.weight),
                    alpha_values=row.contig_weights,
                    eta=float(hypergraph_weight_eta),
                )
            else:
                weight = float(row.weight)
            if weight <= 0.0:
                continue
            members = np.asarray([idx for idx, _alpha in active], dtype=np.int64)
            alpha_values = np.asarray([alpha for _idx, alpha in active], dtype=np.float64)
            alpha_total = float(alpha_values.sum())
            if alpha_total <= 0.0:
                continue
            alpha_values = alpha_values / alpha_total
            feature_dispersion = _weighted_feature_dispersion(
                X=X,
                members=members,
                alpha=alpha_values,
                coverage_feature_present=bool(coverage_feature_present),
            )
            edges.append(
                _TrainingEdgeCandidate(
                    contact_id=contact_id,
                    members=members,
                    alpha=alpha_values.astype(np.float32),
                    base_weight=float(weight),
                    feature_dispersion=feature_dispersion,
                )
            )
    except ContactEvidenceError as exc:
        raise HyperedgeEmbeddingError(str(exc)) from exc
    return edges


def _finalize_training_edges(
    *,
    edge_candidates: list[_TrainingEdgeCandidate],
    n_contigs: int,
    feature_guard_enabled: bool,
) -> tuple[list[_TrainingEdge], object, float | None]:
    import numpy as np

    dispersions = [
        float(edge.feature_dispersion)
        for edge in edge_candidates
        if edge.feature_dispersion is not None and float(edge.feature_dispersion) > 0.0
    ]
    compatibility_scale = _median(dispersions) if feature_guard_enabled and dispersions else None

    edges: list[_TrainingEdge] = []
    support = np.zeros(n_contigs, dtype=np.float64)
    for candidate in edge_candidates:
        compatibility = _feature_compatibility(
            candidate.feature_dispersion,
            scale=compatibility_scale,
            enabled=feature_guard_enabled,
        )
        weight = float(candidate.base_weight) * float(compatibility)
        if weight <= 0.0:
            continue
        members = np.asarray(candidate.members, dtype=np.int64)
        alpha = np.asarray(candidate.alpha, dtype=np.float32)
        edges.append(
            _TrainingEdge(
                contact_id=int(candidate.contact_id),
                members=members,
                alpha=alpha,
                weight=float(weight),
                base_weight=float(candidate.base_weight),
                feature_dispersion=candidate.feature_dispersion,
                feature_compatibility=float(compatibility),
            )
        )
        support[members] += float(weight) * alpha.astype(np.float64)
    return edges, support, compatibility_scale


def _torch_hypergraph_loss(*, z, edges: list[_TrainingEdge], batch_size: int, rng: object):
    import numpy as np
    import torch

    if not edges:
        return z.sum() * 0.0

    if int(batch_size) > 0 and int(batch_size) < len(edges):
        weights = np.asarray([max(float(edge.weight), 0.0) for edge in edges], dtype=np.float64)
        total = float(weights.sum())
        probs = None if total <= 0.0 else weights / total
        edge_indices = rng.choice(len(edges), size=int(batch_size), replace=True, p=probs)
        selected = [edges[int(idx)] for idx in edge_indices.tolist()]
    else:
        selected = edges

    terms = []
    weight_sum = 0.0
    for edge in selected:
        members = torch.as_tensor(edge.members, dtype=torch.long, device=z.device)
        alpha = torch.as_tensor(edge.alpha, dtype=z.dtype, device=z.device)
        vectors = z.index_select(0, members)
        centroid = torch.sum(vectors * alpha[:, None], dim=0)
        dispersion = torch.sum(alpha * torch.sum((vectors - centroid) ** 2, dim=1))
        terms.append(float(edge.weight) * dispersion)
        weight_sum += float(edge.weight)
    if not terms or weight_sum <= 0.0:
        return z.sum() * 0.0
    return torch.stack(terms).sum() / float(weight_sum)


def _weighted_feature_dispersion(
    *,
    X: object,
    members: object,
    alpha: object,
    coverage_feature_present: bool,
) -> float:
    import numpy as np

    matrix = np.asarray(X, dtype=np.float32)[np.asarray(members, dtype=np.int64)]
    weights = np.asarray(alpha, dtype=np.float64)
    total = float(weights.sum())
    if matrix.size == 0 or total <= 0.0:
        return 0.0
    weights = weights / total
    centroid = np.average(matrix, axis=0, weights=weights)
    residual = matrix - centroid
    if bool(coverage_feature_present) and matrix.shape[1] > 1:
        composition = residual[:, :-1]
        coverage = residual[:, -1:]
        composition_dispersion = float(
            np.average(np.mean(composition * composition, axis=1), weights=weights)
        )
        coverage_dispersion = float(
            np.average(np.mean(coverage * coverage, axis=1), weights=weights)
        )
        return float(0.5 * composition_dispersion + 0.5 * coverage_dispersion)
    return float(np.average(np.mean(residual * residual, axis=1), weights=weights))


def _feature_compatibility(
    dispersion: float | None,
    *,
    scale: float | None,
    enabled: bool,
) -> float:
    if not enabled or dispersion is None or scale is None or float(scale) <= 0.0:
        return 1.0
    return float(1.0 / (1.0 + max(0.0, float(dispersion)) / float(scale)))


def _normalize_rows(matrix: object) -> object:
    import numpy as np

    values = np.asarray(matrix, dtype=np.float32).copy()
    norms = np.linalg.norm(values, axis=1)
    mask = norms > 0.0
    values[mask] = (values[mask].T / norms[mask]).T
    values[~np.isfinite(values)] = 0.0
    return values.astype(np.float32, copy=False)


def _write_embedding_tsv(
    path: Path,
    *,
    idx_to_name: list[str],
    embedding: object,
    support: object,
) -> None:
    import numpy as np

    matrix = np.asarray(embedding, dtype=np.float32)
    header = ["contig_id"] + [f"z{idx}" for idx in range(matrix.shape[1])] + ["support_weight"]
    support_values = np.asarray(support, dtype=np.float64)
    write_tsv_rows(
        path,
        header,
        (
            (
                str(contig_id),
                *[f"{float(value):.8g}" for value in matrix[idx]],
                f"{float(support_values[idx]):.8g}",
            )
            for idx, contig_id in enumerate(idx_to_name)
        ),
    )


def _write_meta(
    path: Path,
    *,
    result: HyperedgeEmbeddingResult,
    contacts_path: Path,
    contact_weight_mode: str,
    hypergraph_weight_eta: float,
    alpha_min: float,
    epochs: int,
    learning_rate: float,
    patience: int,
    beta_kl: float,
    lambda_hypergraph: float,
    hyperedge_batch_size: int,
    hidden_dim: int,
    input_dim: int,
    feature_guard_enabled: bool,
    feature_compatibility_scale: float | None,
    mean_feature_compatibility: float | None,
    seed: int,
    losses: list[dict[str, float | int]],
    warning: str | None,
) -> None:
    method = (
        "feature_guarded_feature_anchored_hypergraph_vae_embedding"
        if bool(feature_guard_enabled)
        else "feature_anchored_hypergraph_vae_embedding"
    )
    payload: dict[str, object] = {
        "stage": "hyperedge_embedding",
        "method": method,
        "inputs": {
            "contacts_parquet": str(contacts_path),
            "contact_weight_mode": str(contact_weight_mode),
            "feature_matrix": "TNF136 plus optional log1p coverage from coarse feature construction",
        },
        "outputs": {
            "hyperedge_embedding_tsv": str(result.embedding_tsv),
            "hyperedge_embedding_meta_json": str(result.meta_json),
        },
        "formula": {
            "feature_input": "x_i = [TNF136_i, log1p(coverage_i)] after z-score normalization",
            "vae": "z_i = Encoder(x_i), xhat_i = Decoder(z_i)",
            "reconstruction_loss": "L_rec = mean_i ||x_i - xhat_i||^2",
            "kl_loss": "L_kl = mean_i KL(q(z_i|x_i) || N(0,I))",
            "hyperedge_centroid": "mu_e = sum_i alpha_ie z_i",
            "feature_centroid": "xbar_e = sum_i alpha_ie x_i",
            "feature_dispersion": "d_e = sum_i alpha_ie ||x_i - xbar_e||^2",
            "feature_compatibility": "g_e = 1 / (1 + d_e / median_positive_d)",
            "hyperedge_loss": "L_hg = sum_e W_e g_e sum_i alpha_ie ||z_i - mu_e||^2",
            "total_loss": "L = L_rec + beta_kl * L_kl + lambda_hypergraph * L_hg",
        },
        "contact_weight_mode": str(contact_weight_mode),
        "hypergraph_weight_eta": float(hypergraph_weight_eta),
        "alpha_min": float(alpha_min),
        "feature_guard_enabled": bool(feature_guard_enabled),
        "feature_compatibility_scale": feature_compatibility_scale,
        "mean_feature_compatibility": mean_feature_compatibility,
        "input_dim": int(input_dim),
        "hidden_dim": int(hidden_dim),
        "embedding_dim": int(result.embedding_dim),
        "epochs_requested": int(epochs),
        "epochs_completed": int(result.epochs_completed),
        "learning_rate": float(learning_rate),
        "patience": int(patience),
        "beta_kl": float(beta_kl),
        "lambda_hypergraph": float(lambda_hypergraph),
        "hyperedge_batch_size": int(hyperedge_batch_size),
        "seed": int(seed),
        "n_contigs": int(result.n_contigs),
        "n_training_edges": int(result.n_training_edges),
        "n_supported_contigs": int(result.n_supported_contigs),
        "final_loss": result.final_loss,
        "reconstruction_loss": result.reconstruction_loss,
        "kl_loss": result.kl_loss,
        "hypergraph_loss": result.hypergraph_loss,
        "losses": losses,
    }
    if warning is not None:
        payload["warning"] = str(warning)
    write_json(path, payload)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    xs = sorted(float(value) for value in values)
    mid = len(xs) // 2
    if len(xs) % 2:
        return float(xs[mid])
    return float(0.5 * (xs[mid - 1] + xs[mid]))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / float(len(values)))
