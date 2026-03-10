from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from porebin.utils import iter_fasta_records
from porebin.tnf_constants import CANON_TO_IDX, TNF136_LIST, build_idx256_to_idx136, canonical_4mer


class JointSpectralError(RuntimeError):
    pass


@dataclass(frozen=True)
class ContactComponentPostprocessMeta:
    components_total: int
    components_promoted_from_noise: int
    noise_reassigned_by_component: int
    labels_split_by_component: int


def postprocess_labels_by_contact_components(
    labels: "object",
    contact_H_csr: "object",
    *,
    min_cluster_size: int,
) -> tuple["object", ContactComponentPostprocessMeta]:
    """
    Postprocess HDBSCAN labels using the contact hypergraph connectivity.

    Goals:
      1) Prevent a single bin/label from spanning multiple disconnected contact components.
      2) If HDBSCAN marks vertices as noise (-1) inside a contact-connected component that otherwise has an assigned label,
         reassign those noise vertices to the component's majority non-noise label.
      3) If an entire contact-connected component is noise but its size >= min_cluster_size, promote it to a new label.

    This does not construct a dense |V|x|V| matrix and only uses hyperedge membership for connectivity.
    """
    import numpy as np

    labels = np.asarray(labels, dtype=int).copy()
    H = contact_H_csr
    V = int(H.shape[0])
    if labels.shape[0] != V:
        raise JointSpectralError("Internal error: labels length mismatch in contact-component postprocess.")

    # Build contact-connected components via union-find over hyperedges.
    Hcsc = H.tocsc()
    parent = np.arange(V, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    indptr = Hcsc.indptr
    indices = Hcsc.indices
    for e in range(int(Hcsc.shape[1])):
        start = int(indptr[e])
        end = int(indptr[e + 1])
        if end - start < 2:
            continue
        vs = indices[start:end]
        root = int(vs[0])
        for v in vs[1:]:
            union(root, int(v))

    roots = np.asarray([find(i) for i in range(V)], dtype=np.int32)
    uniq_roots, inv = np.unique(roots, return_inverse=True)
    comp_id = inv  # 0..C-1
    num_components = int(uniq_roots.shape[0])

    promoted = 0
    reassigned = 0

    # Per-component majority non-noise label (if any)
    # (Component ids are 0..C-1, small; use python lists for speed.)
    comp_members: list[list[int]] = [[] for _ in range(num_components)]
    for v in range(V):
        comp_members[int(comp_id[v])].append(v)

    next_label = int(labels.max(initial=-1)) + 1
    for cid in range(num_components):
        members = comp_members[cid]
        if not members:
            continue
        labs = labels[members]
        non_noise = labs[labs != -1]
        majority = None
        if non_noise.size > 0:
            vals, counts = np.unique(non_noise, return_counts=True)
            majority = int(vals[int(np.argmax(counts))])
        else:
            if len(members) >= int(min_cluster_size):
                majority = next_label
                next_label += 1
                promoted += 1

        if majority is None:
            continue
        # Reassign noise vertices within this component to the majority label.
        for v in members:
            if int(labels[v]) == -1:
                labels[v] = int(majority)
                reassigned += 1

    # Finally, split labels by contact component id so bins do not span disconnected components.
    mapping: dict[tuple[int, int], int] = {}
    split_count = 0
    out = labels.copy()
    for v in range(V):
        lab = int(labels[v])
        if lab == -1:
            continue
        key = (lab, int(comp_id[v]))
        new = mapping.get(key)
        if new is None:
            new = len(mapping)
            mapping[key] = new
        out[v] = new
    split_count = len(mapping)

    return out, ContactComponentPostprocessMeta(
        components_total=num_components,
        components_promoted_from_noise=int(promoted),
        noise_reassigned_by_component=int(reassigned),
        labels_split_by_component=int(split_count),
    )


def load_contig_index(graph_dir: Path) -> tuple[dict[str, int], list[str]]:
    """
    Load contig index mapping for v2 joint spectral clustering.

    Required file format: graph/contigs.tsv with rows:
      contig_idx<TAB>contig_name
    (no header). contig_idx must match build_graph contig indices (0..|V|-1).

    Returns:
      contig_name_to_idx, idx_to_name
    """
    graph_dir = graph_dir.resolve()
    contigs_tsv = graph_dir / "contigs.tsv"
    if not contigs_tsv.exists():
        # Backwards-compat fallback: derive from contig_index.tsv without writing anything.
        contig_index = graph_dir / "contig_index.tsv"
        if not contig_index.exists():
            raise JointSpectralError(f"Missing contig index files: {contigs_tsv} (and no fallback {contig_index})")
        mapping: dict[str, int] = {}
        idx_to_name: list[str] = []
        with contig_index.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh, delimiter="\t")
            for row in reader:
                if not row:
                    continue
                if row[0] in {"contig_name", "contig"}:
                    continue
                if len(row) < 2:
                    continue
                name = row[0].strip()
                idx = int(row[1])
                mapping[name] = idx
        if not mapping:
            raise JointSpectralError(f"No contigs found in {contig_index}")
        idx_to_name = [None] * (max(mapping.values()) + 1)
        for name, idx in mapping.items():
            idx_to_name[idx] = name
        if any(x is None for x in idx_to_name):
            raise JointSpectralError("Non-contiguous contig indices in contig_index.tsv fallback.")
        return mapping, list(idx_to_name)

    mapping: dict[str, int] = {}
    idx_to_name: list[str] = []
    with contigs_tsv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if len(row) < 2:
                raise JointSpectralError(f"Invalid contigs.tsv row in {contigs_tsv}: {row}")
            idx = int(row[0])
            name = row[1].strip()
            if idx != len(idx_to_name):
                raise JointSpectralError(
                    f"contigs.tsv must be 0..|V|-1 in order. Got idx={idx}, expected={len(idx_to_name)}."
                )
            idx_to_name.append(name)
            mapping[name] = idx
    if not idx_to_name:
        raise JointSpectralError(f"No contigs found in {contigs_tsv}")
    return mapping, idx_to_name


@dataclass(frozen=True)
class ContactIncidence:
    H_csr: "object"  # scipy.sparse.csr_matrix
    W: "object"  # np.ndarray float64, shape (E,)
    De: "object"  # np.ndarray float64, shape (E,)
    Dv: "object"  # np.ndarray float64, shape (V,)
    dropped_edges_singleton_contact: int
    isolated_mask_contact: "object"  # np.ndarray bool, shape (V,)


def build_contact_incidence_from_parquet(
    contacts_path: Path,
    contig_name_to_idx: dict[str, int],
    *,
    parquet_batch_size: int = 200_000,
    logger: Optional[logging.Logger] = None,
) -> ContactIncidence:
    """
    Build contact hypergraph incidence Hc and weights Wc from contacts.parquet.

    Definitions (must match spec):
      - Hc[v,e] = P_{e,v}  (soft incidence from contig_weights)
      - Drop contacts with k<2, count dropped_edges_singleton_contact
      - Wc[e] = q(e) / (k - 1), where q(e) is contacts.parquet.weight
      - Dec[e] = sum_v Hc[v,e]
      - Dvc[v] = sum_e Wc[e] * Hc[v,e]
    """
    logger = logger or logging.getLogger("porebin")
    contacts_path = contacts_path.resolve()
    if not contacts_path.exists():
        raise FileNotFoundError(f"Contacts Parquet not found: {contacts_path}")

    try:
        import numpy as np
        import pyarrow.parquet as pq
        import scipy.sparse as sp
    except Exception as exc:  # pragma: no cover
        raise JointSpectralError(
            "joint spectral clustering requires optional deps: pyarrow + scipy. "
            "Install them (e.g. pip install 'porebin[spectral]' or conda install scipy)."
        ) from exc

    V = len(contig_name_to_idx)
    dropped = 0
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    W: list[float] = []
    De: list[float] = []

    parquet = pq.ParquetFile(contacts_path)
    schema = parquet.schema_arrow
    cols_present = set(schema.names)
    for req in ("contigs", "contig_weights", "k", "weight"):
        if req not in cols_present:
            raise JointSpectralError(f"contacts.parquet missing required column {req!r}. Found: {schema.names}")

    edge_idx = 0
    logger.info("JointSpectral: reading contacts.parquet: %s", contacts_path)
    for batch in parquet.iter_batches(batch_size=int(parquet_batch_size), columns=["contigs", "contig_weights", "k", "weight"]):
        b = batch.to_pydict()
        contigs_list = b["contigs"]
        weights_list = b["contig_weights"]
        k_list = b["k"]
        q_list = b["weight"]
        n = len(contigs_list)
        for i in range(n):
            contigs_raw = contigs_list[i] or []
            p_raw = weights_list[i] or []
            k = int(k_list[i]) if k_list[i] is not None else len(contigs_raw)
            if k < 2:
                dropped += 1
                continue
            if not isinstance(contigs_raw, list) or not isinstance(p_raw, list) or len(contigs_raw) != len(p_raw):
                continue

            q = float(q_list[i]) if q_list[i] is not None else 1.0
            Wc = float(q) / float(k - 1)
            # accumulate Hc entries for this edge
            dec = 0.0
            for c, p in zip(contigs_raw, p_raw, strict=True):
                name = str(c)
                if not name:
                    continue
                idx = contig_name_to_idx.get(name)
                if idx is None:
                    raise JointSpectralError(f"Contig {name!r} in contacts.parquet not found in contigs.tsv mapping.")
                pv = float(p)
                if pv <= 0.0:
                    continue
                rows.append(int(idx))
                cols.append(int(edge_idx))
                data.append(float(pv))
                dec += float(pv)
            if not (dec > 0.0):
                dropped += 1
                continue

            W.append(float(Wc))
            De.append(float(dec))
            edge_idx += 1

    E = len(W)
    if E == 0:
        raise JointSpectralError("No usable contact hyperedges (k>=2) in contacts.parquet.")

    H = sp.coo_matrix((np.asarray(data, dtype=float), (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int))), shape=(V, E)).tocsr()
    W_arr = np.asarray(W, dtype=float)
    De_arr = np.asarray(De, dtype=float)
    Dv_arr = np.asarray(H @ W_arr).ravel()
    isolated = Dv_arr <= 0.0
    return ContactIncidence(
        H_csr=H,
        W=W_arr,
        De=De_arr,
        Dv=Dv_arr,
        dropped_edges_singleton_contact=int(dropped),
        isolated_mask_contact=isolated,
    )


def compute_tetranuc_features_from_fasta(
    contigs_fasta: Path,
    contig_name_to_idx: dict[str, int],
    *,
    logger: Optional[logging.Logger] = None,
) -> "object":
    """
    4-mer composition (256D) without reverse-complement collapsing.
    Only count 4-mers fully in {A,C,G,T}; ignore any window containing other chars.
    Output is per-contig frequency: cnt / max(1, sum(cnt)), stored as float32.
    """
    logger = logger or logging.getLogger("porebin")
    contigs_fasta = contigs_fasta.resolve()
    if not contigs_fasta.exists():
        raise FileNotFoundError(f"Contigs FASTA not found: {contigs_fasta}")

    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise JointSpectralError("joint spectral clustering requires numpy (via scipy).") from exc

    V = len(contig_name_to_idx)
    X = np.zeros((V, 256), dtype=np.float32)

    base = {"A": 0, "C": 1, "G": 2, "T": 3}
    seen = 0
    for name, _header, seq in iter_fasta_records(contigs_fasta):
        idx = contig_name_to_idx.get(name)
        if idx is None:
            continue
        seen += 1
        s = (seq or "").upper()
        cnt = [0] * 256
        total = 0
        if len(s) >= 4:
            for i in range(len(s) - 3):
                a = base.get(s[i])
                b = base.get(s[i + 1])
                c = base.get(s[i + 2])
                d = base.get(s[i + 3])
                if a is None or b is None or c is None or d is None:
                    continue
                k = (((a * 4) + b) * 4 + c) * 4 + d
                cnt[k] += 1
                total += 1
        denom = float(total) if total > 0 else 1.0
        X[idx, :] = np.asarray(cnt, dtype=np.float32) / denom

    if seen == 0:
        raise JointSpectralError(f"No contigs from contigs.tsv were found in FASTA: {contigs_fasta}")
    logger.info("JointSpectral: tetranuc features computed for %s contigs", seen)
    return X


TNF136_ORDER_TAG = "TNF136_LIST_v1"


def compute_tnf136_features_from_fasta(
    contigs_fasta: Path,
    contig_name_to_idx: dict[str, int],
    *,
    logger: Optional[logging.Logger] = None,
) -> "object":
    """
    Canonical TNF (136D) with a fixed column order (TNF136_LIST).

    Counting definition:
      - Slide a 4-bp window across the contig sequence (upper-cased).
      - Ignore windows containing any char not in {A,C,G,T}.
      - Let canonical_4mer(s)=min(s, revcomp(s)).
      - Increment counts[CANON_TO_IDX[canonical_4mer(s)]] by 1.
      - Normalize to frequency: cnt / max(1, sum(cnt)).

    Output float32 of shape (|V|, 136).
    """
    logger = logger or logging.getLogger("porebin")
    contigs_fasta = contigs_fasta.resolve()
    if not contigs_fasta.exists():
        raise FileNotFoundError(f"Contigs FASTA not found: {contigs_fasta}")

    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise JointSpectralError("joint spectral clustering requires numpy (via scipy).") from exc

    V = len(contig_name_to_idx)
    X = np.zeros((V, 136), dtype=np.float32)

    seen = 0
    for name, _header, seq in iter_fasta_records(contigs_fasta):
        idx = contig_name_to_idx.get(name)
        if idx is None:
            continue
        seen += 1
        s = (seq or "").upper()
        cnt = np.zeros((136,), dtype=np.float32)
        total = 0
        if len(s) >= 4:
            for i in range(len(s) - 3):
                kmer = s[i : i + 4]
                if any(c not in "ACGT" for c in kmer):
                    continue
                canon = canonical_4mer(kmer)
                cnt[int(CANON_TO_IDX[canon])] += 1.0
                total += 1
        denom = float(total) if total > 0 else 1.0
        X[idx, :] = cnt / denom

    if seen == 0:
        raise JointSpectralError(f"No contigs from contigs.tsv were found in FASTA: {contigs_fasta}")
    logger.info("JointSpectral: TNF136 features computed for %s contigs", seen)
    return X


def _read_tnf136_meta(meta_path: Path) -> Optional[dict]:
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_tnf136_meta(meta_path: Path) -> None:
    meta_path.write_text(
        json.dumps({"tnf_order": TNF136_ORDER_TAG, "tnf_dim": 136}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_or_build_tnf136_features(
    *,
    graph_dir: Path,
    contigs_fasta: Path,
    contig_name_to_idx: dict[str, int],
    logger: Optional[logging.Logger] = None,
) -> "object":
    """
    Cache-safe TNF136 feature loader/builder.

    Files:
      - graph/features_tnf136.npy
      - graph/tnf136_meta.json: {"tnf_order":"TNF136_LIST_v1","tnf_dim":136}

    If meta is missing or mismatched, warn and rebuild the npy then write meta.
    Rebuild strategy:
      - If graph/features_tnf256.npy exists and is (V,256), convert 256->136 using TNF136_LIST order.
      - Else compute TNF136 directly from FASTA.
    """
    logger = logger or logging.getLogger("porebin")

    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise JointSpectralError("joint spectral clustering requires numpy (via scipy).") from exc

    graph_dir = graph_dir.resolve()
    features_path = graph_dir / "features_tnf136.npy"
    meta_path = graph_dir / "tnf136_meta.json"

    meta = _read_tnf136_meta(meta_path) if meta_path.exists() else None
    meta_ok = bool(meta) and meta.get("tnf_order") == TNF136_ORDER_TAG and int(meta.get("tnf_dim", -1)) == 136

    if meta_ok and features_path.exists():
        X = np.load(features_path)
        if X.ndim == 2 and X.shape[1] == 136 and X.shape[0] == len(contig_name_to_idx):
            return X.astype(np.float32, copy=False)
        logger.warning("TNF136 cache shape mismatch; rebuilding: %s", features_path)

    if not meta_ok:
        logger.warning("TNF136 meta missing/mismatched; rebuilding features_tnf136.npy: %s", meta_path)

    # Rebuild.
    features_256_path = graph_dir / "features_tnf256.npy"
    X136 = None
    if features_256_path.exists():
        try:
            X256 = np.load(features_256_path)
            if X256.ndim == 2 and X256.shape[1] == 256 and X256.shape[0] == len(contig_name_to_idx):
                idx256_to_136 = np.asarray(build_idx256_to_idx136(), dtype=np.int32)
                X136 = np.zeros((X256.shape[0], 136), dtype=np.float32)
                for j in range(256):
                    X136[:, int(idx256_to_136[j])] += X256[:, j].astype(np.float32, copy=False)
        except Exception:
            X136 = None

    if X136 is None:
        X136 = compute_tnf136_features_from_fasta(contigs_fasta, contig_name_to_idx, logger=logger)

    features_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(features_path, np.asarray(X136, dtype=np.float32))
    _write_tnf136_meta(meta_path)
    return np.asarray(X136, dtype=np.float32)

def load_coverage_feature_optional(
    coverage_tsv: Optional[Path],
    contig_name_to_idx: dict[str, int],
) -> tuple["object", int, bool]:
    """
    coverage feature: cov_feat = log1p(cov).
    If coverage is missing or file absent, use 0 and count missing.
    """
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise JointSpectralError("joint spectral clustering requires numpy (via scipy).") from exc

    V = len(contig_name_to_idx)
    x = np.zeros((V,), dtype=np.float32)
    if coverage_tsv is None or not coverage_tsv.exists():
        return x, V, False

    present = 0
    with coverage_tsv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if row[0] in {"contig_name", "contig"}:
                continue
            if len(row) < 2:
                continue
            name = row[0].strip()
            idx = contig_name_to_idx.get(name)
            if idx is None:
                continue
            try:
                cov = float(row[1])
            except ValueError:
                continue
            x[idx] = float(math.log1p(max(0.0, cov)))
            present += 1

    missing = V - present
    return x, int(missing), True


def zscore_features(X: "object") -> "object":
    """
    Per-dimension z-score. If variance is 0, that dimension is set to 0.
    Output float32.
    """
    import numpy as np

    X = np.asarray(X, dtype=np.float32)
    mu = X.mean(axis=0, dtype=np.float64)
    var = ((X.astype(np.float64) - mu) ** 2).mean(axis=0)
    std = np.sqrt(var)
    out = np.zeros_like(X, dtype=np.float32)
    mask = std > 0
    out[:, mask] = ((X[:, mask].astype(np.float64) - mu[mask]) / std[mask]).astype(np.float32)
    return out


def build_feature_knn_edges(X: "object", knn_k: int) -> "object":
    """
    Compute kNN (excluding self) using sklearn.neighbors.NearestNeighbors.
    Returns indices array of shape (V, k_eff), where k_eff=min(knn_k, V-1).
    """
    try:
        import numpy as np
        from sklearn.neighbors import NearestNeighbors
    except Exception as exc:  # pragma: no cover
        raise JointSpectralError(
            "feature hypergraph requires scikit-learn (sklearn). Install it, e.g. pip install scikit-learn."
        ) from exc

    X = np.asarray(X, dtype=np.float32)
    V = int(X.shape[0])
    if V <= 1:
        return np.zeros((V, 0), dtype=np.int32)

    k_eff = int(min(int(knn_k), V - 1))
    nn = NearestNeighbors(metric="euclidean", n_neighbors=int(k_eff) + 1)
    nn.fit(X)
    idx = nn.kneighbors(X, return_distance=False)
    return np.asarray(idx[:, 1:], dtype=np.int32)


@dataclass(frozen=True)
class FeatureIncidence:
    H_csr: "object"
    W: "object"
    De: "object"
    Dv: "object"


def build_feature_incidence(neighbors: "object", knn_k: int) -> FeatureIncidence:
    """
    Build feature hypergraph incidence:
      - one hyperedge per vertex v (edge_id == v)
      - e_v = {v} ∪ kNN(v)
      - Hf[u, e_v] = 1 iff u in e_v
      - Wf[e] = 1
      - Def[e] = sum_v Hf[v,e]
      - Dvf[v] = sum_e Hf[v,e]
    """
    try:
        import numpy as np
        import scipy.sparse as sp
    except Exception as exc:  # pragma: no cover
        raise JointSpectralError("feature hypergraph requires scipy.") from exc

    neigh = np.asarray(neighbors, dtype=np.int32)
    V = int(neigh.shape[0])
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for v in range(V):
        members = {int(v)}
        for u in neigh[v].tolist():
            uu = int(u)
            if uu < 0 or uu >= V or uu == v:
                continue
            members.add(uu)
        for u in members:
            rows.append(int(u))
            cols.append(int(v))
            data.append(1.0)

    H = sp.coo_matrix((np.asarray(data, dtype=float), (np.asarray(rows, dtype=int), np.asarray(cols, dtype=int))), shape=(V, V)).tocsr()
    W = np.ones((V,), dtype=float)
    De = np.asarray(H.sum(axis=0)).ravel()
    Dv = np.asarray(H.sum(axis=1)).ravel()
    return FeatureIncidence(H_csr=H, W=W, De=De, Dv=Dv)


@dataclass(frozen=True)
class ThetaOperator:
    op: "object"  # scipy.sparse.linalg.LinearOperator
    Dv_inv_sqrt: "object"  # np.ndarray float64
    isolated_mask: "object"  # np.ndarray bool


def make_theta_operator(H_csr: "object", W: "object", De: "object", Dv: "object") -> ThetaOperator:
    """
    Make a LinearOperator for Theta = Dv^{-1/2} * H * diag(W) * diag(De^{-1}) * H^T * Dv^{-1/2},
    implemented via matvec without constructing Theta explicitly.
    """
    import numpy as np
    import scipy.sparse.linalg as spla

    H = H_csr
    W = np.asarray(W, dtype=float)
    De = np.asarray(De, dtype=float)
    Dv = np.asarray(Dv, dtype=float)

    if H.shape[1] != W.shape[0] or H.shape[1] != De.shape[0]:
        raise JointSpectralError("Internal error: H column dimension mismatch with W/De.")
    if H.shape[0] != Dv.shape[0]:
        raise JointSpectralError("Internal error: H row dimension mismatch with Dv.")

    Dv_inv_sqrt = np.zeros_like(Dv, dtype=float)
    nz = Dv > 0
    Dv_inv_sqrt[nz] = 1.0 / np.sqrt(Dv[nz])
    isolated = ~nz

    De_safe = np.zeros_like(De, dtype=float)
    nz_e = De > 0
    De_safe[nz_e] = 1.0 / De[nz_e]

    scale = W * De_safe  # elementwise W/De

    def matvec(x):
        x = np.asarray(x, dtype=float)
        y = Dv_inv_sqrt * x
        z = H.T @ y
        z = z * scale
        y2 = H @ z
        return Dv_inv_sqrt * np.asarray(y2).ravel()

    op = spla.LinearOperator((H.shape[0], H.shape[0]), matvec=matvec, dtype=float)
    return ThetaOperator(op=op, Dv_inv_sqrt=Dv_inv_sqrt, isolated_mask=isolated)


def _auto_d(V: int) -> int:
    if V <= 0:
        return 0
    d = min(128, max(32, int(math.floor(math.log2(float(V)))) * 4))
    d = min(d, max(1, V - 2))
    return int(d)


def spectral_embed_joint(
    *,
    contact_op: "object",
    feature_op: "object",
    lambda_contact: float,
    d: int,
    seed: int,
) -> "object":
    """
    Compute joint spectral embedding:
      - joint = lambda*Thetac + (1-lambda)*Thetaf
      - eigsh on joint_op for top (d+1) eigenvectors (which='LA')
      - drop first eigenvector, keep d dims
      - L2-normalize each row
    """
    try:
        import numpy as np
        import scipy.sparse.linalg as spla
    except Exception as exc:  # pragma: no cover
        raise JointSpectralError("spectral embedding requires scipy.") from exc

    V = int(contact_op.shape[0])
    if int(feature_op.shape[0]) != V:
        raise JointSpectralError("Internal error: contact/feature operator shape mismatch.")

    lam = float(lambda_contact)
    if not (0.0 <= lam <= 1.0):
        raise JointSpectralError(f"lambda_contact must be in [0,1], got {lambda_contact}")

    k = int(min(int(d) + 1, V - 1))
    if k < 2:
        raise JointSpectralError(f"Too few contigs for joint spectral embedding (V={V}).")

    def matvec(x):
        return lam * (contact_op @ x) + (1.0 - lam) * (feature_op @ x)

    joint = spla.LinearOperator((V, V), matvec=matvec, dtype=float)
    rng = np.random.default_rng(int(seed))
    v0 = rng.standard_normal(V)
    evals, evecs = spla.eigsh(joint, k=k, which="LA", v0=v0)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order]

    # Drop the first (trivial) vector, keep up to d dims.
    d_eff = min(int(d), evecs.shape[1] - 1)
    Z = evecs[:, 1 : 1 + d_eff].astype(np.float32, copy=False)

    # Row-wise L2 normalization
    norms = np.linalg.norm(Z, axis=1)
    nz = norms > 0
    Z[nz] = (Z[nz].T / norms[nz]).T
    return Z


def hdbscan_cluster(Z: "object", *, min_cluster_size: int, threads: int) -> tuple["object", dict]:
    """
    Cluster embedding with HDBSCAN.
    Returns (labels, meta)
    """
    import numpy as np

    try:
        import hdbscan
    except Exception as exc:  # pragma: no cover
        raise JointSpectralError(
            "spectral clustering requires the 'hdbscan' package. Install it, e.g. pip install hdbscan."
        ) from exc

    Z = np.asarray(Z, dtype=np.float32)
    if Z.ndim != 2:
        raise JointSpectralError("Embedding must be 2D.")

    # Use leaf selection to avoid artificially limiting the number of clusters in coarse binning.
    # This is still fully unsupervised (no fixed K) and works well for many small bins.
    kwargs = {
        "min_cluster_size": int(min_cluster_size),
        "min_samples": None,
        "metric": "euclidean",
        "cluster_selection_method": "leaf",
    }
    # hdbscan supports parallelization via core_dist_n_jobs
    if int(threads) > 1:
        kwargs["core_dist_n_jobs"] = int(threads)
    clusterer = hdbscan.HDBSCAN(**kwargs)
    labels = clusterer.fit_predict(Z)
    version = getattr(hdbscan, "__version__", None)
    if version is None:  # pragma: no cover
        try:
            import importlib.metadata as ilm

            version = ilm.version("hdbscan")
        except Exception:
            version = None
    meta = {"impl": "hdbscan", "version": version, **kwargs}
    return labels, meta


def write_bins_tsv(out_bins_tsv: Path, idx_to_name: list[str], labels: "object") -> tuple[int, int]:
    """
    Write bins.tsv as contig_name -> bin_id with bin_id remapped to 0..num_bins-1.
    Returns (num_bins, unbinned_count).
    """
    import numpy as np

    out_bins_tsv = out_bins_tsv.resolve()
    labels = np.asarray(labels, dtype=int)
    if labels.shape[0] != len(idx_to_name):
        raise JointSpectralError("labels length mismatch with idx_to_name.")

    uniq = sorted({int(x) for x in labels.tolist() if int(x) != -1})
    remap = {old: new for new, old in enumerate(uniq)}
    unbinned = int(np.sum(labels == -1))

    out_bins_tsv.parent.mkdir(parents=True, exist_ok=True)
    with out_bins_tsv.open("w", encoding="utf-8", newline="") as fh:
        fh.write("contig_name\tbin_id\n")
        for i, name in enumerate(idx_to_name):
            lab = int(labels[i])
            if lab == -1:
                continue
            fh.write(f"{name}\t{remap[lab]}\n")
    return len(uniq), unbinned


def json_dumps_small(obj) -> str:
    return json.dumps(obj, indent=2, sort_keys=True)
