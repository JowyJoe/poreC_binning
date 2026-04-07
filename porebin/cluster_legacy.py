from __future__ import annotations

import csv
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

from porebin.cluster import GraphClusterError
from porebin.export import MIN_BIN_BP
from porebin.utils import iter_fasta_records


def _choose_k_by_eigengap(evals, *, conservative: bool) -> tuple[int, dict]:
    """
    Pick K clusters from ascending Laplacian eigenvalues via eigengap.

    conservative=True chooses the *largest* K whose eigengap is close to the maximum gap,
    which tends to avoid overly coarse (potentially mixed) clusters.
    """
    import numpy as np

    vals = np.asarray(evals, dtype=float)
    if vals.size < 3:
        return 2, {"method": "fallback_small_n", "n_eigs": int(vals.size)}

    gaps = np.diff(vals)
    start = 1
    if gaps.size <= start:
        return 2, {"method": "fallback_no_gaps", "n_eigs": int(vals.size)}

    usable = gaps[start:]
    max_gap = float(np.max(usable))
    if max_gap <= 0:
        return 2, {"method": "fallback_nonpos_gap", "max_gap": max_gap, "n_eigs": int(vals.size)}

    ratio = 0.8 if conservative else 1.0
    candidates = np.where(gaps >= (ratio * max_gap))[0]
    candidates = candidates[candidates >= start]
    if candidates.size == 0:
        i = int(np.argmax(usable) + start)
        return i + 1, {"method": "eigengap_argmax", "gap_ratio": ratio, "max_gap": max_gap, "n_eigs": int(vals.size)}

    i = int(np.max(candidates) if conservative else np.min(candidates))
    return i + 1, {
        "method": "eigengap_conservative" if conservative else "eigengap",
        "gap_ratio": ratio,
        "max_gap": max_gap,
        "n_eigs": int(vals.size),
    }


def _read_contig_lengths(contigs_fasta: Path) -> dict[str, int]:
    lengths: dict[str, int] = {}
    for name, _header, seq in iter_fasta_records(contigs_fasta):
        lengths[name] = len(seq)
    return lengths


def _auto_min_contig_len(lengths) -> tuple[int, dict]:
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise GraphClusterError("spectral clustering requires numpy (via scipy).") from exc

    vals = np.asarray(lengths, dtype=int)
    total_bp = int(np.sum(vals))
    bp_1000_2500 = int(np.sum(vals[(vals >= 1000) & (vals <= 2500)]))
    ratio = (bp_1000_2500 / total_bp) if total_bp else 0.0
    min_len = 1000 if ratio >= 0.05 else 2500
    meta = {
        "type": "semiBin_ratio_1000_2500",
        "ratio_1000_2500_bp": ratio,
        "ratio_threshold": 0.05,
        "rule": "min_contig_len=1000 if ratio>=0.05 else 2500",
    }
    return min_len, meta


def _coverage_from_bam(
    bam_path: Path,
    *,
    contig_names: list[str],
    contig_len: dict[str, int],
    min_contig_len: int,
    logger: logging.Logger,
) -> dict[str, float]:
    """
    Coverage proxy (fast, no per-base depth): aligned_bases_sum / contig_length.

    This mirrors refine's coverage computation, but is kept local to clustering so spectral can use BAM
    as an automatic stopping signal.
    """
    try:
        import pysam
    except Exception as exc:  # pragma: no cover
        raise GraphClusterError(
            "spectral clustering with BAM requires 'pysam'. Install it (pip install 'porebin[bam]' or conda install pysam)."
        ) from exc

    logger.info("Spectral clustering: computing coverage from BAM (aligned_bases_sum/contig_length)")
    bam = pysam.AlignmentFile(str(bam_path), "rb")
    ref_names = list(bam.references)
    aligned_bases: defaultdict[str, int] = defaultdict(int)
    for aln in bam.fetch(until_eof=True):
        if aln.is_unmapped:
            continue
        rid = aln.reference_id
        if rid is None or rid < 0 or rid >= len(ref_names):
            continue
        ref = ref_names[rid]
        aln_len = aln.query_alignment_length
        if not aln_len and aln.reference_end is not None and aln.reference_start is not None:
            aln_len = aln.reference_end - aln.reference_start
        if not aln_len:
            continue
        aligned_bases[ref] += int(aln_len)
    bam.close()

    cov: dict[str, float] = {}
    for name in contig_names:
        L = contig_len.get(name, 0)
        if L < min_contig_len:
            continue
        cov[name] = aligned_bases.get(name, 0) / float(L) if L else 0.0
    return cov


def _coverage_from_tsv(
    path: Path,
    *,
    contig_len: dict[str, int],
    min_contig_len: int,
) -> dict[str, float]:
    cov: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if row[0] in {"contig_name", "contig"}:
                continue
            if len(row) < 2:
                continue
            name = row[0].strip()
            if contig_len.get(name, 0) < min_contig_len:
                continue
            try:
                cov[name] = float(row[1])
            except ValueError:
                continue
    return cov


def _graph_bic_trigger(A_sub, *, left, right) -> tuple[bool, dict]:
    """
    Decide whether to split based on a simple 1-block vs 2-block Bernoulli graph model (BIC) on edge presence.
    """
    import numpy as np

    A_sub = A_sub.tocsr()
    n = int(A_sub.shape[0])
    left = np.asarray(left, dtype=int)
    right = np.asarray(right, dtype=int)
    if left.size + right.size != n:
        return False, {"type": "skip_bad_partition", "n": n}

    n0 = int(left.size)
    n1 = int(right.size)
    n_total = n * (n - 1) // 2
    n_in = (n0 * (n0 - 1) // 2) + (n1 * (n1 - 1) // 2)
    n_out = n0 * n1
    if n_total <= 0 or n_out <= 0:
        return False, {"type": "skip_too_small", "n": n, "n0": n0, "n1": n1}

    side = np.zeros(n, dtype=np.int8)
    side[right] = 1

    m_in = 0
    m_out = 0
    indptr = A_sub.indptr
    indices = A_sub.indices
    for i in range(n):
        si = int(side[i])
        for j in indices[int(indptr[i]) : int(indptr[i + 1])]:
            j = int(j)
            if j <= i:
                continue
            if int(side[j]) == si:
                m_in += 1
            else:
                m_out += 1
    m_total = m_in + m_out

    def clamp_prob(p: float, eps: float = 1e-12) -> float:
        if p < eps:
            return eps
        if p > 1.0 - eps:
            return 1.0 - eps
        return p

    p0 = clamp_prob(m_total / n_total)
    pin = clamp_prob(m_in / n_in) if n_in > 0 else p0
    pout = clamp_prob(m_out / n_out)

    ll1 = m_total * math.log(p0) + (n_total - m_total) * math.log(1.0 - p0)
    ll2 = (
        m_in * math.log(pin)
        + (n_in - m_in) * math.log(1.0 - pin)
        + m_out * math.log(pout)
        + (n_out - m_out) * math.log(1.0 - pout)
    )

    bic1 = -2.0 * ll1 + 1.0 * math.log(n_total)
    bic2 = -2.0 * ll2 + 2.0 * math.log(n_total)
    return (bic2 < bic1), {
        "type": "bic_1v2_bernoulli_edge_presence",
        "n": n,
        "n0": n0,
        "n1": n1,
        "m_total": int(m_total),
        "m_in": int(m_in),
        "m_out": int(m_out),
        "p0": float(p0),
        "p_in": float(pin),
        "p_out": float(pout),
        "bic1": float(bic1),
        "bic2": float(bic2),
        "delta": float(bic1 - bic2),
    }


def _divisive_bisect_by_auto_bic(
    *,
    A,
    lengths,
    coverage,
    seed: int,
    logger: logging.Logger,
    discard_small: bool,
) -> tuple[list[Optional[int]], dict]:
    """
    Top-down spectral bisection on the contig-contig similarity graph.

    Stop rule (automatic, no user thresholds):
      - Split if graph_BIC(2-block) < graph_BIC(1-block) on edge presence, OR
      - If coverage is provided: split if coverage_BIC(2-GMM) < coverage_BIC(1-Gauss) on log1p(coverage).

    Disconnected components are always split (parameter-free).

    discard_small=True: clusters with total_bp < MIN_BIN_BP are treated as unbinned (labels=None).
    discard_small=False: small clusters are kept but are not split further.
    """
    import numpy as np
    import scipy.sparse.csgraph as csgraph

    A = A.tocsr()
    n = int(A.shape[0])

    lens = np.asarray(lengths, dtype=int)
    if lens.size != n:
        raise GraphClusterError("Internal error: lengths array does not match A shape.")
    cov = np.asarray(coverage, dtype=float) if coverage is not None else None

    labels: list[Optional[int]] = [None] * n
    final_clusters: list[np.ndarray] = []

    splits_attempted = 0
    splits_accepted = 0
    component_splits = 0
    graph_bic_triggered = 0
    coverage_bic_triggered = 0
    stopped_unimodal = 0
    stopped_small = 0
    max_depth = 0

    graph_bic_top: list[dict] = []

    stack: list[tuple[np.ndarray, int]] = [(np.arange(n, dtype=int), 0)]
    while stack:
        idx, depth = stack.pop()
        max_depth = max(max_depth, depth)
        if idx.size < 2:
            if discard_small:
                stopped_small += 1
                continue
            final_clusters.append(idx)
            continue

        bp = int(np.sum(lens[idx]))
        if bp < MIN_BIN_BP and discard_small:
            stopped_small += 1
            continue

        A_sub = A[idx][:, idx].tocsr()
        if A_sub.nnz == 0:
            if discard_small:
                stopped_small += 1
                continue
            final_clusters.append(idx)
            continue

        n_comp, comp = csgraph.connected_components(A_sub, directed=False, return_labels=True)
        if n_comp > 1:
            comps: list[np.ndarray] = [idx[np.where(comp == c)[0]] for c in range(n_comp)]
            big = [c for c in comps if int(np.sum(lens[c])) >= MIN_BIN_BP]
            if discard_small and not big:
                final_clusters.append(idx)
                continue
            component_splits += 1
            splits_accepted += 1
            for sub in comps:
                if sub.size >= 1:
                    stack.append((sub, depth + 1))
            continue

        cov_trig = False
        if cov is not None:
            cov_trig, _cov_meta = _coverage_bic_trigger(cov[idx])
            if cov_trig:
                coverage_bic_triggered += 1

        splits_attempted += 1
        left_local, right_local, _split_meta = _spectral_sweep_bisect(A_sub, seed=seed)
        if left_local.size == 0 or right_local.size == 0:
            stopped_unimodal += 1
            final_clusters.append(idx)
            continue

        graph_trig, graph_meta = _graph_bic_trigger(A_sub, left=left_local, right=right_local)
        if graph_meta.get("type") == "bic_1v2_bernoulli_edge_presence":
            graph_bic_top.append(
                {
                    "n": int(graph_meta.get("n", 0)),
                    "delta": float(graph_meta.get("delta", 0.0)),
                    "p_in": float(graph_meta.get("p_in", 0.0)),
                    "p_out": float(graph_meta.get("p_out", 0.0)),
                }
            )
            graph_bic_top.sort(key=lambda x: x["delta"], reverse=True)
            if len(graph_bic_top) > 50:
                graph_bic_top.pop()

        if graph_trig:
            graph_bic_triggered += 1

        if not (cov_trig or graph_trig):
            stopped_unimodal += 1
            final_clusters.append(idx)
            continue

        left = idx[left_local]
        right = idx[right_local]

        if discard_small:
            left_bp = int(np.sum(lens[left]))
            right_bp = int(np.sum(lens[right]))
            if left_bp < MIN_BIN_BP and right_bp < MIN_BIN_BP:
                final_clusters.append(idx)
                continue

        splits_accepted += 1
        stack.append((left, depth + 1))
        stack.append((right, depth + 1))

    kept: list[tuple[int, np.ndarray]] = []
    for idx in final_clusters:
        bp = int(np.sum(lens[idx]))
        if bp >= MIN_BIN_BP or not discard_small:
            kept.append((bp, idx))

    kept.sort(key=lambda x: (-x[0], int(np.min(x[1]))))
    for bin_id, (_bp, idx) in enumerate(kept):
        for i in idx.tolist():
            labels[int(i)] = int(bin_id)

    contigs_assigned = sum(1 for x in labels if x is not None)
    logger.info(
        "Spectral bisection (auto-BIC): bins=%s assigned_contigs=%s/%s splits_attempted=%s splits_accepted=%s",
        len(kept),
        contigs_assigned,
        n,
        splits_attempted,
        splits_accepted,
    )

    return labels, {
        "k_selected": int(len(kept)),
        "bins_written": int(len(kept)),
        "contigs_assigned": int(contigs_assigned),
        "contigs_unbinned": int(n - contigs_assigned),
        "splits": {
            "attempted": int(splits_attempted),
            "accepted": int(splits_accepted),
            "component_splits": int(component_splits),
            "graph_bic_triggered": int(graph_bic_triggered),
            "coverage_bic_triggered": int(coverage_bic_triggered),
            "stopped_unimodal": int(stopped_unimodal),
            "stopped_small": int(stopped_small),
            "max_depth": int(max_depth),
        },
        "graph_bic": {"type": "bic_1v2_bernoulli_edge_presence"},
        "coverage_bic": {"min_n": 20, "type": "bic_1gauss_vs_2gmm_on_log1p_cov"},
        "graph_bic_top": graph_bic_top,
    }


def _divisive_bisect_by_coverage_bic(
    *,
    A,
    contig_names: list[str],
    lengths,
    coverage,
    seed: int,
    logger: logging.Logger,
) -> tuple[list[Optional[int]], dict]:
    """
    Top-down spectral bisection on the contig-contig similarity graph, driven by BAM coverage multi-modality.
    """
    import numpy as np
    import scipy.sparse.csgraph as csgraph

    A = A.tocsr()
    n = int(A.shape[0])
    if n != len(contig_names):
        raise GraphClusterError("Internal error: contig_names length does not match A shape.")

    lens = np.asarray(lengths, dtype=int)
    cov = np.asarray(coverage, dtype=float)

    labels: list[Optional[int]] = [None] * n
    final_clusters: list[np.ndarray] = []

    splits_attempted = 0
    splits_accepted = 0
    bic_triggered = 0
    stopped_unimodal = 0
    stopped_small = 0
    max_depth = 0

    stack: list[tuple[np.ndarray, int]] = [(np.arange(n, dtype=int), 0)]
    while stack:
        idx, depth = stack.pop()
        max_depth = max(max_depth, depth)
        if idx.size < 2:
            stopped_small += 1
            continue

        bp = int(np.sum(lens[idx]))
        if bp < MIN_BIN_BP:
            stopped_small += 1
            continue

        trig, _bic_meta = _coverage_bic_trigger(cov[idx])
        if not trig:
            stopped_unimodal += 1
            final_clusters.append(idx)
            continue
        bic_triggered += 1

        A_sub = A[idx][:, idx].tocsr()
        if A_sub.nnz == 0:
            final_clusters.append(idx)
            continue

        n_comp, comp = csgraph.connected_components(A_sub, directed=False, return_labels=True)
        if n_comp > 1:
            for c in range(n_comp):
                sub = idx[np.where(comp == c)[0]]
                if sub.size >= 2:
                    stack.append((sub, depth + 1))
            splits_accepted += 1
            continue

        splits_attempted += 1
        left_local, right_local, _split_meta = _spectral_sweep_bisect(A_sub, seed=seed)
        if left_local.size == 0 or right_local.size == 0:
            final_clusters.append(idx)
            continue
        splits_accepted += 1
        stack.append((idx[left_local], depth + 1))
        stack.append((idx[right_local], depth + 1))

    kept: list[tuple[int, np.ndarray]] = []
    for idx in final_clusters:
        bp = int(np.sum(lens[idx]))
        if bp >= MIN_BIN_BP:
            kept.append((bp, idx))

    kept.sort(key=lambda x: (-x[0], int(np.min(x[1]))))
    for bin_id, (_bp, idx) in enumerate(kept):
        for i in idx.tolist():
            labels[int(i)] = int(bin_id)

    contigs_assigned = sum(1 for x in labels if x is not None)
    logger.info(
        "Spectral bisection: bins=%s assigned_contigs=%s/%s splits_attempted=%s splits_accepted=%s",
        len(kept),
        contigs_assigned,
        n,
        splits_attempted,
        splits_accepted,
    )

    return labels, {
        "k_selected": int(len(kept)),
        "bins_written": int(len(kept)),
        "contigs_assigned": int(contigs_assigned),
        "contigs_unbinned": int(n - contigs_assigned),
        "splits": {
            "attempted": int(splits_attempted),
            "accepted": int(splits_accepted),
            "bic_triggered": int(bic_triggered),
            "stopped_unimodal": int(stopped_unimodal),
            "stopped_small": int(stopped_small),
            "max_depth": int(max_depth),
        },
        "coverage_bic": {"min_n": 20, "method": "bic_1gauss_vs_2gmm_on_log1p_cov"},
    }


def _spectral_sweep_bisect(A_sub, *, seed: int) -> tuple["np.ndarray", "np.ndarray", dict]:
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    n = int(A_sub.shape[0])
    if n < 2:
        return np.asarray([], dtype=int), np.asarray([], dtype=int), {"method": "fallback_small_n", "n": n}

    deg = np.asarray(A_sub.sum(axis=1)).ravel()
    d_inv_sqrt = np.zeros_like(deg, dtype=float)
    nz = deg > 0
    d_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])

    if n <= 3:
        A_dense = A_sub.toarray() if sp.issparse(A_sub) else np.asarray(A_sub, dtype=float)
        B = (d_inv_sqrt[:, None] * A_dense) * d_inv_sqrt[None, :]
        L_dense = np.eye(n, dtype=float) - B
        evals, evecs = np.linalg.eigh(L_dense)
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]
    else:

        def matvec(x):
            y = d_inv_sqrt * x
            y = A_sub @ y
            y = d_inv_sqrt * y
            return x - y

        L = spla.LinearOperator((n, n), matvec=matvec, dtype=float)
        rng = np.random.default_rng(int(seed))
        v0 = rng.standard_normal(n)
        evals, evecs = spla.eigsh(L, k=2, which="SA", v0=v0)
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]

    f = evecs[:, 1]
    node_order = np.argsort(f)
    best_t, best_phi = _sweep_best_conductance(A_sub, degrees=deg, order=node_order)
    left = node_order[:best_t]
    right = node_order[best_t:]
    return left, right, {"lambda2": float(evals[1]), "phi": float(best_phi), "method": "spectral_sweep_cut"}


def _sweep_best_conductance(A_sub, *, degrees, order):
    import numpy as np

    n = int(A_sub.shape[0])
    if n < 2:
        return 0, float("inf")

    deg = np.asarray(degrees, dtype=float)
    vol_total = float(np.sum(deg))
    if vol_total <= 0:
        return n // 2, float("inf")

    in_s = np.zeros(n, dtype=bool)
    cut = 0.0
    vol_s = 0.0
    best_phi = float("inf")
    best_t = n // 2

    for t in range(1, n):
        v = int(order[t - 1])
        in_s[v] = True
        vol_s += float(deg[v])

        row_start = int(A_sub.indptr[v])
        row_end = int(A_sub.indptr[v + 1])
        neigh = A_sub.indices[row_start:row_end]
        wts = A_sub.data[row_start:row_end]
        for u, w in zip(neigh, wts, strict=True):
            cut += -float(w) if in_s[int(u)] else float(w)

        vol_t = vol_total - vol_s
        if vol_s <= 0 or vol_t <= 0:
            continue
        phi = cut / min(vol_s, vol_t)
        if phi < best_phi:
            best_phi = phi
            best_t = t

    return best_t, best_phi


def _coverage_bic_trigger(values) -> tuple[bool, dict]:
    xs = [math.log1p(float(v)) for v in values if float(v) >= 0.0]
    n = len(xs)
    if n < 20:
        return False, {"type": "skip_small_n", "n": n}
    bic1 = _bic(xs, k=1)
    bic2 = _bic(xs, k=2)
    return (bic2 < bic1), {"type": "bic_1v2", "n": n, "bic1": bic1, "bic2": bic2, "delta": bic1 - bic2}


def _bic(xs: list[float], *, k: int) -> float:
    if not xs:
        return 0.0
    if k == 1:
        ll = _loglik_1gauss(xs)
        p = 2
    elif k == 2:
        ll = _loglik_2gmm(xs)
        p = 5
    else:
        raise ValueError("k must be 1 or 2")
    n = len(xs)
    return -2.0 * ll + p * math.log(n)


def _loglik_1gauss(xs: list[float]) -> float:
    n = len(xs)
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / n
    var = max(var, 1e-6)
    return sum(_log_norm_pdf(x, mu, var) for x in xs)


def _loglik_2gmm(xs: list[float]) -> float:
    n = len(xs)
    xs_sorted = sorted(xs)
    mu1 = xs_sorted[n // 4]
    mu2 = xs_sorted[(3 * n) // 4]
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / n
    var1 = var2 = max(var, 1e-3)
    w1 = 0.5

    for _ in range(50):
        r1_sum = 0.0
        r2_sum = 0.0
        mu1_num = 0.0
        mu2_num = 0.0
        var1_num = 0.0
        var2_num = 0.0

        for x in xs:
            l1 = math.log(w1) + _log_norm_pdf(x, mu1, var1)
            l2 = math.log(1.0 - w1) + _log_norm_pdf(x, mu2, var2)
            m = max(l1, l2)
            d = math.exp(l1 - m) + math.exp(l2 - m)
            r1 = math.exp(l1 - m) / d
            r2 = 1.0 - r1
            r1_sum += r1
            r2_sum += r2
            mu1_num += r1 * x
            mu2_num += r2 * x

        if r1_sum <= 1e-9 or r2_sum <= 1e-9:
            break

        w1 = max(1e-3, min(1.0 - 1e-3, r1_sum / n))
        mu1 = mu1_num / r1_sum
        mu2 = mu2_num / r2_sum

        for x in xs:
            l1 = math.log(w1) + _log_norm_pdf(x, mu1, var1)
            l2 = math.log(1.0 - w1) + _log_norm_pdf(x, mu2, var2)
            m = max(l1, l2)
            d = math.exp(l1 - m) + math.exp(l2 - m)
            r1 = math.exp(l1 - m) / d
            r2 = 1.0 - r1
            var1_num += r1 * (x - mu1) ** 2
            var2_num += r2 * (x - mu2) ** 2

        var1 = max(var1_num / r1_sum, 1e-6)
        var2 = max(var2_num / r2_sum, 1e-6)

    ll = 0.0
    for x in xs:
        l1 = math.log(w1) + _log_norm_pdf(x, mu1, var1)
        l2 = math.log(1.0 - w1) + _log_norm_pdf(x, mu2, var2)
        m = max(l1, l2)
        ll += m + math.log(math.exp(l1 - m) + math.exp(l2 - m))
    return ll


def _log_norm_pdf(x: float, mu: float, var: float) -> float:
    return -0.5 * (math.log(2.0 * math.pi * var) + ((x - mu) ** 2) / var)
