from __future__ import annotations

import csv
import gzip
import logging
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from porebin import __version__
from porebin.build_graph import order_norm
from porebin.export import MIN_BIN_BP
from porebin.utils import (
    dedupe_preserve_order,
    ensure_dir,
    iter_fasta_records,
    utc_now_iso,
    write_json,
)


class RefineError(RuntimeError):
    pass


@dataclass
class RefineStats:
    contigs_total: int = 0
    bins_total: int = 0
    bins_kept_initial: int = 0
    bins_kept_final: int = 0
    contigs_short: int = 0
    contigs_assigned_initial: int = 0
    contigs_assigned_final: int = 0

    reads_total: int = 0
    reads_kept: int = 0
    reads_skipped_k_lt_2: int = 0
    input_sorted_by_readid: bool = True

    decontam_removed: int = 0
    reassign_moved: int = 0
    reassign_unbinned: int = 0
    recruit_assigned: int = 0
    split_triggered: int = 0
    split_applied: int = 0


def _entropy(probs: list[float]) -> float:
    """Natural entropy (nats)."""
    h = 0.0
    for p in probs:
        p = float(p)
        if p > 0.0:
            h -= p * math.log(p)
    return float(h)


def _effective_hosts(entropy_nats: float) -> float:
    """Effective support count: exp(entropy)."""
    return float(math.exp(float(entropy_nats)))


def refine_bins(
    *,
    contigs_fasta: Path,
    ppl_contacts: Path,
    bins_tsv: Path,
    bam: Optional[Path],
    out_dir: Path,
    threads: int = 1,
    seed: int = 0,
    logger: Optional[logging.Logger] = None,
    ) -> Path:
    """
    Refine coarse bins using automatic/default rules (no user thresholds).

    Outputs:
      - out_dir/bins.refined.tsv
      - out_dir/recruit_assign.tsv
      - out_dir/decontam_removed.tsv
      - out_dir/split_map.tsv
      - out_dir/run_refine.json
    """
    logger = logger or logging.getLogger("porebin")
    out_dir = out_dir.resolve()
    ensure_dir(out_dir)

    if not contigs_fasta.exists():
        raise FileNotFoundError(f"Contigs FASTA not found: {contigs_fasta}")
    if not ppl_contacts.exists():
        raise FileNotFoundError(f"PPL contacts not found: {ppl_contacts}")
    if not bins_tsv.exists():
        raise FileNotFoundError(f"Bins TSV not found: {bins_tsv}")
    if bam is not None and not bam.exists():
        raise FileNotFoundError(f"BAM not found: {bam}")

    run_json = out_dir / "run_refine.json"
    record: dict[str, Any] = {
        "porebin_version": __version__,
        "command": "refine",
        "started_at": utc_now_iso(),
        "ended_at": None,
        "status": "running",
        "inputs": {
            "contigs_fasta": str(contigs_fasta),
            "ppl_contacts": str(ppl_contacts),
            "bins_tsv": str(bins_tsv),
            "bam": str(bam) if bam is not None else None,
        },
        "seed": seed,
        "threads": threads,
        "thresholds": {},
        "decisions": {},
        "stats": {},
        "outputs": {},
        "cwd": os.getcwd(),
    }
    write_json(run_json, record)

    stats = RefineStats()
    try:
        contig_len = _read_contig_lengths(contigs_fasta)
        if not contig_len:
            raise RefineError(f"No contigs found in FASTA: {contigs_fasta}")
        stats.contigs_total = len(contig_len)

        min_contig_len, min_contig_len_meta = auto_min_contig_len(contig_len)

        coarse = _read_bins_tsv(bins_tsv)
        stats.bins_total = len(set(coarse.values()))

        # keep_bins is defined the same way as export: total_bp >= 200kb
        coarse_bin_bp, _coarse_bin_n = _bin_sizes(coarse, contig_len, min_contig_len=min_contig_len)
        keep_bins = {b for b, bp in coarse_bin_bp.items() if bp >= MIN_BIN_BP}
        stats.bins_kept_initial = len(keep_bins)

        contig_to_bin: dict[str, str] = {}
        bin_to_contigs: dict[str, list[str]] = defaultdict(list)
        unbinned_reason: dict[str, str] = {}
        for contig, L in contig_len.items():
            if L < min_contig_len:
                stats.contigs_short += 1
                unbinned_reason[contig] = "short_contig"
                continue
            b = coarse.get(contig)
            if b is None:
                unbinned_reason[contig] = "unassigned"
                continue
            if b not in keep_bins:
                unbinned_reason[contig] = "tiny_bin"
                continue
            contig_to_bin[contig] = b
            bin_to_contigs[b].append(contig)
        stats.contigs_assigned_initial = len(contig_to_bin)

        cov: Optional[dict[str, float]] = None
        bin_cov_stats: Optional[dict[str, dict[str, float]]] = None
        coverage_source_meta: Optional[dict[str, Any]] = None
        if bam is not None:
            cov_tsv = bins_tsv.parent / "coverage" / "coverage.tsv"
            if cov_tsv.exists():
                logger.info(f"Refine: using cached coverage TSV: {cov_tsv}")
                cov = _coverage_from_tsv(cov_tsv, contig_len=contig_len, min_contig_len=min_contig_len)
                coverage_source_meta = {"type": "coverage_tsv", "path": str(cov_tsv)}
            else:
                cov = _coverage_from_bam(bam, contig_len, min_contig_len=min_contig_len, logger=logger)
                coverage_source_meta = {"type": "bam_scan", "bam": str(bam)}
            bin_cov_stats = _bin_coverage_stats(bin_to_contigs, cov)

        intra, other, affinity, contact_meta = _scan_contacts_support_and_affinity(
            ppl_contacts=ppl_contacts,
            contig_len=contig_len,
            min_contig_len=min_contig_len,
            contig_to_bin=contig_to_bin,
        )
        stats.reads_total = int(contact_meta["reads_total"])
        stats.reads_kept = int(contact_meta["reads_kept"])
        stats.reads_skipped_k_lt_2 = int(contact_meta["reads_skipped_k_lt_2"])
        stats.input_sorted_by_readid = bool(contact_meta["input_sorted_by_readid"])

        decontam_path = out_dir / "decontam_removed.tsv"
        removed = _decontam(
            bin_to_contigs=bin_to_contigs,
            contig_len=contig_len,
            contig_to_bin=contig_to_bin,
            intra_support=intra,
            other_support=other,
            cov=cov,
            bin_cov_stats=bin_cov_stats,
            out_path=decontam_path,
        )
        stats.decontam_removed = len(removed)
        for c in removed:
            unbinned_reason[c] = "removed_by_refine"

        # Recompute keep bins after decontam (align with export)
        bin_bp2, _ = _bin_sizes_from_assignment(contig_to_bin, contig_len)
        keep_bins2 = {b for b, bp in bin_bp2.items() if bp >= MIN_BIN_BP}
        _drop_bins_below_min(bin_to_contigs, contig_to_bin, keep_bins2, unbinned_reason)

        recruit_path = out_dir / "recruit_assign.tsv"
        recruit_result = _recruit(
            affinity=affinity,
            contig_len=contig_len,
            min_contig_len=min_contig_len,
            contig_to_bin=contig_to_bin,
            bin_to_contigs=bin_to_contigs,
            keep_bins=keep_bins2,
            cov=cov,
            bin_cov_stats=bin_cov_stats,
            out_path=recruit_path,
        )
        recruited = recruit_result["assigned"]
        stats.recruit_assigned = len(recruited)
        for c in recruited:
            unbinned_reason.pop(c, None)

        split_map = out_dir / "split_map.tsv"
        split_meta = _split_bins(
            ppl_contacts=ppl_contacts,
            contig_len=contig_len,
            min_contig_len=min_contig_len,
            contig_to_bin=contig_to_bin,
            bin_to_contigs=bin_to_contigs,
            cov=cov,
            out_path=split_map,
            seed=seed,
        )
        stats.split_triggered = int(split_meta["bins_triggered"])
        stats.split_applied = int(split_meta["bins_split"])
        for c in split_meta["contigs_unbinned"]:
            unbinned_reason[c] = "removed_by_refine"

        # Final keep bins (align with export)
        final_bp, _ = _bin_sizes_from_assignment(contig_to_bin, contig_len)
        keep_bins_final = {b for b, bp in final_bp.items() if bp >= MIN_BIN_BP}
        stats.bins_kept_final = len(keep_bins_final)

        refined_bins = out_dir / "bins.refined.tsv"
        with refined_bins.open("w", encoding="utf-8", newline="") as fh:
            fh.write("contig_name\tbin_id\n")
            for contig, b in contig_to_bin.items():
                if b in keep_bins_final:
                    fh.write(f"{contig}\t{b}\n")
        stats.contigs_assigned_final = sum(1 for _c, b in contig_to_bin.items() if b in keep_bins_final)

        record["thresholds"] = {
            "MIN_BIN_BP": MIN_BIN_BP,
            "MIN_CONTIG_LEN": min_contig_len,
            "MIN_CONTIG_LEN_method": min_contig_len_meta,
        }
        record["decisions"] = {
            "status_filter": "passed",
            "order_norm": "2/(k*(k-1))  # == 1/C(k,2)",
            "coverage_source": coverage_source_meta,
            "keep_bins_rule": "total_bp >= 200kb",
            "recruit": {
                "topK": 3,
                "threshold": float(recruit_result["threshold"]),
                "threshold_method": recruit_result["threshold_method"],
                "ratio_threshold": 3.0,
                "coverage_veto": {
                    "enabled": bam is not None,
                    "rule": "abs(cov_u - bin_median) <= 3*MAD",
                    "k": 3.0,
                },
            },
            "decontam_rule": "per-bin: remove if intra_support < max(0, median - 3*MAD)",
            "split": {
                **split_meta["meta"],
                "bins_triggered": split_meta.get("bins_triggered_list", []),
                "bic_top": split_meta.get("bic_top", []),
            },
        }
        record["stats"] = stats.__dict__
        record["outputs"] = {
            "bins_refined_tsv": str(refined_bins),
            "recruit_assign_tsv": str(recruit_path),
            "decontam_removed_tsv": str(decontam_path),
            "split_map_tsv": str(split_map),
        }
        record["status"] = "ok"
        return refined_bins
    except Exception as exc:
        record["status"] = "error"
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        record["ended_at"] = utc_now_iso()
        write_json(run_json, record)


def refine_bins_parquet(
    *,
    contigs_fasta: Path,
    contacts_parquet: Path,
    bins_tsv: Path,
    coverage_tsv: Optional[Path],
    out_dir: Path,
    threads: int = 1,
    seed: int = 0,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """
    Refine *candidate host communities* (coarse bins) into final host-centric assignments.

    Evidence-layer semantics (must hold):
      - Each Pore-C read r is a hyperedge (one row per contact in contacts.parquet).
      - contacts.parquet.contig_weights are pi_{r,c}: normalized evidence shares from read r to contig c.
        They are NOT posterior probabilities.
      - contacts.parquet.weight is q(r): read-level alignment quality weight in [0,1].

    Refine-layer semantics (this function):
      - Coarse bins are candidate host communities only (not final truth).
      - Refine computes posterior-like host support scores theta_{c,b} for each contig c over candidate hosts b,
        primarily from contact evidence, with a weak coarse-label prior/regularizer (indicator on coarse_host).
      - Refine emits uncertainty summaries and a separate accessory/MGE-like association head (structural only).

    High-order read treatment:
      - Total read information is governed mainly by q(r).
      - No strong pairwise-style normalization (e.g. 2/[k(k-1)]) is used in this refine inference pass.

    Note:
      - coverage_tsv is currently accepted for API compatibility; this MVP refine path does not directly use it.
    """
    logger = logger or logging.getLogger("porebin")
    out_dir = out_dir.resolve()
    ensure_dir(out_dir)

    if not contigs_fasta.exists():
        raise FileNotFoundError(f"Contigs FASTA not found: {contigs_fasta}")
    if not contacts_parquet.exists():
        raise FileNotFoundError(f"Contacts Parquet not found: {contacts_parquet}")
    if not bins_tsv.exists():
        raise FileNotFoundError(f"Bins TSV not found: {bins_tsv}")

    run_json = out_dir / "run_refine.json"
    record: dict[str, Any] = {
        "porebin_version": __version__,
        "command": "refine",
        "started_at": utc_now_iso(),
        "ended_at": None,
        "status": "running",
        "inputs": {
            "contigs_fasta": str(contigs_fasta),
            "contacts_parquet": str(contacts_parquet),
            "bins_tsv": str(bins_tsv),
            "coverage_tsv": str(coverage_tsv) if coverage_tsv is not None else None,
        },
        "seed": seed,
        "threads": threads,
        "thresholds": {},
        "decisions": {},
        "stats": {},
        "outputs": {},
        "cwd": os.getcwd(),
    }
    write_json(run_json, record)

    stats = RefineStats()
    try:
        # Numerical stability constant used in prior, normalization, and entropy.
        eps = 1e-12

        contig_len = _read_contig_lengths(contigs_fasta)
        if not contig_len:
            raise RefineError(f"No contigs found in FASTA: {contigs_fasta}")
        stats.contigs_total = len(contig_len)

        coarse_raw = _read_bins_tsv(bins_tsv)
        coarse_filtered: dict[str, str] = {}
        for c, b in coarse_raw.items():
            c = str(c).strip()
            b = str(b).strip()
            if not c:
                continue
            if c not in contig_len:
                continue
            coarse_filtered[c] = b

        # Candidate host bins set B (exclude -1).
        B: list[str] = sorted({b for b in coarse_filtered.values() if b and b != "-1"})
        B_set = set(B)
        if not B:
            raise RefineError("No candidate host bins found in coarse bins.tsv (empty or only -1).")

        stats.bins_total = int(len(B))
        stats.bins_kept_initial = int(stats.bins_total)

        # Coarse label per contig in the FASTA; use "-1" for unassigned/noise.
        contig_to_coarse: dict[str, str] = {}
        assigned = 0
        for c in contig_len.keys():
            b = coarse_filtered.get(c, "-1")
            b = str(b).strip() if b is not None else "-1"
            if b not in B_set:
                b = "-1"
            else:
                assigned += 1
            contig_to_coarse[c] = b
        stats.contigs_assigned_initial = int(assigned)

        # support[c][b] accumulates contact-driven host support mass for contig c towards candidate host bin b.
        # This is an inference-layer quantity (not pi_{r,c}).
        support: defaultdict[str, dict[str, float]] = defaultdict(dict)

        reads_total = 0
        reads_used = 0
        reads_skipped_k_lt_2 = 0
        for contigs_raw, pi_raw, q in _iter_contacts_parquet(contacts_parquet, batch_size=100_000):
            reads_total += 1
            if not contigs_raw or not pi_raw:
                continue

            # Merge duplicates and renormalize pi.
            seen: dict[str, int] = {}
            contigs: list[str] = []
            pi: list[float] = []
            for c, w in zip(contigs_raw, pi_raw, strict=True):
                c = str(c).strip()
                if not c:
                    continue
                if c not in contig_len:
                    continue
                ww = float(w)
                if ww <= 0.0:
                    continue
                if c in seen:
                    pi[seen[c]] += ww
                else:
                    seen[c] = len(contigs)
                    contigs.append(c)
                    pi.append(ww)

            if len(contigs) < 2:
                reads_skipped_k_lt_2 += 1
                continue

            s = float(sum(pi))
            if not (s > 0.0):
                continue
            pi = [float(x) / s for x in pi]

            reads_used += 1
            q_read = float(q)

            # Bin mass per candidate host: M_r(b) = sum_{u in bin b} pi_{r,u}
            m_by_host: dict[str, float] = {}
            host_of: list[str] = []
            for c, pc in zip(contigs, pi, strict=True):
                b = contig_to_coarse.get(c, "-1")
                host_of.append(b)
                if b == "-1":
                    continue
                m_by_host[b] = m_by_host.get(b, 0.0) + float(pc)

            if not m_by_host:
                continue

            # MVP v1 support update (strict definition):
            #   support[c][b] += q(r) * pi_{r,c} * max(0, M_r(b) - self_term)
            # where:
            #   M_r(b) = sum_{u in bin b} pi_{r,u}
            #   self_term = pi_{r,c} if coarse_bin(c) == b else 0
            # High-order read philosophy: eta(k)=1.0 (no extra k penalty in refine).
            for c, pc, b_c in zip(contigs, pi, host_of, strict=True):
                pc = float(pc)
                if pc <= 0.0:
                    continue
                d = support.get(c)
                if d is None:
                    d = {}
                    support[c] = d
                for b, m in m_by_host.items():
                    m_eff = float(m)
                    if b_c == b:
                        m_eff = max(0.0, m_eff - pc)
                    if m_eff <= 0.0:
                        continue
                    d[b] = float(d.get(b, 0.0)) + q_read * pc * m_eff

        stats.reads_total = int(reads_total)
        stats.reads_kept = int(reads_used)
        stats.reads_skipped_k_lt_2 = int(reads_skipped_k_lt_2)
        stats.input_sorted_by_readid = True

        # Fixed MVP constants/thresholds (must be explicit and recorded).
        prior_strength = 0.05

        core_top1_score_thresh = 0.80
        core_margin_thresh = 0.50
        core_eff_hosts_thresh = 1.5

        accessory_eff_hosts_thresh = 2.0
        accessory_entropy_thresh = float(math.log(2.0))
        accessory_top1_lt = 0.85

        contig_scores_path = out_dir / "contig_host_scores.tsv"
        assoc_path = out_dir / "accessory_associations.tsv"
        refined_bins = out_dir / "bins.refined.tsv"

        B_count = int(len(B))
        contigs_all = sorted(contig_len.keys())

        n_core = 0
        n_accessory = 0
        n_ambiguous = 0
        final_bins_used: set[str] = set()

        with refined_bins.open("w", encoding="utf-8", newline="") as fh_bins, contig_scores_path.open(
            "w", encoding="utf-8", newline=""
        ) as fh_scores, assoc_path.open("w", encoding="utf-8", newline="") as fh_assoc:
            fh_bins.write("contig_name\tbin_id\n")
            fh_scores.write(
                "contig_name\tcoarse_host\ttop1_host\ttop2_host\t"
                "top1_score\ttop2_score\tmargin\tentropy\teffective_hosts\t"
                "has_contact_support\tis_core_like\tis_ambiguous\tis_accessory_candidate\n"
            )
            fh_assoc.write(
                "contig_name\ttop_hosts\thost_weights\thost_entropy\teffective_hosts\t"
                "association_confidence\tsingle_host_like\tbroad_host_like\n"
            )

            for c in contigs_all:
                coarse_host = contig_to_coarse.get(c, "-1")
                sdict = support.get(c, {})

                # MVP prior: weak coarse-label regularizer (indicator on coarse_host), not a direct feature prior.
                sum_support = float(sum(float(v) for v in sdict.values()))
                has_contact_support = bool(sum_support > 0.0)
                alpha_c = float(prior_strength * (sum_support + eps))

                # If a contig has no contact support and has no coarse prior (coarse_host == -1),
                # treat it as unresolved-no-contact rather than accessory-like.
                if (not has_contact_support) and (coarse_host == "-1"):
                    top1_host = "-1"
                    top2_host = "-1"
                    top1_score = 0.0
                    top2_score = 0.0
                    margin = 0.0
                    ent = 0.0
                    effective_hosts = 0.0
                    is_core_like = False
                    is_accessory = False
                    is_ambiguous = True
                    fh_scores.write(
                        f"{c}\t{coarse_host}\t{top1_host}\t{top2_host}\t"
                        f"{top1_score:.6g}\t{top2_score:.6g}\t{margin:.6g}\t{ent:.6g}\t{effective_hosts:.6g}\t"
                        f"{1 if has_contact_support else 0}\t"
                        f"{1 if is_core_like else 0}\t{1 if is_ambiguous else 0}\t{1 if is_accessory else 0}\n"
                    )
                    n_ambiguous += 1
                    continue

                score_nz: dict[str, float] = {b: float(v) for b, v in sdict.items() if b in B_set and float(v) > 0.0}
                if coarse_host in B_set:
                    score_nz[coarse_host] = float(score_nz.get(coarse_host, 0.0)) + alpha_c

                n_nonzero_hosts = int(len(score_nz))
                sum_score = float(sum_support + (alpha_c if coarse_host in B_set else 0.0))
                denom = float(sum_score + (B_count * eps))
                if denom <= 0.0:
                    denom = float(B_count * eps)

                def theta_of(b: str) -> float:
                    # theta[c][b] = (score[c][b] + eps) / sum_{b'}(score[c][b'] + eps)
                    return float((float(score_nz.get(b, 0.0)) + eps) / denom)

                # Rank bins by score (desc, tie by bin_id asc), then fill with zero-score bins in B order.
                ranked: list[str] = [bb for bb, _v in sorted(score_nz.items(), key=lambda kv: (-kv[1], kv[0]))]
                if len(ranked) < 3:
                    for bb in B:
                        if bb in score_nz:
                            continue
                        ranked.append(bb)
                        if len(ranked) >= 3:
                            break
                if not ranked:
                    ranked = B[:3]

                top1_host = ranked[0]
                top2_host = ranked[1] if B_count >= 2 else ""

                top1_score = theta_of(top1_host)
                top2_score = theta_of(top2_host) if top2_host else 0.0
                margin = float(top1_score - top2_score)

                # entropy = -sum_b theta[c][b] * log(theta[c][b] + eps)
                theta0 = float(eps / denom)
                ent = 0.0
                for _bb, sc in score_nz.items():
                    th = float((float(sc) + eps) / denom)
                    ent -= th * math.log(th + eps)
                n0 = int(B_count - len(score_nz))
                if n0 > 0:
                    ent -= float(n0) * theta0 * math.log(theta0 + eps)
                effective_hosts = float(math.exp(ent))

                is_core_like = bool(
                    (top1_score >= core_top1_score_thresh)
                    and (margin >= core_margin_thresh)
                    and (effective_hosts <= core_eff_hosts_thresh)
                )
                is_accessory = bool(
                    (effective_hosts >= accessory_eff_hosts_thresh)
                    or ((ent >= accessory_entropy_thresh) and (top1_score < accessory_top1_lt))
                )
                is_ambiguous = bool(not is_core_like)

                if is_core_like:
                    n_core += 1
                    final_bins_used.add(top1_host)
                    fh_bins.write(f"{c}\t{top1_host}\n")
                if is_accessory:
                    n_accessory += 1
                if is_ambiguous:
                    n_ambiguous += 1

                fh_scores.write(
                    f"{c}\t{coarse_host}\t{top1_host}\t{top2_host}\t"
                    f"{top1_score:.6g}\t{top2_score:.6g}\t{margin:.6g}\t{ent:.6g}\t{effective_hosts:.6g}\t"
                    f"{1 if has_contact_support else 0}\t"
                    f"{1 if is_core_like else 0}\t{1 if is_ambiguous else 0}\t{1 if is_accessory else 0}\n"
                )

                if is_accessory:
                    # Accessory association head: report only hosts with nonzero scores (no zero-padding).
                    if not score_nz:
                        continue
                    ranked_nz = [bb for bb, _v in sorted(score_nz.items(), key=lambda kv: (-kv[1], kv[0]))]
                    topK = min(3, len(ranked_nz))
                    top_hosts = ranked_nz[:topK]
                    theta_top = [theta_of(bb) for bb in top_hosts]
                    z = float(sum(theta_top))
                    if z <= 0.0:
                        host_weights = [1.0 / float(topK)] * int(topK)
                    else:
                        host_weights = [float(x) / z for x in theta_top]

                    # association_confidence = 1 - entropy / log(max(2, number_of_nonzero_hosts)), clip to [0,1]
                    denom_k = float(math.log(max(2, n_nonzero_hosts)))
                    conf = float(1.0 - (ent / denom_k)) if denom_k > 0.0 else 0.0
                    conf = float(max(0.0, min(1.0, conf)))

                    single_host_like = bool((top1_score >= 0.80) and (effective_hosts < 1.5))
                    broad_host_like = bool(effective_hosts >= 2.0)

                    fh_assoc.write(
                        f"{c}\t{','.join(top_hosts)}\t{','.join(f'{w:.6g}' for w in host_weights)}\t"
                        f"{ent:.6g}\t{effective_hosts:.6g}\t{conf:.6g}\t"
                        f"{1 if single_host_like else 0}\t{1 if broad_host_like else 0}\n"
                    )

        stats.contigs_assigned_final = int(n_core)
        stats.bins_kept_final = int(len(final_bins_used))

        record["thresholds"] = {
            "eps": eps,
            "core_like_thresholds": {
                "top1_score_min": core_top1_score_thresh,
                "margin_min": core_margin_thresh,
                "effective_hosts_max": core_eff_hosts_thresh,
            },
            "accessory_thresholds": {
                "effective_hosts_min": accessory_eff_hosts_thresh,
                "entropy_min": accessory_entropy_thresh,
                "top1_score_lt_if_entropy_trigger": accessory_top1_lt,
            },
            "ambiguous_rule": "not core-like (accessory may also be ambiguous)",
        }
        record["decisions"] = {
            "coarse_bins_semantics": "candidate_host_communities",
            "contig_weights_semantics": "normalized_evidence_share",
            "refine_method": "host_assignment_inference",
            "high_order_read_philosophy": "q_controls_total_information",
            "eta_k": 1.0,
            "coarse_prior_used": True,
            "coarse_prior_strength": prior_strength,
            "direct_feature_prior_used": False,
        }
        record["stats"] = {
            **stats.__dict__,
            "candidate_bins_count": int(B_count),
            "contigs_core_like": int(n_core),
            "contigs_ambiguous": int(n_ambiguous),
            "contigs_accessory_candidates": int(n_accessory),
        }
        record["outputs"] = {
            "bins_refined_tsv": str(refined_bins),
            "contig_host_scores_tsv": str(contig_scores_path),
            "accessory_associations_tsv": str(assoc_path),
        }
        record["status"] = "ok"
        return refined_bins
    except Exception as exc:
        record["status"] = "error"
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        record["ended_at"] = utc_now_iso()
        write_json(run_json, record)


def auto_min_contig_len(contig_len: dict[str, int]) -> tuple[int, dict[str, Any]]:
    """
    SemiBin-like heuristic: choose MIN_CONTIG_LEN=1000 or 2500 depending on how much
    sequence mass sits in the 1000-2500bp range.
    """
    total_bp = sum(contig_len.values())
    bp_1000_2500 = sum(L for L in contig_len.values() if 1000 <= L <= 2500)
    ratio = (bp_1000_2500 / total_bp) if total_bp else 0.0
    min_len = 1000 if ratio >= 0.05 else 2500
    meta = {
        "type": "semiBin_ratio_1000_2500",
        "ratio_1000_2500_bp": ratio,
        "ratio_threshold": 0.05,
        "rule": "min_contig_len=1000 if ratio>=0.05 else 2500",
    }
    return min_len, meta


def _read_contig_lengths(contigs_fasta: Path) -> dict[str, int]:
    contig_len: dict[str, int] = {}
    for name, _header, seq in iter_fasta_records(contigs_fasta):
        contig_len[name] = len(seq)
    return contig_len


def _read_bins_tsv(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if row[0] in {"contig_name", "contig"}:
                continue
            if len(row) < 2:
                raise RefineError(f"Invalid bins row in {path}: {row}")
            mapping[row[0]] = row[1]
    return mapping


def _bin_sizes(
    coarse: dict[str, str],
    contig_len: dict[str, int],
    *,
    min_contig_len: int,
) -> tuple[Counter[str], Counter[str]]:
    bp: Counter[str] = Counter()
    n: Counter[str] = Counter()
    for contig, bin_id in coarse.items():
        L = contig_len.get(contig)
        if L is None or L < min_contig_len:
            continue
        bp[bin_id] += L
        n[bin_id] += 1
    return bp, n


def _bin_sizes_from_assignment(
    contig_to_bin: dict[str, str],
    contig_len: dict[str, int],
) -> tuple[Counter[str], Counter[str]]:
    bp: Counter[str] = Counter()
    n: Counter[str] = Counter()
    for contig, bin_id in contig_to_bin.items():
        L = contig_len.get(contig)
        if L is None:
            continue
        bp[bin_id] += L
        n[bin_id] += 1
    return bp, n


def _drop_bins_below_min(
    bin_to_contigs: dict[str, list[str]],
    contig_to_bin: dict[str, str],
    keep_bins: set[str],
    unbinned_reason: dict[str, str],
) -> None:
    for bin_id in list(bin_to_contigs.keys()):
        if bin_id in keep_bins:
            continue
        for c in bin_to_contigs[bin_id]:
            contig_to_bin.pop(c, None)
            unbinned_reason[c] = "tiny_bin"
        bin_to_contigs.pop(bin_id, None)


def _coverage_from_bam(
    bam_path: Path,
    contig_len: dict[str, int],
    *,
    min_contig_len: int,
    logger: logging.Logger,
) -> dict[str, float]:
    """
    Coverage proxy (fast, no per-base depth): aligned_bases_sum / contig_length.
    """
    try:
        import pysam
    except Exception as exc:  # pragma: no cover
        raise RefineError("Coverage requested but 'pysam' is not installed. Install it or omit --bam.") from exc

    logger.info("Refine: computing coverage from BAM (aligned_bases_sum/contig_length)")
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
    for contig, L in contig_len.items():
        if L < min_contig_len:
            continue
        cov[contig] = aligned_bases.get(contig, 0) / float(L) if L else 0.0
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


def _bin_coverage_stats(
    bin_to_contigs: dict[str, list[str]],
    cov: dict[str, float],
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for bin_id, contigs in bin_to_contigs.items():
        vals = [cov.get(c, 0.0) for c in contigs]
        if not vals:
            continue
        med = _median(vals)
        mad = _mad(vals, center=med)
        stats[bin_id] = {"median": med, "mad": mad}
    return stats


def _scan_contacts_support_and_affinity(
    *,
    ppl_contacts: Path,
    contig_len: dict[str, int],
    min_contig_len: int,
    contig_to_bin: dict[str, str],
) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, float]], dict[str, Any]]:
    """
    Streaming pass over PPL contacts (segment-level TSV, passed-only) grouped by readID.
    Computes:
      - intra_support for contigs already assigned to keep bins
      - other_support (best other-bin support) for the same contigs
      - affinity scores for unbinned contigs towards keep bins
    """
    intra: defaultdict[str, float] = defaultdict(float)
    other: defaultdict[str, float] = defaultdict(float)
    affinity: dict[str, dict[str, float]] = {}

    reads_total = 0
    reads_kept = 0
    reads_skipped = 0
    input_sorted = True

    current_read: Optional[str] = None
    current_contigs: list[str] = []
    prev_read: Optional[str] = None

    def flush() -> None:
        nonlocal reads_kept, reads_skipped
        if current_read is None:
            return
        contigs = dedupe_preserve_order(current_contigs)
        contigs = [c for c in contigs if contig_len.get(c, 0) >= min_contig_len]
        k = len(contigs)
        if k < 2:
            reads_skipped += 1
            return
        reads_kept += 1
        w = order_norm(k, method="pair")

        bin_counts: dict[str, int] = {}
        for c in contigs:
            b = contig_to_bin.get(c)
            if b is None:
                continue
            bin_counts[b] = bin_counts.get(b, 0) + 1
        if not bin_counts:
            return

        best_other: dict[str, int] = {}
        for c in contigs:
            b = contig_to_bin.get(c)
            if b is not None:
                same = max(0, bin_counts.get(b, 0) - 1)
                if same:
                    intra[c] += w * same
                if len(bin_counts) > 1:
                    if b not in best_other:
                        best_other[b] = max((cnt for bb, cnt in bin_counts.items() if bb != b), default=0)
                    if best_other[b]:
                        other[c] += w * best_other[b]
            else:
                top = affinity.get(c)
                if top is None:
                    top = {}
                    affinity[c] = top
                for bb, cnt in bin_counts.items():
                    top[bb] = top.get(bb, 0.0) + w * cnt
                if len(top) > 32:
                    _prune_topk(top, k=3)

    for contig, read_id, status in parse_contacts_stream(ppl_contacts):
        if str(status).strip().lower() != "passed":
            continue
        if prev_read is not None and read_id < prev_read:
            input_sorted = False
        prev_read = read_id

        if current_read is None:
            current_read = read_id
            reads_total += 1
        elif read_id != current_read:
            flush()
            current_read = read_id
            current_contigs = []
            reads_total += 1
        current_contigs.append(contig)
    flush()

    return dict(intra), dict(other), affinity, {
        "reads_total": reads_total,
        "reads_kept": reads_kept,
        "reads_skipped_k_lt_2": reads_skipped,
        "input_sorted_by_readid": input_sorted,
    }


def _iter_contacts_parquet(
    contacts_parquet: Path,
    *,
    batch_size: int,
) -> Iterable[tuple[list[str], list[float], float]]:
    """
    Stream contacts.parquet rows.

    Expected columns:
      - contigs: list[str]
      - contig_weights: list[float]  (pi_{r,c} normalized evidence shares; should sum to 1 per read)
      - weight: float (q(r), read quality weight; default 1.0 if missing)
    """
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover
        raise RefineError(
            "refine_parquet requires 'pyarrow' to read contacts.parquet. Install it, e.g. pip install pyarrow."
        ) from exc

    parquet = pq.ParquetFile(contacts_parquet)
    schema = parquet.schema_arrow
    cols = set(schema.names)
    if "contigs" not in cols or "contig_weights" not in cols:
        raise RefineError(
            "contacts.parquet must contain columns: contigs, contig_weights. "
            f"Found: {schema.names}"
        )
    has_weight = "weight" in cols
    read_cols = ["contigs", "contig_weights"] + (["weight"] if has_weight else [])
    for batch in parquet.iter_batches(batch_size=int(batch_size), columns=read_cols):
        data = batch.to_pydict()
        contigs_list = data["contigs"]
        weights_list = data["contig_weights"]
        w_list = data.get("weight")
        n = len(contigs_list)
        for i in range(n):
            contigs_raw = contigs_list[i] or []
            p_raw = weights_list[i] or []
            if not isinstance(contigs_raw, list) or not isinstance(p_raw, list):
                continue
            if len(p_raw) != len(contigs_raw):
                continue
            contigs = [str(c) for c in contigs_raw]
            p = [float(x) for x in p_raw]
            w = float(w_list[i]) if w_list is not None and w_list[i] is not None else 1.0
            yield contigs, p, w


def _scan_contacts_support_and_affinity_parquet(
    *,
    contacts_parquet: Path,
    contig_len: dict[str, int],
    min_contig_len: int,
    contig_to_bin: dict[str, str],
) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, float]], dict[str, Any]]:
    """
    Streaming pass over BAM-derived contacts.parquet (one row per read/contact).

    This implements the soft-hyperedge support in docs/BAM_HYPERGRAPH_SPECTRAL_PIPELINE.md:
      - P_{r,c} is contig_weights (renormalized after filtering)
      - w(r) = OrderNorm(k) * q(r) where q(r) is contact weight
      - Support towards a bin uses S(v,b)=sum_r w(r) P_{r,v} * (sum_{u in b} P_{r,u})
    """
    intra: defaultdict[str, float] = defaultdict(float)
    other: defaultdict[str, float] = defaultdict(float)
    affinity: dict[str, dict[str, float]] = {}

    reads_total = 0
    reads_kept = 0
    reads_skipped = 0

    batch_size = 100_000

    for contigs_raw, p_raw, R in _iter_contacts_parquet(contacts_parquet, batch_size=batch_size):
        reads_total += 1

        # Filter contigs by length and merge duplicates by summing P.
        seen: dict[str, int] = {}
        contigs: list[str] = []
        p: list[float] = []
        for c, w in zip(contigs_raw, p_raw, strict=True):
            if not c:
                continue
            if contig_len.get(c, 0) < min_contig_len:
                continue
            ww = float(w)
            if ww <= 0.0:
                continue
            if c in seen:
                p[seen[c]] += ww
            else:
                seen[c] = len(contigs)
                contigs.append(c)
                p.append(ww)

        if len(contigs) < 2:
            reads_skipped += 1
            continue

        s = float(sum(p))
        if not (s > 0.0):
            reads_skipped += 1
            continue
        p = [x / s for x in p]

        k = len(contigs)
        reads_kept += 1
        w_contact = order_norm(k, method="pair") * float(R)

        bin_mass: dict[str, float] = {}
        for c, pc in zip(contigs, p, strict=True):
            b = contig_to_bin.get(c)
            if b is None:
                continue
            bin_mass[b] = bin_mass.get(b, 0.0) + float(pc)
        if not bin_mass:
            continue

        # Track top-2 bin masses for "best other" without per-contig dicts.
        max1_bin = None
        max1 = -1.0
        max2_bin = None
        max2 = -1.0
        for bb, m in bin_mass.items():
            m = float(m)
            if m > max1:
                max2, max2_bin = max1, max1_bin
                max1, max1_bin = m, bb
            elif m > max2:
                max2, max2_bin = m, bb

        for c, pc in zip(contigs, p, strict=True):
            b0 = contig_to_bin.get(c)
            if b0 is not None:
                m0 = float(bin_mass.get(b0, 0.0))
                intra[c] += float(w_contact) * float(pc) * max(0.0, m0 - float(pc))
                if len(bin_mass) > 1:
                    if max1_bin != b0:
                        other[c] += float(w_contact) * float(pc) * float(max1)
                    elif max2_bin is not None and max2 > 0.0:
                        other[c] += float(w_contact) * float(pc) * float(max2)
            else:
                top = affinity.get(c)
                if top is None:
                    top = {}
                    affinity[c] = top
                for bb, m in bin_mass.items():
                    top[bb] = top.get(bb, 0.0) + float(w_contact) * float(pc) * float(m)
                if len(top) > 32:
                    _prune_topk(top, k=3)

    return dict(intra), dict(other), affinity, {
        "reads_total": reads_total,
        "reads_kept": reads_kept,
        "reads_skipped_k_lt_2": reads_skipped,
        "input_sorted_by_readid": True,
        "source": "contacts_parquet",
    }


def _fit_2gmm_params(xs: list[float]) -> dict[str, float]:
    """
    Fit a 1D 2-component Gaussian mixture with EM.
    Returns parameters: w1, mu1, var1, mu2, var2.
    """
    if not xs:
        return {"w1": 0.5, "mu1": 0.0, "var1": 1.0, "mu2": 0.0, "var2": 1.0}
    n = len(xs)
    xs_sorted = sorted(xs)
    mu1 = float(xs_sorted[n // 4])
    mu2 = float(xs_sorted[(3 * n) // 4])
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / n
    var1 = var2 = max(float(var), 1e-3)
    w1 = 0.5

    for _ in range(50):
        r1_sum = 0.0
        r2_sum = 0.0
        mu1_num = 0.0
        mu2_num = 0.0
        var1_num = 0.0
        var2_num = 0.0

        # E-step: responsibilities
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

        # M-step
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

    return {"w1": float(w1), "mu1": float(mu1), "var1": float(var1), "mu2": float(mu2), "var2": float(var2)}


def _gmm_post_comp1(x: float, *, w1: float, mu1: float, var1: float, mu2: float, var2: float) -> float:
    l1 = math.log(float(w1)) + _log_norm_pdf(float(x), float(mu1), float(var1))
    l2 = math.log(1.0 - float(w1)) + _log_norm_pdf(float(x), float(mu2), float(var2))
    m = max(l1, l2)
    d = math.exp(l1 - m) + math.exp(l2 - m)
    return float(math.exp(l1 - m) / d)


def _reassign_or_unbin(
    *,
    contacts_parquet: Path,
    contig_len: dict[str, int],
    min_contig_len: int,
    contig_to_bin: dict[str, str],
    bin_to_contigs: dict[str, list[str]],
    intra_support: dict[str, float],
    other_support: dict[str, float],
    cov: Optional[dict[str, float]],
    bin_cov_stats: Optional[dict[str, dict[str, float]]],
    out_path: Path,
    logger: logging.Logger,
) -> dict[str, Any]:
    """
    Reassign (or unbin) contigs based on a data-driven split of LLR=log(intra)-log(other) via 1v2-GMM BIC.

    If 2-component model is preferred, contigs in the lower-mean component are treated as "suspects".
    For suspects, we find the best alternative bin by rescanning contacts.parquet and accumulating S(v,b).
    """
    eps = 1e-12
    llr_by_contig: dict[str, float] = {}
    xs: list[float] = []
    for c, b in contig_to_bin.items():
        if contig_len.get(c, 0) < min_contig_len:
            continue
        s_intra = float(intra_support.get(c, 0.0))
        s_other = float(other_support.get(c, 0.0))
        if s_other <= 0.0:
            continue
        llr = math.log(s_intra + eps) - math.log(s_other + eps)
        llr_by_contig[c] = float(llr)
        xs.append(float(llr))

    meta: dict[str, Any] = {
        "method": "llr_gmm_bic_1v2",
        "llr": "LLR(v)=log(intra_support+eps)-log(best_other_support+eps)",
        "n": int(len(xs)),
        "bic1": None,
        "bic2": None,
        "enabled": False,
        "moved": 0,
        "unbinned": 0,
    }

    # Default: no changes, but still write the file for traceability.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(
            "contig_name\told_bin\tnew_bin\taction\tllr\tintra_support\tother_support\tbest_bin_support\tcov\tcov_z\treason\n"
        )

        if len(xs) < 200:
            meta["reason"] = "skip_small_n"
            return meta

        bic1 = _bic(xs, k=1)
        bic2 = _bic(xs, k=2)
        meta["bic1"] = float(bic1)
        meta["bic2"] = float(bic2)
        if bic2 >= bic1:
            meta["reason"] = "bic_prefers_1_component"
            return meta

        params = _fit_2gmm_params(xs)
        # Identify low-mean component.
        mu1 = float(params["mu1"])
        mu2 = float(params["mu2"])
        low_is_1 = mu1 <= mu2

        suspects: set[str] = set()
        for c, llr in llr_by_contig.items():
            r1 = _gmm_post_comp1(llr, **params)
            r_low = r1 if low_is_1 else (1.0 - r1)
            if r_low > 0.5:
                suspects.add(c)

        meta["enabled"] = True
        meta["params"] = params
        meta["low_component"] = 1 if low_is_1 else 2
        meta["suspects"] = int(len(suspects))

        if not suspects:
            meta["reason"] = "no_suspects"
            return meta

        # Rescan contacts.parquet to find best alternative bin for suspects.
        best_scores: dict[str, dict[str, float]] = {c: {} for c in suspects}
        for contigs_raw, p_raw, R in _iter_contacts_parquet(contacts_parquet, batch_size=100_000):
            # Filter/normalize (same as scan pass)
            seen: dict[str, int] = {}
            contigs: list[str] = []
            p: list[float] = []
            for cc, ww in zip(contigs_raw, p_raw, strict=True):
                if not cc:
                    continue
                if contig_len.get(cc, 0) < min_contig_len:
                    continue
                wv = float(ww)
                if wv <= 0.0:
                    continue
                if cc in seen:
                    p[seen[cc]] += wv
                else:
                    seen[cc] = len(contigs)
                    contigs.append(cc)
                    p.append(wv)
            if len(contigs) < 2:
                continue
            s = float(sum(p))
            if not (s > 0.0):
                continue
            p = [x / s for x in p]
            k = len(contigs)
            w_contact = order_norm(k, method="pair") * float(R)

            bin_mass: dict[str, float] = {}
            for cc, pc in zip(contigs, p, strict=True):
                bb = contig_to_bin.get(cc)
                if bb is None:
                    continue
                bin_mass[bb] = bin_mass.get(bb, 0.0) + float(pc)
            if not bin_mass:
                continue

            for cc, pc in zip(contigs, p, strict=True):
                if cc not in suspects:
                    continue
                old_b = contig_to_bin.get(cc)
                if old_b is None:
                    continue
                top = best_scores.get(cc)
                if top is None:
                    continue
                for bb, m in bin_mass.items():
                    if bb == old_b:
                        continue
                    top[bb] = top.get(bb, 0.0) + float(w_contact) * float(pc) * float(m)
                if len(top) > 16:
                    _prune_topk(top, k=3)

        # Apply moves
        moved = 0
        unbinned = 0
        new_assign: dict[str, str] = dict(contig_to_bin)
        for c in suspects:
            old_b = contig_to_bin.get(c)
            if old_b is None:
                continue
            cand = best_scores.get(c, {})
            best_b = None
            best_s = 0.0
            for bb, ss in cand.items():
                if float(ss) > float(best_s):
                    best_b = bb
                    best_s = float(ss)

            s_intra = float(intra_support.get(c, 0.0))
            s_other = float(other_support.get(c, 0.0))
            llr = float(llr_by_contig.get(c, 0.0))

            cov_val = cov.get(c) if cov is not None else None
            cov_z = None
            if cov_val is not None and bin_cov_stats is not None and old_b in bin_cov_stats:
                cov_med = float(bin_cov_stats[old_b].get("median", 0.0))
                cov_mad = float(bin_cov_stats[old_b].get("mad", 0.0))
                denom = cov_mad if cov_mad > 0 else 1e-12
                cov_z = abs(float(cov_val) - cov_med) / denom

            if best_b is not None and best_s > 0.0:
                new_assign[c] = str(best_b)
                moved += 1
                fh.write(
                    f"{c}\t{old_b}\t{best_b}\tmoved\t{llr:.6g}\t{s_intra:.6g}\t{s_other:.6g}\t{best_s:.6g}\t"
                    f"{'' if cov_val is None else f'{cov_val:.6g}'}\t"
                    f"{'' if cov_z is None else f'{cov_z:.6g}'}\tllr_low_component\n"
                )
            else:
                new_assign.pop(c, None)
                unbinned += 1
                fh.write(
                    f"{c}\t{old_b}\t\tunbinned\t{llr:.6g}\t{s_intra:.6g}\t{s_other:.6g}\t\t"
                    f"{'' if cov_val is None else f'{cov_val:.6g}'}\t"
                    f"{'' if cov_z is None else f'{cov_z:.6g}'}\tllr_low_component_no_target\n"
                )

        contig_to_bin.clear()
        contig_to_bin.update(new_assign)
        bin_to_contigs.clear()
        for c, b in contig_to_bin.items():
            bin_to_contigs.setdefault(b, []).append(c)

        meta["moved"] = int(moved)
        meta["unbinned"] = int(unbinned)
        meta["reason"] = "applied"
        return meta


def _recruit_gmm(
    *,
    affinity: dict[str, dict[str, float]],
    contig_len: dict[str, int],
    min_contig_len: int,
    contig_to_bin: dict[str, str],
    bin_to_contigs: dict[str, list[str]],
    keep_bins: set[str],
    cov: Optional[dict[str, float]],
    bin_cov_stats: Optional[dict[str, dict[str, float]]],
    out_path: Path,
) -> dict[str, Any]:
    """
    Recruit unbinned contigs using a data-driven split of LLR=log(top1)-log(top2) via 1v2-GMM BIC.
    """
    eps = 1e-12
    candidates: dict[str, tuple[str, float, float, float]] = {}
    xs: list[float] = []
    for c, scores in affinity.items():
        if c in contig_to_bin or contig_len.get(c, 0) < min_contig_len:
            continue
        best = _top_candidates(scores, keep_bins=keep_bins, topk=2)
        if not best:
            continue
        b1, s1 = best[0]
        s2 = best[1][1] if len(best) > 1 else 0.0
        llr = math.log(float(s1) + eps) - math.log(float(s2) + eps)
        candidates[c] = (b1, float(s1), float(s2), float(llr))
        xs.append(float(llr))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("contig_name\tbin_id\tscore1\tscore2\tllr\taction\tcov\tcov_z\treason\n")

        meta: dict[str, Any] = {"method": "llr_gmm_bic_1v2", "n": int(len(xs)), "bic1": None, "bic2": None}
        if len(xs) < 200:
            meta["reason"] = "skip_small_n"
            return {"assigned": [], **meta}

        bic1 = _bic(xs, k=1)
        bic2 = _bic(xs, k=2)
        meta["bic1"] = float(bic1)
        meta["bic2"] = float(bic2)
        if bic2 >= bic1:
            meta["reason"] = "bic_prefers_1_component"
            return {"assigned": [], **meta}

        params = _fit_2gmm_params(xs)
        mu1 = float(params["mu1"])
        mu2 = float(params["mu2"])
        high_is_1 = mu1 >= mu2

        assigned: list[str] = []
        for c, (b1, s1, s2, llr) in candidates.items():
            r1 = _gmm_post_comp1(llr, **params)
            r_high = r1 if high_is_1 else (1.0 - r1)
            if r_high <= 0.5:
                continue

            cov_val = cov.get(c) if cov is not None else None
            cov_z = None
            if cov_val is not None and bin_cov_stats is not None and b1 in bin_cov_stats:
                cov_med = float(bin_cov_stats[b1].get("median", 0.0))
                cov_mad = float(bin_cov_stats[b1].get("mad", 0.0))
                denom = cov_mad if cov_mad > 0 else 1e-12
                cov_z = abs(float(cov_val) - cov_med) / denom

            contig_to_bin[c] = b1
            bin_to_contigs.setdefault(b1, []).append(c)
            assigned.append(c)
            fh.write(
                f"{c}\t{b1}\t{s1:.6g}\t{s2:.6g}\t{llr:.6g}\tassigned\t"
                f"{'' if cov_val is None else f'{cov_val:.6g}'}\t"
                f"{'' if cov_z is None else f'{cov_z:.6g}'}\tllr_high_component\n"
            )

        meta["params"] = params
        meta["high_component"] = 1 if high_is_1 else 2
        meta["assigned"] = int(len(assigned))
        meta["reason"] = "applied"
        return {"assigned": assigned, **meta}


def _split_bins_parquet(
    *,
    contacts_parquet: Path,
    contig_len: dict[str, int],
    min_contig_len: int,
    contig_to_bin: dict[str, str],
    bin_to_contigs: dict[str, list[str]],
    cov: Optional[dict[str, float]],
    out_path: Path,
    seed: int,
) -> dict[str, Any]:
    """
    Optional refinement split step (coverage BIC trigger), implemented on contacts.parquet hyperedges.

    Trigger:
      - If cov is provided, for each bin: compare BIC(1-Gauss) vs BIC(2-GMM) on log1p(cov).
    Partition:
      - Build an induced bipartite graph (contig-contact) inside triggered bins using weighted incidences.
      - Run Leiden modularity partition and keep sub-bins >= MIN_BIN_BP.
    """
    import leidenalg

    meta = {
        "enabled": cov is not None,
        "trigger": {"type": "gmm_bic_1v2_on_log1p_cov", "rule": "trigger if BIC2 < BIC1"},
        "partition": {"type": "leiden_modularity_on_induced_bipartite", "seed": seed},
        "min_subbin_bp": MIN_BIN_BP,
    }

    if cov is None:
        with out_path.open("w", encoding="utf-8", newline="") as fh:
            fh.write("contig_name\told_bin\tnew_bin\taction\treason\n")
        write_json(out_path.with_suffix(".json"), {"meta": meta, "bic": {}, "bins_triggered": []})
        return {
            "bins_triggered": 0,
            "bins_split": 0,
            "contigs_unbinned": set(),
            "meta": meta,
            "bins_triggered_list": [],
        }

    bic_table: dict[str, dict[str, float]] = {}
    bins_triggered: list[str] = []
    for bin_id, contigs in bin_to_contigs.items():
        vals = [cov.get(c, 0.0) for c in contigs if contig_len.get(c, 0) >= min_contig_len]
        if len(vals) < 20:
            continue
        xs = [math.log1p(v) for v in vals]
        bic1 = _bic(xs, k=1)
        bic2 = _bic(xs, k=2)
        bic_table[bin_id] = {"bic1": bic1, "bic2": bic2, "delta": bic1 - bic2}
        if bic2 < bic1:
            bins_triggered.append(bin_id)

    if not bins_triggered:
        with out_path.open("w", encoding="utf-8", newline="") as fh:
            fh.write("contig_name\told_bin\tnew_bin\taction\treason\n")
        write_json(out_path.with_suffix(".json"), {"meta": meta, "bic": bic_table, "bins_triggered": []})
        return {
            "bins_triggered": 0,
            "bins_split": 0,
            "contigs_unbinned": set(),
            "meta": meta,
            "bins_triggered_list": [],
        }

    builders: dict[str, _InducedBuilder] = {}
    for bin_id in bins_triggered:
        contigs = [c for c in bin_to_contigs.get(bin_id, []) if contig_len.get(c, 0) >= min_contig_len]
        if len(contigs) >= 2:
            builders[bin_id] = _InducedBuilder(bin_id, contigs)

    contig_to_trigger_bin: dict[str, str] = {}
    for bin_id, bld in builders.items():
        for c in bld.contigs:
            contig_to_trigger_bin[c] = bin_id

    # Stream contacts.parquet and add within-bin incidences.
    for contigs_raw, p_raw, R in _iter_contacts_parquet(contacts_parquet, batch_size=100_000):
        # filter / normalize
        seen: dict[str, int] = {}
        contigs: list[str] = []
        p: list[float] = []
        for c, w in zip(contigs_raw, p_raw, strict=True):
            if not c:
                continue
            if contig_len.get(c, 0) < min_contig_len:
                continue
            ww = float(w)
            if ww <= 0.0:
                continue
            if c in seen:
                p[seen[c]] += ww
            else:
                seen[c] = len(contigs)
                contigs.append(c)
                p.append(ww)
        if len(contigs) < 2:
            continue
        s = float(sum(p))
        if not (s > 0.0):
            continue
        p = [x / s for x in p]
        k = len(contigs)
        w_contact = order_norm(k, method="pair") * float(R)

        by_bin: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for c, pc in zip(contigs, p, strict=True):
            b = contig_to_trigger_bin.get(c)
            if b is not None:
                by_bin[b].append((c, float(w_contact) * float(pc)))
        for b, items in by_bin.items():
            if len(items) >= 2:
                builders[b].add_contact_weighted(items)

    next_bin_id = _next_bin_id(contig_to_bin.values())
    contigs_unbinned: set[str] = set()
    bins_split = 0

    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("contig_name\told_bin\tnew_bin\taction\treason\n")
        for bin_id, bld in builders.items():
            if bld.n_contacts < 2:
                continue
            g = bld.to_igraph()
            partition = leidenalg.find_partition(
                g,
                leidenalg.ModularityVertexPartition,
                weights="weight",
                seed=seed,
            )
            membership = partition.membership[: bld.n_contigs]
            comm_to_contigs: dict[int, list[str]] = defaultdict(list)
            for c, m in zip(bld.contigs, membership, strict=True):
                comm_to_contigs[int(m)].append(c)

            comm_bp: dict[int, int] = {
                comm: sum(contig_len.get(c, 0) for c in cs) for comm, cs in comm_to_contigs.items()
            }
            kept_comms = [comm for comm, bp in comm_bp.items() if bp >= MIN_BIN_BP]
            if len(kept_comms) < 2:
                for c in bld.contigs:
                    fh.write(f"{c}\t{bin_id}\t\tkept\tno_split\n")
                continue

            bins_split += 1
            new_bins: dict[int, str] = {}
            for comm in sorted(kept_comms, key=lambda x: comm_bp[x], reverse=True):
                new_bins[comm] = str(next_bin_id)
                next_bin_id += 1

            # Remove old bin and replace with new bins.
            for c in list(bin_to_contigs.get(bin_id, [])):
                contig_to_bin.pop(c, None)
            bin_to_contigs.pop(bin_id, None)

            for comm, cs in comm_to_contigs.items():
                if comm in new_bins:
                    nb = new_bins[comm]
                    for c in cs:
                        contig_to_bin[c] = nb
                        bin_to_contigs[nb].append(c)
                        fh.write(f"{c}\t{bin_id}\t{nb}\tsplit\tcommunity\n")
                else:
                    for c in cs:
                        contigs_unbinned.add(c)
                        fh.write(f"{c}\t{bin_id}\t\tunbinned\tsplit_tiny\n")

    write_json(out_path.with_suffix(".json"), {"meta": meta, "bic": bic_table, "bins_triggered": bins_triggered})
    return {
        "bins_triggered": len(bins_triggered),
        "bins_split": bins_split,
        "contigs_unbinned": contigs_unbinned,
        "meta": meta,
        "bins_triggered_list": bins_triggered,
    }

def _decontam(
    *,
    bin_to_contigs: dict[str, list[str]],
    contig_len: dict[str, int],
    contig_to_bin: dict[str, str],
    intra_support: dict[str, float],
    other_support: dict[str, float],
    cov: Optional[dict[str, float]],
    bin_cov_stats: Optional[dict[str, dict[str, float]]],
    out_path: Path,
) -> set[str]:
    """
    Remove contigs with low intra_support using per-bin median/MAD rule.
    Coverage (if provided) is used as a veto: cov outlier + weak support => remove.
    """
    removed: set[str] = set()
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("contig_name\tbin_id\tlength\tintra_support\tthreshold\tcov\tcov_z\treason\n")
        for bin_id, contigs in list(bin_to_contigs.items()):
            vals = [float(intra_support.get(c, 0.0)) for c in contigs]
            if not vals:
                continue
            med = _median(vals)
            mad = _mad(vals, center=med)
            thr = max(0.0, med - 3.0 * mad)

            cov_med = cov_mad = None
            if cov is not None and bin_cov_stats is not None and bin_id in bin_cov_stats:
                cov_med = float(bin_cov_stats[bin_id].get("median", 0.0))
                cov_mad = float(bin_cov_stats[bin_id].get("mad", 0.0))

            keep: list[str] = []
            for c in contigs:
                s_intra = float(intra_support.get(c, 0.0))
                remove = s_intra < thr
                reason = "low_intra_support" if remove else ""

                cov_z = None
                cov_val = cov.get(c) if cov is not None else None
                if cov_val is not None and cov_med is not None and cov_mad is not None:
                    denom = cov_mad if cov_mad > 0 else 1e-12
                    cov_z = abs(float(cov_val) - cov_med) / denom
                    if cov_z > 3.0 and s_intra < med:
                        remove = True
                        reason = reason + "|cov_outlier" if reason else "cov_outlier"

                if remove:
                    removed.add(c)
                    contig_to_bin.pop(c, None)
                    fh.write(
                        f"{c}\t{bin_id}\t{contig_len.get(c, 0)}\t{s_intra:.6g}\t{thr:.6g}\t"
                        f"{'' if cov_val is None else f'{cov_val:.6g}'}\t"
                        f"{'' if cov_z is None else f'{cov_z:.6g}'}\t{reason}\n"
                    )
                else:
                    keep.append(c)
            bin_to_contigs[bin_id] = keep
    return removed


def _recruit(
    *,
    affinity: dict[str, dict[str, float]],
    contig_len: dict[str, int],
    min_contig_len: int,
    contig_to_bin: dict[str, str],
    bin_to_contigs: dict[str, list[str]],
    keep_bins: set[str],
    cov: Optional[dict[str, float]],
    bin_cov_stats: Optional[dict[str, dict[str, float]]],
    out_path: Path,
) -> dict[str, Any]:
    """
    Recruit unbinned contigs to kept bins.
      - keep topK=3 candidates
      - threshold T from Otsu on log1p(top1_score)
      - ratio threshold fixed at 3.0 (recorded in run_refine.json)
      - optional coverage veto: abs(cov_u - bin_median) <= 3*MAD
    """
    top1_scores = []
    for c, scores in affinity.items():
        if c in contig_to_bin or contig_len.get(c, 0) < min_contig_len:
            continue
        best = _top_candidates(scores, keep_bins=keep_bins, topk=1)
        if best:
            top1_scores.append(best[0][1])

    threshold, thr_meta = _auto_threshold_otsu(top1_scores)
    ratio_threshold = 3.0

    assigned: set[str] = set()
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(
            "contig_name\tassigned_bin\ttop1_bin\ttop1_score\ttop2_score\tratio\tthreshold\tcov\tveto_reason\n"
        )
        for c, scores in affinity.items():
            if c in contig_to_bin or contig_len.get(c, 0) < min_contig_len:
                continue
            cand = _top_candidates(scores, keep_bins=keep_bins, topk=3)
            if not cand:
                continue
            b1, s1 = cand[0]
            s2 = cand[1][1] if len(cand) > 1 else 0.0
            ratio = (s1 / s2) if s2 > 0 else float("inf")

            veto_reason = ""
            cov_val = cov.get(c) if cov is not None else None
            if cov_val is not None and bin_cov_stats is not None and b1 in bin_cov_stats:
                bmed = float(bin_cov_stats[b1].get("median", 0.0))
                bmad = float(bin_cov_stats[b1].get("mad", 0.0))
                denom = bmad if bmad > 0 else 1e-12
                if abs(float(cov_val) - bmed) > 3.0 * denom:
                    veto_reason = "cov_veto"

            ok = (s1 >= threshold) and (ratio >= ratio_threshold) and not veto_reason
            fh.write(
                f"{c}\t{b1 if ok else ''}\t{b1}\t{s1:.6g}\t{s2:.6g}\t{ratio:.6g}\t{threshold:.6g}\t"
                f"{'' if cov_val is None else f'{cov_val:.6g}'}\t{veto_reason}\n"
            )
            if ok:
                contig_to_bin[c] = b1
                bin_to_contigs[b1].append(c)
                assigned.add(c)

    return {"assigned": assigned, "threshold": threshold, "threshold_method": thr_meta}


def _split_bins(
    *,
    ppl_contacts: Path,
    contig_len: dict[str, int],
    min_contig_len: int,
    contig_to_bin: dict[str, str],
    bin_to_contigs: dict[str, list[str]],
    cov: Optional[dict[str, float]],
    out_path: Path,
    seed: int,
) -> dict[str, Any]:
    """
    Split bins based on coverage multi-modality (BIC: 1-Gaussian vs 2-GMM on log1p(cov)).
    If triggered, build an induced contig-contact bipartite graph and partition it with Leiden modularity
    (no resolution parameter).

    Rule: apply split only if >=2 sub-bins each >= MIN_BIN_BP; otherwise revoke.
    """
    meta: dict[str, Any] = {"enabled": False}
    if cov is None:
        with out_path.open("w", encoding="utf-8", newline="") as fh:
            fh.write("contig_name\told_bin\tnew_bin\taction\treason\n")
        return {
            "bins_triggered": 0,
            "bins_split": 0,
            "contigs_unbinned": set(),
            "meta": meta,
        }

    try:
        import igraph as ig  # noqa: F401
        import leidenalg
    except Exception as exc:  # pragma: no cover
        raise RefineError(
            "split requires 'python-igraph' and 'leidenalg' (already required by porebin)."
        ) from exc

    meta = {
        "enabled": True,
        "trigger": {"type": "gmm_bic_1v2_on_log1p_cov", "rule": "trigger if BIC2 < BIC1"},
        "partition": {"type": "leiden_modularity", "seed": seed},
        "min_subbin_bp": MIN_BIN_BP,
    }

    bic_table: dict[str, dict[str, float]] = {}
    bins_triggered: list[str] = []
    for bin_id, contigs in bin_to_contigs.items():
        vals = [cov.get(c, 0.0) for c in contigs if contig_len.get(c, 0) >= min_contig_len]
        if len(vals) < 20:
            continue
        xs = [math.log1p(v) for v in vals]
        bic1 = _bic(xs, k=1)
        bic2 = _bic(xs, k=2)
        bic_table[bin_id] = {"bic1": bic1, "bic2": bic2, "delta": bic1 - bic2}
        if bic2 < bic1:
            bins_triggered.append(bin_id)

    if not bins_triggered:
        with out_path.open("w", encoding="utf-8", newline="") as fh:
            fh.write("contig_name\told_bin\tnew_bin\taction\treason\n")
        write_json(out_path.with_suffix(".json"), {"meta": meta, "bic": bic_table, "bins_triggered": []})
        return {
            "bins_triggered": 0,
            "bins_split": 0,
            "contigs_unbinned": set(),
            "meta": meta,
            "bins_triggered_list": [],
            "bic_top": [],
        }

    builders: dict[str, _InducedBuilder] = {}
    for bin_id in bins_triggered:
        contigs = [c for c in bin_to_contigs.get(bin_id, []) if contig_len.get(c, 0) >= min_contig_len]
        if len(contigs) >= 2:
            builders[bin_id] = _InducedBuilder(bin_id, contigs)

    contig_to_trigger_bin: dict[str, str] = {}
    for bin_id, bld in builders.items():
        for c in bld.contigs:
            contig_to_trigger_bin[c] = bin_id

    current_read: Optional[str] = None
    current_contigs: list[str] = []

    def flush_contact() -> None:
        if current_read is None:
            return
        contigs = dedupe_preserve_order(current_contigs)
        contigs = [c for c in contigs if contig_len.get(c, 0) >= min_contig_len]
        if len(contigs) < 2:
            return
        by_bin: dict[str, list[str]] = defaultdict(list)
        for c in contigs:
            b = contig_to_trigger_bin.get(c)
            if b is not None:
                by_bin[b].append(c)
        for b, cs in by_bin.items():
            if len(cs) >= 2:
                builders[b].add_contact(cs, order_norm(len(cs), method="pair"))

    for contig, read_id, status in parse_contacts_stream(ppl_contacts):
        if str(status).strip().lower() != "passed":
            continue
        if current_read is None:
            current_read = read_id
        elif read_id != current_read:
            flush_contact()
            current_read = read_id
            current_contigs = []
        current_contigs.append(contig)
    flush_contact()

    next_bin_id = _next_bin_id(contig_to_bin.values())
    contigs_unbinned: set[str] = set()
    bins_split = 0

    with out_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("contig_name\told_bin\tnew_bin\taction\treason\n")
        for bin_id, bld in builders.items():
            if bld.n_contacts < 2:
                continue
            g = bld.to_igraph()
            partition = leidenalg.find_partition(
                g,
                leidenalg.ModularityVertexPartition,
                weights="weight",
                seed=seed,
            )
            membership = partition.membership[: bld.n_contigs]
            comm_to_contigs: dict[int, list[str]] = defaultdict(list)
            for c, m in zip(bld.contigs, membership, strict=True):
                comm_to_contigs[int(m)].append(c)

            comm_bp: dict[int, int] = {
                comm: sum(contig_len.get(c, 0) for c in cs) for comm, cs in comm_to_contigs.items()
            }
            kept_comms = [comm for comm, bp in comm_bp.items() if bp >= MIN_BIN_BP]
            if len(kept_comms) < 2:
                for c in bld.contigs:
                    fh.write(f"{c}\t{bin_id}\t\tkept\tno_split\n")
                continue

            bins_split += 1
            new_bins: dict[int, str] = {}
            for comm in sorted(kept_comms, key=lambda x: comm_bp[x], reverse=True):
                new_bins[comm] = str(next_bin_id)
                next_bin_id += 1

            # Remove old bin and replace with new bins.
            for c in list(bin_to_contigs.get(bin_id, [])):
                contig_to_bin.pop(c, None)
            bin_to_contigs.pop(bin_id, None)

            for comm, cs in comm_to_contigs.items():
                if comm in new_bins:
                    nb = new_bins[comm]
                    for c in cs:
                        contig_to_bin[c] = nb
                        bin_to_contigs[nb].append(c)
                        fh.write(f"{c}\t{bin_id}\t{nb}\tsplit\tcommunity\n")
                else:
                    for c in cs:
                        contigs_unbinned.add(c)
                        fh.write(f"{c}\t{bin_id}\t\tunbinned\tsplit_tiny\n")

    write_json(out_path.with_suffix(".json"), {"meta": meta, "bic": bic_table, "bins_triggered": bins_triggered})
    return {
        "bins_triggered": len(bins_triggered),
        "bins_split": bins_split,
        "contigs_unbinned": contigs_unbinned,
        "meta": meta,
        "bins_triggered_list": bins_triggered,
        "bic_top": [
            {"bin_id": b, **m}
            for b, m in sorted(bic_table.items(), key=lambda kv: kv[1]["delta"], reverse=True)[:50]
        ],
    }


class _InducedBuilder:
    def __init__(self, bin_id: str, contigs: list[str]):
        self.bin_id = bin_id
        self.contigs = contigs
        self.contig_to_idx = {c: i for i, c in enumerate(contigs)}
        self.n_contigs = len(contigs)
        self.edges: list[tuple[int, int]] = []
        self.weights: list[float] = []
        self.n_contacts = 0

    def add_contact(self, contigs_in_bin: list[str], w: float) -> None:
        contact_node = self.n_contigs + self.n_contacts
        self.n_contacts += 1
        for c in contigs_in_bin:
            idx = self.contig_to_idx.get(c)
            if idx is None:
                continue
            self.edges.append((idx, contact_node))
            self.weights.append(w)

    def add_contact_weighted(self, items: list[tuple[str, float]]) -> None:
        """
        Add a contact node where each incidence can have its own weight.

        This is used by the BAM/parquet refine split step, where we build an induced bipartite
        graph inside one coarse bin using weighted incidences:
          edge_weight(c,r) = OrderNorm(k(r)) * q(r) * P_{r,c}
        (see docs/BAM_HYPERGRAPH_SPECTRAL_PIPELINE.md).
        """
        contact_node = self.n_contigs + self.n_contacts
        self.n_contacts += 1
        for c, w in items:
            idx = self.contig_to_idx.get(c)
            if idx is None:
                continue
            ww = float(w)
            if ww <= 0.0:
                continue
            self.edges.append((idx, contact_node))
            self.weights.append(ww)

    def to_igraph(self):
        import igraph as ig

        g = ig.Graph(n=self.n_contigs + self.n_contacts, edges=self.edges, directed=False)
        g.es["weight"] = self.weights
        g.vs["type"] = [False] * self.n_contigs + [True] * self.n_contacts
        return g


def _next_bin_id(existing: Iterable[str]) -> int:
    mx = -1
    for b in existing:
        try:
            mx = max(mx, int(str(b)))
        except Exception:
            continue
    return mx + 1


def _top_candidates(
    scores: dict[str, float],
    *,
    keep_bins: set[str],
    topk: int,
) -> list[tuple[str, float]]:
    items = [(b, float(s)) for b, s in scores.items() if b in keep_bins and s > 0]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:topk]


def _prune_topk(scores: dict[str, float], *, k: int) -> None:
    best = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    scores.clear()
    scores.update(best)


def _auto_threshold_otsu(values: list[float]) -> tuple[float, dict[str, Any]]:
    if not values:
        return 0.0, {"type": "none", "note": "no candidates"}
    xs = [math.log1p(v) for v in values if v > 0]
    if len(xs) < 50:
        return math.expm1(_median(xs)), {"type": "fallback_median_log1p", "n": len(xs)}
    thr_log = _otsu_threshold(xs, bins=256)
    return math.expm1(thr_log), {"type": "otsu_log1p", "bins": 256, "n": len(xs)}


def _otsu_threshold(values: list[float], *, bins: int = 256) -> float:
    vmin = min(values)
    vmax = max(values)
    if vmin == vmax:
        return vmin
    step = (vmax - vmin) / bins
    hist = [0] * bins
    for v in values:
        idx = int((v - vmin) / step)
        if idx >= bins:
            idx = bins - 1
        hist[idx] += 1

    total = sum(hist)
    if total <= 0:
        return vmin
    prob = [h / total for h in hist]

    omega = [0.0] * bins
    mu = [0.0] * bins
    omega[0] = prob[0]
    mu[0] = prob[0] * 0.0
    for i in range(1, bins):
        omega[i] = omega[i - 1] + prob[i]
        mu[i] = mu[i - 1] + prob[i] * i

    mu_t = mu[-1]
    best_i = 0
    best_sigma = -1.0
    for i in range(bins - 1):
        if omega[i] <= 0.0 or omega[i] >= 1.0:
            continue
        mu0 = mu[i] / omega[i]
        mu1 = (mu_t - mu[i]) / (1.0 - omega[i])
        sigma = omega[i] * (1.0 - omega[i]) * (mu0 - mu1) ** 2
        if sigma > best_sigma:
            best_sigma = sigma
            best_i = i
    return vmin + (best_i + 0.5) * step


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return float(xs[mid])
    return 0.5 * (xs[mid - 1] + xs[mid])


def _mad(values: list[float], *, center: Optional[float] = None) -> float:
    if not values:
        return 0.0
    c = float(center) if center is not None else _median(values)
    return _median([abs(x - c) for x in values])


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
            # var numerator uses previous mu; update after mu update in next iteration

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
