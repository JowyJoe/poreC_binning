"""
Downstream association module for the current porebin architecture.

Public role:
  - consume `bins.refined.tsv`
  - consume `residual_pool.tsv`
  - emit `associations.tsv`

This module is intentionally one-way. It does not rewrite final bins and should
not be interpreted as part of the core bin-assignment stage.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

from porebin.utils import ensure_dir


class AssociateError(RuntimeError):
    pass


def associate_residuals(
    *,
    bins_refined_tsv: Path,
    residual_pool_tsv: Path,
    contacts_parquet: Path,
    out_dir: Path,
    contig_scores_tsv: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """
    Downstream residual association entrypoint.

    Primary output:
      - `associations.tsv`

    Compatibility / transition input:
      - optional `contig_host_scores.tsv`

    This module is intentionally one-way:
      - it consumes final refined bins plus the residual pool
      - it emits `associations.tsv`
      - it never rewrites `bins.refined.tsv`

    Phase-2 compatibility behavior:
      - if `contig_host_scores.tsv` is available, reuse its residual host-support summary
        to produce a compatibility association table
      - if compatibility scores are unavailable, emit an empty associations table with the
        correct schema rather than trying to reconstruct legacy relation mining implicitly
    """
    logger = logger or logging.getLogger("porebin")
    bins_refined_tsv = bins_refined_tsv.resolve()
    residual_pool_tsv = residual_pool_tsv.resolve()
    contacts_parquet = contacts_parquet.resolve()
    out_dir = out_dir.resolve()
    ensure_dir(out_dir)

    if not bins_refined_tsv.exists():
        raise FileNotFoundError(f"Refined bins TSV not found: {bins_refined_tsv}")
    if not residual_pool_tsv.exists():
        raise FileNotFoundError(f"Residual pool TSV not found: {residual_pool_tsv}")
    if not contacts_parquet.exists():
        raise FileNotFoundError(f"Contacts Parquet not found: {contacts_parquet}")

    final_bins = _read_final_bins(bins_refined_tsv)
    residual_rows = _read_residual_pool(residual_pool_tsv)
    scores = _read_contig_scores(contig_scores_tsv) if contig_scores_tsv is not None and contig_scores_tsv.exists() else {}

    assoc_path = out_dir / "associations.tsv"
    with assoc_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(
            [
                "contig_name",
                "association_type",
                "host_bin_ids",
                "primary_host_bin_id",
                "support_strength",
                "support_class",
                "uncertainty",
                "evidence_summary",
            ]
        )

        for row in residual_rows:
            if row.get("refined_status") != "residual_associate_candidate":
                continue
            contig = row["contig_name"]
            score = scores.get(contig)
            if score is None:
                continue

            host_ids = _compat_host_ids(score, final_bins=final_bins)
            if not host_ids:
                continue

            primary = host_ids[0]
            top1_score = _parse_float(score.get("top1_score"))
            entropy = _parse_float(score.get("entropy"))
            eff_hosts = _parse_float(score.get("effective_hosts"))
            association_type = _classify_association_type(host_ids=host_ids, effective_hosts=eff_hosts)
            support_class = _classify_support(top1_score)

            writer.writerow(
                [
                    contig,
                    association_type,
                    ",".join(host_ids),
                    primary,
                    _fmt_float(top1_score),
                    support_class,
                    _fmt_float(entropy),
                    "compat_from_contig_host_scores",
                ]
            )

    logger.info("Associate: wrote %s", assoc_path)
    return assoc_path


def _read_final_bins(path: Path) -> set[str]:
    bins: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if row[0] in {"contig_name", "contig"}:
                continue
            if len(row) < 2:
                raise AssociateError(f"Invalid refined bins row in {path}: {row}")
            bins.add(str(row[1]).strip())
    return bins


def _read_residual_pool(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        required = {"contig_name", "reason", "stage", "coarse_bin_id", "refined_status", "note"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise AssociateError(
                f"Residual pool schema mismatch in {path}. Expected at least: {sorted(required)}"
            )
        for row in reader:
            if not row:
                continue
            rows.append({k: str(v).strip() for k, v in row.items() if k is not None})
    return rows


def _read_contig_scores(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if not row:
                continue
            contig = str(row.get("contig_name", "")).strip()
            if not contig:
                continue
            out[contig] = {str(k): str(v).strip() for k, v in row.items() if k is not None}
    return out


def _compat_host_ids(score: dict[str, str], *, final_bins: set[str]) -> list[str]:
    host_ids: list[str] = []
    top1 = str(score.get("top1_host", "")).strip()
    top2 = str(score.get("top2_host", "")).strip()
    top2_score = _parse_float(score.get("top2_score"))

    if top1 and top1 != "-1" and top1 in final_bins:
        host_ids.append(top1)
    if top2 and top2 != "-1" and top2 not in host_ids and top2 in final_bins and (top2_score or 0.0) > 0.0:
        host_ids.append(top2)
    return host_ids


def _classify_association_type(*, host_ids: list[str], effective_hosts: Optional[float]) -> str:
    if len(host_ids) <= 1:
        return "single_host"
    if effective_hosts is not None and effective_hosts >= 2.0:
        return "broad_host"
    return "multi_host"


def _classify_support(value: Optional[float]) -> str:
    if value is None:
        return "provisional"
    if value >= 0.8:
        return "high"
    if value >= 0.5:
        return "moderate"
    if value > 0.0:
        return "low"
    return "provisional"


def _parse_float(raw: object) -> Optional[float]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.upper() == "NA":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _fmt_float(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.6g}"
