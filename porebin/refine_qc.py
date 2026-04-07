from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from porebin.contact_hypergraph import iter_canonical_contact_rows
from porebin.scg import BinScgQc


@dataclass(frozen=True)
class BinQcRow:
    bin_id: str
    total_bp: int
    n_contigs: int
    contact_consistency: Optional[float]
    coverage_dispersion: Optional[float]
    tnf_dispersion: Optional[float]
    scg_status: str
    unique_scg: Optional[int]
    duplicated_scg: Optional[int]
    completeness_like: Optional[float]
    contamination_like: Optional[float]
    suspect_flag: bool
    suspect_reasons: tuple[str, ...]
    split_check_flag: bool
    split_check_reasons: tuple[str, ...]
    split_priority_source: Optional[str]
    contact_components: int
    largest_component_share: float
    low_support_contigs: int
    low_support_ratio: float
    median_intra_support: float
    coverage_median: Optional[float]
    coverage_mad: Optional[float]
    scg_implicated_contigs: int


@dataclass(frozen=True)
class SuspectBinRecord:
    bin_id: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SplitCheckRecord:
    bin_id: str
    trigger_reasons: tuple[str, ...]
    auxiliary_reasons: tuple[str, ...]
    priority_source: str


def compute_contact_component_qc(
    *,
    contacts_parquet: Path,
    contig_to_bin: dict[str, str],
    contig_len: dict[str, int],
    min_contig_len: int,
) -> dict[str, tuple[int, float]]:
    """
    Return per-bin:
      - number of contact-connected components
      - largest component share in [0,1]
    """
    memberships = compute_contact_component_memberships(
        contacts_parquet=contacts_parquet,
        contig_to_bin=contig_to_bin,
        contig_len=contig_len,
        min_contig_len=min_contig_len,
    )

    out: dict[str, tuple[int, float]] = {}
    for bin_id, components in memberships.items():
        total = sum(len(component) for component in components)
        if total <= 0:
            out[bin_id] = (0, 0.0)
            continue
        largest = max((len(component) for component in components), default=0)
        out[bin_id] = (len(components), (largest / float(total)) if total > 0 else 0.0)
    return out


def compute_contact_component_memberships(
    *,
    contacts_parquet: Path,
    contig_to_bin: dict[str, str],
    contig_len: dict[str, int],
    min_contig_len: int,
) -> dict[str, tuple[tuple[str, ...], ...]]:
    """
    Return per-bin contact-connected components as tuples of contig names.
    Components are ordered by descending size, then lexical content.
    """
    eligible_by_bin: dict[str, list[str]] = defaultdict(list)
    for contig, bin_id in contig_to_bin.items():
        if contig_len.get(contig, 0) < min_contig_len:
            continue
        if not bin_id or str(bin_id).strip() == "-1":
            continue
        eligible_by_bin[str(bin_id)].append(contig)

    parent_by_bin: dict[str, dict[str, str]] = {}
    size_by_bin: dict[str, Counter[str]] = {}
    for bin_id, contigs in eligible_by_bin.items():
        parent_by_bin[bin_id] = {c: c for c in contigs}
        size_by_bin[bin_id] = Counter({c: 1 for c in contigs})

    def find(parent: dict[str, str], x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(bin_id: str, a: str, b: str) -> None:
        parent = parent_by_bin[bin_id]
        size = size_by_bin[bin_id]
        ra = find(parent, a)
        rb = find(parent, b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]
        del size[rb]

    for row in iter_canonical_contact_rows(contacts_parquet, require_contig_weights=False):
        by_bin: dict[str, list[str]] = defaultdict(list)
        for contig in row.contigs:
            if contig_len.get(contig, 0) < min_contig_len:
                continue
            bin_id = contig_to_bin.get(contig)
            if not bin_id or str(bin_id).strip() == "-1":
                continue
            if contig not in parent_by_bin.get(str(bin_id), {}):
                continue
            by_bin[str(bin_id)].append(contig)
        for bin_id, members in by_bin.items():
            if len(members) < 2:
                continue
            root = members[0]
            for contig in members[1:]:
                union(bin_id, root, contig)

    out: dict[str, tuple[tuple[str, ...], ...]] = {}
    for bin_id, contigs in eligible_by_bin.items():
        if not contigs:
            out[bin_id] = ()
            continue
        comp_sizes = size_by_bin.get(bin_id)
        parent = parent_by_bin.get(bin_id)
        if not comp_sizes or parent is None:
            out[bin_id] = ()
            continue
        groups: dict[str, list[str]] = defaultdict(list)
        for contig in contigs:
            groups[find(parent, contig)].append(contig)
        ordered = sorted(
            (tuple(sorted(group)) for group in groups.values()),
            key=lambda group: (-len(group), tuple(group)),
        )
        out[bin_id] = tuple(ordered)
    return out


def compute_bin_qc_rows(
    *,
    contig_to_bin: dict[str, str],
    contig_len: dict[str, int],
    intra_support: dict[str, float],
    other_support: dict[str, float],
    bin_cov_stats: Optional[dict[str, dict[str, float]]],
    contact_component_qc: dict[str, tuple[int, float]],
    bin_scg_qc: dict[str, BinScgQc],
    scg_status: str,
    scg_expected_markers_count: int,
) -> list[BinQcRow]:
    bp = Counter()
    n = Counter()
    support_by_bin: dict[str, list[float]] = defaultdict(list)
    low_support_by_bin: Counter[str] = Counter()

    for contig, bin_id in contig_to_bin.items():
        if not bin_id or str(bin_id).strip() == "-1":
            continue
        bb = str(bin_id)
        L = int(contig_len.get(contig, 0))
        bp[bb] += L
        n[bb] += 1
        s_intra = float(intra_support.get(contig, 0.0))
        s_other = float(other_support.get(contig, 0.0))
        support_by_bin[bb].append(s_intra)
        if s_intra <= s_other:
            low_support_by_bin[bb] += 1

    rows: list[BinQcRow] = []
    for bin_id in sorted(bp.keys(), key=_natural_bin_sort_key):
        vals = sorted(support_by_bin.get(bin_id, []))
        median = 0.0
        if vals:
            mid = len(vals) // 2
            if len(vals) % 2 == 1:
                median = float(vals[mid])
            else:
                median = float(vals[mid - 1] + vals[mid]) / 2.0
        low_support = int(low_support_by_bin.get(bin_id, 0))
        n_contigs = int(n.get(bin_id, 0))
        comp_count, largest_share = contact_component_qc.get(bin_id, (0, 0.0))
        scg = bin_scg_qc.get(
            bin_id,
            BinScgQc(
                bin_id=bin_id,
                expected_markers=scg_expected_markers_count,
                unique_markers=0,
                duplicated_markers=0,
                total_marker_hits=0,
                completeness_like=(0.0 if scg_expected_markers_count > 0 and scg_status != "disabled" else None),
                contamination_like=(0.0 if scg_expected_markers_count > 0 and scg_status != "disabled" else None),
                implicated_contigs=(),
            ),
        )
        cov = (bin_cov_stats or {}).get(bin_id, {})
        unique_scg: Optional[int] = None
        duplicated_scg: Optional[int] = None
        completeness_like: Optional[float] = None
        contamination_like: Optional[float] = None
        if scg_status != "disabled":
            unique_scg = int(scg.unique_markers)
            duplicated_scg = int(scg.duplicated_markers)
            completeness_like = scg.completeness_like
            contamination_like = scg.contamination_like
        rows.append(
            BinQcRow(
                bin_id=bin_id,
                total_bp=int(bp[bin_id]),
                n_contigs=n_contigs,
                contact_consistency=(float(largest_share) if comp_count > 0 else None),
                coverage_dispersion=(float(cov["mad"]) if "mad" in cov else None),
                tnf_dispersion=None,
                scg_status=scg_status,
                unique_scg=unique_scg,
                duplicated_scg=duplicated_scg,
                completeness_like=completeness_like,
                contamination_like=contamination_like,
                suspect_flag=False,
                suspect_reasons=(),
                split_check_flag=False,
                split_check_reasons=(),
                split_priority_source=None,
                contact_components=int(comp_count),
                largest_component_share=float(largest_share),
                low_support_contigs=low_support,
                low_support_ratio=(low_support / float(n_contigs)) if n_contigs > 0 else 0.0,
                median_intra_support=float(median),
                coverage_median=(float(cov["median"]) if "median" in cov else None),
                coverage_mad=(float(cov["mad"]) if "mad" in cov else None),
                scg_implicated_contigs=len(scg.implicated_contigs),
            )
        )
    return rows


def annotate_suspect_flags(rows: list[BinQcRow], suspects: list[SuspectBinRecord]) -> list[BinQcRow]:
    reasons_by_bin = {rec.bin_id: rec.reasons for rec in suspects}
    out: list[BinQcRow] = []
    for row in rows:
        reasons = reasons_by_bin.get(row.bin_id, ())
        out.append(replace(row, suspect_flag=bool(reasons), suspect_reasons=tuple(reasons)))
    return out


def annotate_split_check_flags(rows: list[BinQcRow], split_checks: list[SplitCheckRecord]) -> list[BinQcRow]:
    checks_by_bin = {rec.bin_id: rec for rec in split_checks}
    out: list[BinQcRow] = []
    for row in rows:
        rec = checks_by_bin.get(row.bin_id)
        if rec is None:
            out.append(replace(row, split_check_flag=False, split_check_reasons=(), split_priority_source=None))
            continue
        reasons = tuple(rec.trigger_reasons + rec.auxiliary_reasons)
        out.append(
            replace(
                row,
                split_check_flag=True,
                split_check_reasons=reasons,
                split_priority_source=str(rec.priority_source),
            )
        )
    return out


def select_suspect_bins(rows: list[BinQcRow]) -> list[SuspectBinRecord]:
    suspects: list[SuspectBinRecord] = []
    for row in rows:
        reasons_struct: list[str] = []
        reasons_scg: list[str] = []
        if row.duplicated_scg is not None and row.duplicated_scg > 0:
            reasons_scg.append("duplicated_scg")
        if row.contamination_like is not None and row.contamination_like > 0.1:
            reasons_scg.append("high_contamination_like")
        if row.completeness_like is not None and row.completeness_like < 0.5:
            reasons_scg.append("low_completeness_like")
        if row.contact_components > 1:
            reasons_struct.append("multi_contact_component")
        if row.largest_component_share > 0.0 and row.largest_component_share < 0.9:
            reasons_struct.append("fragmented_contact_component")
        if row.low_support_ratio >= 0.25:
            reasons_struct.append("high_low_support_ratio")
        if row.coverage_dispersion is not None and row.coverage_dispersion > 0.0:
            reasons_struct.append("coverage_dispersion_present")

        use_scg = row.scg_status != "disabled"
        suspect = False
        if use_scg:
            suspect = bool((reasons_struct and reasons_scg) or len(reasons_struct) >= 2 or len(reasons_scg) >= 2)
        else:
            suspect = bool(len(reasons_struct) >= 2)
        if suspect:
            suspects.append(SuspectBinRecord(bin_id=row.bin_id, reasons=tuple(reasons_scg + reasons_struct)))
    return suspects


def select_split_check_bins(rows: list[BinQcRow]) -> list[SplitCheckRecord]:
    records: list[SplitCheckRecord] = []
    for row in rows:
        if int(row.n_contigs) < 4:
            continue
        trigger_reasons: list[str] = []
        auxiliary_reasons: list[str] = []
        if row.duplicated_scg is not None and row.duplicated_scg > 0:
            trigger_reasons.append("duplicated_scg")
        if row.contamination_like is not None and row.contamination_like > 0.1:
            trigger_reasons.append("high_contamination_like")

        if row.contact_components > 1:
            auxiliary_reasons.append("multi_contact_component")
        if row.largest_component_share > 0.0 and row.largest_component_share < 0.9:
            auxiliary_reasons.append("fragmented_contact_component")
        if row.low_support_ratio >= 0.25:
            auxiliary_reasons.append("high_low_support_ratio")
        if row.coverage_dispersion is not None and row.coverage_dispersion > 0.0:
            auxiliary_reasons.append("coverage_dispersion_present")
        if row.tnf_dispersion is not None and row.tnf_dispersion > 0.0:
            auxiliary_reasons.append("tnf_dispersion_present")

        if trigger_reasons and auxiliary_reasons:
            records.append(
                SplitCheckRecord(
                    bin_id=row.bin_id,
                    trigger_reasons=tuple(trigger_reasons),
                    auxiliary_reasons=tuple(auxiliary_reasons),
                    priority_source="scg_priority",
                )
            )
    return records


def write_bin_qc_tsv(path: Path, rows: list[BinQcRow]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(
            [
                "bin_id",
                "total_bp",
                "n_contigs",
                "contact_consistency",
                "coverage_dispersion",
                "tnf_dispersion",
                "scg_status",
                "unique_scg",
                "duplicated_scg",
                "completeness_like",
                "contamination_like",
                "suspect_flag",
                "suspect_reasons",
                "split_check_flag",
                "split_check_reasons",
                "split_priority_source",
                "contact_components",
                "largest_component_share",
                "low_support_contigs",
                "low_support_ratio",
                "median_intra_support",
                "coverage_median",
                "coverage_mad",
                "scg_implicated_contigs",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.bin_id,
                    row.total_bp,
                    row.n_contigs,
                    _fmt_float(row.contact_consistency),
                    _fmt_float(row.coverage_dispersion),
                    _fmt_float(row.tnf_dispersion),
                    row.scg_status,
                    _fmt_int(row.unique_scg),
                    _fmt_int(row.duplicated_scg),
                    _fmt_float(row.completeness_like),
                    _fmt_float(row.contamination_like),
                    1 if row.suspect_flag else 0,
                    ",".join(row.suspect_reasons),
                    1 if row.split_check_flag else 0,
                    ",".join(row.split_check_reasons),
                    (row.split_priority_source or ""),
                    row.contact_components,
                    f"{row.largest_component_share:.6g}",
                    row.low_support_contigs,
                    f"{row.low_support_ratio:.6g}",
                    f"{row.median_intra_support:.6g}",
                    _fmt_float(row.coverage_median),
                    _fmt_float(row.coverage_mad),
                    row.scg_implicated_contigs,
                ]
            )


def write_suspect_bins_tsv(path: Path, suspects: list[SuspectBinRecord]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["bin_id", "reasons"])
        for rec in suspects:
            writer.writerow([rec.bin_id, ",".join(rec.reasons)])


def _fmt_float(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.6g}"


def _fmt_int(value: Optional[int]) -> str:
    if value is None:
        return "NA"
    return str(int(value))


def _natural_bin_sort_key(bin_id: str) -> tuple[int, str]:
    try:
        return (0, f"{int(bin_id):012d}")
    except Exception:
        return (1, str(bin_id))
