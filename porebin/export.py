from __future__ import annotations

import csv
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from porebin import __version__
from porebin.utils import ensure_dir, iter_fasta_records, write_json

MIN_BIN_BP = 200_000  # 200kb (MetaBAT2/VAMB/SemiBin common default)


class ExportError(RuntimeError):
    pass


@dataclass
class ExportStats:
    contigs_total: int = 0
    bp_total: int = 0
    bins_total: int = 0
    bins_kept: int = 0
    contigs_in_bins: int = 0
    contigs_exported: int = 0
    contigs_unbinned: int = 0
    min_contig_len: int = 0
    min_contig_len_ratio_1000_2500_bp: float = 0.0


def export_bins(
    *,
    contigs_fasta: Path,
    bins_tsv: Path,
    out_dir: Path,
    threads: int = 1,
    logger: Optional[logging.Logger] = None,
) -> ExportStats:
    logger = logger or logging.getLogger("porebin")
    out_dir = out_dir.resolve()
    ensure_dir(out_dir)

    if not contigs_fasta.exists():
        raise FileNotFoundError(f"Contigs FASTA not found: {contigs_fasta}")
    if not bins_tsv.exists():
        raise FileNotFoundError(f"Bins TSV not found: {bins_tsv}")

    contig_to_bin = _read_bins_tsv(bins_tsv)
    if not contig_to_bin:
        raise ExportError(f"No bins found in {bins_tsv}")

    removed_by_refine = _load_removed_by_refine(bins_tsv.parent)

    contig_len: dict[str, int] = {}
    total_bp = 0
    bp_1000_2500 = 0
    try:
        for name, _header, seq in iter_fasta_records(contigs_fasta):
            L = len(seq)
            contig_len[name] = L
            total_bp += L
            if 1000 <= L <= 2500:
                bp_1000_2500 += L
    except ValueError as exc:
        raise ExportError(str(exc)) from exc

    if not contig_len:
        raise ExportError(f"No contigs found in FASTA: {contigs_fasta}")

    stats = ExportStats()
    stats.contigs_total = len(contig_len)
    stats.bp_total = total_bp
    stats.contigs_in_bins = len(contig_to_bin)

    ratio = (bp_1000_2500 / total_bp) if total_bp else 0.0
    min_contig_len = 1000 if ratio >= 0.05 else 2500
    stats.min_contig_len = min_contig_len
    stats.min_contig_len_ratio_1000_2500_bp = ratio

    bin_bp: Counter[str] = Counter()
    bin_n: Counter[str] = Counter()
    bin_ids = sorted(set(contig_to_bin.values()), key=_natural_bin_sort_key)
    for contig, bin_id in contig_to_bin.items():
        L = contig_len.get(contig)
        if L is None:
            raise ExportError(
                f"Contig '{contig}' in {bins_tsv} not found in FASTA '{contigs_fasta}'."
            )
        if L < min_contig_len:
            continue
        bin_bp[bin_id] += L
        bin_n[bin_id] += 1

    keep_bins = {b for b in bin_ids if bin_bp[b] >= MIN_BIN_BP}

    bins_summary_path = out_dir / "bins.summary.tsv"
    with bins_summary_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("bin_id\ttotal_bp\tn_contigs\tkeep\n")
        for bin_id in bin_ids:
            fh.write(
                f"{bin_id}\t{bin_bp[bin_id]}\t{bin_n[bin_id]}\t{1 if bin_id in keep_bins else 0}\n"
            )

    out_bins_dir = out_dir / "bins_fasta"
    ensure_dir(out_bins_dir)
    for existing in out_bins_dir.glob("bin_*.fasta"):
        existing.unlink()

    unbinned_fasta = out_dir / "unbinned.fasta"
    unbinned_tsv = out_dir / "unbinned.tsv"
    if unbinned_fasta.exists():
        unbinned_fasta.unlink()
    if unbinned_tsv.exists():
        unbinned_tsv.unlink()

    stats.bins_total = len(bin_ids)
    stats.bins_kept = len(keep_bins)

    with unbinned_fasta.open("w", encoding="utf-8", newline="") as unb_fa, unbinned_tsv.open(
        "w", encoding="utf-8", newline=""
    ) as unb_tsv:
        unb_tsv.write("contig_name\treason\tbin_id\tlength\n")

        try:
            for name, header, seq in iter_fasta_records(contigs_fasta):
                bin_id = contig_to_bin.get(name)
                if name in removed_by_refine:
                    _write_fasta_record(unb_fa, header, seq)
                    unb_tsv.write(f"{name}\tremoved_by_refine\t{bin_id or ''}\t{len(seq)}\n")
                    stats.contigs_unbinned += 1
                    continue
                if len(seq) < min_contig_len:
                    _write_fasta_record(unb_fa, header, seq)
                    unb_tsv.write(f"{name}\tshort_contig\t{bin_id or ''}\t{len(seq)}\n")
                    stats.contigs_unbinned += 1
                    continue
                if bin_id is not None and bin_id in keep_bins:
                    out_path = out_bins_dir / f"bin_{bin_id}.fasta"
                    with out_path.open("a", encoding="utf-8", newline="") as fh:
                        _write_fasta_record(fh, header, seq)
                    stats.contigs_exported += 1
                    continue

                reason = _unbinned_reason(
                    name,
                    length=len(seq),
                    bin_id=bin_id,
                    keep_bins=keep_bins,
                    removed_by_refine=removed_by_refine,
                    min_contig_len=min_contig_len,
                )
                _write_fasta_record(unb_fa, header, seq)
                unb_tsv.write(f"{name}\t{reason}\t{bin_id or ''}\t{len(seq)}\n")
                stats.contigs_unbinned += 1
        except ValueError as exc:
            raise ExportError(str(exc)) from exc

    meta = {
        "porebin_version": __version__,
        "min_bin_bp": MIN_BIN_BP,
        "min_contig_len": min_contig_len,
        "min_contig_len_method": {
            "type": "semiBin_ratio_1000_2500",
            "ratio_1000_2500_bp": ratio,
            "ratio_threshold": 0.05,
            "rule": "min_contig_len=1000 if ratio>=0.05 else 2500",
        },
        "counts": {
            "contigs_total": stats.contigs_total,
            "bins_total": stats.bins_total,
            "bins_kept": stats.bins_kept,
            "contigs_exported": stats.contigs_exported,
            "contigs_unbinned": stats.contigs_unbinned,
        },
        "inputs": {
            "contigs_fasta": str(contigs_fasta),
            "bins_tsv": str(bins_tsv),
        },
        "threads": threads,
    }
    write_json(out_dir / "export_meta.json", meta)

    logger.info(
        "Exported bins: kept=%s/%s (min_bin_bp=%s), exported_contigs=%s, unbinned_contigs=%s",
        stats.bins_kept,
        stats.bins_total,
        MIN_BIN_BP,
        f"{stats.contigs_exported:,}",
        f"{stats.contigs_unbinned:,}",
    )
    logger.info(f"Wrote: {bins_summary_path}")
    logger.info(f"Wrote: {unbinned_fasta}")
    logger.info(f"Wrote: {unbinned_tsv}")
    return stats


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
                raise ExportError(f"Invalid bins row in {path}: {row}")
            mapping[row[0]] = row[1]
    return mapping


def _load_removed_by_refine(refine_dir: Path) -> set[str]:
    removed: set[str] = set()
    decontam = refine_dir / "decontam_removed.tsv"
    if decontam.exists():
        with decontam.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh, delimiter="\t")
            for row in reader:
                if not row:
                    continue
                if row[0] in {"contig_name", "contig"}:
                    continue
                removed.add(row[0])
    split_map = refine_dir / "split_map.tsv"
    if split_map.exists():
        with split_map.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh, delimiter="\t")
            for row in reader:
                if not row:
                    continue
                if row[0] in {"contig_name", "contig"}:
                    continue
                # contig_name, old_bin, new_bin, action, reason, ...
                if len(row) >= 4 and row[3].strip().lower() == "unbinned":
                    removed.add(row[0])
    return removed


def _unbinned_reason(
    contig: str,
    *,
    length: int,
    bin_id: Optional[str],
    keep_bins: set[str],
    removed_by_refine: set[str],
    min_contig_len: int,
) -> str:
    if contig in removed_by_refine:
        return "removed_by_refine"
    if length < min_contig_len:
        return "short_contig"
    if bin_id is not None and bin_id not in keep_bins:
        return "tiny_bin"
    return "unassigned"


def _write_fasta_record(fh, header: str, seq: str, *, wrap: int = 80) -> None:
    fh.write(f">{header}\n")
    for i in range(0, len(seq), wrap):
        fh.write(seq[i : i + wrap] + "\n")


def _natural_bin_sort_key(bin_id: str) -> tuple[int, str]:
    # Prefer numeric ordering for common integer bin ids.
    try:
        return (0, f"{int(bin_id):012d}")
    except Exception:
        return (1, bin_id)
