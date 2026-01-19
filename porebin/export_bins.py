from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

from porebin.utils import iter_fasta_records


class ExportBinsError(RuntimeError):
    pass


def export_bins_fasta(
    *,
    contigs_fasta: Path,
    bins_tsv: Path,
    out_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> None:
    logger = logger or logging.getLogger("porebin")

    if not contigs_fasta.exists():
        raise FileNotFoundError(f"Contigs FASTA not found: {contigs_fasta}")
    if not bins_tsv.exists():
        raise FileNotFoundError(f"Bins TSV not found: {bins_tsv}")

    contig_to_bin = _read_bins_tsv(bins_tsv)
    if not contig_to_bin:
        raise ExportBinsError(f"No bins found in {bins_tsv}")

    out_bins_dir = out_dir / "bins_fasta"
    out_bins_dir.mkdir(parents=True, exist_ok=True)
    for existing in out_bins_dir.glob("bin_*.fasta"):
        existing.unlink()

    written = 0
    seen = 0
    try:
        for name, header, seq in iter_fasta_records(contigs_fasta):
            seen += 1
            bin_id = contig_to_bin.get(name)
            if bin_id is None:
                continue
            out_path = out_bins_dir / f"bin_{bin_id}.fasta"
            with out_path.open("a", encoding="utf-8", newline="") as fh:
                _write_fasta_record(fh, header, seq)
            written += 1
    except ValueError as exc:
        raise ExportBinsError(str(exc)) from exc

    logger.info(f"Wrote {written} contigs into {out_bins_dir} (from {seen} FASTA records)")


def _read_bins_tsv(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if row[0] == "contig_name":
                continue
            if len(row) < 2:
                raise ExportBinsError(f"Invalid bins row in {path}: {row}")
            mapping[row[0]] = row[1]
    return mapping


def _write_fasta_record(fh, header: str, seq: str, *, wrap: int = 80) -> None:
    fh.write(f">{header}\n")
    for i in range(0, len(seq), wrap):
        fh.write(seq[i : i + wrap] + "\n")
