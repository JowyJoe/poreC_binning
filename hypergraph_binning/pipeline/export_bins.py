from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def _sanitize_bin_name(name: str) -> str:
    # Keep alnum, dash, underscore; replace others with underscore
    import re
    s = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    if not s:
        s = "bin"
    return s


def export_bins_fasta(
    contigs_fasta: Path,
    bins_tsv: Path,
    out_dir: Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read bin assignments
    df = pd.read_csv(bins_tsv, sep="\t")
    if "contig" not in df.columns or "bin" not in df.columns:
        raise ValueError("bins.tsv must have columns: contig, bin")

    # Map contig -> bin
    c2b: Dict[str, str] = dict(zip(df["contig"].astype(str), df["bin"].astype(str)))

    # Prepare writers lazily per-bin
    handles: Dict[str, any] = {}
    bin_counts: Dict[str, int] = {}
    bin_bases: Dict[str, int] = {}

    try:
        for rec in SeqIO.parse(str(contigs_fasta), "fasta"):
            contig = rec.id
            b = c2b.get(contig)
            if b is None:
                continue  # contig not assigned, skip
            b_san = _sanitize_bin_name(b)
            if b_san not in handles:
                fpath = out_dir / f"{b_san}.fasta"
                handles[b_san] = open(fpath, "w", encoding="utf-8")
                bin_counts[b_san] = 0
                bin_bases[b_san] = 0
            SeqIO.write(rec, handles[b_san], "fasta")
            bin_counts[b_san] += 1
            bin_bases[b_san] += len(rec.seq)
    finally:
        for h in handles.values():
            try:
                h.close()
            except Exception:
                pass

    # Write simple stats
    stats = (
        pd.DataFrame({
            "bin": list(bin_counts.keys()),
            "num_contigs": list(bin_counts.values()),
            "total_bases": [bin_bases[k] for k in bin_counts.keys()],
        })
        .sort_values(["num_contigs", "total_bases"], ascending=[False, False])
    )
    stats.to_csv(out_dir / "bin_fasta_stats.tsv", sep="\t", index=False)
