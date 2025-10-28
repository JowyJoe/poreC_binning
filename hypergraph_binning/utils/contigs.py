from __future__ import annotations
from typing import Dict, List, Tuple
from pathlib import Path

from Bio import SeqIO


def read_contigs_fasta(fasta_path: str) -> Tuple[List[str], Dict[str, int]]:
    """Return (ordered_names, length_map)."""
    names: List[str] = []
    lengths: Dict[str, int] = {}
    for rec in SeqIO.parse(str(fasta_path), "fasta"):
        name = rec.id
        names.append(name)
        lengths[name] = int(len(rec.seq))
    return names, lengths
