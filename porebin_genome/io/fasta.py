"""FASTA readers for contig-centric workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional


def iter_fasta_records(path: Path) -> Iterator[tuple[str, str, str]]:
    """Yield `(name, header, sequence)` records from a FASTA file."""
    header: Optional[str] = None
    name: Optional[str] = None
    seq_parts: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip().lstrip("\ufeff")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None and name is not None:
                    yield name, header, "".join(seq_parts)
                header = line[1:].strip()
                name = header.split()[0] if header else ""
                seq_parts = []
            else:
                if header is None:
                    raise ValueError(f"Invalid FASTA (sequence before header) at {path}:{line_no}")
                seq_parts.append(line)
        if header is not None and name is not None:
            yield name, header, "".join(seq_parts)


def iter_fasta_names(path: Path) -> Iterator[str]:
    """Yield contig names from a FASTA file."""
    for name, _header, _seq in iter_fasta_records(path):
        yield name


def read_contig_lengths(path: Path) -> dict[str, int]:
    """Read contig lengths keyed by contig name."""
    out: dict[str, int] = {}
    for name, _header, seq in iter_fasta_records(path):
        out[name] = len(seq)
    return out
