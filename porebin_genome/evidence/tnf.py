"""TNF136 constants and sequence-composition helpers for contig features."""

from __future__ import annotations

from typing import Final, Optional

from porebin_genome.io.fasta import iter_fasta_records


TNF136_LIST: Final[list[str]] = [
    "AAAA",
    "AAAC",
    "AAAG",
    "AAAT",
    "AACA",
    "AACC",
    "AACG",
    "AACT",
    "AAGA",
    "AAGC",
    "AAGG",
    "AAGT",
    "AATA",
    "AATC",
    "AATG",
    "AATT",
    "ACAA",
    "ACAC",
    "ACAG",
    "ACAT",
    "ACCA",
    "ACCC",
    "ACCG",
    "ACCT",
    "ACGA",
    "ACGC",
    "ACGG",
    "ACGT",
    "ACTA",
    "ACTC",
    "ACTG",
    "AGAA",
    "AGAC",
    "AGAG",
    "AGAT",
    "AGCA",
    "AGCC",
    "AGCG",
    "AGCT",
    "AGGA",
    "AGGC",
    "AGGG",
    "AGTA",
    "AGTC",
    "AGTG",
    "ATAA",
    "ATAC",
    "ATAG",
    "ATAT",
    "ATCA",
    "ATCC",
    "ATCG",
    "ATGA",
    "ATGC",
    "ATGG",
    "ATTA",
    "ATTC",
    "ATTG",
    "CAAA",
    "CAAC",
    "CAAG",
    "CACA",
    "CACC",
    "CACG",
    "CAGA",
    "CAGC",
    "CAGG",
    "CATA",
    "CATC",
    "CATG",
    "CCAA",
    "CCAC",
    "CCAG",
    "CCCA",
    "CCCC",
    "CCCG",
    "CCGA",
    "CCGC",
    "CCGG",
    "CCTA",
    "CCTC",
    "CGAA",
    "CGAC",
    "CGAG",
    "CGCA",
    "CGCC",
    "CGCG",
    "CGGA",
    "CGGC",
    "CGTA",
    "CGTC",
    "CTAA",
    "CTAC",
    "CTAG",
    "CTCA",
    "CTCC",
    "CTGA",
    "CTGC",
    "CTTA",
    "CTTC",
    "GAAA",
    "GAAC",
    "GACA",
    "GACC",
    "GAGA",
    "GAGC",
    "GATA",
    "GATC",
    "GCAA",
    "GCAC",
    "GCCA",
    "GCCC",
    "GCGA",
    "GCGC",
    "GCTA",
    "GGAA",
    "GGAC",
    "GGCA",
    "GGCC",
    "GGGA",
    "GGTA",
    "GTAA",
    "GTAC",
    "GTCA",
    "GTGA",
    "GTTA",
    "TAAA",
    "TACA",
    "TAGA",
    "TATA",
    "TCAA",
    "TCCA",
    "TCGA",
    "TGAA",
    "TGCA",
    "TTAA",
]

CANON_TO_IDX: Final[dict[str, int]] = {kmer: idx for idx, kmer in enumerate(TNF136_LIST)}

_COMP: Final[dict[str, str]] = {"A": "T", "C": "G", "G": "C", "T": "A"}


def revcomp_4mer(sequence: str) -> str:
    """Reverse-complement a 4-mer."""
    if len(sequence) != 4:
        raise ValueError(f"Expected 4-mer, got {sequence!r}")
    try:
        return "".join(_COMP[base] for base in reversed(sequence))
    except KeyError as exc:
        raise ValueError(f"Invalid base in 4-mer {sequence!r} (expected only A/C/G/T).") from exc


def canonical_4mer(sequence: str) -> str:
    """Collapse a 4-mer against its reverse complement."""
    revcomp = revcomp_4mer(sequence)
    return sequence if sequence <= revcomp else revcomp


def enumerate_directed_4mers_lex() -> list[str]:
    """Enumerate the 256 directed 4-mers in lexicographic order."""
    alphabet = "ACGT"
    out: list[str] = []
    for a in alphabet:
        for b in alphabet:
            for c in alphabet:
                for d in alphabet:
                    out.append(a + b + c + d)
    return out


def build_idx256_to_idx136() -> list[int]:
    """Build the 256 -> 136 canonical 4-mer index map."""
    all_256 = enumerate_directed_4mers_lex()
    idx256_to_idx136 = [0] * 256
    for idx, sequence in enumerate(all_256):
        idx256_to_idx136[idx] = CANON_TO_IDX[canonical_4mer(sequence)]
    return idx256_to_idx136


def compute_tnf136_features(
    contigs_fasta,
    contig_name_to_idx: dict[str, int],
    *,
    logger: Optional[object] = None,
) -> "object":
    """Compute canonical TNF136 frequencies keyed by contig order."""
    _ = logger
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Computing TNF136 features requires numpy.") from exc

    X = np.zeros((len(contig_name_to_idx), 136), dtype=np.float32)
    seen = 0
    for name, _header, seq in iter_fasta_records(contigs_fasta):
        idx = contig_name_to_idx.get(name)
        if idx is None:
            continue
        seen += 1
        s = (seq or "").upper()
        counts = np.zeros((136,), dtype=np.float32)
        total = 0
        if len(s) >= 4:
            for start in range(len(s) - 3):
                kmer = s[start : start + 4]
                if any(base not in "ACGT" for base in kmer):
                    continue
                counts[int(CANON_TO_IDX[canonical_4mer(kmer)])] += 1.0
                total += 1
        denom = float(total) if total > 0 else 1.0
        X[idx, :] = counts / denom
    if seen == 0:
        raise RuntimeError(f"No contigs from contig index were found in FASTA: {contigs_fasta}")
    return X


assert len(TNF136_LIST) == 136
assert len(set(TNF136_LIST)) == 136
