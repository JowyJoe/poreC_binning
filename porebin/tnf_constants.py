from __future__ import annotations

from typing import Final


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

CANON_TO_IDX: Final[dict[str, int]] = {kmer: i for i, kmer in enumerate(TNF136_LIST)}

_COMP: Final[dict[str, str]] = {"A": "T", "C": "G", "G": "C", "T": "A"}


def revcomp_4mer(s: str) -> str:
    if len(s) != 4:
        raise ValueError(f"Expected 4-mer, got {s!r}")
    try:
        return "".join(_COMP[c] for c in reversed(s))
    except KeyError as exc:
        raise ValueError(f"Invalid base in 4-mer {s!r} (expected only A/C/G/T).") from exc


def canonical_4mer(s: str) -> str:
    rc = revcomp_4mer(s)
    return s if s <= rc else rc


def enumerate_directed_4mers_lex() -> list[str]:
    """
    Directed 256 4-mers in lexicographic order with alphabet A<C<G<T:
      AAAA, AAAC, ..., TTTT.
    """
    alphabet = "ACGT"
    out: list[str] = []
    for a in alphabet:
        for b in alphabet:
            for c in alphabet:
                for d in alphabet:
                    out.append(a + b + c + d)
    return out


def build_idx256_to_idx136() -> list[int]:
    """
    Build 256->136 index mapping with TNF136_LIST as the single source of truth.
    """
    all_256 = enumerate_directed_4mers_lex()
    idx256_to_idx136 = [0] * 256
    for i, s in enumerate(all_256):
        canon = canonical_4mer(s)
        idx136 = CANON_TO_IDX[canon]
        idx256_to_idx136[i] = idx136

    canon_set = {canonical_4mer(s) for s in all_256}
    assert canon_set == set(TNF136_LIST)
    assert len(canon_set) == 136
    return idx256_to_idx136


assert len(TNF136_LIST) == 136
assert len(set(TNF136_LIST)) == 136
assert TNF136_LIST == sorted(TNF136_LIST)
assert all(len(x) == 4 and all(c in "ACGT" for c in x) for x in TNF136_LIST)

