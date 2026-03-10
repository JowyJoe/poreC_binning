from __future__ import annotations

import pytest


def test_tnf136_list_matches_canon() -> None:
    from porebin.tnf_constants import TNF136_LIST, canonical_4mer, enumerate_directed_4mers_lex

    all_256 = enumerate_directed_4mers_lex()
    canon_set = {canonical_4mer(s) for s in all_256}
    assert canon_set == set(TNF136_LIST)
    assert len(canon_set) == 136


def test_256_to_136_merge() -> None:
    from porebin.tnf_constants import (
        CANON_TO_IDX,
        build_idx256_to_idx136,
        enumerate_directed_4mers_lex,
        revcomp_4mer,
    )

    all_256 = enumerate_directed_4mers_lex()
    idx_map = build_idx256_to_idx136()

    aaac = "AAAC"
    aaac_rc = revcomp_4mer(aaac)
    assert aaac_rc == "GTTT"

    i_aaac = all_256.index(aaac)
    i_rc = all_256.index(aaac_rc)
    assert idx_map[i_aaac] == idx_map[i_rc]

    # Palindrome should not "double": only one directed 4-mer exists for it.
    aatt = "AATT"
    assert revcomp_4mer(aatt) == aatt
    i_aatt = all_256.index(aatt)

    counts256 = [0.0] * 256
    counts256[i_aaac] = 2.0
    counts256[i_rc] = 3.0
    counts256[i_aatt] = 7.0

    counts136 = [0.0] * 136
    for i, v in enumerate(counts256):
        counts136[idx_map[i]] += float(v)

    assert counts136[CANON_TO_IDX["AAAC"]] == pytest.approx(5.0)
    assert counts136[CANON_TO_IDX["AATT"]] == pytest.approx(7.0)

