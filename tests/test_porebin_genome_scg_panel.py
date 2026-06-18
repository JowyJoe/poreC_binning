from __future__ import annotations

import inspect
from pathlib import Path

from porebin_genome.evidence.scg.panel import (
    DEFAULT_SCG_HMM,
    DEFAULT_SCG_MARKERS,
    load_hmm_profile_ids,
    load_scg_panel,
)


def test_bundled_scg_panel_matches_fixed_107_marker_definition() -> None:
    panel = load_scg_panel()

    assert panel.hmm_path == DEFAULT_SCG_HMM.resolve()
    assert len(panel.raw_profile_ids) == 111
    assert panel.profiles_with_tc == 111
    assert panel.marker_count == 107
    assert len(set(panel.canonical_marker_ids)) == 107
    assert panel.canonical_marker_ids == DEFAULT_SCG_MARKERS


def test_bundled_scg_panel_is_process_cached_and_has_no_runtime_inputs() -> None:
    first = load_scg_panel()
    second = load_scg_panel()

    assert first is second
    assert not inspect.signature(load_scg_panel).parameters


def test_bundled_scg_alias_pairs_collapse_to_one_canonical_marker() -> None:
    panel = load_scg_panel()

    assert panel.canonicalize("TIGR00388") == "TIGR00389"
    assert panel.canonicalize("TIGR00471") == "TIGR00472"
    assert panel.canonicalize("TIGR00408") == "TIGR00409"
    assert panel.canonicalize("TIGR02386") == "TIGR02387"


def test_hmm_reader_reports_profile_without_trusted_cutoff(tmp_path: Path) -> None:
    hmm = tmp_path / "bad.hmm"
    hmm.write_text(
        "HMMER3/f\nNAME  marker_a\nACC   marker_a\n//\n",
        encoding="utf-8",
    )

    profile_ids, profiles_with_tc = load_hmm_profile_ids(hmm)

    assert profile_ids == ("marker_a",)
    assert profiles_with_tc == 0
