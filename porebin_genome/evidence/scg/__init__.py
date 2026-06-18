"""Fixed-panel single-copy-gene evidence generation."""

from __future__ import annotations

from porebin_genome.evidence.scg.discovery import (
    ContigScgProfile,
    ScgHit,
    ScgRunResult,
    collapse_scg_hits_to_contigs,
    ensure_scg_toolchain_available,
    parse_scg_hits,
    prepare_scg_profiles,
)
from porebin_genome.evidence.scg.panel import (
    DEFAULT_SCG_HMM,
    DEFAULT_SCG_MARKERS,
    ScgPanel,
    canonicalize_marker_id,
    load_scg_panel,
)

__all__ = [
    "ContigScgProfile",
    "DEFAULT_SCG_HMM",
    "DEFAULT_SCG_MARKERS",
    "ScgHit",
    "ScgPanel",
    "ScgRunResult",
    "canonicalize_marker_id",
    "collapse_scg_hits_to_contigs",
    "ensure_scg_toolchain_available",
    "load_scg_panel",
    "parse_scg_hits",
    "prepare_scg_profiles",
]
