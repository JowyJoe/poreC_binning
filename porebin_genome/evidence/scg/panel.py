"""Fixed SCG panel loading and canonical marker normalization."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


SCG_ALIAS_SCHEMA_VERSION = 1
SCG_ALIASES = {
    "TIGR00388": "TIGR00389",
    "TIGR00471": "TIGR00472",
    "TIGR00408": "TIGR00409",
    "TIGR02386": "TIGR02387",
}

DEFAULT_SCG_HMM = Path(__file__).resolve().parent / "marker.hmm"
# Stable 107-marker order corresponding to LorBin's single-copy-gene table.
# Keep this tuple explicit: its order is part of the SCG feature contract.
DEFAULT_SCG_MARKERS = (
    "GrpE",
    "Methyltransf_5",
    "PGK",
    "Ribosomal_L10",
    "Ribosomal_L23",
    "Ribosomal_L3",
    "Ribosomal_L4",
    "Ribosomal_L5",
    "Ribosomal_L6",
    "Ribosomal_S11",
    "Ribosomal_S13",
    "Ribosomal_S17",
    "Ribosomal_S8",
    "Ribosomal_S9",
    "TIGR00001",
    "TIGR00002",
    "TIGR00009",
    "TIGR00012",
    "TIGR00019",
    "TIGR00029",
    "TIGR00043",
    "TIGR00059",
    "TIGR00060",
    "TIGR00061",
    "TIGR00062",
    "TIGR00064",
    "TIGR00082",
    "TIGR00086",
    "TIGR00092",
    "TIGR00115",
    "TIGR00116",
    "TIGR00152",
    "TIGR00158",
    "TIGR00165",
    "TIGR00166",
    "TIGR00168",
    "TIGR00234",
    "TIGR00337",
    "TIGR00344",
    "TIGR00362",
    "TIGR00389",
    "TIGR00392",
    "TIGR00396",
    "TIGR00409",
    "TIGR00414",
    "TIGR00418",
    "TIGR00420",
    "TIGR00422",
    "TIGR00435",
    "TIGR00436",
    "TIGR00442",
    "TIGR00459",
    "TIGR00460",
    "TIGR00468",
    "TIGR00472",
    "TIGR00487",
    "TIGR00496",
    "TIGR00575",
    "TIGR00631",
    "TIGR00663",
    "TIGR00755",
    "TIGR00810",
    "TIGR00855",
    "TIGR00922",
    "TIGR00952",
    "TIGR00959",
    "TIGR00963",
    "TIGR00964",
    "TIGR00967",
    "TIGR00981",
    "TIGR01009",
    "TIGR01011",
    "TIGR01017",
    "TIGR01021",
    "TIGR01024",
    "TIGR01029",
    "TIGR01030",
    "TIGR01031",
    "TIGR01032",
    "TIGR01044",
    "TIGR01049",
    "TIGR01050",
    "TIGR01059",
    "TIGR01063",
    "TIGR01066",
    "TIGR01067",
    "TIGR01071",
    "TIGR01079",
    "TIGR01164",
    "TIGR01169",
    "TIGR01171",
    "TIGR01391",
    "TIGR01393",
    "TIGR01632",
    "TIGR01953",
    "TIGR02012",
    "TIGR02013",
    "TIGR02027",
    "TIGR02191",
    "TIGR02350",
    "TIGR02387",
    "TIGR02397",
    "TIGR02432",
    "TIGR02729",
    "TIGR03263",
    "TIGR03594",
    "tRNA-synt_1d",
)


class ScgPanelError(RuntimeError):
    """Raised when the bundled SCG HMM and marker order are inconsistent."""


@dataclass(frozen=True)
class ScgPanel:
    """The validated bundled HMM panel and canonical marker order."""

    hmm_path: Path
    raw_profile_ids: tuple[str, ...]
    canonical_marker_ids: tuple[str, ...]
    hmm_sha256: str
    marker_order_sha256: str
    profiles_with_tc: int

    @property
    def marker_count(self) -> int:
        return len(self.canonical_marker_ids)

    def canonicalize(self, marker_id: str) -> str:
        return canonicalize_marker_id(marker_id)

    def metadata(self) -> dict[str, object]:
        return {
            "hmm_path": str(self.hmm_path),
            "hmm_sha256": self.hmm_sha256,
            "marker_order_sha256": self.marker_order_sha256,
            "raw_profile_count": len(self.raw_profile_ids),
            "canonical_marker_count": self.marker_count,
            "profiles_with_tc": self.profiles_with_tc,
            "alias_schema_version": SCG_ALIAS_SCHEMA_VERSION,
        }


def canonicalize_marker_id(marker_id: str) -> str:
    """Map an HMM profile name to the canonical SCG identifier."""
    value = str(marker_id).strip()
    return SCG_ALIASES.get(value, value)


@lru_cache(maxsize=1)
def load_scg_panel() -> ScgPanel:
    """Load and validate the fixed bundled SCG panel."""
    resolved_hmm = DEFAULT_SCG_HMM.resolve()
    if not resolved_hmm.exists():
        raise FileNotFoundError(f"SCG HMM database not found: {resolved_hmm}")

    raw_profile_ids, profiles_with_tc = load_hmm_profile_ids(resolved_hmm)
    raw_unique = set(raw_profile_ids)
    if len(raw_unique) != len(raw_profile_ids):
        duplicates = sorted(
            marker_id
            for marker_id in raw_unique
            if raw_profile_ids.count(marker_id) > 1
        )
        raise ScgPanelError(
            "SCG HMM contains duplicate profile names: " + ", ".join(duplicates[:8])
        )
    if profiles_with_tc != len(raw_profile_ids):
        raise ScgPanelError(
            "Every SCG HMM profile must define a trusted cutoff (TC): "
            f"profiles={len(raw_profile_ids)}, profiles_with_tc={profiles_with_tc}"
        )

    canonical_hmm = tuple(
        sorted({canonicalize_marker_id(marker_id) for marker_id in raw_profile_ids})
    )
    canonical_marker_ids = _validate_marker_order(
        marker_ids=DEFAULT_SCG_MARKERS,
        canonical_hmm=canonical_hmm,
    )

    return ScgPanel(
        hmm_path=resolved_hmm,
        raw_profile_ids=tuple(raw_profile_ids),
        canonical_marker_ids=tuple(canonical_marker_ids),
        hmm_sha256=sha256_file(resolved_hmm),
        marker_order_sha256=_marker_order_sha256(DEFAULT_SCG_MARKERS),
        profiles_with_tc=profiles_with_tc,
    )


def load_hmm_profile_ids(path: Path) -> tuple[tuple[str, ...], int]:
    """Return HMM profile names and the number of profiles defining TC."""
    profile_ids: list[str] = []
    profiles_with_tc = 0
    current_name: str | None = None
    current_has_tc = False

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("NAME"):
                if current_name is not None:
                    profile_ids.append(current_name)
                    profiles_with_tc += int(current_has_tc)
                fields = line.split()
                if len(fields) < 2:
                    raise ScgPanelError(f"Invalid NAME line in SCG HMM: {line.rstrip()}")
                current_name = str(fields[1]).strip()
                current_has_tc = False
            elif line.startswith("TC") and current_name is not None:
                current_has_tc = True
            elif line.startswith("//") and current_name is not None:
                profile_ids.append(current_name)
                profiles_with_tc += int(current_has_tc)
                current_name = None
                current_has_tc = False

    if current_name is not None:
        profile_ids.append(current_name)
        profiles_with_tc += int(current_has_tc)
    if not profile_ids:
        raise ScgPanelError(f"No HMM profiles found in SCG database: {path}")
    return tuple(profile_ids), int(profiles_with_tc)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest().upper()


def _validate_marker_order(
    *,
    marker_ids: tuple[str, ...],
    canonical_hmm: tuple[str, ...],
) -> tuple[str, ...]:
    canonical_markers = tuple(
        canonicalize_marker_id(marker_id) for marker_id in marker_ids
    )
    if len(set(canonical_markers)) != len(canonical_markers):
        raise ScgPanelError("Embedded SCG marker order contains duplicates.")
    if set(canonical_markers) != set(canonical_hmm):
        missing = sorted(set(canonical_hmm) - set(canonical_markers))
        extra = sorted(set(canonical_markers) - set(canonical_hmm))
        raise ScgPanelError(
            "Embedded SCG marker order does not match canonical HMM profiles: "
            f"missing_from_order={missing[:8]}; extra_in_order={extra[:8]}"
        )
    return canonical_markers


def _marker_order_sha256(marker_ids: tuple[str, ...]) -> str:
    content = "marker_index,marker_id\n" + "".join(
        f"{index},{marker_id}\n"
        for index, marker_id in enumerate(marker_ids)
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest().upper()
