"""Suspect-bin detection and bin-level snapshot metrics for refine MVP."""

from __future__ import annotations

from porebin_genome.refine.models import BinFeatureProfile, BinSnapshot, RefineState, SupportSummary, SuspectBin
from porebin_genome.refine.support import compute_contact_consistency


def build_bin_snapshots(
    *,
    state: RefineState,
    support_summary: SupportSummary,
    profiles: dict[str, BinFeatureProfile],
) -> dict[str, BinSnapshot]:
    """Build bin-level metric snapshots for suspect detection and QC."""
    snapshots: dict[str, BinSnapshot] = {}
    for bin_id, members in state.bin_to_contigs().items():
        total_length = int(sum(state.contig_lengths.get(contig_id, 0) for contig_id in members))
        coverage_values = [
            float(state.coverage_by_contig[contig_id])
            for contig_id in members
            if contig_id in state.coverage_by_contig
        ]
        profile = profiles.get(bin_id)
        contact_consistency = compute_contact_consistency(
            bin_id=bin_id,
            members=members,
            support_summary=support_summary,
        )
        low_support = 0
        for contig_id in members:
            total = float(support_summary.total_support.get(contig_id, 0.0))
            own = float(support_summary.support_by_contig_bin.get(contig_id, {}).get(bin_id, 0.0))
            if total <= 0.0 or own / total < 0.55:
                low_support += 1
        suspect_reasons = _detect_suspect_reasons(
            members=members,
            contact_consistency=contact_consistency,
            contact_components=len(support_summary.contact_components_by_bin.get(bin_id, ()) or ()),
            low_support_ratio=(float(low_support) / float(len(members))) if members else 0.0,
            feature_dispersion=(float(profile.feature_dispersion) if profile is not None else 0.0),
            coverage_dispersion=(profile.coverage_mad if profile is not None else None),
        )
        snapshots[bin_id] = BinSnapshot(
            bin_id=bin_id,
            members=tuple(sorted(members)),
            n_contigs=len(members),
            total_length=total_length,
            median_coverage=(
                float(profile.median_coverage)
                if profile is not None and profile.median_coverage is not None
                else None
            ),
            coverage_dispersion=(
                float(profile.coverage_mad)
                if profile is not None and profile.coverage_mad is not None
                else None
            ),
            feature_dispersion=(float(profile.feature_dispersion) if profile is not None else 0.0),
            contact_consistency=float(contact_consistency),
            contact_components=len(support_summary.contact_components_by_bin.get(bin_id, ()) or ()),
            low_support_ratio=(float(low_support) / float(len(members))) if members else 0.0,
            suspect_flag=bool(suspect_reasons),
            suspect_reasons=tuple(suspect_reasons),
            refine_status=("suspect" if suspect_reasons else "stable"),
        )
    return snapshots


def detect_suspect_bins(*, snapshots: dict[str, BinSnapshot]) -> list[SuspectBin]:
    """Return suspect bins that qualify for local split attempts."""
    out: list[SuspectBin] = []
    for snapshot in snapshots.values():
        if snapshot.suspect_flag:
            out.append(SuspectBin(bin_id=snapshot.bin_id, reasons=snapshot.suspect_reasons))
    return sorted(out, key=lambda item: item.bin_id)


def _detect_suspect_reasons(
    *,
    members: list[str],
    contact_consistency: float,
    contact_components: int,
    low_support_ratio: float,
    feature_dispersion: float,
    coverage_dispersion: float | None,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if len(members) < 4:
        return ()
    if contact_components > 1:
        reasons.append("multiple_contact_components")
    if contact_consistency < 0.60:
        reasons.append("low_contact_consistency")
    if low_support_ratio >= 0.25:
        reasons.append("high_low_support_ratio")
    if feature_dispersion > 0.85:
        reasons.append("high_feature_dispersion")
    if coverage_dispersion is not None and coverage_dispersion > 5.0:
        reasons.append("high_coverage_dispersion")

    if "multiple_contact_components" in reasons:
        return tuple(reasons)
    if len(reasons) >= 2:
        return tuple(reasons)
    return ()
