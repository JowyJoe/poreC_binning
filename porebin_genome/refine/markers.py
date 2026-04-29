"""Internal SCG discovery and action-level SCG transition veto for refinement."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from porebin_genome.io.runtime import ensure_dir, write_json
from porebin_genome.refine.models import RefineState


DEFAULT_BACTERIAL_SCG_HMM = (
    Path(__file__).resolve().parents[1] / "evidence" / "scg" / "core_bacterial_scg.hmm"
)
MIN_HMM_COVERAGE = 0.40


@dataclass(frozen=True)
class ScgHit:
    """One ORF-level SCG hit retained from hmmsearch output."""

    orf_id: str
    contig_id: str
    marker_id: str
    domain: str
    bitscore: float
    evalue: float
    hmm_coverage: float


@dataclass(frozen=True)
class ContigScgProfile:
    """Collapsed contig-level SCG profile."""

    contig_id: str
    marker_ids: tuple[str, ...]
    marker_to_domain: dict[str, str]
    n_markers: int
    domain_counts: dict[str, int]


@dataclass(frozen=True)
class BinScgState:
    """Per-bin SCG state under one contig assignment."""

    bin_id: str
    present_markers: frozenset[str]
    duplicate_markers: frozenset[str]
    marker_to_contigs: dict[str, tuple[str, ...]]
    dominant_domain: str | None
    domain_counts: dict[str, int]
    domain_conflict: bool


@dataclass(frozen=True)
class ScgTransitionDecision:
    """Action-level SCG transition result."""

    status: str  # veto | support | abstain
    reason: str
    note: str


@dataclass(frozen=True)
class ScgRunResult:
    """Outputs produced by the internal SCG discovery sub-pipeline."""

    scg_dir: Path
    scg_hmm_path: Path
    proteins_faa: Path
    prodigal_gff: Path
    domtblout_tsv: Path
    contig_index_json: Path
    contig_profiles: dict[str, ContigScgProfile]


def summarize_bin_scg_state(state: BinScgState | None) -> tuple[str, int]:
    """Return a compact public SCG status label and duplicate-marker count."""
    if state is None or not state.present_markers:
        return "no_scg", 0
    duplicate_count = int(len(state.duplicate_markers))
    if duplicate_count > 0:
        return "scg_duplicate_present", duplicate_count
    return "scg_clean", 0


def prepare_scg_profiles(
    *,
    contigs_fasta: Path,
    out_dir: Path,
    scg_hmm_path: Path | None = None,
    prodigal_executable: str = "prodigal",
    hmmsearch_executable: str = "hmmsearch",
) -> ScgRunResult:
    """Run internal SCG discovery and return contig-level SCG profiles."""
    scg_hmm_path = (scg_hmm_path or DEFAULT_BACTERIAL_SCG_HMM).resolve()
    if not scg_hmm_path.exists():
        raise FileNotFoundError(f"SCG HMM database not found: {scg_hmm_path}")

    scg_dir = out_dir.resolve()
    ensure_dir(scg_dir)
    proteins_faa = scg_dir / "orfs.faa"
    prodigal_gff = scg_dir / "orfs.gff"
    hmmsearch_txt = scg_dir / "hmmsearch.txt"
    domtblout_tsv = scg_dir / "raw_hits.domtblout"
    contig_index_json = scg_dir / "contig_scg_index.json"

    prodigal_path, hmmsearch_path = ensure_scg_toolchain_available(
        prodigal_executable=prodigal_executable,
        hmmsearch_executable=hmmsearch_executable,
    )
    _run_prodigal(
        prodigal_path=prodigal_path,
        contigs_fasta=contigs_fasta.resolve(),
        proteins_faa=proteins_faa,
        prodigal_gff=prodigal_gff,
    )
    _run_hmmsearch(
        hmmsearch_path=hmmsearch_path,
        scg_hmm_path=scg_hmm_path,
        proteins_faa=proteins_faa,
        domtblout_tsv=domtblout_tsv,
        hmmsearch_txt=hmmsearch_txt,
    )

    hits = parse_scg_hits(
        domtblout_tsv=domtblout_tsv,
        prodigal_gff=prodigal_gff,
        domain="bacteria",
    )
    contig_profiles = collapse_scg_hits_to_contigs(hits)
    write_json(
        contig_index_json,
        {
            "scg_hmm_path": str(scg_hmm_path),
            "n_contigs_with_scg": len(contig_profiles),
            "profiles": {
                contig_id: {
                    "marker_ids": list(profile.marker_ids),
                    "marker_to_domain": dict(profile.marker_to_domain),
                    "n_markers": int(profile.n_markers),
                    "domain_counts": dict(profile.domain_counts),
                }
                for contig_id, profile in sorted(contig_profiles.items(), key=lambda item: item[0])
            },
        },
    )
    return ScgRunResult(
        scg_dir=scg_dir,
        scg_hmm_path=scg_hmm_path,
        proteins_faa=proteins_faa,
        prodigal_gff=prodigal_gff,
        domtblout_tsv=domtblout_tsv,
        contig_index_json=contig_index_json,
        contig_profiles=contig_profiles,
    )


def ensure_scg_toolchain_available(
    *,
    prodigal_executable: str = "prodigal",
    hmmsearch_executable: str = "hmmsearch",
) -> tuple[str, str]:
    """Resolve required external SCG tools or raise a dependency error."""
    prodigal_path = shutil.which(prodigal_executable)
    hmmsearch_path = shutil.which(hmmsearch_executable)
    missing: list[str] = []
    if not prodigal_path:
        missing.append(str(prodigal_executable))
    if not hmmsearch_path:
        missing.append(str(hmmsearch_executable))
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            "SCG refinement requires external dependencies that were not found in PATH: "
            f"{missing_str}. Install Prodigal and HMMER (hmmsearch), or rerun with SCG disabled."
        )
    return str(prodigal_path), str(hmmsearch_path)


def parse_scg_hits(
    *,
    domtblout_tsv: Path,
    prodigal_gff: Path,
    domain: str,
) -> list[ScgHit]:
    """Parse hmmsearch domtblout into filtered ORF-level SCG hits."""
    orf_to_contig = _parse_prodigal_gff_orf_map(prodigal_gff)
    best_by_orf: dict[str, ScgHit] = {}
    with domtblout_tsv.open("r", encoding="utf-8", newline="") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split(maxsplit=22)
            if len(fields) < 22:
                continue
            orf_id = str(fields[0])
            marker_id = str(fields[3])
            contig_id = orf_to_contig.get(orf_id)
            if not contig_id:
                continue
            hmm_len = int(fields[5])
            i_evalue = float(fields[12])
            bitscore = float(fields[13])
            hmm_from = int(fields[15])
            hmm_to = int(fields[16])
            hmm_coverage = float((hmm_to - hmm_from + 1) / max(hmm_len, 1))
            if hmm_coverage < MIN_HMM_COVERAGE:
                continue
            hit = ScgHit(
                orf_id=orf_id,
                contig_id=contig_id,
                marker_id=marker_id,
                domain=domain,
                bitscore=bitscore,
                evalue=i_evalue,
                hmm_coverage=hmm_coverage,
            )
            current = best_by_orf.get(orf_id)
            if current is None or _scg_hit_sort_key(hit) > _scg_hit_sort_key(current):
                best_by_orf[orf_id] = hit
    return list(best_by_orf.values())


def collapse_scg_hits_to_contigs(hits: list[ScgHit]) -> dict[str, ContigScgProfile]:
    """Collapse ORF-level hits into one marker-presence profile per contig."""
    best_by_contig_marker: dict[tuple[str, str], ScgHit] = {}
    for hit in hits:
        key = (str(hit.contig_id), str(hit.marker_id))
        current = best_by_contig_marker.get(key)
        if current is None or _scg_hit_sort_key(hit) > _scg_hit_sort_key(current):
            best_by_contig_marker[key] = hit

    hits_by_contig: dict[str, list[ScgHit]] = {}
    for hit in best_by_contig_marker.values():
        hits_by_contig.setdefault(str(hit.contig_id), []).append(hit)

    out: dict[str, ContigScgProfile] = {}
    for contig_id, contig_hits in sorted(hits_by_contig.items(), key=lambda item: item[0]):
        contig_hits = sorted(contig_hits, key=lambda item: (str(item.marker_id), -float(item.bitscore), float(item.evalue)))
        marker_to_domain = {str(hit.marker_id): str(hit.domain) for hit in contig_hits}
        domain_counts: dict[str, int] = {}
        for hit in contig_hits:
            domain_counts[str(hit.domain)] = int(domain_counts.get(str(hit.domain), 0) + 1)
        marker_ids = tuple(sorted(marker_to_domain.keys()))
        out[contig_id] = ContigScgProfile(
            contig_id=contig_id,
            marker_ids=marker_ids,
            marker_to_domain=marker_to_domain,
            n_markers=len(marker_ids),
            domain_counts=domain_counts,
        )
    return out


def build_bin_scg_states(
    *,
    assignment: Mapping[str, str],
    contig_profiles: Mapping[str, ContigScgProfile],
) -> dict[str, BinScgState]:
    """Build per-bin SCG states from current assignment and contig-level profiles."""
    markers_by_bin: dict[str, dict[str, list[str]]] = {}
    domain_counts_by_bin: dict[str, dict[str, int]] = {}

    for contig_id, bin_id in assignment.items():
        bin_id = str(bin_id).strip()
        if not bin_id:
            continue
        profile = contig_profiles.get(str(contig_id))
        if profile is None:
            continue
        marker_to_contigs = markers_by_bin.setdefault(bin_id, {})
        for marker_id in profile.marker_ids:
            marker_to_contigs.setdefault(str(marker_id), []).append(str(contig_id))
            domain = str(profile.marker_to_domain.get(marker_id, "bacteria"))
            domain_counts = domain_counts_by_bin.setdefault(bin_id, {})
            domain_counts[domain] = int(domain_counts.get(domain, 0) + 1)

    out: dict[str, BinScgState] = {}
    for bin_id, marker_to_contigs in sorted(markers_by_bin.items(), key=lambda item: item[0]):
        duplicate_markers = frozenset(
            marker_id
            for marker_id, contigs in marker_to_contigs.items()
            if len(set(contigs)) > 1
        )
        domain_counts = domain_counts_by_bin.get(bin_id, {})
        dominant_domain = None
        if domain_counts:
            dominant_domain = max(
                sorted(domain_counts.keys()),
                key=lambda domain: (int(domain_counts[domain]), domain),
            )
        domain_conflict = sum(1 for value in domain_counts.values() if int(value) >= 2) >= 2
        out[bin_id] = BinScgState(
            bin_id=bin_id,
            present_markers=frozenset(marker_to_contigs.keys()),
            duplicate_markers=duplicate_markers,
            marker_to_contigs={
                marker_id: tuple(sorted(set(contigs)))
                for marker_id, contigs in sorted(marker_to_contigs.items(), key=lambda item: item[0])
            },
            dominant_domain=dominant_domain,
            domain_counts={str(key): int(value) for key, value in sorted(domain_counts.items(), key=lambda item: item[0])},
            domain_conflict=bool(domain_conflict),
        )
    return out


def evaluate_split_scg_transition(
    *,
    source_bin: str,
    child_groups: tuple[tuple[str, ...], tuple[str, ...]],
    state: RefineState,
    contig_profiles: Mapping[str, ContigScgProfile],
    new_bin_id: str,
) -> ScgTransitionDecision:
    """Evaluate SCG change caused by splitting one bin into two child bins."""
    relevant = _relevant_contig_profiles(source_bin=source_bin, state=state, contig_profiles=contig_profiles)
    if not relevant:
        return ScgTransitionDecision(status="abstain", reason="no_scg_information", note="source bin has no SCG-bearing contigs")

    before_assignment = dict(state.current_assignment)
    before_states = build_bin_scg_states(assignment=before_assignment, contig_profiles=relevant)
    after_assignment = dict(before_assignment)
    for contig_id in child_groups[1]:
        after_assignment[str(contig_id)] = str(new_bin_id)
    after_states = build_bin_scg_states(assignment=after_assignment, contig_profiles=relevant)

    before_burden = _bin_duplicate_burden(before_states.get(source_bin))
    after_burden = _bin_duplicate_burden(after_states.get(source_bin)) + _bin_duplicate_burden(after_states.get(str(new_bin_id)))
    if after_burden > before_burden:
        return ScgTransitionDecision(
            status="veto",
            reason="split_scg_duplication_worsened",
            note=f"pre_duplicate_burden={before_burden};post_duplicate_burden={after_burden}",
        )
    if after_burden < before_burden:
        return ScgTransitionDecision(
            status="support",
            reason="split_resolves_scg_duplication",
            note=f"pre_duplicate_burden={before_burden};post_duplicate_burden={after_burden}",
        )
    return ScgTransitionDecision(
        status="abstain",
        reason="split_scg_neutral",
        note=f"duplicate_burden={before_burden}",
    )


def evaluate_merge_scg_transition(
    *,
    source_bin: str,
    target_bin: str,
    state: RefineState,
    contig_profiles: Mapping[str, ContigScgProfile],
) -> ScgTransitionDecision:
    """Evaluate SCG change caused by merging target bin into source bin."""
    relevant = _relevant_contig_profiles(
        source_bin=source_bin,
        target_bin=target_bin,
        state=state,
        contig_profiles=contig_profiles,
    )
    if not relevant:
        return ScgTransitionDecision(status="abstain", reason="no_scg_information", note="merge candidate bins have no SCG-bearing contigs")

    before_assignment = dict(state.current_assignment)
    before_states = build_bin_scg_states(assignment=before_assignment, contig_profiles=relevant)
    after_assignment = dict(before_assignment)
    for contig_id, bin_id in list(after_assignment.items()):
        if str(bin_id) == str(target_bin):
            after_assignment[contig_id] = str(source_bin)
    after_states = build_bin_scg_states(assignment=after_assignment, contig_profiles=relevant)

    before_burden = _bin_duplicate_burden(before_states.get(source_bin)) + _bin_duplicate_burden(before_states.get(target_bin))
    after_burden = _bin_duplicate_burden(after_states.get(source_bin))
    if after_burden > before_burden:
        newly_duplicated = sorted(
            set(after_states.get(source_bin, _empty_bin_scg_state(source_bin)).duplicate_markers)
            - set(before_states.get(source_bin, _empty_bin_scg_state(source_bin)).duplicate_markers)
            - set(before_states.get(target_bin, _empty_bin_scg_state(target_bin)).duplicate_markers)
        )
        marker_note = ",".join(newly_duplicated[:8]) if newly_duplicated else "duplicate_markers_increased"
        return ScgTransitionDecision(
            status="veto",
            reason="merge_scg_duplicate_conflict",
            note=(
                f"pre_duplicate_burden={before_burden};"
                f"post_duplicate_burden={after_burden};"
                f"markers={marker_note}"
            ),
        )
    return ScgTransitionDecision(
        status="abstain",
        reason="merge_scg_nonconflicting",
        note=f"pre_duplicate_burden={before_burden};post_duplicate_burden={after_burden}",
    )


def evaluate_reassign_scg_transition(
    *,
    contig_id: str,
    source_bin: str,
    target_bin: str,
    state: RefineState,
    contig_profiles: Mapping[str, ContigScgProfile],
) -> ScgTransitionDecision:
    """Evaluate SCG change caused by moving one contig between bins."""
    contig_profile = contig_profiles.get(str(contig_id))
    if contig_profile is None or contig_profile.n_markers == 0:
        return ScgTransitionDecision(status="abstain", reason="no_scg_information", note="contig carries no SCG")

    relevant = _relevant_contig_profiles(
        source_bin=source_bin,
        target_bin=target_bin,
        state=state,
        contig_profiles=contig_profiles,
    )
    before_assignment = dict(state.current_assignment)
    before_states = build_bin_scg_states(assignment=before_assignment, contig_profiles=relevant)
    after_assignment = dict(before_assignment)
    after_assignment[str(contig_id)] = str(target_bin)
    after_states = build_bin_scg_states(assignment=after_assignment, contig_profiles=relevant)

    source_before = _bin_duplicate_burden(before_states.get(source_bin))
    source_after = _bin_duplicate_burden(after_states.get(source_bin))
    target_before = _bin_duplicate_burden(before_states.get(target_bin))
    target_after = _bin_duplicate_burden(after_states.get(target_bin))
    if target_after > target_before:
        new_target_duplicates = sorted(
            set(after_states.get(target_bin, _empty_bin_scg_state(target_bin)).duplicate_markers)
            - set(before_states.get(target_bin, _empty_bin_scg_state(target_bin)).duplicate_markers)
        )
        return ScgTransitionDecision(
            status="veto",
            reason="reassign_scg_duplicate_conflict",
            note=(
                f"source_pre={source_before};source_post={source_after};"
                f"target_pre={target_before};target_post={target_after};"
                f"markers={','.join(new_target_duplicates[:8]) if new_target_duplicates else 'duplicate_markers_increased'}"
            ),
        )
    if source_after < source_before:
        return ScgTransitionDecision(
            status="support",
            reason="reassign_resolves_source_scg_duplication",
            note=(
                f"source_pre={source_before};source_post={source_after};"
                f"target_pre={target_before};target_post={target_after}"
            ),
        )
    return ScgTransitionDecision(
        status="abstain",
        reason="reassign_scg_neutral",
        note=(
            f"source_pre={source_before};source_post={source_after};"
            f"target_pre={target_before};target_post={target_after}"
        ),
    )


def evaluate_recruit_scg_transition(
    *,
    contig_id: str,
    target_bin: str,
    state: RefineState,
    contig_profiles: Mapping[str, ContigScgProfile],
) -> ScgTransitionDecision:
    """Evaluate SCG change caused by recruiting one unbinned contig into a bin."""
    contig_profile = contig_profiles.get(str(contig_id))
    if contig_profile is None or contig_profile.n_markers == 0:
        return ScgTransitionDecision(status="abstain", reason="no_scg_information", note="contig carries no SCG")

    relevant = _relevant_contig_profiles(target_bin=target_bin, state=state, contig_profiles=contig_profiles)
    relevant = dict(relevant)
    relevant[str(contig_id)] = contig_profile
    before_assignment = dict(state.current_assignment)
    before_states = build_bin_scg_states(assignment=before_assignment, contig_profiles=relevant)
    after_assignment = dict(before_assignment)
    after_assignment[str(contig_id)] = str(target_bin)
    after_states = build_bin_scg_states(assignment=after_assignment, contig_profiles=relevant)

    target_before = _bin_duplicate_burden(before_states.get(target_bin))
    target_after = _bin_duplicate_burden(after_states.get(target_bin))
    if target_after > target_before:
        new_target_duplicates = sorted(
            set(after_states.get(target_bin, _empty_bin_scg_state(target_bin)).duplicate_markers)
            - set(before_states.get(target_bin, _empty_bin_scg_state(target_bin)).duplicate_markers)
        )
        return ScgTransitionDecision(
            status="veto",
            reason="recruit_scg_duplicate_conflict",
            note=(
                f"target_pre={target_before};target_post={target_after};"
                f"markers={','.join(new_target_duplicates[:8]) if new_target_duplicates else 'duplicate_markers_increased'}"
            ),
        )
    return ScgTransitionDecision(
        status="abstain",
        reason="recruit_scg_nonconflicting",
        note=f"target_pre={target_before};target_post={target_after}",
    )


def _run_prodigal(
    *,
    prodigal_path: str,
    contigs_fasta: Path,
    proteins_faa: Path,
    prodigal_gff: Path,
) -> None:
    command = [
        str(prodigal_path),
        "-i",
        str(contigs_fasta),
        "-a",
        str(proteins_faa),
        "-o",
        str(prodigal_gff),
        "-f",
        "gff",
        "-p",
        "meta",
        "-q",
    ]
    _run_external_command(command, tool_name="prodigal")


def _run_hmmsearch(
    *,
    hmmsearch_path: str,
    scg_hmm_path: Path,
    proteins_faa: Path,
    domtblout_tsv: Path,
    hmmsearch_txt: Path,
) -> None:
    command = [
        str(hmmsearch_path),
        "--noali",
        "--cut_tc",
        "--domtblout",
        str(domtblout_tsv),
        "-o",
        str(hmmsearch_txt),
        str(scg_hmm_path),
        str(proteins_faa),
    ]
    _run_external_command(command, tool_name="hmmsearch")


def _run_external_command(command: list[str], *, tool_name: str) -> None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - protected by preflight
        raise RuntimeError(f"Required external dependency not found: {tool_name}") from exc
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        message = stderr or stdout or f"{tool_name} exited with status {completed.returncode}"
        raise RuntimeError(f"{tool_name} failed while running internal SCG discovery: {message}")


def _parse_prodigal_gff_orf_map(prodigal_gff: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with prodigal_gff.open("r", encoding="utf-8", newline="") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split("\t")
            if len(fields) < 9 or fields[2] != "CDS":
                continue
            contig_id = str(fields[0]).strip()
            attributes = _parse_gff_attributes(fields[8])
            orf_id = str(attributes.get("ID", "")).strip()
            if contig_id and orf_id:
                out[orf_id] = contig_id
    return out


def _parse_gff_attributes(field: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in str(field).strip().split(";"):
        if not token or "=" not in token:
            continue
        key, value = token.split("=", 1)
        out[str(key).strip()] = str(value).strip()
    return out


def _scg_hit_sort_key(hit: ScgHit) -> tuple[float, float, float]:
    return (float(hit.bitscore), float(hit.hmm_coverage), -float(hit.evalue))


def _relevant_contig_profiles(
    *,
    state: RefineState,
    contig_profiles: Mapping[str, ContigScgProfile],
    source_bin: str | None = None,
    target_bin: str | None = None,
) -> dict[str, ContigScgProfile]:
    bins = {str(bin_id) for bin_id in (source_bin, target_bin) if bin_id}
    out: dict[str, ContigScgProfile] = {}
    for contig_id, bin_id in state.current_assignment.items():
        if str(bin_id) not in bins:
            continue
        profile = contig_profiles.get(str(contig_id))
        if profile is not None:
            out[str(contig_id)] = profile
    return out


def _bin_duplicate_burden(state: BinScgState | None) -> int:
    if state is None:
        return 0
    burden = 0
    for contigs in state.marker_to_contigs.values():
        burden += max(0, len(tuple(sorted(set(contigs)))) - 1)
    return int(burden)


def _empty_bin_scg_state(bin_id: str) -> BinScgState:
    return BinScgState(
        bin_id=str(bin_id),
        present_markers=frozenset(),
        duplicate_markers=frozenset(),
        marker_to_contigs={},
        dominant_domain=None,
        domain_counts={},
        domain_conflict=False,
    )
