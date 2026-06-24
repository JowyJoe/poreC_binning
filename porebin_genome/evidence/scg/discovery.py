"""Generate fixed-panel contig-level SCG evidence."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from porebin_genome.evidence.scg.panel import (
    SCG_ALIAS_SCHEMA_VERSION,
    ScgPanel,
    canonicalize_marker_id,
    load_scg_panel,
)
from porebin_genome.io.runtime import ensure_dir, write_json


MIN_HMM_COVERAGE = 0.40
SCG_CACHE_SCHEMA_VERSION = 1
SCG_PARSER_SCHEMA_VERSION = 3


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
    raw_marker_id: str = ""


@dataclass(frozen=True)
class ContigScgProfile:
    """Collapsed contig-level SCG profile."""

    contig_id: str
    marker_ids: tuple[str, ...]
    marker_to_domain: dict[str, str]
    n_markers: int
    domain_counts: dict[str, int]
    marker_orf_counts: dict[str, int] = field(default_factory=dict)


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
    run_meta_json: Path
    panel: ScgPanel
    cache_status: str


def prepare_scg_profiles(
    *,
    contigs_fasta: Path,
    out_dir: Path,
    prodigal_executable: str = "prodigal",
    hmmsearch_executable: str = "hmmsearch",
) -> ScgRunResult:
    """Run internal SCG discovery and return contig-level SCG profiles."""
    panel = load_scg_panel()

    scg_dir = out_dir.resolve()
    ensure_dir(scg_dir)
    proteins_faa = scg_dir / "orfs.faa"
    prodigal_gff = scg_dir / "orfs.gff"
    hmmsearch_txt = scg_dir / "hmmsearch.txt"
    domtblout_tsv = scg_dir / "raw_hits.domtblout"
    contig_index_json = scg_dir / "contig_scg_index.json"
    run_meta_json = scg_dir / "scg_run_meta.json"

    fasta_fingerprint = _file_fingerprint(contigs_fasta.resolve())
    prodigal_fingerprint = {
        "schema_version": 1,
        "fasta": fasta_fingerprint,
        "executable": str(prodigal_executable),
        "mode": "meta",
        "output_format": "gff",
    }
    hmmsearch_fingerprint = {
        "schema_version": 1,
        "prodigal": prodigal_fingerprint,
        "executable": str(hmmsearch_executable),
        "hmm_sha256": panel.hmm_sha256,
        "cutoff": "trusted_cutoff",
        "no_alignment": True,
    }
    parser_fingerprint = {
        "schema_version": SCG_PARSER_SCHEMA_VERSION,
        "hmmsearch": hmmsearch_fingerprint,
        "marker_order_sha256": panel.marker_order_sha256,
        "alias_schema_version": SCG_ALIAS_SCHEMA_VERSION,
        "min_hmm_coverage": MIN_HMM_COVERAGE,
    }
    cached_meta = _read_json_if_available(run_meta_json)
    prodigal_cached = (
        cached_meta.get("prodigal_fingerprint") == prodigal_fingerprint
        and proteins_faa.exists()
        and prodigal_gff.exists()
    )
    hmmsearch_cached = (
        prodigal_cached
        and cached_meta.get("hmmsearch_fingerprint") == hmmsearch_fingerprint
        and domtblout_tsv.exists()
        and hmmsearch_txt.exists()
    )
    parser_cached = (
        hmmsearch_cached
        and cached_meta.get("parser_fingerprint") == parser_fingerprint
        and contig_index_json.exists()
    )
    if parser_cached:
        contig_profiles = _load_contig_profiles(contig_index_json)
        return ScgRunResult(
            scg_dir=scg_dir,
            scg_hmm_path=panel.hmm_path,
            proteins_faa=proteins_faa,
            prodigal_gff=prodigal_gff,
            domtblout_tsv=domtblout_tsv,
            contig_index_json=contig_index_json,
            contig_profiles=contig_profiles,
            run_meta_json=run_meta_json,
            panel=panel,
            cache_status="complete_cache_hit",
        )

    if not prodigal_cached:
        prodigal_path = _resolve_required_executable(
            prodigal_executable,
            purpose="SCG ORF prediction",
        )
        _run_prodigal(
            prodigal_path=prodigal_path,
            contigs_fasta=contigs_fasta.resolve(),
            proteins_faa=proteins_faa,
            prodigal_gff=prodigal_gff,
        )
        cached_meta = {
            "schema_version": SCG_CACHE_SCHEMA_VERSION,
            "prodigal_fingerprint": prodigal_fingerprint,
            "hmmsearch_fingerprint": None,
            "parser_fingerprint": None,
            "panel": panel.metadata(),
        }
        write_json(run_meta_json, cached_meta)
        hmmsearch_cached = False

    if not hmmsearch_cached:
        hmmsearch_path = _resolve_required_executable(
            hmmsearch_executable,
            purpose="SCG HMM search",
        )
        _run_hmmsearch(
            hmmsearch_path=hmmsearch_path,
            scg_hmm_path=panel.hmm_path,
            proteins_faa=proteins_faa,
            domtblout_tsv=domtblout_tsv,
            hmmsearch_txt=hmmsearch_txt,
        )
        cached_meta = {
            "schema_version": SCG_CACHE_SCHEMA_VERSION,
            "prodigal_fingerprint": prodigal_fingerprint,
            "hmmsearch_fingerprint": hmmsearch_fingerprint,
            "parser_fingerprint": None,
            "panel": panel.metadata(),
        }
        write_json(run_meta_json, cached_meta)

    hits = parse_scg_hits(
        domtblout_tsv=domtblout_tsv,
        prodigal_gff=prodigal_gff,
        domain="bacteria",
    )
    contig_profiles = collapse_scg_hits_to_contigs(hits)
    write_json(
        contig_index_json,
        {
            "panel": panel.metadata(),
            "n_contigs_with_scg": len(contig_profiles),
            "profiles": {
                contig_id: {
                    "marker_ids": list(profile.marker_ids),
                    "marker_to_domain": dict(profile.marker_to_domain),
                    "n_markers": int(profile.n_markers),
                    "domain_counts": dict(profile.domain_counts),
                    "marker_orf_counts": dict(profile.marker_orf_counts),
                }
                for contig_id, profile in sorted(
                    contig_profiles.items(),
                    key=lambda item: item[0],
                )
            },
        },
    )
    write_json(
        run_meta_json,
        {
            "schema_version": SCG_CACHE_SCHEMA_VERSION,
            "prodigal_fingerprint": prodigal_fingerprint,
            "hmmsearch_fingerprint": hmmsearch_fingerprint,
            "parser_fingerprint": parser_fingerprint,
            "panel": panel.metadata(),
        },
    )
    return ScgRunResult(
        scg_dir=scg_dir,
        scg_hmm_path=panel.hmm_path,
        proteins_faa=proteins_faa,
        prodigal_gff=prodigal_gff,
        domtblout_tsv=domtblout_tsv,
        contig_index_json=contig_index_json,
        contig_profiles=contig_profiles,
        run_meta_json=run_meta_json,
        panel=panel,
        cache_status=(
            "parser_rebuilt"
            if hmmsearch_cached
            else ("hmmsearch_rebuilt" if prodigal_cached else "full_rebuild")
        ),
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
            "SCG refinement requires external dependencies that were not "
            f"found in PATH: {missing_str}. Install Prodigal and HMMER "
            "(hmmsearch), or rerun with SCG disabled."
        )
    return str(prodigal_path), str(hmmsearch_path)


def parse_scg_hits(
    *,
    domtblout_tsv: Path,
    prodigal_gff: Path,
    domain: str,
) -> list[ScgHit]:
    """Parse hmmsearch domtblout into filtered ORF-level SCG hits."""
    panel = load_scg_panel()
    raw_profile_ids = set(panel.raw_profile_ids)
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
            raw_marker_id = str(fields[3])
            if raw_marker_id not in raw_profile_ids:
                raise RuntimeError(
                    "SCG domtblout contains a marker that is absent from the "
                    f"configured HMM panel: {raw_marker_id}"
                )
            marker_id = panel.canonicalize(raw_marker_id)
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
                raw_marker_id=raw_marker_id,
            )
            current = best_by_orf.get(orf_id)
            if current is None or _scg_hit_sort_key(hit) > _scg_hit_sort_key(current):
                best_by_orf[orf_id] = hit
    return list(best_by_orf.values())


def collapse_scg_hits_to_contigs(hits: list[ScgHit]) -> dict[str, ContigScgProfile]:
    """Collapse ORF-level hits into one marker-presence profile per contig."""
    best_by_contig_marker: dict[tuple[str, str], ScgHit] = {}
    orfs_by_contig_marker: dict[tuple[str, str], set[str]] = {}
    for hit in hits:
        canonical_marker = canonicalize_marker_id(hit.marker_id)
        key = (str(hit.contig_id), canonical_marker)
        orfs_by_contig_marker.setdefault(key, set()).add(str(hit.orf_id))
        current = best_by_contig_marker.get(key)
        if current is None or _scg_hit_sort_key(hit) > _scg_hit_sort_key(current):
            best_by_contig_marker[key] = hit

    hits_by_contig: dict[str, list[tuple[str, ScgHit]]] = {}
    for (contig_id, marker_id), hit in best_by_contig_marker.items():
        hits_by_contig.setdefault(str(contig_id), []).append((marker_id, hit))

    out: dict[str, ContigScgProfile] = {}
    for contig_id, contig_hits in sorted(
        hits_by_contig.items(),
        key=lambda item: item[0],
    ):
        contig_hits = sorted(
            contig_hits,
            key=lambda item: (
                str(item[0]),
                -float(item[1].bitscore),
                float(item[1].evalue),
            ),
        )
        marker_to_domain = {
            str(marker_id): str(hit.domain)
            for marker_id, hit in contig_hits
        }
        domain_counts: dict[str, int] = {}
        for _marker_id, hit in contig_hits:
            domain_counts[str(hit.domain)] = int(domain_counts.get(str(hit.domain), 0) + 1)
        marker_ids = tuple(sorted(marker_to_domain.keys()))
        marker_orf_counts = {
            marker_id: len(orfs_by_contig_marker.get((contig_id, marker_id), ()))
            for marker_id in marker_ids
        }
        out[contig_id] = ContigScgProfile(
            contig_id=contig_id,
            marker_ids=marker_ids,
            marker_to_domain=marker_to_domain,
            n_markers=len(marker_ids),
            domain_counts=domain_counts,
            marker_orf_counts=marker_orf_counts,
        )
    return out


def _file_fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _read_json_if_available(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_contig_profiles(path: Path) -> dict[str, ContigScgProfile]:
    payload = _read_json_if_available(path)
    raw_profiles = payload.get("profiles", {})
    if not isinstance(raw_profiles, dict):
        raise RuntimeError(f"Invalid SCG contig index: {path}")

    out: dict[str, ContigScgProfile] = {}
    for contig_id, raw_profile in raw_profiles.items():
        if not isinstance(raw_profile, dict):
            continue
        marker_ids = tuple(
            str(marker_id)
            for marker_id in raw_profile.get("marker_ids", [])
        )
        out[str(contig_id)] = ContigScgProfile(
            contig_id=str(contig_id),
            marker_ids=marker_ids,
            marker_to_domain={
                str(marker_id): str(domain)
                for marker_id, domain in dict(
                    raw_profile.get("marker_to_domain", {})
                ).items()
            },
            n_markers=int(raw_profile.get("n_markers", len(marker_ids))),
            domain_counts={
                str(domain): int(count)
                for domain, count in dict(
                    raw_profile.get("domain_counts", {})
                ).items()
            },
            marker_orf_counts={
                str(marker_id): int(count)
                for marker_id, count in dict(
                    raw_profile.get("marker_orf_counts", {})
                ).items()
            },
        )
    return out


def _resolve_required_executable(executable: str, *, purpose: str) -> str:
    resolved = shutil.which(executable)
    if resolved:
        return str(resolved)
    raise RuntimeError(
        f"{purpose} requires an external dependency that was not found in PATH: "
        f"{executable}"
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
        raise RuntimeError(
            f"Required external dependency not found: {tool_name}"
        ) from exc
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        message = (
            stderr
            or stdout
            or f"{tool_name} exited with status {completed.returncode}"
        )
        raise RuntimeError(
            f"{tool_name} failed while running internal SCG discovery: "
            f"{message}"
        )


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
                suffix = orf_id.rsplit("_", 1)[-1]
                if suffix and suffix != orf_id:
                    out[f"{contig_id}_{suffix}"] = contig_id
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
