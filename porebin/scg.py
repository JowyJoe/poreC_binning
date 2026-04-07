from __future__ import annotations

import csv
import json
import logging
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class ScgError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScgResources:
    db_dir: Path
    marker_hmm: Path
    manifest_json: Path
    marker_set_id: str
    db_version: Optional[str]
    expected_markers: tuple[str, ...]


@dataclass(frozen=True)
class ScgRunResult:
    state: str
    enabled: bool
    hits_tsv: Optional[Path]
    proteins_faa: Optional[Path]
    domtblout: Optional[Path]
    resources: Optional[ScgResources]
    expected_markers: tuple[str, ...]
    reused_cache: bool
    reason: str
    gene_caller: Optional[str] = None
    marker_search: Optional[str] = None


@dataclass(frozen=True)
class ScgHit:
    contig_name: str
    gene_id: str
    marker_id: str
    bitscore: float
    evalue: float
    hmm_from: int
    hmm_to: int
    ali_from: int
    ali_to: int
    env_from: int
    env_to: int
    orf_start: int
    orf_end: int
    strand: str


@dataclass(frozen=True)
class BinScgQc:
    bin_id: str
    expected_markers: int
    unique_markers: int
    duplicated_markers: int
    total_marker_hits: int
    completeness_like: Optional[float]
    contamination_like: Optional[float]
    implicated_contigs: tuple[str, ...]


_SEQHDR_RE = re.compile(r'seqhdr="([^"]+)"')


def resolve_scg_resources(db_dir: Optional[Path] = None) -> ScgResources:
    root = (db_dir or (Path(__file__).resolve().parent / "scg_db")).resolve()
    manifest_json = root / "manifest.json"
    if not manifest_json.exists():
        raise ScgError(f"SCG manifest missing: {manifest_json}")

    try:
        manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        raise ScgError(f"Invalid SCG manifest JSON: {manifest_json}") from exc

    marker_hmm_name = str(manifest.get("marker_hmm", "")).strip()
    if marker_hmm_name:
        marker_hmm = root / marker_hmm_name
    else:
        default_marker_hmm = root / "marker.hmm"
        hmm_candidates = sorted(root.glob("*.hmm"))
        if default_marker_hmm.exists():
            marker_hmm = default_marker_hmm
        elif len(hmm_candidates) == 1:
            marker_hmm = hmm_candidates[0]
        else:
            marker_hmm = default_marker_hmm

    if not marker_hmm.exists():
        raise ScgError(f"SCG marker HMM missing: {marker_hmm}")

    expected_markers = tuple(str(x).strip() for x in manifest.get("expected_markers", []) if str(x).strip())
    return ScgResources(
        db_dir=root,
        marker_hmm=marker_hmm,
        manifest_json=manifest_json,
        marker_set_id=str(manifest.get("marker_set_id", "unknown")).strip() or "unknown",
        db_version=(str(manifest.get("db_version")).strip() if manifest.get("db_version") is not None else None),
        expected_markers=expected_markers,
    )


def ensure_scg_hits(
    *,
    contigs_fasta: Path,
    out_dir: Path,
    db_dir: Optional[Path] = None,
    prodigal_exe: str = "prodigal",
    hmmsearch_exe: str = "hmmsearch",
    threads: int = 1,
    force: bool = False,
    logger: Optional[logging.Logger] = None,
) -> ScgRunResult:
    """
    Ensure a cached SCG hit table exists.

    Primary path:
      contigs.fasta -> prodigal proteins.faa -> hmmsearch domtblout -> scg_hits.tsv

    If the built-in SCG database or required executables are unavailable, return a disabled
    result rather than failing hard. This keeps the current refine path usable while SCG-aware
    refinement is being phased in.
    """
    logger = logger or logging.getLogger("porebin")
    contigs_fasta = contigs_fasta.resolve()
    out_dir = out_dir.resolve()
    scg_dir = out_dir / "scg"
    scg_dir.mkdir(parents=True, exist_ok=True)

    hits_tsv = scg_dir / "scg_hits.tsv"
    proteins_faa = scg_dir / "proteins.faa"
    domtblout = scg_dir / "hits.domtblout"

    resources: Optional[ScgResources] = None
    resource_reason = ""
    try:
        resources = resolve_scg_resources(db_dir=db_dir)
    except ScgError as exc:
        resource_reason = str(exc)

    if hits_tsv.exists() and not force:
        expected_markers = resources.expected_markers if resources is not None else ()
        state = "enabled" if expected_markers else "partial"
        reason = "cached_hits" if resources is not None else f"cached_hits_without_resources: {resource_reason}"
        return ScgRunResult(
            state=state,
            enabled=True,
            hits_tsv=hits_tsv,
            proteins_faa=proteins_faa if proteins_faa.exists() else None,
            domtblout=domtblout if domtblout.exists() else None,
            resources=resources,
            expected_markers=tuple(expected_markers),
            reused_cache=True,
            reason=reason,
        )

    if resources is None:
        logger.info("SCG disabled: %s", resource_reason)
        return ScgRunResult(
            state="disabled",
            enabled=False,
            hits_tsv=None,
            proteins_faa=None,
            domtblout=None,
            resources=None,
            expected_markers=(),
            reused_cache=False,
            reason=resource_reason,
        )

    try:
        subprocess.run(
            [
                prodigal_exe,
                "-i",
                str(contigs_fasta),
                "-a",
                str(proteins_faa),
                "-p",
                "meta",
                "-q",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        reason = f"SCG disabled: gene caller not found: {prodigal_exe}"
        logger.info(reason)
        return ScgRunResult(
            state="disabled",
            enabled=False,
            hits_tsv=None,
            proteins_faa=None,
            domtblout=None,
            resources=resources,
            expected_markers=tuple(resources.expected_markers),
            reused_cache=False,
            reason=reason,
            gene_caller=prodigal_exe,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        raise ScgError(f"Prodigal failed: {exc.stderr or exc.stdout or exc}") from exc

    gene_meta = _load_prodigal_gene_meta(proteins_faa)
    if not gene_meta:
        raise ScgError(f"No proteins were produced for SCG calling: {proteins_faa}")

    hmm_cmd = [
        hmmsearch_exe,
        "--noali",
        "--domtblout",
        str(domtblout),
        "--cpu",
        str(max(1, int(threads))),
        str(resources.marker_hmm),
        str(proteins_faa),
    ]
    try:
        subprocess.run(hmm_cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        reason = f"SCG disabled: marker search tool not found: {hmmsearch_exe}"
        logger.info(reason)
        return ScgRunResult(
            state="disabled",
            enabled=False,
            hits_tsv=None,
            proteins_faa=proteins_faa,
            domtblout=None,
            resources=resources,
            expected_markers=tuple(resources.expected_markers),
            reused_cache=False,
            reason=reason,
            gene_caller=prodigal_exe,
            marker_search=hmmsearch_exe,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        raise ScgError(f"hmmsearch failed: {exc.stderr or exc.stdout or exc}") from exc

    hits = parse_hmmsearch_domtblout(domtblout, gene_meta=gene_meta)
    write_scg_hits(hits_tsv, hits)
    state = "enabled" if resources.expected_markers else "partial"
    return ScgRunResult(
        state=state,
        enabled=True,
        hits_tsv=hits_tsv,
        proteins_faa=proteins_faa,
        domtblout=domtblout,
        resources=resources,
        expected_markers=tuple(resources.expected_markers),
        reused_cache=False,
        reason="generated",
        gene_caller=prodigal_exe,
        marker_search=hmmsearch_exe,
    )


def read_scg_hits(path: Path) -> list[ScgHit]:
    path = path.resolve()
    hits: list[ScgHit] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if not row:
                continue
            hits.append(
                ScgHit(
                    contig_name=str(row["contig_name"]).strip(),
                    gene_id=str(row["gene_id"]).strip(),
                    marker_id=str(row["marker_id"]).strip(),
                    bitscore=float(row["bitscore"]),
                    evalue=float(row["evalue"]),
                    hmm_from=int(row["hmm_from"]),
                    hmm_to=int(row["hmm_to"]),
                    ali_from=int(row["ali_from"]),
                    ali_to=int(row["ali_to"]),
                    env_from=int(row["env_from"]),
                    env_to=int(row["env_to"]),
                    orf_start=int(row["orf_start"]),
                    orf_end=int(row["orf_end"]),
                    strand=str(row["strand"]).strip(),
                )
            )
    return hits


def write_scg_hits(path: Path, hits: list[ScgHit]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(
            "contig_name\tgene_id\tmarker_id\tbitscore\tevalue\thmm_from\thmm_to\tali_from\tali_to\t"
            "env_from\tenv_to\torf_start\torf_end\tstrand\n"
        )
        for hit in hits:
            fh.write(
                f"{hit.contig_name}\t{hit.gene_id}\t{hit.marker_id}\t{hit.bitscore:.6g}\t{hit.evalue:.6g}\t"
                f"{hit.hmm_from}\t{hit.hmm_to}\t{hit.ali_from}\t{hit.ali_to}\t"
                f"{hit.env_from}\t{hit.env_to}\t{hit.orf_start}\t{hit.orf_end}\t{hit.strand}\n"
            )


def compute_bin_scg_qc(
    *,
    contig_to_bin: dict[str, str],
    hits: list[ScgHit],
    expected_markers: Optional[set[str]] = None,
) -> dict[str, BinScgQc]:
    hits_by_bin_marker: dict[str, dict[str, dict[str, set[str]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    implicated_by_bin: dict[str, set[str]] = defaultdict(set)

    for hit in hits:
        bin_id = contig_to_bin.get(hit.contig_name)
        if not bin_id or str(bin_id).strip() == "-1":
            continue
        hits_by_bin_marker[str(bin_id)][hit.marker_id][hit.contig_name].add(hit.gene_id)

    expected_count = int(len(expected_markers)) if expected_markers else 0
    out: dict[str, BinScgQc] = {}
    for bin_id, marker_map in hits_by_bin_marker.items():
        unique_markers = 0
        duplicated_markers = 0
        total_hits = 0
        for marker_id, contig_gene_map in marker_map.items():
            copies = sum(len(gene_ids) for gene_ids in contig_gene_map.values())
            if copies <= 0:
                continue
            unique_markers += 1
            total_hits += copies
            if copies > 1:
                duplicated_markers += 1
                implicated_by_bin[bin_id].update(contig_gene_map.keys())

        completeness_like = None
        contamination_like = None
        if expected_count > 0:
            completeness_like = unique_markers / float(expected_count)
            contamination_like = duplicated_markers / float(expected_count)

        out[bin_id] = BinScgQc(
            bin_id=bin_id,
            expected_markers=expected_count,
            unique_markers=unique_markers,
            duplicated_markers=duplicated_markers,
            total_marker_hits=total_hits,
            completeness_like=completeness_like,
            contamination_like=contamination_like,
            implicated_contigs=tuple(sorted(implicated_by_bin.get(bin_id, set()))),
        )
    return out


def parse_hmmsearch_domtblout(
    domtblout: Path,
    *,
    gene_meta: dict[str, tuple[str, int, int, str]],
) -> list[ScgHit]:
    """
    Parse hmmsearch --domtblout output.

    For hmmsearch, target_name is the sequence / ORF identifier and query_name is the marker HMM.
    """
    domtblout = domtblout.resolve()
    hits: list[ScgHit] = []
    with domtblout.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            row = line.strip().split()
            if len(row) < 23:
                continue
            gene_id = str(row[0]).strip()
            marker_id = str(row[3]).strip()
            meta = gene_meta.get(gene_id)
            if meta is None:
                continue
            contig_name, orf_start, orf_end, strand = meta
            hits.append(
                ScgHit(
                    contig_name=contig_name,
                    gene_id=gene_id,
                    marker_id=marker_id,
                    bitscore=float(row[13]),
                    evalue=float(row[12]),
                    hmm_from=int(row[15]),
                    hmm_to=int(row[16]),
                    ali_from=int(row[17]),
                    ali_to=int(row[18]),
                    env_from=int(row[19]),
                    env_to=int(row[20]),
                    orf_start=int(orf_start),
                    orf_end=int(orf_end),
                    strand=strand,
                )
            )
    return hits


def _load_prodigal_gene_meta(path: Path) -> dict[str, tuple[str, int, int, str]]:
    """
    Load prodigal protein headers and map gene_id -> (contig_name, start, end, strand).

    Prodigal protein headers commonly look like:
      >1_1 # 3 # 302 # 1 # ID=1_1;...;seqhdr="contigA"
    """
    path = path.resolve()
    meta: dict[str, tuple[str, int, int, str]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            header = line[1:].strip()
            if not header:
                continue
            parts = [p.strip() for p in header.split("#")]
            gene_id = parts[0].split()[0]
            orf_start = int(parts[1]) if len(parts) > 1 and parts[1].strip().isdigit() else 0
            orf_end = int(parts[2]) if len(parts) > 2 and parts[2].strip().isdigit() else 0
            strand_raw = parts[3].strip() if len(parts) > 3 else ""
            strand = "+" if strand_raw in {"1", "+"} else "-" if strand_raw in {"-1", "-"} else strand_raw

            seqhdr_match = _SEQHDR_RE.search(header)
            if seqhdr_match is not None:
                contig_name = seqhdr_match.group(1).strip().split()[0]
            else:
                contig_name = gene_id.rsplit("_", 1)[0] if "_" in gene_id else gene_id
            meta[gene_id] = (contig_name, orf_start, orf_end, strand)
    return meta
