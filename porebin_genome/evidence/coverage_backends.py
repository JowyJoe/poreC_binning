"""Coverage/depth backends for evidence construction."""

from __future__ import annotations

import csv
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from porebin_genome.io.coverage import read_coverage_tsv, validate_coverage_tsv
from porebin_genome.io.fasta import iter_fasta_records
from porebin_genome.io.runtime import ensure_dir


class CoverageBackendError(RuntimeError):
    """Raised when coverage/depth construction fails."""


@dataclass(frozen=True)
class CoverageBuildResult:
    """A normalized coverage table produced or adopted by an evidence backend."""

    coverage_tsv: Path
    source: str
    raw_output_tsv: Optional[Path]
    n_contigs: int
    n_missing_contigs: int


def adopt_coverage_tsv(*, source_tsv: Path, out_tsv: Path) -> CoverageBuildResult:
    """Validate and copy a user-provided coverage table into the evidence layout."""
    source_tsv = source_tsv.resolve()
    out_tsv = out_tsv.resolve()
    if not source_tsv.exists():
        raise FileNotFoundError(f"Coverage TSV not found: {source_tsv}")
    validate_coverage_tsv(source_tsv)
    coverage_by_contig = read_coverage_tsv(source_tsv)

    ensure_dir(out_tsv.parent)
    if source_tsv != out_tsv:
        shutil.copyfile(source_tsv, out_tsv)
    return CoverageBuildResult(
        coverage_tsv=out_tsv,
        source="tsv",
        raw_output_tsv=source_tsv,
        n_contigs=len(coverage_by_contig),
        n_missing_contigs=0,
    )


def compute_coverm_coverage_tsv(
    *,
    coverage_bam: Path,
    contigs_fasta: Path,
    out_tsv: Path,
    raw_output_tsv: Path,
    threads: int = 1,
) -> CoverageBuildResult:
    """Run CoverM contig mean-depth and normalize it to porebin's coverage TSV."""
    coverage_bam = coverage_bam.resolve()
    contigs_fasta = contigs_fasta.resolve()
    out_tsv = out_tsv.resolve()
    raw_output_tsv = raw_output_tsv.resolve()

    if not coverage_bam.exists():
        raise FileNotFoundError(f"Coverage BAM not found: {coverage_bam}")
    if not contigs_fasta.exists():
        raise FileNotFoundError(f"Contigs FASTA not found: {contigs_fasta}")

    coverm = shutil.which("coverm")
    if coverm is None:
        raise CoverageBackendError(
            "CoverM is required for --coverage-method coverm. Install it with:\n"
            "conda install -c conda-forge -c bioconda coverm"
        )

    ensure_dir(raw_output_tsv.parent)
    command = [
        coverm,
        "contig",
        "--bam-files",
        str(coverage_bam),
        "--methods",
        "mean",
        "--contig-end-exclusion",
        "0",
        "--min-covered-fraction",
        "0",
        "--output-format",
        "dense",
        "--output-file",
        str(raw_output_tsv),
        "--threads",
        str(max(1, int(threads))),
    ]
    _run_external_command(command, tool_name="CoverM")

    contig_names = _read_contig_names(contigs_fasta)
    coverage_by_contig = parse_coverm_mean_depth_tsv(raw_output_tsv)
    missing = [name for name in contig_names if name not in coverage_by_contig]
    if missing:
        preview = ", ".join(missing[:5])
        raise CoverageBackendError(
            "CoverM output did not contain all FASTA contigs. "
            "Make sure --coverage-bam was aligned to the same --contigs FASTA. "
            f"Missing {len(missing)} contigs; examples: {preview}"
        )

    _write_porebin_coverage_tsv(
        out_tsv=out_tsv,
        contig_names=contig_names,
        coverage_by_contig=coverage_by_contig,
    )
    return CoverageBuildResult(
        coverage_tsv=out_tsv,
        source="coverm",
        raw_output_tsv=raw_output_tsv,
        n_contigs=len(contig_names),
        n_missing_contigs=0,
    )


def parse_coverm_mean_depth_tsv(path: Path) -> dict[str, float]:
    """Parse CoverM dense contig output produced with `--methods mean`."""
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"CoverM output TSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as exc:
            raise CoverageBackendError(f"CoverM output is empty: {path}") from exc
        if not header:
            raise CoverageBackendError(f"CoverM output has an empty header: {path}")

        contig_idx = _select_contig_column(header)
        mean_idx = _select_mean_depth_column(header, contig_idx=contig_idx)
        out: dict[str, float] = {}
        for row in reader:
            if not row or len(row) <= max(contig_idx, mean_idx):
                continue
            contig_name = str(row[contig_idx]).strip()
            if not contig_name:
                continue
            try:
                depth = float(row[mean_idx])
            except ValueError:
                continue
            out[contig_name] = depth

    if not out:
        raise CoverageBackendError(f"No contig mean-depth values parsed from CoverM output: {path}")
    return out


def _select_contig_column(header: list[str]) -> int:
    for idx, name in enumerate(header):
        lowered = str(name).strip().lower()
        if lowered in {"contig", "contig_name", "contigname", "#rname", "rname"}:
            return idx
    return 0


def _select_mean_depth_column(header: list[str], *, contig_idx: int) -> int:
    candidates: list[int] = []
    for idx, name in enumerate(header):
        if idx == contig_idx:
            continue
        lowered = str(name).strip().lower()
        if "mean" in lowered and "trimmed" not in lowered:
            candidates.append(idx)
    if len(candidates) == 1:
        return candidates[0]
    if len(header) == 2:
        return 1 if contig_idx == 0 else 0
    raise CoverageBackendError(
        "Could not identify the CoverM mean-depth column. "
        "Run CoverM with `coverm contig --methods mean --output-format dense`."
    )


def _read_contig_names(contigs_fasta: Path) -> list[str]:
    names = [name for name, _header, _seq in iter_fasta_records(contigs_fasta)]
    if not names:
        raise CoverageBackendError(f"No contigs found in FASTA: {contigs_fasta}")
    return names


def _write_porebin_coverage_tsv(
    *,
    out_tsv: Path,
    contig_names: list[str],
    coverage_by_contig: dict[str, float],
) -> None:
    ensure_dir(out_tsv.parent)
    with out_tsv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(["contig_name", "coverage"])
        for contig_name in contig_names:
            writer.writerow([contig_name, f"{float(coverage_by_contig[contig_name]):.12g}"])


def _run_external_command(command: list[str], *, tool_name: str) -> None:
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:  # pragma: no cover - protected by shutil.which
        raise CoverageBackendError(f"{tool_name} executable was not found.") from exc
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip()
        raise CoverageBackendError(
            f"{tool_name} failed while computing contig mean depth: {message}"
        )
