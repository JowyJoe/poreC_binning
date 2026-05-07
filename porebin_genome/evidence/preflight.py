"""Fast input and dependency checks for evidence construction."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from porebin_genome.io.coverage import read_coverage_tsv, validate_coverage_tsv
from porebin_genome.io.fasta import iter_fasta_records


class EvidencePreflightError(RuntimeError):
    """Raised when evidence construction cannot start safely."""


@dataclass(frozen=True)
class BamHeaderSummary:
    """Small BAM header summary used by preflight checks."""

    sort_order: Optional[str]
    references: frozenset[str]


@dataclass(frozen=True)
class EvidencePreflightReport:
    """Collected evidence preflight checks."""

    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    checks: dict[str, object]

    @property
    def passed(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            message = "Evidence preflight failed:\n" + "\n".join(
                f"- {error}" for error in self.errors
            )
            raise EvidencePreflightError(message)


def run_evidence_preflight(
    *,
    bam: Path,
    contigs_fasta: Path,
    coverage_method: str,
    coverage_bam: Optional[Path],
    coverage_tsv: Optional[Path],
    parquet_batch_size: int,
    coverage_threads: int,
) -> EvidencePreflightReport:
    """Run cheap checks before streaming BAM evidence or invoking CoverM."""
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, object] = {}

    coverage_method = str(coverage_method).strip().lower()
    if int(parquet_batch_size) <= 0:
        errors.append("--parquet-batch-size must be > 0.")
    if int(coverage_threads) < 1:
        errors.append("--coverage-threads must be >= 1.")

    contig_names = _read_fasta_names(contigs_fasta, errors=errors)
    checks["n_fasta_contigs"] = len(contig_names)

    contact_header = _read_bam_header_summary(bam, errors=errors, label="--bam")
    if contact_header is not None:
        checks["contact_bam_sort_order"] = contact_header.sort_order
        checks["n_contact_bam_references"] = len(contact_header.references)
        _check_contact_bam_sort_order(contact_header.sort_order, errors=errors, warnings=warnings)
        _check_reference_overlap(
            label="--bam",
            bam_references=contact_header.references,
            contig_names=contig_names,
            errors=errors,
            warnings=warnings,
        )

    if coverage_tsv is not None:
        _check_coverage_tsv(
            coverage_tsv=coverage_tsv,
            contig_names=contig_names,
            errors=errors,
            warnings=warnings,
            checks=checks,
        )
    elif coverage_method == "internal":
        warnings.append(
            "--coverage-method internal uses the legacy MAPQ-weighted depth fallback; "
            "CoverM mean depth is recommended for production runs."
        )
        checks["coverage_source"] = "internal"
    elif coverage_method == "coverm":
        _check_coverm(
            coverage_bam=coverage_bam,
            contig_names=contig_names,
            errors=errors,
            warnings=warnings,
            checks=checks,
        )
    else:
        errors.append("coverage_method must be one of: coverm, internal.")

    return EvidencePreflightReport(
        errors=tuple(errors),
        warnings=tuple(warnings),
        checks=checks,
    )


def _read_fasta_names(contigs_fasta: Path, *, errors: list[str]) -> frozenset[str]:
    contigs_fasta = contigs_fasta.resolve()
    if not contigs_fasta.exists():
        errors.append(f"Contigs FASTA not found: {contigs_fasta}")
        return frozenset()
    try:
        names = [name for name, _header, _seq in iter_fasta_records(contigs_fasta)]
    except Exception as exc:
        errors.append(f"Could not read contigs FASTA {contigs_fasta}: {exc}")
        return frozenset()
    if not names:
        errors.append(f"Contigs FASTA is empty: {contigs_fasta}")
    return frozenset(names)


def _read_bam_header_summary(
    path: Path,
    *,
    errors: list[str],
    label: str,
) -> Optional[BamHeaderSummary]:
    path = path.resolve()
    if not path.exists():
        errors.append(f"{label} not found: {path}")
        return None
    try:
        import pysam
    except Exception as exc:  # pragma: no cover
        errors.append("BAM preflight requires pysam. Install with: conda install -c bioconda pysam")
        return None
    try:
        with pysam.AlignmentFile(str(path), "rb") as bam_fh:
            header_dict = bam_fh.header.to_dict()
            sort_order_raw = (header_dict.get("HD") or {}).get("SO")
            sort_order = str(sort_order_raw).lower() if sort_order_raw is not None else None
            references = frozenset(str(name) for name in bam_fh.references)
    except Exception as exc:
        errors.append(f"Could not read BAM header for {label} {path}: {exc}")
        return None
    return BamHeaderSummary(sort_order=sort_order, references=references)


def _check_contact_bam_sort_order(
    sort_order: Optional[str],
    *,
    errors: list[str],
    warnings: list[str],
) -> None:
    if sort_order == "queryname":
        return
    if sort_order == "coordinate":
        errors.append(
            "--bam appears to be coordinate-sorted, but Pore-C contact evidence requires "
            "a queryname-sorted BAM. Use: samtools sort -n -o reads.namesorted.bam reads.bam"
        )
        return
    warnings.append(
        "--bam sort order is not declared as queryname in the BAM header; "
        "evidence construction expects records grouped by read name."
    )


def _check_coverm(
    *,
    coverage_bam: Optional[Path],
    contig_names: frozenset[str],
    errors: list[str],
    warnings: list[str],
    checks: dict[str, object],
) -> None:
    checks["coverage_source"] = "coverm"
    if shutil.which("coverm") is None:
        errors.append(
            "CoverM is required for --coverage-method coverm. Install with: "
            "conda install -c conda-forge -c bioconda coverm"
        )
    if coverage_bam is None:
        errors.append(
            "--coverage-bam is required for CoverM coverage. Provide a reference-sorted "
            "contig-aligned BAM, or use --coverage-tsv to provide precomputed mean depth."
        )
        return

    coverage_header = _read_bam_header_summary(coverage_bam, errors=errors, label="--coverage-bam")
    if coverage_header is None:
        return
    checks["coverage_bam_sort_order"] = coverage_header.sort_order
    checks["n_coverage_bam_references"] = len(coverage_header.references)

    if coverage_header.sort_order == "queryname":
        errors.append(
            "--coverage-bam appears to be queryname-sorted. CoverM expects a reference-sorted "
            "BAM. Use: samtools sort -o reads.coordsorted.bam reads.bam"
        )
    elif coverage_header.sort_order not in {"coordinate", "unknown"}:
        warnings.append(
            "--coverage-bam sort order is not declared as coordinate/reference-sorted; "
            "CoverM may fail if the BAM is not reference-sorted."
        )

    if not _bam_index_exists(coverage_bam):
        errors.append(
            f"--coverage-bam has no .bai or .csi index: {coverage_bam.resolve()}. "
            "Create one with: samtools index reads.coordsorted.bam"
        )

    _check_reference_overlap(
        label="--coverage-bam",
        bam_references=coverage_header.references,
        contig_names=contig_names,
        errors=errors,
        warnings=warnings,
    )


def _check_coverage_tsv(
    *,
    coverage_tsv: Path,
    contig_names: frozenset[str],
    errors: list[str],
    warnings: list[str],
    checks: dict[str, object],
) -> None:
    coverage_tsv = coverage_tsv.resolve()
    checks["coverage_source"] = "tsv"
    if not coverage_tsv.exists():
        errors.append(f"Coverage TSV not found: {coverage_tsv}")
        return
    try:
        validate_coverage_tsv(coverage_tsv)
        coverage_by_contig = read_coverage_tsv(coverage_tsv)
    except Exception as exc:
        errors.append(f"Invalid coverage TSV {coverage_tsv}: {exc}")
        return
    coverage_names = frozenset(coverage_by_contig)
    checks["n_coverage_tsv_contigs"] = len(coverage_names)
    if contig_names and not (contig_names & coverage_names):
        errors.append(
            "--coverage-tsv has no contig names in common with --contigs. "
            "The coverage table must use FASTA contig IDs."
        )
    elif contig_names:
        missing = contig_names - coverage_names
        if missing:
            warnings.append(
                f"--coverage-tsv is missing coverage for {len(missing)} FASTA contigs; "
                "missing contigs will have weaker coverage constraints."
            )


def _check_reference_overlap(
    *,
    label: str,
    bam_references: frozenset[str],
    contig_names: frozenset[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    if not contig_names or not bam_references:
        return
    overlap = contig_names & bam_references
    if not overlap:
        errors.append(
            f"{label} reference names have no overlap with --contigs FASTA IDs. "
            "Make sure the BAM was aligned against the same contigs FASTA."
        )
        return
    missing = contig_names - bam_references
    if missing:
        warnings.append(
            f"{label} header is missing {len(missing)} FASTA contigs; "
            "those contigs may receive no evidence from this BAM."
        )


def _bam_index_exists(path: Path) -> bool:
    path = path.resolve()
    candidates = [
        path.with_suffix(path.suffix + ".bai"),
        path.with_suffix(".bai"),
        path.with_suffix(path.suffix + ".csi"),
        path.with_suffix(".csi"),
    ]
    return any(candidate.exists() for candidate in candidates)
