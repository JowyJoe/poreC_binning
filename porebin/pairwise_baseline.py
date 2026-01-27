from __future__ import annotations

import gzip
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Iterator, Optional

from porebin.utils import dedupe_preserve_order, ensure_dir


class PairwiseBaselineError(RuntimeError):
    pass


@dataclass
class PairwiseBuildStats:
    segments_total: int = 0
    segments_kept: int = 0
    reads_total: int = 0
    reads_skipped_k_lt_2: int = 0
    raw_pairs_written: int = 0
    unique_edges: int = 0
    input_sorted_by_readid: bool = True
    input_sorted_by_readid_verified: bool = True


def parse_contacts_stream(contacts_path: Path) -> Iterator[tuple[str, str, str]]:
    """
    Stream a PPL .contacts TSV (optionally .gz).

    Expected columns (1-based):
      1 contig/chr, 4 readID, 11 status
    Yields (contig, read_id, status).
    """
    opener = gzip.open if contacts_path.suffix == ".gz" else open
    with opener(contacts_path, "rt", encoding="utf-8", errors="replace", newline="") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            row = line.rstrip("\n").split("\t")
            if line_no == 1 and _looks_like_header_row(row):
                continue
            if len(row) < 11:
                raise PairwiseBaselineError(
                    f"Unexpected column count at {contacts_path}:{line_no}: got {len(row)}, expected >=11."
                )
            contig = row[0].strip()
            read_id = row[3].strip()
            status = row[10].strip()
            if not contig or not read_id:
                raise PairwiseBaselineError(
                    f"Empty contig/readID at {contacts_path}:{line_no}: contig={contig!r} readID={read_id!r}"
                )
            yield contig, read_id, status


def group_by_readid(
    stream: Iterable[tuple[str, str, str]],
    *,
    assume_sorted: bool,
) -> Iterator[tuple[str, list[str]]]:
    """
    Group a stream that is sorted/grouped by readID.

    Input yields (contig, read_id, status). Output yields (read_id, contigs_seen_in_order).
    """
    current_read: Optional[str] = None
    contigs: list[str] = []
    prev_read: Optional[str] = None

    for contig, read_id, _status in stream:
        if not assume_sorted and prev_read is not None and read_id < prev_read:
            raise PairwiseBaselineError(
                "Input .contacts does not appear sorted by readID (column 4). "
                "Please sort first, e.g.: sort -k4,4 contacts > contacts.sorted"
            )
        prev_read = read_id

        if current_read is None:
            current_read = read_id
        elif read_id != current_read:
            yield current_read, contigs
            current_read = read_id
            contigs = []
        contigs.append(contig)

    if current_read is not None:
        yield current_read, contigs


def expand_to_pairs(contigs: list[str]) -> Iterator[tuple[str, str, float]]:
    """
    Clique expansion with fair per-read normalization:
      w_pair = 2/(k*(k-1)) == 1/C(k,2)
    """
    unique = dedupe_preserve_order([c for c in contigs if c])
    k = len(unique)
    if k < 2:
        return iter(())
    w = 2.0 / (k * (k - 1))

    def gen() -> Iterator[tuple[str, str, float]]:
        for u, v in combinations(unique, 2):
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            yield a, b, w

    return gen()


def emit_raw_pairs_from_contacts(
    contacts_path: Path,
    *,
    tmp_dir: Path,
    chunk_lines: int = 5_000_000,
    gzip_raw: bool = False,
    assume_sorted: bool = False,
    require_passed: bool = True,
    logger: Optional[logging.Logger] = None,
) -> tuple[list[Path], PairwiseBuildStats]:
    """
    Stage 1: stream + per-read clique expansion, writing raw (contigA, contigB, w) lines.

    Writes multiple parts to tmp_dir as raw.part000.tsv[.gz], raw.part001.tsv[.gz], ...
    """
    logger = logger or logging.getLogger("porebin")
    ensure_dir(tmp_dir)

    for old in tmp_dir.glob("raw.part*.tsv*"):
        old.unlink(missing_ok=True)

    stats = PairwiseBuildStats()
    stats.input_sorted_by_readid_verified = not assume_sorted
    stats.input_sorted_by_readid = True

    opener = gzip.open if gzip_raw else open
    part_paths: list[Path] = []
    part_idx = 0
    lines_in_part = 0
    out_fh = None

    def open_part() -> None:
        nonlocal out_fh, part_idx, lines_in_part
        if out_fh is not None:
            out_fh.close()
        suffix = ".tsv.gz" if gzip_raw else ".tsv"
        path = tmp_dir / f"raw.part{part_idx:03d}{suffix}"
        part_paths.append(path)
        out_fh = opener(path, "wt", encoding="utf-8", newline="")
        part_idx += 1
        lines_in_part = 0

    open_part()

    current_read: Optional[str] = None
    current_contigs: list[str] = []
    prev_read: Optional[str] = None

    def write_raw(a: str, b: str, w: float) -> None:
        nonlocal lines_in_part, out_fh
        assert out_fh is not None
        out_fh.write(f"{a}\t{b}\t{w:.10g}\n")
        stats.raw_pairs_written += 1
        lines_in_part += 1
        if chunk_lines > 0 and lines_in_part >= chunk_lines:
            open_part()

    def flush_current() -> None:
        unique = dedupe_preserve_order([c for c in current_contigs if c])
        k = len(unique)
        if k < 2:
            stats.reads_skipped_k_lt_2 += 1
            return
        w = 2.0 / (k * (k - 1))
        for u, v in combinations(unique, 2):
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            write_raw(a, b, w)

    for contig, read_id, status in parse_contacts_stream(contacts_path):
        stats.segments_total += 1
        if require_passed and str(status).strip().lower() != "passed":
            continue
        stats.segments_kept += 1

        if not assume_sorted and prev_read is not None and read_id < prev_read:
            stats.input_sorted_by_readid = False
            raise PairwiseBaselineError(
                "Input .contacts does not appear sorted by readID (column 4). "
                "Please sort first, e.g.: sort -k4,4 contacts > contacts.sorted"
            )
        prev_read = read_id

        if current_read is None:
            current_read = read_id
            stats.reads_total += 1
        elif read_id != current_read:
            flush_current()
            current_read = read_id
            current_contigs = []
            stats.reads_total += 1

        current_contigs.append(contig)

    if current_read is not None:
        flush_current()

    if out_fh is not None:
        out_fh.close()

    if stats.reads_total == 0:
        raise PairwiseBaselineError("No reads found in contacts input.")
    if stats.raw_pairs_written == 0:
        raise PairwiseBaselineError("No raw pairs written (all reads had k<2 after filtering).")

    logger.info(
        "Pairwise stage1: segments=%s kept=%s reads=%s skipped_k<2=%s raw_pairs=%s parts=%s",
        f"{stats.segments_total:,}",
        f"{stats.segments_kept:,}",
        f"{stats.reads_total:,}",
        f"{stats.reads_skipped_k_lt_2:,}",
        f"{stats.raw_pairs_written:,}",
        len(part_paths),
    )
    return part_paths, stats


def external_sort_and_reduce(
    raw_parts: list[Path],
    *,
    out_edges_path: Path,
    tmp_dir: Path,
    sort_threads: int = 1,
    memory: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> int:
    """
    Stage 2: external sort (system `sort`) + streaming reduce (sum weights).

    Output: gzip TSV with columns: contigA, contigB, weight (no header).
    Returns the number of unique edges written.
    """
    logger = logger or logging.getLogger("porebin")
    ensure_dir(out_edges_path.parent)
    ensure_dir(tmp_dir)

    if out_edges_path.exists():
        out_edges_path.unlink()

    if not raw_parts:
        raise PairwiseBaselineError("No raw parts provided for sorting/reducing.")

    sort_exe = shutil.which("sort")
    if sort_exe is None or not _is_gnu_sort(sort_exe):
        logger.warning(
            "GNU 'sort' not available; falling back to in-memory sort (tests/small inputs only). "
            "On Linux, ensure coreutils 'sort' is on PATH."
        )
        acc: dict[tuple[str, str], float] = {}
        for part in raw_parts:
            opener = gzip.open if part.suffix == ".gz" else open
            with opener(part, "rt", encoding="utf-8", newline="") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    a, b, w_s = line.rstrip("\n").split("\t")
                    acc[(a, b)] = acc.get((a, b), 0.0) + float(w_s)
        with gzip.open(out_edges_path, "wt", encoding="utf-8", newline="") as out_fh:
            for (a, b) in sorted(acc.keys()):
                out_fh.write(f"{a}\t{b}\t{acc[(a, b)]:.10g}\n")
        return len(acc)

    cmd = [sort_exe, "-k1,1", "-k2,2"]
    if sort_threads and sort_threads > 1:
        cmd.append(f"--parallel={int(sort_threads)}")
    if memory:
        cmd.extend(["-S", str(memory)])
    cmd.extend(["-T", str(tmp_dir)])
    cmd.extend([str(p) for p in raw_parts])

    env = os.environ.copy()
    env.setdefault("LC_ALL", "C")

    logger.info("Pairwise stage2: sorting %s raw part(s) via system sort", len(raw_parts))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert proc.stdout is not None
    assert proc.stderr is not None

    unique_edges = 0
    last_a: Optional[str] = None
    last_b: Optional[str] = None
    acc_w = 0.0

    with gzip.open(out_edges_path, "wt", encoding="utf-8", newline="") as out_fh:
        for line in proc.stdout:
            if not line.strip():
                continue
            try:
                a, b, w_s = line.rstrip("\n").split("\t")
                w = float(w_s)
            except Exception as exc:
                raise PairwiseBaselineError(f"Invalid raw pair line from sort: {line!r}") from exc

            if last_a is None:
                last_a, last_b, acc_w = a, b, w
                continue

            if a == last_a and b == last_b:
                acc_w += w
                continue

            out_fh.write(f"{last_a}\t{last_b}\t{acc_w:.10g}\n")
            unique_edges += 1
            last_a, last_b, acc_w = a, b, w

        if last_a is not None:
            out_fh.write(f"{last_a}\t{last_b}\t{acc_w:.10g}\n")
            unique_edges += 1

    stderr = proc.stderr.read()
    rc = proc.wait()
    if rc != 0:
        raise PairwiseBaselineError(f"External sort failed (exit {rc}). stderr:\n{stderr}")

    logger.info("Pairwise stage2: wrote %s unique edges to %s", f"{unique_edges:,}", out_edges_path)
    return unique_edges


def build_pairwise_edges(
    *,
    contacts_path: Path,
    out_edges_path: Path,
    tmp_dir: Path,
    assume_sorted: bool = False,
    sort_threads: int = 1,
    memory: Optional[str] = None,
    chunk_lines: int = 5_000_000,
    logger: Optional[logging.Logger] = None,
) -> PairwiseBuildStats:
    """
    Build a contig-contig edge list baseline via clique expansion + external aggregation.
    """
    logger = logger or logging.getLogger("porebin")
    if not contacts_path.exists():
        raise FileNotFoundError(f"Contacts not found: {contacts_path}")

    ensure_dir(tmp_dir)
    raw_parts, stats = emit_raw_pairs_from_contacts(
        contacts_path,
        tmp_dir=tmp_dir,
        chunk_lines=chunk_lines,
        gzip_raw=False,
        assume_sorted=assume_sorted,
        require_passed=True,
        logger=logger,
    )
    stats.unique_edges = external_sort_and_reduce(
        raw_parts,
        out_edges_path=out_edges_path,
        tmp_dir=tmp_dir,
        sort_threads=sort_threads,
        memory=memory,
        logger=logger,
    )
    return stats


def _looks_like_header_row(row: list[str]) -> bool:
    if len(row) < 11:
        return False
    c0 = row[0].strip().lower()
    c3 = row[3].strip().lower()
    c10 = row[10].strip().lower()
    return (c0 in {"chr", "chrom", "contig", "reference"} or c3 in {"readid", "read_id"}) and (
        c10 in {"status", "filter"}
    )


def _is_gnu_sort(sort_exe: str) -> bool:
    try:
        proc = subprocess.run(
            [sort_exe, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception:
        return False
    if proc.returncode != 0:
        return False
    out = (proc.stdout or "") + (proc.stderr or "")
    return "GNU coreutils" in out or "sort (GNU coreutils)" in out
