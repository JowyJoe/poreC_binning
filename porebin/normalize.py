from __future__ import annotations

import csv
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from porebin import __version__
from porebin.utils import dedupe_preserve_order, ensure_dir, write_json


class NormalizeError(RuntimeError):
    pass


_STATUS_TOKENS = {"passed", "failed", "fail", "filtered", "rejected"}


@dataclass(frozen=True)
class PplColumns:
    read_id: int
    chrom: int
    status: int
    score: Optional[int]


def normalize_contacts(
    *,
    ppl_contacts: Path,
    out_dir: Path,
    include_tags: Iterable[str] = ("mapq", "AS", "n_segments"),
    assume_no_header: bool = False,
    threads: int = 1,
    logger: Optional[logging.Logger] = None,
) -> Path:
    logger = logger or logging.getLogger("porebin")
    out_contacts_dir = out_dir / "contacts"
    ensure_dir(out_contacts_dir)

    out_parquet = out_contacts_dir / "contacts.parquet"
    out_qc = out_contacts_dir / "qc.json"

    if not ppl_contacts.exists():
        raise FileNotFoundError(f"PPL contacts not found: {ppl_contacts}")

    include = {t.strip().lower() for t in include_tags if t and t.strip()}
    if "all" in include:
        include = {"mapq", "as", "n_segments"}

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover
        raise NormalizeError(
            "normalize requires 'pyarrow' for Parquet output. Install it, e.g. pip install pyarrow."
        ) from exc

    if out_parquet.exists():
        out_parquet.unlink()

    logger.info(f"Normalizing PPL contacts: {ppl_contacts}")
    logger.debug(f"Threads hint: {threads} (currently not used in v0.1)")

    contact_id = 0
    segment_rows_total = 0
    segment_rows_kept = 0
    read_groups_total = 0
    contacts_written = 0
    contacts_skipped_k_lt_2 = 0
    status_counter: Counter[str] = Counter()
    k_counter: Counter[int] = Counter()
    degree_counter: Counter[str] = Counter()

    current_read: Optional[str] = None
    current_contigs: list[str] = []
    current_contig_set: set[str] = set()
    current_mapq_vals: list[float] = []
    current_as_vals: list[float] = []
    current_segments = 0

    schema = _make_contacts_schema(pa, include)
    writer = _open_parquet_writer(pq, out_parquet, schema=schema, logger=logger)
    batch_size_contacts = 10_000
    buffer: dict[str, list[Any]] = {name: [] for name in schema.names}

    def flush_buffer() -> None:
        if not buffer[schema.names[0]]:
            return
        writer.write_table(pa.table(buffer, schema=schema))
        for name in schema.names:
            buffer[name].clear()

    def reset_current() -> None:
        nonlocal current_read, current_contigs, current_contig_set
        nonlocal current_mapq_vals, current_as_vals, current_segments
        current_read = None
        current_contigs = []
        current_contig_set = set()
        current_mapq_vals = []
        current_as_vals = []
        current_segments = 0

    def flush_current() -> None:
        nonlocal contact_id, contacts_written, contacts_skipped_k_lt_2
        if current_read is None:
            return
        contigs = dedupe_preserve_order(current_contigs)
        k = len(contigs)
        if k < 2:
            contacts_skipped_k_lt_2 += 1
            reset_current()
            return

        buffer["contact_id"].append(contact_id)
        buffer["contigs"].append(contigs)
        buffer["k"].append(k)
        buffer["weight"].append(1.0)
        buffer["support_count"].append(1)
        if "n_segments" in include:
            buffer["n_segments"].append(current_segments)
        if "mapq" in include:
            if current_mapq_vals:
                buffer["mapq_min"].append(min(current_mapq_vals))
                buffer["mapq_mean"].append(sum(current_mapq_vals) / len(current_mapq_vals))
            else:
                buffer["mapq_min"].append(None)
                buffer["mapq_mean"].append(None)
        if "as" in include:
            buffer["as_sum"].append(sum(current_as_vals) if current_as_vals else None)
        if len(buffer["contact_id"]) >= batch_size_contacts:
            flush_buffer()

        k_counter[k] += 1
        for c in contigs:
            degree_counter[c] += 1

        contact_id += 1
        contacts_written += 1
        reset_current()

    with ppl_contacts.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")

        first_row: Optional[list[str]] = None
        first_row_line_no = 0
        for line_no, row in enumerate(reader, start=1):
            if not row or all(not c.strip() for c in row):
                continue
            first_row = row
            first_row_line_no = line_no
            break

        if first_row is None:
            raise NormalizeError(f"Empty contacts file: {ppl_contacts}")

        data_first_row: Optional[list[str]] = None
        if not assume_no_header and _looks_like_header(first_row):
            columns = _columns_from_header(first_row)
        else:
            data_first_row = first_row
            columns = _columns_from_example_row(first_row)

        def process_row(row: list[str], *, line_no: int) -> None:
            nonlocal segment_rows_total, segment_rows_kept, read_groups_total
            nonlocal current_read, current_contigs, current_contig_set
            nonlocal current_mapq_vals, current_as_vals, current_segments

            if len(row) not in (11, 12):
                raise NormalizeError(
                    f"Unexpected column count at {ppl_contacts}:{line_no}: got {len(row)}, expected 11 or 12."
                )

            segment_rows_total += 1

            try:
                read_id = row[columns.read_id].strip().lstrip("\ufeff")
                chrom = row[columns.chrom].strip().lstrip("\ufeff")
                status = row[columns.status].strip().lower()
                score = row[columns.score].strip() if columns.score is not None and columns.score < len(row) else ""
            except Exception as exc:
                raise NormalizeError(f"Failed to parse row at {ppl_contacts}:{line_no}: {row}") from exc

            if not read_id:
                raise NormalizeError(f"Empty readID at {ppl_contacts}:{line_no}")
            if not chrom:
                raise NormalizeError(f"Empty chr/contig at {ppl_contacts}:{line_no}")

            status_counter[status] += 1

            if current_read is None:
                current_read = read_id
                read_groups_total += 1
            elif read_id != current_read:
                flush_current()
                current_read = read_id
                read_groups_total += 1

            if status != "passed":
                return

            segment_rows_kept += 1
            current_segments += 1

            if chrom not in current_contig_set:
                current_contig_set.add(chrom)
                current_contigs.append(chrom)

            if score:
                tags = _parse_score_tags(score)
                if "mapq" in include and "mapq" in tags:
                    try:
                        current_mapq_vals.append(float(tags["mapq"]))
                    except ValueError:
                        pass
                if "as" in include and "as" in tags:
                    try:
                        current_as_vals.append(float(tags["as"]))
                    except ValueError:
                        pass

        if data_first_row is not None:
            process_row(data_first_row, line_no=first_row_line_no)

        for line_no, row in enumerate(reader, start=first_row_line_no + 1):
            if not row or all(not c.strip() for c in row):
                continue
            process_row(row, line_no=line_no)

        flush_current()

    flush_buffer()
    writer.close()

    qc = {
        "porebin_version": __version__,
        "input_ppl_contacts": str(ppl_contacts),
        "out_contacts_parquet": str(out_parquet),
        "segments_total": segment_rows_total,
        "segments_kept_status_passed": segment_rows_kept,
        "reads_total": read_groups_total,
        "contacts_written": contacts_written,
        "contacts_skipped_k_lt_2": contacts_skipped_k_lt_2,
        "k_distribution": {str(k): v for k, v in k_counter.items()},
        "status_distribution": dict(status_counter),
        "degree_top_hubs": [{"contig": c, "degree": d} for c, d in degree_counter.most_common(20)],
        "include_tags": sorted(include),
    }
    write_json(out_qc, qc)

    if contacts_written == 0:
        raise NormalizeError(
            "No contacts written. Check that your input has status='passed' segments and each read spans >=2 contigs."
        )

    logger.info(f"Wrote contacts: {out_parquet} (n={contacts_written})")
    logger.info(f"Wrote QC: {out_qc}")
    return out_parquet


def _make_contacts_schema(pa, include: set[str]):
    fields = [
        pa.field("contact_id", pa.int64(), nullable=False),
        pa.field("contigs", pa.list_(pa.string()), nullable=False),
        pa.field("k", pa.int32(), nullable=False),
        pa.field("weight", pa.float64(), nullable=False),
        pa.field("support_count", pa.int32(), nullable=False),
    ]
    if "n_segments" in include:
        fields.append(pa.field("n_segments", pa.int32(), nullable=True))
    if "mapq" in include:
        fields.append(pa.field("mapq_min", pa.float64(), nullable=True))
        fields.append(pa.field("mapq_mean", pa.float64(), nullable=True))
    if "as" in include:
        fields.append(pa.field("as_sum", pa.float64(), nullable=True))
    return pa.schema(fields)


def _open_parquet_writer(pq, path: Path, *, schema, logger: logging.Logger):
    try:
        return pq.ParquetWriter(path, schema=schema, compression="zstd")
    except Exception:
        logger.warning("Parquet compression 'zstd' unavailable; falling back to 'snappy'.")
    try:
        return pq.ParquetWriter(path, schema=schema, compression="snappy")
    except Exception:
        logger.warning("Parquet compression 'snappy' unavailable; falling back to uncompressed.")
    return pq.ParquetWriter(path, schema=schema, compression=None)


def _looks_like_header(row: list[str]) -> bool:
    tokens = {_norm_header_token(c) for c in row if c and c.strip()}
    return bool(tokens & {"readid", "read_id", "chr", "chrom", "contig", "status"})


def _columns_from_header(header: list[str]) -> PplColumns:
    norm = [_norm_header_token(c) for c in header]
    read_id = _find_first(norm, {"readid", "read_id", "read", "qname", "readname"})
    chrom = _find_first(norm, {"chr", "chrom", "contig", "ref", "reference", "rname"})
    status = _find_first(norm, {"status"})
    score = _find_first(norm, {"score", "tags", "tag"})
    if read_id is None or chrom is None or status is None:
        raise NormalizeError(
            f"Header missing required columns. Need readID/chr/status. Found: {header}"
        )
    return PplColumns(read_id=read_id, chrom=chrom, status=status, score=score)


def _columns_from_example_row(row: list[str]) -> PplColumns:
    if len(row) not in (11, 12):
        raise NormalizeError(
            f"Unexpected column count in first row: got {len(row)}, expected 11 or 12. "
            "If your file has a header, omit --assume-no-header."
        )

    status_idx = None
    last = row[-1].strip().lower()
    if last in _STATUS_TOKENS:
        status_idx = len(row) - 1
    else:
        for i, v in enumerate(row):
            if v.strip().lower() in _STATUS_TOKENS:
                status_idx = i
    if status_idx is None:
        raise NormalizeError(
            "Could not locate a 'status' column in headerless input. "
            "Provide a header or verify that one column contains tokens like 'passed'."
        )

    score_idx: Optional[int] = None
    if len(row) >= 2 and _looks_like_score(row[-2]):
        score_idx = len(row) - 2
    else:
        for i, v in enumerate(row):
            if _looks_like_score(v):
                score_idx = i
                break

    read_id_idx = 0

    chrom_idx = None
    excluded = {read_id_idx, status_idx}
    if score_idx is not None:
        excluded.add(score_idx)
    for i, v in enumerate(row):
        if i in excluded:
            continue
        if _looks_like_contig(v):
            chrom_idx = i
            break
    if chrom_idx is None:
        chrom_idx = 1

    return PplColumns(read_id=read_id_idx, chrom=chrom_idx, status=status_idx, score=score_idx)


def _find_first(names: list[str], candidates: set[str]) -> Optional[int]:
    for i, n in enumerate(names):
        if n in candidates:
            return i
    return None


def _norm_header_token(value: str) -> str:
    # Handle UTF-8 BOM that is common in Windows-written TSV headers.
    return value.strip().lstrip("\ufeff").lower()


_SCORE_SPLIT_RE = re.compile(r"[;,\s|]+")


def _parse_score_tags(score: str) -> dict[str, str]:
    score = score.strip()
    if not score or score == ".":
        return {}
    tags: dict[str, str] = {}
    for part in _SCORE_SPLIT_RE.split(score):
        if not part:
            continue
        if ":" in part:
            k, v = part.split(":", 1)
        elif "=" in part:
            k, v = part.split("=", 1)
        else:
            continue
        k = k.strip().lower()
        v = v.strip()
        if not k or not v:
            continue
        tags[k] = v
    return tags


def _looks_like_score(value: str) -> bool:
    v = value.strip()
    if not v:
        return False
    low = v.lower()
    return ("mapq" in low) or ("as:" in low) or ("as=" in low)


def _looks_like_contig(value: str) -> bool:
    v = value.strip()
    if not v:
        return False
    low = v.lower()
    if low in _STATUS_TOKENS:
        return False
    if v in {"+", "-"}:
        return False
    if _looks_like_score(v):
        return False
    return not _looks_numeric(v)


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False
