from __future__ import annotations

import json
import logging
import os
import platform
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from rich.console import Console
from rich.logging import RichHandler

from porebin import __version__

console = Console(stderr=True)


def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
    )
    return logging.getLogger("porebin")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@contextmanager
def record_run(
    out_dir: Path,
    *,
    command: str,
    params: dict[str, Any],
    seed: Optional[int] = None,
) -> Iterator[dict[str, Any]]:
    run_path = out_dir / "run.json"
    started_at = utc_now_iso()
    record: dict[str, Any] = {
        "porebin_version": __version__,
        "command": command,
        "params": params,
        "seed": seed,
        "started_at": started_at,
        "ended_at": None,
        "status": "running",
        "python": sys.version.split()[0],
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "cwd": os.getcwd(),
    }
    write_json(run_path, record)

    try:
        yield record
    except Exception as exc:
        record["status"] = "error"
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        record["ended_at"] = utc_now_iso()
        if record["status"] == "running":
            record["status"] = "ok"
        write_json(run_path, record)


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def iter_fasta_records(path: Path):
    header: Optional[str] = None
    name: Optional[str] = None
    seq_parts: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip().lstrip("\ufeff")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None and name is not None:
                    yield name, header, "".join(seq_parts)
                header = line[1:].strip()
                name = header.split()[0] if header else ""
                seq_parts = []
            else:
                if header is None:
                    raise ValueError(f"Invalid FASTA (sequence before header) at {path}:{line_no}")
                seq_parts.append(line)
        if header is not None and name is not None:
            yield name, header, "".join(seq_parts)


def iter_fasta_names(path: Path):
    for name, _, _ in iter_fasta_records(path):
        yield name
