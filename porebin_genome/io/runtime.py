"""Runtime helpers shared by the new genome-centric CLI and orchestrators."""

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

from porebin_genome import __version__


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure a plain logging setup for the new CLI surface."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")
    return logging.getLogger("porebin_genome")


def ensure_dir(path: Path) -> None:
    """Create a directory path if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON file, creating parent directories as needed."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


@contextmanager
def record_run(
    out_dir: Path,
    *,
    command: str,
    params: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """Write a minimal `run.json` envelope for CLI commands."""
    run_path = out_dir / "run.json"
    record: dict[str, Any] = {
        "tool": "porebin_genome",
        "version": __version__,
        "command": command,
        "params": params,
        "started_at": utc_now_iso(),
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
