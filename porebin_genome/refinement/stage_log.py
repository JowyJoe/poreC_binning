"""Append-only stage logging for the replacement refinement engine."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from porebin_genome.io.runtime import ensure_dir, utc_now_iso


class RefinementStageLog:
    """Write small, immediately visible JSONL progress records."""

    def __init__(self, path: Path, *, logger: object | None = None) -> None:
        self.path = path.resolve()
        ensure_dir(self.path.parent)
        self._stream = self.path.open("w", encoding="utf-8", newline="\n")
        self._started = time.monotonic()
        self._logger = logger or logging.getLogger("porebin_genome")
        self.stage_durations: dict[str, float] = {}

    def close(self) -> None:
        if not self._stream.closed:
            self._stream.flush()
            self._stream.close()

    def emit(
        self,
        event: str,
        *,
        stage: str,
        message: str,
        **details: Any,
    ) -> None:
        record = {
            "time": utc_now_iso(),
            "event": str(event),
            "stage": str(stage),
            "elapsed_seconds": round(time.monotonic() - self._started, 6),
            "message": str(message),
        }
        record.update(details)
        self._stream.write(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        )
        self._stream.flush()
        log_fn = getattr(self._logger, "info", None)
        if callable(log_fn):
            log_fn(f"refine[{stage}] {message}")

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        message: str,
        **start_details: Any,
    ) -> Iterator[dict[str, Any]]:
        started = time.monotonic()
        self.emit(
            "stage_start",
            stage=name,
            message=message,
            **start_details,
        )
        completion: dict[str, Any] = {}
        try:
            yield completion
        except Exception as exc:
            elapsed = round(time.monotonic() - started, 6)
            self.stage_durations[str(name)] = elapsed
            self.emit(
                "stage_error",
                stage=name,
                message=f"{name} failed: {exc}",
                stage_elapsed_seconds=elapsed,
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            raise
        else:
            elapsed = round(time.monotonic() - started, 6)
            self.stage_durations[str(name)] = elapsed
            complete_message = str(
                completion.pop("message", f"{name} completed")
            )
            self.emit(
                "stage_complete",
                stage=name,
                message=complete_message,
                stage_elapsed_seconds=elapsed,
                **completion,
            )
