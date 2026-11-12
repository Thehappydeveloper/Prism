"""Compatibility helpers for compressed Prism event logs.

New runs store their event log as ``events.json.gz``.  Older uncompressed
``events.json`` logs remain readable so results folders can be mixed safely.
"""

from __future__ import annotations

import gzip
from pathlib import Path


EVENTS_JSON_NAME = "events.json"
EVENTS_GZIP_NAME = "events.json.gz"


def event_log_path(run_dir: Path) -> Path | None:
    """Return the preferred existing event-log path for one run."""
    compressed = run_dir / EVENTS_GZIP_NAME
    if compressed.exists():
        return compressed
    plain = run_dir / EVENTS_JSON_NAME
    return plain if plain.exists() else None


def event_log_exists(run_dir: Path) -> bool:
    return event_log_path(run_dir) is not None


def read_event_log_text(path: Path) -> str:
    """Read either a plain or gzip-compressed JSON event log as text."""
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8")
