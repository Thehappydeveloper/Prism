"""Durable lifecycle markers for restart-safe experiment runs."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


COMPLETION_FILE_NAME = "completion.json"
INCOMPLETE_DIR_NAME = ".incomplete"


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically replace a small JSON control file after flushing it to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        # os.replace is atomic on the target filesystem and remains safe when
        # two progress callbacks write the same status file concurrently.
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def completion_path(run_dir: Path) -> Path:
    return run_dir / COMPLETION_FILE_NAME


def mark_run_complete(run_dir: Path, run_id: str) -> None:
    """Commit a run only after every final artifact has been written."""
    atomic_write_json(
        completion_path(run_dir),
        {
            "schema": "prism.run-completion.v1",
            "run_id": str(run_id),
            "completed_at": datetime.now().isoformat(),
        },
    )


def has_completion_marker(run_dir: Path) -> bool:
    return completion_path(run_dir).exists()


def prepare_incomplete_dir(run_dir: Path) -> Path:
    """Create an empty staging directory for one new runtime attempt."""
    staging = run_dir / INCOMPLETE_DIR_NAME
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    return staging


def promote_incomplete_dir(run_dir: Path) -> None:
    """Atomically promote completed runtime artifacts into the canonical run directory."""
    staging = run_dir / INCOMPLETE_DIR_NAME
    if not staging.exists():
        raise FileNotFoundError(f"Run staging directory is missing: {staging}")
    for artifact in staging.iterdir():
        target = run_dir / artifact.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        artifact.replace(target)
    staging.rmdir()
