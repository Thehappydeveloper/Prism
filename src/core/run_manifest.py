"""Versioned, portable provenance manifests for Prism run artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.run_lifecycle import atomic_write_json


RUN_MANIFEST_SCHEMA = "prism.run-manifest.v1"
RUN_MANIFEST_FILE = "manifest.json"


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _file_fingerprint(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    stat = path.stat()
    # Hash small control inputs.  Large media is represented by stable size and
    # mtime metadata so finalization does not reread a complete PLY sequence.
    digest = None
    if stat.st_size <= 32 * 1024 * 1024:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": digest,
    }


def _source_revision(project_root: Path) -> str | None:
    head = project_root / ".git" / "HEAD"
    try:
        value = head.read_text(encoding="utf-8").strip()
        if value.startswith("ref: "):
            return (project_root / ".git" / value[5:]).read_text(encoding="utf-8").strip() or None
        return value or None
    except OSError:
        return None


def build_run_manifest(run_id: str, config: Any, video: Any, run_dir: Path) -> dict[str, Any]:
    """Build the completion manifest without changing any legacy artifacts."""
    config_value = _json_value(config)
    if not isinstance(config_value, dict):
        raise ValueError("Run manifest requires a mapping-like playback configuration.")
    mpd_path = Path(config_value["mpd"]) if config_value.get("mpd") else None
    content_dir = Path(config_value["content_dir"]) if config_value.get("content_dir") else None
    fov_trace = Path(config_value["fov_trace"]) if config_value.get("fov_trace") else None
    network_trace = Path(config_value["network_trace"]) if config_value.get("network_trace") else None
    project_root = Path(__file__).resolve().parents[2]
    artifact_names = ("config.json", "events.json.gz", "events.json", "predictor_diagnostics.json", "timing.json")
    artifacts = {name: _file_fingerprint(run_dir / name) for name in artifact_names if (run_dir / name).exists()}
    return {
        "schema": RUN_MANIFEST_SCHEMA,
        "run_id": str(run_id),
        "state": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source": {"revision": _source_revision(project_root), "dirty": "unverified"},
        "inputs": {
            "dataset_id": config_value.get("dataset_id"),
            "mpd": _file_fingerprint(mpd_path),
            "content": {
                "path": str(content_dir) if content_dir else None,
                "frame_count": int(getattr(video, "total_frames", 0) or 0),
                "first_frame": int(getattr(video, "start_frame_idx", 0) or 0),
            },
            "timeline": {
                "duration_s": float(getattr(video, "total_playback_time", 0.0) or 0.0),
                "segment_count": int(getattr(video, "segment_count", 0) or 0),
                "frame_rate_hz": float(getattr(video, "frame_rate_hz", 30.0) or 30.0),
            },
            "fov_trace": _file_fingerprint(fov_trace),
            "network_trace": _file_fingerprint(network_trace),
            "network_profile": config_value.get("network_profile"),
        },
        # The run launcher has already expanded all selected parameters into
        # this concrete configuration.  Keeping both fields makes the schema
        # forward-compatible with a later preflight requested/resolved split.
        "configuration": {"requested": config_value, "resolved": config_value},
        "artifacts": artifacts,
    }


def validate_run_manifest(manifest: Any) -> list[str]:
    """Return schema errors; an empty list means the manifest is valid."""
    if not isinstance(manifest, dict):
        return ["manifest must be an object"]
    errors: list[str] = []
    if manifest.get("schema") != RUN_MANIFEST_SCHEMA:
        errors.append(f"schema must be {RUN_MANIFEST_SCHEMA}")
    for key in ("run_id", "state", "completed_at", "source", "inputs", "configuration", "artifacts"):
        if key not in manifest:
            errors.append(f"missing required field: {key}")
    if manifest.get("state") != "complete":
        errors.append("state must be complete")
    if not isinstance(manifest.get("configuration"), dict):
        errors.append("configuration must be an object")
    return errors


def write_run_manifest(run_id: str, config: Any, video: Any, run_dir: Path) -> Path:
    manifest = build_run_manifest(run_id, config, video, run_dir)
    errors = validate_run_manifest(manifest)
    if errors:
        raise ValueError("Invalid run manifest: " + "; ".join(errors))
    path = run_dir / RUN_MANIFEST_FILE
    atomic_write_json(path, manifest)
    return path
