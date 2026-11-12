"""Small, rebuildable SQLite catalogue of Prism result folders."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from core.run_lifecycle import has_completion_marker
from core.run_manifest import RUN_MANIFEST_FILE, RUN_MANIFEST_SCHEMA


RUN_INDEX_FILE = "_run_index.sqlite"
RUN_INDEX_SCHEMA = "prism.run-index.v1"


def index_path(results_root: Path) -> Path:
    return results_root / RUN_INDEX_FILE


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _connect(results_root: Path) -> sqlite3.Connection:
    results_root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(index_path(results_root))
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            created_at TEXT,
            started_at TEXT,
            finished_at TEXT,
            dataset_id TEXT,
            abr TEXT,
            predictor TEXT,
            predictor_preset TEXT,
            bandwidth_predictor TEXT,
            run_group_id TEXT,
            run_group_name TEXT,
            manifest_schema TEXT,
            artifact_state TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )"""
    )
    connection.execute("CREATE INDEX IF NOT EXISTS runs_finished_idx ON runs(finished_at DESC)")
    connection.execute("CREATE INDEX IF NOT EXISTS runs_group_idx ON runs(run_group_id)")
    connection.execute("CREATE INDEX IF NOT EXISTS runs_predictor_idx ON runs(predictor)")
    connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)", ("schema", RUN_INDEX_SCHEMA)
    )
    return connection


def _payload_for_directory(run_dir: Path) -> dict[str, Any] | None:
    config = _read_json(run_dir / "config.json")
    status = _read_json(run_dir / "status.json")
    manifest = _read_json(run_dir / RUN_MANIFEST_FILE)
    if not config and not status and not manifest:
        return None
    run_id = str(manifest.get("run_id") or status.get("run_id") or run_dir.name)
    complete = has_completion_marker(run_dir)
    state = str(status.get("status") or manifest.get("state") or ("finished" if complete else "unknown"))
    return {
        "run_id": run_id,
        "status": state,
        "created_at": status.get("queued_at"),
        "started_at": status.get("started"),
        "finished_at": status.get("finished") or manifest.get("completed_at"),
        # Keep the compact index payload compatible with the established run
        # browser contract.  The *_at fields remain useful to external tools,
        # while these aliases avoid a second per-folder lookup in the UI.
        "queued_at": status.get("queued_at"),
        "started": status.get("started"),
        "finished": status.get("finished") or manifest.get("completed_at"),
        "dataset_id": config.get("dataset_id") or (manifest.get("inputs") or {}).get("dataset_id"),
        "abr": config.get("abr"),
        "predictor": config.get("predictor") or "None",
        "predictor_preset": config.get("predictor_preset_name"),
        "bandwidth_predictor": config.get("bandwidth_predictor") or "None",
        "run_group_id": config.get("run_group_id") or status.get("run_group_id"),
        "run_group_name": config.get("run_group_name") or status.get("run_group_name"),
        "run_group_source": config.get("run_group_source") or status.get("run_group_source"),
        "manifest_schema": manifest.get("schema"),
        "artifact_state": "complete" if complete else "legacy_or_incomplete",
        "config_summary": {
            "abr": config.get("abr"), "predictor": config.get("predictor") or "None",
            "bandwidth_predictor": config.get("bandwidth_predictor") or "None",
            "network_profile": config.get("network_profile"), "network_trace": str(config.get("network_trace") or ""),
            "fov_trace": str(config.get("fov_trace") or ""), "ground_truth_mode": config.get("ground_truth_mode") or "visible_fraction",
            "oracle_mode": config.get("oracle_mode") or "none",
        },
        "error": status.get("error"),
        "progress": status.get("progress"),
        "events_path": f"/api/runs/{run_id}/events" if (run_dir / "events.json").exists() else None,
        "log_path": f"/results/{run_id}/{run_id}.log",
        "predictor_diagnostics_path": (
            f"/results/{run_id}/predictor_diagnostics.json"
            if (run_dir / "predictor_diagnostics.json").exists() else None
        ),
        "oracle_mode": config.get("oracle_mode", "none"),
        "oracle_group_id": config.get("oracle_group_id"),
        "baseline_run_id": config.get("baseline_run_id"),
    }


def upsert_run_directory(results_root: Path, run_dir: Path) -> bool:
    payload = _payload_for_directory(run_dir)
    if payload is None:
        return False
    with _connect(results_root) as connection:
        connection.execute(
            """INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
              status=excluded.status, created_at=excluded.created_at, started_at=excluded.started_at,
              finished_at=excluded.finished_at, dataset_id=excluded.dataset_id, abr=excluded.abr,
              predictor=excluded.predictor, predictor_preset=excluded.predictor_preset,
              bandwidth_predictor=excluded.bandwidth_predictor, run_group_id=excluded.run_group_id,
              run_group_name=excluded.run_group_name, manifest_schema=excluded.manifest_schema,
              artifact_state=excluded.artifact_state, payload_json=excluded.payload_json""",
            (
                payload["run_id"], payload["status"], payload["created_at"], payload["started_at"],
                payload["finished_at"], payload["dataset_id"], payload["abr"], payload["predictor"],
                payload["predictor_preset"], payload["bandwidth_predictor"], payload["run_group_id"],
                payload["run_group_name"], payload["manifest_schema"], payload["artifact_state"],
                json.dumps(payload, sort_keys=True),
            ),
        )
    return True


def rebuild_run_index(results_root: Path) -> int:
    """Recreate the catalogue from manifests and legacy result folders."""
    with _connect(results_root) as connection:
        connection.execute("DELETE FROM runs")
    count = 0
    for run_dir in results_root.iterdir() if results_root.exists() else ():
        if run_dir.is_dir() and not run_dir.name.startswith("_") and upsert_run_directory(results_root, run_dir):
            count += 1
    return count


def ensure_run_index(results_root: Path) -> int:
    """Build the index once for pre-existing result folders, if needed."""
    if not index_path(results_root).exists():
        return rebuild_run_index(results_root)
    return indexed_run_count(results_root)


def indexed_run_count(results_root: Path) -> int:
    with _connect(results_root) as connection:
        return int(connection.execute("SELECT COUNT(*) FROM runs").fetchone()[0])


def remove_indexed_run(results_root: Path, run_id: str) -> bool:
    """Remove a run from the rebuildable catalogue after its artifacts are deleted."""
    with _connect(results_root) as connection:
        cursor = connection.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
    return cursor.rowcount > 0


def query_run_index(
    results_root: Path,
    *,
    limit: int = 200,
    offset: int = 0,
    search: str = "",
    status: str = "",
    group_id: str = "",
    ungrouped: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """Return one compact, filtered page for the run browser."""
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))
    clauses, values = [], []
    if status and status != "all": clauses.append("status = ?"); values.append(status)
    if group_id: clauses.append("run_group_id = ?"); values.append(group_id)
    if ungrouped: clauses.append("(run_group_id IS NULL OR run_group_id = '')")
    if search:
        clauses.append("(run_id LIKE ? OR dataset_id LIKE ? OR abr LIKE ? OR predictor LIKE ? OR run_group_name LIKE ?)")
        values.extend([f"%{search}%"] * 5)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    with _connect(results_root) as connection:
        total = int(connection.execute("SELECT COUNT(*) FROM runs" + where, values).fetchone()[0])
        rows = connection.execute("SELECT payload_json FROM runs" + where + " ORDER BY COALESCE(finished_at, created_at, run_id) DESC LIMIT ? OFFSET ?", values + [limit, offset]).fetchall()
    return [json.loads(row[0]) for row in rows], total


def query_run_group_summaries(results_root: Path, *, search: str = "", status: str = "") -> dict[str, dict[str, int]]:
    """Count matching indexed runs per batch, including lifecycle state."""
    clauses, values = ["run_group_id IS NOT NULL", "run_group_id != ''"], []
    if status and status != "all":
        clauses.append("status = ?")
        values.append(status)
    if search:
        clauses.append("(run_id LIKE ? OR dataset_id LIKE ? OR abr LIKE ? OR predictor LIKE ? OR run_group_name LIKE ?)")
        values.extend([f"%{search}%"] * 5)
    where = " WHERE " + " AND ".join(clauses)
    sql = (
        "SELECT run_group_id, COUNT(*), "
        "SUM(status = 'queued'), SUM(status = 'running'), SUM(status = 'paused'), "
        "SUM(status = 'finished'), SUM(status = 'failed'), SUM(status = 'canceled') "
        "FROM runs" + where + " GROUP BY run_group_id"
    )
    with _connect(results_root) as connection:
        rows = connection.execute(sql, values).fetchall()
    fields = ("total", "queued", "running", "paused", "finished", "failed", "canceled")
    return {str(row[0]): {field: int(row[index + 1] or 0) for index, field in enumerate(fields)} for row in rows}
