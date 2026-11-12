from __future__ import annotations

import json
import shutil
import threading
import time
from datetime import datetime
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List

from app import PlaybackConfig, PlaybackGroupCanceled, run_playback, run_playback_group, save_config
from abr import baselines as abr_baselines
from core.network_profiles import get_network_profile
from core.event_storage import event_log_path, read_event_log_text
from core.run_lifecycle import atomic_write_json, completion_path, has_completion_marker
from core.run_index import ensure_run_index, upsert_run_directory
from core.compute_processes import begin_compute_session, force_stop_compute_pools

from .config import (
    RESULTS_ROOT,
    generate_run_id,
    get_dataset_profile,
    navis_settings,
    parse_bool,
    prism_settings,
    read_json,
    require_nonempty_path,
    resolve_headset_config,
    safe_path,
)


RUNS_LOCK = threading.Lock()
RUNS: Dict[str, "RunJob"] = {}
RUN_QUEUE: List[str] = []
RUN_QUEUE_CONDITION = threading.Condition(RUNS_LOCK)
RUN_QUEUE_WORKER: threading.Thread | None = None
RUN_LOAD_CACHE: Dict[str, Dict[str, Any]] = {}
RUN_LOAD_CACHE_LOCK = threading.Lock()
RUN_LOAD_CACHE_MAX_ENTRIES = 64
RUN_GROUPS_FILE = RESULTS_ROOT / "_run_groups.json"
INCOMPLETE_RUNTIME_FILES = (
    "events.json",
    "events.json.gz",
    ".events.json.tmp",
    ".events.json.gz.tmp",
    "timing.json",
    "predictor_diagnostics.json",
    "completion.json",
    ".incomplete",
)

ORACLE_MODE_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "bandwidth": {
        "bandwidth_predictor": "GroundTruthBandwidthPredictor",
        "bandwidth_predictor_params": {},
    },
    "fov_trace": {
        "predictor": "GroundTruthFovPredictor",
        "predictor_params": {},
    },
    "fov_tile_scores": {
        "predictor": "GroundTruthTileFovPredictor",
        "predictor_params": {},
    },
    "bandwidth_and_fov_trace": {
        "bandwidth_predictor": "GroundTruthBandwidthPredictor",
        "bandwidth_predictor_params": {},
        "predictor": "GroundTruthFovPredictor",
        "predictor_params": {},
    },
    "bandwidth_and_fov_tile_scores": {
        "bandwidth_predictor": "GroundTruthBandwidthPredictor",
        "bandwidth_predictor_params": {},
        "predictor": "GroundTruthTileFovPredictor",
        "predictor_params": {},
    },
    "offline_optimal": {
        "abr": "OfflineGreedyOptimalABR",
        "abr_params": {},
        "bandwidth_predictor": "GroundTruthBandwidthPredictor",
        "bandwidth_predictor_params": {},
        "predictor": "GroundTruthTileFovPredictor",
        "predictor_params": {},
    },
}

ORACLE_FOV_MODES = {"fov_trace", "fov_tile_scores", "bandwidth_and_fov_trace", "bandwidth_and_fov_tile_scores", "offline_optimal"}


def _config_summary(config: Any) -> Dict[str, Any]:
    if hasattr(config, "__dict__"):
        getter = lambda key, default=None: getattr(config, key, default)
    elif isinstance(config, dict):
        getter = config.get
    else:
        getter = lambda _key, default=None: default
    return {
        "abr": getter("abr"),
        "predictor": getter("predictor") or "None",
        "bandwidth_predictor": getter("bandwidth_predictor") or "None",
        "network_profile": getter("network_profile") or None,
        "network_trace": str(getter("network_trace") or ""),
        "fov_trace": str(getter("fov_trace") or ""),
        "ground_truth_mode": getter("ground_truth_mode") or "visible_fraction",
        "oracle_mode": getter("oracle_mode") or "none",
    }


def _now_iso() -> str:
    return datetime.now().isoformat()


def _safe_group_id(value: Any, fallback_prefix: str = "group") -> str:
    raw = str(value or "").strip().lower()
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or f"{fallback_prefix}_{datetime.now().strftime('%y%m%d_%H%M%S')}"


def _read_run_groups() -> Dict[str, Any]:
    raw = read_json(RUN_GROUPS_FILE)
    groups = raw.get("groups") if isinstance(raw, dict) else {}
    return {"version": 1, "groups": groups if isinstance(groups, dict) else {}}


def _write_run_groups(payload: Dict[str, Any]) -> None:
    RUN_GROUPS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = RUN_GROUPS_FILE.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp_path.replace(RUN_GROUPS_FILE)


def _run_group_payload(group_id: str, group: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(group)
    item["id"] = str(item.get("id") or group_id)
    item["run_ids"] = sorted({str(run_id) for run_id in item.get("run_ids") or []})
    item["archived"] = bool(item.get("archived", False))
    return item


def list_run_groups() -> List[Dict[str, Any]]:
    groups = _read_run_groups()["groups"]
    return sorted((_run_group_payload(group_id, group) for group_id, group in groups.items()), key=lambda item: str(item.get("created_at") or item["id"]), reverse=True)


def upsert_run_group(payload: Dict[str, Any]) -> Dict[str, Any]:
    name = str(payload.get("name") or payload.get("id") or "Run group").strip()
    group_id = _safe_group_id(payload.get("id") or name)
    registry = _read_run_groups()
    groups = dict(registry["groups"])
    existing = dict(groups.get(group_id) or {})
    now = _now_iso()
    run_ids = sorted({*(str(v) for v in existing.get("run_ids") or []), *(str(v) for v in payload.get("run_ids") or [])})
    groups[group_id] = {
        "id": group_id,
        "name": name,
        "description": str(payload.get("description") if payload.get("description") is not None else existing.get("description") or ""),
        "source": str(payload.get("source") or existing.get("source") or "manual"),
        "plan_path": str(payload.get("plan_path") or existing.get("plan_path") or ""),
        "created_at": existing.get("created_at") or now,
        "updated_at": now,
        "archived": bool(payload.get("archived", existing.get("archived", False))),
        "run_ids": run_ids,
    }
    _write_run_groups({"version": 1, "groups": groups})
    return _run_group_payload(group_id, groups[group_id])


def _update_run_config_group(run_id: str, group_id: str | None, group_name: str | None, group_source: str | None = None) -> None:
    cfg_file = RESULTS_ROOT / run_id / "config.json"
    if not cfg_file.exists():
        return
    cfg = read_json(cfg_file)
    if group_id:
        cfg["run_group_id"] = group_id
        cfg["run_group_name"] = group_name or group_id
        cfg["run_group_source"] = group_source or cfg.get("run_group_source") or "manual"
    else:
        cfg.pop("run_group_id", None)
        cfg.pop("run_group_name", None)
        cfg.pop("run_group_source", None)
    atomic_write_json(cfg_file, cfg)


def _update_run_status_group(run_id: str, group_id: str | None, group_name: str | None, group_source: str | None = None) -> None:
    status_file = RESULTS_ROOT / run_id / "status.json"
    if not status_file.exists():
        return
    status = read_json(status_file)
    if group_id:
        status["run_group_id"] = group_id
        status["run_group_name"] = group_name or group_id
        status["run_group_source"] = group_source or status.get("run_group_source") or "manual"
    else:
        status.pop("run_group_id", None)
        status.pop("run_group_name", None)
        status.pop("run_group_source", None)
    atomic_write_json(status_file, status)


def _group_run_ids(group_id: str, group: Dict[str, Any]) -> List[str]:
    run_ids = {str(run_id) for run_id in group.get("run_ids") or [] if str(run_id).strip()}
    group_name = str(group.get("name") or "").strip()
    if RESULTS_ROOT.exists():
        for run_dir in RESULTS_ROOT.iterdir():
            if not run_dir.is_dir() or run_dir.name.startswith("_"):
                continue
            cfg = read_json(run_dir / "config.json")
            status = read_json(run_dir / "status.json")
            run_group_id = cfg.get("run_group_id") or status.get("run_group_id")
            run_group_name = cfg.get("run_group_name") or status.get("run_group_name")
            if run_group_id == group_id or (not run_group_id and group_name and run_group_name == group_name):
                run_ids.add(run_dir.name)
    return sorted(run_ids)


def _control_persisted_run(run_id: str, action: str) -> bool:
    status_file = RESULTS_ROOT / run_id / "status.json"
    if not status_file.exists():
        return False
    status = read_json(status_file)
    current_status = str(status.get("status") or "").strip().lower()
    if action == "cancel":
        if current_status not in {"queued", "running", "paused"}:
            return False
        status["status"] = "canceled"
        status["error"] = "Canceled by group control."
        status["finished"] = _now_iso()
        progress = dict(status.get("progress") or {})
        progress.update(
            {
                "phase": "canceled",
                "detail": {"reason": status["error"], "previous_status": current_status},
                "phase_started_at": status["finished"],
                "heartbeat_at": status["finished"],
            }
        )
        status["progress"] = progress
    elif action == "pause":
        if current_status not in {"queued", "running"}:
            return False
        now = _now_iso()
        status["status"] = "paused"
        progress = dict(status.get("progress") or {})
        progress.update(
            {
                "phase": "paused",
                "detail": {"reason": "Paused by group control.", "previous_status": current_status},
                "phase_started_at": now,
                "heartbeat_at": now,
            }
        )
        status["progress"] = progress
    else:
        return False
    atomic_write_json(status_file, status)
    return True


def assign_runs_to_group(payload: Dict[str, Any]) -> Dict[str, Any]:
    run_ids = [str(v) for v in payload.get("run_ids") or [] if str(v).strip()]
    if not run_ids:
        raise ValueError("Select at least one run.")
    group = upsert_run_group({**payload, "run_ids": run_ids})
    group_id = str(group["id"])
    group_name = str(group.get("name") or group_id)
    group_source = str(group.get("source") or "manual")
    registry = _read_run_groups()
    groups = dict(registry["groups"])
    for gid, item in list(groups.items()):
        if gid == group_id:
            continue
        item = dict(item)
        item["run_ids"] = [run_id for run_id in item.get("run_ids") or [] if run_id not in run_ids]
        groups[gid] = item
    target = dict(groups.get(group_id) or group)
    target["run_ids"] = sorted({*(str(v) for v in target.get("run_ids") or []), *run_ids})
    target["updated_at"] = _now_iso()
    groups[group_id] = target
    _write_run_groups({"version": 1, "groups": groups})
    for run_id in run_ids:
        _update_run_config_group(run_id, group_id, group_name, group_source)
        _update_run_status_group(run_id, group_id, group_name, group_source)
        with RUNS_LOCK:
            job = RUNS.get(run_id)
            if job:
                job.config.run_group_id = group_id
                job.config.run_group_name = group_name
                job.config.run_group_source = group_source
                job._persist_status()  # pylint: disable=protected-access
    return {"groups": list_run_groups()}


def remove_runs_from_groups(payload: Dict[str, Any]) -> Dict[str, Any]:
    run_ids = {str(v) for v in payload.get("run_ids") or [] if str(v).strip()}
    if not run_ids:
        raise ValueError("Select at least one run.")
    registry = _read_run_groups()
    groups = dict(registry["groups"])
    now = _now_iso()
    for gid, item in list(groups.items()):
        item = dict(item)
        item["run_ids"] = [run_id for run_id in item.get("run_ids") or [] if run_id not in run_ids]
        item["updated_at"] = now
        groups[gid] = item
    _write_run_groups({"version": 1, "groups": groups})
    for run_id in run_ids:
        _update_run_config_group(run_id, None, None)
        _update_run_status_group(run_id, None, None)
        with RUNS_LOCK:
            job = RUNS.get(run_id)
            if job:
                job.config.run_group_id = None
                job.config.run_group_name = None
                job.config.run_group_source = None
                job._persist_status()  # pylint: disable=protected-access
    return {"groups": list_run_groups()}


def control_run_group(group_id: str, action: str) -> Dict[str, Any]:
    action = str(action or "").strip().lower()
    if action not in {"pause", "resume", "cancel", "force_cancel"}:
        raise ValueError("Group action must be 'pause', 'resume', 'cancel', or 'force_cancel'.")
    registry = _read_run_groups()
    group = dict((registry.get("groups") or {}).get(group_id) or {})
    run_ids = _group_run_ids(group_id, group)
    if not run_ids:
        raise ValueError("Run group not found or empty.")

    affected = 0
    with RUN_QUEUE_CONDITION:
        for run_id in run_ids:
            job = RUNS.get(run_id)
            if job is None:
                if _control_persisted_run(run_id, action):
                    affected += 1
                continue
            if action in {"cancel", "force_cancel"}:
                if job.status in {"queued", "paused"} and run_id in RUN_QUEUE:
                    RUN_QUEUE.remove(run_id)
                if job.status in {"queued", "running", "paused"}:
                    job.request_cancel()
                    affected += 1
            elif action == "pause":
                if job.status == "queued" and run_id in RUN_QUEUE:
                    RUN_QUEUE.remove(run_id)
                if job.status in {"queued", "running"}:
                    job.request_pause()
                    affected += 1
            elif action == "resume":
                was_paused = job.status == "paused"
                job.request_resume()
                if was_paused and job.status == "queued" and run_id not in RUN_QUEUE:
                    RUN_QUEUE.append(run_id)
                    affected += 1
                elif job.status == "running":
                    affected += 1
        if action == "resume" and affected:
            _ensure_queue_worker_locked()
        RUN_QUEUE_CONDITION.notify_all()
    force_stop = force_stop_compute_pools() if action == "force_cancel" else None
    return {
        "status": action,
        "group_id": group_id,
        "affected": affected,
        "force_stop": force_stop,
        "groups": list_run_groups(),
    }


def archive_run_group(group_id: str, archived: bool) -> Dict[str, Any]:
    registry = _read_run_groups()
    groups = dict(registry["groups"])
    if group_id not in groups:
        raise ValueError("Run group not found.")
    groups[group_id] = {**dict(groups[group_id]), "archived": bool(archived), "updated_at": _now_iso()}
    _write_run_groups({"version": 1, "groups": groups})
    return {"groups": list_run_groups()}


def delete_run_group(group_id: str) -> Dict[str, Any]:
    registry = _read_run_groups()
    groups = dict(registry["groups"])
    group = dict(groups.pop(group_id, {}) or {})
    run_ids = [str(v) for v in group.get("run_ids") or []]
    _write_run_groups({"version": 1, "groups": groups})
    for run_id in run_ids:
        _update_run_config_group(run_id, None, None)
        _update_run_status_group(run_id, None, None)
        with RUNS_LOCK:
            job = RUNS.get(run_id)
            if job:
                job.config.run_group_id = None
                job.config.run_group_name = None
                job.config.run_group_source = None
                job._persist_status()  # pylint: disable=protected-access
    return {"groups": list_run_groups()}


def delete_run_group_with_runs(group_id: str) -> Dict[str, Any]:
    """Delete a group registry entry and every completed run it owns.

    Live playback owns open files and worker state, so deleting a group with an
    active run is deliberately rejected.  The caller can cancel the group,
    wait for its runs to settle, and then delete it safely.
    """
    registry = _read_run_groups()
    groups = dict(registry["groups"])
    group = dict(groups.get(group_id) or {})
    if not group:
        raise ValueError("Run group not found.")
    run_ids = _group_run_ids(group_id, group)

    active: list[str] = []
    with RUN_QUEUE_CONDITION:
        for run_id in run_ids:
            job = RUNS.get(run_id)
            status = str(job.status if job else read_json(RESULTS_ROOT / run_id / "status.json").get("status") or "").lower()
            if status in {"queued", "running", "paused"}:
                active.append(run_id)
        if active:
            raise ValueError(
                "Cannot delete a group containing active runs: "
                + ", ".join(active[:5])
                + (" …" if len(active) > 5 else "")
                + ". Cancel them and wait for them to finish first."
            )

        # Remove the group first, then erase only inactive run directories.
        groups.pop(group_id, None)
        _write_run_groups({"version": 1, "groups": groups})
        for run_id in run_ids:
            if run_id in RUN_QUEUE:
                RUN_QUEUE.remove(run_id)
            RUNS.pop(run_id, None)
            run_dir = RESULTS_ROOT / run_id
            if run_dir.exists() and run_dir.is_dir():
                shutil.rmtree(run_dir)

    with RUN_LOAD_CACHE_LOCK:
        for cache_key in list(RUN_LOAD_CACHE):
            if any(f'"run_id":"{run_id}"' in cache_key for run_id in run_ids):
                RUN_LOAD_CACHE.pop(cache_key, None)
    return {"groups": list_run_groups(), "deleted_run_ids": run_ids}


def run_artifact_fingerprint(run_dir: Path) -> Dict[str, Dict[str, Any] | None]:
    fingerprint: Dict[str, Dict[str, Any] | None] = {}
    for name in ("config.json", "status.json"):
        path = run_dir / name
        if not path.exists():
            fingerprint[name] = None
            continue
        stat = path.stat()
        fingerprint[name] = {
            "path": str(path),
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
        }
    events_file = event_log_path(run_dir)
    if events_file is None:
        fingerprint["events.json"] = None
    else:
        stat = events_file.stat()
        fingerprint["events.json"] = {
            "path": str(events_file),
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
        }
    return fingerprint


class RunJob(threading.Thread):
    def __init__(self, config: PlaybackConfig):
        super().__init__(daemon=True)
        self.config = config
        self.run_id = config.run_id or generate_run_id()
        self.config.run_id = self.run_id
        self.run_dir = config.output_dir / self.run_id
        self.status = "queued"
        self.error: str | None = None
        self.navis_export: Dict[str, Any] | None = None
        self.fov_comparison: Dict[str, Any] | None = None
        self.queued_at = datetime.now().isoformat()
        self.progress_current = 0
        self.progress_total = 0
        self.progress_group_id: str | None = None
        self.progress_group_size = 1
        self.progress_phase = "queued"
        self.progress_detail: Dict[str, Any] = {}
        self.progress_phase_started_at = self.queued_at
        self.progress_heartbeat_at = self.progress_phase_started_at
        self.pause_requested = False
        self.cancel_requested = False
        self.started: str | None = None
        self.finished: str | None = None
        self.attempt = 1
        save_config(self.run_dir, self.config)
        self._persist_status()

    def _persist_status(self) -> None:
        status_file = self.run_dir / "status.json"
        atomic_write_json(status_file, self.to_dict())
        upsert_run_directory(self.run_dir.parent, self.run_dir)

    def run(self):
        if self.status == "canceled":
            return
        completion_path(self.run_dir).unlink(missing_ok=True)
        self.status = "running"
        self.started = datetime.now().isoformat()
        self.progress_group_id = None
        self.progress_group_size = 1
        self.update_progress_detail("running", {"mode": "single_run"}, persist=False)
        self._persist_status()
        try:
            run_playback(self.config, progress_callback=self.update_progress)
            self.status = "finished"
            if navis_settings().get("auto_export"):
                from .navis_export import export_run_to_navis

                self.navis_export = export_run_to_navis(self.run_id)
        except Exception as exc:  # pylint: disable=broad-except
            self.status = "failed"
            self.error = str(exc)
        finally:
            self.finished = datetime.now().isoformat()
            if self.status == "finished":
                self.update_progress_detail("finished", {"status": "finished"}, persist=False)
            self._persist_status()
            if self.status == "finished":
                from .diagnostics import schedule_default_run_summary

                schedule_default_run_summary(self.run_id)

    def cancel_queued(self) -> bool:
        if self.status not in {"queued", "paused"}:
            return False
        self.status = "canceled"
        self.error = "Canceled before start."
        self.finished = datetime.now().isoformat()
        self.cancel_requested = True
        self.update_progress_detail("canceled", {"reason": self.error}, persist=False)
        self._persist_status()
        return True

    def request_cancel(self) -> None:
        self.cancel_requested = True
        self.pause_requested = False
        if self.status in {"queued", "paused"}:
            self.status = "canceled"
            self.error = "Canceled."
            self.finished = datetime.now().isoformat()
            self.update_progress_detail("canceled", {"reason": self.error}, persist=False)
        else:
            self.update_progress_detail("cancel_requested", {"status": self.status}, persist=False)
        self._persist_status()

    def request_pause(self) -> None:
        if self.status == "queued":
            self.pause_requested = True
            self.status = "paused"
            self.update_progress_detail("paused", {"reason": "Paused before start."}, persist=False)
            self._persist_status()
        elif self.status == "running":
            self.pause_requested = True
            self.update_progress_detail("pause_requested", {"status": self.status}, persist=False)
            self._persist_status()

    def request_resume(self) -> None:
        if self.status == "paused":
            self.status = "running" if self.started and not self.finished and self.progress_group_id else "queued"
        self.pause_requested = False
        if self.status in {"queued", "running"}:
            self.update_progress_detail("resume_requested", {"status": self.status}, persist=False)
        self._persist_status()

    def update_progress(self, _run_id: str, current: int, total: int) -> None:
        self.progress_current = max(0, int(current or 0))
        self.progress_total = max(0, int(total or 0))
        self.progress_heartbeat_at = _now_iso()

    def update_progress_detail(self, phase: str | None = None, detail: Dict[str, Any] | None = None, persist: bool = False) -> None:
        now = _now_iso()
        next_phase = str(phase or self.progress_phase or "running")
        if next_phase != self.progress_phase:
            self.progress_phase = next_phase
            self.progress_phase_started_at = now
        self.progress_heartbeat_at = now
        if detail is not None:
            safe_detail: Dict[str, Any] = {}
            for key, value in detail.items():
                if not isinstance(key, str):
                    continue
                if isinstance(value, (str, int, float, bool)) or value is None:
                    safe_detail[key] = value
                else:
                    safe_detail[key] = str(value)
            self.progress_detail = safe_detail
        if persist:
            self._persist_status()

    def to_dict(self) -> Dict:
        from .fov_comparison import fov_comparison_paths

        predictor_diagnostics_file = self.run_dir / "predictor_diagnostics.json"
        events_file = event_log_path(self.run_dir)
        return {
            "run_id": self.run_id,
            "status": self.status,
            "error": self.error,
            "cancel_requested": self.cancel_requested,
            "navis_export": self.navis_export,
            "fov_comparison": self.fov_comparison or fov_comparison_paths(self.run_id),
            "queued_at": self.queued_at,
            "started": self.started,
            "finished": self.finished,
            "attempt": self.attempt,
            "events_path": (f"/api/runs/{self.run_id}/events" if events_file is not None else None),
            "log_path": f"/results/{self.run_id}/{self.run_id}.log",
            "predictor_diagnostics_path": (f"/results/{self.run_id}/predictor_diagnostics.json" if predictor_diagnostics_file.exists() else None),
            "artifact_fingerprint": run_artifact_fingerprint(self.run_dir),
            "config_summary": _config_summary(self.config),
            "progress": {
                "current": self.progress_current,
                "total": self.progress_total,
                "group_id": self.progress_group_id,
                "group_size": self.progress_group_size,
                "phase": self.progress_phase,
                "detail": self.progress_detail,
                "phase_started_at": self.progress_phase_started_at,
                "heartbeat_at": self.progress_heartbeat_at,
            },
            "oracle_mode": getattr(self.config, "oracle_mode", "none"),
            "oracle_group_id": getattr(self.config, "oracle_group_id", None),
            "baseline_run_id": getattr(self.config, "baseline_run_id", None),
            "run_group_id": getattr(self.config, "run_group_id", None),
            "run_group_name": getattr(self.config, "run_group_name", None),
            "run_group_source": getattr(self.config, "run_group_source", None),
        }


def normalize_param_payload(raw: Any, label: str) -> Dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be an object/dict.")
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, (bool, int, float, str)) or v is None:
            out[k] = v
            continue
        raise ValueError(f"{label}.{k} has unsupported type '{type(v).__name__}'.")
    return out


def find_mpd_in_folder(folder: str | Path) -> Path | None:
    candidate = safe_path(folder)
    if candidate.is_file() and candidate.suffix.lower() == ".mpd":
        return candidate
    if candidate.is_dir():
        matches = [p for p in candidate.iterdir() if p.is_file() and p.suffix.lower() == ".mpd"]
        if matches:
            return matches[0].resolve()
    return None


def load_existing_runs() -> List[Dict]:
    from .fov_comparison import fov_comparison_paths

    runs: List[Dict] = []
    if not RESULTS_ROOT.exists():
        return runs
    for run_dir in RESULTS_ROOT.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        status_file = run_dir / "status.json"
        events_file = event_log_path(run_dir)
        cfg_file = run_dir / "config.json"
        predictor_diagnostics_file = run_dir / "predictor_diagnostics.json"
        status = read_json(status_file) if status_file.exists() else {}
        cfg = read_json(cfg_file) if cfg_file.exists() else {}
        inferred_status = status.get("status") or ("finished" if events_file is not None and cfg_file.exists() else "unknown")
        runs.append(
            {
                "run_id": run_dir.name,
                "status": inferred_status,
                "error": status.get("error"),
                "navis_export": status.get("navis_export"),
                "fov_comparison": status.get("fov_comparison") or fov_comparison_paths(run_dir.name),
                "queued_at": status.get("queued_at"),
                "started": status.get("started"),
                "finished": status.get("finished"),
                "events_path": (f"/api/runs/{run_dir.name}/events" if events_file is not None else None),
                "log_path": f"/results/{run_dir.name}/{run_dir.name}.log",
                "predictor_diagnostics_path": (f"/results/{run_dir.name}/predictor_diagnostics.json" if predictor_diagnostics_file.exists() else None),
                "artifact_fingerprint": run_artifact_fingerprint(run_dir),
                "config_summary": _config_summary(cfg),
                "progress": status.get("progress"),
                "oracle_mode": cfg.get("oracle_mode", "none"),
                "oracle_group_id": cfg.get("oracle_group_id"),
                "baseline_run_id": cfg.get("baseline_run_id"),
                "run_group_id": cfg.get("run_group_id") or status.get("run_group_id"),
                "run_group_name": cfg.get("run_group_name") or status.get("run_group_name"),
                "run_group_source": cfg.get("run_group_source") or status.get("run_group_source"),
            }
        )
    return sorted(runs, key=lambda r: r["run_id"], reverse=True)


def load_run(run_id: str) -> Dict | None:
    events_file = event_log_path(RESULTS_ROOT / run_id)
    cfg_file = RESULTS_ROOT / run_id / "config.json"
    if events_file is None or not cfg_file.exists():
        return None
    config = json.loads(cfg_file.read_text())
    manifest = read_json(RESULTS_ROOT / run_id / "manifest.json")
    resolved = (manifest.get("configuration") or {}).get("resolved") if isinstance(manifest, dict) else None
    if isinstance(resolved, dict):
        config = resolved
    try:
        events = json.loads(read_event_log_text(events_file))
    except json.JSONDecodeError as exc:
        return {
            "run_id": run_id,
            "config": config,
            "manifest": manifest or None,
            "events_error": str(exc),
            "artifact_fingerprint": run_artifact_fingerprint(RESULTS_ROOT / run_id),
        }
    return {
        "run_id": run_id,
        "events": events,
        "config": config,
        "manifest": manifest or None,
        "artifact_fingerprint": run_artifact_fingerprint(RESULTS_ROOT / run_id),
    }


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _result_fingerprint(fingerprint: Dict[str, Any]) -> Dict[str, Any]:
    return {name: fingerprint.get(name) for name in ("config.json", "events.json")}


def load_run_cached(run_id: str) -> Dict | None:
    run_dir = RESULTS_ROOT / run_id
    fingerprint = run_artifact_fingerprint(run_dir)
    cache_key = _stable_json({"run_id": run_id, "fingerprint": _result_fingerprint(fingerprint)})
    started = time.perf_counter()
    with RUN_LOAD_CACHE_LOCK:
        cached = RUN_LOAD_CACHE.get(cache_key)
    if cached is not None:
        print(f"[graph-timing] load_run cache_hit run={run_id} elapsed_ms={int((time.perf_counter() - started) * 1000)}")
        return cached.get("run")

    events_file = event_log_path(run_dir)
    cfg_file = run_dir / "config.json"
    if events_file is None or not cfg_file.exists():
        return None

    config_started = time.perf_counter()
    config_text = cfg_file.read_text()
    config_read_ms = int((time.perf_counter() - config_started) * 1000)
    config_parse_started = time.perf_counter()
    config = json.loads(config_text)
    manifest = read_json(run_dir / "manifest.json")
    resolved = (manifest.get("configuration") or {}).get("resolved") if isinstance(manifest, dict) else None
    if isinstance(resolved, dict):
        # New artifacts preserve the configuration actually resolved at launch.
        config = resolved
    config_parse_ms = int((time.perf_counter() - config_parse_started) * 1000)

    events_started = time.perf_counter()
    events_text = read_event_log_text(events_file)
    events_read_ms = int((time.perf_counter() - events_started) * 1000)
    events_parse_started = time.perf_counter()
    try:
        events = json.loads(events_text)
        run = {"run_id": run_id, "events": events, "config": config, "manifest": manifest or None, "artifact_fingerprint": fingerprint}
    except json.JSONDecodeError as exc:
        run = {"run_id": run_id, "config": config, "events_error": str(exc), "artifact_fingerprint": fingerprint}
    events_parse_ms = int((time.perf_counter() - events_parse_started) * 1000)

    with RUN_LOAD_CACHE_LOCK:
        while len(RUN_LOAD_CACHE) >= RUN_LOAD_CACHE_MAX_ENTRIES:
            RUN_LOAD_CACHE.pop(next(iter(RUN_LOAD_CACHE)), None)
        RUN_LOAD_CACHE[cache_key] = {"run": run, "fingerprint": fingerprint}

    print(
        "[graph-timing] load_run cache_miss "
        f"run={run_id} config_read_ms={config_read_ms} config_parse_ms={config_parse_ms} "
        f"events_read_ms={events_read_ms} events_parse_ms={events_parse_ms} "
        f"elapsed_ms={int((time.perf_counter() - started) * 1000)}"
    )
    return run


def list_runs() -> List[Dict]:
    # The list API still reads legacy folders in this foundation phase, but the
    # first visit creates the compact catalogue needed by the later paginated
    # API without requiring an explicit migration step.
    ensure_run_index(RESULTS_ROOT)
    with RUNS_LOCK:
        _reconcile_completed_in_memory_jobs_locked()
        in_memory = {job.run_id: job.to_dict() for job in RUNS.values()}
        queued_order = {run_id: idx + 1 for idx, run_id in enumerate(RUN_QUEUE)}
        for run_id, payload in in_memory.items():
            payload["queue_position"] = queued_order.get(run_id)
    disk_runs = {run["run_id"]: run for run in load_existing_runs()}
    return sorted({**disk_runs, **in_memory}.values(), key=lambda r: r["run_id"], reverse=True)


def _reconcile_completed_in_memory_jobs_locked() -> int:
    """Discard stale active states when the immutable completion marker exists.

    A server restart can occur after grouped playback writes its artifacts but
    before the parent queue updates every in-memory RunJob.  Those objects
    must never override their completed on-disk state in the live queue.
    Caller holds ``RUNS_LOCK``.
    """
    reconciled = 0
    for job in RUNS.values():
        if job.status not in {"queued", "running", "paused", "canceling"}:
            continue
        if not has_completion_marker(job.run_dir):
            continue
        job.status = "finished"
        job.error = None
        job.finished = job.finished or _now_iso()
        job.update_progress_detail(
            "finished",
            {"status": "finished", "reconciled": True, "reason": "Completion marker verified."},
            persist=False,
        )
        job._persist_status()
        reconciled += 1
    if reconciled:
        RUN_QUEUE[:] = [run_id for run_id in RUN_QUEUE if RUNS.get(run_id) is not None and RUNS[run_id].status == "queued"]
        print(f"[run-queue] reconciled_completed_in_memory_jobs count={reconciled}")
    return reconciled


def active_runs_snapshot() -> List[Dict]:
    """Return in-memory work only, for the live UI stream.

    Completed results are immutable and come from the paginated SQLite index;
    repeatedly rescanning every result folder just to refresh active progress is
    unnecessarily expensive once an experiment collection becomes large.
    """
    with RUNS_LOCK:
        _reconcile_completed_in_memory_jobs_locked()
        queued_order = {run_id: idx + 1 for idx, run_id in enumerate(RUN_QUEUE)}
        payloads = []
        for job in RUNS.values():
            if job.status not in {"queued", "running", "paused", "canceling"}:
                continue
            payload = job.to_dict()
            payload["queue_position"] = queued_order.get(job.run_id)
            payloads.append(payload)
    return sorted(payloads, key=lambda row: row["run_id"], reverse=True)


def run_queue_debug() -> Dict[str, Any]:
    with RUNS_LOCK:
        _reconcile_completed_in_memory_jobs_locked()
        queued_ids = [run_id for run_id in RUN_QUEUE if RUNS.get(run_id) is not None and RUNS[run_id].status == "queued"]
        running_jobs = [job for job in RUNS.values() if job.status == "running"]
        groups: Dict[str, Dict[str, Any]] = {}
        for job in running_jobs:
            group_id = job.progress_group_id or job.run_id
            item = groups.setdefault(
                group_id,
                {
                    "group_id": group_id,
                    "size": 0,
                    "phase_counts": {},
                    "oldest_heartbeat_at": None,
                    "sample_runs": [],
                },
            )
            item["size"] += 1
            item["phase_counts"][job.progress_phase] = int(item["phase_counts"].get(job.progress_phase, 0)) + 1
            heartbeat = job.progress_heartbeat_at
            if heartbeat and (item["oldest_heartbeat_at"] is None or heartbeat < item["oldest_heartbeat_at"]):
                item["oldest_heartbeat_at"] = heartbeat
            if len(item["sample_runs"]) < 12:
                item["sample_runs"].append(
                    {
                        "run_id": job.run_id,
                        "current": job.progress_current,
                        "total": job.progress_total,
                        "phase": job.progress_phase,
                        "detail": job.progress_detail,
                        "heartbeat_at": job.progress_heartbeat_at,
                    }
                )
        buckets: Dict[str, int] = {}
        for run_id in queued_ids:
            job = RUNS.get(run_id)
            if job is None:
                continue
            key = "|".join(_batch_group_key(job.config))
            buckets[key] = buckets.get(key, 0) + 1
        return {
            "queued_count": len(queued_ids),
            "running_count": len(running_jobs),
            "queue_worker_alive": bool(RUN_QUEUE_WORKER and RUN_QUEUE_WORKER.is_alive()),
            "queued_batch_buckets": buckets,
            "running_groups": list(groups.values()),
        }


def _build_playback_config(payload: Dict, run_id: str) -> PlaybackConfig:
    dataset_id = str(payload.get("dataset_id") or "").strip()
    dataset = get_dataset_profile(dataset_id)
    if not dataset:
        raise ValueError("Select a dataset profile from the dataset registry before starting a run.")
    mpd_path = dataset.get("mpd")
    if not mpd_path:
        raise ValueError("Selected dataset profile does not contain an MPD path.")
    content_dir = dataset.get("content_dir")
    if not content_dir:
        raise ValueError("Selected dataset profile does not contain a content/PLY directory.")

    headset_cfg = resolve_headset_config(
        payload.get("headset_preset"),
        payload.get("fov_h_deg"),
        payload.get("fov_v_deg"),
        dataset.get("angle_convention") or payload.get("angle_convention"),
    )
    raw_buffer_window = payload.get("buffer_window_segments")
    raw_segment_limit = payload.get("segment_limit")
    try:
        segment_limit = 0 if raw_segment_limit in (None, "") else int(raw_segment_limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("segment_limit must be a non-negative integer.") from exc
    if segment_limit < 0:
        raise ValueError("segment_limit must be a non-negative integer.")
    raw_qoe_warmup = payload.get("analysis_qoe_warmup_segments")
    try:
        qoe_warmup_segments = 0 if raw_qoe_warmup in (None, "") else int(raw_qoe_warmup)
    except (TypeError, ValueError) as exc:
        raise ValueError("analysis_qoe_warmup_segments must be a non-negative integer.") from exc
    if qoe_warmup_segments < 0:
        raise ValueError("analysis_qoe_warmup_segments must be a non-negative integer.")
    network_profile = (payload.get("network_profile") or None)
    network_trace_raw = str(payload.get("network_trace") or "").strip()
    if network_profile and network_profile != "None":
        if get_network_profile(network_profile) is None:
            raise ValueError(f"Unknown network profile '{network_profile}'.")
        if network_trace_raw:
            raise ValueError("Select either a network profile or a network trace, not both.")
        network_trace = None
    else:
        network_profile = None
        network_trace = safe_path(require_nonempty_path(payload, "network_trace", "Network trace"))
    raw_segment_workers_override = payload.get("segment_workers_override")
    segment_workers_override = None
    if raw_segment_workers_override not in (None, ""):
        try:
            segment_workers_override = int(raw_segment_workers_override)
        except (TypeError, ValueError) as exc:
            raise ValueError("segment_workers_override must be an integer from 0 to 16.") from exc
        if not 0 <= segment_workers_override <= 16:
            raise ValueError("segment_workers_override must be from 0 to 16.")

    return PlaybackConfig(
        mpd=safe_path(mpd_path),
        content_dir=safe_path(content_dir),
        fov_trace=safe_path(require_nonempty_path(payload, "fov_trace", "FoV trace")),
        network_trace=network_trace,
        network_profile=network_profile,
        dataset_id=dataset_id,
        abr=payload.get("abr") or "BandwidthAwareABR",
        predictor=payload.get("predictor"),
        bandwidth_predictor=payload.get("bandwidth_predictor"),
        abr_params=normalize_param_payload(payload.get("abr_params"), "abr_params"),
        predictor_params=normalize_param_payload(payload.get("predictor_params"), "predictor_params"),
        bandwidth_predictor_params=normalize_param_payload(payload.get("bandwidth_predictor_params"), "bandwidth_predictor_params"),
        abr_preset_name=(payload.get("abr_preset_name") or None),
        predictor_preset_name=(payload.get("predictor_preset_name") or None),
        bandwidth_predictor_preset_name=(payload.get("bandwidth_predictor_preset_name") or None),
        headset_preset=headset_cfg["headset_preset"],
        fov_h_deg=float(headset_cfg["fov_h_deg"]),
        fov_v_deg=float(headset_cfg["fov_v_deg"]),
        angle_convention=headset_cfg["angle_convention"],
        rtt=float(payload.get("rtt") or 0.06),
        block_size_bits=int(payload.get("block_size_bits") or 250_000),
        enable_fov_upgrades=parse_bool(payload.get("enable_fov_upgrades")),
        enable_mid_download_cancel=parse_bool(payload.get("enable_mid_download_cancel")),
        buffer_window_segments=2 if raw_buffer_window is None else int(raw_buffer_window),
        buffer_unbounded=parse_bool(payload.get("buffer_unbounded")),
        segment_limit=segment_limit,
        show_progress=False,
        output_dir=RESULTS_ROOT,
        run_id=run_id,
        record_fov_stage_latency=bool(prism_settings().get("record_fov_stage_latency")),
        ground_truth_mode=str(payload.get("ground_truth_mode") or "visible_fraction"),
        ground_truth_visible_threshold=float(payload.get("ground_truth_visible_threshold") or 0.01),
        ground_truth_peak_weight=float(payload.get("ground_truth_peak_weight") or 0.3),
        motion_limits=dict(dataset.get("motion_limits") or {}),
        oracle_mode=str(payload.get("oracle_mode") or "none"),
        oracle_group_id=(payload.get("oracle_group_id") or None),
        baseline_run_id=(payload.get("baseline_run_id") or None),
        run_group_id=(payload.get("run_group_id") or None),
        run_group_name=(payload.get("run_group_name") or None),
        run_group_source=(payload.get("run_group_source") or None),
        experiment_id=(payload.get("experiment_id") or None),
        experiment_name=(payload.get("experiment_name") or None),
        experiment_purpose=(payload.get("experiment_purpose") or None),
        experiment_kind=(payload.get("experiment_kind") or None),
        experiment_primary_metrics=[str(item) for item in (payload.get("experiment_primary_metrics") or []) if str(item).strip()],
        analysis_qoe_function=(payload.get("analysis_qoe_function") or None),
        analysis_qoe_scoring_version=(payload.get("analysis_qoe_scoring_version") or None),
        analysis_qoe_warmup_segments=qoe_warmup_segments,
        segment_workers_override=segment_workers_override,
    )


def _playback_config_from_saved(raw: Dict[str, Any], run_id: str) -> PlaybackConfig:
    field_names = {field.name for field in fields(PlaybackConfig)}
    payload = {key: value for key, value in raw.items() if key in field_names}
    payload["run_id"] = run_id
    payload["output_dir"] = RESULTS_ROOT
    for key in ("mpd", "content_dir", "fov_trace"):
        if payload.get(key) is not None:
            payload[key] = Path(payload[key])
    if payload.get("network_trace") is not None:
        payload["network_trace"] = Path(payload["network_trace"])
    return PlaybackConfig(**payload)


def _discard_incomplete_runtime_artifacts(run_dir: Path) -> None:
    """Remove artifacts that cannot be trusted after an interrupted attempt."""
    for name in INCOMPLETE_RUNTIME_FILES:
        path = run_dir / name
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    for pattern in ("*.log", "*.ndjson"):
        for path in run_dir.glob(pattern):
            path.unlink(missing_ok=True)
    artifacts_dir = run_dir / "artifacts"
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)


def _persist_recovery_state(status_file: Path, status: Dict[str, Any], *, phase: str, detail: Dict[str, Any]) -> None:
    now = _now_iso()
    progress = dict(status.get("progress") or {})
    progress.update(
        {
            "phase": phase,
            "detail": detail,
            "phase_started_at": now,
            "heartbeat_at": now,
        }
    )
    status["progress"] = progress
    status["finished"] = None
    atomic_write_json(status_file, status)


def recover_queued_runs() -> int:
    """Recover queued and interrupted runs after a server restart.

    A run without a completion marker is never accepted as finished merely
    because a process was stopped late in its output phase.  Interrupted runs
    restart from segment zero using their immutable saved configuration.
    """
    recovered: List[RunJob] = []
    if not RESULTS_ROOT.exists():
        return 0
    # A grouped cancel can be persisted by some members before another member
    # finishes its current segment step.  Preserve that group-level intent if
    # the server is stopped during the unwind, rather than restarting the
    # members whose latest progress callback overwrote ``cancel_requested``.
    canceled_group_ids: set[str] = set()
    for run_dir in RESULTS_ROOT.iterdir():
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        status = read_json(run_dir / "status.json")
        progress = status.get("progress") if isinstance(status.get("progress"), dict) else {}
        if not (bool(status.get("cancel_requested")) or str(progress.get("phase") or "").lower() == "cancel_requested"):
            continue
        config = read_json(run_dir / "config.json")
        group_id = str(config.get("run_group_id") or status.get("run_group_id") or "").strip()
        if group_id:
            canceled_group_ids.add(group_id)
    with RUN_QUEUE_CONDITION:
        existing = set(RUNS) | set(RUN_QUEUE)
        for run_dir in sorted(RESULTS_ROOT.iterdir(), key=lambda path: path.name):
            if not run_dir.is_dir() or run_dir.name.startswith("_") or run_dir.name in existing:
                continue
            status_file = run_dir / "status.json"
            config_file = run_dir / "config.json"
            if not status_file.exists() or not config_file.exists():
                continue
            status = read_json(status_file)
            previous_status = str(status.get("status") or "").lower()
            progress = status.get("progress") if isinstance(status.get("progress"), dict) else {}
            config = read_json(config_file)
            group_id = str(config.get("run_group_id") or status.get("run_group_id") or "").strip()
            # A server may be stopped while a grouped segment is still
            # unwinding.  A persisted cancel request is intentional and must
            # never be mistaken for an interrupted run to restart.
            if previous_status in {"queued", "running", "paused"} and (
                bool(status.get("cancel_requested"))
                or str(progress.get("phase") or "").lower() == "cancel_requested"
                or group_id in canceled_group_ids
            ):
                status["status"] = "canceled"
                status["error"] = "Canceled before the server stopped."
                status["finished"] = _now_iso()
                _persist_recovery_state(
                    status_file,
                    status,
                    phase="canceled",
                    detail={"recovered": True, "reason": status["error"]},
                )
                status["finished"] = _now_iso()
                atomic_write_json(status_file, status)
                continue
            legacy_complete = previous_status == "finished" and event_log_path(run_dir) is not None
            if has_completion_marker(run_dir):
                if previous_status != "finished":
                    _persist_recovery_state(
                        status_file,
                        status,
                        phase="finished",
                        detail={"recovered": True, "reason": "Completion marker verified after server restart."},
                    )
                    status["status"] = "finished"
                    status["error"] = None
                    status["finished"] = _now_iso()
                    atomic_write_json(status_file, status)
                    # The Control Center group counts come from the compact
                    # SQLite index, not a directory scan.  Keep it in sync
                    # with a completion-marker recovery so completed runs do
                    # not remain displayed as running after a restart.
                    upsert_run_directory(RESULTS_ROOT, run_dir)
                continue
            if legacy_complete:
                continue
            if previous_status not in {"queued", "running", "paused", "committing", "interrupted"}:
                continue
            interrupted = previous_status in {"running", "paused", "committing", "interrupted"}
            attempt = max(1, int(status.get("attempt") or 1)) + (1 if interrupted else 0)
            if interrupted:
                _persist_recovery_state(
                    status_file,
                    status,
                    phase="interrupted",
                    detail={
                        "reason": "Server stopped before the run completion marker was written.",
                        "previous_status": previous_status,
                        "restart_policy": "restart_from_beginning",
                    },
                )
                _discard_incomplete_runtime_artifacts(run_dir)
            try:
                config = _playback_config_from_saved(config, run_dir.name)
                job = RunJob(config)
            except Exception as exc:  # pylint: disable=broad-except
                status["status"] = "failed"
                status["error"] = f"Failed to recover queued run: {exc}"
                status["finished"] = _now_iso()
                atomic_write_json(status_file, status)
                continue
            job.attempt = attempt
            saved_progress = status.get("progress") if isinstance(status.get("progress"), dict) else {}
            job.progress_current = max(0, int(saved_progress.get("current") or 0)) if interrupted else 0
            job.progress_total = max(0, int(saved_progress.get("total") or 0)) if interrupted else 0
            job.update_progress_detail(
                "queued",
                {
                    "recovered": True,
                    "previous_status": previous_status,
                    "restart_policy": "restart_from_beginning" if interrupted else "continue_queue",
                },
                persist=False,
            )
            job._persist_status()
            RUNS[job.run_id] = job
            RUN_QUEUE.append(job.run_id)
            recovered.append(job)
        if recovered:
            _ensure_queue_worker_locked()
            RUN_QUEUE_CONDITION.notify_all()
    if recovered:
        print(f"[run-queue] recovered {len(recovered)} queued run(s) from disk")
    return len(recovered)


def _run_queue_worker() -> None:
    while True:
        with RUN_QUEUE_CONDITION:
            while not RUN_QUEUE:
                RUN_QUEUE_CONDITION.wait()
            _wait_for_queue_to_settle_locked()
            group_run_ids = _largest_batch_group_locked()
            if not group_run_ids:
                continue
            group_jobs = [RUNS[run_id] for run_id in group_run_ids if run_id in RUNS and RUNS[run_id].status == "queued"]
            RUN_QUEUE[:] = [run_id for run_id in RUN_QUEUE if run_id not in set(group_run_ids)]

        if len(group_jobs) == 1:
            group_jobs[0].run()
        else:
            _run_group_jobs(group_jobs)


def _largest_batch_group_locked() -> List[str]:
    buckets: Dict[tuple[str, str, int | None], List[str]] = {}
    order: Dict[tuple[str, str, int | None], int] = {}
    for idx, run_id in enumerate(RUN_QUEUE):
        job = RUNS.get(run_id)
        if job is None or job.status != "queued":
            continue
        key = _batch_group_key(job.config)
        buckets.setdefault(key, []).append(run_id)
        order.setdefault(key, idx)
    if not buckets:
        RUN_QUEUE[:] = [run_id for run_id in RUN_QUEUE if RUNS.get(run_id) is not None and RUNS[run_id].status == "queued"]
        return []
    best_key = max(buckets, key=lambda key: (len(buckets[key]), -order[key]))
    group = buckets[best_key]
    max_group = max(0, int(prism_settings().get("batch_group_max_runs", 0) or 0))
    return group[:max_group] if max_group > 0 else group


def _wait_for_queue_to_settle_locked() -> None:
    settings = prism_settings()
    quiet_s = max(0.0, float(settings.get("batch_queue_coalesce_ms", 250) or 0) / 1000.0)
    max_wait_s = max(quiet_s, float(settings.get("batch_queue_max_coalesce_ms", 2500) or 0) / 1000.0)
    if quiet_s <= 0 or max_wait_s <= 0:
        return

    deadline = time.monotonic() + max_wait_s
    last_queue_snapshot = tuple(RUN_QUEUE)
    while True:
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            return
        RUN_QUEUE_CONDITION.wait(timeout=min(quiet_s, remaining_s))
        next_queue_snapshot = tuple(RUN_QUEUE)
        if next_queue_snapshot == last_queue_snapshot:
            return
        last_queue_snapshot = next_queue_snapshot


def _ensure_queue_worker_locked() -> None:
    global RUN_QUEUE_WORKER  # pylint: disable=global-statement
    if RUN_QUEUE_WORKER is not None and RUN_QUEUE_WORKER.is_alive():
        return
    RUN_QUEUE_WORKER = threading.Thread(target=_run_queue_worker, name="PrismRunQueue", daemon=True)
    RUN_QUEUE_WORKER.start()


def _batch_group_key(config: PlaybackConfig) -> tuple[str, str, int | None]:
    # A fixed worker override is part of experimental semantics: do not merge
    # plan runs that require different concurrency into one grouped execution.
    return (
        str(config.mpd.resolve()),
        str(config.content_dir.resolve()),
        config.segment_workers_override,
    )


def _run_group_jobs(jobs: List[RunJob]) -> None:
    started = datetime.now().isoformat()
    group_id = f"group_{jobs[0].run_id}" if jobs else None
    for job in jobs:
        if job.status == "canceled":
            continue
        job.status = "running"
        job.started = started
        job.progress_group_id = group_id
        job.progress_group_size = len([candidate for candidate in jobs if candidate.status != "canceled"])
        job.update_progress_detail("group_starting", {"group_size": job.progress_group_size}, persist=False)
        job._persist_status()  # pylint: disable=protected-access

    active_jobs = [job for job in jobs if job.status == "running"]
    if not active_jobs:
        return
    try:
        begin_compute_session()
        first_config = active_jobs[0].config
        workers = max(1, int(prism_settings().get("batch_ply_io_workers", 1) or 1))
        configured_segment_workers = int(prism_settings().get("batch_segment_workers", 1) or 0)
        plan_segment_workers = first_config.segment_workers_override
        segment_workers = max(0, int(plan_segment_workers if plan_segment_workers is not None else configured_segment_workers))
        segment_workers_auto_cap = max(1, int(prism_settings().get("batch_segment_workers_auto_cap", 4) or 4))
        prefetch_segments = max(0, min(1, int(prism_settings().get("batch_prefetch_segments", 1) or 0)))
        print(
            "[batch-group] running "
            f"size={len(active_jobs)} mpd={first_config.mpd} content_dir={first_config.content_dir} "
            f"ply_io_workers={workers} segment_workers={'auto' if segment_workers == 0 else segment_workers} "
            f"segment_workers_source={'plan_override' if plan_segment_workers is not None else 'server_setting'} "
            f"segment_worker_cap={segment_workers_auto_cap} prefetch_segments={prefetch_segments}"
        )
        job_by_run_id = {job.run_id: job for job in active_jobs}

        def update_group_progress(run_id: str, current: int, total: int) -> None:
            target = job_by_run_id.get(run_id)
            if target is not None:
                target.update_progress(run_id, current, total)
                target._persist_status()  # pylint: disable=protected-access

        def update_group_detail(run_id: str, detail: Dict[str, Any]) -> None:
            target = job_by_run_id.get(run_id)
            if target is not None:
                phase = str(detail.get("phase") or target.progress_phase or "running")
                clean_detail = {key: value for key, value in detail.items() if key != "phase"}
                target.update_progress_detail(phase, clean_detail, persist=True)

        def apply_group_control() -> None:
            with RUN_QUEUE_CONDITION:
                if any(job.cancel_requested for job in active_jobs):
                    for job in active_jobs:
                        job.cancel_requested = True
                        job.status = "canceled"
                        job.error = "Canceled by group control."
                        job.finished = _now_iso()
                        job.update_progress_detail("canceled", {"reason": job.error}, persist=False)
                        job._persist_status()  # pylint: disable=protected-access
                    raise PlaybackGroupCanceled("Grouped run canceled.")
                while any(job.pause_requested for job in active_jobs):
                    for job in active_jobs:
                        if job.status == "running":
                            job.status = "paused"
                            job.update_progress_detail("paused", {"reason": "Paused by group control."}, persist=False)
                            job._persist_status()  # pylint: disable=protected-access
                    RUN_QUEUE_CONDITION.wait(timeout=1.0)
                    if any(job.cancel_requested for job in active_jobs):
                        for job in active_jobs:
                            job.cancel_requested = True
                            job.status = "canceled"
                            job.error = "Canceled while paused."
                            job.finished = _now_iso()
                            job.update_progress_detail("canceled", {"reason": job.error}, persist=False)
                            job._persist_status()  # pylint: disable=protected-access
                        raise PlaybackGroupCanceled("Grouped run canceled while paused.")
                for job in active_jobs:
                    if job.status == "paused":
                        job.status = "running"
                        job.update_progress_detail("running", {"reason": "Resumed by group control."}, persist=False)
                        job._persist_status()  # pylint: disable=protected-access

        heartbeat_stop = threading.Event()

        def heartbeat_loop() -> None:
            while not heartbeat_stop.wait(2.0):
                now = _now_iso()
                for job in active_jobs:
                    if job.status not in {"running", "paused"}:
                        continue
                    job.progress_heartbeat_at = now
                    if not job.progress_detail:
                        job.progress_detail = {"heartbeat": "group runner alive"}
                    job._persist_status()  # pylint: disable=protected-access

        heartbeat_thread = threading.Thread(target=heartbeat_loop, name=f"run-group-heartbeat-{group_id}", daemon=True)
        heartbeat_thread.start()
        try:
            run_playback_group(
                [job.config for job in active_jobs],
                ply_io_workers=workers,
                segment_workers=segment_workers,
                segment_workers_auto_cap=segment_workers_auto_cap,
                prefetch_segments=prefetch_segments,
                progress_callback=update_group_progress,
                progress_detail_callback=update_group_detail,
                control_callback=apply_group_control,
            )
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=3.0)
        for job in active_jobs:
            if job.status == "canceled":
                continue
            job.status = "finished"
            if navis_settings().get("auto_export"):
                from .navis_export import export_run_to_navis

                job.navis_export = export_run_to_navis(job.run_id)
    except PlaybackGroupCanceled as exc:
        for job in active_jobs:
            if job.status != "canceled":
                job.status = "canceled"
                job.error = str(exc)
    except Exception as exc:  # pylint: disable=broad-except
        cancellation_requested = any(job.cancel_requested or job.status == "canceled" for job in active_jobs)
        for job in active_jobs:
            if cancellation_requested:
                job.status = "canceled"
                job.error = "Force-canceled while terminating active compute workers."
            else:
                job.status = "failed"
                job.error = str(exc)
    finally:
        finished = datetime.now().isoformat()
        for job in active_jobs:
            job.finished = finished
            if job.status == "finished":
                job.update_progress_detail("finished", {"status": "finished"}, persist=False)
            job._persist_status()  # pylint: disable=protected-access
            if job.status == "finished":
                from .diagnostics import schedule_default_run_summary

                schedule_default_run_summary(job.run_id)


def _enqueue_config(config: PlaybackConfig) -> RunJob:
    job = RunJob(config)
    if config.run_group_id or config.run_group_name:
        assign_runs_to_group(
            {
                "id": config.run_group_id or config.run_group_name,
                "name": config.run_group_name or config.run_group_id,
                "source": config.run_group_source or "run",
                "description": config.experiment_purpose or "",
                "run_ids": [job.run_id],
            }
        )
    with RUN_QUEUE_CONDITION:
        RUNS[job.run_id] = job
        RUN_QUEUE.append(job.run_id)
        _ensure_queue_worker_locked()
        RUN_QUEUE_CONDITION.notify()
    return job


def _allocate_run_id(payload: Dict, reserved_ids: set[str] | None = None) -> str:
    """Return a run ID that cannot overwrite an existing or batch-local job."""
    reserved = reserved_ids if reserved_ids is not None else set()
    requested = str(payload.get("run_id") or "").strip()
    if requested:
        if requested in reserved or (RESULTS_ROOT / requested).exists():
            raise ValueError(f"Run id '{requested}' is already in use.")
        with RUNS_LOCK:
            if requested in RUNS:
                raise ValueError(f"Run id '{requested}' is already in use.")
        reserved.add(requested)
        return requested

    for _ in range(32):
        candidate = generate_run_id()
        if candidate in reserved or (RESULTS_ROOT / candidate).exists():
            continue
        with RUNS_LOCK:
            if candidate in RUNS:
                continue
        reserved.add(candidate)
        return candidate
    raise RuntimeError("Could not allocate a unique run id.")


def start_runs_batch(payloads: List[Dict]) -> List[RunJob]:
    jobs: List[RunJob] = []
    group_payloads: Dict[tuple[str, str, str, str], List[str]] = {}
    reserved_ids: set[str] = set()
    for payload in payloads:
        run_id = _allocate_run_id(payload, reserved_ids)
        config = _build_playback_config(payload, run_id)
        job = RunJob(config)
        jobs.append(job)
        if config.run_group_id or config.run_group_name:
            group_key = (
                str(config.run_group_id or config.run_group_name),
                str(config.run_group_name or config.run_group_id),
                str(config.run_group_source or "run"), str(config.experiment_purpose or ""),
            )
            group_payloads.setdefault(group_key, []).append(job.run_id)

    for (group_id, group_name, group_source, description), run_ids in group_payloads.items():
        assign_runs_to_group(
            {
                "id": group_id,
                "name": group_name,
                "source": group_source,
                "description": description,
                "run_ids": run_ids,
            }
        )

    with RUN_QUEUE_CONDITION:
        for job in jobs:
            RUNS[job.run_id] = job
            RUN_QUEUE.append(job.run_id)
        if jobs:
            _ensure_queue_worker_locked()
            RUN_QUEUE_CONDITION.notify_all()
    return jobs


def start_run(payload: Dict) -> RunJob:
    run_id = _allocate_run_id(payload)
    config = _build_playback_config(payload, run_id)
    return _enqueue_config(config)


def _validate_oracle_mode(base_payload: Dict, mode: str) -> None:
    if mode not in ORACLE_MODE_OVERRIDES:
        raise ValueError(f"Unknown oracle gap mode '{mode}'.")
    if mode not in ORACLE_FOV_MODES:
        return
    abr_name = str(base_payload.get("abr") or "BandwidthAwareABR")
    if mode == "offline_optimal":
        return
    abr_cls = getattr(abr_baselines, abr_name, None)
    if not abr_cls or not getattr(abr_cls, "abr_supports_predictor", False):
        raise ValueError(f"Oracle mode '{mode}' requires an ABR that supports FoV predictors.")


def start_oracle_gap_suite(payload: Dict) -> List[RunJob]:
    requested_modes = [str(mode).strip() for mode in (payload.get("oracle_gap_modes") or []) if str(mode).strip()]
    if not requested_modes:
        requested_modes = ["bandwidth", "fov_tile_scores", "bandwidth_and_fov_tile_scores", "offline_optimal"]
    requested_modes = list(dict.fromkeys(requested_modes))
    for mode in requested_modes:
        _validate_oracle_mode(payload, mode)

    baseline_run_id = payload.get("run_id") or generate_run_id()
    oracle_group_id = payload.get("oracle_group_id") or f"oracle_gap_{baseline_run_id}"
    jobs: List[RunJob] = []

    baseline_payload = dict(payload)
    baseline_payload.update(
        {
            "run_id": baseline_run_id,
            "oracle_mode": "none",
            "oracle_group_id": oracle_group_id,
            "baseline_run_id": baseline_run_id,
        }
    )
    jobs.append(_enqueue_config(_build_playback_config(baseline_payload, baseline_run_id)))

    for mode in requested_modes:
        run_id = f"{baseline_run_id}_{mode}"
        oracle_payload = dict(payload)
        oracle_payload.update(ORACLE_MODE_OVERRIDES[mode])
        oracle_payload.update(
            {
                "run_id": run_id,
                "oracle_mode": mode,
                "oracle_group_id": oracle_group_id,
                "baseline_run_id": baseline_run_id,
            }
        )
        jobs.append(_enqueue_config(_build_playback_config(oracle_payload, run_id)))

    return jobs


def forget_run(run_id: str) -> Dict[str, Any]:
    with RUN_QUEUE_CONDITION:
        job = RUNS.get(run_id)
        if job and job.status == "running":
            return {"status": "running", "run_id": run_id, "deleted": False, "error": "Running jobs cannot be deleted or canceled safely."}
        if job and job.status in {"queued", "paused"}:
            job.cancel_queued()
        if run_id in RUN_QUEUE:
            RUN_QUEUE.remove(run_id)
        RUNS.pop(run_id, None)
        RUN_QUEUE_CONDITION.notify_all()
    return {"status": "deleted", "run_id": run_id, "deleted": True}
