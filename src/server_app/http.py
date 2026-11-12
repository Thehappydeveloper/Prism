from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import time
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import pandas as pd
from core.trace_columns import normalize_network_dataframe
from core.event_storage import event_log_path
from core.run_index import ensure_run_index, query_run_group_summaries, query_run_index, remove_indexed_run
from reporting import build_report, list_report_builds, list_reports, load_report_batch_plan, update_latest_figure_font_size, validate_report
from reporting.service import GENERATED_ROOT

from .config import (
    RESULTS_ROOT,
    WEB_DIST_ROOT,
    dataset_profiles_payload,
    delete_dataset_profile,
    delete_run_form_session,
    headset_config_payload,
    navis_config_payload,
    navis_status,
    parse_bool,
    prism_config_payload,
    run_form_sessions_payload,
    safe_path,
    save_headset_settings,
    save_navis_file_settings,
    save_prism_settings,
    save_dataset_profile,
    save_run_form_session,
    save_trace_column_settings,
    save_trace_settings,
    save_world_transform,
    trace_column_config_payload,
    trace_config_payload,
    world_transform_payload,
)
from .diagnostics import METRIC_DATASETS, gather_diagnostics, gather_diagnostics_for_run_ids
from .preflight import preflight_runs
from .discovery import discover_options, resolve_qoe_model
from .extra_tools import (
    analyze_point_cloud_sequence,
    analyze_speed_acceleration,
    discover_point_cloud_sequence,
    discover_speed_accel_columns,
    inspect_ply_point_cloud,
    normalize_trace_height,
    transform_dataset,
)
from .fov_comparison import FovComparisonError, build_fov_comparison_segment, generate_fov_comparison_video
from .navis_export import export_run_to_navis
from .oracle_gap import build_oracle_gap
from .run_summaries import build_log_story, build_predictor_story, build_readable_summary
from .runs import (
    archive_run_group,
    active_runs_snapshot,
    assign_runs_to_group,
    control_run_group,
    delete_run_group,
    delete_run_group_with_runs,
    forget_run,
    list_run_groups,
    list_runs,
    load_run_cached,
    remove_runs_from_groups,
    run_queue_debug,
    start_oracle_gap_suite,
    start_run,
    start_runs_batch,
    upsert_run_group,
)
from training.export import delete_model, export_model, list_training_models
from training.plans import delete_training_plan, save_training_plan, training_plans_payload
from training.registry import delete_training_dataset, save_training_dataset, training_datasets_payload
from training.trainer import delete_training_run, evaluate_model, job_status, list_training_runs, start_training_run, training_results_payload, control_training_run


DIAGNOSTICS_CACHE: dict[str, dict[str, Any]] = {}
DIAGNOSTICS_CACHE_MAX_BYTES = 96 * 1024 * 1024
DIAGNOSTICS_CACHE_BYTES = 0
DIAGNOSTICS_DISK_CACHE_VERSION = 4
DIAGNOSTICS_DISK_CACHE_ROOT = RESULTS_ROOT.parent / ".prism_cache" / "graphs" / "diagnostics"
TRACE_SAMPLE_CACHE: dict[str, dict[str, Any]] = {}
TRACE_SAMPLE_CACHE_MAX_ENTRIES = 64
TRACE_SAMPLE_DISK_CACHE_VERSION = 2
TRACE_SAMPLE_DISK_CACHE_ROOT = RESULTS_ROOT.parent / ".prism_cache" / "graphs" / "traces"
ORACLE_GAP_DISK_CACHE_VERSION = 1
ORACLE_GAP_DISK_CACHE_ROOT = RESULTS_ROOT.parent / ".prism_cache" / "graphs" / "oracle_gap"


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _cache_digest(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _run_file_fingerprint(run_id: str) -> dict[str, Any]:
    run_dir = RESULTS_ROOT / run_id
    files = {}
    for name in ("config.json", "events.json", "predictor_diagnostics.json"):
        path = event_log_path(run_dir) if name == "events.json" else run_dir / name
        if path is None:
            files[name] = None
            continue
        if not path.exists():
            files[name] = None
            continue
        stat = path.stat()
        files[name] = {"path": str(path), "mtime_ns": stat.st_mtime_ns, "size": stat.st_size}
    return {"run_id": run_id, "files": files}


def _diagnostics_cache_key(run_ids: list[str], payload: dict[str, Any]) -> str:
    return _stable_json(
        {
            "run_ids": sorted(str(run_id) for run_id in run_ids),
            "qoe_function": payload.get("qoe_function"),
            "qoe_params": payload.get("qoe_params") or {},
            "datasets": sorted(str(value) for value in (payload.get("datasets") or [])),
            "runs": [_run_file_fingerprint(str(run_id)) for run_id in sorted(run_ids)],
        }
    )


def _remember_diagnostics(key: str, metrics: dict[str, Any]) -> None:
    global DIAGNOSTICS_CACHE_BYTES
    size_bytes = len(json.dumps(metrics, separators=(",", ":"), default=str).encode("utf-8"))
    previous = DIAGNOSTICS_CACHE.pop(key, None) or {}
    DIAGNOSTICS_CACHE_BYTES -= int(previous.get("size_bytes", 0))
    while DIAGNOSTICS_CACHE and DIAGNOSTICS_CACHE_BYTES + size_bytes > DIAGNOSTICS_CACHE_MAX_BYTES:
        oldest_key = next(iter(DIAGNOSTICS_CACHE))
        removed = DIAGNOSTICS_CACHE.pop(oldest_key, None) or {}
        DIAGNOSTICS_CACHE_BYTES -= int(removed.get("size_bytes", 0))
        print(f"[graph-cache] diagnostics invalidate oldest key={oldest_key[:12]}")
    DIAGNOSTICS_CACHE[key] = {"metrics": metrics, "created": time.time(), "size_bytes": size_bytes}
    DIAGNOSTICS_CACHE_BYTES += size_bytes


def _diagnostics_disk_cache_path(key: str) -> Path:
    return DIAGNOSTICS_DISK_CACHE_ROOT / f"{_cache_digest(key)}.json"


def _load_diagnostics_from_disk(key: str) -> dict[str, Any] | None:
    started = time.perf_counter()
    path = _diagnostics_disk_cache_path(key)
    if not path.exists():
        print(f"[graph-cache] diagnostics disk_miss elapsed_ms={int((time.perf_counter() - started) * 1000)}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[graph-cache] diagnostics disk_read_failed path={path} error={exc}")
        return None
    if payload.get("version") != DIAGNOSTICS_DISK_CACHE_VERSION or payload.get("key") != key:
        print(f"[graph-cache] diagnostics disk_stale path={path} elapsed_ms={int((time.perf_counter() - started) * 1000)}")
        return None
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return None
    _remember_diagnostics(key, metrics)
    print(f"[graph-cache] diagnostics disk_hit path={path} elapsed_ms={int((time.perf_counter() - started) * 1000)}")
    return metrics


def _write_diagnostics_to_disk(key: str, metrics: dict[str, Any]) -> None:
    started = time.perf_counter()
    path = _diagnostics_disk_cache_path(key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(
                {
                    "version": DIAGNOSTICS_DISK_CACHE_VERSION,
                    "key": key,
                    "created": time.time(),
                    "metrics": metrics,
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        tmp_path.replace(path)
        print(f"[graph-cache] diagnostics disk_write path={path} elapsed_ms={int((time.perf_counter() - started) * 1000)}")
    except OSError as exc:
        print(f"[graph-cache] diagnostics disk_write_failed path={path} error={exc}")


def _oracle_gap_cache_key(run_ids: list[str], payload: dict[str, Any]) -> str:
    return _stable_json(
        {
            "run_ids": sorted(str(run_id) for run_id in run_ids),
            "qoe_function": payload.get("qoe_function"),
            "qoe_params": payload.get("qoe_params") or {},
            "runs": [_run_file_fingerprint(str(run_id)) for run_id in sorted(run_ids)],
        }
    )


def _oracle_gap_disk_cache_path(key: str) -> Path:
    return ORACLE_GAP_DISK_CACHE_ROOT / f"{_cache_digest(key)}.json"


def _load_oracle_gap_from_disk(key: str) -> dict[str, Any] | None:
    started = time.perf_counter()
    path = _oracle_gap_disk_cache_path(key)
    if not path.exists():
        print(f"[graph-cache] oracle_gap disk_miss elapsed_ms={int((time.perf_counter() - started) * 1000)}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[graph-cache] oracle_gap disk_read_failed path={path} error={exc}")
        return None
    if payload.get("version") != ORACLE_GAP_DISK_CACHE_VERSION or payload.get("key") != key:
        print(f"[graph-cache] oracle_gap disk_stale path={path} elapsed_ms={int((time.perf_counter() - started) * 1000)}")
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    print(f"[graph-cache] oracle_gap disk_hit path={path} elapsed_ms={int((time.perf_counter() - started) * 1000)}")
    return data


def _write_oracle_gap_to_disk(key: str, data: dict[str, Any]) -> None:
    started = time.perf_counter()
    path = _oracle_gap_disk_cache_path(key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(
                {
                    "version": ORACLE_GAP_DISK_CACHE_VERSION,
                    "key": key,
                    "created": time.time(),
                    "data": data,
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        tmp_path.replace(path)
        print(f"[graph-cache] oracle_gap disk_write path={path} elapsed_ms={int((time.perf_counter() - started) * 1000)}")
    except OSError as exc:
        print(f"[graph-cache] oracle_gap disk_write_failed path={path} error={exc}")


def _file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "mtime_ns": stat.st_mtime_ns, "size": stat.st_size}


def _trace_cache_key(path: Path) -> str:
    return _stable_json({"version": TRACE_SAMPLE_DISK_CACHE_VERSION, "trace": _file_fingerprint(path)})


def _trace_disk_cache_path(key: str) -> Path:
    return TRACE_SAMPLE_DISK_CACHE_ROOT / f"{_cache_digest(key)}.json"


def _remember_trace_samples(key: str, samples: list[dict[str, float]]) -> None:
    while len(TRACE_SAMPLE_CACHE) >= TRACE_SAMPLE_CACHE_MAX_ENTRIES:
        TRACE_SAMPLE_CACHE.pop(next(iter(TRACE_SAMPLE_CACHE)), None)
    TRACE_SAMPLE_CACHE[key] = {"samples": samples, "created": time.time()}


def _load_trace_samples_from_disk(key: str) -> list[dict[str, float]] | None:
    started = time.perf_counter()
    path = _trace_disk_cache_path(key)
    if not path.exists():
        print(f"[graph-cache] trace_samples disk_miss elapsed_ms={int((time.perf_counter() - started) * 1000)}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[graph-cache] trace_samples disk_read_failed path={path} error={exc}")
        return None
    if payload.get("version") != TRACE_SAMPLE_DISK_CACHE_VERSION or payload.get("key") != key:
        print(f"[graph-cache] trace_samples disk_stale path={path} elapsed_ms={int((time.perf_counter() - started) * 1000)}")
        return None
    samples = payload.get("samples")
    if not isinstance(samples, list):
        return None
    _remember_trace_samples(key, samples)
    print(f"[graph-cache] trace_samples disk_hit path={path} samples={len(samples)} elapsed_ms={int((time.perf_counter() - started) * 1000)}")
    return samples


def _write_trace_samples_to_disk(key: str, samples: list[dict[str, float]]) -> None:
    started = time.perf_counter()
    path = _trace_disk_cache_path(key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(
                {
                    "version": TRACE_SAMPLE_DISK_CACHE_VERSION,
                    "key": key,
                    "created": time.time(),
                    "samples": samples,
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        tmp_path.replace(path)
        print(f"[graph-cache] trace_samples disk_write path={path} samples={len(samples)} elapsed_ms={int((time.perf_counter() - started) * 1000)}")
    except OSError as exc:
        print(f"[graph-cache] trace_samples disk_write_failed path={path} error={exc}")


class RequestHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @staticmethod
    def _web_entrypoint() -> Path | None:
        dist_index = WEB_DIST_ROOT / "index.html"
        if dist_index.exists():
            return dist_index
        return None

    @staticmethod
    def _resolve_static_asset(path_only: str) -> Path:
        rel = path_only.replace("/static/", "", 1)
        dist_path = (WEB_DIST_ROOT / rel).resolve()
        try:
            dist_path.relative_to(WEB_DIST_ROOT.resolve())
        except ValueError:
            dist_path = WEB_DIST_ROOT / rel
        return dist_path

    @staticmethod
    def _is_web_asset(path: Path) -> bool:
        try:
            path.resolve().relative_to(WEB_DIST_ROOT.resolve())
            return True
        except ValueError:
            return False

    def _read_json_body(self) -> Any:
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else b"{}"
        return json.loads(body.decode("utf-8") or "{}")

    def _serve_web_build_missing(self):
        message = (
            "Web UI build not found at src/web/dist.\n"
            "Build the frontend with:\n"
            "  cd src/web\n"
            "  npm ci\n"
            "  npm run build\n"
        )
        data = message.encode("utf-8")
        self.send_response(503)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path_only = self.path.split("?", 1)[0]
        query = parse_qs(urlparse(self.path).query)
        if path_only == "/":
            entrypoint = self._web_entrypoint()
            if entrypoint is None:
                return self._serve_web_build_missing()
            return self._serve_file(entrypoint)
        if path_only.startswith("/static/"):
            if path_only in (
                "/static/home.html",
                "/static/index.html",
                "/static/aggregate.html",
                "/static/compare.html",
                "/static/reports.html",
                "/static/datasets.html",
                "/static/extra.html",
                "/static/speed-acceleration.html",
                "/static/point-cloud-sequence-analyzer.html",
                "/static/dataset-transformation.html",
                "/static/trace-height-normalizer.html",
                "/static/settings.html",
                "/static/training.html",
            ):
                entrypoint = self._web_entrypoint()
                if entrypoint is None:
                    return self._serve_web_build_missing()
                return self._serve_file(entrypoint)
            return self._serve_file(self._resolve_static_asset(path_only))
        if path_only.startswith("/reports/"):
            report_path = (GENERATED_ROOT / path_only.removeprefix("/reports/")).resolve()
            try:
                report_path.relative_to(GENERATED_ROOT.resolve())
            except ValueError:
                return self._respond_json({"error": "Invalid report path"}, status=400)
            return self._serve_file(report_path)
        if path_only.startswith("/results/"):
            return self._serve_file(RESULTS_ROOT / path_only.replace("/results/", ""))
        if path_only == "/api/options":
            return self._respond_json(discover_options())
        if path_only == "/api/navis/config":
            return self._respond_json(navis_config_payload())
        if path_only == "/api/prism/config":
            return self._respond_json(prism_config_payload())
        if path_only == "/api/headsets/config":
            return self._respond_json(headset_config_payload())
        if path_only == "/api/traces/config":
            return self._respond_json(trace_config_payload())
        if path_only == "/api/trace-columns/config":
            return self._respond_json(trace_column_config_payload())
        if path_only == "/api/world-transforms":
            return self._respond_json(world_transform_payload((query.get("content_dir") or [None])[0]))
        if path_only == "/api/datasets":
            return self._respond_json(dataset_profiles_payload())
        if path_only == "/api/training/datasets":
            return self._respond_json(training_datasets_payload())
        if path_only == "/api/training/plans":
            return self._respond_json(training_plans_payload())
        if path_only == "/api/training/runs":
            return self._respond_json(list_training_runs())
        if path_only.startswith("/api/training/runs/") and path_only.endswith("/metrics"):
            run_id = path_only.split("/")[4]
            metrics_file = RESULTS_ROOT / "training" / "runs" / run_id / "metrics.json"
            if metrics_file.exists():
                return self._serve_file(metrics_file, content_type="application/json")
            return self._respond_json({"error": "not found"}, status=404)
        if path_only.startswith("/api/training/runs/") and path_only.endswith("/results"):
            run_id = path_only.split("/")[4]
            return self._respond_json(training_results_payload(run_id))
        if path_only.startswith("/api/training/runs/"):
            run_id = path_only.split("/")[4]
            return self._respond_json(job_status(run_id))
        if path_only == "/api/training/models":
            return self._respond_json({"models": list_training_models()})
        if path_only == "/api/run-form-sessions":
            return self._respond_json(run_form_sessions_payload())
        if path_only == "/api/navis/status":
            return self._respond_json(navis_status())
        if path_only == "/api/runs":
            if any(key in query for key in ("limit", "offset", "search", "status", "group_id", "ungrouped")):
                ensure_run_index(RESULTS_ROOT)
                try:
                    limit = int((query.get("limit") or [200])[0])
                    offset = int((query.get("offset") or [0])[0])
                except ValueError:
                    return self._respond_json({"error": "limit and offset must be integers"}, status=400)
                items, total = query_run_index(
                    RESULTS_ROOT,
                    limit=limit, offset=offset,
                    search=str((query.get("search") or [""])[0]), status=str((query.get("status") or [""])[0]),
                    group_id=str((query.get("group_id") or [""])[0]),
                    ungrouped=str((query.get("ungrouped") or [""])[0]).lower() in {"1", "true", "yes"},
                )
                return self._respond_json({"items": items, "total": total, "offset": offset, "limit": limit})
            return self._respond_json(list_runs())
        if path_only == "/api/runs/group-summaries":
            ensure_run_index(RESULTS_ROOT)
            return self._respond_json({"groups": query_run_group_summaries(
                RESULTS_ROOT,
                search=str((query.get("search") or [""])[0]),
                status=str((query.get("status") or [""])[0]),
            )})
        if path_only == "/api/reports":
            return self._respond_json({"reports": list_reports()})
        if path_only.startswith("/api/reports/") and path_only.endswith("/builds"):
            report_id = unquote(path_only.split("/")[3])
            try:
                return self._respond_json({"builds": list_report_builds(report_id)})
            except ValueError as exc:
                return self._respond_json({"error": str(exc)}, status=404)
        if path_only.startswith("/api/reports/") and "/batch-plans/" in path_only:
            parts = path_only.split("/")
            if len(parts) != 6:
                return self._respond_json({"error": "Unsupported report batch-plan endpoint"}, status=404)
            try:
                return self._respond_json(load_report_batch_plan(unquote(parts[3]), unquote(parts[5])))
            except ValueError as exc:
                return self._respond_json({"error": str(exc)}, status=404)
        if path_only == "/api/runs/debug":
            return self._respond_json(run_queue_debug())
        if path_only == "/api/run-groups":
            return self._respond_json({"groups": list_run_groups()})
        if path_only == "/api/runs/stream":
            return self._serve_runs_stream()
        if path_only.startswith("/api/runs/") and path_only.endswith("/events"):
            run_id = path_only.split("/")[3]
            events_file = event_log_path(RESULTS_ROOT / run_id)
            if events_file is not None:
                return self._serve_file(
                    events_file,
                    content_type="application/json",
                    content_encoding="gzip" if events_file.name.endswith(".gz") else None,
                )
            return self._respond_json({"error": "not found"}, status=404)
        if path_only.startswith("/api/runs/") and path_only.endswith("/config"):
            run_id = path_only.split("/")[3]
            cfg_file = RESULTS_ROOT / run_id / "config.json"
            if cfg_file.exists():
                return self._serve_file(cfg_file, content_type="application/json")
            return self._respond_json({"error": "not found"}, status=404)
        if path_only.startswith("/api/runs/") and path_only.endswith("/readable-summary"):
            run_id = path_only.split("/")[3]
            run_dir = RESULTS_ROOT / run_id
            if run_dir.exists() and run_dir.is_dir():
                return self._respond_json(build_readable_summary(run_dir, run_id))
            return self._respond_json({"error": "not found"}, status=404)
        if path_only.startswith("/api/runs/") and path_only.endswith("/predictor-story"):
            run_id = path_only.split("/")[3]
            run_dir = RESULTS_ROOT / run_id
            if run_dir.exists() and run_dir.is_dir():
                return self._respond_json(build_predictor_story(run_dir))
            return self._respond_json({"error": "not found"}, status=404)
        if path_only.startswith("/api/runs/") and path_only.endswith("/log-story"):
            run_id = path_only.split("/")[3]
            run_dir = RESULTS_ROOT / run_id
            if run_dir.exists() and run_dir.is_dir():
                return self._respond_json(build_log_story(run_dir, run_id))
            return self._respond_json({"error": "not found"}, status=404)
        if path_only.startswith("/api/runs/") and (path_only.endswith("/predictor-diagnostics") or path_only.endswith("/fov-latency")):
            run_id = path_only.split("/")[3]
            diagnostics_file = RESULTS_ROOT / run_id / "predictor_diagnostics.json"
            if diagnostics_file.exists():
                return self._serve_file(diagnostics_file, content_type="application/json")
            return self._respond_json({"error": "not found"}, status=404)
        if self.path.startswith("/api/trace?path="):
            try:
                return self._serve_trace(self.path.split("path=")[-1])
            except Exception as exc:  # pylint: disable=broad-except
                return self._respond_json({"error": str(exc)}, status=400)
        return super().do_GET()

    def do_POST(self):
        path_only = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path_only == "/api/preflight":
            payload = self._read_json_body()
            raw_runs = payload.get("runs") if isinstance(payload, dict) else None
            if not isinstance(raw_runs, list) or not raw_runs:
                return self._respond_json({"error": "Preflight requires a non-empty runs list."}, status=400)
            return self._respond_json(preflight_runs([item for item in raw_runs if isinstance(item, dict)]))
        if path_only.startswith("/api/reports/"):
            parts = path_only.split("/")
            if len(parts) != 5 or parts[4] not in {"validate", "build", "svg-font-size"}:
                return self._respond_json({"error": "Unsupported report endpoint"}, status=404)
            report_id, action = unquote(parts[3]), parts[4]
            payload = self._read_json_body()
            try:
                if action == "validate":
                    result = validate_report(report_id, RESULTS_ROOT)
                elif action == "svg-font-size":
                    result = update_latest_figure_font_size(report_id, float((payload or {}).get("font_size_pt")))
                else:
                    result = build_report(report_id, RESULTS_ROOT, payload.get("render_options") if isinstance(payload, dict) else None)
            except ValueError as exc:
                return self._respond_json({"error": str(exc)}, status=404)
            except Exception as exc:  # pylint: disable=broad-except
                # A report renderer may fail while loading a newly completed
                # cohort.  Return its diagnostic to the Reports page instead
                # of closing the request and surfacing only "Failed to fetch".
                return self._respond_json({"error": f"Report {action} failed: {exc}"}, status=500)
            return self._respond_json(result, status=400 if result.get("status") == "error" else 200)
        if path_only == "/api/navis/config":
            payload = self._read_json_body()
            try:
                save_navis_file_settings(payload.get("settings", payload))
            except ValueError as exc:
                return self._respond_json({"error": str(exc)}, status=400)
            return self._respond_json(navis_config_payload())
        if path_only == "/api/headsets/config":
            payload = self._read_json_body()
            try:
                save_headset_settings(payload.get("settings", payload))
            except ValueError as exc:
                return self._respond_json({"error": str(exc)}, status=400)
            return self._respond_json(headset_config_payload())
        if path_only == "/api/prism/config":
            payload = self._read_json_body()
            try:
                save_prism_settings(payload.get("settings", payload))
            except ValueError as exc:
                return self._respond_json({"error": str(exc)}, status=400)
            return self._respond_json(prism_config_payload())
        if path_only == "/api/traces/config":
            payload = self._read_json_body()
            try:
                save_trace_settings(payload.get("settings", payload))
            except ValueError as exc:
                return self._respond_json({"error": str(exc)}, status=400)
            return self._respond_json(trace_config_payload())
        if path_only == "/api/trace-columns/config":
            payload = self._read_json_body()
            try:
                save_trace_column_settings(payload.get("settings", payload))
            except ValueError as exc:
                return self._respond_json({"error": str(exc)}, status=400)
            return self._respond_json(trace_column_config_payload())
        if path_only == "/api/world-transforms":
            try:
                return self._respond_json(save_world_transform(self._read_json_body()))
            except ValueError as exc:
                return self._respond_json({"error": str(exc)}, status=400)
        if path_only == "/api/datasets":
            try:
                return self._respond_json(save_dataset_profile(self._read_json_body()))
            except ValueError as exc:
                return self._respond_json({"error": str(exc)}, status=400)
        if path_only == "/api/training/datasets":
            try:
                return self._respond_json(save_training_dataset(self._read_json_body()))
            except ValueError as exc:
                return self._respond_json({"error": str(exc)}, status=400)
        if path_only == "/api/training/plans":
            try:
                return self._respond_json(save_training_plan(self._read_json_body()))
            except ValueError as exc:
                return self._respond_json({"error": str(exc)}, status=400)
        if path_only == "/api/training/runs":
            payload = self._read_json_body()
            try:
                return self._respond_json(start_training_run(str(payload.get("plan_id") or ""), str(payload.get("resume_run_id") or "") or None))
            except Exception as exc:  # pylint: disable=broad-except
                return self._respond_json({"error": str(exc)}, status=400)
        if path_only.startswith("/api/training/runs/") and path_only.endswith("/control"):
            run_id = path_only.split("/")[4]
            payload = self._read_json_body()
            try:
                return self._respond_json(control_training_run(run_id, str(payload.get("action") or "")))
            except ValueError as exc:
                return self._respond_json({"error": str(exc)}, status=400)
        if path_only == "/api/training/export":
            payload = self._read_json_body()
            try:
                return self._respond_json(export_model(str(payload.get("model_id") or ""), str(payload.get("export_name") or "") or None))
            except ValueError as exc:
                return self._respond_json({"error": str(exc)}, status=400)
        if path_only == "/api/training/evaluate":
            payload = self._read_json_body()
            try:
                return self._respond_json(evaluate_model(str(payload.get("model_id") or ""), str(payload.get("split") or "test")))
            except Exception as exc:  # pylint: disable=broad-except
                return self._respond_json({"error": str(exc)}, status=400)
        if path_only == "/api/run-form-sessions":
            try:
                save_run_form_session(self._read_json_body())
            except ValueError as exc:
                return self._respond_json({"error": str(exc)}, status=400)
            return self._respond_json(run_form_sessions_payload())
        if path_only == "/api/navis/export":
            payload = self._read_json_body()
            run_id = payload.get("run_id")
            if not run_id:
                return self._respond_json({"error": "run_id is required"}, status=400)
            result = export_run_to_navis(
                run_id=str(run_id),
                base_url=payload.get("base_url"),
                export_path=payload.get("export_path"),
                health_path=payload.get("health_path"),
            )
            return self._respond_json(result, status=404 if result.get("error") == "Run not found." else 200)
        if path_only.startswith("/api/runs/") and path_only.endswith("/fov-comparison-segment"):
            run_id = path_only.split("/")[3]
            payload = self._read_json_body()
            try:
                return self._respond_json(build_fov_comparison_segment(str(run_id), int(payload.get("segment_idx"))))
            except (TypeError, ValueError, FovComparisonError) as exc:
                return self._respond_json({"error": str(exc)}, status=400)
        if path_only.startswith("/api/runs/") and path_only.endswith("/fov-comparison"):
            run_id = path_only.split("/")[3]
            payload = self._read_json_body()
            result = generate_fov_comparison_video(run_id=str(run_id), options=payload.get("options") if isinstance(payload, dict) else None)
            return self._respond_json(result)
        if path_only == "/api/extra/speed-acceleration/discover":
            payload = self._read_json_body()
            try:
                return self._respond_json(discover_speed_accel_columns(str(payload.get("root_folder") or "")))
            except ValueError as exc:
                return self._respond_json({"error": str(exc)}, status=400)
        if path_only == "/api/extra/speed-acceleration/inspect-ply":
            payload = self._read_json_body()
            try:
                return self._respond_json(inspect_ply_point_cloud(str(payload.get("ply_path") or "")))
            except ValueError as exc:
                return self._respond_json({"error": str(exc)}, status=400)
        if path_only == "/api/extra/speed-acceleration/analyze":
            payload = self._read_json_body()
            try:
                return self._respond_json(
                    analyze_speed_acceleration(
                        root_folder=str(payload.get("root_folder") or ""),
                        x_column=str(payload.get("x_column") or ""),
                        y_column=str(payload.get("y_column") or ""),
                        z_column=str(payload.get("z_column") or ""),
                        fps=float(payload.get("fps") or 0),
                        ply_path=str(payload.get("ply_path") or ""),
                        height_axis=str(payload.get("height_axis") or ""),
                        real_height_m=float(payload.get("real_height_m") or 0),
                        filter_outliers=parse_bool(payload.get("filter_outliers", True)),
                        filter_method=str(payload.get("filter_method") or "iqr"),
                        percentile_keep=float(payload.get("percentile_keep") or 99.0),
                        histogram_source=str(payload.get("histogram_source") or "filtered"),
                        exclude_suspicious_jumps=parse_bool(payload.get("exclude_suspicious_jumps", True)),
                    )
                )
            except ValueError as exc:
                return self._respond_json({"error": str(exc)}, status=400)
        if path_only == "/api/extra/point-cloud-sequence/discover":
            payload = self._read_json_body()
            try:
                return self._respond_json(
                    discover_point_cloud_sequence(
                        root_folder=str(payload.get("root_folder") or ""),
                        recursive=parse_bool(payload.get("recursive", True)),
                    )
                )
            except ValueError as exc:
                return self._respond_json({"error": str(exc)}, status=400)
        if path_only == "/api/extra/point-cloud-sequence/analyze":
            payload = self._read_json_body()
            try:
                return self._respond_json(
                    analyze_point_cloud_sequence(
                        root_folder=str(payload.get("root_folder") or ""),
                        recursive=parse_bool(payload.get("recursive", True)),
                        segment_size=int(payload.get("segment_size") or 30),
                        aggregation=str(payload.get("aggregation") or "mean"),
                        voxel_bits=int(payload.get("voxel_bits") or 10),
                        use_native_voxels=parse_bool(payload.get("use_native_voxels", False)),
                        max_points_for_qp=int(payload.get("max_points_for_qp") or 120000),
                    )
                )
            except ValueError as exc:
                return self._respond_json({"error": str(exc)}, status=400)
        if path_only == "/api/extra/trace-height-normalizer/discover":
            payload = self._read_json_body()
            try:
                return self._respond_json(discover_speed_accel_columns(str(payload.get("root_folder") or "")))
            except ValueError as exc:
                return self._respond_json({"error": str(exc)}, status=400)
        if path_only == "/api/extra/dataset-transformation/discover":
            payload = self._read_json_body()
            try:
                return self._respond_json(discover_speed_accel_columns(str(payload.get("root_folder") or "")))
            except ValueError as exc:
                return self._respond_json({"error": str(exc)}, status=400)
        if path_only == "/api/extra/trace-height-normalizer/inspect-ply":
            payload = self._read_json_body()
            try:
                return self._respond_json(inspect_ply_point_cloud(str(payload.get("ply_path") or "")))
            except ValueError as exc:
                return self._respond_json({"error": str(exc)}, status=400)
        if path_only == "/api/extra/dataset-transformation/inspect-ply":
            payload = self._read_json_body()
            try:
                return self._respond_json(inspect_ply_point_cloud(str(payload.get("ply_path") or "")))
            except ValueError as exc:
                return self._respond_json({"error": str(exc)}, status=400)
        if path_only == "/api/extra/trace-height-normalizer/normalize":
            payload = self._read_json_body()
            try:
                return self._respond_json(
                    normalize_trace_height(
                        root_folder=str(payload.get("root_folder") or ""),
                        output_folder=str(payload.get("output_folder") or ""),
                        x_column=str(payload.get("x_column") or ""),
                        y_column=str(payload.get("y_column") or ""),
                        z_column=str(payload.get("z_column") or ""),
                        ply_path=str(payload.get("ply_path") or ""),
                        height_axis=str(payload.get("height_axis") or ""),
                        real_height_m=float(payload.get("real_height_m") or 0),
                        output_height_axis=str(payload.get("output_height_axis") or "") or None,
                    )
                )
            except ValueError as exc:
                return self._respond_json({"error": str(exc)}, status=400)
        if path_only == "/api/extra/dataset-transformation/transform":
            payload = self._read_json_body()
            try:
                voxel_raw = payload.get("voxel_bits")
                voxel_bits = int(voxel_raw) if str(voxel_raw or "").strip() else None
                resample_raw = payload.get("fov_resample_target")
                fov_resample_target = int(resample_raw) if str(resample_raw or "").strip() else None
                return self._respond_json(
                    transform_dataset(
                        trace_root_folder=str(payload.get("trace_root_folder") or payload.get("root_folder") or ""),
                        ply_input_path=str(payload.get("ply_input_path") or payload.get("ply_path") or ""),
                        output_folder=str(payload.get("output_folder") or ""),
                        x_column=str(payload.get("x_column") or ""),
                        y_column=str(payload.get("y_column") or ""),
                        z_column=str(payload.get("z_column") or ""),
                        reference_ply_path=str(payload.get("reference_ply_path") or payload.get("ply_path") or ""),
                        height_axis=str(payload.get("height_axis") or "y"),
                        real_height_m=float(payload.get("real_height_m") or 0),
                        output_height_axis=str(payload.get("output_height_axis") or "") or None,
                        transform_traces=parse_bool(payload.get("transform_traces", True)),
                        transform_point_clouds=parse_bool(payload.get("transform_point_clouds", False)),
                        height_axis_starts_at_zero=parse_bool(payload.get("height_axis_starts_at_zero", True)),
                        positive_coordinates=parse_bool(payload.get("positive_coordinates", False)),
                        voxel_bits=voxel_bits,
                        export_mode=str(payload.get("export_mode") or "normalized"),
                        worker_count=int(payload.get("worker_count") or 1),
                        fov_resample_enabled=parse_bool(payload.get("fov_resample_enabled", False)),
                        fov_resample_target=fov_resample_target,
                    )
                )
            except ValueError as exc:
                return self._respond_json({"error": str(exc)}, status=400)
        if path_only == "/api/diagnostics/fingerprint":
            payload = self._read_json_body()
            requested_run_ids = [str(rid) for rid in (payload.get("run_ids") or [])]
            if requested_run_ids:
                resolved_run_ids = requested_run_ids
            elif RESULTS_ROOT.exists():
                resolved_run_ids = sorted(path.name for path in RESULTS_ROOT.iterdir() if path.is_dir())
            else:
                resolved_run_ids = []
            cache_key = _diagnostics_cache_key(resolved_run_ids, payload)
            return self._respond_json({"fingerprint": f"v{DIAGNOSTICS_DISK_CACHE_VERSION}:{_cache_digest(cache_key)}"})
        if path_only == "/api/diagnostics":
            request_started = time.perf_counter()
            payload = self._read_json_body()
            requested_datasets = {
                str(value) for value in (payload.get("datasets") or METRIC_DATASETS) if str(value) in METRIC_DATASETS
            }
            requested_run_ids = [str(rid) for rid in (payload.get("run_ids") or [])]
            qoe_model = resolve_qoe_model(payload.get("qoe_function"), payload.get("qoe_params") or {})
            if requested_run_ids:
                resolved_run_ids = requested_run_ids
            elif RESULTS_ROOT.exists():
                resolved_run_ids = sorted(p.name for p in RESULTS_ROOT.iterdir() if p.is_dir())
            else:
                resolved_run_ids = []

            key_started = time.perf_counter()
            cache_key = _diagnostics_cache_key(resolved_run_ids, payload)
            key_ms = int((time.perf_counter() - key_started) * 1000)
            cached = DIAGNOSTICS_CACHE.get(cache_key)
            if cached:
                age_ms = int((time.time() - float(cached.get("created", 0))) * 1000)
                print(f"[graph-cache] diagnostics memory_hit runs={len(resolved_run_ids)} age_ms={age_ms} key_ms={key_ms}")
                return self._respond_json(cached["metrics"])
            disk_metrics = _load_diagnostics_from_disk(cache_key)
            if disk_metrics is not None:
                print(
                    "[graph-timing] diagnostics served_from_disk "
                    f"runs={len(resolved_run_ids)} key_ms={key_ms} elapsed_ms={int((time.perf_counter() - request_started) * 1000)}"
                )
                return self._respond_json(disk_metrics)

            started = time.perf_counter()
            print(f"[graph-cache] diagnostics miss runs={len(resolved_run_ids)} key_ms={key_ms}")
            existing_run_ids = [
                rid for rid in resolved_run_ids
                if (RESULTS_ROOT / rid / "config.json").exists() and event_log_path(RESULTS_ROOT / rid) is not None
            ]
            if not existing_run_ids:
                return self._respond_json({"error": "no runs found"}, status=404)
            metrics_started = time.perf_counter()
            metrics = gather_diagnostics_for_run_ids(existing_run_ids, qoe_model, datasets=requested_datasets)
            metrics_ms = int((time.perf_counter() - metrics_started) * 1000)
            _remember_diagnostics(cache_key, metrics)
            _write_diagnostics_to_disk(cache_key, metrics)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            print(
                "[graph-timing] diagnostics generated "
                f"runs={len(existing_run_ids)} datasets={sorted(requested_datasets)} metrics_ms={metrics_ms} elapsed_ms={elapsed_ms}"
            )
            return self._respond_json(metrics)
        if path_only == "/api/oracle-gap":
            request_started = time.perf_counter()
            payload = self._read_json_body()
            requested_run_ids = [str(rid) for rid in (payload.get("run_ids") or [])]
            group_id = str(payload.get("oracle_group_id") or "").strip()
            if requested_run_ids:
                resolved_run_ids = requested_run_ids
            elif group_id:
                resolved_run_ids = [
                    str(run.get("run_id"))
                    for run in list_runs()
                    if str(run.get("oracle_group_id") or "") == group_id
                ]
            else:
                resolved_run_ids = [
                    str(run.get("run_id"))
                    for run in list_runs()
                    if run.get("oracle_group_id")
                ]
            if not resolved_run_ids:
                return self._respond_json({"error": "no oracle gap runs found"}, status=404)

            qoe_model = resolve_qoe_model(payload.get("qoe_function"), payload.get("qoe_params") or {})
            cache_key = _oracle_gap_cache_key(resolved_run_ids, payload)
            disk_data = _load_oracle_gap_from_disk(cache_key)
            if disk_data is not None:
                print(
                    "[graph-timing] oracle_gap served_from_disk "
                    f"runs={len(resolved_run_ids)} elapsed_ms={int((time.perf_counter() - request_started) * 1000)}"
                )
                return self._respond_json(disk_data)

            runs = []
            for rid in resolved_run_ids:
                loaded = load_run_cached(rid)
                if loaded:
                    runs.append(loaded)
            if not runs:
                return self._respond_json({"error": "no runs found"}, status=404)
            data = build_oracle_gap(runs, qoe_model)
            _write_oracle_gap_to_disk(cache_key, data)
            print(
                "[graph-timing] oracle_gap generated "
                f"runs={len(runs)} elapsed_ms={int((time.perf_counter() - request_started) * 1000)}"
            )
            return self._respond_json(data)
        if path_only == "/api/runs":
            try:
                payload = self._read_json_body()
                if parse_bool(payload.get("oracle_gap_suite")):
                    jobs = start_oracle_gap_suite(payload)
                    return self._respond_json(
                        {
                            "run_id": jobs[0].run_id if jobs else None,
                            "run_ids": [job.run_id for job in jobs],
                            "status": "queued",
                        }
                    )
                job = start_run(payload)
                return self._respond_json({"run_id": job.run_id, "run_ids": [job.run_id], "status": job.status})
            except Exception as exc:  # pylint: disable=broad-except
                return self._respond_json({"error": str(exc)}, status=400)
        if path_only == "/api/runs/batch":
            try:
                payload = self._read_json_body()
                raw_runs = payload.get("runs") if isinstance(payload, dict) else None
                if not isinstance(raw_runs, list) or not raw_runs:
                    raise ValueError("Batch payload must include a non-empty 'runs' list.")
                jobs = start_runs_batch([run for run in raw_runs if isinstance(run, dict)])
                if not jobs:
                    raise ValueError("Batch payload did not include any valid run configs.")
                return self._respond_json(
                    {
                        "run_id": jobs[0].run_id,
                        "run_ids": [job.run_id for job in jobs],
                        "status": "queued",
                    }
                )
            except Exception as exc:  # pylint: disable=broad-except
                return self._respond_json({"error": str(exc)}, status=400)
        if path_only == "/api/run-groups":
            try:
                return self._respond_json({"groups": [upsert_run_group(self._read_json_body())]})
            except Exception as exc:  # pylint: disable=broad-except
                return self._respond_json({"error": str(exc)}, status=400)
        if path_only == "/api/run-groups/assign":
            try:
                return self._respond_json(assign_runs_to_group(self._read_json_body()))
            except Exception as exc:  # pylint: disable=broad-except
                return self._respond_json({"error": str(exc)}, status=400)
        if path_only == "/api/run-groups/remove-runs":
            try:
                return self._respond_json(remove_runs_from_groups(self._read_json_body()))
            except Exception as exc:  # pylint: disable=broad-except
                return self._respond_json({"error": str(exc)}, status=400)
        if path_only == "/api/run-groups/archive":
            try:
                payload = self._read_json_body()
                return self._respond_json(archive_run_group(str(payload.get("group_id") or ""), parse_bool(payload.get("archived"))))
            except Exception as exc:  # pylint: disable=broad-except
                return self._respond_json({"error": str(exc)}, status=400)
        if path_only == "/api/run-groups/control":
            try:
                payload = self._read_json_body()
                return self._respond_json(control_run_group(str(payload.get("group_id") or ""), str(payload.get("action") or "")))
            except Exception as exc:  # pylint: disable=broad-except
                return self._respond_json({"error": str(exc)}, status=400)
        if path_only == "/api/run-groups/delete-with-runs":
            try:
                payload = self._read_json_body()
                return self._respond_json(delete_run_group_with_runs(str(payload.get("group_id") or "")))
            except Exception as exc:  # pylint: disable=broad-except
                return self._respond_json({"error": str(exc)}, status=400)
        return self._respond_json({"error": "Unsupported endpoint"}, status=404)

    def do_DELETE(self):
        if self.path.startswith("/api/datasets/"):
            dataset_id = unquote(self.path.split("/", 3)[3])
            return self._respond_json(delete_dataset_profile(dataset_id))
        if self.path.startswith("/api/training/datasets/"):
            dataset_id = unquote(self.path.split("/", 4)[4])
            return self._respond_json(delete_training_dataset(dataset_id))
        if self.path.startswith("/api/training/plans/"):
            plan_id = unquote(self.path.split("/", 4)[4])
            return self._respond_json(delete_training_plan(plan_id))
        if self.path.startswith("/api/training/runs/"):
            run_id = unquote(self.path.split("/", 4)[4])
            return self._respond_json(delete_training_run(run_id))
        if self.path.startswith("/api/training/models/"):
            model_id = unquote(self.path.split("/", 4)[4])
            return self._respond_json(delete_model(model_id))
        if self.path.startswith("/api/run-form-sessions/"):
            name = unquote(self.path.split("/", 3)[3])
            delete_run_form_session(name)
            return self._respond_json(run_form_sessions_payload())
        if self.path.startswith("/api/run-groups/"):
            group_id = unquote(self.path.split("/", 3)[3])
            return self._respond_json(delete_run_group(group_id))
        if self.path.startswith("/api/runs/"):
            run_id = self.path.split("/")[3]
            result = forget_run(run_id)
            if not result.get("deleted"):
                return self._respond_json(result, status=409)
            run_dir = RESULTS_ROOT / run_id
            if run_dir.exists() and run_dir.is_dir():
                shutil.rmtree(run_dir)
            remove_indexed_run(RESULTS_ROOT, run_id)
            try:
                remove_runs_from_groups({"run_ids": [run_id]})
            except Exception:
                pass
            return self._respond_json(result)
        return self._respond_json({"error": "Unsupported endpoint"}, status=404)

    def _serve_file(self, path: Path, content_type: str | None = None, content_encoding: str | None = None):
        if not path.exists():
            self.send_response(404)
            self.end_headers()
            return
        if content_type is None:
            content_type = "text/html" if path.suffix == ".html" else "text/plain"
            if path.suffix == ".css":
                content_type = "text/css"
            elif path.suffix == ".js":
                content_type = "application/javascript"
            elif path.suffix == ".json":
                content_type = "application/json"
            elif path.suffix == ".pdf":
                content_type = "application/pdf"
            elif path.suffix == ".svg":
                content_type = "image/svg+xml"
            elif path.suffix == ".csv":
                content_type = "text/csv; charset=utf-8"
            elif path.suffix == ".md":
                content_type = "text/markdown; charset=utf-8"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if content_encoding:
            self.send_header("Content-Encoding", content_encoding)
            self.send_header("Vary", "Accept-Encoding")
        if self._is_web_asset(path):
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(data)

    def _respond_json(self, payload: Any, status: int = 200):
        started = time.perf_counter()
        data = json.dumps(payload).encode("utf-8")
        encode_ms = int((time.perf_counter() - started) * 1000)
        use_gzip = "gzip" in (self.headers.get("Accept-Encoding") or "").lower() and len(data) >= 16 * 1024
        if use_gzip:
            data = gzip.compress(data, compresslevel=5)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        if use_gzip:
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        self.end_headers()
        self.wfile.write(data)
        if self.path.startswith("/api/diagnostics"):
            print(
                "[graph-timing] diagnostics response "
                f"status={status} bytes={len(data)} encode_ms={encode_ms} "
                f"elapsed_ms={int((time.perf_counter() - started) * 1000)}"
            )

    def _serve_trace(self, encoded_path: str):
        started = time.perf_counter()
        trace_path = safe_path(unquote(encoded_path.split("?")[0]))
        if not trace_path.exists():
            return self._respond_json({"error": "Trace not found"}, status=404)
        if trace_path.stat().st_size == 0:
            return self._respond_json({"error": "Trace is empty"}, status=400)
        cache_key = _trace_cache_key(trace_path)
        cached = TRACE_SAMPLE_CACHE.get(cache_key)
        if cached:
            samples = cached.get("samples") or []
            print(
                "[graph-cache] trace_samples memory_hit "
                f"path={trace_path} samples={len(samples)} elapsed_ms={int((time.perf_counter() - started) * 1000)}"
            )
            return self._respond_json({"samples": samples})
        disk_samples = _load_trace_samples_from_disk(cache_key)
        if disk_samples is not None:
            print(
                "[graph-timing] trace_csv served_from_disk "
                f"path={trace_path} samples={len(disk_samples)} elapsed_ms={int((time.perf_counter() - started) * 1000)}"
            )
            return self._respond_json({"samples": disk_samples})
        csv_started = time.perf_counter()
        raw_df = pd.read_csv(trace_path)
        csv_ms = int((time.perf_counter() - csv_started) * 1000)
        normalize_started = time.perf_counter()
        df = normalize_network_dataframe(raw_df, source=trace_path)
        normalize_ms = int((time.perf_counter() - normalize_started) * 1000)
        serialize_started = time.perf_counter()
        samples = []
        for _, row in df.iterrows():
            try:
                samples.append({"time_s": float(row["Time"]) - 1.0, "bandwidth_mbps": float(row["Bandwidth"])})
            except (TypeError, ValueError):
                continue
        serialize_ms = int((time.perf_counter() - serialize_started) * 1000)
        print(
            "[graph-timing] trace_csv "
            f"path={trace_path} csv_parse_ms={csv_ms} normalize_ms={normalize_ms} "
            f"serialize_ms={serialize_ms} elapsed_ms={int((time.perf_counter() - started) * 1000)}"
        )
        _remember_trace_samples(cache_key, samples)
        _write_trace_samples_to_disk(cache_key, samples)
        return self._respond_json({"samples": samples})

    def _serve_runs_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        last_payload = None
        last_ping = time.monotonic()
        try:
            while True:
                payload = json.dumps(active_runs_snapshot())
                if payload != last_payload:
                    self.wfile.write(f"event: runs\ndata: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    last_payload = payload
                    last_ping = time.monotonic()
                elif time.monotonic() - last_ping > 15:
                    self.wfile.write(b"event: ping\ndata: {}\n\n")
                    self.wfile.flush()
                    last_ping = time.monotonic()
                time.sleep(1.0)
        except (BrokenPipeError, ConnectionResetError):
            return
