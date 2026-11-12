from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List

from integrations.navis_client import _join_url, post_run_to_navis

from .config import RESULTS_ROOT, navis_settings, navis_status, read_json
from .runs import RUNS, RUNS_LOCK, load_run


def tile_catalog(events: Dict[str, Any]) -> Dict[str, Any]:
    catalog: Dict[str, Any] = {}
    for seg in events.get("enqueued_segments", []) or []:
        for tile_id, tile in (seg.get("tiles") or {}).items():
            tid = str(tile_id)
            if tid in catalog:
                continue
            catalog[tid] = {
                "bounds": tile.get("bounds") or {},
                "components": sorted((tile.get("components") or {}).keys()),
            }
    return catalog


def component_max_rep_idx(decisions: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for entry in decisions:
        if not isinstance(entry, dict):
            continue
        for stage in ("requested", "applied"):
            stage_map = entry.get(stage) or {}
            if not isinstance(stage_map, dict):
                continue
            for comp_map in stage_map.values():
                if not isinstance(comp_map, dict):
                    continue
                for comp_name, rep_idx in comp_map.items():
                    try:
                        rep_val = int(rep_idx)
                    except (TypeError, ValueError):
                        continue
                    out[comp_name] = max(out.get(comp_name, 0), rep_val)
    return out


def fov_visibility_by_segment(entries: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            seg_idx = int(entry.get("segment_idx"))
        except (TypeError, ValueError):
            continue
        visibility = entry.get("tile_visibility")
        if isinstance(visibility, dict):
            out[seg_idx] = {str(tile_id): value for tile_id, value in visibility.items()}
    return out


def build_navis_payload(run_id: str, run: Dict[str, Any], status: Dict[str, Any]) -> Dict[str, Any]:
    events = run.get("events", {}) if isinstance(run, dict) else {}
    config = run.get("config", {}) if isinstance(run, dict) else {}
    abr_decisions = events.get("abr_decisions") or []
    fov_predictions = events.get("fov_predictions") or []
    fov_ground_truth = events.get("fov_ground_truth") or []
    enqueued_segments = events.get("enqueued_segments", []) or []
    segment_duration_s = None
    if enqueued_segments:
        try:
            segment_duration_s = float(enqueued_segments[0].get("duration_s"))
        except (TypeError, ValueError):
            segment_duration_s = None

    fov_prediction_by_segment = fov_visibility_by_segment(fov_predictions)
    fov_ground_truth_by_segment = fov_visibility_by_segment(fov_ground_truth)

    segments: List[Dict[str, Any]] = []
    for entry in abr_decisions:
        if not isinstance(entry, dict):
            continue
        try:
            seg_idx = int(entry.get("segment_idx"))
        except (TypeError, ValueError):
            continue
        fov_prediction = fov_prediction_by_segment.get(seg_idx, {})
        ground_truth = fov_ground_truth_by_segment.get(seg_idx, {})
        segments.append(
            {
                "segment_idx": seg_idx,
                "time_s": entry.get("time_s"),
                "duration_s": segment_duration_s,
                "abr_requested": entry.get("requested") or {},
                "abr_applied": entry.get("applied") or {},
                "fov_prediction": fov_prediction,
                "fov_ground_truth": ground_truth,
                "fov_visibility": fov_prediction,
            }
        )

    return {
        "source": "prism",
        "schema": "prism.navis.compat.v2",
        "exported_at": datetime.now().isoformat(),
        "run": {
            "run_id": run_id,
            "status": status.get("status"),
            "started": status.get("started"),
            "finished": status.get("finished"),
        },
        "algorithms": {
            "abr_name": config.get("abr"),
            "predictor_name": config.get("predictor"),
        },
        "headset": {
            "preset": config.get("headset_preset"),
            "fov_h_deg": config.get("fov_h_deg"),
            "fov_v_deg": config.get("fov_v_deg"),
            "angle_convention": config.get("angle_convention"),
        },
        "content": {
            "ply_dir_path": config.get("content_dir"),
            "mpd_path": config.get("mpd"),
        },
        "traces": {
            "navigation_trace_path": config.get("fov_trace"),
            "bandwidth_trace_path": config.get("network_trace"),
            "network_profile": config.get("network_profile"),
        },
        "playback": {
            "segment_count": len(segments),
            "segment_duration_s": segment_duration_s,
            "buffer_window_segments": config.get("buffer_window_segments"),
            "segment_limit": config.get("segment_limit"),
            "rtt_s": config.get("rtt"),
        },
        "tiling": {
            "tile_catalog": tile_catalog(events),
            "component_max_rep_idx": component_max_rep_idx(abr_decisions),
        },
        "segments": segments,
    }


def store_navis_export_result(run_id: str, export_result: Dict[str, Any]) -> None:
    with RUNS_LOCK:
        job = RUNS.get(run_id)
        if job is not None:
            job.navis_export = export_result
    status_file = RESULTS_ROOT / run_id / "status.json"
    status = read_json(status_file) if status_file.exists() else {"run_id": run_id}
    if not isinstance(status, dict):
        status = {"run_id": run_id}
    status["navis_export"] = export_result
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.write_text(json.dumps(status, indent=2), encoding="utf-8")


def export_run_to_navis(
    run_id: str,
    base_url: str | None = None,
    export_path: str | None = None,
    health_path: str | None = None,
) -> Dict[str, Any]:
    run = load_run(run_id)
    if not run:
        return {"ok": False, "run_id": run_id, "error": "Run not found."}
    if run.get("events_error"):
        return {"ok": False, "run_id": run_id, "error": f"Events file is invalid: {run['events_error']}"}

    cfg = navis_settings()
    effective_base = (base_url or cfg["base_url"]).strip()
    effective_export = (export_path or cfg["export_path"]).strip() or "/api/prism/runs"
    effective_health = (health_path or cfg["health_path"]).strip() or "/health"
    status = navis_status(base_url=effective_base, health_path=effective_health)
    status["export_path"] = effective_export
    if not status.get("reachable"):
        result = {
            "ok": False,
            "run_id": run_id,
            "error": status.get("error") or "Navis is unreachable.",
            "navis_status": status,
        }
        store_navis_export_result(run_id, result)
        return result

    status_file = RESULTS_ROOT / run_id / "status.json"
    payload = build_navis_payload(run_id=run_id, run=run, status=read_json(status_file))
    sent = post_run_to_navis(
        base_url=effective_base,
        export_path=effective_export,
        payload=payload,
        timeout_s=float(cfg["timeout_s"]),
        auth_token=(cfg.get("auth_token") or None),
    )
    response = sent.get("response")
    redirect_url = ""
    if isinstance(response, dict):
        for key in ("redirect_url", "open_url", "url"):
            val = response.get(key)
            if isinstance(val, str) and val.strip():
                redirect_url = val.strip()
                break
    if redirect_url:
        redirect_url = _join_url(effective_base, redirect_url)
    if not redirect_url:
        redirect_url = f"{effective_base.rstrip('/')}/?run_id={run_id}"

    sent["payload_summary"] = {
        "schema": payload.get("schema"),
        "segment_count": len(payload.get("segments") or []),
        "has_fov_prediction": any((segment.get("fov_prediction") or {}) for segment in payload.get("segments") or []),
        "has_fov_ground_truth": any((segment.get("fov_ground_truth") or {}) for segment in payload.get("segments") or []),
        "ply_dir_path": ((payload.get("content") or {}).get("ply_dir_path")),
        "navigation_trace_path": ((payload.get("traces") or {}).get("navigation_trace_path")),
        "bandwidth_trace_path": ((payload.get("traces") or {}).get("bandwidth_trace_path")),
    }
    sent["run_id"] = run_id
    sent["redirect_url"] = redirect_url
    sent["navis_status"] = status
    store_navis_export_result(run_id, sent)
    return sent
