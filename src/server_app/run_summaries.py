from __future__ import annotations

import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from core.event_storage import event_log_path, read_event_log_text
from fov_predictors.display import predictor_label

COMPONENT_ORDER = ("atlas", "geom", "occp", "attr")
USEFUL_METADATA_FIELDS = {
    "active_tiles",
    "active_sample_points",
    "candidate_voxels",
    "free_voxels",
    "reachable_voxels",
    "pruned_voxels",
    "visible_voxels",
    "frame_count",
    "nav_horizon_s",
    "r_nav_m",
    "direction_reliability",
    "speed_mps",
    "auto_prune_budget_cap",
    "auto_prune_mass_kept",
    "auto_prune_mass_target",
    "auto_sample_total",
    "auto_visibility_time_budget_ms",
    "occupancy_voxel_size_l",
    "navigation_voxel_size_l",
    "visibility_voxel_size_l",
    "visibility_floor",
    "source_tile_count",
    "source_tiles_with_score",
    "propagated_tiles_with_score",
    "source_target_pairs_passing_threshold",
    "raycast_processes_requested",
    "raycast_processes_used",
    "raycast_parallel_fallback",
    "raycast_parallel_spawn",
    "path_neighbor_connectivity",
    "path_topology_cache_size",
    "navigation_to_visibility_voxels",
    "navigation_to_visibility_collisions",
    "navigation_to_visibility_origin_offset_mean",
    "navigation_to_visibility_origin_offset_max",
    "effective_voxels_per_tile",
    "selected_voxel_mass_mean",
    "intention_horizon_count",
    "intention_motion_consensus",
    "intention_confidence_scale",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _round(value: Any, digits: int = 3) -> float:
    return round(_safe_float(value), digits)


def _load_json_with_error(path: Path) -> tuple[Dict[str, Any], str | None]:
    if not path.exists():
        return {}, "not found"
    try:
        payload = json.loads(read_event_log_text(path) if path.name.endswith(".gz") else path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, f"invalid JSON: {exc}"
    except OSError as exc:
        return {}, str(exc)
    return payload if isinstance(payload, dict) else {}, None


def _read_json(path: Path) -> Dict[str, Any]:
    payload, error = _load_json_with_error(path)
    return {} if error else payload


def _bits_to_mbps(bits: Any, duration_s: Any) -> float:
    duration = max(_safe_float(duration_s), 1e-9)
    return round((_safe_float(bits) / duration) / 1e6, 3)


def _component_reps(selection: Dict[str, int] | None) -> str:
    if not selection:
        return "-"
    parts = []
    for component in COMPONENT_ORDER:
        if component in selection:
            parts.append(f"{component}{selection[component]}")
    for component in sorted(k for k in selection.keys() if k not in COMPONENT_ORDER):
        parts.append(f"{component}{selection[component]}")
    return " ".join(parts) if parts else "-"


def _decision_summary(selection_by_tile: Dict[str, Dict[str, int]] | None, limit: int = 8) -> str:
    if not selection_by_tile:
        return "No ABR decision recorded"
    tile_ids = sorted(selection_by_tile.keys(), key=lambda value: (_safe_int(value, 10**9), str(value)))
    chunks = [f"T{tile_id}: {_component_reps(selection_by_tile.get(tile_id))}" for tile_id in tile_ids[:limit]]
    if len(tile_ids) > limit:
        chunks.append(f"+{len(tile_ids) - limit} tiles")
    return "; ".join(chunks)


def _top_fov(tile_visibility: Dict[str, Any] | None, limit: int = 3) -> List[Dict[str, float]]:
    if not isinstance(tile_visibility, dict):
        return []
    ranked = [
        {"tile_id": str(tile_id), "visibility": _round(value)}
        for tile_id, value in tile_visibility.items()
        if _safe_float(value) > 0.0
    ]
    ranked.sort(key=lambda item: item["visibility"], reverse=True)
    return ranked[:limit]


def _buffer_after(time_s: Any, times: List[Any], values: List[Any]) -> float | None:
    target = _safe_float(time_s)
    best_idx = None
    for idx, raw_time in enumerate(times):
        if _safe_float(raw_time) <= target + 1e-9:
            best_idx = idx
    if best_idx is None or best_idx >= len(values):
        return None
    return _round(values[best_idx])


def _prediction_by_segment(events: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for entry in events.get("fov_predictions") or []:
        if isinstance(entry, dict):
            out[_safe_int(entry.get("segment_idx"), -1)] = entry
    return out


def _decision_by_segment(events: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for entry in events.get("abr_decisions") or []:
        if isinstance(entry, dict):
            out[_safe_int(entry.get("segment_idx"), -1)] = entry
    return out


def _consumed_by_segment(events: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for entry in events.get("consumed_segments") or []:
        if isinstance(entry, dict):
            out[_safe_int(entry.get("segment_idx"), -1)] = entry
    return out


def _rebuffer_notes(events: Dict[str, Any]) -> Dict[int, str]:
    notes: Dict[int, str] = {}
    enqueued = sorted(
        [seg for seg in events.get("enqueued_segments") or [] if isinstance(seg, dict)],
        key=lambda seg: _safe_float(seg.get("time_s")),
    )
    for rebuffer in events.get("rebuffering_events") or []:
        if not isinstance(rebuffer, dict):
            continue
        time_s = _safe_float(rebuffer.get("time_s"))
        duration_s = _safe_float(rebuffer.get("duration_s"))
        target = None
        for seg in enqueued:
            if _safe_float(seg.get("time_s")) >= time_s:
                target = _safe_int(seg.get("segment_idx"), -1)
                break
        if target is not None and target >= 0:
            notes[target] = f"Rebuffered {duration_s:.2f}s"
    return notes


def _segment_timeline(events: Dict[str, Any], log_story: Dict[str, Any]) -> List[Dict[str, Any]]:
    decisions = _decision_by_segment(events)
    predictions = _prediction_by_segment(events)
    consumed = _consumed_by_segment(events)
    rebuffer_notes = _rebuffer_notes(events)
    wait_notes = {
        _safe_int(item.get("segment_idx"), -1): item.get("message")
        for item in log_story.get("events") or []
        if item.get("kind") == "warning" and _safe_int(item.get("segment_idx"), -1) >= 0
    }
    times = events.get("time_s") or []
    buffers = events.get("buffer_occupancy_s") or []
    rows: List[Dict[str, Any]] = []

    enqueued = sorted(
        [seg for seg in events.get("enqueued_segments") or [] if isinstance(seg, dict)],
        key=lambda seg: _safe_int(seg.get("segment_idx"), 0),
    )
    for seg in enqueued:
        segment_idx = _safe_int(seg.get("segment_idx"), len(rows))
        decision = decisions.get(segment_idx, {})
        prediction = predictions.get(segment_idx, {})
        consume = consumed.get(segment_idx, {})
        notes = [note for note in (wait_notes.get(segment_idx), rebuffer_notes.get(segment_idx)) if note]
        rows.append(
            {
                "segment_idx": segment_idx,
                "decision_time_s": _round(decision.get("time_s", seg.get("time_s"))),
                "enqueue_time_s": _round(seg.get("time_s")),
                "duration_s": _round(seg.get("duration_s")),
                "download_time_s": _round(consume.get("consumed_s")),
                "size_mbit": _round(_safe_float(seg.get("size_bits")) / 1e6, 2),
                "bitrate_mbps": _bits_to_mbps(seg.get("size_bits"), seg.get("duration_s")),
                "buffer_after_s": _buffer_after(seg.get("time_s"), times, buffers),
                "selection_summary": _decision_summary(decision.get("applied") or decision.get("requested")),
                "top_fov_tiles": _top_fov(prediction.get("tile_visibility")),
                "notes": notes,
            }
        )
    return rows


def _overview(run_id: str, run_dir: Path, events: Dict[str, Any], config: Dict[str, Any], events_error: str | None) -> Dict[str, Any]:
    status = _read_json(run_dir / "status.json")
    segments = [seg for seg in events.get("enqueued_segments") or [] if isinstance(seg, dict)]
    total_bits = sum(_safe_float(seg.get("size_bits")) for seg in segments)
    total_duration = sum(_safe_float(seg.get("duration_s")) for seg in segments)
    return {
        "run_id": run_id,
        "status": status.get("status") or ("finished" if segments else "unknown"),
        "error": status.get("error") or events_error,
        "started": status.get("started"),
        "finished": status.get("finished"),
        "abr": config.get("abr") or "-",
        "predictor": config.get("predictor") or "None",
        "dataset_id": config.get("dataset_id"),
        "network_trace": config.get("network_trace"),
        "network_profile": config.get("network_profile"),
        "fov_trace": config.get("fov_trace"),
        "segment_count": len(segments),
        "playback_time_s": _round(events.get("total_playback_time")),
        "rebuffer_time_s": _round(events.get("total_rebuffering_time")),
        "rebuffer_count": len(events.get("rebuffering_events") or []),
        "average_bitrate_mbps": _bits_to_mbps(total_bits, total_duration) if total_duration > 0 else 0.0,
    }


SEGMENT_RE = re.compile(r"Processing segment (?P<segment>\d+)")
BANDWIDTH_RE = re.compile(r"Tile (?P<tile>\S+) component (?P<component>\S+) \(rep (?P<rep>\d+)\) avg bandwidth .*: (?P<mbps>[\d.]+) Mbps")
SIZE_RE = re.compile(r"Segment (?P<segment>\d+) total size: (?P<mbit>[\d.]+) Mb")
DOWNLOADED_RE = re.compile(r"Segment (?P<segment>\d+) downloaded in (?P<seconds>[\d.]+) s")
SUMMARY_RE = re.compile(
    r"Segment (?P<segment>\d+) summary: size (?P<mbit>[\d.]+) Mb, download (?P<seconds>[\d.]+) s, avg bandwidth (?P<mbps>[\d.]+) Mbps"
)
WAIT_RE = re.compile(r"Buffer full, waiting (?P<seconds>[\d.]+) s")
ABORT_RE = re.compile(r"Adaptive abort: tile (?P<tile>\S+) component (?P<component>\S+) from rep (?P<from>\d+) to (?P<to>\d+)")


def _level_from_line(line: str) -> str:
    match = re.search(r"\[(?P<level>[A-Z]+)\]", line)
    return match.group("level") if match else "INFO"


def build_log_story(run_dir: Path, run_id: str) -> Dict[str, Any]:
    log_file = run_dir / f"{run_id}.log"
    if not log_file.exists():
        return {"available": False, "summary": "No run.log file was found.", "segments": [], "events": []}

    current_segment: int | None = None
    segments: Dict[int, Dict[str, Any]] = {}
    events: List[Dict[str, Any]] = []
    line_count = 0
    component_lines = 0

    for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line_count += 1
        message = line.split(": ", 1)[-1] if ": " in line else line
        level = _level_from_line(line)

        if match := SEGMENT_RE.search(message):
            current_segment = _safe_int(match.group("segment"))
            segments.setdefault(current_segment, {"segment_idx": current_segment, "component_count": 0, "avg_component_bandwidth_mbps": None})
            continue

        if match := BANDWIDTH_RE.search(message):
            component_lines += 1
            segment = segments.setdefault(
                current_segment if current_segment is not None else -1,
                {"segment_idx": current_segment, "component_count": 0, "avg_component_bandwidth_mbps": None},
            )
            samples = segment.setdefault("_bandwidth_samples", [])
            samples.append(_safe_float(match.group("mbps")))
            segment["component_count"] = len(samples)
            continue

        if match := SIZE_RE.search(message):
            segment_idx = _safe_int(match.group("segment"))
            segments.setdefault(segment_idx, {"segment_idx": segment_idx})["size_mbit"] = _round(match.group("mbit"), 2)
            continue

        if match := DOWNLOADED_RE.search(message):
            segment_idx = _safe_int(match.group("segment"))
            segments.setdefault(segment_idx, {"segment_idx": segment_idx})["download_time_s"] = _round(match.group("seconds"))
            continue

        if match := SUMMARY_RE.search(message):
            segment_idx = _safe_int(match.group("segment"))
            segment = segments.setdefault(segment_idx, {"segment_idx": segment_idx})
            segment["size_mbit"] = _round(match.group("mbit"), 2)
            segment["download_time_s"] = _round(match.group("seconds"))
            segment["avg_component_bandwidth_mbps"] = _round(match.group("mbps"), 2)
            continue

        if match := WAIT_RE.search(message):
            events.append(
                {
                    "kind": "warning",
                    "segment_idx": current_segment,
                    "message": f"Buffer full; waited {_safe_float(match.group('seconds')):.2f}s before downloading.",
                }
            )
            continue

        if match := ABORT_RE.search(message):
            events.append(
                {
                    "kind": "adaptive_abort",
                    "segment_idx": current_segment,
                    "message": (
                        f"Tile {match.group('tile')} {match.group('component')} switched "
                        f"from rep {match.group('from')} to {match.group('to')}."
                    ),
                }
            )
            continue

        if level in {"WARNING", "ERROR", "CRITICAL"}:
            events.append({"kind": level.lower(), "segment_idx": current_segment, "message": message})

    story_segments = []
    for segment_idx in sorted(segments):
        if segment_idx < 0:
            continue
        item = dict(segments[segment_idx])
        samples = item.pop("_bandwidth_samples", [])
        if samples:
            item["avg_component_bandwidth_mbps"] = _round(mean(samples), 2)
            item["min_component_bandwidth_mbps"] = _round(min(samples), 2)
            item["max_component_bandwidth_mbps"] = _round(max(samples), 2)
        story_segments.append(item)

    return {
        "available": True,
        "summary": f"Collapsed {component_lines} per-component bandwidth lines from {line_count} log lines.",
        "segments": story_segments,
        "events": events,
    }


def build_predictor_story(run_dir: Path) -> Dict[str, Any]:
    diagnostics, error = _load_json_with_error(run_dir / "predictor_diagnostics.json")
    if error:
        return {"available": False, "summary": f"Predictor diagnostics are {error}.", "story": [], "slowest_stages": [], "metadata": []}

    total = diagnostics.get("total") or {}
    stages = [stage for stage in diagnostics.get("stages") or [] if isinstance(stage, dict)]
    stages.sort(key=lambda stage: _safe_float(stage.get("mean_ms")), reverse=True)
    slowest = [
        {
            "name": stage.get("name") or "-",
            "mean_ms": _round(stage.get("mean_ms")),
            "p95_ms": _round(stage.get("p95_ms")),
            "max_ms": _round(stage.get("max_ms")),
            "count": _safe_int(stage.get("count")),
        }
        for stage in stages[:5]
    ]

    metadata_summary = diagnostics.get("metadata_summary") or {}
    metadata = []
    for key in sorted(USEFUL_METADATA_FIELDS.intersection(metadata_summary.keys())):
        value = metadata_summary.get(key) or {}
        metadata.append(
            {
                "name": key,
                "median": _round(value.get("median")),
                "min": _round(value.get("min")),
                "max": _round(value.get("max")),
                "count": _safe_int(value.get("count")),
            }
        )

    predictor_name = predictor_label(diagnostics.get("predictor_name") or "Predictor")
    prediction_count = _safe_int(diagnostics.get("prediction_count"))
    story = [
        f"{predictor_name} produced {prediction_count} prediction{'s' if prediction_count != 1 else ''}.",
        (
            f"Typical end-to-end prediction time was {_round(total.get('median_ms'))} ms "
            f"(mean {_round(total.get('mean_ms'))} ms, p95 {_round(total.get('p95_ms'))} ms)."
        ),
    ]
    if slowest:
        story.append(f"The slowest stage by mean time was {slowest[0]['name']} at {slowest[0]['mean_ms']} ms.")

    anomalies = []
    for stage in stages:
        median_ms = _safe_float(stage.get("median_ms"))
        max_ms = _safe_float(stage.get("max_ms"))
        if median_ms > 0 and max_ms >= median_ms * 2.0:
            anomalies.append(f"{stage.get('name')} had a max latency spike of {_round(max_ms)} ms.")
    story.extend(anomalies[:3])

    return {
        "available": True,
        "predictor_name": predictor_name,
        "prediction_count": prediction_count,
        "total": {
            "median_ms": _round(total.get("median_ms")),
            "mean_ms": _round(total.get("mean_ms")),
            "p95_ms": _round(total.get("p95_ms")),
            "max_ms": _round(total.get("max_ms")),
        },
        "summary": " ".join(story),
        "story": story,
        "slowest_stages": slowest,
        "metadata": metadata,
    }


def build_readable_summary(run_dir: Path, run_id: str) -> Dict[str, Any]:
    events_file = event_log_path(run_dir)
    events, events_error = _load_json_with_error(events_file) if events_file is not None else ({}, "not found")
    config, config_error = _load_json_with_error(run_dir / "config.json")
    log_story = build_log_story(run_dir, run_id)
    predictor_story = build_predictor_story(run_dir)
    return {
        "overview": _overview(run_id, run_dir, events, config, events_error or config_error),
        "timeline": _segment_timeline(events, log_story),
        "predictor_story": predictor_story,
        "log_story": log_story,
        "raw_artifacts": {
            "events_path": f"/api/runs/{run_id}/events" if events_file is not None else None,
            "log_path": f"/results/{run_id}/{run_id}.log" if (run_dir / f"{run_id}.log").exists() else None,
            "predictor_diagnostics_path": (
                f"/results/{run_id}/predictor_diagnostics.json" if (run_dir / "predictor_diagnostics.json").exists() else None
            ),
        },
    }
