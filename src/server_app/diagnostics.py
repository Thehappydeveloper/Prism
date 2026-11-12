from __future__ import annotations

import json
import hashlib
import multiprocessing
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

from qoe.base import QoEFunction
from qoe.heuristics import WeightedQoEHeuristic
from core.tile_fov import (
    DEFAULT_GROUND_TRUTH_MODE,
    DEFAULT_PEAK_WEIGHT,
    DEFAULT_VISIBLE_THRESHOLD,
    reduce_tile_visibility_frames,
)
from core.event_storage import event_log_path, read_event_log_text
from .config import PROJECT_ROOT
from .config import RESULTS_ROOT
from .fov_summary import summarize_fov_segments


RUN_METRICS_CACHE: Dict[str, Dict[str, Any]] = {}
RUN_METRICS_CACHE_LOCK = threading.Lock()
RUN_METRICS_CACHE_MAX_BYTES = 128 * 1024 * 1024
RUN_METRICS_CACHE_BYTES = 0
GRAPH_CACHE_VERSION = 12
GRAPH_CACHE_ROOT = PROJECT_ROOT / ".prism_cache" / "graphs"
RUN_METRICS_DISK_CACHE = GRAPH_CACHE_ROOT / "run_metrics"


def _log(*args, **kwargs) -> None:
    """Keep diagnostic progress out of machine-readable report stdout."""
    print(*args, file=sys.stderr, **kwargs)


METRIC_DATASETS = frozenset(
    {
        "labels", "qoe", "qoe_series", "fov_quality", "fov_prediction", "delta_fov",
        "stall", "component_bits", "bitrate_vs_quality", "buffer", "meta",
        "latency",
    }
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _cache_digest(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _disk_cache_path(key: str) -> Path:
    return RUN_METRICS_DISK_CACHE / f"{_cache_digest(key)}.json"


def _disk_cache_is_current(key: str) -> bool:
    path = _disk_cache_path(key)
    try:
        with path.open("r", encoding="utf-8") as handle:
            prefix = handle.read(96)
    except OSError:
        return False
    return f'"version":{GRAPH_CACHE_VERSION}' in prefix


def _empty_metrics() -> Dict:
    return {
        "labels": {},
        "qoe": {},
        "qoe_series": {},
        "fov_quality": {},
        "fov_prediction": {},
        "fov_segments": {},
        "delta_fov": {},
        "stall": {},
        "component_bits": {},
        "bitrate_vs_quality": {},
        "buffer": {},
        "latency": {},
        "meta": {},
    }


def _qoe_signature(qoe_model: QoEFunction) -> str:
    return _stable_json(
        {
            "class": qoe_model.__class__.__name__,
            "module": qoe_model.__class__.__module__,
            # QoE implementations may change their scoring semantics without
            # changing constructor state.  Include the explicit class-level
            # version so a paper-report rerun cannot silently reuse values
            # cached under an earlier metric definition.
            "scoring_version": getattr(qoe_model, "_scoring_version", None),
            "state": getattr(qoe_model, "__dict__", {}),
        }
    )


def _run_metric_cache_key(run: Dict, qoe_model: QoEFunction) -> str:
    fingerprint = run.get("artifact_fingerprint") or {}
    result_fingerprint = {name: fingerprint.get(name) for name in ("config.json", "events.json")}
    return _stable_json(
        {
            "run_id": run.get("run_id"),
            "artifact_fingerprint": result_fingerprint,
            "qoe": _qoe_signature(qoe_model),
        }
    )


def _remember_run_metrics(key: str, metrics: Dict) -> None:
    global RUN_METRICS_CACHE_BYTES
    encoded = json.dumps(metrics, separators=(",", ":"), default=str)
    size_bytes = len(encoded.encode("utf-8"))
    with RUN_METRICS_CACHE_LOCK:
        while RUN_METRICS_CACHE and RUN_METRICS_CACHE_BYTES + size_bytes > RUN_METRICS_CACHE_MAX_BYTES:
            evicted = next(iter(RUN_METRICS_CACHE))
            removed = RUN_METRICS_CACHE.pop(evicted, None) or {}
            RUN_METRICS_CACHE_BYTES -= int(removed.get("size_bytes", 0))
            _log(f"[graph-cache] run_metrics invalidate key={evicted[:12]}")
        RUN_METRICS_CACHE[key] = {"metrics": metrics, "created": time.time(), "size_bytes": size_bytes}
        RUN_METRICS_CACHE_BYTES += size_bytes
    started = time.perf_counter()
    path = _disk_cache_path(key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp_path.write_text(
            json.dumps(
                {
                    "version": GRAPH_CACHE_VERSION,
                    "key": key,
                    "created": time.time(),
                    "metrics": metrics,
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        tmp_path.replace(path)
        _log(f"[graph-cache] run_metrics disk_write path={path} elapsed_ms={int((time.perf_counter() - started) * 1000)}")
    except OSError as exc:
        _log(f"[graph-cache] run_metrics disk_write_failed path={path} error={exc}")


def _load_run_metrics_from_disk(key: str) -> Dict | None:
    started = time.perf_counter()
    path = _disk_cache_path(key)
    if not path.exists():
        _log(f"[graph-cache] run_metrics disk_miss elapsed_ms={int((time.perf_counter() - started) * 1000)}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        _log(f"[graph-cache] run_metrics disk_read_failed path={path} error={exc}")
        return None
    if payload.get("version") == 10 and payload.get("key") == key and isinstance(payload.get("metrics"), dict):
        metrics = _compact_legacy_metrics(payload["metrics"])
        _remember_run_metrics(key, metrics)
        _log(f"[graph-cache] run_metrics migrated path={path} elapsed_ms={int((time.perf_counter() - started) * 1000)}")
        return metrics
    if payload.get("version") != GRAPH_CACHE_VERSION or payload.get("key") != key:
        _log(f"[graph-cache] run_metrics disk_stale path={path} elapsed_ms={int((time.perf_counter() - started) * 1000)}")
        return None
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        return None
    _log(f"[graph-cache] run_metrics disk_hit path={path} elapsed_ms={int((time.perf_counter() - started) * 1000)}")
    _remember_run_metrics_in_memory(key, metrics, payload.get("created", time.time()))
    return metrics


def _remember_run_metrics_in_memory(key: str, metrics: Dict, created: float | None = None) -> None:
    global RUN_METRICS_CACHE_BYTES
    size_bytes = len(json.dumps(metrics, separators=(",", ":"), default=str).encode("utf-8"))
    with RUN_METRICS_CACHE_LOCK:
        previous = RUN_METRICS_CACHE.pop(key, None) or {}
        RUN_METRICS_CACHE_BYTES -= int(previous.get("size_bytes", 0))
        while RUN_METRICS_CACHE and RUN_METRICS_CACHE_BYTES + size_bytes > RUN_METRICS_CACHE_MAX_BYTES:
            removed = RUN_METRICS_CACHE.pop(next(iter(RUN_METRICS_CACHE)))
            RUN_METRICS_CACHE_BYTES -= int(removed.get("size_bytes", 0))
        RUN_METRICS_CACHE[key] = {"metrics": metrics, "created": created or time.time(), "size_bytes": size_bytes}
        RUN_METRICS_CACHE_BYTES += size_bytes


def _merge_metrics(target: Dict, source: Dict, *, datasets: set[str] | None = None) -> None:
    keys = ("labels", "qoe", "qoe_series", "fov_quality", "fov_prediction", "fov_segments", "delta_fov", "stall", "component_bits", "bitrate_vs_quality", "buffer", "latency", "meta")
    for key in keys:
        if datasets is not None and key not in datasets and key not in {"labels", "meta"}:
            continue
        target[key].update(source.get(key, {}))


def _downsample_points(points: list[Any], max_points: int = 400) -> list[Any]:
    if len(points) <= max_points:
        return points
    step = (len(points) - 1) / (max_points - 1)
    indices = sorted({round(index * step) for index in range(max_points)})
    return [points[index] for index in indices]


def _compact_legacy_metrics(metrics: Dict) -> Dict:
    for run_id, segments in list((metrics.get("fov_segments") or {}).items()):
        metrics.setdefault("fov_prediction", {})[run_id] = summarize_fov_segments(segments or []) or {}
        metrics["fov_segments"][run_id] = []
    for dataset in ("qoe", "qoe_series", "delta_fov"):
        for run_id, values in list((metrics.get(dataset) or {}).items()):
            metrics[dataset][run_id] = _downsample_points(values or [], 400)
    for run_id, split in list((metrics.get("fov_quality") or {}).items()):
        if not isinstance(split, dict):
            continue
        split["fov"] = _downsample_points(split.get("fov") or [], 400)
        split["non"] = _downsample_points(split.get("non") or [], 400)
    return metrics


def parse_quality_id(qid: str | None) -> int | None:
    if not qid:
        return None
    q = qid.lower().replace("quality_", "")
    if q.startswith("q"):
        q = q[1:]
    try:
        return int(q)
    except ValueError:
        return None


def comp_max_levels(events: Dict) -> Dict[str, int]:
    comp_max: Dict[str, int] = {}
    for seg in events.get("enqueued_segments", []) or []:
        for tile in (seg.get("tiles") or {}).values():
            for comp_name, comp in (tile.get("components") or {}).items():
                lvl = parse_quality_id(comp.get("quality_id")) or comp.get("rep_idx", 0)
                comp_max[comp_name] = max(comp_max.get(comp_name, 0), int(lvl))
    return comp_max


def tile_quality(tile: Dict, comp_max: Dict[str, int]) -> float:
    total = 0.0
    w = 0.0
    for comp_name, comp in (tile.get("components") or {}).items():
        lvl = parse_quality_id(comp.get("quality_id")) or comp.get("rep_idx", 0)
        # A component can intentionally remain at its base representation in
        # every selected tile (e.g. atlas when its ABR saliency is zero).
        # Its observed maximum is then zero; it represents zero normalized
        # quality, not an invalid chart value.
        max_lvl = max(int(comp_max.get(comp_name, max(lvl, 1)) or 0), 1)
        total += max(0.0, min(1.0, lvl / max_lvl))
        w += 1.0
    return total / w if w else 0.0


def _visibility_by_segment(entries: List[Dict] | None) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for entry in entries or []:
        try:
            segment_idx = int(entry.get("segment_idx"))
        except (TypeError, ValueError):
            continue
        raw_visibility = entry.get("tile_visibility") or {}
        if not isinstance(raw_visibility, dict):
            continue
        visibility: Dict[str, float] = {}
        for tile_id, value in raw_visibility.items():
            try:
                numeric = float(value or 0.0)
            except (TypeError, ValueError):
                numeric = 0.0
            visibility[str(tile_id)] = max(0.0, min(1.0, numeric))
        out[segment_idx] = visibility
    return out


def _ground_truth_frame_entries_by_segment(entries: List[Dict] | None) -> tuple[Dict[int, List[Dict]], Dict[int, Dict[str, float]], Dict[int, str]]:
    frames_by_segment: Dict[int, List[Dict]] = {}
    target_by_segment: Dict[int, Dict[str, float]] = {}
    mode_by_segment: Dict[int, str] = {}
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        try:
            segment_idx = int(entry.get("segment_idx"))
        except (TypeError, ValueError):
            continue
        raw_frames = entry.get("frames") or []
        frames: List[Dict] = []
        if isinstance(raw_frames, list):
            for frame_entry in raw_frames:
                if not isinstance(frame_entry, dict):
                    continue
                raw_visibility = frame_entry.get("tile_visibility") or {}
                if not isinstance(raw_visibility, dict):
                    continue
                visibility: Dict[str, float] = {}
                for tile_id, value in raw_visibility.items():
                    try:
                        numeric = float(value or 0.0)
                    except (TypeError, ValueError):
                        numeric = 0.0
                    visibility[str(tile_id)] = max(0.0, min(1.0, numeric))
                if not visibility:
                    continue
                try:
                    frame_number = int(frame_entry.get("frame"))
                except (TypeError, ValueError):
                    frame_number = len(frames)
                frames.append({"frame": frame_number, "tile_visibility": visibility})
        if frames:
            frames_by_segment[segment_idx] = frames

        raw_target = entry.get("target_visibility") or {}
        if isinstance(raw_target, dict):
            target: Dict[str, float] = {}
            for tile_id, value in raw_target.items():
                try:
                    numeric = float(value or 0.0)
                except (TypeError, ValueError):
                    numeric = 0.0
                target[str(tile_id)] = max(0.0, min(1.0, numeric))
            if target:
                target_by_segment[segment_idx] = target
        mode_by_segment[segment_idx] = str(entry.get("target_mode") or entry.get("mode") or DEFAULT_GROUND_TRUTH_MODE)
    return frames_by_segment, target_by_segment, mode_by_segment


def _ground_truth_params(events: Dict, config: Dict) -> tuple[str, float, float]:
    mode = str(
        events.get("ground_truth_mode")
        or config.get("ground_truth_mode")
        or DEFAULT_GROUND_TRUTH_MODE
    )
    params = events.get("ground_truth_params") if isinstance(events.get("ground_truth_params"), dict) else {}
    try:
        threshold = float(params.get("visible_threshold", config.get("ground_truth_visible_threshold", DEFAULT_VISIBLE_THRESHOLD)))
    except (TypeError, ValueError):
        threshold = DEFAULT_VISIBLE_THRESHOLD
    try:
        peak_weight = float(params.get("peak_weight", config.get("ground_truth_peak_weight", DEFAULT_PEAK_WEIGHT)))
    except (TypeError, ValueError):
        peak_weight = DEFAULT_PEAK_WEIGHT
    return mode, threshold, peak_weight


def segment_fov_sets(events: Dict) -> tuple[list[set[str]], bool]:
    fov_sets: list[set[str]] = []
    has_fov_info = False
    for seg in events.get("enqueued_segments", []) or []:
        hinted = seg.get("tiles_in_fov") or seg.get("viewport_tiles") or []
        seg_fov = {str(t) for t in hinted}
        if seg_fov:
            has_fov_info = True
        fov_sets.append(seg_fov)
    return fov_sets, has_fov_info


def network_class(path: str | None) -> str | None:
    if not path:
        return None
    low = str(path).lower()
    if "5g" in low:
        return "5G"
    if "4g" in low or "lte" in low:
        return "4G"
    return None


def network_label(trace_path: str | None, profile: str | None) -> str | None:
    if profile:
        return f"profile:{profile}"
    return network_class(trace_path)


def _compute_run_metrics(run: Dict, qoe_model: QoEFunction) -> Dict:
    metrics = _empty_metrics()
    if run.get("events_error"):
        metrics["meta"][run["run_id"]] = {"error": run["events_error"]}
        return metrics

    run_id = run.get("run_id", "unknown_run")
    started = time.perf_counter()
    abr = run["config"].get("abr", "unknown")
    predictor = run["config"].get("predictor") or "None"
    predictor_params = run["config"].get("predictor_params") or {}
    predictor_preset_name = run["config"].get("predictor_preset_name")
    bandwidth_predictor = run["config"].get("bandwidth_predictor") or "None"
    net_trace = run["config"].get("network_trace")
    net_profile = run["config"].get("network_profile")
    fov_trace = run["config"].get("fov_trace")
    short = run_id.replace("run_", "")
    short = short[-6:] if len(short) > 6 else short
    metrics["labels"][run_id] = f"{short} ({abr})"
    events = run["events"]

    comp_started = time.perf_counter()
    comp_max = comp_max_levels(events)
    fov_sets, has_fov_info = segment_fov_sets(events)
    fov_predictions_by_segment = _visibility_by_segment(events.get("fov_predictions"))
    fov_ground_truth_by_segment = _visibility_by_segment(events.get("fov_ground_truth"))
    fov_gt_frames_by_segment, fov_gt_targets_by_segment, fov_gt_modes_by_segment = _ground_truth_frame_entries_by_segment(
        events.get("fov_ground_truth_frames")
    )
    default_gt_mode, default_gt_threshold, default_gt_peak_weight = _ground_truth_params(events, run.get("config") or {})
    comp_ms = int((time.perf_counter() - comp_started) * 1000)

    metrics["qoe"].setdefault(run_id, [])
    metrics["qoe_series"].setdefault(run_id, [])
    metrics["fov_quality"].setdefault(run_id, {"fov": [], "non": []})
    metrics["fov_prediction"].setdefault(run_id, {})
    metrics["fov_segments"].setdefault(run_id, [])
    metrics["delta_fov"].setdefault(run_id, [])
    metrics["stall"].setdefault(run_id, {"duration": 0.0, "count": 0})
    metrics["component_bits"].setdefault(run_id, {})
    metrics["bitrate_vs_quality"].setdefault(run_id, [])
    metrics["buffer"].setdefault(run_id, {})
    metrics["meta"].setdefault(
        run_id,
        {
            "duration_s": 0.0,
            "fov_present": has_fov_info,
            "abr": abr,
            "predictor": predictor,
            "predictor_params": predictor_params,
            "predictor_preset_name": predictor_preset_name,
            "bandwidth_predictor": bandwidth_predictor,
            "network_trace": net_trace,
            "network_profile": net_profile,
            "network_class": network_label(net_trace, net_profile),
            "fov_trace": fov_trace,
            "dataset_id": run["config"].get("dataset_id"),
            "buffer_window_segments": int(run["config"].get("buffer_window_segments") or 0),
        },
    )

    qoe_started = time.perf_counter()
    last_qoe: float | None = None
    for t, q in qoe_model.compute_time_series(events, run_config=run.get("config") or {}):
        score = float(q)
        metrics["qoe_series"][run_id].append({"time_s": float(t), "score": score})
        if last_qoe is None or abs(score - last_qoe) > 1e-9:
            metrics["qoe"][run_id].append(score)
            last_qoe = score
    metrics["qoe_series"][run_id] = _downsample_points(metrics["qoe_series"][run_id], 400)
    qoe_ms = int((time.perf_counter() - qoe_started) * 1000)

    derived_started = time.perf_counter()
    prev_tile_quality: Dict[str, float] = {}
    playback_seconds = 0.0
    for idx, seg in enumerate(events.get("enqueued_segments", []) or []):
        try:
            segment_idx = int(seg.get("segment_idx", idx))
        except (TypeError, ValueError):
            segment_idx = idx
        seg_dur = seg.get("duration_s", 1.0)
        seg_bits = seg.get("size_bits", 0.0)
        playback_seconds += float(seg_dur or 0.0)
        tile_q = {}
        tile_bits = {}
        fov_sum = 0.0
        fov_count = 0
        seg_fov = fov_sets[idx] if idx < len(fov_sets) else set()
        seg_has_fov = bool(seg_fov)

        for tile_id, tile in (seg.get("tiles") or {}).items():
            tid = str(tile_id)
            tq = tile_quality(tile, comp_max)
            tile_q[tid] = tq
            tile_bits[tid] = float(tile.get("size_bits", 0.0) or 0.0)
            if has_fov_info and seg_has_fov:
                in_fov = tid in seg_fov
            elif has_fov_info and not seg_has_fov:
                in_fov = True
            else:
                in_fov = True
            cls = "fov" if in_fov else "non"
            metrics["fov_quality"][run_id][cls].append(tq)
            if in_fov:
                fov_sum += tq
                fov_count += 1
            for comp_name, comp in (tile.get("components") or {}).items():
                metrics["component_bits"][run_id][comp_name] = metrics["component_bits"][run_id].get(comp_name, 0.0) + comp.get("size_bits", 0.0)

        for tid, tq in tile_q.items():
            if has_fov_info and seg_has_fov and tid not in seg_fov:
                prev_tile_quality.pop(tid, None)
                continue
            if tid in prev_tile_quality:
                metrics["delta_fov"][run_id].append(abs(tq - prev_tile_quality[tid]))
            prev_tile_quality[tid] = tq

        if fov_count == 0 and tile_q:
            fov_sum = sum(tile_q.values())
            fov_count = len(tile_q)
        predicted_visibility = fov_predictions_by_segment.get(segment_idx)
        actual_mean_visibility = fov_ground_truth_by_segment.get(segment_idx)
        actual_frames = fov_gt_frames_by_segment.get(segment_idx) or []
        actual_mode = fov_gt_modes_by_segment.get(segment_idx, default_gt_mode)
        actual_visibility = fov_gt_targets_by_segment.get(segment_idx)
        if actual_visibility is None and actual_frames:
            actual_visibility = reduce_tile_visibility_frames(
                actual_frames,
                tile_ids=tile_q.keys(),
                mode=actual_mode,
                visible_threshold=default_gt_threshold,
                peak_weight=default_gt_peak_weight,
            )
        if actual_visibility is None:
            actual_visibility = actual_mean_visibility
        if predicted_visibility and actual_visibility:
            metrics["fov_segments"][run_id].append(
                {
                    "segment_idx": segment_idx,
                    "duration_s": float(seg_dur or 0.0),
                    "predicted": predicted_visibility,
                    "actual": actual_visibility,
                    "actual_mean": actual_mean_visibility or actual_visibility,
                    "actual_frames": actual_frames,
                    "ground_truth_mode": actual_mode if actual_frames else "mean_visibility",
                    "tile_quality": tile_q,
                    "tile_bits": tile_bits,
                }
            )
        avg_fov_q = (fov_sum / fov_count) if fov_count else 0.0
        bitrate_mbps = (seg_bits / seg_dur) / 1e6 if seg_dur else 0.0
        metrics["bitrate_vs_quality"][run_id].append((bitrate_mbps, avg_fov_q))

    stalls = events.get("rebuffering_events") or []
    metrics["stall"][run_id]["duration"] += sum((s.get("duration_s", 0) for s in stalls))
    metrics["stall"][run_id]["count"] += len(stalls)
    metrics["meta"][run_id]["duration_s"] = playback_seconds

    vals = [float(b) for b in events.get("buffer_occupancy_s") or [] if isinstance(b, (float, int))]
    if vals:
        mean_buf = sum(vals) / len(vals)
        var_buf = sum((b - mean_buf) ** 2 for b in vals) / len(vals)
        metrics["buffer"][run_id] = {
            "mean": mean_buf,
            "std": var_buf ** 0.5,
            "min": min(vals),
            "max": max(vals),
        }
    metrics["fov_prediction"][run_id] = summarize_fov_segments(metrics["fov_segments"][run_id]) or {}
    # Raw tile and frame maps belong to the single-run debugger. Advanced Compare
    # persists only the compact sufficient statistics above.
    metrics["fov_segments"][run_id] = []
    metrics["qoe"][run_id] = _downsample_points(metrics["qoe"][run_id], 400)
    metrics["fov_quality"][run_id]["fov"] = _downsample_points(metrics["fov_quality"][run_id]["fov"], 400)
    metrics["fov_quality"][run_id]["non"] = _downsample_points(metrics["fov_quality"][run_id]["non"], 400)
    metrics["delta_fov"][run_id] = _downsample_points(metrics["delta_fov"][run_id], 400)
    derived_ms = int((time.perf_counter() - derived_started) * 1000)
    _log(
        "[graph-timing] run_metrics compute "
        f"run={run_id} comp_ms={comp_ms} qoe_ms={qoe_ms} derived_ms={derived_ms} "
        f"elapsed_ms={int((time.perf_counter() - started) * 1000)}"
    )
    return metrics


def _cached_run_metrics(run: Dict, qoe_model: QoEFunction) -> Dict:
    run_id = run.get("run_id", "unknown_run")
    started = time.perf_counter()
    key = _run_metric_cache_key(run, qoe_model)
    with RUN_METRICS_CACHE_LOCK:
        cached = RUN_METRICS_CACHE.get(key)
    if cached is not None:
        _log(f"[graph-cache] run_metrics hit run={run_id} elapsed_ms={int((time.perf_counter() - started) * 1000)}")
        return cached["metrics"]
    disk_metrics = _load_run_metrics_from_disk(key)
    if disk_metrics is not None:
        _log(f"[graph-cache] run_metrics disk_reuse run={run_id} elapsed_ms={int((time.perf_counter() - started) * 1000)}")
        return disk_metrics
    _log(f"[graph-cache] run_metrics miss run={run_id}")
    metrics = _compute_run_metrics(run, qoe_model)
    _remember_run_metrics(key, metrics)
    return metrics


def _load_run_for_summary(run_id: str) -> Dict | None:
    run_dir = RESULTS_ROOT / run_id
    config_path = run_dir / "config.json"
    events_path = event_log_path(run_dir)
    if not config_path.exists() or events_path is None:
        return None
    fingerprint: Dict[str, Any] = {}
    for name, path in (("config.json", config_path), ("events.json", events_path)):
        stat = path.stat()
        fingerprint[name] = {"path": str(path), "mtime_ns": stat.st_mtime_ns, "size": stat.st_size}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        events = json.loads(read_event_log_text(events_path))
    except json.JSONDecodeError as exc:
        return {"run_id": run_id, "config": config, "events_error": str(exc), "artifact_fingerprint": fingerprint}
    return {"run_id": run_id, "config": config, "events": events, "artifact_fingerprint": fingerprint}


def _load_predictor_latency(run_id: str) -> Dict[str, Any] | None:
    path = RESULTS_ROOT / run_id / "predictor_diagnostics.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    total = payload.get("total") if isinstance(payload.get("total"), dict) else {}
    if not total:
        return None
    stages = []
    for stage in payload.get("stages") or []:
        if not isinstance(stage, dict):
            continue
        stages.append(
            {
                "name": str(stage.get("name") or "unknown"),
                "count": int(stage.get("count") or 0),
                "median_ms": float(stage.get("median_ms") or 0.0),
                "mean_ms": float(stage.get("mean_ms") or 0.0),
                "p95_ms": float(stage.get("p95_ms") or 0.0),
                "max_ms": float(stage.get("max_ms") or 0.0),
            }
        )
    return {
        "count": int(total.get("count") or payload.get("prediction_count") or 0),
        "median_ms": float(total.get("median_ms") or 0.0),
        "mean_ms": float(total.get("mean_ms") or 0.0),
        "p95_ms": float(total.get("p95_ms") or 0.0),
        "max_ms": float(total.get("max_ms") or 0.0),
        "stages": stages,
    }


def _cache_key_for_run_id(run_id: str, qoe_model: QoEFunction) -> str | None:
    run_dir = RESULTS_ROOT / run_id
    fingerprint: Dict[str, Any] = {}
    for name in ("config.json", "events.json"):
        path = event_log_path(run_dir) if name == "events.json" else run_dir / name
        if path is None or not path.exists():
            return None
        stat = path.stat()
        fingerprint[name] = {"path": str(path), "mtime_ns": stat.st_mtime_ns, "size": stat.st_size}
    return _stable_json({"run_id": run_id, "artifact_fingerprint": fingerprint, "qoe": _qoe_signature(qoe_model)})


def _cached_run_metrics_by_id(run_id: str, qoe_model: QoEFunction) -> Dict | None:
    key = _cache_key_for_run_id(run_id, qoe_model)
    if key is None:
        return None
    with RUN_METRICS_CACHE_LOCK:
        cached = RUN_METRICS_CACHE.get(key)
    if cached is not None:
        return cached["metrics"]
    disk_metrics = _load_run_metrics_from_disk(key)
    if disk_metrics is not None:
        return disk_metrics
    run = _load_run_for_summary(run_id)
    if run is None:
        return None
    metrics = _compute_run_metrics(run, qoe_model)
    _remember_run_metrics(key, metrics)
    return metrics


def _process_run_summary(args: tuple[str, QoEFunction]) -> Dict | None:
    run_id, qoe_model = args
    return _cached_run_metrics_by_id(run_id, qoe_model)


def gather_diagnostics_for_run_ids(
    run_ids: List[str],
    qoe_model: QoEFunction | None = None,
    *,
    datasets: set[str] | None = None,
) -> Dict:
    """Load compact summaries directly, using processes only for legacy cache misses."""
    qoe_model = qoe_model or WeightedQoEHeuristic()
    requested = (datasets & METRIC_DATASETS) if datasets is not None else set(METRIC_DATASETS)
    out = _empty_metrics()
    if not run_ids:
        return out

    hits: list[Dict] = []
    misses: list[str] = []
    for run_id in run_ids:
        key = _cache_key_for_run_id(run_id, qoe_model)
        if key is None:
            continue
        with RUN_METRICS_CACHE_LOCK:
            cached = RUN_METRICS_CACHE.get(key)
        part = cached["metrics"] if cached is not None else (_load_run_metrics_from_disk(key) if _disk_cache_is_current(key) else None)
        if part is None:
            misses.append(run_id)
        else:
            hits.append(part)

    built: list[Dict] = []
    requested_workers = max(1, int(os.environ.get("PRISM_DIAGNOSTICS_WORKERS", "4")))
    worker_count = min(requested_workers, len(misses), max(1, os.cpu_count() or 1))
    if worker_count > 1:
        try:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=multiprocessing.get_context("spawn"),
            ) as executor:
                built = [part for part in executor.map(_process_run_summary, ((run_id, qoe_model) for run_id in misses)) if part]
        except Exception as exc:  # pylint: disable=broad-except
            _log(f"[graph-cache] process_backfill_fallback error={exc}")
            with ThreadPoolExecutor(max_workers=min(8, len(misses))) as executor:
                built = [part for part in executor.map(lambda rid: _cached_run_metrics_by_id(rid, qoe_model), misses) if part]
    elif misses:
        for run_id in misses:
            part = _cached_run_metrics_by_id(run_id, qoe_model)
            if part:
                built.append(part)

    for part in hits + built:
        _merge_metrics(out, part, datasets=requested)
    # Meta is deliberately refreshed from the tiny config file. Older per-run
    # summary caches predate these fields, while paired FoV statistics need a
    # stable trace identity and the correct startup window.
    for run_id in run_ids:
        meta = out.get("meta", {}).get(run_id)
        if not isinstance(meta, dict):
            continue
        config_path = RESULTS_ROOT / run_id / "config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        meta["fov_trace"] = config.get("fov_trace")
        meta["dataset_id"] = config.get("dataset_id")
        meta["buffer_window_segments"] = int(config.get("buffer_window_segments") or 0)
    if "latency" in requested:
        for run_id in run_ids:
            latency = _load_predictor_latency(run_id)
            if latency is not None:
                out["latency"][run_id] = latency
    _log(
        "[graph-timing] compact diagnostics "
        f"runs={len(run_ids)} cache_hits={len(hits)} backfilled={len(built)} workers={worker_count}"
    )
    return out


def schedule_default_run_summary(run_id: str) -> None:
    """Leave completed runs immediately available to the control UI.

    Metrics are generated lazily by the graph or report that requests them.
    Prewarming every completion used a server thread to recompute the legacy
    weighted-QoE summary, even when the eventual request needed FoV-aware QoE
    v2 and therefore could not reuse that cache entry.  During a batch this
    saturated the server with avoidable CPU work and made the UI unresponsive.
    """
    _log(f"[graph-cache] run_metrics deferred run={run_id}")


def gather_diagnostics(
    runs: List[Dict],
    qoe_model: QoEFunction | None = None,
    *,
    include_fov_segments: bool = False,
    datasets: set[str] | None = None,
) -> Dict:
    qoe_model = qoe_model or WeightedQoEHeuristic()
    started = time.perf_counter()
    out = _empty_metrics()
    if not runs:
        return out

    worker_count = min(max(1, os.cpu_count() or 1), len(runs), 8)
    if worker_count <= 1:
        parts = [_cached_run_metrics(run, qoe_model) for run in runs]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            parts = list(executor.map(lambda run: _cached_run_metrics(run, qoe_model), runs))

    merge_started = time.perf_counter()
    for part in parts:
        _merge_metrics(out, part, datasets=datasets)
    merge_ms = int((time.perf_counter() - merge_started) * 1000)
    _log(
        "[graph-timing] diagnostics assemble "
        f"runs={len(runs)} workers={worker_count} merge_ms={merge_ms} elapsed_ms={int((time.perf_counter() - started) * 1000)}"
    )
    return out
