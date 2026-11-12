from __future__ import annotations

import math
import time
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .base import QoEFunction, UiField
from core.trace_columns import normalize_fov_dataframe
from fov_predictors.base import FovPredictor
from core.rotations import forward_from_quaternion_xyzw, quaternion_from_row
from core.mpd_parser import V3CMpd


@dataclass
class QoEWeights:
    """Weights controlling the QoE heuristic."""

    component_weights: Dict[str, float] = field(
        default_factory=lambda: {"attr": 0.35, "geom": 0.35, "occp": 0.2, "atlas": 0.0}
    )
    tile_saliency: Dict[str, float] = field(default_factory=dict)
    fov_saliency: float = 1.25
    temporal_penalty: float = 0.2
    stall_per_second: float = 0.15
    stall_fixed: float = 0.05
    default_component: float = 0.1

    def as_params(self) -> Dict[str, float]:
        params = dict(self.component_weights)
        params.update(
            {
                "default_component": self.default_component,
                "fov_saliency": self.fov_saliency,
                "temporal_penalty": self.temporal_penalty,
                "stall_per_second": self.stall_per_second,
                "stall_fixed": self.stall_fixed,
            }
        )
        return params


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TRACE_CACHE: Dict[str, List[Dict[str, float]]] = {}
_MPD_COMPONENT_LEVEL_CACHE: Dict[str, Dict[str, int]] = {}
_MPD_COMPONENT_BITRATE_CACHE: Dict[str, Dict[str, Tuple[float, float]]] = {}
_POSITION_KEY_CANDIDATES: Tuple[Tuple[str, str, str], ...] = (
    ("HMDPX", "HMDPY", "HMDPZ"),
    ("PositionX", "PositionY", "PositionZ"),
    ("pos_x", "pos_y", "pos_z"),
    ("x", "y", "z"),
    ("X", "Y", "Z"),
)
_ROTATION_KEY_CANDIDATES: Tuple[Tuple[str, str, str], ...] = (
    ("HMDRX", "HMDRY", "HMDRZ"),
    ("rot_x", "rot_y", "rot_z"),
    ("RX", "RY", "RZ"),
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _resolve_path(raw_path: Any) -> Optional[Path]:
    if not raw_path:
        return None
    candidate = Path(str(raw_path))
    if candidate.exists():
        return candidate
    project_candidate = (PROJECT_ROOT / candidate).resolve()
    if project_candidate.exists():
        return project_candidate
    return None


def _load_fov_trace_rows(trace_path: Optional[Path]) -> List[Dict[str, float]]:
    if trace_path is None:
        return []
    cache_key = str(trace_path.resolve())
    if cache_key in _TRACE_CACHE:
        return _TRACE_CACHE[cache_key]

    rows: List[Dict[str, float]] = []
    try:
        raw_df = normalize_fov_dataframe(pd.read_csv(trace_path), source=trace_path)
        for raw_row in raw_df.to_dict(orient="records"):
            row: Dict[str, float] = {}
            for key, value in raw_row.items():
                if key is None:
                    continue
                try:
                    row[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
            if row:
                rows.append(row)
    except (OSError, ValueError):
        return []

    if rows and "FrameNumber" in rows[0]:
        rows.sort(key=lambda row: int(row.get("FrameNumber", 0)))
    _TRACE_CACHE[cache_key] = rows
    return rows


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _parse_quality_id(quality_id: Optional[str]) -> Optional[int]:
    if not quality_id:
        return None
    q = quality_id.lower().replace("quality_", "")
    if q.startswith("q"):
        q = q[1:]
    try:
        return int(q)
    except ValueError:
        return None


def _component_level(comp: Mapping[str, Any]) -> int:
    """Return the component ladder index, preferring component rep_idx over combined quality labels."""
    try:
        return int(comp.get("rep_idx", 0) or 0)
    except (TypeError, ValueError):
        pass
    parsed = _parse_quality_id(comp.get("quality_id"))
    return int(parsed or 0)


def _segment_idx(segment: Mapping[str, Any], fallback: int) -> int:
    try:
        return int(segment.get("segment_idx", fallback))
    except (TypeError, ValueError):
        return fallback


def _segment_duration(segment: Mapping[str, Any]) -> float:
    return max(_safe_float(segment.get("duration_s"), 1.0), 1e-6)


def _sorted_segments(events: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    return sorted(events.get("enqueued_segments") or [], key=lambda s: _safe_float(s.get("time_s"), 0.0))


def _max_component_levels(events: Mapping[str, Any]) -> Dict[str, int]:
    max_levels: Dict[str, int] = {}
    for seg in events.get("enqueued_segments", []) or []:
        for tile in (seg.get("tiles") or {}).values():
            for comp_name, comp in (tile.get("components") or {}).items():
                lvl = _component_level(comp)
                max_levels[comp_name] = max(max_levels.get(comp_name, 0), int(lvl))
    return max_levels


def _mpd_component_max_levels(run_config: Mapping[str, Any] | None) -> Dict[str, int]:
    if not run_config:
        return {}
    mpd_path = _resolve_path(run_config.get("mpd"))
    if mpd_path is None:
        return {}
    cache_key = str(mpd_path.resolve())
    if cache_key in _MPD_COMPONENT_LEVEL_CACHE:
        return dict(_MPD_COMPONENT_LEVEL_CACHE[cache_key])
    out: Dict[str, int] = {}
    try:
        mpd = V3CMpd(mpd_path)
        for tile_id in mpd.iter_tile_ids():
            for comp_name in (mpd.tile_components.get(tile_id) or {}).keys():
                count = len(mpd.component_quality_order(tile_id, comp_name))
                if count > 0:
                    out[comp_name] = max(out.get(comp_name, 0), count - 1)
    except Exception:
        out = {}
    _MPD_COMPONENT_LEVEL_CACHE[cache_key] = dict(out)
    return out


def _mpd_component_bitrate_ranges(run_config: Mapping[str, Any] | None) -> Dict[str, Tuple[float, float]]:
    """Return one content-defined rate ladder per component from the MPD.

    The range is intentionally independent of a run's selected qualities, so
    utility values are comparable across predictors and ABR decisions.
    """
    if not run_config:
        return {}
    mpd_path = _resolve_path(run_config.get("mpd"))
    if mpd_path is None:
        return {}
    cache_key = str(mpd_path.resolve())
    if cache_key in _MPD_COMPONENT_BITRATE_CACHE:
        return dict(_MPD_COMPONENT_BITRATE_CACHE[cache_key])
    values: Dict[str, List[float]] = {}
    try:
        mpd = V3CMpd(mpd_path)
        for tile_components in mpd.tile_components.values():
            for component, representations in tile_components.items():
                values.setdefault(component, []).extend(
                    float(rep.bandwidth) for rep in representations.values()
                    if rep.bandwidth is not None and rep.bandwidth > 0
                )
    except Exception:
        values = {}
    ranges = {
        component: (max(min(rates), 1.0), max(max(rates), max(min(rates), 1.0)))
        for component, rates in values.items() if rates
    }
    _MPD_COMPONENT_BITRATE_CACHE[cache_key] = dict(ranges)
    return ranges


def _component_rep_norm(comp: Mapping[str, Any], max_levels: Mapping[str, int], comp_name: str) -> float:
    lvl = _component_level(comp)
    max_lvl = max(int(max_levels.get(comp_name, 0) or 0), lvl, 1)
    return _clamp01(lvl / float(max_lvl))


def _component_bitrate_bps(comp: Mapping[str, Any]) -> float:
    bandwidth = _safe_float(comp.get("bandwidth_bps"), 0.0)
    if bandwidth > 0.0:
        return bandwidth
    duration_s = max(_safe_float(comp.get("duration_s"), 0.0), 0.0)
    size_bits = max(_safe_float(comp.get("size_bits"), 0.0), 0.0)
    if duration_s <= 0.0 or size_bits <= 0.0:
        return 0.0
    return size_bits / duration_s


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = _clamp01(q) * (len(values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def _component_bitrate_ranges(events: Mapping[str, Any]) -> Dict[str, Tuple[float, float]]:
    samples: Dict[str, List[float]] = {}
    for seg in events.get("enqueued_segments", []) or []:
        for tile in (seg.get("tiles") or {}).values():
            for comp_name, comp in (tile.get("components") or {}).items():
                bitrate = _component_bitrate_bps(comp)
                if bitrate > 0.0:
                    samples.setdefault(comp_name, []).append(bitrate)

    ranges: Dict[str, Tuple[float, float]] = {}
    for comp_name, values in samples.items():
        ordered = sorted(values)
        low = _quantile(ordered, 0.1)
        high = _quantile(ordered, 0.9)
        if high <= low:
            low = ordered[0]
            high = ordered[-1]
        ranges[comp_name] = (max(low, 1.0), max(high, max(low, 1.0)))
    return ranges


def _buffer_samples(events: Mapping[str, Any]) -> Tuple[List[float], List[float]]:
    times = events.get("time_s") or []
    values = events.get("buffer_occupancy_s") or []
    pairs = [
        (_safe_float(t), max(_safe_float(v), 0.0))
        for t, v in zip(times, values)
        if math.isfinite(_safe_float(t)) and math.isfinite(_safe_float(v))
    ]
    pairs.sort(key=lambda item: item[0])
    if not pairs:
        return [], []
    return [item[0] for item in pairs], [item[1] for item in pairs]


def _buffer_at_time(times: Sequence[float], values: Sequence[float], time_s: float) -> Optional[float]:
    if not times or not values:
        return None
    idx = bisect_right(times, time_s) - 1
    if idx < 0:
        return values[0]
    return values[min(idx, len(values) - 1)]


def _min_buffer_in_window(
    times: Sequence[float],
    values: Sequence[float],
    start_s: float,
    end_s: float,
) -> Optional[float]:
    if not times or not values:
        return None
    lo = bisect_right(times, start_s) - 1
    hi = bisect_right(times, end_s)
    window_vals: List[float] = []
    if lo >= 0:
        window_vals.append(values[min(lo, len(values) - 1)])
    window_vals.extend(values[max(lo + 1, 0):hi])
    if not window_vals:
        return _buffer_at_time(times, values, start_s)
    return min(window_vals)


def _segment_visibility_maps(events: Mapping[str, Any]) -> Dict[int, Dict[str, float]]:
    visibility_by_segment: Dict[int, Dict[str, float]] = {}

    for fallback_idx, seg in enumerate(_sorted_segments(events)):
        seg_idx = _segment_idx(seg, fallback_idx)
        hinted = seg.get("tiles_in_fov") or seg.get("viewport_tiles") or []
        if hinted:
            visibility_by_segment[seg_idx] = {str(tile_id): 1.0 for tile_id in hinted}

    for entry in events.get("fov_predictions") or []:
        if not isinstance(entry, Mapping):
            continue
        try:
            seg_idx = int(entry.get("segment_idx"))
        except (TypeError, ValueError):
            continue
        raw_visibility = entry.get("tile_visibility") or {}
        if not isinstance(raw_visibility, Mapping):
            continue
        parsed = {
            str(tile_id): _clamp01(_safe_float(score))
            for tile_id, score in raw_visibility.items()
        }
        if parsed:
            visibility_by_segment[seg_idx] = parsed

    return visibility_by_segment


def _segment_ground_truth_visibility_maps(events: Mapping[str, Any]) -> Dict[int, Dict[str, float]]:
    visibility_by_segment: Dict[int, Dict[str, float]] = {}
    for entry in events.get("fov_ground_truth") or []:
        if not isinstance(entry, Mapping):
            continue
        try:
            seg_idx = int(entry.get("segment_idx"))
        except (TypeError, ValueError):
            continue
        raw_visibility = entry.get("tile_visibility") or {}
        if not isinstance(raw_visibility, Mapping):
            continue
        parsed = {
            str(tile_id): _clamp01(_safe_float(score))
            for tile_id, score in raw_visibility.items()
        }
        if parsed:
            visibility_by_segment[seg_idx] = parsed
    return visibility_by_segment


def _segment_ground_truth_frame_maps(events: Mapping[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
    frames_by_segment: Dict[int, List[Dict[str, Any]]] = {}
    for entry in events.get("fov_ground_truth_frames") or []:
        if not isinstance(entry, Mapping):
            continue
        try:
            seg_idx = int(entry.get("segment_idx"))
        except (TypeError, ValueError):
            continue
        parsed_frames: List[Dict[str, Any]] = []
        for raw_frame in entry.get("frames") or []:
            if not isinstance(raw_frame, Mapping):
                continue
            raw_visibility = raw_frame.get("tile_visibility") or {}
            if not isinstance(raw_visibility, Mapping):
                continue
            visibility = {
                str(tile_id): _clamp01(_safe_float(score))
                for tile_id, score in raw_visibility.items()
            }
            if not visibility:
                continue
            try:
                frame_number = int(raw_frame.get("frame"))
            except (TypeError, ValueError):
                frame_number = len(parsed_frames)
            parsed_frames.append({"frame": frame_number, "tile_visibility": visibility})
        if parsed_frames:
            frames_by_segment[seg_idx] = parsed_frames
    return frames_by_segment


def _segment_visibility(
    segment: Mapping[str, Any],
    fallback_idx: int,
    visibility_by_segment: Mapping[int, Dict[str, float]],
    viewport_tiles: Optional[Iterable[str]],
) -> Dict[str, float]:
    if viewport_tiles is not None:
        return {str(tile_id): 1.0 for tile_id in viewport_tiles}
    return dict(visibility_by_segment.get(_segment_idx(segment, fallback_idx), {}))


def _stall_overlap_and_count(
    seg_start: float,
    seg_end: float,
    stalls: Sequence[Mapping[str, Any]],
) -> Tuple[float, int]:
    overlap = 0.0
    count = 0
    for stall in stalls:
        st = _safe_float(stall.get("time_s"), 0.0)
        et = st + max(_safe_float(stall.get("duration_s"), 0.0), 0.0)
        ov = max(0.0, min(seg_end, et) - max(seg_start, st))
        if ov > 0.0:
            overlap += ov
            count += 1
    return overlap, count


def _playback_horizon(events: Mapping[str, Any], segments: Sequence[Mapping[str, Any]]) -> float:
    last_segment_end = max(
        (_safe_float(seg.get("time_s"), 0.0) + _segment_duration(seg) for seg in segments),
        default=0.0,
    )
    consumed_end = max(
        (
            _safe_float(seg.get("time_s"), 0.0) + _safe_float(seg.get("duration_s"), 0.0)
            for seg in (events.get("consumed_segments") or [])
        ),
        default=0.0,
    )
    stall_end = max(
        (
            _safe_float(stall.get("time_s"), 0.0) + _safe_float(stall.get("duration_s"), 0.0)
            for stall in (events.get("rebuffering_events") or [])
        ),
        default=0.0,
    )
    buffer_end = max((_safe_float(t) for t in (events.get("time_s") or [])), default=0.0)
    return max(last_segment_end, consumed_end, stall_end, buffer_end)


def _experience_intervals(events: Mapping[str, Any]) -> List[Dict[str, Any]]:
    segments = _sorted_segments(events)
    if not segments:
        return []

    intervals: List[Dict[str, Any]] = []
    playback_cursor = 0.0
    media_cursor = 0.0
    for idx, seg in enumerate(segments):
        arrival_time = _safe_float(seg.get("time_s"), 0.0)
        if arrival_time > playback_cursor + 1e-9:
            intervals.append(
                {
                    "start": playback_cursor,
                    "end": arrival_time,
                    "media_start_s": media_cursor,
                    "media_end_s": media_cursor,
                    "segment": None,
                    "fallback_idx": idx - 1,
                }
            )
            playback_cursor = arrival_time

        playback_start = max(arrival_time, playback_cursor)
        seg_duration = _segment_duration(seg)
        playback_end = playback_start + seg_duration
        intervals.append(
            {
                "start": playback_start,
                "end": playback_end,
                "media_start_s": media_cursor,
                "media_end_s": media_cursor + seg_duration,
                "segment": seg,
                "fallback_idx": idx,
            }
        )
        playback_cursor = playback_end
        media_cursor += seg_duration
    return intervals


def _total_media_duration(intervals: Sequence[Mapping[str, Any]]) -> float:
    play_intervals = [interval for interval in intervals if interval.get("segment") is not None]
    if not play_intervals:
        return 0.0
    return max(_safe_float(play_intervals[-1].get("media_end_s"), 0.0), 0.0)


def _actual_fov_context(
    events: Mapping[str, Any],
    run_config: Mapping[str, Any] | None,
) -> Optional[Dict[str, Any]]:
    if not run_config:
        return None
    trace_path = _resolve_path(run_config.get("fov_trace"))
    rows = _load_fov_trace_rows(trace_path)
    if not rows:
        return None
    intervals = _experience_intervals(events)
    total_media_s = _total_media_duration(intervals)
    if total_media_s <= 1e-9:
        return None
    frame_dt = total_media_s / max(len(rows) - 1, 1)
    return {
        "rows": rows,
        "total_media_s": total_media_s,
        "frame_dt": max(frame_dt, 1e-6),
        "sample_dt": max(frame_dt, 0.05),
        "fov_h_deg": _safe_float(run_config.get("fov_h_deg"), 97.0),
        "fov_v_deg": _safe_float(run_config.get("fov_v_deg"), 93.0),
        "fov_h_rad": math.radians(_safe_float(run_config.get("fov_h_deg"), 97.0)),
        "fov_v_rad": math.radians(_safe_float(run_config.get("fov_v_deg"), 93.0)),
        "angle_convention": str(run_config.get("angle_convention") or FovPredictor.DEFAULT_ANGLE_CONVENTION).strip().lower(),
        "segment_probe_cache": {},
    }


def _trace_row_at_media_time(actual_ctx: Optional[Mapping[str, Any]], media_time_s: float) -> Optional[Mapping[str, float]]:
    if not actual_ctx:
        return None
    rows = actual_ctx.get("rows") or []
    total_media_s = _safe_float(actual_ctx.get("total_media_s"), 0.0)
    if not rows or total_media_s <= 0.0:
        return None
    ratio = _clamp01(media_time_s / total_media_s)
    idx = int(round(ratio * max(len(rows) - 1, 0)))
    return rows[min(max(idx, 0), len(rows) - 1)]


def _extract_pose_sample(
    row: Optional[Mapping[str, float]],
    angle_convention: str = FovPredictor.DEFAULT_ANGLE_CONVENTION,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if not row:
        return None, None

    position: Optional[np.ndarray] = None
    for keys in _POSITION_KEY_CANDIDATES:
        if all(key in row for key in keys):
            position = np.asarray([_safe_float(row[key]) for key in keys], dtype=float)
            break

    forward: Optional[np.ndarray] = None
    try:
        forward = forward_from_quaternion_xyzw(quaternion_from_row(row, angle_convention=angle_convention))
    except (TypeError, ValueError):
        forward = None

    return position, forward


def _tile_probe_points(tile: Mapping[str, Any]) -> np.ndarray:
    bounds = tile.get("bounds") or {}
    min_corner = np.asarray(bounds.get("min_corner") or (0.0, 0.0, 0.0), dtype=float)
    max_corner = np.asarray(bounds.get("max_corner") or (0.0, 0.0, 0.0), dtype=float)
    center = (min_corner + max_corner) * 0.5
    corners = np.asarray(
        [
            [min_corner[0], min_corner[1], min_corner[2]],
            [min_corner[0], min_corner[1], max_corner[2]],
            [min_corner[0], max_corner[1], min_corner[2]],
            [min_corner[0], max_corner[1], max_corner[2]],
            [max_corner[0], min_corner[1], min_corner[2]],
            [max_corner[0], min_corner[1], max_corner[2]],
            [max_corner[0], max_corner[1], min_corner[2]],
            [max_corner[0], max_corner[1], max_corner[2]],
            center,
        ],
        dtype=float,
    )
    return corners


def _actual_visibility_for_segment(
    segment: Mapping[str, Any],
    row: Optional[Mapping[str, float]],
    actual_ctx: Optional[Mapping[str, Any]],
) -> Dict[str, float]:
    if not row or not actual_ctx:
        return {}

    angle_convention = str(actual_ctx.get("angle_convention") or FovPredictor.DEFAULT_ANGLE_CONVENTION)
    position, forward = _extract_pose_sample(row, angle_convention=angle_convention)
    if position is None or forward is None:
        return {}

    fov_h = _safe_float(actual_ctx.get("fov_h_rad"), math.radians(97.0))
    fov_v = _safe_float(actual_ctx.get("fov_v_rad"), math.radians(93.0))

    probe_cache = actual_ctx.get("segment_probe_cache")
    segment_key = id(segment)
    probe_data = probe_cache.get(segment_key) if isinstance(probe_cache, dict) else None
    if probe_data is None:
        tile_ids: List[str] = []
        slices: List[Tuple[int, int]] = []
        point_chunks: List[np.ndarray] = []
        offset = 0
        for tile_id, tile in (segment.get("tiles") or {}).items():
            points = _tile_probe_points(tile)
            if points.size == 0:
                continue
            count = int(points.shape[0])
            tile_ids.append(str(tile_id))
            slices.append((offset, offset + count))
            point_chunks.append(points)
            offset += count
        points_all = np.vstack(point_chunks) if point_chunks else np.empty((0, 3), dtype=float)
        probe_data = (tile_ids, slices, points_all)
        if isinstance(probe_cache, dict):
            probe_cache[segment_key] = probe_data

    tile_ids, slices, points_all = probe_data
    if not tile_ids or points_all.size == 0:
        return {}

    mask_all = FovPredictor._points_in_fov(
        points_all,
        position=position,
        forward=forward,
        euler_angles=None,
        fov_h=fov_h,
        fov_v=fov_v,
        angle_convention=angle_convention,
    )

    visibility: Dict[str, float] = {}
    for tile_id, (start, end) in zip(tile_ids, slices):
        tile_mask = mask_all[start:end]
        visibility[tile_id] = float(np.mean(tile_mask)) if tile_mask.size else 0.0
    return visibility


def _sample_media_times(interval: Mapping[str, Any], actual_ctx: Optional[Mapping[str, Any]]) -> List[float]:
    start_s = _safe_float(interval.get("media_start_s"), 0.0)
    end_s = max(_safe_float(interval.get("media_end_s"), start_s), start_s)
    if end_s <= start_s + 1e-9:
        return [start_s]
    if not actual_ctx:
        return [start_s, end_s]

    step = _safe_float(actual_ctx.get("sample_dt"), 0.1)
    times = [start_s]
    current = math.floor(start_s / step) * step
    if current <= start_s + 1e-9:
        current += step
    while current < end_s - 1e-9:
        times.append(current)
        current += step
    times.append(end_s)
    out: List[float] = []
    for value in times:
        if not out or abs(value - out[-1]) > 1e-9:
            out.append(value)
    return out


def _sample_media_times_for_frames(interval: Mapping[str, Any], frame_count: int) -> List[float]:
    start_s = _safe_float(interval.get("media_start_s"), 0.0)
    end_s = max(_safe_float(interval.get("media_end_s"), start_s), start_s)
    count = max(int(frame_count), 0)
    if count <= 1 or end_s <= start_s + 1e-9:
        return [start_s]
    duration = end_s - start_s
    return [start_s + duration * (idx / float(count - 1)) for idx in range(count)]


def _visibility_for_frame_sample(
    frames: Sequence[Mapping[str, Any]],
    media_time_s: float,
    segment_media_start_s: float,
    segment_duration_s: float,
) -> Dict[str, float]:
    if not frames:
        return {}
    if len(frames) == 1 or segment_duration_s <= 1e-9:
        idx = 0
    else:
        ratio = _clamp01((media_time_s - segment_media_start_s) / segment_duration_s)
        idx = int(round(ratio * (len(frames) - 1)))
    raw_visibility = frames[max(0, min(idx, len(frames) - 1))].get("tile_visibility") or {}
    if not isinstance(raw_visibility, Mapping):
        return {}
    return {
        str(tile_id): _clamp01(_safe_float(score))
        for tile_id, score in raw_visibility.items()
    }


def _relax_toward(prev_value: Optional[float], target_value: float, dt_s: float, tau_s: float) -> float:
    if prev_value is None:
        return target_value
    tau = max(tau_s, 1e-6)
    dt = max(dt_s, 0.0)
    alpha = 1.0 - math.exp(-dt / tau)
    return prev_value + alpha * (target_value - prev_value)


def _collapse_same_time_points(series: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not series:
        return []
    out: List[Tuple[float, float]] = []
    for time_s, score in sorted(series, key=lambda item: item[0]):
        if out and abs(time_s - out[-1][0]) < 1e-9:
            out[-1] = (time_s, score)
        else:
            out.append((time_s, score))
    return out


def _post_stall_penalty(
    time_s: float,
    stalls: Sequence[Mapping[str, Any]],
    stall_per_second: float,
    stall_fixed: float,
    tau_s: float,
) -> float:
    if not stalls:
        return 0.0
    penalty = 0.0
    tau = max(tau_s, 1e-6)
    for stall in stalls:
        start_s = _safe_float(stall.get("time_s"), 0.0)
        duration_s = max(_safe_float(stall.get("duration_s"), 0.0), 0.0)
        end_s = start_s + duration_s
        if time_s < end_s - 1e-9:
            continue
        amplitude = max(0.0, stall_fixed + stall_per_second * duration_s)
        if amplitude <= 0.0:
            continue
        decay = math.exp(-(time_s - end_s) / tau)
        penalty += amplitude * decay
    return penalty


class _BitrateQoEBase(QoEFunction):
    """Shared helpers for bitrate-aware QoE models."""

    _scoring_version = "visible_tiles_only_v2"
    _bitrate_alpha: float = 6.0
    _buffer_target_segments: float = 1.5
    _buffer_target_cap_s: float = 6.0
    _buffer_exponent: float = 1.0
    _utility_floor: float = 1e-3
    _perceptual_tau_s: float = 0.35
    _stall_memory_tau_s: float = 0.9

    def __init__(self, weights: Optional[QoEWeights] = None, overrides: Mapping[str, Any] | None = None):
        self.weights = weights or QoEWeights()
        self.scoring_version = self._scoring_version
        self.apply_overrides(overrides)

    def apply_overrides(self, overrides: Mapping[str, Any] | None = None) -> None:
        if not overrides:
            return
        comp = self.weights.component_weights
        for key in ["attr", "geom", "occp", "atlas"]:
            if key in overrides:
                try:
                    comp[key] = float(overrides[key])
                except (TypeError, ValueError):
                    continue
        if "default_component" in overrides or "default" in overrides:
            try:
                self.weights.default_component = float(overrides.get("default_component", overrides.get("default")))
            except (TypeError, ValueError):
                pass
        for attr in ["fov_saliency", "temporal_penalty", "stall_per_second", "stall_fixed"]:
            if attr in overrides:
                try:
                    setattr(self.weights, attr, float(overrides[attr]))
                except (TypeError, ValueError):
                    continue

    def _component_weight(self, name: str) -> float:
        return self.weights.component_weights.get(name, self.weights.default_component)

    def _tile_weight(self, tile_id: str, visibility: Mapping[str, float]) -> float:
        base = self.weights.tile_saliency.get(tile_id, 1.0)
        if base <= 0.0:
            return 0.0
        if not visibility:
            return base

        vis_score = _clamp01(_safe_float(visibility.get(tile_id), 0.0))
        if vis_score <= 0.0:
            return 0.0

        saliency = max(_safe_float(self.weights.fov_saliency, 1.0), 1e-6)
        return base * (vis_score ** (1.0 / saliency))

    def _component_utility(
        self,
        comp_name: str,
        comp: Mapping[str, Any],
        max_levels: Mapping[str, int],
        bitrate_ranges: Mapping[str, Tuple[float, float]],
    ) -> float:
        bitrate_bps = _component_bitrate_bps(comp)
        low, high = bitrate_ranges.get(comp_name, (0.0, 0.0))

        if bitrate_bps > 0.0 and low > 0.0 and high > low:
            log_low = math.log(max(low, 1.0))
            log_high = math.log(max(high, low * 1.01))
            if log_high > log_low:
                norm = _clamp01((math.log(max(bitrate_bps, 1.0)) - log_low) / (log_high - log_low))
            else:
                norm = _component_rep_norm(comp, max_levels, comp_name)
        else:
            norm = _component_rep_norm(comp, max_levels, comp_name)

        if norm <= 0.0:
            return 0.0

        alpha = self._bitrate_alpha
        return math.log1p(alpha * norm) / math.log1p(alpha)

    def _segment_quality_linear(
        self,
        segment: Mapping[str, Any],
        max_levels: Mapping[str, int],
        bitrate_ranges: Mapping[str, Tuple[float, float]],
        visibility: Mapping[str, float],
    ) -> float:
        return self._segment_quality_from_tile_map(
            self._segment_tile_quality_linear_map(segment, max_levels, bitrate_ranges),
            visibility,
        )

    def _segment_tile_quality_linear_map(
        self,
        segment: Mapping[str, Any],
        max_levels: Mapping[str, int],
        bitrate_ranges: Mapping[str, Tuple[float, float]],
    ) -> Dict[str, float]:
        qualities: Dict[str, float] = {}
        for tile_id, tile in (segment.get("tiles") or {}).items():
            comp_sum = 0.0
            comp_weights = 0.0
            for comp_name, comp in (tile.get("components") or {}).items():
                cw = self._component_weight(comp_name)
                if cw <= 0.0:
                    continue
                comp_sum += self._component_utility(comp_name, comp, max_levels, bitrate_ranges) * cw
                comp_weights += cw
            qualities[str(tile_id)] = (comp_sum / comp_weights) if comp_weights > 0.0 else 0.0
        return qualities

    def _segment_quality_from_tile_map(
        self,
        tile_qualities: Mapping[str, float],
        visibility: Mapping[str, float],
    ) -> float:
        weighted_sum = 0.0
        weight_total = 0.0
        for tile_id, tile_quality in tile_qualities.items():
            tile_w = self._tile_weight(str(tile_id), visibility)
            weighted_sum += tile_quality * tile_w
            weight_total += tile_w
        return weighted_sum / weight_total if weight_total > 0.0 else 0.0

    def _segment_quality_harmonic(
        self,
        segment: Mapping[str, Any],
        max_levels: Mapping[str, int],
        bitrate_ranges: Mapping[str, Tuple[float, float]],
        visibility: Mapping[str, float],
    ) -> float:
        return self._segment_quality_from_tile_map(
            self._segment_tile_quality_harmonic_map(segment, max_levels, bitrate_ranges),
            visibility,
        )

    def _segment_tile_quality_harmonic_map(
        self,
        segment: Mapping[str, Any],
        max_levels: Mapping[str, int],
        bitrate_ranges: Mapping[str, Tuple[float, float]],
    ) -> Dict[str, float]:
        qualities: Dict[str, float] = {}
        for tile_id, tile in (segment.get("tiles") or {}).items():
            comp_weights = 0.0
            denom = 0.0
            for comp_name, comp in (tile.get("components") or {}).items():
                cw = self._component_weight(comp_name)
                if cw <= 0.0:
                    continue
                util = self._component_utility(comp_name, comp, max_levels, bitrate_ranges)
                comp_weights += cw
                denom += cw / max(util, self._utility_floor)
            tile_quality = (comp_weights / denom) if comp_weights > 0.0 and denom > 0.0 else 0.0
            qualities[str(tile_id)] = tile_quality
        return qualities

    def _buffer_risk(self, buffer_s: Optional[float], seg_duration_s: float) -> float:
        if buffer_s is None:
            return 0.0
        target_s = min(self._buffer_target_cap_s, max(seg_duration_s, self._buffer_target_segments * seg_duration_s))
        if target_s <= 0.0:
            return 0.0
        deficit = max(0.0, target_s - max(buffer_s, 0.0))
        risk = deficit / target_s
        return _clamp01(risk ** self._buffer_exponent)

    def _run_context(self, events: Mapping[str, Any], run_config: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        buffer_times, buffer_values = _buffer_samples(events)
        max_levels = _max_component_levels(events)
        mpd_max_levels = _mpd_component_max_levels(run_config)
        if mpd_max_levels:
            max_levels.update({comp: max(max_levels.get(comp, 0), level) for comp, level in mpd_max_levels.items()})
        return {
            "segments": _sorted_segments(events),
            "max_levels": max_levels,
            # Every run of one content shares this MPD-defined representation
            # ladder.  Never derive the normalization range from the selected
            # representations of an individual run.
            "bitrate_ranges": _mpd_component_bitrate_ranges(run_config),
            "stalls": events.get("rebuffering_events") or [],
            "buffer_times": buffer_times,
            "buffer_values": buffer_values,
            "visibility_by_segment": _segment_visibility_maps(events),
            "ground_truth_visibility_by_segment": _segment_ground_truth_visibility_maps(events),
            "ground_truth_frames_by_segment": _segment_ground_truth_frame_maps(events),
        }

    def _playback_samples(
        self,
        events: Mapping[str, Any],
        run_config: Mapping[str, Any] | None,
        viewport_tiles: Optional[Iterable[str]],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        context = self._run_context(events, run_config=run_config)
        intervals = _experience_intervals(events)
        actual_ctx = _actual_fov_context(events, run_config)
        samples: List[Dict[str, Any]] = []

        for interval in intervals:
            wall_start = _safe_float(interval.get("start"), 0.0)
            wall_end = max(_safe_float(interval.get("end"), wall_start), wall_start)
            seg = interval.get("segment")
            fallback_idx = int(interval.get("fallback_idx", -1))

            if seg is None:
                samples.append({"time_s": wall_start, "segment": None, "segment_idx": fallback_idx, "visibility": {}, "buffer_s": 0.0})
                samples.append({"time_s": wall_end, "segment": None, "segment_idx": fallback_idx, "visibility": {}, "buffer_s": 0.0})
                continue

            media_start_s = _safe_float(interval.get("media_start_s"), 0.0)
            ground_truth_frames = context["ground_truth_frames_by_segment"].get(_segment_idx(seg, fallback_idx), [])
            media_times = (
                _sample_media_times_for_frames(interval, len(ground_truth_frames))
                if ground_truth_frames
                else _sample_media_times(interval, actual_ctx)
            )
            ground_truth_visibility = _segment_visibility(
                seg,
                fallback_idx,
                context["ground_truth_visibility_by_segment"],
                None,
            )
            for media_time_s in media_times:
                wall_time_s = wall_start + max(0.0, media_time_s - media_start_s)
                visibility = _visibility_for_frame_sample(
                    ground_truth_frames,
                    media_time_s,
                    media_start_s,
                    _segment_duration(seg),
                )
                if not visibility:
                    visibility = dict(ground_truth_visibility)
                if not visibility:
                    actual_row = _trace_row_at_media_time(actual_ctx, media_time_s)
                    visibility = _actual_visibility_for_segment(seg, actual_row, actual_ctx)
                if not visibility:
                    visibility = _segment_visibility(seg, fallback_idx, context["visibility_by_segment"], viewport_tiles)
                samples.append(
                    {
                        "time_s": wall_time_s,
                        "segment": seg,
                        "segment_idx": _segment_idx(seg, fallback_idx),
                        "visibility": visibility,
                        "buffer_s": _buffer_at_time(context["buffer_times"], context["buffer_values"], wall_time_s),
                    }
                )
        return context, samples


class WeightedQoEHeuristic(_BitrateQoEBase):
    """Balanced bitrate-aware QoE with weakest-link sensitivity inside a tile."""

    id = "WeightedQoEHeuristic"
    label = "Weighted harmonic QoE"
    equation = "QoE = harmonic(visible-tile bitrate utility) - smoothness - low_buffer - stall"
    description = (
        "Scores delivered bitrate with log utility over visible tiles only, punishes sharp quality drops, "
        "and subtracts low-buffer and stall penalties. Invisible tiles receive no quality credit when "
        "visibility data is available."
    )
    implementation = "weighted"

    _bitrate_alpha = 7.0
    _buffer_target_segments = 1.75
    _buffer_target_cap_s = 8.0
    _buffer_exponent = 1.6
    _temporal_smoothing_rho = 0.45
    _lambda_down_factor = 1.15
    _lambda_up_factor = 0.25
    _stall_saturation_k = 1.7
    _stall_count_penalty = 0.03
    _buffer_penalty_scale = 0.18
    _perceptual_tau_s = 0.45
    _stall_memory_tau_s = 1.1

    @classmethod
    def ui_fields(cls) -> Sequence[UiField]:
        return [
            UiField("attr", "Attr weight", 0.35, min=0.0, step=0.01, help="Relative importance of attribute quality"),
            UiField("geom", "Geom weight", 0.35, min=0.0, step=0.01, help="Relative importance of geometry quality"),
            UiField("occp", "Occp weight", 0.2, min=0.0, step=0.01, help="Relative importance of occupancy quality"),
            UiField("atlas", "Atlas weight", 0.0, min=0.0, step=0.01, help="Atlas is ignored in QoE unless you explicitly weight it"),
            UiField("default_component", "Default component weight", 0.1, min=0.0, step=0.01, help="Fallback for unknown components"),
            UiField("fov_saliency", "FoV saliency", 1.25, min=0.0, step=0.05, help="Shapes visible-tile weights; invisible tiles score zero"),
            UiField("temporal_penalty", "Temporal penalty", 0.2, min=0.0, step=0.01, help="Penalty scale for visible quality changes"),
            UiField("stall_per_second", "Stall /s penalty", 0.15, min=0.0, step=0.01, help="Per-second stall penalty (saturating)"),
            UiField("stall_fixed", "Stall fixed penalty", 0.05, min=0.0, step=0.01, help="Penalty added when a stall overlaps a segment"),
        ]

    def compute_time_series(
        self,
        events: Mapping[str, Any],
        viewport_tiles: Optional[Iterable[str]] = None,
        run_config: Mapping[str, Any] | None = None,
    ) -> List[Tuple[float, float]]:
        started = time.perf_counter()
        context, samples = self._playback_samples(events, run_config, viewport_tiles)
        sample_ms = int((time.perf_counter() - started) * 1000)
        score_started = time.perf_counter()
        max_levels = context["max_levels"]
        bitrate_ranges = context["bitrate_ranges"]
        stalls = context["stalls"]

        results: List[Tuple[float, float]] = []
        prev_base_quality: Optional[float] = None
        prev_eff_quality: Optional[float] = None
        prev_time: Optional[float] = None
        tile_quality_cache: Dict[int, Dict[str, float]] = {}

        for sample in samples:
            sample_time = _safe_float(sample.get("time_s"), 0.0)
            seg = sample.get("segment")
            if seg is None:
                results.append((sample_time, 0.0))
                prev_base_quality = 0.0
                prev_eff_quality = 0.0
                prev_time = sample_time
                continue

            seg_duration = _segment_duration(seg)
            visibility = sample.get("visibility") or {}
            seg_key = id(seg)
            tile_qualities = tile_quality_cache.get(seg_key)
            if tile_qualities is None:
                tile_qualities = self._segment_tile_quality_harmonic_map(seg, max_levels, bitrate_ranges)
                tile_quality_cache[seg_key] = tile_qualities
            base_quality = self._segment_quality_from_tile_map(tile_qualities, visibility)
            dt_s = 0.0 if prev_time is None else max(0.0, sample_time - prev_time)

            eff_quality = _relax_toward(prev_eff_quality, base_quality, dt_s, self._perceptual_tau_s)

            penalty = 0.0
            if prev_base_quality is not None:
                delta = eff_quality - prev_eff_quality if prev_eff_quality is not None else base_quality - prev_base_quality
                down = max(0.0, -delta)
                up = max(0.0, delta)
                penalty += self.weights.temporal_penalty * (
                    self._lambda_down_factor * (down ** 2) + self._lambda_up_factor * (up ** 2)
                )

            buffer_s = sample.get("buffer_s")
            buffer_risk = self._buffer_risk(buffer_s, seg_duration)
            penalty += self._buffer_penalty_scale * (buffer_risk ** 2)
            penalty += _post_stall_penalty(
                sample_time,
                stalls,
                self.weights.stall_per_second,
                self.weights.stall_fixed,
                self._stall_memory_tau_s,
            )

            score = _clamp01(eff_quality - penalty)
            results.append((sample_time, score))
            prev_base_quality = base_quality
            prev_eff_quality = eff_quality
            prev_time = sample_time

        collapsed = _collapse_same_time_points(results)
        print(
            "[graph-timing] qoe_weighted "
            f"samples={len(samples)} output={len(collapsed)} sample_ms={sample_ms} "
            f"score_ms={int((time.perf_counter() - score_started) * 1000)} "
            f"elapsed_ms={int((time.perf_counter() - started) * 1000)}"
        )
        return collapsed


class LinearQoE(_BitrateQoEBase):
    """Transparent bitrate-aware baseline with linear penalties."""

    id = "LinearQoE"
    label = "Linear QoE"
    equation = "QoE = avg(visible-tile bitrate utility) - smoothness - low_buffer - stall"
    description = (
        "Simple arithmetic average of visible-tile quality with linear smoothness, low-buffer, "
        "and stall penalties. Invisible tiles receive no quality credit when visibility data is available."
    )
    implementation = "linear"

    _bitrate_alpha = 4.0
    _buffer_target_segments = 1.35
    _buffer_target_cap_s = 6.0
    _buffer_exponent = 1.0
    _buffer_penalty_scale = 0.12
    _perceptual_tau_s = 0.30
    _stall_memory_tau_s = 0.85

    @classmethod
    def ui_fields(cls) -> Sequence[UiField]:
        return [
            UiField("fov_saliency", "FoV saliency", 1.0, min=0.0, step=0.05, help="Shapes visible-tile weights; invisible tiles score zero"),
            UiField("temporal_penalty", "Temporal penalty", 0.12, min=0.0, step=0.01, help="Smoothness penalty scale"),
            UiField("stall_per_second", "Stall /s penalty", 0.12, min=0.0, step=0.01, help="Linear stall penalty per second"),
            UiField("stall_fixed", "Stall fixed penalty", 0.03, min=0.0, step=0.01, help="Penalty once any stall occurs"),
        ]

    @classmethod
    def defaults(cls) -> Dict[str, float]:
        return {
            "attr": 0.33,
            "geom": 0.33,
            "occp": 0.2,
            "atlas": 0.0,
            "default_component": 0.1,
            "fov_saliency": 1.0,
            "temporal_penalty": 0.12,
            "stall_per_second": 0.12,
            "stall_fixed": 0.03,
        }

    def compute_time_series(
        self,
        events: Mapping[str, Any],
        viewport_tiles: Optional[Iterable[str]] = None,
        run_config: Mapping[str, Any] | None = None,
    ) -> List[Tuple[float, float]]:
        started = time.perf_counter()
        context, samples = self._playback_samples(events, run_config, viewport_tiles)
        sample_ms = int((time.perf_counter() - started) * 1000)
        score_started = time.perf_counter()
        max_levels = context["max_levels"]
        bitrate_ranges = context["bitrate_ranges"]
        stalls = context["stalls"]

        results: List[Tuple[float, float]] = []
        prev_perceived_quality: Optional[float] = None
        prev_time: Optional[float] = None
        tile_quality_cache: Dict[int, Dict[str, float]] = {}

        for sample in samples:
            sample_time = _safe_float(sample.get("time_s"), 0.0)
            seg = sample.get("segment")
            if seg is None:
                results.append((sample_time, 0.0))
                prev_perceived_quality = 0.0
                prev_time = sample_time
                continue

            seg_duration = _segment_duration(seg)
            visibility = sample.get("visibility") or {}
            seg_key = id(seg)
            tile_qualities = tile_quality_cache.get(seg_key)
            if tile_qualities is None:
                tile_qualities = self._segment_tile_quality_linear_map(seg, max_levels, bitrate_ranges)
                tile_quality_cache[seg_key] = tile_qualities
            base_quality = self._segment_quality_from_tile_map(tile_qualities, visibility)
            dt_s = 0.0 if prev_time is None else max(0.0, sample_time - prev_time)
            perceived_quality = _relax_toward(prev_perceived_quality, base_quality, dt_s, self._perceptual_tau_s)

            penalty = 0.0
            if prev_perceived_quality is not None:
                penalty += self.weights.temporal_penalty * max(0.0, prev_perceived_quality - perceived_quality)

            buffer_s = sample.get("buffer_s")
            penalty += self._buffer_penalty_scale * self._buffer_risk(buffer_s, seg_duration)
            penalty += _post_stall_penalty(
                sample_time,
                stalls,
                self.weights.stall_per_second,
                self.weights.stall_fixed,
                self._stall_memory_tau_s,
            )

            score = _clamp01(perceived_quality - penalty)
            results.append((sample_time, score))
            prev_perceived_quality = perceived_quality
            prev_time = sample_time

        collapsed = _collapse_same_time_points(results)
        print(
            "[graph-timing] qoe_linear "
            f"samples={len(samples)} output={len(collapsed)} sample_ms={sample_ms} "
            f"score_ms={int((time.perf_counter() - score_started) * 1000)} "
            f"elapsed_ms={int((time.perf_counter() - started) * 1000)}"
        )
        return collapsed


class VisiblePlaybackUtility(_BitrateQoEBase):
    """Transparent per-instant visible-quality utility for paper reporting.

    The reported paper value is the time average of this series.  Unlike the
    interactive QoE heuristics, it contains no buffer-risk, stall-memory, or
    quality-switch penalty: those are reported separately when relevant.
    This makes it an auditable measure of delivered quality on the tiles that
    were actually visible.
    """

    id = "VisiblePlaybackUtility"
    label = "Mean visible playback utility"
    equation = "Q = U_visible * (0.75 + 0.25 * exp(-mean |log(B_s/B_{s-1})| / log(2)))"
    description = (
        "MPD-normalized visible-tile utility with a 25% bounded reward for stable "
        "visibility-weighted bitrate across adjacent segments. The stability term cannot "
        "make low visible quality score well; stalls remain a separate measurement."
    )
    implementation = "visible_playback_utility"
    _scoring_version = "mpd_stable_visible_utility_v1"
    _bitrate_alpha = 6.0

    def compute_time_series(
        self,
        events: Mapping[str, Any],
        viewport_tiles: Optional[Iterable[str]] = None,
        run_config: Mapping[str, Any] | None = None,
    ) -> List[Tuple[float, float]]:
        context, samples = self._playback_samples(events, run_config, viewport_tiles)
        max_levels = context["max_levels"]
        bitrate_ranges = context["bitrate_ranges"]
        if not bitrate_ranges:
            raise ValueError("VisiblePlaybackUtility requires an accessible MPD with representation bandwidths.")
        tile_quality_cache: Dict[int, Dict[str, float]] = {}
        segment_bitrates: list[float] = []
        segment_utilities: list[float] = []
        results: List[Tuple[float, float]] = []
        for sample in samples:
            seg = sample.get("segment")
            time_s = _safe_float(sample.get("time_s"), 0.0)
            if seg is None:
                results.append((time_s, 0.0))
                continue
            key = id(seg)
            tile_qualities = tile_quality_cache.get(key)
            if tile_qualities is None:
                tile_qualities = self._segment_tile_quality_linear_map(seg, max_levels, bitrate_ranges)
                tile_quality_cache[key] = tile_qualities
            visibility = sample.get("visibility") or {}
            utility = self._segment_quality_from_tile_map(tile_qualities, visibility)
            results.append((time_s, utility))
            # The segment-level stability statistic is computed once per
            # enqueued segment below, not from individual frame samples.

        for index, segment in enumerate(context["segments"]):
            visibility = _segment_visibility(
                segment, index, context["ground_truth_visibility_by_segment"], None
            )
            qualities = self._segment_tile_quality_linear_map(segment, max_levels, bitrate_ranges)
            segment_utilities.append(self._segment_quality_from_tile_map(qualities, visibility))
            visible_bits = 0.0
            visible_weight = 0.0
            for tile_id, tile in (segment.get("tiles") or {}).items():
                weight = self._tile_weight(str(tile_id), visibility)
                if weight <= 0.0:
                    continue
                component_bits = sum(
                    max(_component_bitrate_bps(component), 0.0)
                    for component in (tile.get("components") or {}).values()
                )
                visible_bits += weight * component_bits
                visible_weight += weight
            if visible_weight > 0.0:
                segment_bitrates.append(visible_bits / visible_weight)

        if len(segment_bitrates) >= 2:
            changes = [abs(math.log((current + 1.0) / (previous + 1.0)))
                       for previous, current in zip(segment_bitrates, segment_bitrates[1:])]
            stability = math.exp(-float(np.mean(changes)) / math.log(2.0))
        else:
            stability = 1.0
        multiplier = 0.75 + 0.25 * _clamp01(stability)
        return [(time_s, _clamp01(utility * multiplier)) for time_s, utility in _collapse_same_time_points(results)]


class FovAwarePlaybackUtility(_BitrateQoEBase):
    """Playback utility that makes FoV allocation errors and flicker explicit.

    This metric is intended for the matched bandwidth study, where the
    question is whether a predictor puts the available quality where the
    viewer actually looks.  It uses ground-truth visibility only; predicted
    tile scores never enter the calculation.

    A missing-quality term dominates the score.  Quality on invisible tiles
    is a smaller opportunity-cost penalty, while a third term penalizes a
    representation change for tiles visible on both sides of a segment
    boundary.  Playback gaps retain a score of zero through
    :meth:`_playback_samples`, so rebuffering remains part of the time mean.
    """

    id = "FovAwarePlaybackUtility"
    label = "FoV-aware playback utility"
    equation = "Q = 1 - (0.78 M_{FoV} + 0.12 W_{non-FoV} + 0.10 Delta_{FoV})"
    description = (
        "Ground-truth-FoV utility with a dominant visible-quality-deficit term, "
        "a smaller high-quality non-FoV cost, and a penalty for quality changes "
        "on tiles visible across consecutive segments. The first two segments are "
        "an evaluation-only predictor warm-up and are excluded by default."
    )
    implementation = "fov_aware_playback_utility"
    _scoring_version = "fov_aware_playback_utility_v2"
    _bitrate_alpha = 6.0
    _miss_weight = 0.78
    _waste_weight = 0.12
    _switch_weight = 0.10
    _default_warmup_segments = 2

    @staticmethod
    def _warmup_segments(run_config: Mapping[str, Any] | None) -> int:
        """Return the explicit evaluation-only cold-start prefix length."""
        if not run_config or "analysis_qoe_warmup_segments" not in run_config:
            return FovAwarePlaybackUtility._default_warmup_segments
        return max(0, int(_safe_float(run_config.get("analysis_qoe_warmup_segments"), 0.0)))

    def _segment_switch_penalties(
        self,
        context: Mapping[str, Any],
        max_levels: Mapping[str, int],
        bitrate_ranges: Mapping[str, Tuple[float, float]],
        warmup_segments: int = 0,
    ) -> Dict[int, float]:
        """Return persistent-FoV quality-change penalties by segment identity."""
        penalties: Dict[int, float] = {}
        previous_qualities: Optional[Dict[str, float]] = None
        previous_visibility: Optional[Dict[str, float]] = None
        for index, segment in enumerate(context["segments"]):
            segment_idx = _segment_idx(segment, index)
            if segment_idx < warmup_segments:
                # The first scored segment begins a new stability history, so
                # a warm-up representation cannot affect its switch penalty.
                previous_qualities = None
                previous_visibility = None
                continue
            qualities = self._segment_tile_quality_linear_map(segment, max_levels, bitrate_ranges)
            visibility = _segment_visibility(
                segment, index, context["ground_truth_visibility_by_segment"], None
            )
            penalty = 0.0
            if previous_qualities is not None and previous_visibility is not None:
                weighted_change = 0.0
                persistent_weight = 0.0
                for tile_id, quality in qualities.items():
                    overlap = min(
                        _clamp01(_safe_float(visibility.get(tile_id), 0.0)),
                        _clamp01(_safe_float(previous_visibility.get(tile_id, 0.0))),
                    )
                    if overlap <= 0.0:
                        continue
                    weighted_change += overlap * abs(quality - previous_qualities.get(tile_id, 0.0))
                    persistent_weight += overlap
                if persistent_weight > 0.0:
                    penalty = weighted_change / persistent_weight
            penalties[id(segment)] = _clamp01(penalty)
            previous_qualities = qualities
            previous_visibility = visibility
        return penalties

    def compute_time_series(
        self,
        events: Mapping[str, Any],
        viewport_tiles: Optional[Iterable[str]] = None,
        run_config: Mapping[str, Any] | None = None,
    ) -> List[Tuple[float, float]]:
        context, samples = self._playback_samples(events, run_config, viewport_tiles)
        max_levels = context["max_levels"]
        bitrate_ranges = context["bitrate_ranges"]
        if not bitrate_ranges:
            raise ValueError("FovAwarePlaybackUtility requires an accessible MPD with representation bandwidths.")

        warmup_segments = self._warmup_segments(run_config)
        switch_penalties = self._segment_switch_penalties(
            context, max_levels, bitrate_ranges, warmup_segments
        )
        tile_quality_cache: Dict[int, Dict[str, float]] = {}
        results: List[Tuple[float, float]] = []
        for sample in samples:
            segment = sample.get("segment")
            segment_idx = int(sample.get("segment_idx", -1))
            if warmup_segments and segment_idx < warmup_segments:
                continue
            time_s = _safe_float(sample.get("time_s"), 0.0)
            if segment is None:
                results.append((time_s, 0.0))
                continue

            key = id(segment)
            qualities = tile_quality_cache.get(key)
            if qualities is None:
                qualities = self._segment_tile_quality_linear_map(segment, max_levels, bitrate_ranges)
                tile_quality_cache[key] = qualities
            visibility = sample.get("visibility") or {}
            visible_quality = self._segment_quality_from_tile_map(qualities, visibility)
            visible_deficit = 1.0 - visible_quality
            non_fov_quality = sum(
                quality * (1.0 - _clamp01(_safe_float(visibility.get(tile_id), 0.0)))
                for tile_id, quality in qualities.items()
            ) / max(len(qualities), 1)
            loss = (
                self._miss_weight * visible_deficit
                + self._waste_weight * non_fov_quality
                + self._switch_weight * switch_penalties.get(key, 0.0)
            )
            results.append((time_s, _clamp01(1.0 - loss)))
        return _collapse_same_time_points(results)


class StallAwareQoE(_BitrateQoEBase):
    """QoE model that heavily punishes impending or actual playback failure."""

    id = "StallAwareQoE"
    label = "Stall-aware QoE"
    equation = "QoE = quality * exp(-(stall + low_buffer)) - smoothness"
    description = (
        "Favors stable playback over aggressive quality by sharply discounting low-buffer and stall-heavy "
        "segments, using only visible tiles for quality credit when visibility data is available."
    )
    implementation = "stall_first"

    _bitrate_alpha = 5.0
    _buffer_target_segments = 2.0
    _buffer_target_cap_s = 10.0
    _buffer_exponent = 1.25
    _buffer_decay_strength = 1.35
    _stall_decay_strength = 1.8
    _perceptual_tau_s = 0.35
    _stall_memory_tau_s = 1.25

    def __init__(self, weights: Optional[QoEWeights] = None, overrides: Mapping[str, Any] | None = None):
        tuned = weights or QoEWeights(
            component_weights={"attr": 0.35, "geom": 0.35, "occp": 0.2, "atlas": 0.0},
            fov_saliency=1.1,
            temporal_penalty=0.15,
            stall_per_second=0.25,
            stall_fixed=0.08,
            default_component=0.1,
        )
        super().__init__(tuned, overrides)

    @classmethod
    def ui_fields(cls) -> Sequence[UiField]:
        return [
            UiField("fov_saliency", "FoV saliency", 1.1, min=0.0, step=0.05, help="Shapes visible-tile weights; invisible tiles score zero"),
            UiField("temporal_penalty", "Temporal penalty", 0.15, min=0.0, step=0.01, help="Smoothness penalty scale"),
            UiField("stall_per_second", "Stall /s penalty", 0.25, min=0.0, step=0.01, help="Aggressive stall penalty per second"),
            UiField("stall_fixed", "Stall fixed penalty", 0.08, min=0.0, step=0.01, help="Penalty once any stall occurs"),
        ]

    @classmethod
    def defaults(cls) -> Dict[str, float]:
        return {
            "attr": 0.35,
            "geom": 0.35,
            "occp": 0.2,
            "atlas": 0.0,
            "default_component": 0.1,
            "fov_saliency": 1.1,
            "temporal_penalty": 0.15,
            "stall_per_second": 0.25,
            "stall_fixed": 0.08,
        }

    def compute_time_series(
        self,
        events: Mapping[str, Any],
        viewport_tiles: Optional[Iterable[str]] = None,
        run_config: Mapping[str, Any] | None = None,
    ) -> List[Tuple[float, float]]:
        started = time.perf_counter()
        context, samples = self._playback_samples(events, run_config, viewport_tiles)
        sample_ms = int((time.perf_counter() - started) * 1000)
        score_started = time.perf_counter()
        max_levels = context["max_levels"]
        bitrate_ranges = context["bitrate_ranges"]
        stalls = context["stalls"]

        results: List[Tuple[float, float]] = []
        prev_perceived_quality: Optional[float] = None
        prev_time: Optional[float] = None
        tile_quality_cache: Dict[int, Dict[str, float]] = {}

        for sample in samples:
            sample_time = _safe_float(sample.get("time_s"), 0.0)
            seg = sample.get("segment")
            if seg is None:
                results.append((sample_time, 0.0))
                prev_perceived_quality = 0.0
                prev_time = sample_time
                continue

            seg_duration = _segment_duration(seg)
            visibility = sample.get("visibility") or {}
            seg_key = id(seg)
            tile_qualities = tile_quality_cache.get(seg_key)
            if tile_qualities is None:
                tile_qualities = self._segment_tile_quality_harmonic_map(seg, max_levels, bitrate_ranges)
                tile_quality_cache[seg_key] = tile_qualities
            base_quality = self._segment_quality_from_tile_map(tile_qualities, visibility)
            dt_s = 0.0 if prev_time is None else max(0.0, sample_time - prev_time)
            perceived_quality = _relax_toward(prev_perceived_quality, base_quality, dt_s, self._perceptual_tau_s)

            smooth_penalty = 0.0
            if prev_perceived_quality is not None:
                delta = max(0.0, prev_perceived_quality - perceived_quality)
                smooth_penalty = self.weights.temporal_penalty * delta

            buffer_s = sample.get("buffer_s")
            buffer_risk = self._buffer_risk(buffer_s, seg_duration)
            quality_factor = math.exp(-self._buffer_decay_strength * buffer_risk)
            post_stall = _post_stall_penalty(
                sample_time,
                stalls,
                self.weights.stall_per_second,
                self.weights.stall_fixed,
                self._stall_memory_tau_s,
            )

            score = _clamp01(perceived_quality * quality_factor * math.exp(-self._stall_decay_strength * post_stall) - smooth_penalty)
            results.append((sample_time, score))
            prev_perceived_quality = perceived_quality
            prev_time = sample_time

        collapsed = _collapse_same_time_points(results)
        print(
            "[graph-timing] qoe_stall_aware "
            f"samples={len(samples)} output={len(collapsed)} sample_ms={sample_ms} "
            f"score_ms={int((time.perf_counter() - score_started) * 1000)} "
            f"elapsed_ms={int((time.perf_counter() - started) * 1000)}"
        )
        return collapsed
