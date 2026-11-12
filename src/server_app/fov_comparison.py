from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from core.tile_fov import (
    DEFAULT_GROUND_TRUTH_MODE,
    DEFAULT_PEAK_WEIGHT,
    DEFAULT_VISIBLE_THRESHOLD,
    reduce_tile_visibility_frames,
)
from core.event_storage import event_log_path, read_event_log_text
from utils.ply_reader import PlyReader


ARTIFACT_DIR = "artifacts"
DATA_NAME = "fov_prediction_comparison.json"
VIDEO_NAME = "fov_prediction_comparison.mp4"
RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"


class FovComparisonError(ValueError):
    """Raised when a run cannot produce a FoV comparison artifact."""


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _load_pose_trace(config: Dict[str, Any]) -> Any | None:
    trace_path = config.get("fov_trace")
    if not trace_path:
        return None
    path = Path(str(trace_path))
    if not path.exists():
        return None
    try:
        import pandas as pd

        from core.rotations import ensure_fov_quaternions
        from core.trace_columns import normalize_fov_dataframe
        from core.world_transform_registry import apply_world_transform_to_trace

        df = normalize_fov_dataframe(pd.read_csv(path), source=path)
        content_dir = config.get("content_dir")
        if content_dir:
            df = apply_world_transform_to_trace(path, df, content_dir)
        df = ensure_fov_quaternions(df, angle_convention=str(config.get("angle_convention") or "unity_zxy"))
        df = df.set_index("FrameNumber")
        df.index = df.index.astype(int)
        return df.sort_index()
    except Exception:
        return None


def _portable_report_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Rebind bundled run metadata to locally fetched Argus inputs.

    Completed artifacts retain their original absolute paths as provenance. A
    reviewer may, however, generate the visual report on another machine.
    Prefer the local dataset profile and locate the same trace by its stable
    path suffix when those historical paths no longer exist.
    """
    resolved = dict(config)
    try:
        from server_app.config import get_dataset_profile

        profile = get_dataset_profile(str(config.get("dataset_id") or ""))
    except Exception:
        profile = None
    if profile:
        for key in ("mpd", "content_dir"):
            if not Path(str(resolved.get(key) or "")).exists() and profile.get(key):
                resolved[key] = profile[key]
    trace = Path(str(resolved.get("fov_trace") or ""))
    if trace.exists():
        return resolved
    marker = "data/fov_traces/"
    raw_trace = str(config.get("fov_trace") or "").replace("\\", "/")
    if marker in raw_trace:
        candidate = Path(__file__).resolve().parents[2] / "data" / "fov_traces" / raw_trace.split(marker, 1)[1]
        if candidate.is_file():
            resolved["fov_trace"] = str(candidate)
            return resolved
    trace_root = Path(__file__).resolve().parents[2] / "data" / "fov_traces"
    matches = list(trace_root.rglob(trace.name)) if trace.name else []
    if len(matches) == 1:
        resolved["fov_trace"] = str(matches[0])
    return resolved


def _pose_sample(row: Any, frame_idx: int, phase: str, angle_convention: str) -> Dict[str, Any] | None:
    from core.rotations import forward_from_quaternion_xyzw, quaternion_from_row

    try:
        position = [float(row["HMDPX"]), float(row["HMDPY"]), float(row["HMDPZ"])]
    except Exception:
        return None
    if not all(math.isfinite(value) for value in position):
        return None
    out: Dict[str, Any] = {
        "frame": int(frame_idx),
        "phase": phase,
        "position": position,
    }
    try:
        quat = quaternion_from_row(row, angle_convention=angle_convention)
        forward = forward_from_quaternion_xyzw(quat)
        out["forward"] = [float(value) for value in forward]
    except Exception:
        pass
    return out


def _sample_existing_pose_frames(
    pose_df: Any,
    frame_numbers: Iterable[int],
    phase: str,
    angle_convention: str,
    max_samples: int,
) -> List[Dict[str, Any]]:
    frames = sorted({int(frame) for frame in frame_numbers if int(frame) in pose_df.index})
    if not frames:
        return []
    step = max(1, math.ceil(len(frames) / max(1, int(max_samples))))
    out: List[Dict[str, Any]] = []
    for frame_idx in frames[::step]:
        row = pose_df.loc[frame_idx]
        if hasattr(row, "iloc") and hasattr(row, "columns"):
            row = row.iloc[-1]
        sample = _pose_sample(row, frame_idx, phase, angle_convention)
        if sample is not None:
            out.append(sample)
    return out


def _user_trace_for_frame(
    pose_df: Any | None,
    frame_numbers: Iterable[int],
    prediction_debug: Dict[str, Any] | None,
    config: Dict[str, Any],
) -> Dict[str, Any] | None:
    if pose_df is None:
        return None
    actual_frames = sorted({int(frame) for frame in frame_numbers})
    observed_frame = None
    if isinstance(prediction_debug, dict) and prediction_debug.get("observed_frame_idx") is not None:
        observed_frame = _safe_int(prediction_debug.get("observed_frame_idx"), -1)
    if observed_frame is None or observed_frame < 0:
        # The target is the segment after the buffered geometry segment.  A
        # baseline has no predictor-debug record, so recover the same
        # decision instant from the target's frame span: one target-segment
        # duration before its start, then one preceding-frame boundary.
        # For example, frames 1351–1500 target an observation at frame 1200.
        target_start = min(actual_frames) if actual_frames else None
        target_end = max(actual_frames) if actual_frames else None
        target_frame_count = (target_end - target_start + 1) if target_start is not None and target_end is not None else 0
        observed_frame = target_start - target_frame_count - 1 if target_start is not None and target_frame_count > 0 else None
    if observed_frame is None:
        return None

    angle_convention = str(config.get("angle_convention") or "unity_zxy")
    observed_start = max(int(pose_df.index.min()), int(observed_frame) - 90)
    observed_samples = _sample_existing_pose_frames(
        pose_df=pose_df,
        frame_numbers=range(observed_start, int(observed_frame) + 1),
        phase="observed",
        angle_convention=angle_convention,
        max_samples=60,
    )
    future_samples = _sample_existing_pose_frames(
        pose_df=pose_df,
        frame_numbers=actual_frames,
        phase="actual",
        angle_convention=angle_convention,
        max_samples=90,
    )
    current = None
    if int(observed_frame) in pose_df.index:
        row = pose_df.loc[int(observed_frame)]
        if hasattr(row, "iloc") and hasattr(row, "columns"):
            row = row.iloc[-1]
        current = _pose_sample(row, int(observed_frame), "current", angle_convention)
    return {
        "observed_frame_idx": int(observed_frame),
        "target_start_frame": int(min(actual_frames)) if actual_frames else None,
        "target_end_frame": int(max(actual_frames)) if actual_frames else None,
        "observed": observed_samples,
        "actual": future_samples,
        "current": current,
    }


def _visibility_by_segment(entries: Iterable[Dict[str, Any]] | None) -> Dict[int, Dict[str, float]]:
    out: Dict[int, Dict[str, float]] = {}
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        segment_idx = _safe_int(entry.get("segment_idx"), -1)
        if segment_idx < 0:
            continue
        raw = entry.get("tile_visibility")
        if not isinstance(raw, dict):
            continue
        out[segment_idx] = {str(tile_id): _safe_float(value) for tile_id, value in raw.items()}
    return out


def _ground_truth_frames_by_segment(entries: Iterable[Dict[str, Any]] | None) -> tuple[Dict[int, List[Dict[str, Any]]], Dict[int, Dict[str, float]], Dict[int, str]]:
    frames_by_segment: Dict[int, List[Dict[str, Any]]] = {}
    target_by_segment: Dict[int, Dict[str, float]] = {}
    mode_by_segment: Dict[int, str] = {}
    for entry in entries or []:
        if not isinstance(entry, dict):
            continue
        segment_idx = _safe_int(entry.get("segment_idx"), -1)
        if segment_idx < 0:
            continue
        frames: List[Dict[str, Any]] = []
        for raw_frame in entry.get("frames") or []:
            if not isinstance(raw_frame, dict):
                continue
            raw_visibility = raw_frame.get("tile_visibility")
            if not isinstance(raw_visibility, dict):
                continue
            frames.append(
                {
                    "frame": _safe_int(raw_frame.get("frame"), len(frames)),
                    "tile_visibility": {str(tile_id): _safe_float(value) for tile_id, value in raw_visibility.items()},
                }
            )
        if frames:
            frames_by_segment[segment_idx] = frames
        raw_target = entry.get("target_visibility")
        if isinstance(raw_target, dict):
            target_by_segment[segment_idx] = {str(tile_id): _safe_float(value) for tile_id, value in raw_target.items()}
        mode_by_segment[segment_idx] = str(entry.get("target_mode") or entry.get("mode") or DEFAULT_GROUND_TRUTH_MODE)
    return frames_by_segment, target_by_segment, mode_by_segment


def _bounds_for_tile(tile: Dict[str, Any]) -> tuple[float, float, float, float] | None:
    bounds = tile.get("bounds") if isinstance(tile, dict) else None
    if not isinstance(bounds, dict):
        return None
    min_corner = bounds.get("min_corner")
    max_corner = bounds.get("max_corner")
    if not isinstance(min_corner, (list, tuple)) or not isinstance(max_corner, (list, tuple)):
        return None
    if len(min_corner) < 3 or len(max_corner) < 3:
        return None
    return (
        _safe_float(min_corner[0]),
        _safe_float(min_corner[2]),
        _safe_float(max_corner[0]),
        _safe_float(max_corner[2]),
    )


def _first_ply_for_config(config: Dict[str, Any]) -> Path | None:
    reference = str(config.get("reference_ply") or "").strip()
    candidates: List[Path] = []
    if reference:
        ref_path = Path(reference)
        if ref_path.is_file() and ref_path.suffix.lower() == ".ply":
            return ref_path
        if ref_path.is_dir():
            candidates.extend(sorted(ref_path.glob("*.ply")))
    content_dir = str(config.get("content_dir") or "").strip()
    if content_dir:
        content_path = Path(content_dir)
        if content_path.is_file() and content_path.suffix.lower() == ".ply":
            return content_path
        if content_path.is_dir():
            candidates.extend(sorted(content_path.glob("*.ply")))
            if not candidates:
                candidates.extend(sorted(content_path.rglob("*.ply")))
    return candidates[0] if candidates else None


def _middle_height_point_slice(config: Dict[str, Any], max_points: int = 3500) -> Dict[str, Any] | None:
    ply_path = _first_ply_for_config(config)
    if ply_path is None or not ply_path.exists():
        return None
    try:
        points_df = PlyReader().read(ply_path)
        points = points_df.to_numpy()
    except Exception:
        return None
    if points.ndim != 2 or points.shape[1] < 3 or points.shape[0] <= 0:
        return None

    rows = []
    for row in points[:, :3]:
        x, y, z = _safe_float(row[0], float("nan")), _safe_float(row[1], float("nan")), _safe_float(row[2], float("nan"))
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
            rows.append((x, y, z))
    if not rows:
        return None

    ys = sorted(row[1] for row in rows)
    min_y = ys[0]
    max_y = ys[-1]
    mid_y = (min_y + max_y) * 0.5
    extent_y = max(max_y - min_y, 1e-9)
    thickness = max(extent_y * 0.04, 0.04)
    selected = [row for row in rows if abs(row[1] - mid_y) <= thickness * 0.5]
    if not selected:
        ranked = sorted(rows, key=lambda row: abs(row[1] - mid_y))
        selected = ranked[: min(len(ranked), max(1, int(max_points)))]
        if selected:
            thickness = max(abs(selected[-1][1] - mid_y) * 2.0, thickness)
    if len(selected) > max_points:
        step = max(1, math.ceil(len(selected) / max(1, int(max_points))))
        selected = selected[::step][:max_points]

    return {
        "source": str(ply_path),
        "height_axis": "y",
        "height": float(mid_y),
        "thickness": float(thickness),
        "point_count": int(len(selected)),
        "source_point_count": int(len(rows)),
        "points": [[float(x), float(z)] for x, _y, z in selected],
    }


def _segment_time_s(segment: Dict[str, Any], decision_by_segment: Dict[int, Dict[str, Any]]) -> float:
    segment_idx = _safe_int(segment.get("segment_idx"), -1)
    decision = decision_by_segment.get(segment_idx) or {}
    if "time_s" in decision:
        return _safe_float(decision.get("time_s"))
    return _safe_float(segment.get("time_s"))


def _soft_iou(actual: Dict[str, float], predicted: Dict[str, float], tile_ids: Iterable[str]) -> float:
    numerator = 0.0
    denominator = 0.0
    for tile_id in tile_ids:
        a = max(0.0, min(1.0, actual.get(tile_id, 0.0)))
        p = max(0.0, min(1.0, predicted.get(tile_id, 0.0)))
        numerator += min(a, p)
        denominator += max(a, p)
    return numerator / denominator if denominator else 1.0


def _mae(actual: Dict[str, float], predicted: Dict[str, float], tile_ids: Iterable[str]) -> float:
    values = [abs(max(0.0, min(1.0, actual.get(t, 0.0))) - max(0.0, min(1.0, predicted.get(t, 0.0)))) for t in tile_ids]
    return sum(values) / len(values) if values else 0.0


def _rank_key(tile_id: str) -> tuple[int, str]:
    try:
        return (int(tile_id), tile_id)
    except ValueError:
        return (10**9, tile_id)


def build_fov_comparison_dataset(run_id: str) -> Dict[str, Any]:
    run_dir = RESULTS_ROOT / run_id
    events_file = event_log_path(run_dir)
    config_file = run_dir / "config.json"
    if events_file is None:
        raise FovComparisonError("events.json is missing for this run.")
    if not config_file.exists():
        raise FovComparisonError("config.json is missing for this run.")
    try:
        events = json.loads(read_event_log_text(events_file))
    except json.JSONDecodeError as exc:
        raise FovComparisonError(f"events.json is invalid: {exc}") from exc
    if not isinstance(events, dict):
        raise FovComparisonError("events.json did not contain an object.")
    config = _portable_report_config(_read_json(config_file))
    pose_df = _load_pose_trace(config)
    point_cloud_slice = _middle_height_point_slice(config)

    segments = [seg for seg in events.get("enqueued_segments") or [] if isinstance(seg, dict)]
    if not segments:
        raise FovComparisonError("No enqueued segment metadata was found.")
    predicted_by_segment = _visibility_by_segment(events.get("fov_predictions"))
    debug_by_segment = {
        _safe_int(entry.get("segment_idx"), -1): entry
        for entry in events.get("fov_prediction_debug") or []
        if isinstance(entry, dict)
    }
    actual_mean_by_segment = _visibility_by_segment(events.get("fov_ground_truth"))
    actual_frames_by_segment, actual_target_by_segment, actual_mode_by_segment = _ground_truth_frames_by_segment(events.get("fov_ground_truth_frames"))
    actual_by_segment = dict(actual_mean_by_segment)
    params = events.get("ground_truth_params") if isinstance(events.get("ground_truth_params"), dict) else {}
    threshold = _safe_float(params.get("visible_threshold"), DEFAULT_VISIBLE_THRESHOLD)
    peak_weight = _safe_float(params.get("peak_weight"), DEFAULT_PEAK_WEIGHT)
    for seg_idx, frames_for_segment in actual_frames_by_segment.items():
        actual_by_segment[seg_idx] = actual_target_by_segment.get(seg_idx) or reduce_tile_visibility_frames(
            frames_for_segment,
            mode=actual_mode_by_segment.get(seg_idx, DEFAULT_GROUND_TRUTH_MODE),
            visible_threshold=threshold,
            peak_weight=peak_weight,
        )
    if not predicted_by_segment:
        raise FovComparisonError("No FoV predictor tile scores were logged.")
    if not actual_by_segment:
        raise FovComparisonError("No FoV ground-truth tile scores were logged.")

    decisions = {
        _safe_int(entry.get("segment_idx"), -1): entry
        for entry in events.get("abr_decisions") or []
        if isinstance(entry, dict)
    }

    frames: List[Dict[str, Any]] = []
    for idx, segment in enumerate(segments):
        segment_idx = _safe_int(segment.get("segment_idx"), idx)
        predicted = predicted_by_segment.get(segment_idx)
        actual = actual_by_segment.get(segment_idx)
        if predicted is None or actual is None:
            continue
        raw_tiles = segment.get("tiles")
        if not isinstance(raw_tiles, dict) or not raw_tiles:
            continue
        tiles: Dict[str, Any] = {}
        for tile_id, tile in raw_tiles.items():
            bounds = _bounds_for_tile(tile)
            if bounds is None:
                continue
            tiles[str(tile_id)] = {
                "bounds_xz": bounds,
                "raw_bounds": (tile.get("bounds") or {}),
            }
        if not tiles:
            continue
        tile_ids = sorted(set(tiles.keys()) | set(predicted.keys()) | set(actual.keys()), key=_rank_key)
        missing_bounds = [tile_id for tile_id in tile_ids if tile_id not in tiles]
        if missing_bounds:
            raise FovComparisonError(
                f"Segment {segment_idx} has FoV scores for tiles without bounds: {', '.join(missing_bounds[:8])}"
            )
        prediction_debug = debug_by_segment.get(segment_idx)
        actual_frames = actual_frames_by_segment.get(segment_idx, [])
        user_trace = _user_trace_for_frame(
            pose_df=pose_df,
            frame_numbers=[_safe_int(frame.get("frame"), -1) for frame in actual_frames if isinstance(frame, dict)],
            prediction_debug=prediction_debug,
            config=config,
        )
        frames.append(
            {
                "segment_idx": segment_idx,
                "time_s": _segment_time_s(segment, decisions),
                "duration_s": _safe_float(segment.get("duration_s")),
                "tiles": tiles,
                "tile_ids": tile_ids,
                "actual": actual,
                "actual_mean": actual_mean_by_segment.get(segment_idx, actual),
                "actual_frames": actual_frames,
                "ground_truth_mode": actual_mode_by_segment.get(segment_idx, "mean_visibility"),
                "predicted": predicted,
                "prediction_debug": prediction_debug,
                "user_trace": user_trace,
                "metrics": {
                    "soft_iou": _soft_iou(actual, predicted, tile_ids),
                    "mae": _mae(actual, predicted, tile_ids),
                },
            }
        )

    if not frames:
        raise FovComparisonError("No segment had both prediction and ground-truth FoV data with tile bounds.")

    return {
        "schema": "prism.fov_comparison.v1",
        "generated_at": datetime.now().isoformat(),
        "run_id": run_id,
        "source": {
            "events_path": str(events_file),
            "config_path": str(config_file),
        },
        "display": {
            "projection": "x/z top-down",
            "color_scale": [0.0, 1.0],
            "score_display": "clamped to [0, 1]; raw values preserved in actual/predicted",
        },
        "point_cloud_slice": point_cloud_slice,
        "frames": frames,
    }


def build_fov_comparison_segment(run_id: str, segment_idx: int) -> Dict[str, Any]:
    """Return one debugger frame without rendering a comparison video."""
    dataset = build_fov_comparison_dataset(run_id)
    requested = int(segment_idx)
    frame = next((item for item in dataset["frames"] if int(item.get("segment_idx", -1)) == requested), None)
    if frame is None:
        raise FovComparisonError(f"Segment {requested} has no prediction/ground-truth comparison data.")
    return {
        "schema": "prism.fov_comparison.segment.v1",
        "run_id": run_id,
        "segment_idx": requested,
        "display": dataset.get("display") or {},
        "frame": frame,
    }


def _color_for_score(value: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, value))
    return (255, int(245 - 215 * t), int(238 - 225 * t))


def _set_pixel(buf: bytearray, width: int, height: int, x: int, y: int, color: tuple[int, int, int]) -> None:
    if x < 0 or y < 0 or x >= width or y >= height:
        return
    idx = (y * width + x) * 3
    buf[idx : idx + 3] = bytes(color)


def _fill_rect(buf: bytearray, width: int, height: int, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    for y in range(y0, y1):
        row = y * width * 3
        pixel = bytes(color)
        for x in range(x0, x1):
            idx = row + x * 3
            buf[idx : idx + 3] = pixel


def _stroke_rect(buf: bytearray, width: int, height: int, box: tuple[int, int, int, int], color: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    for x in range(x0, x1):
        _set_pixel(buf, width, height, x, y0, color)
        _set_pixel(buf, width, height, x, y1 - 1, color)
    for y in range(y0, y1):
        _set_pixel(buf, width, height, x0, y, color)
        _set_pixel(buf, width, height, x1 - 1, y, color)


FONT_5X7: Dict[str, tuple[str, ...]] = {
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    "-": ("00000", "00000", "00000", "11110", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    "/": ("00001", "00010", "00100", "01000", "10000", "00000", "00000"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01110", "10001", "10000", "10111", "10001", "10001", "01110"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("01110", "00100", "00100", "00100", "00100", "00100", "01110"),
    "J": ("00111", "00010", "00010", "00010", "10010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "01010", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "01010", "00100", "00100", "00100", "01010", "10001"),
    "Y": ("10001", "01010", "00100", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
}


def _draw_text(
    buf: bytearray,
    width: int,
    height: int,
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int] = (24, 24, 24),
    scale: int = 2,
) -> None:
    cursor = x
    for char in text.upper():
        glyph = FONT_5X7.get(char, FONT_5X7[" "])
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit != "1":
                    continue
                _fill_rect(
                    buf,
                    width,
                    height,
                    (cursor + gx * scale, y + gy * scale, cursor + (gx + 1) * scale, y + (gy + 1) * scale),
                    color,
                )
        cursor += 6 * scale


def _text_size(text: str, scale: int = 1) -> tuple[int, int]:
    if not text:
        return (0, 0)
    return (max(0, len(text) * 6 - 1) * scale, 7 * scale)


def _panel_bounds(frame: Dict[str, Any]) -> tuple[float, float, float, float]:
    xs: List[float] = []
    zs: List[float] = []
    for tile_id in frame["tile_ids"]:
        x0, z0, x1, z1 = frame["tiles"][tile_id]["bounds_xz"]
        xs.extend([x0, x1])
        zs.extend([z0, z1])
    if not xs or not zs:
        return (0.0, 0.0, 1.0, 1.0)
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)
    if abs(max_x - min_x) < 1e-9:
        max_x += 1.0
    if abs(max_z - min_z) < 1e-9:
        max_z += 1.0
    return (min_x, min_z, max_x, max_z)


def _project(
    bounds: tuple[float, float, float, float],
    world: tuple[float, float, float, float],
    panel: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    min_x, min_z, max_x, max_z = world
    px0, py0, px1, py1 = panel
    x0, z0, x1, z1 = bounds
    sx0 = px0 + int((min(x0, x1) - min_x) / (max_x - min_x) * (px1 - px0))
    sx1 = px0 + int((max(x0, x1) - min_x) / (max_x - min_x) * (px1 - px0))
    sy0 = py1 - int((max(z0, z1) - min_z) / (max_z - min_z) * (py1 - py0))
    sy1 = py1 - int((min(z0, z1) - min_z) / (max_z - min_z) * (py1 - py0))
    return (sx0, sy0, max(sx1, sx0 + 1), max(sy1, sy0 + 1))


def _project_point(
    point: Iterable[float],
    world: tuple[float, float, float, float],
    panel: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    values = list(point)
    if len(values) < 2:
        return None
    x, z = _safe_float(values[0], float("nan")), _safe_float(values[1], float("nan"))
    min_x, min_z, max_x, max_z = world
    px0, py0, px1, py1 = panel
    if not math.isfinite(x) or not math.isfinite(z) or max_x <= min_x or max_z <= min_z:
        return None
    sx = px0 + int((x - min_x) / (max_x - min_x) * (px1 - px0))
    sy = py1 - int((z - min_z) / (max_z - min_z) * (py1 - py0))
    return (sx, sy)


def _draw_point_marker(buf: bytearray, width: int, height: int, x: int, y: int) -> None:
    """Draw a legible point-cloud marker over either light or dark tiles."""
    _fill_rect(buf, width, height, (x - 2, y - 2, x + 3, y + 3), (255, 255, 255))
    _fill_rect(buf, width, height, (x - 1, y - 1, x + 2, y + 2), (17, 24, 39))


def _draw_panel(
    buf: bytearray,
    width: int,
    height: int,
    frame: Dict[str, Any],
    scores: Dict[str, float],
    panel: tuple[int, int, int, int],
    point_cloud_slice: Dict[str, Any] | None = None,
    show_scores: bool = True,
) -> None:
    world = _panel_bounds(frame)
    _fill_rect(buf, width, height, panel, (252, 252, 252))
    _stroke_rect(buf, width, height, panel, (28, 28, 28))
    for tile_id in frame["tile_ids"]:
        box = _project(frame["tiles"][tile_id]["bounds_xz"], world, panel)
        score = max(0.0, min(1.0, scores.get(tile_id, 0.0)))
        color = _color_for_score(score)
        _fill_rect(buf, width, height, box, color)
        _stroke_rect(buf, width, height, box, (232, 222, 216))
        if show_scores:
            label = f"{score:.2f}"
            label_w, label_h = _text_size(label, scale=1)
            box_w = box[2] - box[0]
            box_h = box[3] - box[1]
            if box_w >= label_w + 4 and box_h >= label_h + 4:
                label_x = box[0] + (box_w - label_w) // 2
                label_y = box[1] + (box_h - label_h) // 2
                label_color = (255, 255, 255) if score >= 0.55 else (48, 38, 38)
                _draw_text(buf, width, height, label_x, label_y, label, color=label_color, scale=1)
    # Draw the retained horizontal point-cloud slice above the score tiles.
    # The points provide geometric context without changing the common score
    # color scale or hiding the tile boundaries.
    # Retain one marker per small screen-space cell.  This preserves the
    # shape of dense geometry without turning it into an opaque mesh.
    occupied: set[tuple[int, int]] = set()
    for point in (point_cloud_slice or {}).get("points") or []:
        projected = _project_point(point, world, panel)
        if projected is not None and (cell := (projected[0] // 9, projected[1] // 9)) not in occupied:
            occupied.add(cell)
            _draw_point_marker(buf, width, height, projected[0], projected[1])


def _write_ppm_frame(
    path: Path,
    frame: Dict[str, Any],
    size: tuple[int, int],
    point_cloud_slice: Dict[str, Any] | None = None,
) -> None:
    width, height = size
    buf = bytearray([255, 255, 255]) * (width * height)
    margin = 36
    top = 72
    bottom = 50
    gap = 44
    panel_w = (width - margin * 2 - gap) // 2
    left = (margin, top, margin + panel_w, height - bottom)
    right = (margin + panel_w + gap, top, width - margin, height - bottom)
    segment_idx = _safe_int(frame.get("segment_idx"))
    time_s = _safe_float(frame.get("time_s"))
    duration_s = _safe_float(frame.get("duration_s"))
    metrics = frame.get("metrics") or {}
    _draw_text(
        buf,
        width,
        height,
        margin,
        22,
        f"SEGMENT {segment_idx} {time_s:.2f}-{time_s + duration_s:.2f}S",
        scale=2,
    )
    _draw_text(buf, width, height, left[0], 50, "GROUND TRUTH", scale=2)
    _draw_text(buf, width, height, right[0], 50, "PREDICTION", scale=2)
    _draw_panel(buf, width, height, frame, frame["actual"], left, point_cloud_slice)
    _draw_panel(buf, width, height, frame, frame["predicted"], right, point_cloud_slice)
    _draw_text(
        buf,
        width,
        height,
        margin,
        height - 30,
        f"SOFT-IOU {_safe_float(metrics.get('soft_iou')):.3f}  MAE {_safe_float(metrics.get('mae')):.3f}",
        scale=2,
    )
    with path.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        handle.write(buf)


def _encode_mp4(frame_dir: Path, output_path: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FovComparisonError("ffmpeg is required to encode fov_prediction_comparison.mp4 but was not found on PATH.")
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "frame_%05d.ppm"),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    if completed.returncode != 0:
        raise FovComparisonError(f"ffmpeg failed to encode video: {completed.stderr[-1200:]}")


def fov_comparison_paths(run_id: str) -> Dict[str, str | None]:
    artifact_dir = RESULTS_ROOT / run_id / ARTIFACT_DIR
    data_file = artifact_dir / DATA_NAME
    video_file = artifact_dir / VIDEO_NAME
    return {
        "data_path": f"/results/{run_id}/{ARTIFACT_DIR}/{DATA_NAME}" if data_file.exists() else None,
        "video_path": f"/results/{run_id}/{ARTIFACT_DIR}/{VIDEO_NAME}" if video_file.exists() else None,
    }


def store_fov_comparison_result(run_id: str, result: Dict[str, Any]) -> None:
    try:
        from .runs import RUNS, RUNS_LOCK

        with RUNS_LOCK:
            job = RUNS.get(run_id)
            if job is not None:
                job.fov_comparison = result
    except ImportError:
        pass
    status_file = RESULTS_ROOT / run_id / "status.json"
    status = _read_json(status_file) if status_file.exists() else {"run_id": run_id}
    if not isinstance(status, dict):
        status = {"run_id": run_id}
    status["fov_comparison"] = result
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")


def generate_fov_comparison_video(run_id: str, options: Dict[str, Any] | None = None) -> Dict[str, Any]:
    options = options or {}
    fps = max(1, min(30, _safe_int(options.get("fps"), 2)))
    width = max(640, min(3840, _safe_int(options.get("width"), 1280)))
    height = max(360, min(2160, _safe_int(options.get("height"), 720)))
    artifact_dir = RESULTS_ROOT / run_id / ARTIFACT_DIR
    data_file = artifact_dir / DATA_NAME
    video_file = artifact_dir / VIDEO_NAME
    try:
        dataset = build_fov_comparison_dataset(run_id)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        data_file.write_text(json.dumps(dataset, indent=2, default=str), encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix="prism_fov_cmp_") as tmp:
            frame_dir = Path(tmp)
            for idx, frame in enumerate(dataset["frames"]):
                _write_ppm_frame(
                    frame_dir / f"frame_{idx:05d}.ppm", frame, (width, height), dataset.get("point_cloud_slice")
                )
            _encode_mp4(frame_dir, video_file, fps=fps)
        result = {
            "ok": True,
            "run_id": run_id,
            "status": "done",
            "generated_at": dataset["generated_at"],
            "frame_count": len(dataset["frames"]),
            "video_path": f"/results/{run_id}/{ARTIFACT_DIR}/{VIDEO_NAME}",
            "data_path": f"/results/{run_id}/{ARTIFACT_DIR}/{DATA_NAME}",
        }
    except FovComparisonError as exc:
        result = {
            "ok": False,
            "run_id": run_id,
            "status": "failed",
            "generated_at": datetime.now().isoformat(),
            "error": str(exc),
            **fov_comparison_paths(run_id),
        }
    store_fov_comparison_result(run_id, result)
    return result
