"""Aggregate launch validation without creating runs or noisy per-run output."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any
from core.video import Video

from .runs import _build_playback_config


TIME_COLUMNS = {"time", "time_s", "timestamp"}


def _has_timestamp(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return bool({str(name).strip().lower() for name in (csv.reader(handle).__next__() or [])} & TIME_COLUMNS)
    except (OSError, StopIteration):
        return False


def _frame_range(path: Path) -> tuple[int | None, int | None, int]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            field = next((name for name in (reader.fieldnames or []) if name.strip().lower() in {"frame", "framenumber"}), None)
            if not field: return None, None, 0
            values = [int(float(row[field])) for row in reader if row.get(field, "").strip()]
            return (min(values), max(values), len(values)) if values else (None, None, 0)
    except (OSError, ValueError): return None, None, 0


def preflight_runs(raw_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate a launch set and reduce repeated results to compact checks."""
    errors: dict[str, list[dict[str, Any]]] = defaultdict(list)
    warnings: dict[str, list[dict[str, Any]]] = defaultdict(list)
    configs = []
    videos: dict[tuple[str, str], Video] = {}
    for index, raw in enumerate(raw_runs):
        try:
            config = _build_playback_config(raw, f"preflight_{index}")
            configs.append(config)
            video_key = (str(config.mpd), str(config.content_dir))
            if video_key not in videos:
                videos[video_key] = Video(config.mpd, config.content_dir)
            video = videos[video_key]
            first, last, count = _frame_range(config.fov_trace)
            if count and (first is None or last is None or first > video.start_frame_idx or last < video.start_frame_idx + video.total_frames - 1):
                warnings["FoV trace does not cover the complete content frame range"].append({"dataset": config.dataset_id, "trace": config.fov_trace.name})
            if config.predictor in {"LinearExtrapolationPredictor", "LinearPoseExtrapolationPredictor", "ReachabilityRaycastPredictor"} and not _has_timestamp(config.fov_trace):
                warnings["FoV traces have no timestamps"].append({"dataset": config.dataset_id, "trace": config.fov_trace.name})
        except Exception as exc:  # validation must report all independent bad entries
            errors[str(exc)].append({"index": index, "dataset": raw.get("dataset_id"), "trace": Path(str(raw.get("fov_trace") or "")).name})

    checks = []
    for title, affected in errors.items():
        checks.append({"severity": "error", "title": title, "affected_count": len(affected), "samples": affected[:5]})
    for title, affected in warnings.items():
        checks.append({
            "severity": "warning", "title": title,
            "detail": "Time-based predictors will use the MPD + PLY content timeline when trace timestamps are absent.",
            "affected_count": len(affected), "samples": affected[:5],
        })
    dimensions = {
        "ABR": [str(config.abr) for config in configs],
        "FoV predictor": [str(config.predictor or "None") for config in configs],
        "FoV preset": [str(config.predictor_preset_name or "Default") for config in configs],
        "Network": [str(config.network_profile or config.network_trace or "") for config in configs],
        "Dataset": [str(config.dataset_id or "") for config in configs],
    }
    fixed = {name: values[0] for name, values in dimensions.items() if values and len(set(values)) == 1}
    varied = [{"label": name, "values": sorted(set(values)), "count": len(set(values))} for name, values in dimensions.items() if len(set(values)) > 1]
    status = "error" if errors else ("warning" if warnings else "ready")
    timelines = sorted({f"{video.frame_rate_hz:g} Hz · {video.segment_count} segments" for video in videos.values()})
    return {"status": status, "run_count": len(raw_runs), "valid_count": len(configs), "checks": checks, "fixed": fixed, "varied": varied, "timelines": timelines}
