from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from uuid import uuid4

from core.mpd_parser import V3CMpd
from core.trace_columns import (
    save_trace_column_settings,
    trace_column_config_payload,
)
from core.world_transform_registry import (
    WORLD_TRANSFORMS_FILE,
    detect_header_transform,
    get_world_transform,
    list_world_transforms,
    normalize_content_dir,
    set_world_transform,
)
from integrations.navis_client import probe_navis

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
WEB_DIST_ROOT = WEB_ROOT / "dist"
RESULTS_ROOT = ROOT / ".." / "results"
PROJECT_ROOT = ROOT.parent
SETTINGS_ROOT = PROJECT_ROOT / ".prism_settings"
NAVIS_CONFIG_FILE = SETTINGS_ROOT / "navis.json"
HEADSET_CONFIG_FILE = SETTINGS_ROOT / "headsets.json"
TRACE_CONFIG_FILE = SETTINGS_ROOT / "trace_roots.json"
PRISM_CONFIG_FILE = SETTINGS_ROOT / "prism.json"
FORM_SESSIONS_FILE = SETTINGS_ROOT / "run_form_sessions.json"
DATASET_PROFILES_FILE = SETTINGS_ROOT / "datasets.json"

DEFAULT_HEADSET_PRESETS: list[dict[str, Any]] = [
    {"id": "prism_default", "label": "Prism Default", "fov_h_deg": 97.0, "fov_v_deg": 93.0},
    {"id": "meta_quest_2", "label": "Meta Quest 2", "fov_h_deg": 96.0, "fov_v_deg": 90.0},
    {"id": "meta_quest_3", "label": "Meta Quest 3", "fov_h_deg": 110.0, "fov_v_deg": 96.0},
    {"id": "apple_vision_pro", "label": "Apple Vision Pro", "fov_h_deg": 100.0, "fov_v_deg": 96.0},
    {"id": "pico_4", "label": "PICO 4", "fov_h_deg": 105.0, "fov_v_deg": 105.0},
]
ANGLE_CONVENTIONS: list[dict[str, str]] = [
    {"id": "unity_zxy", "label": "Y-up / Z-forward, Unity order ZXY"},
    {"id": "unity_zyx", "label": "Y-up / Z-forward, order ZYX"},
    {"id": "unity_xyz", "label": "Y-up / Z-forward, order XYZ"},
    {"id": "unity_xzy", "label": "Y-up / Z-forward, order XZY"},
    {"id": "unity_yxz", "label": "Y-up / Z-forward, order YXZ"},
    {"id": "unity_yzx", "label": "Y-up / Z-forward, order YZX"},
]
DEFAULT_HEADSET_CONFIG: dict[str, Any] = {
    "default_preset": "prism_default",
    "angle_convention": "unity_zxy",
    "presets": DEFAULT_HEADSET_PRESETS,
}
DEFAULT_TRACE_CONFIG: dict[str, str] = {
    "network_root": "data/network_traces",
    "fov_root": "data/fov_traces",
}
DEFAULT_PRISM_CONFIG: dict[str, Any] = {
    "record_fov_stage_latency": False,
    "batch_queue_coalesce_ms": 250,
    "batch_queue_max_coalesce_ms": 2500,
    "batch_ply_io_workers": 1,
    "batch_segment_workers": 0,
    "batch_segment_workers_auto_cap": 4,
    "batch_group_max_runs": 0,
    "batch_prefetch_segments": 1,
    "training_max_parallel_runs": 1,
    "training_cpu_core_limit": max(1, os.cpu_count() or 1),
    "training_require_single_worker_for_parallel": True,
}


def generate_run_id() -> str:
    now = datetime.now()
    # Batch plans may create hundreds of runs in one second.  Four hex digits
    # make a collision likely enough to silently coalesce two queued jobs.
    suffix = uuid4().hex[:8]
    return f"run_{now.strftime('%y%m%d_%H%M%S')}_{suffix}"


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def safe_path(path_str: str | Path) -> Path:
    path_obj = Path(path_str).expanduser()
    path = path_obj if path_obj.is_absolute() else (PROJECT_ROOT / path_obj)
    return path.resolve()


def path_for_client(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def parse_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on", "y")
    return False


def require_nonempty_path(payload: Dict[str, Any], key: str, label: str) -> str:
    raw = payload.get(key)
    value = str(raw or "").strip()
    if not value:
        raise ValueError(f"{label} is required.")
    return value


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def safe_id(value: Any, fallback: str) -> str:
    raw = str(value or "").strip().lower()
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw)
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    return cleaned or fallback


def sanitize_trace_settings(raw_payload: Any) -> Dict[str, str]:
    if raw_payload is None:
        raw_payload = {}
    if not isinstance(raw_payload, dict):
        raise ValueError("Trace root settings payload must be an object/dict.")
    network_root = str(raw_payload.get("network_root") or DEFAULT_TRACE_CONFIG["network_root"]).strip()
    fov_root = str(raw_payload.get("fov_root") or DEFAULT_TRACE_CONFIG["fov_root"]).strip()
    if not network_root:
        network_root = DEFAULT_TRACE_CONFIG["network_root"]
    if not fov_root:
        fov_root = DEFAULT_TRACE_CONFIG["fov_root"]
    return {
        "network_root": path_for_client(safe_path(network_root)),
        "fov_root": path_for_client(safe_path(fov_root)),
    }


def trace_settings() -> Dict[str, str]:
    raw = read_json(TRACE_CONFIG_FILE)
    if not raw:
        return dict(DEFAULT_TRACE_CONFIG)
    try:
        return sanitize_trace_settings(raw)
    except ValueError:
        return dict(DEFAULT_TRACE_CONFIG)


def save_trace_settings(raw_payload: Any) -> Dict[str, str]:
    settings = sanitize_trace_settings(raw_payload)
    TRACE_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with TRACE_CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
    return settings


def trace_config_payload() -> Dict[str, Any]:
    return {
        "settings": trace_settings(),
        "settings_path": str(TRACE_CONFIG_FILE.relative_to(PROJECT_ROOT)),
    }


def sanitize_prism_settings(raw_payload: Any) -> Dict[str, Any]:
    if raw_payload is None:
        raw_payload = {}
    if not isinstance(raw_payload, dict):
        raise ValueError("Prism settings payload must be an object/dict.")
    def bounded_int(key: str, default: int, *, minimum: int, maximum: int) -> int:
        try:
            value = int(raw_payload.get(key, default) or default)
        except (TypeError, ValueError):
            value = default
        return max(minimum, min(maximum, value))

    coalesce_ms = bounded_int("batch_queue_coalesce_ms", DEFAULT_PRISM_CONFIG["batch_queue_coalesce_ms"], minimum=0, maximum=10_000)
    max_coalesce_ms = bounded_int(
        "batch_queue_max_coalesce_ms",
        DEFAULT_PRISM_CONFIG["batch_queue_max_coalesce_ms"],
        minimum=coalesce_ms,
        maximum=60_000,
    )
    ply_io_workers = bounded_int("batch_ply_io_workers", DEFAULT_PRISM_CONFIG["batch_ply_io_workers"], minimum=1, maximum=16)
    # Zero means automatic per-segment sizing; positive values remain fixed.
    segment_workers = bounded_int("batch_segment_workers", DEFAULT_PRISM_CONFIG["batch_segment_workers"], minimum=0, maximum=16)
    segment_workers_auto_cap = bounded_int(
        "batch_segment_workers_auto_cap",
        DEFAULT_PRISM_CONFIG["batch_segment_workers_auto_cap"],
        minimum=1,
        maximum=16,
    )
    group_max_runs = bounded_int("batch_group_max_runs", DEFAULT_PRISM_CONFIG["batch_group_max_runs"], minimum=0, maximum=10_000)
    prefetch_segments = bounded_int("batch_prefetch_segments", DEFAULT_PRISM_CONFIG["batch_prefetch_segments"], minimum=0, maximum=1)
    training_max_parallel_runs = bounded_int(
        "training_max_parallel_runs",
        DEFAULT_PRISM_CONFIG["training_max_parallel_runs"],
        minimum=1,
        maximum=256,
    )
    training_cpu_core_limit = bounded_int(
        "training_cpu_core_limit",
        DEFAULT_PRISM_CONFIG["training_cpu_core_limit"],
        minimum=1,
        maximum=max(1, (os.cpu_count() or 1) * 4),
    )
    return {
        "record_fov_stage_latency": parse_bool(
            raw_payload.get("record_fov_stage_latency", DEFAULT_PRISM_CONFIG["record_fov_stage_latency"])
        ),
        "batch_queue_coalesce_ms": min(coalesce_ms, 10_000),
        "batch_queue_max_coalesce_ms": min(max_coalesce_ms, 60_000),
        "batch_ply_io_workers": ply_io_workers,
        "batch_segment_workers": segment_workers,
        "batch_segment_workers_auto_cap": segment_workers_auto_cap,
        "batch_group_max_runs": group_max_runs,
        "batch_prefetch_segments": prefetch_segments,
        "training_max_parallel_runs": training_max_parallel_runs,
        "training_cpu_core_limit": training_cpu_core_limit,
        "training_require_single_worker_for_parallel": parse_bool(
            raw_payload.get(
                "training_require_single_worker_for_parallel",
                DEFAULT_PRISM_CONFIG["training_require_single_worker_for_parallel"],
            )
        ),
    }


def prism_settings() -> Dict[str, Any]:
    raw = read_json(PRISM_CONFIG_FILE)
    if not raw:
        return dict(DEFAULT_PRISM_CONFIG)
    try:
        return sanitize_prism_settings(raw)
    except ValueError:
        return dict(DEFAULT_PRISM_CONFIG)


def save_prism_settings(raw_payload: Any) -> Dict[str, Any]:
    settings = sanitize_prism_settings(raw_payload)
    PRISM_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with PRISM_CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
    return settings


def prism_config_payload() -> Dict[str, Any]:
    return {
        "settings": prism_settings(),
        "settings_path": str(PRISM_CONFIG_FILE.relative_to(PROJECT_ROOT)),
    }


def sanitize_world_transform_payload(raw_payload: Any) -> Dict[str, Any]:
    if not isinstance(raw_payload, dict):
        raise ValueError("World transform payload must be an object/dict.")
    content_dir = raw_payload.get("content_dir")
    if not isinstance(content_dir, str) or not content_dir.strip():
        raise ValueError("content_dir is required.")
    transform = raw_payload.get("transform")
    if not isinstance(transform, dict):
        raise ValueError("transform is required.")
    translation = transform.get("translation")
    if not isinstance(translation, list) or len(translation) < 3:
        raise ValueError("transform.translation must contain three numeric values.")
    return {
        "content_dir": normalize_content_dir(content_dir),
        "transform": {
            "scale": float(transform.get("scale", 1.0)),
            "translation": [float(translation[0]), float(translation[1]), float(translation[2])],
            "units": str(transform.get("units") or "meters"),
            "source": str(transform.get("source") or "manual"),
            "notes": str(transform.get("notes") or ""),
            "apply_to_fov_trace": parse_bool(transform.get("apply_to_fov_trace", True)),
        },
    }


def world_transform_payload(content_dir: str | None = None) -> Dict[str, Any]:
    if content_dir:
        normalized = normalize_content_dir(content_dir)
        return {
            "content_dir": normalized,
            "transform": get_world_transform(normalized),
            "detected_header_transform": detect_header_transform(normalized),
            "settings_path": str(WORLD_TRANSFORMS_FILE.relative_to(PROJECT_ROOT)),
        }
    return {
        "transforms": list_world_transforms(),
        "settings_path": str(WORLD_TRANSFORMS_FILE.relative_to(PROJECT_ROOT)),
    }


def save_world_transform(raw_payload: Any) -> Dict[str, Any]:
    payload = sanitize_world_transform_payload(raw_payload)
    transform = payload["transform"]
    saved = set_world_transform(
        content_dir=payload["content_dir"],
        scale=transform["scale"],
        translation=transform["translation"],
        units=transform["units"],
        source=transform["source"],
        notes=transform["notes"],
        apply_to_fov_trace=transform["apply_to_fov_trace"],
    )
    return {
        "content_dir": payload["content_dir"],
        "transform": saved,
        "settings_path": str(WORLD_TRANSFORMS_FILE.relative_to(PROJECT_ROOT)),
    }


def find_mpd_in_dataset_folder(folder: str | Path) -> Path | None:
    candidate = safe_path(folder)
    if candidate.is_file() and candidate.suffix.lower() == ".mpd":
        return candidate
    if candidate.is_dir():
        matches = sorted(p for p in candidate.iterdir() if p.is_file() and p.suffix.lower() == ".mpd")
        if matches:
            return matches[0].resolve()
    return None


def mpd_tile_extent_stats(mpd_path: str | Path) -> Dict[str, Any]:
    mpd = V3CMpd(safe_path(mpd_path))
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    count = 0
    for bounds_by_tile in mpd.bounds_by_segment.values():
        for bounds in bounds_by_tile.values():
            for idx, value in enumerate(bounds.min_corner):
                mins[idx] = min(mins[idx], float(value))
            for idx, value in enumerate(bounds.max_corner):
                maxs[idx] = max(maxs[idx], float(value))
            count += 1
    if count <= 0 or not all(math.isfinite(v) for v in mins + maxs):
        raise ValueError("No tile bounds were found in the MPD.")
    extents = [maxs[idx] - mins[idx] for idx in range(3)]
    return {
        "source": "mpd_tile_bounds",
        "tile_bounds_count": int(count),
        "axis": {
            "x": {"min": float(mins[0]), "max": float(maxs[0]), "extent": float(extents[0])},
            "y": {"min": float(mins[1]), "max": float(maxs[1]), "extent": float(extents[1])},
            "z": {"min": float(mins[2]), "max": float(maxs[2]), "extent": float(extents[2])},
        },
    }


def load_dataset_registry() -> Dict[str, Any]:
    raw = read_json(DATASET_PROFILES_FILE)
    entries = raw.get("entries")
    if not isinstance(entries, dict):
        entries = {}
    return {"version": 1, "entries": entries}


def save_dataset_registry(registry: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {"version": 1, "entries": dict(registry.get("entries") or {})}
    DATASET_PROFILES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with DATASET_PROFILES_FILE.open("w", encoding="utf-8") as handle:
        json.dump(normalized, handle, indent=2)
    return normalized


def get_dataset_profile(dataset_id: str | None) -> Dict[str, Any] | None:
    if not dataset_id:
        return None
    registry = load_dataset_registry()
    profile = registry["entries"].get(str(dataset_id))
    return dict(profile) if isinstance(profile, dict) else None


def list_dataset_profiles() -> list[Dict[str, Any]]:
    registry = load_dataset_registry()
    profiles = []
    for dataset_id, profile in sorted((registry.get("entries") or {}).items()):
        if not isinstance(profile, dict):
            continue
        item = dict(profile)
        item["id"] = str(item.get("id") or dataset_id)
        profiles.append(item)
    return profiles


def dataset_profiles_payload() -> Dict[str, Any]:
    return {
        "datasets": list_dataset_profiles(),
        "settings_path": str(DATASET_PROFILES_FILE.relative_to(PROJECT_ROOT)),
    }


def resolve_dataset_scale(raw: Dict[str, Any], mpd_path: Path) -> tuple[float, Dict[str, Any], Dict[str, Any] | None]:
    mode = str(raw.get("scale_mode") or "manual").strip().lower()
    calibration = dict(raw.get("calibration") or {})
    if mode in ("already_meters", "none", "identity"):
        return 1.0, {"mode": "already_meters"}, None
    if mode == "manual":
        scale = safe_optional_float(raw.get("scale"))
        if scale is None or scale <= 0:
            raise ValueError("Manual scale must be greater than zero.")
        return float(scale), {"mode": "manual"}, None

    if mode not in ("known_extent_manual", "known_extent_mpd"):
        raise ValueError(f"Unsupported dataset scale mode '{mode}'.")

    axis = str(calibration.get("axis") or raw.get("axis") or "y").strip().lower()
    if axis not in ("x", "y", "z"):
        raise ValueError("Calibration axis must be x, y, or z.")
    dimension = str(calibration.get("dimension") or raw.get("dimension") or "height").strip().lower()
    real_extent_m = safe_optional_float(calibration.get("real_extent_m", raw.get("real_extent_m")))
    if real_extent_m is None or real_extent_m <= 0:
        raise ValueError("Real extent in meters must be greater than zero.")

    extent_stats = None
    if mode == "known_extent_mpd":
        extent_stats = mpd_tile_extent_stats(mpd_path)
        source_extent = safe_optional_float((extent_stats.get("axis") or {}).get(axis, {}).get("extent"))
    else:
        source_extent = safe_optional_float(calibration.get("source_extent_units", raw.get("source_extent_units")))
    if source_extent is None or source_extent <= 0:
        raise ValueError("Source extent must be greater than zero.")

    scale = float(real_extent_m) / float(source_extent)
    return scale, {
        "mode": mode,
        "dimension": dimension,
        "axis": axis,
        "real_extent_m": float(real_extent_m),
        "source_extent_units": float(source_extent),
    }, extent_stats


def sanitize_dataset_profile_payload(raw_payload: Any) -> Dict[str, Any]:
    if not isinstance(raw_payload, dict):
        raise ValueError("Dataset profile payload must be an object/dict.")
    name = str(raw_payload.get("name") or "").strip()
    if not name:
        raise ValueError("Dataset name is required.")
    mpd_folder_raw = str(raw_payload.get("mpd_folder") or raw_payload.get("v3c_folder") or "").strip()
    if not mpd_folder_raw:
        raise ValueError("MPD folder is required.")
    mpd_path = find_mpd_in_dataset_folder(mpd_folder_raw)
    if mpd_path is None:
        raise ValueError("No MPD file found in the dataset MPD folder.")
    content_dir = safe_path(require_nonempty_path(raw_payload, "content_dir", "Content dir"))
    if not content_dir.is_dir():
        raise ValueError("Content dir must be an existing PLY directory.")

    already_normalized = parse_bool(raw_payload.get("already_normalized", False))
    translation = raw_payload.get("translation") or [0.0, 0.0, 0.0]
    if not isinstance(translation, list) or len(translation) < 3:
        raise ValueError("translation must contain three numeric values.")
    if already_normalized:
        scale, calibration, extent_stats = 1.0, {"mode": "already_normalized"}, None
        translation = [0.0, 0.0, 0.0]
    else:
        scale, calibration, extent_stats = resolve_dataset_scale(raw_payload, mpd_path)
    dataset_id = safe_id(raw_payload.get("id"), safe_id(name, f"dataset_{uuid4().hex[:8]}"))
    apply_to_fov_trace = False if already_normalized else parse_bool(raw_payload.get("apply_to_fov_trace", True))
    valid_conventions = {item["id"] for item in ANGLE_CONVENTIONS}
    angle_convention = str(raw_payload.get("angle_convention") or DEFAULT_HEADSET_CONFIG["angle_convention"]).strip().lower()
    if angle_convention not in valid_conventions:
        angle_convention = DEFAULT_HEADSET_CONFIG["angle_convention"]
    motion_limits_raw = raw_payload.get("motion_limits") or {}
    if not isinstance(motion_limits_raw, dict):
        motion_limits_raw = {}
    max_speed = safe_optional_float(motion_limits_raw.get("max_speed", raw_payload.get("max_speed")))
    max_acceleration = safe_optional_float(motion_limits_raw.get("max_acceleration", raw_payload.get("max_acceleration")))
    if max_speed is None or max_speed <= 0:
        max_speed = 1.5
    if max_acceleration is None or max_acceleration <= 0:
        max_acceleration = 3.0
    notes = str(raw_payload.get("notes") or "").strip()
    units = "meters"
    transform = {
        "scale": float(scale),
        "translation": [float(translation[0]), float(translation[1]), float(translation[2])],
        "units": units,
        "source": "dataset_profile",
        "notes": notes,
        "apply_to_fov_trace": bool(apply_to_fov_trace),
    }
    return {
        "id": dataset_id,
        "name": name,
        "mpd_folder": path_for_client(mpd_path.parent),
        "mpd": path_for_client(mpd_path),
        "content_dir": path_for_client(content_dir),
        "scale_mode": calibration.get("mode"),
        "angle_convention": angle_convention,
        "calibration": calibration,
        "motion_limits": {
            "max_speed": float(max_speed),
            "max_acceleration": float(max_acceleration),
        },
        "extent_stats": extent_stats,
        "transform": transform,
        "apply_to_fov_trace": bool(apply_to_fov_trace),
        "already_normalized": bool(already_normalized),
        "notes": notes,
        "updated_at": datetime.now().isoformat(),
    }


def save_dataset_profile(raw_payload: Any) -> Dict[str, Any]:
    profile = sanitize_dataset_profile_payload(raw_payload)
    registry = load_dataset_registry()
    entries = dict(registry.get("entries") or {})
    update_existing = parse_bool(raw_payload.get("update_existing")) if isinstance(raw_payload, dict) else False
    if profile["id"] in entries and not update_existing:
        base_id = safe_id(profile.get("name"), f"dataset_{uuid4().hex[:8]}")
        candidate = base_id
        suffix = 2
        while candidate in entries:
            candidate = f"{base_id}_{suffix}"
            suffix += 1
        profile["id"] = candidate
    entries[profile["id"]] = profile
    save_dataset_registry({"entries": entries})

    transform = profile["transform"]
    set_world_transform(
        content_dir=safe_path(profile["content_dir"]),
        scale=transform["scale"],
        translation=transform["translation"],
        units=transform["units"],
        source=transform["source"],
        notes=transform["notes"],
        apply_to_fov_trace=transform["apply_to_fov_trace"],
        metadata={
            "dataset_id": profile["id"],
            "dataset_name": profile["name"],
            "scale_mode": profile["scale_mode"],
            "angle_convention": profile["angle_convention"],
            "calibration": profile["calibration"],
            "motion_limits": profile["motion_limits"],
        },
    )
    return dataset_profiles_payload()


def delete_dataset_profile(dataset_id: str) -> Dict[str, Any]:
    registry = load_dataset_registry()
    entries = dict(registry.get("entries") or {})
    entries.pop(str(dataset_id), None)
    save_dataset_registry({"entries": entries})
    return dataset_profiles_payload()


def run_form_sessions() -> Dict[str, Any]:
    raw = read_json(FORM_SESSIONS_FILE)
    sessions = raw.get("sessions")
    if not isinstance(sessions, dict):
        sessions = {}
    return {"sessions": sessions}


def save_run_form_sessions(payload: Dict[str, Any]) -> Dict[str, Any]:
    FORM_SESSIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with FORM_SESSIONS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def run_form_sessions_payload() -> Dict[str, Any]:
    data = run_form_sessions()
    sessions = []
    for name, entry in sorted((data.get("sessions") or {}).items()):
        if not isinstance(entry, dict):
            continue
        sessions.append(
            {
                "name": name,
                "config": entry.get("config") or {},
                "updated_at": entry.get("updated_at"),
            }
        )
    return {
        "sessions": sessions,
        "settings_path": str(FORM_SESSIONS_FILE.relative_to(PROJECT_ROOT)),
    }


def save_run_form_session(raw_payload: Any) -> Dict[str, Any]:
    if not isinstance(raw_payload, dict):
        raise ValueError("Run form session payload must be an object/dict.")
    name = str(raw_payload.get("name") or "").strip()
    if not name:
        raise ValueError("Session name is required.")
    config = raw_payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("config is required.")
    data = run_form_sessions()
    sessions = dict(data.get("sessions") or {})
    sessions[name] = {
        "config": dict(config),
        "updated_at": datetime.now().isoformat(),
    }
    return save_run_form_sessions({"sessions": sessions})


def delete_run_form_session(name: str) -> Dict[str, Any]:
    data = run_form_sessions()
    sessions = dict(data.get("sessions") or {})
    sessions.pop(name, None)
    return save_run_form_sessions({"sessions": sessions})


def navis_env_settings() -> Dict[str, Any]:
    return {
        "base_url": (os.getenv("PRISM_NAVIS_URL") or "").strip(),
        "health_path": (os.getenv("PRISM_NAVIS_HEALTH_PATH") or "/health").strip() or "/health",
        "export_path": (os.getenv("PRISM_NAVIS_EXPORT_PATH") or "/api/prism/runs").strip() or "/api/prism/runs",
        "timeout_s": max(env_float("PRISM_NAVIS_TIMEOUT_S", 0.35), 0.05),
        "auto_export": parse_bool(os.getenv("PRISM_NAVIS_AUTO_EXPORT") or "0"),
        "auth_token": (os.getenv("PRISM_NAVIS_AUTH_TOKEN") or "").strip(),
    }


def navis_file_settings() -> Dict[str, Any]:
    raw = read_json(NAVIS_CONFIG_FILE)
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Any] = {}
    for key in ("base_url", "health_path", "export_path", "auth_token"):
        if key in raw:
            out[key] = str(raw.get(key) or "").strip()
    if "timeout_s" in raw:
        out["timeout_s"] = max(safe_float(raw.get("timeout_s"), 0.35), 0.05)
    if "auto_export" in raw:
        out["auto_export"] = parse_bool(raw.get("auto_export"))
    return out


def navis_settings() -> Dict[str, Any]:
    cfg = navis_env_settings()
    cfg.update(navis_file_settings())
    cfg["health_path"] = str(cfg.get("health_path") or "/health").strip() or "/health"
    cfg["export_path"] = str(cfg.get("export_path") or "/api/prism/runs").strip() or "/api/prism/runs"
    cfg["timeout_s"] = max(safe_float(cfg.get("timeout_s"), 0.35), 0.05)
    cfg["auto_export"] = parse_bool(cfg.get("auto_export"))
    cfg["base_url"] = str(cfg.get("base_url") or "").strip()
    cfg["auth_token"] = str(cfg.get("auth_token") or "").strip()
    return cfg


def save_navis_file_settings(raw_payload: Any) -> Dict[str, Any]:
    if raw_payload is None:
        raw_payload = {}
    if not isinstance(raw_payload, dict):
        raise ValueError("Navis settings payload must be an object/dict.")
    updated = dict(navis_file_settings())
    for key in ("base_url", "health_path", "export_path", "auth_token"):
        if key in raw_payload:
            updated[key] = str(raw_payload.get(key) or "").strip()
    if "timeout_s" in raw_payload:
        updated["timeout_s"] = max(safe_float(raw_payload.get("timeout_s"), 0.35), 0.05)
    if "auto_export" in raw_payload:
        updated["auto_export"] = parse_bool(raw_payload.get("auto_export"))
    NAVIS_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with NAVIS_CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2)
    return updated


def navis_status(base_url: str | None = None, health_path: str | None = None) -> Dict[str, Any]:
    cfg = navis_settings()
    base = (base_url or cfg["base_url"]).strip()
    health = (health_path or cfg["health_path"]).strip() or "/health"
    status = probe_navis(base, health_path=health, timeout_s=float(cfg["timeout_s"]))
    status["export_path"] = cfg["export_path"]
    status["auto_export"] = bool(cfg["auto_export"])
    return status


def navis_config_payload() -> Dict[str, Any]:
    settings = navis_settings()
    status = navis_status(base_url=settings["base_url"], health_path=settings["health_path"])
    status["export_path"] = settings["export_path"]
    status["auto_export"] = bool(settings["auto_export"])
    return {
        "settings": settings,
        "status": status,
        "settings_path": str(NAVIS_CONFIG_FILE.relative_to(PROJECT_ROOT)),
    }


def sanitize_headset_preset(raw: Any, idx: int) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Headset preset #{idx + 1} must be an object/dict.")
    preset_id = str(raw.get("id") or "").strip()
    label = str(raw.get("label") or "").strip()
    if not preset_id:
        raise ValueError(f"Headset preset #{idx + 1} is missing an id.")
    if not label:
        raise ValueError(f"Headset preset '{preset_id}' is missing a label.")
    fov_h = safe_float(raw.get("fov_h_deg"), 0.0)
    fov_v = safe_float(raw.get("fov_v_deg"), 0.0)
    if not (1.0 <= fov_h <= 360.0):
        raise ValueError(f"Headset preset '{preset_id}' has invalid horizontal FoV.")
    if not (1.0 <= fov_v <= 360.0):
        raise ValueError(f"Headset preset '{preset_id}' has invalid vertical FoV.")
    return {
        "id": preset_id,
        "label": label,
        "fov_h_deg": round(float(fov_h), 3),
        "fov_v_deg": round(float(fov_v), 3),
    }


def sanitize_headset_settings(raw_payload: Any) -> Dict[str, Any]:
    if raw_payload is None:
        raw_payload = {}
    if not isinstance(raw_payload, dict):
        raise ValueError("Headset settings payload must be an object/dict.")
    raw_presets = raw_payload.get("presets")
    if not isinstance(raw_presets, list) or not raw_presets:
        raise ValueError("Headset settings must include a non-empty presets list.")
    presets = [sanitize_headset_preset(item, idx) for idx, item in enumerate(raw_presets)]
    seen: set[str] = set()
    for preset in presets:
        if preset["id"] in seen:
            raise ValueError(f"Duplicate headset preset id '{preset['id']}'.")
        seen.add(preset["id"])
    default_preset = str(raw_payload.get("default_preset") or presets[0]["id"]).strip()
    if default_preset not in seen:
        default_preset = presets[0]["id"]
    valid_conventions = {item["id"] for item in ANGLE_CONVENTIONS}
    angle_convention = str(raw_payload.get("angle_convention") or DEFAULT_HEADSET_CONFIG["angle_convention"]).strip().lower()
    if angle_convention not in valid_conventions:
        angle_convention = DEFAULT_HEADSET_CONFIG["angle_convention"]
    return {
        "default_preset": default_preset,
        "angle_convention": angle_convention,
        "presets": presets,
    }


def headset_settings() -> Dict[str, Any]:
    raw = read_json(HEADSET_CONFIG_FILE)
    if not raw:
        return dict(DEFAULT_HEADSET_CONFIG)
    try:
        return sanitize_headset_settings(raw)
    except ValueError:
        return dict(DEFAULT_HEADSET_CONFIG)


def save_headset_settings(raw_payload: Any) -> Dict[str, Any]:
    settings = sanitize_headset_settings(raw_payload)
    HEADSET_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with HEADSET_CONFIG_FILE.open("w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
    return settings


def headset_config_payload() -> Dict[str, Any]:
    return {
        "settings": headset_settings(),
        "angle_conventions": ANGLE_CONVENTIONS,
        "settings_path": str(HEADSET_CONFIG_FILE.relative_to(PROJECT_ROOT)),
    }


def headset_preset_map() -> Dict[str, Dict[str, Any]]:
    presets = headset_settings().get("presets") or []
    return {str(preset.get("id")): preset for preset in presets if isinstance(preset, dict)}


def resolve_headset_config(preset_id: Any, fov_h_deg: Any, fov_v_deg: Any, angle_convention: Any) -> Dict[str, Any]:
    settings = headset_settings()
    presets = headset_preset_map()
    selected_id = str(preset_id or settings.get("default_preset") or "").strip()
    selected = presets.get(selected_id)
    if selected is None and presets:
        fallback_id = str(settings.get("default_preset") or next(iter(presets.keys())))
        selected = presets.get(fallback_id) or next(iter(presets.values()))
        selected_id = str(selected.get("id") or fallback_id)
    resolved_h = safe_float(fov_h_deg, float(selected.get("fov_h_deg") if selected else 97.0))
    resolved_v = safe_float(fov_v_deg, float(selected.get("fov_v_deg") if selected else 93.0))
    valid_conventions = {item["id"] for item in ANGLE_CONVENTIONS}
    resolved_convention = str(angle_convention or settings.get("angle_convention") or DEFAULT_HEADSET_CONFIG["angle_convention"]).strip().lower()
    if resolved_convention not in valid_conventions:
        resolved_convention = DEFAULT_HEADSET_CONFIG["angle_convention"]
    return {
        "headset_preset": selected_id or None,
        "fov_h_deg": resolved_h,
        "fov_v_deg": resolved_v,
        "angle_convention": resolved_convention,
    }
