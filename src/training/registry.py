from __future__ import annotations

from typing import Any

from core.rotations import QUAT_COLUMNS, EULER_COLUMNS, ensure_fov_quaternions
from core.trace_columns import normalize_fov_dataframe, normalize_network_dataframe
from server_app.config import get_dataset_profile, path_for_client, safe_path

from .config import TRAINING_DATASETS_FILE, list_csv_files, new_id, now_iso, read_json, tree_fingerprint, write_json


def _registry() -> dict[str, Any]:
    raw = read_json(TRAINING_DATASETS_FILE, {"version": 1, "entries": {}})
    entries = raw.get("entries") if isinstance(raw, dict) else {}
    return {"version": 1, "entries": entries if isinstance(entries, dict) else {}}


def list_training_datasets() -> list[dict[str, Any]]:
    entries = _registry()["entries"]
    out = []
    for dataset_id, entry in sorted(entries.items()):
        if isinstance(entry, dict):
            item = dict(entry)
            item["id"] = str(item.get("id") or dataset_id)
            out.append(item)
    return out


def get_training_dataset(dataset_id: str | None) -> dict[str, Any] | None:
    if not dataset_id:
        return None
    entry = _registry()["entries"].get(str(dataset_id))
    return dict(entry) if isinstance(entry, dict) else None


def _validate_trace_sample(family: str, path: str, angle_convention: str | None = None) -> dict[str, Any]:
    files = list_csv_files(path)
    if not files:
        raise ValueError("At least one CSV trace file is required.")
    import pandas as pd

    sample = files[0]
    df = pd.read_csv(sample, nrows=10)
    if family == "fov":
        normalized = normalize_fov_dataframe(df, source=sample)
        has_quat = all(column in normalized.columns for column in QUAT_COLUMNS)
        has_euler = all(column in normalized.columns for column in EULER_COLUMNS)
        if not has_quat and not has_euler:
            raise ValueError("FoV traces must contain either quaternion columns or Euler rotation columns.")
        normalized = ensure_fov_quaternions(normalized, angle_convention=angle_convention)
        required = ["FrameNumber", "HMDPX", "HMDPY", "HMDPZ", *QUAT_COLUMNS]
    else:
        normalized = normalize_network_dataframe(df, source=sample)
        required = ["Time", "Bandwidth"]
    missing = [name for name in required if name not in normalized.columns]
    if missing:
        raise ValueError(f"Trace sample is missing required normalized columns: {', '.join(missing)}")
    return {"file_count": len(files), "sample_file": path_for_client(sample)}


def _prism_dataset_snapshot(profile: dict[str, Any] | None) -> dict[str, Any] | None:
    if not profile:
        return None
    keys = (
        "id",
        "name",
        "mpd_folder",
        "mpd",
        "content_dir",
        "scale_mode",
        "calibration",
        "motion_limits",
        "extent_stats",
        "transform",
        "angle_convention",
        "apply_to_fov_trace",
        "already_normalized",
        "updated_at",
    )
    return {key: profile.get(key) for key in keys if key in profile}


def save_training_dataset(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Training dataset payload must be an object/dict.")
    family = str(payload.get("family") or "").strip().lower()
    if family not in {"fov", "bandwidth"}:
        raise ValueError("Training dataset family must be 'fov' or 'bandwidth'.")
    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError("Dataset name is required.")
    trace_path_raw = str(payload.get("trace_path") or "").strip()
    if not trace_path_raw:
        raise ValueError("Trace path is required.")
    trace_path = safe_path(trace_path_raw)
    if not trace_path.exists():
        raise ValueError("Trace path does not exist.")
    dataset_id = str(payload.get("id") or "").strip() or new_id("tds")
    imported = None
    prism_dataset_id = str(payload.get("prism_dataset_id") or "").strip()
    if prism_dataset_id:
        imported = get_dataset_profile(prism_dataset_id)
        if imported is None:
            raise ValueError("Referenced Prism dataset profile was not found.")
    elif family == "fov":
        raise ValueError("FoV training datasets must reference a Prism dataset profile for geometry and calibration metadata.")
    angle_convention = str((imported or {}).get("angle_convention") or payload.get("angle_convention") or "unity_zxy")
    validation = _validate_trace_sample(family, str(trace_path), angle_convention=angle_convention)
    content_dir_raw = str((imported or {}).get("content_dir") or payload.get("content_dir") or "").strip()
    reference_ply_raw = str(payload.get("reference_ply") or "").strip()
    if family == "fov" and content_dir_raw:
        content_dir_path = safe_path(content_dir_raw)
        if not content_dir_path.exists() or not content_dir_path.is_dir():
            raise ValueError("Imported Prism dataset content_dir must be an existing PLY directory.")
        if not reference_ply_raw:
            first_ply = next(iter(sorted(content_dir_path.rglob("*.ply"))), None)
            if first_ply is not None:
                reference_ply_raw = path_for_client(first_ply)
    elif family == "fov":
        raise ValueError("Referenced Prism dataset must define a content_dir for FoV training.")
    if family == "fov" and reference_ply_raw:
        reference_ply_path = safe_path(reference_ply_raw)
        if not reference_ply_path.exists() or reference_ply_path.suffix.lower() != ".ply":
            raise ValueError("reference_ply must be an existing .ply file.")

    entry = {
        "id": dataset_id,
        "family": family,
        "name": name,
        "trace_path": path_for_client(trace_path),
        "fps": float(payload.get("fps") or (30.0 if family == "fov" else 1.0)),
        "units": str(payload.get("units") or ("frames" if family == "fov" else "mbps")),
        "angle_convention": angle_convention,
        "content_dir": content_dir_raw,
        "reference_ply": reference_ply_raw,
        "prism_dataset_id": prism_dataset_id or None,
        "prism_dataset_snapshot": _prism_dataset_snapshot(imported),
        "transform": (imported or {}).get("transform"),
        "motion_limits": (imported or {}).get("motion_limits"),
        "notes": str(payload.get("notes") or ""),
        "validation": validation,
        "fingerprint": tree_fingerprint(trace_path),
        "updated_at": now_iso(),
    }
    registry = _registry()
    entries = dict(registry["entries"])
    entries[dataset_id] = entry
    write_json(TRAINING_DATASETS_FILE, {"version": 1, "entries": entries})
    return {"datasets": list_training_datasets(), "settings_path": str(TRAINING_DATASETS_FILE)}


def delete_training_dataset(dataset_id: str) -> dict[str, Any]:
    registry = _registry()
    entries = dict(registry["entries"])
    entries.pop(str(dataset_id), None)
    write_json(TRAINING_DATASETS_FILE, {"version": 1, "entries": entries})
    return {"datasets": list_training_datasets(), "settings_path": str(TRAINING_DATASETS_FILE)}


def training_datasets_payload() -> dict[str, Any]:
    return {"datasets": list_training_datasets(), "settings_path": str(TRAINING_DATASETS_FILE)}
