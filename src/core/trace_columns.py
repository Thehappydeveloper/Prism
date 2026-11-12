from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_ROOT = PROJECT_ROOT / ".prism_settings"
TRACE_COLUMNS_FILE = SETTINGS_ROOT / "trace_columns.json"


TRACE_COLUMN_FIELDS: dict[str, dict[str, dict[str, Any]]] = {
    "fov": {
        "frame_number": {
            "label": "Frame number / timestamp",
            "canonical": "FrameNumber",
            "aliases": ["Frame", "frame", "frame_number", "frame_id", "FrameId", "FrameID", "timestamp", "Timestamp", "time", "Time", "Time_s"],
            "required": True,
        },
        "position_x": {
            "label": "Position X",
            "canonical": "HMDPX",
            "aliases": ["Position_X", "PositionX", "pos_x", "x", "X", "HMDPositionX"],
            "required": True,
        },
        "position_y": {
            "label": "Position Y",
            "canonical": "HMDPY",
            "aliases": ["Position_Y", "PositionY", "pos_y", "y", "Y", "HMDPositionY"],
            "required": True,
        },
        "position_z": {
            "label": "Position Z",
            "canonical": "HMDPZ",
            "aliases": ["Position_Z", "PositionZ", "pos_z", "z", "Z", "HMDPositionZ"],
            "required": True,
        },
        "rotation_x": {
            "label": "Euler / rotation X",
            "canonical": "HMDRX",
            "aliases": ["Euler_X", "EulerX", "HeadRotX", "Rotation_X", "RotationX", "rot_x", "rx", "RX"],
            "required": True,
        },
        "rotation_y": {
            "label": "Euler / rotation Y",
            "canonical": "HMDRY",
            "aliases": ["Euler_Y", "EulerY", "HeadRotY", "Rotation_Y", "RotationY", "rot_y", "ry", "RY"],
            "required": True,
        },
        "rotation_z": {
            "label": "Euler / rotation Z",
            "canonical": "HMDRZ",
            "aliases": ["Euler_Z", "EulerZ", "HeadRotZ", "Rotation_Z", "RotationZ", "rot_z", "rz", "RZ"],
            "required": True,
        },
        "quaternion_x": {
            "label": "Quaternion X",
            "canonical": "Quat_X",
            "aliases": ["QuatX", "quat_x", "qx", "QX"],
            "required": False,
        },
        "quaternion_y": {
            "label": "Quaternion Y",
            "canonical": "Quat_Y",
            "aliases": ["QuatY", "quat_y", "qy", "QY"],
            "required": False,
        },
        "quaternion_z": {
            "label": "Quaternion Z",
            "canonical": "Quat_Z",
            "aliases": ["QuatZ", "quat_z", "qz", "QZ"],
            "required": False,
        },
        "quaternion_w": {
            "label": "Quaternion W",
            "canonical": "Quat_W",
            "aliases": ["QuatW", "quat_w", "qw", "QW"],
            "required": False,
        },
    },
    "network": {
        "time": {
            "label": "Time",
            "canonical": "Time",
            "aliases": ["Time_s", "time", "time_s", "timestamp", "Timestamp", "Seconds", "second"],
            "required": True,
        },
        "bandwidth": {
            "label": "Bandwidth",
            "canonical": "Bandwidth",
            "aliases": ["Bandwidth_Mbps", "bandwidth", "bandwidth_mbps", "throughput", "Throughput", "Mbps", "mbps"],
            "required": True,
        },
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def default_trace_column_settings() -> dict[str, dict[str, list[str]]]:
    return {
        trace_type: {
            field_id: list(field["aliases"])
            for field_id, field in fields.items()
        }
        for trace_type, fields in TRACE_COLUMN_FIELDS.items()
    }


def _clean_aliases(value: Any) -> list[str]:
    if isinstance(value, str):
        parts = value.replace("\n", ",").split(",")
    elif isinstance(value, list):
        parts = value
    else:
        parts = []
    aliases: list[str] = []
    seen: set[str] = set()
    for item in parts:
        alias = str(item or "").strip()
        key = alias.lower()
        if alias and key not in seen:
            aliases.append(alias)
            seen.add(key)
    return aliases


def sanitize_trace_column_settings(raw_payload: Any) -> dict[str, dict[str, list[str]]]:
    if raw_payload is None:
        raw_payload = {}
    if not isinstance(raw_payload, dict):
        raise ValueError("Trace column mapping settings payload must be an object/dict.")

    defaults = default_trace_column_settings()
    settings: dict[str, dict[str, list[str]]] = {}
    for trace_type, fields in TRACE_COLUMN_FIELDS.items():
        raw_trace = raw_payload.get(trace_type, {})
        if raw_trace is None:
            raw_trace = {}
        if not isinstance(raw_trace, dict):
            raise ValueError(f"Trace column mapping '{trace_type}' must be an object/dict.")
        settings[trace_type] = {}
        for field_id in fields:
            aliases = _clean_aliases(raw_trace.get(field_id, defaults[trace_type][field_id]))
            settings[trace_type][field_id] = aliases
    return settings


def trace_column_settings() -> dict[str, dict[str, list[str]]]:
    raw = _read_json(TRACE_COLUMNS_FILE)
    try:
        return sanitize_trace_column_settings(raw)
    except ValueError:
        return default_trace_column_settings()


def save_trace_column_settings(raw_payload: Any) -> dict[str, dict[str, list[str]]]:
    settings = sanitize_trace_column_settings(raw_payload)
    TRACE_COLUMNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TRACE_COLUMNS_FILE.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return settings


def trace_column_config_payload() -> dict[str, Any]:
    return {
        "settings": trace_column_settings(),
        "fields": TRACE_COLUMN_FIELDS,
        "settings_path": str(TRACE_COLUMNS_FILE.relative_to(PROJECT_ROOT)),
    }


def _attempted_names(trace_type: str, field_id: str, settings: Mapping[str, Mapping[str, list[str]]]) -> list[str]:
    field = TRACE_COLUMN_FIELDS[trace_type][field_id]
    return [str(field["canonical"]), *list(settings.get(trace_type, {}).get(field_id, []))]


def resolve_column(
    columns: Iterable[str],
    trace_type: str,
    field_id: str,
    settings: Mapping[str, Mapping[str, list[str]]] | None = None,
) -> tuple[str | None, list[str]]:
    active_settings = settings or trace_column_settings()
    attempted = _attempted_names(trace_type, field_id, active_settings)
    column_by_lower = {str(column).strip().lower(): str(column) for column in columns}
    for name in attempted:
        match = column_by_lower.get(str(name).strip().lower())
        if match is not None:
            return match, attempted
    return None, attempted


def normalize_trace_dataframe(
    df: pd.DataFrame,
    trace_type: str,
    source: str | Path | None = None,
    required_only: bool = True,
) -> pd.DataFrame:
    settings = trace_column_settings()
    rename: dict[str, str] = {}
    missing: list[tuple[str, str, list[str]]] = []
    for field_id, field in TRACE_COLUMN_FIELDS[trace_type].items():
        if required_only and not bool(field.get("required")):
            continue
        actual, attempted = resolve_column(df.columns, trace_type, field_id, settings)
        if actual is None:
            if bool(field.get("required")):
                missing.append((field_id, str(field["label"]), attempted))
            continue
        canonical = str(field["canonical"])
        if actual != canonical:
            rename[actual] = canonical

    if missing:
        details = []
        for field_id, label, attempted in missing:
            details.append(f"{label} ({field_id}); attempted: {', '.join(attempted)}")
        source_text = f" in {source}" if source else ""
        headers = ", ".join(str(col) for col in df.columns)
        raise ValueError(
            f"Could not parse {trace_type} CSV{source_text}. Missing required field(s): "
            f"{'; '.join(details)}. Detected headers: {headers}"
        )

    return df.rename(columns=rename)


def normalize_fov_dataframe(df: pd.DataFrame, source: str | Path | None = None) -> pd.DataFrame:
    return normalize_trace_dataframe(df, "fov", source=source, required_only=False)


def normalize_network_dataframe(df: pd.DataFrame, source: str | Path | None = None) -> pd.DataFrame:
    return normalize_trace_dataframe(df, "network", source=source, required_only=True)


def read_trace_csv_subset(path: str | Path, trace_type: str, include_tile_columns: bool = False) -> pd.DataFrame:
    source = Path(path)
    headers = pd.read_csv(source, nrows=0)
    settings = trace_column_settings()
    selected: list[str] = []
    seen: set[str] = set()
    for field_id in TRACE_COLUMN_FIELDS[trace_type]:
        actual, _ = resolve_column(headers.columns, trace_type, field_id, settings)
        if actual is not None and actual not in seen:
            selected.append(actual)
            seen.add(actual)
    if include_tile_columns:
        for column in headers.columns:
            column_name = str(column)
            if column_name.startswith("tile_") and column_name not in seen:
                selected.append(column_name)
                seen.add(column_name)
    return pd.read_csv(source, usecols=selected) if selected else pd.read_csv(source)
