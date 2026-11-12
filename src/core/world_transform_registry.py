from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from core.media_segment import TileBounds

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results"
SETTINGS_ROOT = PROJECT_ROOT / ".prism_settings"
WORLD_TRANSFORMS_FILE = SETTINGS_ROOT / "world_transforms.json"

UNIT_TO_METERS = {
    "meters": 1.0,
    "meter": 1.0,
    "m": 1.0,
    "centimeters": 0.01,
    "centimeter": 0.01,
    "cm": 0.01,
    "millimeters": 0.001,
    "millimeter": 0.001,
    "mm": 0.001,
}


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def normalize_content_dir(content_dir: str | Path) -> str:
    return str(Path(content_dir).expanduser().resolve())


def load_world_transform_registry() -> Dict[str, Any]:
    raw = _read_json(WORLD_TRANSFORMS_FILE)
    entries = raw.get("entries")
    if not isinstance(entries, dict):
        entries = {}
    return {"version": 1, "entries": entries}


def save_world_transform_registry(registry: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {"version": 1, "entries": dict(registry.get("entries") or {})}
    _write_json(WORLD_TRANSFORMS_FILE, normalized)
    return normalized


def get_world_transform(content_dir: str | Path) -> Dict[str, Any] | None:
    key = normalize_content_dir(content_dir)
    registry = load_world_transform_registry()
    value = registry["entries"].get(key)
    return dict(value) if isinstance(value, dict) else None


def set_world_transform(
    content_dir: str | Path,
    scale: float,
    translation: list[float] | tuple[float, float, float],
    units: str = "meters",
    source: str = "manual",
    notes: str = "",
    apply_to_fov_trace: bool = True,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    key = normalize_content_dir(content_dir)
    registry = load_world_transform_registry()
    registry["entries"][key] = {
        "scale": float(scale),
        "translation": [float(v) for v in translation[:3]],
        "units": str(units or "meters"),
        "source": str(source or "manual"),
        "notes": str(notes or ""),
        "apply_to_fov_trace": bool(apply_to_fov_trace),
    }
    if metadata:
        registry["entries"][key]["metadata"] = dict(metadata)
    save_world_transform_registry(registry)
    return dict(registry["entries"][key])


def list_world_transforms() -> list[Dict[str, Any]]:
    registry = load_world_transform_registry()
    out = []
    for content_dir, transform in sorted(registry["entries"].items()):
        if not isinstance(transform, dict):
            continue
        out.append({"content_dir": content_dir, "transform": dict(transform)})
    return out


def detect_header_transform(content_dir: str | Path) -> Dict[str, Any] | None:
    directory = Path(content_dir).expanduser().resolve()
    if not directory.is_dir():
        return None
    first_ply = next(iter(sorted(directory.glob("*.ply"))), None)
    if first_ply is None:
        return None

    scale = None
    translation = None
    try:
        with first_ply.open("rb") as handle:
            for raw_line in handle:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if line == "end_header":
                    break
                if line.startswith("comment frame_to_world_scale "):
                    _, _, value = line.partition("comment frame_to_world_scale ")
                    try:
                        scale = float(value.strip())
                    except ValueError:
                        scale = None
                elif line.startswith("comment frame_to_world_translation "):
                    _, _, value = line.partition("comment frame_to_world_translation ")
                    parts = value.strip().split()
                    if len(parts) >= 3:
                        try:
                            translation = [float(parts[0]), float(parts[1]), float(parts[2])]
                        except ValueError:
                            translation = None
    except OSError:
        return None

    if scale is None or translation is None:
        return None
    return {
        "scale": float(scale),
        "translation": translation,
        "units": "meters",
        "source": "ply_header",
        "notes": f"Detected from {first_ply.name}",
    }


def _coordinate_output_dtype(frame_df: pd.DataFrame, columns: list[str] | tuple[str, ...]) -> np.dtype:
    dtypes = [frame_df[col].dtype for col in columns if col in frame_df.columns]
    if dtypes and all(np.issubdtype(dtype, np.floating) for dtype in dtypes):
        return np.dtype(np.result_type(*dtypes))
    return np.dtype(np.float64)


def _assign_coordinate_columns(frame_df: pd.DataFrame, columns: list[str] | tuple[str, ...], values: np.ndarray) -> None:
    dtype = _coordinate_output_dtype(frame_df, columns)
    cast_values = np.asarray(values, dtype=dtype)
    for idx, col in enumerate(columns):
        frame_df[col] = cast_values[:, idx]


def apply_world_transform(frame_path: str | Path, frame_df: pd.DataFrame) -> pd.DataFrame:
    content_dir = Path(frame_path).expanduser().resolve().parent
    transform = get_world_transform(content_dir)
    if not transform:
        return frame_df
    if not {"x", "y", "z"}.issubset(frame_df.columns):
        return frame_df

    unit_scale = float(UNIT_TO_METERS.get(str(transform.get("units") or "meters").strip().lower(), 1.0))
    scale = float(transform.get("scale", 1.0)) * unit_scale
    translation = np.asarray(transform.get("translation") or [0.0, 0.0, 0.0], dtype=float)
    if translation.shape != (3,):
        translation = np.asarray([0.0, 0.0, 0.0], dtype=float)
    translation = translation * unit_scale

    out = frame_df.copy()
    xyz = out.loc[:, ["x", "y", "z"]].to_numpy(dtype=float)
    _assign_coordinate_columns(out, ["x", "y", "z"], (xyz * scale) + translation)
    return out


def apply_world_transform_to_trace(csv_path: str | Path, trace_df: pd.DataFrame, content_dir: str | Path) -> pd.DataFrame:
    """Apply the saved sequence world transform to known FoV trace position columns."""
    transform = get_world_transform(content_dir)
    if not transform:
        return trace_df
    if not bool(transform.get("apply_to_fov_trace", True)):
        return trace_df

    position_column_sets = [
        ("HMDPX", "HMDPY", "HMDPZ"),
        ("PositionX", "PositionY", "PositionZ"),
        ("pos_x", "pos_y", "pos_z"),
        ("x", "y", "z"),
        ("X", "Y", "Z"),
    ]
    unit_scale = float(UNIT_TO_METERS.get(str(transform.get("units") or "meters").strip().lower(), 1.0))
    scale = float(transform.get("scale", 1.0)) * unit_scale
    translation = np.asarray(transform.get("translation") or [0.0, 0.0, 0.0], dtype=float)
    if translation.shape != (3,):
        translation = np.asarray([0.0, 0.0, 0.0], dtype=float)
    translation = translation * unit_scale

    out = trace_df.copy()
    for cols in position_column_sets:
        if not all(col in out.columns for col in cols):
            continue
        xyz = out.loc[:, list(cols)].to_numpy(dtype=float)
        _assign_coordinate_columns(out, list(cols), (xyz * scale) + translation)
    return out


def apply_world_transform_to_tile_bounds(bounds: TileBounds, content_dir: str | Path) -> TileBounds:
    """Apply the saved sequence world transform to MPD tile bounds and convert them to meters."""
    transform = get_world_transform(content_dir)
    if not transform:
        return bounds

    unit_scale = float(UNIT_TO_METERS.get(str(transform.get("units") or "meters").strip().lower(), 1.0))
    scale = float(transform.get("scale", 1.0)) * unit_scale
    translation = np.asarray(transform.get("translation") or [0.0, 0.0, 0.0], dtype=float)
    if translation.shape != (3,):
        translation = np.asarray([0.0, 0.0, 0.0], dtype=float)
    translation = translation * unit_scale

    min_corner = (np.asarray(bounds.min_corner, dtype=float) * scale) + translation
    max_corner = (np.asarray(bounds.max_corner, dtype=float) * scale) + translation
    transformed_min = np.minimum(min_corner, max_corner)
    transformed_max = np.maximum(min_corner, max_corner)
    return TileBounds(
        min_corner=tuple(float(v) for v in transformed_min),
        max_corner=tuple(float(v) for v in transformed_max),
    )
