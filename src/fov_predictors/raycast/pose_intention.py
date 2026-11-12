"""Pose-history feature extraction for the raycast predictor.

This module translates FoV trace rows into position, gaze, motion, and a single
confidence value that says how strongly recent movement should influence the
reachable-voxel distribution computed in `reachability_pipeline.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from fov_predictors.base import FovPredictor
from core.rotations import forward_from_quaternion_xyzw, quaternion_from_row


@dataclass
class IntentionEstimate:
    """Compact summary of inferred movement and gaze intent for the current frame."""
    move_dir: np.ndarray
    latest_velocity_dir: np.ndarray
    gaze_dir: np.ndarray
    speed: float
    accel: float
    mismatch: float
    normalized_speed: float
    normalized_acceleration: float
    motion_intensity: float
    motion_stability: float
    direction_reliability: float
    confidence: float
    effective_confidence: float


def _safe_get(row: Mapping, key: str):
    """Read a key from a row-like object without assuming a concrete type."""
    if hasattr(row, "get"):
        return row.get(key, None)
    return None


def _unit(vec: Sequence[float]) -> np.ndarray:
    """Return a normalized 3D vector, or zeros for degenerate input."""
    arr = np.asarray(vec, dtype=float)
    n = float(np.linalg.norm(arr))
    if n <= 1e-12:
        return np.zeros(3, dtype=float)
    return arr / n


def extract_position(row: Mapping) -> np.ndarray:
    """Extract the user position from one pose row using known column aliases."""
    candidates = [
        ("HMDPX", "HMDPY", "HMDPZ"),
        ("PositionX", "PositionY", "PositionZ"),
        ("pos_x", "pos_y", "pos_z"),
        ("x", "y", "z"),
        ("X", "Y", "Z"),
    ]
    for keys in candidates:
        vals = [_safe_get(row, k) for k in keys]
        if any(v is None for v in vals):
            continue
        arr = np.asarray(vals, dtype=float)
        if np.all(np.isfinite(arr)):
            return arr
    raise KeyError("Could not find position columns in FoV row")


def compute_gaze_dir(row: Mapping, angle_convention: str = FovPredictor.DEFAULT_ANGLE_CONVENTION) -> np.ndarray:
    """Extract or reconstruct the current gaze/forward direction from a pose row."""
    vec_candidates = [
        ("ForwardX", "ForwardY", "ForwardZ"),
        ("forward_x", "forward_y", "forward_z"),
        ("LookX", "LookY", "LookZ"),
    ]
    for keys in vec_candidates:
        vals = [_safe_get(row, k) for k in keys]
        if any(v is None for v in vals):
            continue
        forward = _unit(vals)
        if np.any(forward):
            return forward

    try:
        forward = _unit(forward_from_quaternion_xyzw(quaternion_from_row(row, angle_convention=angle_convention)))
        if np.any(forward):
            return forward
    except (TypeError, ValueError):
        pass

    euler_candidates = [
        ("HMDRX", "HMDRY", "HMDRZ"),
        ("HeadRotX", "HeadRotY", "HeadRotZ"),
        ("rot_x", "rot_y", "rot_z"),
        ("RX", "RY", "RZ"),
    ]
    for keys in euler_candidates:
        vals = [_safe_get(row, k) for k in keys]
        if any(v is None for v in vals):
            continue
        rx, ry, rz = [float(v) for v in vals]
        forward = _unit(FovPredictor.forward_from_euler_deg(rx, ry, rz, angle_convention=angle_convention))
        if np.any(forward):
            return forward

    return np.array([0.0, 0.0, 1.0], dtype=float)


def _extract_times(history_df, assumed_fps: float) -> np.ndarray:
    """Resolve timestamps for the history window, falling back to frame index time."""
    time_columns = ["Time", "Time_s", "Timestamp", "time_s", "time"]
    for col in time_columns:
        if hasattr(history_df, "columns") and col in history_df.columns:
            arr = history_df[col].to_numpy(dtype=float)
            if arr.size and np.all(np.isfinite(arr)):
                return arr

    idx = history_df.index.to_numpy(dtype=float)
    if idx.size == 0:
        return np.array([], dtype=float)
    fps = max(float(assumed_fps), 1e-6)
    return idx / fps


def _moving_average_rows(values: np.ndarray, window: int) -> np.ndarray:
    """Apply a trailing moving average across row vectors."""
    if values.size == 0:
        return values
    w = max(1, int(window))
    out = np.zeros_like(values, dtype=float)
    for i in range(values.shape[0]):
        s = max(0, i - w + 1)
        out[i] = np.mean(values[s : i + 1], axis=0)
    return out


def compute_motion_estimates(history_df, assumed_fps: float = 30.0, smooth_window: int = 3):
    """Estimate current speed, acceleration, and motion direction from pose history."""
    if history_df is None or len(history_df) < 2:
        return 0.0, 0.0, np.zeros(3, dtype=float), np.zeros(3, dtype=float)

    positions = np.vstack([extract_position(history_df.iloc[i]) for i in range(len(history_df))])
    times = _extract_times(history_df, assumed_fps=assumed_fps)

    if positions.shape[0] < 2 or times.shape[0] < 2:
        return 0.0, 0.0, np.zeros(3, dtype=float), np.zeros(3, dtype=float)

    dpos = np.diff(positions, axis=0)
    dt = np.diff(times)
    dt = np.where(dt > 1e-6, dt, 1e-3)

    velocities = dpos / dt[:, None]
    velocities = _moving_average_rows(velocities, smooth_window)

    speed_series = np.linalg.norm(velocities, axis=1)
    speed = float(speed_series[-1]) if speed_series.size else 0.0
    latest_velocity_dir = _unit(velocities[-1]) if speed > 1e-6 else np.zeros(3, dtype=float)

    accel = 0.0
    if velocities.shape[0] >= 2:
        dv = velocities[-1] - velocities[-2]
        accel = float(np.linalg.norm(dv) / max(dt[-1], 1e-3))

    # The reachability predictor evaluates multiple history horizons. Using
    # only the last velocity makes those horizons collapse to the same direction;
    # use the net displacement over the selected window for the direction while
    # keeping latest speed/acceleration for confidence.
    displacement = positions[-1] - positions[0]
    move_dir = _unit(displacement) if np.linalg.norm(displacement) > 1e-6 else (
        latest_velocity_dir if speed > 1e-6 else np.zeros(3, dtype=float)
    )
    return speed, accel, move_dir, latest_velocity_dir


def confidence(
    speed: float,
    accel: float,
    move_dir: np.ndarray,
    gaze_dir: np.ndarray,
    max_speed: float,
    max_acceleration: float,
    stability_weight: float,
    mismatch_penalty_strength: float,
) -> tuple[float, float, float, float, float, float, float, float]:
    """Compute bounded motion-regime confidence and direction reliability."""
    max_speed = max(float(max_speed), 1e-6)
    max_acceleration = max(float(max_acceleration), 1e-6)
    stability_weight = float(np.clip(float(stability_weight), 0.0, 1.0))
    mismatch_penalty_strength = float(np.clip(float(mismatch_penalty_strength), 0.0, 1.0))

    # Normalize dataset-scaled physical quantities so confidence stays bounded
    # and comparable across traces with different units or motion capture ranges.
    normalized_speed = float(np.clip(float(speed) / max_speed, 0.0, 1.0))
    normalized_acceleration = float(np.clip(abs(float(accel)) / max_acceleration, 0.0, 1.0))

    # Intensity captures any strong motion; stability captures predictable motion.
    motion_intensity = max(normalized_speed, normalized_acceleration)
    motion_stability = normalized_speed * (1.0 - normalized_acceleration)
    motion_confidence = (
        stability_weight * motion_stability
        + (1.0 - stability_weight) * motion_intensity
    )

    if np.linalg.norm(move_dir) <= 1e-9 or np.linalg.norm(gaze_dir) <= 1e-9:
        mismatch = 0.0
    else:
        dot = float(np.clip(np.dot(move_dir, gaze_dir), -1.0, 1.0))
        mismatch = (1.0 - dot) * 0.5

    direction_reliability = 1.0 - mismatch_penalty_strength * (1.0 - float(np.exp(-mismatch)))
    direction_reliability = float(np.clip(direction_reliability, 0.0, 1.0))
    effective_confidence = float(np.clip(motion_confidence * direction_reliability, 0.0, 1.0))
    return (
        float(motion_confidence),
        mismatch,
        effective_confidence,
        normalized_speed,
        normalized_acceleration,
        float(motion_intensity),
        float(motion_stability),
        direction_reliability,
    )


def estimate_intention(
    pose_now,
    pose_history,
    max_speed: float,
    max_acceleration: float,
    stability_weight: float,
    mismatch_penalty_strength: float,
    assumed_fps: float = 30.0,
    smooth_window: int = 3,
    angle_convention: str = FovPredictor.DEFAULT_ANGLE_CONVENTION,
) -> IntentionEstimate:
    """Build the intention estimate consumed by the reachability pipeline."""
    gaze_dir = compute_gaze_dir(pose_now, angle_convention=angle_convention)
    speed, accel, move_dir, latest_velocity_dir = compute_motion_estimates(
        pose_history,
        assumed_fps=assumed_fps,
        smooth_window=smooth_window,
    )

    (
        motion_confidence,
        mismatch,
        effective_confidence,
        normalized_speed,
        normalized_acceleration,
        motion_intensity,
        motion_stability,
        direction_reliability,
    ) = confidence(
        speed=speed,
        accel=accel,
        move_dir=move_dir,
        gaze_dir=gaze_dir,
        max_speed=max_speed,
        max_acceleration=max_acceleration,
        stability_weight=stability_weight,
        mismatch_penalty_strength=mismatch_penalty_strength,
    )

    return IntentionEstimate(
        move_dir=move_dir,
        latest_velocity_dir=latest_velocity_dir,
        gaze_dir=gaze_dir,
        speed=speed,
        accel=accel,
        mismatch=mismatch,
        normalized_speed=normalized_speed,
        normalized_acceleration=normalized_acceleration,
        motion_intensity=motion_intensity,
        motion_stability=motion_stability,
        direction_reliability=direction_reliability,
        confidence=float(motion_confidence),
        effective_confidence=float(effective_confidence),
    )
