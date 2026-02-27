from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class IntentionEstimate:
    move_dir: np.ndarray
    gaze_dir: np.ndarray
    speed: float
    accel: float
    mismatch: float
    confidence: float
    effective_confidence: float


def _safe_get(row: Mapping, key: str):
    if hasattr(row, "get"):
        return row.get(key, None)
    return None


def _unit(vec: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    n = float(np.linalg.norm(arr))
    if n <= 1e-12:
        return np.zeros(3, dtype=float)
    return arr / n


def extract_position(row: Mapping) -> np.ndarray:
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


def compute_gaze_dir(row: Mapping) -> np.ndarray:
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
        # Keep the same convention as existing Prism baseline predictors.
        rot = R.from_euler("ZXY", [rz, rx, ry], degrees=True)
        forward = _unit(rot.apply([0, 0, 1]))
        if np.any(forward):
            return forward

    return np.array([0.0, 0.0, 1.0], dtype=float)


def _extract_times(history_df, assumed_fps: float) -> np.ndarray:
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
    if values.size == 0:
        return values
    w = max(1, int(window))
    out = np.zeros_like(values, dtype=float)
    for i in range(values.shape[0]):
        s = max(0, i - w + 1)
        out[i] = np.mean(values[s : i + 1], axis=0)
    return out


def compute_motion_estimates(history_df, assumed_fps: float = 30.0, smooth_window: int = 3):
    if history_df is None or len(history_df) < 2:
        return 0.0, 0.0, np.zeros(3, dtype=float)

    positions = np.vstack([extract_position(history_df.iloc[i]) for i in range(len(history_df))])
    times = _extract_times(history_df, assumed_fps=assumed_fps)

    if positions.shape[0] < 2 or times.shape[0] < 2:
        return 0.0, 0.0, np.zeros(3, dtype=float)

    dpos = np.diff(positions, axis=0)
    dt = np.diff(times)
    dt = np.where(dt > 1e-6, dt, 1e-3)

    velocities = dpos / dt[:, None]
    velocities = _moving_average_rows(velocities, smooth_window)

    speed_series = np.linalg.norm(velocities, axis=1)
    speed = float(speed_series[-1]) if speed_series.size else 0.0

    accel = 0.0
    if velocities.shape[0] >= 2:
        dv = velocities[-1] - velocities[-2]
        accel = float(np.linalg.norm(dv) / max(dt[-1], 1e-3))

    move_dir = _unit(velocities[-1]) if speed > 1e-6 else np.zeros(3, dtype=float)
    return speed, accel, move_dir


def confidence(
    speed: float,
    accel: float,
    move_dir: np.ndarray,
    gaze_dir: np.ndarray,
    s0: float,
    kappa_s: float,
    kappa_a: float,
    gamma: float,
) -> tuple[float, float, float]:
    z = float(kappa_s) * (float(speed) - float(s0)) + float(kappa_a) * float(accel)
    z = float(np.clip(z, -60.0, 60.0))
    c = 1.0 / (1.0 + float(np.exp(-z)))

    if np.linalg.norm(move_dir) <= 1e-9 or np.linalg.norm(gaze_dir) <= 1e-9:
        mismatch = 0.0
    else:
        dot = float(np.clip(np.dot(move_dir, gaze_dir), -1.0, 1.0))
        mismatch = (1.0 - dot) * 0.5

    c_eff = float(c * np.exp(-float(gamma) * mismatch))
    return c, mismatch, c_eff


def estimate_intention(
    pose_now,
    pose_history,
    s0: float,
    kappa_s: float,
    kappa_a: float,
    gamma: float,
    assumed_fps: float = 30.0,
    smooth_window: int = 3,
) -> IntentionEstimate:
    gaze_dir = compute_gaze_dir(pose_now)
    speed, accel, move_dir = compute_motion_estimates(
        pose_history,
        assumed_fps=assumed_fps,
        smooth_window=smooth_window,
    )

    c, mismatch, c_eff = confidence(
        speed=speed,
        accel=accel,
        move_dir=move_dir,
        gaze_dir=gaze_dir,
        s0=s0,
        kappa_s=kappa_s,
        kappa_a=kappa_a,
        gamma=gamma,
    )

    return IntentionEstimate(
        move_dir=move_dir,
        gaze_dir=gaze_dir,
        speed=speed,
        accel=accel,
        mismatch=mismatch,
        confidence=float(c),
        effective_confidence=float(c_eff),
    )
