from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

DEFAULT_ANGLE_CONVENTION = "unity_zxy"
QUAT_COLUMNS = ("Quat_X", "Quat_Y", "Quat_Z", "Quat_W")
EULER_COLUMNS = ("HMDRX", "HMDRY", "HMDRZ")


def euler_order(angle_convention: str | None = None) -> str:
    convention = (angle_convention or DEFAULT_ANGLE_CONVENTION).strip().lower()
    if not convention.startswith("unity_"):
        convention = DEFAULT_ANGLE_CONVENTION
    order = convention.split("_", 1)[1].upper()
    if len(order) != 3 or sorted(order) != ["X", "Y", "Z"]:
        return DEFAULT_ANGLE_CONVENTION.split("_", 1)[1].upper()
    return order


def normalize_quaternion_xyzw(value: Any) -> np.ndarray:
    quat = np.asarray(value, dtype=float)
    if quat.shape != (4,):
        raise ValueError("Quaternion must contain x, y, z, w.")
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Quaternion norm must be finite and non-zero.")
    quat = quat / norm
    if quat[3] < 0:
        quat = -quat
    return quat


def quaternion_from_euler_deg(rx: float, ry: float, rz: float, angle_convention: str | None = None) -> np.ndarray:
    order = euler_order(angle_convention)
    values = {"X": float(rx), "Y": float(ry), "Z": float(rz)}
    quat = R.from_euler(order, [values[axis] for axis in order], degrees=True).as_quat()
    return normalize_quaternion_xyzw(quat)


def quaternion_from_row(row: Mapping[str, Any], angle_convention: str | None = None) -> np.ndarray:
    if all(name in row for name in QUAT_COLUMNS):
        try:
            values = [float(row[name]) for name in QUAT_COLUMNS]
            if all(np.isfinite(values)):
                return normalize_quaternion_xyzw(values)
        except (TypeError, ValueError):
            pass
    if all(name in row for name in EULER_COLUMNS):
        return quaternion_from_euler_deg(
            float(row["HMDRX"]),
            float(row["HMDRY"]),
            float(row["HMDRZ"]),
            angle_convention=angle_convention,
        )
    raise ValueError("FoV row must contain either quaternion columns or Euler rotation columns.")


def rotation_from_quaternion_xyzw(quaternion: Any) -> R:
    return R.from_quat(normalize_quaternion_xyzw(quaternion))


def forward_from_quaternion_xyzw(quaternion: Any) -> np.ndarray:
    return rotation_from_quaternion_xyzw(quaternion).apply([0, 0, 1])


def up_from_quaternion_xyzw(quaternion: Any) -> np.ndarray:
    return rotation_from_quaternion_xyzw(quaternion).apply([0, 1, 0])


def ensure_fov_quaternions(df: pd.DataFrame, angle_convention: str | None = None) -> pd.DataFrame:
    out = df.copy()
    has_quat = all(name in out.columns for name in QUAT_COLUMNS)
    has_euler = all(name in out.columns for name in EULER_COLUMNS)
    if not has_quat and not has_euler:
        return out
    quats = []
    for _, row in out.iterrows():
        quats.append(quaternion_from_row(row, angle_convention=angle_convention))
    quat_array = np.asarray(quats, dtype=float)
    for idx, column in enumerate(QUAT_COLUMNS):
        out[column] = quat_array[:, idx]
    return out

