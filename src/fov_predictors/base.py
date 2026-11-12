# fov_predictors/fov_predictors.py
from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple, List, Dict
from core.rotations import (
    DEFAULT_ANGLE_CONVENTION,
    forward_from_quaternion_xyzw,
    quaternion_from_euler_deg,
    quaternion_from_row,
    up_from_quaternion_xyzw,
)
from core.fov import Fov
from core.media_segment import TileComponentOptions as Tile
from pathlib import Path
from utils.ply_reader import PlyReader
from fov_predictors.latency import NullLatencySession


class FovPredictor(ABC):
    """
    Base class for FoV predictors.
    Each subclass implements `predict()`.
    """
    device = "cpu"
    DEFAULT_ANGLE_CONVENTION = DEFAULT_ANGLE_CONVENTION

    def __init__(
        self,
        ply_reader: PlyReader = PlyReader(),
        fov_h: float = np.deg2rad(97),
        fov_v: float = np.deg2rad(93),
        angle_convention: str = DEFAULT_ANGLE_CONVENTION,
    ):
        self.fov_h = fov_h
        self.fov_v = fov_v
        self.ply_reader = ply_reader
        self.angle_convention = angle_convention or self.DEFAULT_ANGLE_CONVENTION

    @classmethod
    def forward_from_euler_deg(cls, rx: float, ry: float, rz: float, angle_convention: str | None = None) -> np.ndarray:
        convention = (angle_convention or cls.DEFAULT_ANGLE_CONVENTION).strip().lower()
        if not convention.startswith("unity_"):
            convention = cls.DEFAULT_ANGLE_CONVENTION
        order = convention.split("_", 1)[1].upper()
        if len(order) != 3 or sorted(order) != ["X", "Y", "Z"]:
            order = cls.DEFAULT_ANGLE_CONVENTION.split("_", 1)[1].upper()
        values = {"X": float(rx), "Y": float(ry), "Z": float(rz)}
        rotation = R.from_euler(order, [values[axis] for axis in order], degrees=True)
        return rotation.apply([0, 0, 1])

    @classmethod
    def quaternion_from_euler_deg(cls, rx: float, ry: float, rz: float, angle_convention: str | None = None) -> np.ndarray:
        return quaternion_from_euler_deg(rx, ry, rz, angle_convention=angle_convention or cls.DEFAULT_ANGLE_CONVENTION)

    @classmethod
    def forward_from_quaternion_xyzw(cls, quaternion) -> np.ndarray:
        return forward_from_quaternion_xyzw(quaternion)

    @classmethod
    def up_from_quaternion_xyzw(cls, quaternion) -> np.ndarray:
        return up_from_quaternion_xyzw(quaternion)

    def _quaternion_from_frame(self, frame) -> np.ndarray:
        return quaternion_from_row(frame, angle_convention=self.angle_convention)

    def _forward_from_frame(self, frame) -> np.ndarray:
        return forward_from_quaternion_xyzw(self._quaternion_from_frame(frame))

    def _forward_from_euler_deg(self, rx: float, ry: float, rz: float) -> np.ndarray:
        return self.forward_from_euler_deg(rx, ry, rz, angle_convention=self.angle_convention)

    @abstractmethod
    def predict(
        self, tiles: List[Tile], frame_paths: List[Path], fov_data: Fov
    ) -> Dict[str, float]:
        """
        Predict per-tile FoV/importance weights for the current decision point.
        Returns {tile_id: score_in_[0,1]}.
        """
        pass

    def _latency_session(self, fov_data: Fov):
        recorder = getattr(fov_data, "latency_recorder", None)
        if recorder is None:
            return NullLatencySession()
        return recorder.start_prediction(self.__class__.__name__)

    def required_ply_frame_paths(self, frame_paths: List[Path]) -> List[Path]:
        """Return the subset of frame paths this predictor may read as PLY data."""
        return list(frame_paths or [])

    def predict_for_request(self, tiles, request, fov_data):
        """Predict using a request with separated geometry and observed FoV access."""
        return self.predict(tiles, request.geometry_frame_paths, fov_data)

    @staticmethod
    def _points_in_fov(
        points: np.ndarray,
        position: Optional[np.ndarray],
        forward: Optional[np.ndarray],
        euler_angles: Optional[Tuple[float, float, float]] = None,
        up: Optional[np.ndarray] = None,
        fov_h: float = np.deg2rad(97),
        fov_v: float = np.deg2rad(93),
        angle_convention: str = DEFAULT_ANGLE_CONVENTION,
        near_distance: Optional[float] = None,
        max_distance: Optional[float] = None,
    ) -> np.ndarray:
        """
        Vectorized perspective FoV check in Unity coordinates (Y-up, Z-forward).

        Returns a boolean mask of points inside the FoV.

        Important:
        The cone check must compare lateral projection relative to forward depth:

            x_ratio = x_proj / z_proj
            y_ratio = y_proj / z_proj

        not simply x_proj and y_proj directly. Otherwise the FoV becomes much
        wider than intended, especially near the side of the front hemisphere.
        """
        points = np.asarray(points)
        if points.size == 0:
            return np.zeros(0, dtype=bool)

        points_xyz = points[:, :3]
        n_points = points_xyz.shape[0]

        # Default position.
        if position is None:
            position = np.zeros(3, dtype=float)
        else:
            position = np.asarray(position, dtype=float)

        # Compute forward/up from Euler if forward was not provided.
        if forward is None:
            if euler_angles is None:
                raise ValueError("Either forward vector or euler_angles must be provided.")

            rx, ry, rz = euler_angles
            forward = FovPredictor.forward_from_euler_deg(
                rx,
                ry,
                rz,
                angle_convention=angle_convention,
            )

            if up is None:
                convention = (angle_convention or FovPredictor.DEFAULT_ANGLE_CONVENTION).strip().lower()
                if convention.startswith("unity_"):
                    order = convention.split("_", 1)[1].upper()
                else:
                    order = FovPredictor.DEFAULT_ANGLE_CONVENTION.split("_", 1)[1].upper()

                if len(order) != 3 or sorted(order) != ["X", "Y", "Z"]:
                    order = FovPredictor.DEFAULT_ANGLE_CONVENTION.split("_", 1)[1].upper()

                values = {"X": float(rx), "Y": float(ry), "Z": float(rz)}
                rotation = R.from_euler(order, [values[axis] for axis in order], degrees=True)
                up = rotation.apply([0, 1, 0])

        # Normalize forward.
        forward = np.asarray(forward, dtype=float)
        forward_norm = np.linalg.norm(forward)
        if forward_norm <= 1e-12:
            raise ValueError("Forward vector has near-zero length.")
        forward = forward / forward_norm

        # Normalize up, then re-orthogonalize it against forward.
        if up is None:
            up = np.array([0, 1, 0], dtype=float)
        else:
            up = np.asarray(up, dtype=float)

        up_norm = np.linalg.norm(up)
        if up_norm <= 1e-12:
            up = np.array([0, 1, 0], dtype=float)
            up_norm = 1.0
        up = up / up_norm

        # If up is almost parallel to forward, choose a fallback up.
        if abs(float(np.dot(up, forward))) > 0.999:
            fallback_up = np.array([0, 1, 0], dtype=float)
            if abs(float(np.dot(fallback_up, forward))) > 0.999:
                fallback_up = np.array([1, 0, 0], dtype=float)
            up = fallback_up

        # Unity-like right-handed local basis for projection:
        # forward = +Z, up = +Y, right = +X.
        #
        # np.cross(up, forward) gives +X for default Unity axes:
        # up [0,1,0] × forward [0,0,1] = right [1,0,0].
        right = np.cross(up, forward)
        right_norm = np.linalg.norm(right)
        if right_norm <= 1e-12:
            raise ValueError("Could not construct a valid right vector from forward/up.")
        right = right / right_norm

        # Recompute up so the basis is orthonormal.
        up = np.cross(forward, right)
        up_norm = np.linalg.norm(up)
        if up_norm <= 1e-12:
            raise ValueError("Could not construct a valid up vector from forward/right.")
        up = up / up_norm

        # Vectors from camera/user position to points.
        vecs = points_xyz - position
        distances = np.linalg.norm(vecs, axis=1)

        # Distance mask.
        mask_distance = distances > 1e-12
        if near_distance is not None:
            mask_distance &= distances >= float(near_distance)
        if max_distance is not None:
            mask_distance &= distances <= float(max_distance)

        if not np.any(mask_distance):
            return np.zeros(n_points, dtype=bool)

        valid_vecs = vecs[mask_distance]
        valid_distances = distances[mask_distance]

        # Normalize directions.
        vecs_norm = valid_vecs / valid_distances[:, None]

        # Project into local camera basis.
        x_proj = np.dot(vecs_norm, right)
        y_proj = np.dot(vecs_norm, up)
        z_proj = np.dot(vecs_norm, forward)

        # Only points in front of the camera.
        eps = 1e-9
        mask_front = z_proj > eps

        # Perspective FoV cone check.
        #
        # For a perspective camera:
        #   abs(x / z) <= tan(horizontal_fov / 2)
        #   abs(y / z) <= tan(vertical_fov / 2)
        #
        # For an elliptical cone, combine the two ratios:
        #   (x/z / h_limit)^2 + (y/z / v_limit)^2 <= 1
        h_limit = np.tan(float(fov_h) / 2.0)
        v_limit = np.tan(float(fov_v) / 2.0)

        x_ratio = np.zeros_like(x_proj)
        y_ratio = np.zeros_like(y_proj)
        x_ratio[mask_front] = x_proj[mask_front] / z_proj[mask_front]
        y_ratio[mask_front] = y_proj[mask_front] / z_proj[mask_front]

        mask_cone = (
            (x_ratio / h_limit) ** 2
            + (y_ratio / v_limit) ** 2
        ) <= 1.0

        # Combine masks back into original point indexing.
        final_mask = np.zeros(n_points, dtype=bool)
        valid_indices = np.where(mask_distance)[0]
        final_mask[valid_indices[mask_front & mask_cone]] = True

        return final_mask


    @staticmethod
    def tile_fov_proportion(tiles: list, mask: np.ndarray, points: np.ndarray) -> dict:
        """
        Compute the proportion of points inside the FoV for each tile.
        
        Args:
            tiles: List of TileComponentOptions objects.
            mask: Boolean numpy array of shape (N,) indicating which points are in the FoV.
            points: N x 3+ array of point coordinates (first 3 columns are XYZ).
        
        Returns:
            Dictionary mapping tile unique_id to proportion of points in the FoV.
        """
        points_np = np.asarray(points)
        if points_np.shape[1] > 3:
            points_np = points_np[:, :3]

        if not tiles:
            return {}

        mask_np = np.asarray(mask, dtype=bool)
        mins = np.asarray([tile.min_corner for tile in tiles], dtype=float)
        maxs = np.asarray([tile.max_corner for tile in tiles], dtype=float)
        visible_counts = np.zeros(len(tiles), dtype=np.int64)
        total_counts = np.zeros(len(tiles), dtype=np.int64)
        chunk_size = 200_000
        for start in range(0, points_np.shape[0], chunk_size):
            end = min(start + chunk_size, points_np.shape[0])
            chunk = points_np[start:end, :3]
            in_tiles = np.all((chunk[:, None, :] >= mins[None, :, :]) & (chunk[:, None, :] <= maxs[None, :, :]), axis=2)
            total_counts += np.sum(in_tiles, axis=0)
            visible = mask_np[start:end]
            if np.any(visible):
                visible_counts += np.sum(in_tiles & visible[:, None], axis=0)

        proportions = {}
        for idx, tile in enumerate(tiles):
            total_points = int(total_counts[idx])
            proportion = float(visible_counts[idx]) / total_points if total_points > 0 else 0.0
            proportions[tile.unique_id] = proportion

        return proportions
