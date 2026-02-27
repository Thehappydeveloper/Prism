# fov_predictors/fov_predictors.py
from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple, List, Dict
from core.fov import Fov
from core.media_segment import TileComponentOptions as Tile
from pathlib import Path
from utils.ply_reader import PlyReader


class FovPredictor(ABC):
    """
    Base class for FoV predictors.
    Each subclass implements `predict()`.
    """
    device = "cpu"

    def __init__(self, ply_reader:PlyReader = PlyReader(),fov_h: float = np.deg2rad(97), fov_v: float = np.deg2rad(93)):
        self.fov_h = fov_h
        self.fov_v = fov_v
        self.ply_reader = ply_reader

    @abstractmethod
    def predict(
        self, tiles: List[Tile], frame_paths: List[Path], fov_data: Fov
    ) -> Dict[str, float]:
        """
        Predict per-tile FoV/importance weights for the current decision point.
        Returns {tile_id: score_in_[0,1]}.
        """
        pass

    @staticmethod
    def _points_in_fov(
        points: np.ndarray,
        position: Optional[np.ndarray],
        forward: Optional[np.ndarray],
        euler_angles: Optional[Tuple[float,float,float]] = None,
        up: Optional[np.ndarray] = None,
        fov_h: float = np.deg2rad(97),
        fov_v: float = np.deg2rad(93),
        near_distance: Optional[float] = None,
        max_distance: Optional[float] = None
    ) -> np.ndarray:
        """
        Vectorized elliptical cone FoV check in Unity coordinates (Y-up, Z-forward).
        Returns boolean mask of visible points.
        """
        points = np.asarray(points)
        points_xyz = points[:, :3]
        N = points_xyz.shape[0]

        # Default position
        if position is None:
            position = np.zeros(3, dtype=float)
        else:
            position = np.asarray(position, dtype=float)

        # Compute forward/up from euler if not provided
        if forward is None:
            if euler_angles is not None:
                rx, ry, rz = euler_angles  # Unity X,Y,Z rotation in degrees
                # Use ZYX order to match Unity: Yaw (around Y), Pitch (X), Roll (Z)
                r = R.from_euler("ZYX", [rz, ry, rx], degrees=True)
                forward = r.apply([0, 0, 1])  # Unity forward
                if up is None:
                    up = r.apply([0, 1, 0])   # Unity up
            else:
                raise ValueError("Either forward vector or euler_angles must be provided.")

        forward = np.asarray(forward, dtype=float)
        forward /= np.linalg.norm(forward)

        if up is None:
            up = np.array([0, 1, 0], dtype=float)
        else:
            up = np.asarray(up, dtype=float)
        up /= np.linalg.norm(up)

        # Right vector in Unity axes
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)

        # Vector from position to points
        vecs = points_xyz - position
        distances = np.linalg.norm(vecs, axis=1)

        # Distance mask
        mask_distance = np.ones(N, dtype=bool)
        if near_distance is not None:
            mask_distance &= distances >= near_distance
        if max_distance is not None:
            mask_distance &= distances <= max_distance
        if not np.any(mask_distance):
            return np.zeros(N, dtype=bool)

        # Normalize vectors for angle check
        vecs_norm = vecs[mask_distance] / distances[mask_distance][:, None]
        x_proj = np.dot(vecs_norm, right)
        y_proj = np.dot(vecs_norm, up)
        z_proj = np.dot(vecs_norm, forward)

        # Only points in front
        mask_front = z_proj > 0

        # Elliptical cone check (FoV)
        h_limit = np.tan(fov_h / 2)
        v_limit = np.tan(fov_v / 2)
        mask_cone = (x_proj / h_limit) ** 2 + (y_proj / v_limit) ** 2 <= 1

        # Combine masks
        final_mask = np.zeros(N, dtype=bool)
        idx = np.where(mask_distance)[0]
        final_mask[idx[mask_front & mask_cone]] = True

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
        # Ensure numpy array and only XYZ
        points_np = np.asarray(points)
        if points_np.shape[1] > 3:
            points_np = points_np[:, :3]

        proportions = {}
        for tile in tiles:
            min_corner = np.array(tile.min_corner)
            max_corner = np.array(tile.max_corner)

            # Select points inside this tile's bounding box
            in_tile = np.all((points_np >= min_corner) & (points_np <= max_corner), axis=1)

            # Compute proportion in FoV
            points_in_fov = np.sum(mask[in_tile])
            total_points = np.sum(in_tile)
            proportion = points_in_fov / total_points if total_points > 0 else 0.0

            proportions[tile.unique_id] = proportion

        return proportions
