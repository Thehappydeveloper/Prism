"""Voxel-grid primitives shared by the reachability raycast modules.

This module defines the grid coordinate system used across the package and a
small set of geometry helpers for voxelization, neighborhood traversal, and
distance queries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np

type VoxelIndex = Tuple[int, int, int]


@dataclass(frozen=True)
class VoxelGrid:
    """Axis-aligned voxel grid over a world-space AABB."""

    min_corner: np.ndarray
    max_corner: np.ndarray
    voxel_size: float

    def __post_init__(self):
        """Precompute grid dimensions and valid index bounds."""
        if self.voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        span = np.maximum(self.max_corner - self.min_corner, 0.0)
        dims = np.floor(span / self.voxel_size).astype(int) + 1
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "max_index", dims - 1)

    def in_bounds(self, idx: Sequence[int]) -> bool:
        """Return whether a voxel index lies inside the grid bounds."""
        i = np.asarray(idx, dtype=int)
        return bool(np.all(i >= 0) and np.all(i < self.dims))

    def clamp_index(self, idx: Sequence[int]) -> VoxelIndex:
        """Clamp an index to the nearest valid voxel in the grid."""
        i = np.asarray(idx, dtype=int)
        i = np.minimum(np.maximum(i, 0), self.max_index)
        return int(i[0]), int(i[1]), int(i[2])

    def world_to_voxel(self, point: Sequence[float], clamp: bool = False) -> VoxelIndex:
        """Convert a world-space position into integer voxel coordinates."""
        p = np.asarray(point, dtype=float)
        idx = np.floor((p - self.min_corner) / self.voxel_size).astype(int)
        if clamp:
            return self.clamp_index(idx)
        return int(idx[0]), int(idx[1]), int(idx[2])

    def voxel_to_world_center(self, idx: Sequence[int]) -> np.ndarray:
        """Return the world-space center point of a voxel index."""
        i = np.asarray(idx, dtype=float)
        return self.min_corner + (i + 0.5) * self.voxel_size


_NEIGHBOR_OFFSETS_26: List[VoxelIndex] = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
]


def neighbor_offsets_26() -> List[VoxelIndex]:
    """Return the 26 offsets around a voxel in a cubic neighborhood."""
    return list(_NEIGHBOR_OFFSETS_26)


def compute_expanded_aabb(points_xyz: np.ndarray, expansion: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute a point-cloud bounding box padded by the requested margin."""
    if points_xyz.size == 0:
        raise ValueError("Point cloud is empty")
    mins = np.min(points_xyz[:, :3], axis=0).astype(float)
    maxs = np.max(points_xyz[:, :3], axis=0).astype(float)
    return mins - float(expansion), maxs + float(expansion)


def enumerate_voxels_in_sphere(
    voxel_grid: VoxelGrid,
    center_world: Sequence[float],
    radius: float,
) -> List[VoxelIndex]:
    """List voxels whose centers fall inside a world-space sphere."""
    if radius <= 0:
        return [voxel_grid.world_to_voxel(center_world, clamp=True)]

    center = np.asarray(center_world, dtype=float)
    radius_vec = np.array([radius, radius, radius], dtype=float)
    min_idx = voxel_grid.world_to_voxel(center - radius_vec, clamp=True)
    max_idx = voxel_grid.world_to_voxel(center + radius_vec, clamp=True)
    min_corner = voxel_grid.min_corner
    voxel_size = float(voxel_grid.voxel_size)
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    r2 = float(radius) * float(radius)

    voxels: List[VoxelIndex] = []
    for i in range(min_idx[0], max_idx[0] + 1):
        wx = float(min_corner[0]) + (float(i) + 0.5) * voxel_size
        dx2 = (wx - cx) * (wx - cx)
        for j in range(min_idx[1], max_idx[1] + 1):
            wy = float(min_corner[1]) + (float(j) + 0.5) * voxel_size
            dxy2 = dx2 + ((wy - cy) * (wy - cy))
            if dxy2 > r2:
                continue
            for k in range(min_idx[2], max_idx[2] + 1):
                wz = float(min_corner[2]) + (float(k) + 0.5) * voxel_size
                if dxy2 + ((wz - cz) * (wz - cz)) <= r2:
                    voxels.append((i, j, k))
    return voxels


def build_occupancy_set(points_xyz: np.ndarray, voxel_grid: VoxelGrid) -> Set[VoxelIndex]:
    """Map point-cloud samples onto the set of occupied voxels.

    Optimized approach:
    - Vectorize world->voxel index conversion for all points at once.
    - Filter out-of-bounds indices via boolean mask.
    - Convert 3D indices to a single flat index and call np.unique on that 1D array
      (faster than np.unique on 2D rows or repeated Python set.add).
    - Reconstruct only the unique voxel tuples for the final set.
    """
    if points_xyz.size == 0:
        return set()

    # Work with first three coordinates and ensure float computation
    pts = np.asarray(points_xyz[:, :3], dtype=float, copy=False)

    # Compute fractional voxel indices and floor to integer voxel indices.
    rel = (pts - voxel_grid.min_corner) / float(voxel_grid.voxel_size)
    idx = np.floor(rel).astype(np.int64, copy=False)  # shape (N,3)

    # Bounds check using precomputed dims / max_index
    valid_mask = np.all((idx >= 0) & (idx <= voxel_grid.max_index), axis=1)
    idx_valid = idx[valid_mask]
    if idx_valid.size == 0:
        return set()

    # Convert 3D index to 1D flat index for fast uniqueness
    # flat = i * (Y*Z) + j * Z + k
    dims = np.asarray(voxel_grid.dims, dtype=np.int64)
    stride0 = int(dims[1]) * int(dims[2])
    stride1 = int(dims[2])
    flat = idx_valid[:, 0] * stride0 + idx_valid[:, 1] * stride1 + idx_valid[:, 2]

    unique_flat = np.unique(flat)
    if unique_flat.size == 0:
        return set()

    # Reconstruct 3D indices from flat unique indices
    i = (unique_flat // stride0).astype(int)
    rem = unique_flat % stride0
    j = (rem // stride1).astype(int)
    k = (rem % stride1).astype(int)

    # Build and return set of tuples (iterate only over unique voxels)
    return {(int(a), int(b), int(c)) for a, b, c in zip(i, j, k)}


def find_nearest_free_voxel(
    start: VoxelIndex,
    free_voxels: Set[VoxelIndex],
    max_shell_radius: int = 6,
) -> VoxelIndex | None:
    """Find the closest free voxel to a blocked start location by shell search."""
    if start in free_voxels:
        return start

    sx, sy, sz = start
    for r in range(1, max_shell_radius + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    if max(abs(dx), abs(dy), abs(dz)) != r:
                        continue
                    cand = (sx + dx, sy + dy, sz + dz)
                    if cand in free_voxels:
                        return cand
    return None


def point_to_aabb_distance(point: Sequence[float], min_corner: Sequence[float], max_corner: Sequence[float]) -> float:
    """Compute Euclidean distance from a point to an axis-aligned box."""
    p = np.asarray(point, dtype=float)
    min_c = np.asarray(min_corner, dtype=float)
    max_c = np.asarray(max_corner, dtype=float)
    clamped = np.minimum(np.maximum(p, min_c), max_c)
    return float(np.linalg.norm(p - clamped))
