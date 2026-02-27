from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Set, Tuple

import numpy as np

VoxelIndex = Tuple[int, int, int]


@dataclass(frozen=True)
class VoxelGrid:
    """Axis-aligned voxel grid over a world-space AABB."""

    min_corner: np.ndarray
    max_corner: np.ndarray
    voxel_size: float

    def __post_init__(self):
        if self.voxel_size <= 0:
            raise ValueError("voxel_size must be positive")
        span = np.maximum(self.max_corner - self.min_corner, 0.0)
        dims = np.floor(span / self.voxel_size).astype(int) + 1
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "max_index", dims - 1)

    def in_bounds(self, idx: Sequence[int]) -> bool:
        i = np.asarray(idx, dtype=int)
        return bool(np.all(i >= 0) and np.all(i < self.dims))

    def clamp_index(self, idx: Sequence[int]) -> VoxelIndex:
        i = np.asarray(idx, dtype=int)
        i = np.minimum(np.maximum(i, 0), self.max_index)
        return int(i[0]), int(i[1]), int(i[2])

    def world_to_voxel(self, point: Sequence[float], clamp: bool = False) -> VoxelIndex:
        p = np.asarray(point, dtype=float)
        idx = np.floor((p - self.min_corner) / self.voxel_size).astype(int)
        if clamp:
            return self.clamp_index(idx)
        return int(idx[0]), int(idx[1]), int(idx[2])

    def voxel_to_world_center(self, idx: Sequence[int]) -> np.ndarray:
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
    return list(_NEIGHBOR_OFFSETS_26)


def compute_expanded_aabb(points_xyz: np.ndarray, expansion: float) -> tuple[np.ndarray, np.ndarray]:
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
    if radius <= 0:
        return [voxel_grid.world_to_voxel(center_world, clamp=True)]

    center = np.asarray(center_world, dtype=float)
    radius_vec = np.array([radius, radius, radius], dtype=float)
    min_idx = voxel_grid.world_to_voxel(center - radius_vec, clamp=True)
    max_idx = voxel_grid.world_to_voxel(center + radius_vec, clamp=True)
    r2 = float(radius) * float(radius)

    voxels: List[VoxelIndex] = []
    for i in range(min_idx[0], max_idx[0] + 1):
        for j in range(min_idx[1], max_idx[1] + 1):
            for k in range(min_idx[2], max_idx[2] + 1):
                idx = (i, j, k)
                c = voxel_grid.voxel_to_world_center(idx)
                if float(np.sum((c - center) ** 2)) <= r2:
                    voxels.append(idx)
    return voxels


def build_occupancy_set(points_xyz: np.ndarray, voxel_grid: VoxelGrid) -> Set[VoxelIndex]:
    if points_xyz.size == 0:
        return set()

    pts = np.asarray(points_xyz[:, :3], dtype=float)
    idx = np.floor((pts - voxel_grid.min_corner) / voxel_grid.voxel_size).astype(int)
    valid = np.all((idx >= 0) & (idx < voxel_grid.dims), axis=1)
    idx_valid = idx[valid]
    if idx_valid.size == 0:
        return set()

    unique = np.unique(idx_valid, axis=0)
    return {(int(i), int(j), int(k)) for i, j, k in unique}


def find_nearest_free_voxel(
    start: VoxelIndex,
    free_voxels: Set[VoxelIndex],
    max_shell_radius: int = 6,
) -> VoxelIndex | None:
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
    p = np.asarray(point, dtype=float)
    min_c = np.asarray(min_corner, dtype=float)
    max_c = np.asarray(max_corner, dtype=float)
    clamped = np.minimum(np.maximum(p, min_c), max_c)
    return float(np.linalg.norm(p - clamped))
