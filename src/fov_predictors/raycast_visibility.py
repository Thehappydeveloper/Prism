from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

from .voxel_grid import VoxelGrid, VoxelIndex, point_to_aabb_distance


def uniform_tile_distribution(tiles) -> Dict[str, float]:
    if not tiles:
        return {}
    w = 1.0 / float(len(tiles))
    return {str(tile.unique_id): w for tile in tiles}


def sample_points_per_tile(
    points_xyz: np.ndarray,
    tiles,
    samples_per_tile: int,
    rng_seed: int | None,
) -> Dict[str, np.ndarray]:
    """Sample representative points inside each tile AABB."""
    pts = np.asarray(points_xyz[:, :3], dtype=float)
    rng = np.random.default_rng(rng_seed)
    m = max(1, int(samples_per_tile))

    samples: Dict[str, np.ndarray] = {}
    for tile in tiles:
        tile_id = str(tile.unique_id)
        min_c = np.asarray(tile.min_corner, dtype=float)
        max_c = np.asarray(tile.max_corner, dtype=float)

        in_tile = np.all((pts >= min_c) & (pts <= max_c), axis=1)
        tile_pts = pts[in_tile]

        if tile_pts.shape[0] == 0:
            center = (min_c + max_c) * 0.5
            samples[tile_id] = np.repeat(center[None, :], m, axis=0)
            continue

        if tile_pts.shape[0] >= m:
            idx = rng.choice(tile_pts.shape[0], size=m, replace=False)
            samples[tile_id] = tile_pts[idx]
        else:
            idx = rng.choice(tile_pts.shape[0], size=m, replace=True)
            samples[tile_id] = tile_pts[idx]

    return samples


def dda_voxel_traversal(
    start_world: Sequence[float],
    end_world: Sequence[float],
    voxel_grid: VoxelGrid,
) -> list[VoxelIndex]:
    """Amanatides-Woo style 3D voxel traversal from start to end."""
    start = np.asarray(start_world, dtype=float)
    end = np.asarray(end_world, dtype=float)

    start_idx = voxel_grid.world_to_voxel(start, clamp=True)
    end_idx = voxel_grid.world_to_voxel(end, clamp=True)

    current = np.array(start_idx, dtype=int)
    target = np.array(end_idx, dtype=int)

    voxels: list[VoxelIndex] = [tuple(current.tolist())]
    if np.all(current == target):
        return voxels

    direction = end - start
    step = np.sign(direction).astype(int)

    t_max = np.full(3, np.inf, dtype=float)
    t_delta = np.full(3, np.inf, dtype=float)

    for axis in range(3):
        if abs(direction[axis]) <= 1e-12:
            continue

        if step[axis] > 0:
            boundary = voxel_grid.min_corner[axis] + (current[axis] + 1) * voxel_grid.voxel_size
        else:
            boundary = voxel_grid.min_corner[axis] + current[axis] * voxel_grid.voxel_size

        t_max[axis] = (boundary - start[axis]) / direction[axis]
        t_delta[axis] = voxel_grid.voxel_size / abs(direction[axis])

    max_iters = int(np.sum(np.abs(target - current)) + 1024)
    for _ in range(max_iters):
        if np.all(current == target):
            break

        axis = int(np.argmin(t_max))
        current[axis] += step[axis]

        if not voxel_grid.in_bounds(current):
            break

        voxels.append(tuple(current.tolist()))
        t_max[axis] += t_delta[axis]

    return voxels


def ray_visible(
    start_voxel: VoxelIndex,
    target_point_world: Sequence[float],
    occupied_set: set[VoxelIndex],
    voxel_grid: VoxelGrid,
) -> bool:
    start_world = voxel_grid.voxel_to_world_center(start_voxel)
    traversed = dda_voxel_traversal(start_world, target_point_world, voxel_grid)
    if not traversed:
        return False

    # Ignore the start and terminal voxel to avoid self-occlusion at the target point.
    for vox in traversed[1:-1]:
        if vox in occupied_set:
            return False
    return True


def tile_visibility_from_voxel(
    voxel_idx: VoxelIndex,
    tile_samples: Mapping[str, np.ndarray],
    occupied_set: set[VoxelIndex],
    voxel_grid: VoxelGrid,
    max_rays_per_tile: int | None = None,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for tile_id, samples in tile_samples.items():
        arr = np.asarray(samples, dtype=float)
        if arr.size == 0:
            out[tile_id] = 0.0
            continue

        total = arr.shape[0] if max_rays_per_tile is None else min(arr.shape[0], int(max_rays_per_tile))
        if total <= 0:
            out[tile_id] = 0.0
            continue

        visible = 0
        for i in range(total):
            if ray_visible(voxel_idx, arr[i], occupied_set, voxel_grid):
                visible += 1

        out[tile_id] = float(visible) / float(total)
    return out


def aggregate_tile_importance(
    tiles,
    voxel_probabilities: Mapping[VoxelIndex, float],
    tile_samples: Mapping[str, np.ndarray],
    occupied_set: set[VoxelIndex],
    voxel_grid: VoxelGrid,
    user_position: Sequence[float],
    r_vis: float,
    sigma_g: float,
    max_rays_per_tile: int | None = None,
) -> Dict[str, float]:
    if not tiles:
        return {}

    sigma = max(float(sigma_g), 1e-6)
    user_pos = np.asarray(user_position, dtype=float)

    gate_by_tile: Dict[str, float] = {}
    for tile in tiles:
        tile_id = str(tile.unique_id)
        d_min = point_to_aabb_distance(user_pos, tile.min_corner, tile.max_corner)
        if d_min > float(r_vis):
            gate = 0.0
        else:
            gate = float(np.exp(-(d_min * d_min) / (2.0 * sigma * sigma)))
        gate_by_tile[tile_id] = gate

    mass = {str(tile.unique_id): 0.0 for tile in tiles}
    for vox, p_vox in voxel_probabilities.items():
        if p_vox <= 0:
            continue
        vis = tile_visibility_from_voxel(
            voxel_idx=vox,
            tile_samples=tile_samples,
            occupied_set=occupied_set,
            voxel_grid=voxel_grid,
            max_rays_per_tile=max_rays_per_tile,
        )
        for tile_id, v in vis.items():
            g = gate_by_tile.get(tile_id, 0.0)
            if g <= 0:
                continue
            mass[tile_id] += float(p_vox) * g * float(v)

    total = float(sum(mass.values()))
    if total <= 1e-12:
        return uniform_tile_distribution(tiles)

    return {tile_id: val / total for tile_id, val in mass.items()}
