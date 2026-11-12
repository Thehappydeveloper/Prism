"""Visibility and aggregation helpers for the reachability raycast pipeline.

These functions take probable user voxels and convert them into tile-level
importance by sampling points inside tile bounds, tracing rays through the
voxel grid, and aggregating visibility with distance-based gating.
"""

from __future__ import annotations

import multiprocessing as mp
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np

from core.compute_processes import force_stop_requested, register_compute_pool, unregister_compute_pool

from .voxel_grid import VoxelGrid, VoxelIndex, point_to_aabb_distance


@dataclass(frozen=True)
class WeightedVoxelSample:
    """A raycast origin with its contribution weight in the full voxel expectation."""
    voxel: VoxelIndex
    probability: float
    world_center: np.ndarray
    estimator_weight: float
    stratum: str


_RAYCAST_WORKER_CONTEXT: dict[str, Any] | None = None


def _initialize_raycast_worker(context: dict[str, Any]) -> None:
    """Install one explicitly serialized context in a spawned raycast worker.

    Raycast evaluation is invoked from the web application's threaded batch
    runner.  A ``fork`` child would inherit that runner's threads, locks, and
    any active process-pool state, which can leave a child blocked forever.
    ``spawn`` starts a clean interpreter, so the worker must receive its
    immutable evaluation context explicitly rather than inheriting this module
    global from the parent process.
    """
    global _RAYCAST_WORKER_CONTEXT
    _RAYCAST_WORKER_CONTEXT = context


def uniform_tile_distribution(tiles) -> Dict[str, float]:
    """Return an all-visible independent score for every tile."""
    if not tiles:
        return {}
    return {str(tile.unique_id): 1.0 for tile in tiles}


def collect_points_per_tile(points_xyz: np.ndarray, tiles) -> Dict[str, np.ndarray]:
    """Collect all point-cloud samples that fall inside each tile AABB."""
    pts = np.asarray(points_xyz[:, :3], dtype=float)
    tile_list = list(tiles)
    if not tile_list:
        return {}

    tile_ids = [str(tile.unique_id) for tile in tile_list]
    mins = np.asarray([tile.min_corner for tile in tile_list], dtype=float)
    maxs = np.asarray([tile.max_corner for tile in tile_list], dtype=float)
    collected: Dict[str, list[np.ndarray]] = {tile_id: [] for tile_id in tile_ids}

    # Chunk to avoid building a huge (points x tiles x xyz) boolean array for
    # dense V-PCC frames while still scanning the point cloud only once.
    max_bool_cells = 48_000_000
    chunk_size = max(1, min(len(pts), max_bool_cells // max(1, len(tile_list) * 3)))
    for start in range(0, len(pts), chunk_size):
        chunk = pts[start:start + chunk_size]
        if chunk.size == 0:
            continue
        in_tile = np.all((chunk[:, None, :] >= mins[None, :, :]) & (chunk[:, None, :] <= maxs[None, :, :]), axis=2)
        point_idx, tile_idx = np.nonzero(in_tile)
        if point_idx.size == 0:
            continue
        for local_tile_idx in np.unique(tile_idx):
            mask = tile_idx == local_tile_idx
            collected[tile_ids[int(local_tile_idx)]].append(chunk[point_idx[mask]])

    out: Dict[str, np.ndarray] = {}
    for tile_id in tile_ids:
        parts = collected[tile_id]
        out[tile_id] = np.vstack(parts) if parts else np.empty((0, 3), dtype=float)
    return out


def sample_points_per_tile(
    points_xyz: np.ndarray,
    tiles,
    samples_per_tile: int | Mapping[str, int],
    rng_seed: int | None,
) -> Dict[str, np.ndarray]:
    """Choose representative point-cloud samples for each tile's AABB."""
    tile_points = collect_points_per_tile(points_xyz=points_xyz, tiles=tiles)
    return sample_precollected_points_per_tile(
        tile_point_sets=tile_points,
        tiles=tiles,
        samples_per_tile=samples_per_tile,
        rng_seed=rng_seed,
    )


def sample_precollected_points_per_tile(
    tile_point_sets: Mapping[str, np.ndarray],
    tiles,
    samples_per_tile: int | Mapping[str, int],
    rng_seed: int | None,
) -> Dict[str, np.ndarray]:
    """Sample representative points once tile point support has already been collected."""
    rng = np.random.default_rng(rng_seed)

    samples: Dict[str, np.ndarray] = {}
    for tile in tiles:
        tile_id = str(tile.unique_id)
        if isinstance(samples_per_tile, Mapping):
            m = max(0, int(samples_per_tile.get(tile_id, 0)))
        else:
            m = max(1, int(samples_per_tile))
        tile_pts = tile_point_sets.get(tile_id)
        if tile_pts is None:
            tile_pts = np.empty((0, 3), dtype=float)

        if m <= 0:
            samples[tile_id] = np.empty((0, 3), dtype=float)
            continue

        if tile_pts.shape[0] == 0:
            samples[tile_id] = np.empty((0, 3), dtype=float)
            continue

        if tile_pts.shape[0] >= m:
            idx = rng.choice(tile_pts.shape[0], size=m, replace=False)
            samples[tile_id] = tile_pts[idx]
        else:
            idx = rng.choice(tile_pts.shape[0], size=m, replace=True)
            samples[tile_id] = tile_pts[idx]

    return samples


def compute_tile_gates(
    tiles,
    user_position: Sequence[float],
    r_vis: float,
    sigma_g: float,
) -> Dict[str, float]:
    """Compute the distance gate applied to each tile before visibility aggregation."""
    sigma = max(float(sigma_g), 1e-6)
    user_pos = np.asarray(user_position, dtype=float)
    r_vis_f = float(r_vis)

    gate_by_tile: Dict[str, float] = {}
    for tile in tiles:
        tile_id = str(tile.unique_id)
        d_min = point_to_aabb_distance(user_pos, tile.min_corner, tile.max_corner)
        if d_min > r_vis_f:
            gate = 0.0
        else:
            gate = float(np.exp(-(d_min * d_min) / (2.0 * sigma * sigma)))
        gate_by_tile[tile_id] = gate
    return gate_by_tile


def summarize_tile_gates(
    tiles,
    user_position: Sequence[float],
    r_vis: float,
    sigma_g: float,
) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float], list[dict]]:
    """Return tile gates plus distance/gate diagnostics for predictor reporting."""
    sigma = max(float(sigma_g), 1e-6)
    user_pos = np.asarray(user_position, dtype=float)
    r_vis_f = float(r_vis)

    gate_by_tile: Dict[str, float] = {}
    distance_by_tile: Dict[str, float] = {}
    gate_stats: Dict[str, float] = {
        "active_tiles": 0.0,
        "gated_out_tiles": 0.0,
        "nearest_tile_distance": 0.0,
        "farthest_tile_distance": 0.0,
        "max_tile_gate": 0.0,
        "mean_tile_gate": 0.0,
    }
    details: list[dict] = []

    distances = []
    gates = []
    for tile in tiles:
        tile_id = str(tile.unique_id)
        d_min = float(point_to_aabb_distance(user_pos, tile.min_corner, tile.max_corner))
        if d_min > r_vis_f:
            gate = 0.0
        else:
            gate = float(np.exp(-(d_min * d_min) / (2.0 * sigma * sigma)))
        gate_by_tile[tile_id] = gate
        distance_by_tile[tile_id] = d_min
        distances.append(d_min)
        gates.append(gate)
        details.append(
            {
                "tile_id": tile_id,
                "distance": round(d_min, 6),
                "gate": round(gate, 6),
                "within_r_vis": bool(d_min <= r_vis_f),
            }
        )

    if distances:
        gate_stats = {
            "active_tiles": float(sum(1 for g in gates if g > 0.0)),
            "gated_out_tiles": float(sum(1 for g in gates if g <= 0.0)),
            "nearest_tile_distance": float(min(distances)),
            "farthest_tile_distance": float(max(distances)),
            "max_tile_gate": float(max(gates)),
            "mean_tile_gate": float(sum(gates) / len(gates)),
        }
    details.sort(key=lambda item: (item["distance"], item["tile_id"]))
    return gate_by_tile, distance_by_tile, gate_stats, details


def allocate_samples_per_tile(
    tile_point_sets: Mapping[str, np.ndarray],
    tile_weights: Mapping[str, float],
    kept_voxel_count: int,
    ray_budget: int,
    target_confidence: float,
    support_points: int,
    min_samples: int,
    max_samples: int,
) -> tuple[Dict[str, int], Dict[str, float]]:
    """Allocate per-tile sample counts under a global ray budget."""
    counts: Dict[str, int] = {}
    stats: Dict[str, float] = {}
    if kept_voxel_count <= 0 or ray_budget <= 0 or not tile_point_sets:
        return counts, {
            "auto_sample_allocated_tiles": 0.0,
            "auto_sample_total": 0.0,
            "auto_sample_confidence_mean": 0.0,
            "auto_sample_min": 0.0,
            "auto_sample_max": 0.0,
        }

    delta = float(np.clip(1.0 - float(target_confidence), 1e-6, 1.0 - 1e-6))
    min_s = max(0, int(min_samples))
    max_s = max(min_s, int(max_samples))
    support_pts = max(1, int(support_points))
    per_sample_cost = max(1, int(kept_voxel_count))

    entries = []
    for tile_id, pts in tile_point_sets.items():
        n_points = int(np.asarray(pts).shape[0])
        weight = max(float(tile_weights.get(tile_id, 0.0)), 0.0)
        if n_points <= 0:
            p_t = 1.0
            n_req = 1
        else:
            p_t = float(np.clip(min(1.0, float(support_pts) / float(n_points)), 1e-6, 1.0))
            if p_t >= 1.0:
                n_req = 1
            else:
                n_req = int(math.ceil(math.log(delta) / math.log(1.0 - p_t)))
        n_req = max(1, min(n_req, max_s))
        entries.append(
            {
                "tile_id": tile_id,
                "weight": weight,
                "p_t": p_t,
                "n_req": n_req,
                "n_alloc": min(min_s, n_req),
            }
        )

    rays_used = per_sample_cost * sum(int(entry["n_alloc"]) for entry in entries)
    if rays_used > ray_budget:
        entries.sort(key=lambda entry: float(entry["weight"]), reverse=True)
        rays_used = 0
        for entry in entries:
            grant = min(
                int(entry["n_req"]),
                max(0, (ray_budget - rays_used) // per_sample_cost),
            )
            entry["n_alloc"] = grant
            rays_used += grant * per_sample_cost
    else:
        remaining = max(0, int(ray_budget - rays_used))
        while remaining >= per_sample_cost:
            best_entry = None
            best_score = -1.0
            for entry in entries:
                n_alloc = int(entry["n_alloc"])
                n_req = int(entry["n_req"])
                if n_alloc >= n_req:
                    continue
                score = float(entry["weight"]) * ((1.0 - float(entry["p_t"])) ** n_alloc) * float(entry["p_t"])
                if score > best_score:
                    best_score = score
                    best_entry = entry
            if best_entry is None or best_score <= 0.0:
                break
            best_entry["n_alloc"] = int(best_entry["n_alloc"]) + 1
            remaining -= per_sample_cost

    confidences = []
    for entry in entries:
        tile_id = str(entry["tile_id"])
        n_alloc = int(entry["n_alloc"])
        counts[tile_id] = n_alloc
        confidence = 1.0 - ((1.0 - float(entry["p_t"])) ** n_alloc) if n_alloc > 0 else 0.0
        confidences.append(confidence)

    allocated = [int(entry["n_alloc"]) for entry in entries] if entries else [0]
    stats.update(
        {
            "auto_sample_allocated_tiles": float(len(entries)),
            "auto_sample_total": float(sum(allocated)),
            "auto_sample_confidence_mean": float(np.mean(confidences)) if confidences else 0.0,
            "auto_sample_min": float(min(allocated)) if allocated else 0.0,
            "auto_sample_max": float(max(allocated)) if allocated else 0.0,
        }
    )
    return counts, stats


def dda_voxel_traversal(
    start_world: Sequence[float],
    end_world: Sequence[float],
    voxel_grid: VoxelGrid,
) -> list[VoxelIndex]:
    """Enumerate voxels intersected by a ray segment in grid coordinates."""
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


def _world_points_to_voxels(points_world: np.ndarray, voxel_grid: VoxelGrid) -> np.ndarray:
    """Map many world-space points to voxel indices with one vectorized transform."""
    if points_world.size == 0:
        return np.empty((0, 3), dtype=np.int64)

    pts = np.asarray(points_world, dtype=float)
    idx = np.floor((pts - voxel_grid.min_corner) / voxel_grid.voxel_size).astype(np.int64, copy=False)
    return np.minimum(np.maximum(idx, 0), voxel_grid.max_index)


def select_candidate_voxels_probability_diverse(
    voxel_items: Sequence[tuple[VoxelIndex, float, np.ndarray]],
    max_candidate_voxels: int | None,
    voxel_selection_core_fraction: float = 0.7,
) -> list[tuple[VoxelIndex, float, np.ndarray]]:
    """Select target-independent ray origins with high-probability and spatial support."""
    items = [(vox, float(prob), np.asarray(center, dtype=float)) for vox, prob, center in voxel_items if float(prob) > 0.0]
    if not items:
        return []
    sorted_items = sorted(items, key=lambda item: item[1], reverse=True)
    if max_candidate_voxels is None or len(sorted_items) <= max(1, int(max_candidate_voxels)):
        return sorted_items

    cap = max(1, int(max_candidate_voxels))
    core_fraction = float(np.clip(float(voxel_selection_core_fraction), 0.0, 1.0))
    k_core = min(cap, int(round(float(cap) * core_fraction)))
    selected = list(sorted_items[:k_core])
    remaining = list(sorted_items[k_core:])

    if not selected:
        selected.append(sorted_items[0])
        remaining = sorted_items[1:]

    while remaining and len(selected) < cap:
        selected_centers = np.asarray([center for _, _, center in selected], dtype=float)
        best_idx = 0
        best_score = -1.0
        for idx, (_vox, prob, center) in enumerate(remaining):
            dists = np.linalg.norm(selected_centers - center, axis=1)
            nearest_dist = float(np.min(dists)) if dists.size else 0.0
            score = float(prob) * nearest_dist
            if score > best_score:
                best_score = score
                best_idx = idx
        selected.append(remaining.pop(best_idx))

    return selected


def select_candidate_voxels(
    voxel_items: Sequence[tuple[VoxelIndex, float, np.ndarray]],
    max_candidate_voxels: int | None,
    voxel_selection_mode: str = "probability_diverse",
    voxel_selection_core_fraction: float = 0.7,
) -> list[tuple[VoxelIndex, float, np.ndarray]]:
    """Select a global target-independent set of raycast origins."""
    items = [(vox, float(prob), np.asarray(center, dtype=float)) for vox, prob, center in voxel_items if float(prob) > 0.0]
    sorted_items = sorted(items, key=lambda item: item[1], reverse=True)
    if not sorted_items:
        return []
    if max_candidate_voxels is None:
        return sorted_items

    cap = max(1, int(max_candidate_voxels))
    mode = str(voxel_selection_mode or "probability_diverse")
    if mode == "probability_topk":
        return sorted_items[:cap]
    if mode == "probability_random":
        if len(sorted_items) <= cap:
            return sorted_items
        probs = np.asarray([prob for _, prob, _ in sorted_items], dtype=float)
        total = float(np.sum(probs))
        if total <= 1e-12:
            return sorted_items[:cap]
        rng = np.random.default_rng(0)
        idx = rng.choice(len(sorted_items), size=cap, replace=False, p=probs / total)
        return [sorted_items[int(i)] for i in idx]
    if mode == "probability_diverse":
        return select_candidate_voxels_probability_diverse(
            voxel_items=sorted_items,
            max_candidate_voxels=cap,
            voxel_selection_core_fraction=voxel_selection_core_fraction,
        )
    return sorted_items[:cap]


def select_weighted_candidate_voxels(
    voxel_items: Sequence[tuple[VoxelIndex, float, np.ndarray]],
    max_candidate_voxels: int | None,
    voxel_selection_mode: str = "probability_diverse",
    voxel_selection_core_fraction: float = 0.7,
) -> tuple[list[WeightedVoxelSample], Dict[str, float]]:
    """Select raycast origins as a weighted estimator over the full voxel distribution."""
    items = [
        (vox, float(prob), np.asarray(center, dtype=float))
        for vox, prob, center in voxel_items
        if float(prob) > 0.0
    ]
    sorted_items = sorted(items, key=lambda item: item[1], reverse=True)
    if not sorted_items:
        return [], {
            "candidate_voxel_estimator_mode": "empty",
            "candidate_voxel_core_count": 0.0,
            "candidate_voxel_core_mass": 0.0,
            "candidate_voxel_tail_count": 0.0,
            "candidate_voxel_tail_mass": 0.0,
            "candidate_voxel_estimator_weight_sum": 0.0,
        }

    mode = str(voxel_selection_mode or "probability_diverse")
    if max_candidate_voxels is None or len(sorted_items) <= max(1, int(max_candidate_voxels)):
        samples = [
            WeightedVoxelSample(
                voxel=vox,
                probability=prob,
                world_center=center,
                estimator_weight=prob,
                stratum="exact",
            )
            for vox, prob, center in sorted_items
        ]
        total_mass = float(sum(sample.estimator_weight for sample in samples))
        return samples, {
            "candidate_voxel_estimator_mode": "exact_all",
            "candidate_voxel_core_count": float(len(samples)),
            "candidate_voxel_core_mass": total_mass,
            "candidate_voxel_tail_count": 0.0,
            "candidate_voxel_tail_mass": 0.0,
            "candidate_voxel_estimator_weight_sum": total_mass,
        }

    cap = max(1, int(max_candidate_voxels))
    if mode in {"probability_topk", "tile_distance_biased_legacy"}:
        selected = sorted_items[:cap]
        samples = [
            WeightedVoxelSample(
                voxel=vox,
                probability=prob,
                world_center=center,
                estimator_weight=prob,
                stratum="truncated",
            )
            for vox, prob, center in selected
        ]
        selected_mass = float(sum(sample.probability for sample in samples))
        return samples, {
            "candidate_voxel_estimator_mode": "truncated",
            "candidate_voxel_core_count": float(len(samples)),
            "candidate_voxel_core_mass": selected_mass,
            "candidate_voxel_tail_count": 0.0,
            "candidate_voxel_tail_mass": max(0.0, float(sum(prob for _, prob, _ in sorted_items)) - selected_mass),
            "candidate_voxel_estimator_weight_sum": selected_mass,
        }

    core_fraction = float(np.clip(float(voxel_selection_core_fraction), 0.0, 1.0))
    core_count = min(cap, max(0, int(round(float(cap) * core_fraction))))
    if cap > 1 and core_count >= cap:
        core_count = cap - 1
    if core_count <= 0 and cap > 1:
        core_count = 1

    core_items = sorted_items[:core_count]
    tail_items = sorted_items[core_count:]
    tail_budget = max(0, cap - len(core_items))

    samples_by_voxel: Dict[VoxelIndex, WeightedVoxelSample] = {}
    for vox, prob, center in core_items:
        samples_by_voxel[vox] = WeightedVoxelSample(
            voxel=vox,
            probability=prob,
            world_center=center,
            estimator_weight=prob,
            stratum="core",
        )

    tail_mass = float(sum(prob for _, prob, _ in tail_items))
    tail_selected_count = 0
    if tail_items and tail_budget > 0 and tail_mass > 1e-12:
        cumulative = np.cumsum(np.asarray([prob for _, prob, _ in tail_items], dtype=float))
        # Deterministic probability-mass strata. Each stratum represents the
        # same tail probability mass, so selected tail origins estimate the
        # omitted distribution without normalizing the core.
        targets = ((np.arange(tail_budget, dtype=float) + 0.5) / float(tail_budget)) * tail_mass
        per_sample_weight = tail_mass / float(tail_budget)
        for target in targets:
            idx = int(np.searchsorted(cumulative, float(target), side="left"))
            idx = min(max(idx, 0), len(tail_items) - 1)
            vox, prob, center = tail_items[idx]
            existing = samples_by_voxel.get(vox)
            if existing is None:
                samples_by_voxel[vox] = WeightedVoxelSample(
                    voxel=vox,
                    probability=prob,
                    world_center=center,
                    estimator_weight=per_sample_weight,
                    stratum="tail",
                )
                tail_selected_count += 1
            else:
                samples_by_voxel[vox] = WeightedVoxelSample(
                    voxel=existing.voxel,
                    probability=existing.probability,
                    world_center=existing.world_center,
                    estimator_weight=existing.estimator_weight + per_sample_weight,
                    stratum=existing.stratum,
                )

    samples = sorted(
        samples_by_voxel.values(),
        key=lambda sample: (0 if sample.stratum == "core" else 1, -float(sample.estimator_weight)),
    )
    core_mass = float(sum(prob for _, prob, _ in core_items))
    weight_sum = float(sum(sample.estimator_weight for sample in samples))
    return samples, {
        "candidate_voxel_estimator_mode": "core_tail_stratified",
        "candidate_voxel_core_count": float(len(core_items)),
        "candidate_voxel_core_mass": core_mass,
        "candidate_voxel_tail_count": float(tail_selected_count),
        "candidate_voxel_tail_mass": tail_mass,
        "candidate_voxel_estimator_weight_sum": weight_sum,
    }


def _tile_bounds_by_id(tiles) -> Dict[str, tuple[np.ndarray, np.ndarray]]:
    bounds: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tile in tiles:
        bounds[str(tile.unique_id)] = (
            np.asarray(tile.min_corner, dtype=float),
            np.asarray(tile.max_corner, dtype=float),
        )
    return bounds


def _voxel_intersects_bounds(
    voxel: Sequence[int],
    voxel_grid: VoxelGrid,
    bounds: tuple[np.ndarray, np.ndarray],
) -> bool:
    voxel_idx = np.asarray(voxel, dtype=float)
    voxel_min = np.asarray(voxel_grid.min_corner, dtype=float) + voxel_idx * float(voxel_grid.voxel_size)
    voxel_max = voxel_min + float(voxel_grid.voxel_size)
    tile_min, tile_max = bounds
    eps = 1e-9
    return bool(np.all(voxel_max > tile_min + eps) and np.all(voxel_min < tile_max - eps))


def _voxel_center_or_none(voxel: Sequence[int], voxel_grid: VoxelGrid) -> list[float] | None:
    dims = voxel_grid.dims
    if any(int(voxel[idx]) < 0 or int(voxel[idx]) >= int(dims[idx]) for idx in range(3)):
        return None
    center = voxel_grid.voxel_to_world_center((int(voxel[0]), int(voxel[1]), int(voxel[2])))
    return [float(v) for v in center]


def _occupied_at(
    voxel: VoxelIndex,
    occupied_set: set[VoxelIndex],
    occupied_lookup: np.ndarray | None,
    dims: tuple[int, int, int],
    stride0: int,
    stride1: int,
) -> bool:
    x, y, z = int(voxel[0]), int(voxel[1]), int(voxel[2])
    if x < 0 or y < 0 or z < 0 or x >= dims[0] or y >= dims[1] or z >= dims[2]:
        return False
    if occupied_lookup is None:
        return (x, y, z) in occupied_set
    return bool(occupied_lookup[(x * stride0) + (y * stride1) + z])


def _step_axis_indices(t_max_x: float, t_max_y: float, t_max_z: float) -> list[int]:
    min_t = min(t_max_x, t_max_y, t_max_z)
    eps = max(1e-12, abs(min_t) * 1e-12)
    axes = []
    if abs(t_max_x - min_t) <= eps:
        axes.append(0)
    if abs(t_max_y - min_t) <= eps:
        axes.append(1)
    if abs(t_max_z - min_t) <= eps:
        axes.append(2)
    return axes


def _conservative_step_candidates(
    current: VoxelIndex,
    axes: Sequence[int],
    steps: tuple[int, int, int],
) -> list[VoxelIndex]:
    """Return all voxels touched when stepping across one or more tied axes."""
    if not axes:
        return []
    candidates: list[VoxelIndex] = []
    seen: set[VoxelIndex] = set()
    axis_list = list(axes)
    for mask in range(1, 1 << len(axis_list)):
        nxt = [int(current[0]), int(current[1]), int(current[2])]
        for bit, axis in enumerate(axis_list):
            if mask & (1 << bit):
                nxt[axis] += int(steps[axis])
        vox = (int(nxt[0]), int(nxt[1]), int(nxt[2]))
        if vox not in seen:
            seen.add(vox)
            candidates.append(vox)
    return candidates


def _target_arrival_visible(
    voxel: VoxelIndex,
    occupied_set: set[VoxelIndex],
    occupied_lookup: np.ndarray | None,
    voxel_grid: VoxelGrid,
    target_tile_bounds: tuple[np.ndarray, np.ndarray] | None,
    dims: tuple[int, int, int],
    stride0: int,
    stride1: int,
) -> bool:
    if target_tile_bounds is None:
        return True
    return bool(
        _occupied_at(voxel, occupied_set, occupied_lookup, dims, stride0, stride1)
        and _voxel_intersects_bounds(voxel, voxel_grid, target_tile_bounds)
    )


def _source_tile_scores_from_candidate_positions(
    tiles,
    selected_voxel_samples: Sequence[WeightedVoxelSample],
    source_sigma_l: float,
    source_score_tau: float,
    source_min_candidates: int,
) -> tuple[Dict[str, float], Dict[str, list[int]], Dict[str, Dict[str, float]], Dict[str, float]]:
    """Convert predicted candidate positions into source tile scores.

    Candidate positions are assigned to the nearest tile AABB when they are
    inside or near it. Scores are relative to the median positive source tile,
    which makes a dense tile high without allowing raw counts to explode.
    """
    tile_list = list(tiles)
    tile_ids = [str(tile.unique_id) for tile in tile_list]
    bounds_by_tile = _tile_bounds_by_id(tile_list)
    source_radius = max(float(source_sigma_l), 1e-6)
    tau = max(float(source_score_tau), 1e-6)
    min_candidates = max(1, int(source_min_candidates))

    source_members: Dict[str, list[int]] = {tile_id: [] for tile_id in tile_ids}
    raw_stats: Dict[str, Dict[str, float]] = {
        tile_id: {"candidate_count": 0.0, "candidate_mass": 0.0}
        for tile_id in tile_ids
    }

    for sample_idx, sample in enumerate(selected_voxel_samples):
        best_tile_id = None
        best_distance = float("inf")
        center = np.asarray(sample.world_center, dtype=float)
        for tile_id, (tile_min, tile_max) in bounds_by_tile.items():
            distance = point_to_aabb_distance(center, tile_min, tile_max)
            if distance < best_distance:
                best_distance = float(distance)
                best_tile_id = tile_id
        if best_tile_id is None or best_distance > source_radius:
            continue

        source_members[best_tile_id].append(sample_idx)
        raw_stats[best_tile_id]["candidate_count"] += 1.0
        raw_stats[best_tile_id]["candidate_mass"] += float(sample.estimator_weight)

    positive_counts = [
        stats["candidate_count"]
        for stats in raw_stats.values()
        if stats["candidate_count"] >= float(min_candidates)
    ]
    positive_masses = [
        stats["candidate_mass"]
        for stats in raw_stats.values()
        if stats["candidate_count"] >= float(min_candidates) and stats["candidate_mass"] > 1e-12
    ]
    median_count = float(np.median(positive_counts)) if positive_counts else 0.0
    median_mass = float(np.median(positive_masses)) if positive_masses else 0.0

    source_scores: Dict[str, float] = {tile_id: 0.0 for tile_id in tile_ids}
    for tile_id, stats in raw_stats.items():
        count = float(stats["candidate_count"])
        mass = float(stats["candidate_mass"])
        if count < float(min_candidates) or mass <= 1e-12:
            stats["relative_count"] = 0.0
            stats["relative_mass"] = 0.0
            stats["source_score"] = 0.0
            continue
        relative_count = count / max(median_count, 1e-12) if median_count > 0.0 else 0.0
        relative_mass = mass / max(median_mass, 1e-12) if median_mass > 0.0 else 0.0
        relative_support = 0.5 * relative_count + 0.5 * relative_mass
        source_score = float(relative_support / (relative_support + tau)) if relative_support > 0.0 else 0.0
        source_scores[tile_id] = float(np.clip(source_score, 0.0, 1.0))
        stats["relative_count"] = float(relative_count)
        stats["relative_mass"] = float(relative_mass)
        stats["relative_support"] = float(relative_support)
        stats["source_score"] = float(source_scores[tile_id])

    stats_summary = {
        "source_tile_count": float(sum(1 for value in source_scores.values() if value > 0.0)),
        "source_median_candidate_count": float(median_count),
        "source_median_candidate_mass": float(median_mass),
        "source_presence_radius_l": float(source_radius),
        "source_score_tau": float(tau),
        "source_min_candidates": float(min_candidates),
    }
    return source_scores, source_members, raw_stats, stats_summary


def _empty_tile_debug_stats() -> Dict[str, Any]:
    return {
        "contributor_count": 0,
        "tested_rays": 0,
        "visible_rays": 0,
        "blocked_rays": 0,
        "stored_ray_count": 0,
        "stored_ray_hit_kinds": {},
        "top_contribution": 0.0,
    }


def _merge_tile_debug_stats(dst: Dict[str, Any], src: Mapping[str, Any]) -> None:
    dst["contributor_count"] = int(dst.get("contributor_count", 0)) + int(src.get("contributor_count", 0))
    dst["tested_rays"] = int(dst.get("tested_rays", 0)) + int(src.get("tested_rays", 0))
    dst["visible_rays"] = int(dst.get("visible_rays", 0)) + int(src.get("visible_rays", 0))
    dst["blocked_rays"] = int(dst.get("blocked_rays", 0)) + int(src.get("blocked_rays", 0))
    dst["stored_ray_count"] = int(dst.get("stored_ray_count", 0)) + int(src.get("stored_ray_count", 0))
    dst["top_contribution"] = max(float(dst.get("top_contribution", 0.0)), float(src.get("top_contribution", 0.0)))
    dst_kinds = dst.setdefault("stored_ray_hit_kinds", {})
    for kind, count in (src.get("stored_ray_hit_kinds", {}) or {}).items():
        dst_kinds[str(kind)] = int(dst_kinds.get(str(kind), 0)) + int(count)


def _compute_tile_entry_batch_visibility(
    active_tile_entries: Sequence[tuple],
    selected_voxel_samples: Sequence[WeightedVoxelSample],
    occupied_set: set[VoxelIndex],
    occupied_lookup: np.ndarray | None,
    voxel_grid: VoxelGrid,
    floor: float,
    aggregation: str,
    source_tiles_by_sample_idx: Mapping[int, Sequence[str]],
    debug_enabled: bool,
    debug_max_rays: int,
    debug_trace_voxels: bool,
) -> Dict[str, Any]:
    """Compute ray visibility for a target-tile batch."""
    batch_mass: Dict[str, float] = {}
    batch_pair_stats: Dict[tuple[str, str], Dict[str, float]] = {}
    batch_debug_tile_stats: Dict[str, Dict[str, Any]] = {}
    batch_debug_contributors: Dict[str, list[Dict[str, Any]]] = {}

    for tile_id, gate, sample_worlds, target_voxels, target_tile_bounds, sample_count in active_tile_entries:
        batch_mass.setdefault(tile_id, 0.0)
        for sample_idx, voxel_sample in enumerate(selected_voxel_samples):
            visible = 0
            debug_rays: list[Dict[str, Any]] = []
            for i in range(int(sample_count)):
                if debug_enabled and len(debug_rays) < debug_max_rays:
                    trace = _ray_trace_from_world(
                        start_world=voxel_sample.world_center,
                        start_voxel=voxel_sample.voxel,
                        target_point_world=sample_worlds[i],
                        occupied_set=occupied_set,
                        voxel_grid=voxel_grid,
                        target_voxel=target_voxels[i],
                        occupied_lookup=occupied_lookup,
                        target_tile_bounds=target_tile_bounds,
                        include_traversed_voxels=debug_trace_voxels,
                    )
                    ray_is_visible = bool(trace.get("visible"))
                    debug_rays.append(trace)
                else:
                    ray_is_visible = _ray_visible_from_world(
                        start_world=voxel_sample.world_center,
                        start_voxel=voxel_sample.voxel,
                        target_point_world=sample_worlds[i],
                        occupied_set=occupied_set,
                        voxel_grid=voxel_grid,
                        target_voxel=target_voxels[i],
                        occupied_lookup=occupied_lookup,
                        target_tile_bounds=target_tile_bounds,
                    )
                if ray_is_visible:
                    visible += 1

            if aggregation == "source_propagation":
                for source_tile_id in source_tiles_by_sample_idx.get(int(sample_idx), []):
                    if source_tile_id == tile_id:
                        continue
                    pair_stats = batch_pair_stats.setdefault(
                        (source_tile_id, tile_id),
                        {"visible_rays": 0.0, "tested_rays": 0.0},
                    )
                    pair_stats["visible_rays"] += float(visible)
                    pair_stats["tested_rays"] += float(sample_count)

            if visible <= 0:
                if debug_enabled:
                    stats = batch_debug_tile_stats.setdefault(tile_id, _empty_tile_debug_stats())
                    stats["tested_rays"] += int(sample_count)
                    stats["blocked_rays"] += int(sample_count)
                    for trace in debug_rays:
                        kind = str(trace.get("hit_kind") or "unknown")
                        stats["stored_ray_count"] += 1
                        stats["stored_ray_hit_kinds"][kind] = int(stats["stored_ray_hit_kinds"].get(kind, 0)) + 1
                continue

            visibility = max(float(visible) / sample_count, floor)
            contribution = float(voxel_sample.estimator_weight) * float(gate) * visibility
            batch_mass[tile_id] = float(batch_mass.get(tile_id, 0.0)) + contribution
            if debug_enabled:
                stats = batch_debug_tile_stats.setdefault(tile_id, _empty_tile_debug_stats())
                stats["contributor_count"] += 1
                stats["tested_rays"] += int(sample_count)
                stats["visible_rays"] += int(visible)
                stats["blocked_rays"] += int(sample_count) - int(visible)
                stats["top_contribution"] = max(float(stats["top_contribution"]), float(contribution))
                for trace in debug_rays:
                    kind = str(trace.get("hit_kind") or "unknown")
                    stats["stored_ray_count"] += 1
                    stats["stored_ray_hit_kinds"][kind] = int(stats["stored_ray_hit_kinds"].get(kind, 0)) + 1
                batch_debug_contributors.setdefault(tile_id, []).append(
                    {
                        "voxel": [int(v) for v in voxel_sample.voxel],
                        "center": [float(v) for v in voxel_sample.world_center],
                        "probability": float(voxel_sample.probability),
                        "estimator_weight": float(voxel_sample.estimator_weight),
                        "stratum": str(voxel_sample.stratum),
                        "gate": float(gate),
                        "visibility": float(visibility),
                        "visible_rays": int(visible),
                        "tested_rays": int(sample_count),
                        "contribution": float(contribution),
                        "rays": debug_rays,
                    }
                )

    return {
        "mass": batch_mass,
        "source_target_ray_stats": batch_pair_stats,
        "debug_tile_stats": batch_debug_tile_stats,
        "debug_contributors": batch_debug_contributors,
    }


def _raycast_worker_compute_batch(active_tile_entries: Sequence[tuple]) -> Dict[str, Any]:
    context = _RAYCAST_WORKER_CONTEXT
    if context is None:
        raise RuntimeError("raycast worker context is not initialized")
    return _compute_tile_entry_batch_visibility(
        active_tile_entries=active_tile_entries,
        selected_voxel_samples=context["selected_voxel_samples"],
        occupied_set=context["occupied_set"],
        occupied_lookup=context["occupied_lookup"],
        voxel_grid=context["voxel_grid"],
        floor=float(context["floor"]),
        aggregation=str(context["aggregation"]),
        source_tiles_by_sample_idx=context["source_tiles_by_sample_idx"],
        debug_enabled=bool(context["debug_enabled"]),
        debug_max_rays=int(context["debug_max_rays"]),
        debug_trace_voxels=bool(context["debug_trace_voxels"]),
    )


def _chunk_sequence(items: Sequence[Any], chunk_count: int) -> list[list[Any]]:
    if not items:
        return []
    count = max(1, min(int(chunk_count), len(items)))
    return [list(items[idx::count]) for idx in range(count) if items[idx::count]]


def _merge_tile_batch_result(
    result: Mapping[str, Any],
    mass: Dict[str, float],
    source_target_ray_stats: Dict[tuple[str, str], Dict[str, float]],
    debug_tile_stats: Dict[str, Dict[str, Any]],
    debug_contributors: Dict[str, list[Dict[str, Any]]],
) -> None:
    for tile_id, value in (result.get("mass", {}) or {}).items():
        mass[str(tile_id)] = float(mass.get(str(tile_id), 0.0)) + float(value)
    for pair_key, pair_stats in (result.get("source_target_ray_stats", {}) or {}).items():
        source_tile_id, target_tile_id = pair_key
        dst = source_target_ray_stats.setdefault(
            (str(source_tile_id), str(target_tile_id)),
            {"visible_rays": 0.0, "tested_rays": 0.0},
        )
        dst["visible_rays"] += float(pair_stats.get("visible_rays", 0.0))
        dst["tested_rays"] += float(pair_stats.get("tested_rays", 0.0))
    for tile_id, stats in (result.get("debug_tile_stats", {}) or {}).items():
        dst = debug_tile_stats.setdefault(str(tile_id), _empty_tile_debug_stats())
        _merge_tile_debug_stats(dst, stats)
    for tile_id, contributors in (result.get("debug_contributors", {}) or {}).items():
        debug_contributors.setdefault(str(tile_id), []).extend(list(contributors or []))


def _ray_visible_from_world(
    start_world: Sequence[float],
    start_voxel: VoxelIndex,
    target_point_world: Sequence[float],
    occupied_set: set[VoxelIndex],
    voxel_grid: VoxelGrid,
    target_voxel: Sequence[int] | None = None,
    occupied_lookup: np.ndarray | None = None,
    target_tile_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> bool:
    """Return whether the ray's first occupied hit belongs to the target tile."""
    start_x = float(start_world[0])
    start_y = float(start_world[1])
    start_z = float(start_world[2])
    end_x = float(target_point_world[0])
    end_y = float(target_point_world[1])
    end_z = float(target_point_world[2])

    cur_x, cur_y, cur_z = int(start_voxel[0]), int(start_voxel[1]), int(start_voxel[2])
    if target_voxel is None:
        tgt_x, tgt_y, tgt_z = voxel_grid.world_to_voxel((end_x, end_y, end_z), clamp=True)
    else:
        tgt_x, tgt_y, tgt_z = int(target_voxel[0]), int(target_voxel[1]), int(target_voxel[2])

    dims_x, dims_y, dims_z = int(voxel_grid.dims[0]), int(voxel_grid.dims[1]), int(voxel_grid.dims[2])
    dims = (dims_x, dims_y, dims_z)
    stride0 = dims_y * dims_z
    stride1 = dims_z

    if cur_x == tgt_x and cur_y == tgt_y and cur_z == tgt_z:
        return _target_arrival_visible(
            (cur_x, cur_y, cur_z),
            occupied_set,
            occupied_lookup,
            voxel_grid,
            target_tile_bounds,
            dims,
            stride0,
            stride1,
        )

    dir_x = end_x - start_x
    dir_y = end_y - start_y
    dir_z = end_z - start_z
    step_x = 1 if dir_x > 1e-12 else (-1 if dir_x < -1e-12 else 0)
    step_y = 1 if dir_y > 1e-12 else (-1 if dir_y < -1e-12 else 0)
    step_z = 1 if dir_z > 1e-12 else (-1 if dir_z < -1e-12 else 0)

    min_corner = voxel_grid.min_corner
    voxel_size = float(voxel_grid.voxel_size)

    if step_x == 0:
        t_max_x = np.inf
        t_delta_x = np.inf
    else:
        boundary_x = min_corner[0] + ((cur_x + 1) if step_x > 0 else cur_x) * voxel_size
        t_max_x = (boundary_x - start_x) / dir_x
        t_delta_x = voxel_size / abs(dir_x)

    if step_y == 0:
        t_max_y = np.inf
        t_delta_y = np.inf
    else:
        boundary_y = min_corner[1] + ((cur_y + 1) if step_y > 0 else cur_y) * voxel_size
        t_max_y = (boundary_y - start_y) / dir_y
        t_delta_y = voxel_size / abs(dir_y)

    if step_z == 0:
        t_max_z = np.inf
        t_delta_z = np.inf
    else:
        boundary_z = min_corner[2] + ((cur_z + 1) if step_z > 0 else cur_z) * voxel_size
        t_max_z = (boundary_z - start_z) / dir_z
        t_delta_z = voxel_size / abs(dir_z)

    max_iters = abs(tgt_x - cur_x) + abs(tgt_y - cur_y) + abs(tgt_z - cur_z) + 1024
    for _ in range(max_iters):
        axes = _step_axis_indices(t_max_x, t_max_y, t_max_z)
        candidates = _conservative_step_candidates(
            current=(cur_x, cur_y, cur_z),
            axes=axes,
            steps=(step_x, step_y, step_z),
        )
        target_hits = []
        for cand in candidates:
            if not _occupied_at(cand, occupied_set, occupied_lookup, dims, stride0, stride1):
                continue
            if target_tile_bounds is not None and _voxel_intersects_bounds(cand, voxel_grid, target_tile_bounds):
                target_hits.append(cand)
            else:
                return False

        if 0 in axes:
            cur_x += step_x
            t_max_x += t_delta_x
        if 1 in axes:
            cur_y += step_y
            t_max_y += t_delta_y
        if 2 in axes:
            cur_z += step_z
            t_max_z += t_delta_z

        if cur_x < 0 or cur_y < 0 or cur_z < 0 or cur_x >= dims_x or cur_y >= dims_y or cur_z >= dims_z:
            return False

        if target_hits:
            return True

        if cur_x == tgt_x and cur_y == tgt_y and cur_z == tgt_z:
            return _target_arrival_visible(
                (cur_x, cur_y, cur_z),
                occupied_set,
                occupied_lookup,
                voxel_grid,
                target_tile_bounds,
                dims,
                stride0,
                stride1,
            )

    return False


def _ray_trace_from_world(
    start_world: Sequence[float],
    start_voxel: VoxelIndex,
    target_point_world: Sequence[float],
    occupied_set: set[VoxelIndex],
    voxel_grid: VoxelGrid,
    target_voxel: Sequence[int] | None = None,
    occupied_lookup: np.ndarray | None = None,
    target_tile_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    include_traversed_voxels: bool = True,
    max_traversed_voxels: int = 96,
) -> Dict[str, Any]:
    """Trace one visibility ray and return a compact explanation of the first hit."""
    start = [float(start_world[0]), float(start_world[1]), float(start_world[2])]
    target = [float(target_point_world[0]), float(target_point_world[1]), float(target_point_world[2])]
    start_x, start_y, start_z = start
    end_x, end_y, end_z = target
    cur_x, cur_y, cur_z = int(start_voxel[0]), int(start_voxel[1]), int(start_voxel[2])
    if target_voxel is None:
        tgt_x, tgt_y, tgt_z = voxel_grid.world_to_voxel((end_x, end_y, end_z), clamp=True)
    else:
        tgt_x, tgt_y, tgt_z = int(target_voxel[0]), int(target_voxel[1]), int(target_voxel[2])

    traversed: list[list[int]] = []
    if include_traversed_voxels:
        traversed.append([cur_x, cur_y, cur_z])

    base = {
        "start": start,
        "target": target,
        "start_voxel": [cur_x, cur_y, cur_z],
        "target_voxel": [tgt_x, tgt_y, tgt_z],
        "traversed_voxels": traversed,
    }
    dims_x, dims_y, dims_z = int(voxel_grid.dims[0]), int(voxel_grid.dims[1]), int(voxel_grid.dims[2])
    dims = (dims_x, dims_y, dims_z)
    stride0 = dims_y * dims_z
    stride1 = dims_z
    if cur_x == tgt_x and cur_y == tgt_y and cur_z == tgt_z:
        visible = _target_arrival_visible(
            (cur_x, cur_y, cur_z),
            occupied_set,
            occupied_lookup,
            voxel_grid,
            target_tile_bounds,
            dims,
            stride0,
            stride1,
        )
        return {
            **base,
            "visible": visible,
            "hit_kind": "target_voxel" if visible else "empty_target_voxel",
            "hit_voxel": [cur_x, cur_y, cur_z],
            "hit_center": _voxel_center_or_none((cur_x, cur_y, cur_z), voxel_grid),
        }

    dir_x = end_x - start_x
    dir_y = end_y - start_y
    dir_z = end_z - start_z
    step_x = 1 if dir_x > 1e-12 else (-1 if dir_x < -1e-12 else 0)
    step_y = 1 if dir_y > 1e-12 else (-1 if dir_y < -1e-12 else 0)
    step_z = 1 if dir_z > 1e-12 else (-1 if dir_z < -1e-12 else 0)

    min_corner = voxel_grid.min_corner
    voxel_size = float(voxel_grid.voxel_size)

    if step_x == 0:
        t_max_x = np.inf
        t_delta_x = np.inf
    else:
        boundary_x = min_corner[0] + ((cur_x + 1) if step_x > 0 else cur_x) * voxel_size
        t_max_x = (boundary_x - start_x) / dir_x
        t_delta_x = voxel_size / abs(dir_x)

    if step_y == 0:
        t_max_y = np.inf
        t_delta_y = np.inf
    else:
        boundary_y = min_corner[1] + ((cur_y + 1) if step_y > 0 else cur_y) * voxel_size
        t_max_y = (boundary_y - start_y) / dir_y
        t_delta_y = voxel_size / abs(dir_y)

    if step_z == 0:
        t_max_z = np.inf
        t_delta_z = np.inf
    else:
        boundary_z = min_corner[2] + ((cur_z + 1) if step_z > 0 else cur_z) * voxel_size
        t_max_z = (boundary_z - start_z) / dir_z
        t_delta_z = voxel_size / abs(dir_z)

    max_iters = abs(tgt_x - cur_x) + abs(tgt_y - cur_y) + abs(tgt_z - cur_z) + 1024
    for _ in range(max_iters):
        axes = _step_axis_indices(t_max_x, t_max_y, t_max_z)
        candidates = _conservative_step_candidates(
            current=(cur_x, cur_y, cur_z),
            axes=axes,
            steps=(step_x, step_y, step_z),
        )
        if include_traversed_voxels:
            for cand in candidates:
                if len(traversed) >= max(1, int(max_traversed_voxels)):
                    break
                traversed.append([int(cand[0]), int(cand[1]), int(cand[2])])

        target_hits = []
        for cand in candidates:
            if not _occupied_at(cand, occupied_set, occupied_lookup, dims, stride0, stride1):
                continue
            in_target = bool(
                target_tile_bounds is not None
                and _voxel_intersects_bounds(cand, voxel_grid, target_tile_bounds)
            )
            if in_target:
                target_hits.append(cand)
                continue
            return {
                **base,
                "visible": False,
                "hit_kind": "occluder",
                "hit_voxel": [int(cand[0]), int(cand[1]), int(cand[2])],
                "hit_center": _voxel_center_or_none(cand, voxel_grid),
            }

        if 0 in axes:
            cur_x += step_x
            t_max_x += t_delta_x
        if 1 in axes:
            cur_y += step_y
            t_max_y += t_delta_y
        if 2 in axes:
            cur_z += step_z
            t_max_z += t_delta_z

        if cur_x < 0 or cur_y < 0 or cur_z < 0 or cur_x >= dims_x or cur_y >= dims_y or cur_z >= dims_z:
            return {
                **base,
                "visible": False,
                "hit_kind": "out_of_grid",
                "hit_voxel": [cur_x, cur_y, cur_z],
                "hit_center": None,
            }

        if target_hits:
            hit = target_hits[0]
            return {
                **base,
                "visible": True,
                "hit_kind": "target_tile",
                "hit_voxel": [int(hit[0]), int(hit[1]), int(hit[2])],
                "hit_center": _voxel_center_or_none(hit, voxel_grid),
            }

        if cur_x == tgt_x and cur_y == tgt_y and cur_z == tgt_z:
            visible = _target_arrival_visible(
                (cur_x, cur_y, cur_z),
                occupied_set,
                occupied_lookup,
                voxel_grid,
                target_tile_bounds,
                dims,
                stride0,
                stride1,
            )
            return {
                **base,
                "visible": visible,
                "hit_kind": "target_voxel" if visible else "empty_target_voxel",
                "hit_voxel": [cur_x, cur_y, cur_z],
                "hit_center": _voxel_center_or_none((cur_x, cur_y, cur_z), voxel_grid),
            }

    return {
        **base,
        "visible": False,
        "hit_kind": "max_iters",
        "hit_voxel": [cur_x, cur_y, cur_z],
        "hit_center": _voxel_center_or_none((cur_x, cur_y, cur_z), voxel_grid),
    }


def ray_visible(
    start_voxel: VoxelIndex,
    target_point_world: Sequence[float],
    occupied_set: set[VoxelIndex],
    voxel_grid: VoxelGrid,
) -> bool:
    """Check whether a target sample is visible from a starting voxel center."""
    start_world = voxel_grid.voxel_to_world_center(start_voxel)
    return _ray_visible_from_world(
        start_world=start_world,
        start_voxel=start_voxel,
        target_point_world=target_point_world,
        occupied_set=occupied_set,
        voxel_grid=voxel_grid,
    )


def tile_visibility_from_voxel(
    voxel_idx: VoxelIndex,
    tile_samples: Mapping[str, np.ndarray],
    occupied_set: set[VoxelIndex],
    voxel_grid: VoxelGrid,
    max_rays_per_tile: int | None = None,
    occupied_lookup: np.ndarray | None = None,
    tile_bounds_by_id: Mapping[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> Dict[str, float]:
    """Estimate per-tile visibility from one candidate user voxel."""
    out: Dict[str, float] = {}
    start_world = voxel_grid.voxel_to_world_center(voxel_idx)
    for tile_id, samples in tile_samples.items():
        arr = samples
        if arr.size == 0:
            out[tile_id] = 0.0
            continue

        total = arr.shape[0] if max_rays_per_tile is None else min(arr.shape[0], int(max_rays_per_tile))
        if total <= 0:
            out[tile_id] = 0.0
            continue

        visible = 0
        target_voxels = _world_points_to_voxels(arr[:total], voxel_grid)
        for i in range(total):
            if _ray_visible_from_world(
                start_world=start_world,
                start_voxel=voxel_idx,
                target_point_world=arr[i],
                occupied_set=occupied_set,
                voxel_grid=voxel_grid,
                target_voxel=target_voxels[i],
                occupied_lookup=occupied_lookup,
                target_tile_bounds=(tile_bounds_by_id or {}).get(tile_id),
            ):
                visible += 1

        out[tile_id] = float(visible) / float(total)
    return out


def aggregate_tile_importance(
    tiles,
    voxel_probabilities: Mapping[VoxelIndex, float],
    tile_samples: Mapping[str, np.ndarray],
    occupied_set: set[VoxelIndex],
    occupied_lookup: np.ndarray | None,
    voxel_grid: VoxelGrid,
    user_position: Sequence[float],
    r_vis: float,
    sigma_g: float,
    max_rays_per_tile: int | None = None,
    visibility_floor: float = 0.0,
    max_voxels_per_tile: int | None = None,
    max_candidate_voxels: int | None = None,
    voxel_selection_mode: str = "probability_diverse",
    voxel_selection_core_fraction: float = 0.7,
    aggregation_mode: str = "source_propagation",
    source_presence_sigma_l: float | None = None,
    source_score_tau: float = 1.0,
    source_min_candidates: int = 1,
    propagation_factor: float = 0.5,
    propagation_visibility_threshold: float = 1.0 / 3.0,
    raycast_processes: int = 1,
    voxel_world_centers: Mapping[VoxelIndex, np.ndarray] | None = None,
    voxel_navigation_paths: Mapping[VoxelIndex, Sequence[Sequence[float]]] | None = None,
    visual_debug: Mapping[str, Any] | None = None,
) -> tuple[Dict[str, float], Dict[str, Any]]:
    """Aggregate voxel probabilities and visibility into independent tile scores."""
    if not tiles:
        return {}, {"reason": "empty_tiles"}

    sigma = max(float(sigma_g), 1e-6)
    floor = max(0.0, min(1.0, float(visibility_floor)))
    user_pos = np.asarray(user_position, dtype=float)
    gate_by_tile = compute_tile_gates(tiles=tiles, user_position=user_pos, r_vis=float(r_vis), sigma_g=sigma)
    bounds_by_tile = _tile_bounds_by_id(tiles)
    voxel_items = [
        (vox, float(prob), np.asarray((voxel_world_centers or {}).get(vox, voxel_grid.voxel_to_world_center(vox)), dtype=float))
        for vox, prob in voxel_probabilities.items()
        if float(prob) > 0.0
    ]
    if max_candidate_voxels is None and max_voxels_per_tile is not None:
        max_candidate_voxels = max_voxels_per_tile
    voxel_cap = None if max_candidate_voxels is None else max(1, int(max_candidate_voxels))
    mode = str(voxel_selection_mode or "probability_diverse")
    core_fraction = float(np.clip(float(voxel_selection_core_fraction), 0.0, 1.0))
    selected_voxel_samples, estimator_stats = select_weighted_candidate_voxels(
        voxel_items=voxel_items,
        max_candidate_voxels=voxel_cap,
        voxel_selection_mode=mode,
        voxel_selection_core_fraction=core_fraction,
    )
    selected_mass_global = float(sum(sample.probability for sample in selected_voxel_samples))
    estimator_weight_global = float(sum(sample.estimator_weight for sample in selected_voxel_samples))
    aggregation = str(aggregation_mode or "expected_visibility")
    if aggregation not in {"expected_visibility", "source_propagation"}:
        aggregation = "expected_visibility"
    source_sigma = (
        float(source_presence_sigma_l)
        if source_presence_sigma_l is not None
        else float(voxel_grid.voxel_size) * 1.5
    )
    source_scores, source_members, source_stats_by_tile, source_summary = _source_tile_scores_from_candidate_positions(
        tiles=tiles,
        selected_voxel_samples=selected_voxel_samples,
        source_sigma_l=source_sigma,
        source_score_tau=float(source_score_tau),
        source_min_candidates=int(source_min_candidates),
    )
    source_tiles_by_sample_idx: Dict[int, list[str]] = {}
    for source_tile_id, sample_indices in source_members.items():
        if source_scores.get(source_tile_id, 0.0) <= 0.0:
            continue
        for sample_idx in sample_indices:
            source_tiles_by_sample_idx.setdefault(int(sample_idx), []).append(source_tile_id)
    propagated_scores = {str(tile.unique_id): 0.0 for tile in tiles}
    propagation_stats_by_tile: Dict[str, Dict[str, Any]] = {}
    source_target_ray_stats: Dict[tuple[str, str], Dict[str, float]] = {}
    propagation_gain = float(np.clip(float(propagation_factor), 0.0, 1.0))
    propagation_threshold = float(np.clip(float(propagation_visibility_threshold), 0.0, 1.0))
    debug_enabled = bool((visual_debug or {}).get("enabled", False))
    debug_max_voxels = max(0, int((visual_debug or {}).get("max_voxels", 32)))
    debug_max_tiles = max(0, int((visual_debug or {}).get("max_tiles", 24)))
    debug_max_contributors = max(0, int((visual_debug or {}).get("max_contributors_per_tile", 8)))
    debug_max_rays = max(0, int((visual_debug or {}).get("max_rays_per_contributor", 4)))
    debug_trace_voxels = bool((visual_debug or {}).get("trace_voxels", True))
    debug_contributors: Dict[str, list[Dict[str, Any]]] = {}
    debug_tile_stats: Dict[str, Dict[str, Any]] = {}
    requested_processes = max(1, int(raycast_processes or 1))

    mass = {str(tile.unique_id): 0.0 for tile in tiles}
    active_tile_samples = {
        tile_id: tile_samples[tile_id]
        for tile_id, gate in gate_by_tile.items()
        if gate > 0.0 and tile_id in tile_samples
    }
    active_tile_count = float(len(active_tile_samples))
    total_tile_count = float(len(mass))
    gated_out_tile_count = max(total_tile_count - active_tile_count, 0.0)
    ray_tests_per_visibility_eval = 0.0
    active_sample_points = 0.0
    active_tile_entries = []
    empty_support_tiles = 0.0
    for tile_id, samples in active_tile_samples.items():
        arr = np.asarray(samples, dtype=float)
        if arr.size == 0:
            empty_support_tiles += 1.0
            continue
        active_sample_points += float(arr.shape[0])
        total = arr.shape[0] if max_rays_per_tile is None else min(arr.shape[0], int(max_rays_per_tile))
        ray_tests_per_visibility_eval += float(max(total, 0))
        if total > 0:
            arr = arr[:total]
            active_tile_entries.append(
                (
                    tile_id,
                    float(gate_by_tile.get(tile_id, 0.0)),
                    arr,
                    _world_points_to_voxels(arr, voxel_grid),
                    bounds_by_tile.get(tile_id),
                    float(total),
                )
            )
    positive_voxel_count = float(len(voxel_items))
    effective_voxels_per_tile = float(len(selected_voxel_samples))
    selected_probs = [sample.probability for sample in selected_voxel_samples]
    selected_weights = [sample.estimator_weight for sample in selected_voxel_samples]
    raw_probs = np.asarray([prob for _, prob, _ in voxel_items], dtype=float)
    raw_prob_sum = float(np.sum(raw_probs)) if raw_probs.size else 0.0
    norm_probs = raw_probs / raw_prob_sum if raw_prob_sum > 1e-12 else raw_probs
    positive_norm = norm_probs[norm_probs > 1e-12]
    complexity_stats = {
        "active_tiles": active_tile_count,
        "active_tiles_without_sample_points": empty_support_tiles,
        "gated_out_tiles": gated_out_tile_count,
        "active_sample_points": active_sample_points,
        "ray_tests_per_visibility_eval": ray_tests_per_visibility_eval,
        "voxel_probability_support": positive_voxel_count,
        "effective_voxels_per_tile": effective_voxels_per_tile,
        "total_ray_tests": effective_voxels_per_tile * ray_tests_per_visibility_eval,
        "candidate_voxel_selection_mode": mode,
        "candidate_voxel_core_fraction": core_fraction,
        "candidate_voxel_diversity_fraction": 1.0 - core_fraction,
        "candidate_voxel_count_raw": positive_voxel_count,
        "candidate_voxel_count_selected": float(len(selected_voxel_samples)),
        "candidate_voxel_selected_mass": selected_mass_global,
        "candidate_voxel_estimator_weight_sum": estimator_weight_global,
        "candidate_voxel_min_probability": float(min(selected_probs)) if selected_probs else 0.0,
        "candidate_voxel_max_probability": float(max(selected_probs)) if selected_probs else 0.0,
        "candidate_voxel_mean_probability": float(np.mean(selected_probs)) if selected_probs else 0.0,
        "candidate_voxel_min_weight": float(min(selected_weights)) if selected_weights else 0.0,
        "candidate_voxel_max_weight": float(max(selected_weights)) if selected_weights else 0.0,
        "candidate_voxel_mean_weight": float(np.mean(selected_weights)) if selected_weights else 0.0,
        "voxel_probability_entropy": float(-np.sum(positive_norm * np.log2(positive_norm))) if positive_norm.size else 0.0,
        "voxel_probability_effective_sample_size": float(1.0 / np.sum(norm_probs * norm_probs)) if norm_probs.size and np.sum(norm_probs * norm_probs) > 1e-12 else 0.0,
        "tile_aggregation_mode": aggregation,
        "source_propagation_factor": propagation_gain,
        "source_propagation_visibility_threshold": propagation_threshold,
        "source_propagation_direct_visibility_fallback": 0.0,
        "raycast_processes_requested": float(requested_processes),
    }
    complexity_stats.update(estimator_stats)
    complexity_stats.update(source_summary)
    if max_voxels_per_tile is not None and max_candidate_voxels == max_voxels_per_tile:
        complexity_stats["max_voxels_per_tile_deprecated"] = 1.0
    if not active_tile_entries:
        if aggregation == "source_propagation" and any(value > 1e-12 for value in source_scores.values()):
            return {tile_id: float(np.clip(value, 0.0, 1.0)) for tile_id, value in source_scores.items()}, complexity_stats
        return uniform_tile_distribution(tiles), complexity_stats

    parallel_used = 0
    if estimator_weight_global > 1e-12:
        worker_context = {
            "selected_voxel_samples": selected_voxel_samples,
            "occupied_set": occupied_set,
            "occupied_lookup": occupied_lookup,
            "voxel_grid": voxel_grid,
            "floor": floor,
            "aggregation": aggregation,
            "source_tiles_by_sample_idx": source_tiles_by_sample_idx,
            "debug_enabled": debug_enabled,
            "debug_max_rays": debug_max_rays,
            "debug_trace_voxels": debug_trace_voxels,
        }
        if requested_processes > 1 and len(active_tile_entries) > 1:
            try:
                # Do not fork from the threaded web/batch runner.  In
                # particular, grouped playback can have an active PLY loader
                # process pool when this call is made.  A spawned worker gets
                # a clean interpreter and an explicit, pickled context via
                # the initializer, avoiding inherited locks and zombie pools.
                ctx = mp.get_context("spawn")
                batches = _chunk_sequence(active_tile_entries, requested_processes)
                executor = ProcessPoolExecutor(
                    max_workers=len(batches),
                    mp_context=ctx,
                    initializer=_initialize_raycast_worker,
                    initargs=(worker_context,),
                )
                register_compute_pool(executor)
                try:
                    with executor:
                        for result in executor.map(_raycast_worker_compute_batch, batches):
                            _merge_tile_batch_result(
                                result=result,
                                mass=mass,
                                source_target_ray_stats=source_target_ray_stats,
                                debug_tile_stats=debug_tile_stats,
                                debug_contributors=debug_contributors,
                            )
                finally:
                    unregister_compute_pool(executor)
                parallel_used = len(batches)
            except Exception as exc:  # pylint: disable=broad-except
                if force_stop_requested():
                    raise RuntimeError("Raycast worker pool force-canceled.") from exc
                complexity_stats["raycast_parallel_fallback"] = 1.0
                complexity_stats["raycast_parallel_error"] = str(exc)
                parallel_used = 0
                mass = {str(tile.unique_id): 0.0 for tile in tiles}
                source_target_ray_stats = {}
                debug_tile_stats = {}
                debug_contributors = {}
        if parallel_used <= 0:
            result = _compute_tile_entry_batch_visibility(
                active_tile_entries=active_tile_entries,
                selected_voxel_samples=selected_voxel_samples,
                occupied_set=occupied_set,
                occupied_lookup=occupied_lookup,
                voxel_grid=voxel_grid,
                floor=floor,
                aggregation=aggregation,
                source_tiles_by_sample_idx=source_tiles_by_sample_idx,
                debug_enabled=debug_enabled,
                debug_max_rays=debug_max_rays,
                debug_trace_voxels=debug_trace_voxels,
            )
            _merge_tile_batch_result(
                result=result,
                mass=mass,
                source_target_ray_stats=source_target_ray_stats,
                debug_tile_stats=debug_tile_stats,
                debug_contributors=debug_contributors,
            )
    complexity_stats["raycast_processes_used"] = float(max(1, parallel_used))
    complexity_stats["raycast_parallel_spawn"] = float(parallel_used > 0)

    complexity_stats["selected_voxel_mass_mean"] = selected_mass_global
    complexity_stats["selected_voxel_estimator_weight_mean"] = estimator_weight_global

    final_mass = mass
    if aggregation == "source_propagation":
        for (source_tile_id, target_tile_id), pair_stats in source_target_ray_stats.items():
            tested = float(pair_stats.get("tested_rays", 0.0))
            if tested <= 0.0:
                continue
            visibility_ratio = float(pair_stats.get("visible_rays", 0.0)) / tested
            if visibility_ratio < propagation_threshold:
                continue
            propagated = (
                propagation_gain
                * float(source_scores.get(source_tile_id, 0.0))
                * visibility_ratio
                * float(gate_by_tile.get(target_tile_id, 0.0))
            )
            if propagated <= float(propagated_scores.get(target_tile_id, 0.0)):
                continue
            propagated_scores[target_tile_id] = float(propagated)
            propagation_stats_by_tile[target_tile_id] = {
                "source_tile_id": source_tile_id,
                "source_score": float(source_scores.get(source_tile_id, 0.0)),
                "propagated_score": float(propagated),
                "visibility_ratio": float(visibility_ratio),
                "visible_rays": int(pair_stats.get("visible_rays", 0.0)),
                "tested_rays": int(pair_stats.get("tested_rays", 0.0)),
                "gate": float(gate_by_tile.get(target_tile_id, 0.0)),
            }
        final_mass = {
            tile_id: max(float(source_scores.get(tile_id, 0.0)), float(propagated_scores.get(tile_id, 0.0)))
            for tile_id in mass.keys()
        }
        complexity_stats["source_tiles_with_score"] = float(sum(1 for value in source_scores.values() if value > 0.0))
        complexity_stats["propagated_tiles_with_score"] = float(sum(1 for value in propagated_scores.values() if value > 0.0))
        complexity_stats["source_target_pairs_tested"] = float(len(source_target_ray_stats))
        complexity_stats["source_target_pairs_passing_threshold"] = float(len(propagation_stats_by_tile))
        # Source propagation is an enhancement of direct expected visibility,
        # not a reason to discard it.  Uniform-reachable origins often lie in
        # free space rather than within a source tile's small presence radius;
        # in that case there are no source/propagated scores even though the
        # raycaster has valid direct visibility evidence in ``mass``.
        if not any(value > 1e-12 for value in final_mass.values()) and any(
            value > 1e-12 for value in mass.values()
        ):
            final_mass = dict(mass)
            complexity_stats["source_propagation_direct_visibility_fallback"] = 1.0

    if debug_enabled:
        selected_for_debug = sorted(
            selected_voxel_samples,
            key=lambda sample: float(sample.estimator_weight),
            reverse=True,
        )[:debug_max_voxels or 0]
        top_tile_ids = [
            tile_id
            for tile_id, _value in sorted(final_mass.items(), key=lambda kv: kv[1], reverse=True)[:debug_max_tiles or 0]
        ]
        debug_stats_by_tile = {}
        for tile_id in top_tile_ids:
            stats = dict(debug_tile_stats.get(tile_id, {}))
            score = float(final_mass.get(tile_id, 0.0))
            expected_score = float(mass.get(tile_id, 0.0))
            tested = int(stats.get("tested_rays", 0))
            visible = int(stats.get("visible_rays", 0))
            top = float(stats.get("top_contribution", 0.0))
            stats["visibility_ratio"] = float(visible / tested) if tested > 0 else 0.0
            stats["top_contributor_share"] = float(top / expected_score) if expected_score > 1e-12 else 0.0
            stats["score"] = score
            stats["expected_visibility_score"] = expected_score
            stats["source_score"] = float(source_scores.get(tile_id, 0.0))
            stats["propagated_score"] = float(propagated_scores.get(tile_id, 0.0))
            stats["source_stats"] = source_stats_by_tile.get(tile_id, {})
            stats["propagation"] = propagation_stats_by_tile.get(tile_id, {})
            debug_stats_by_tile[tile_id] = stats
        complexity_stats["_visual_debug"] = {
            "schema": "prism.raycast_visual_debug.v1",
            "voxel_grid": {
                "min_corner": [float(v) for v in voxel_grid.min_corner],
                "voxel_size": float(voxel_grid.voxel_size),
                "dims": [int(v) for v in voxel_grid.dims],
            },
            "user_position": [float(v) for v in user_pos],
            "selected_voxels": [
                {
                    "voxel": [int(v) for v in sample.voxel],
                    "center": [float(v) for v in sample.world_center],
                    "probability": float(sample.probability),
                    "estimator_weight": float(sample.estimator_weight),
                    "stratum": str(sample.stratum),
                    "path": [
                        [float(v) for v in point]
                        for point in (voxel_navigation_paths or {}).get(sample.voxel, [])
                        if len(point) >= 3
                    ],
                }
                for sample in selected_for_debug
            ],
            "distribution_stats": {
                "positive_voxels": int(positive_voxel_count),
                "selected_voxels": int(len(selected_voxel_samples)),
                "selected_mass": float(selected_mass_global),
                "estimator_weight_sum": float(estimator_weight_global),
                "entropy_bits": float(complexity_stats["voxel_probability_entropy"]),
                "effective_sample_size": float(complexity_stats["voxel_probability_effective_sample_size"]),
                "max_probability": float(complexity_stats["candidate_voxel_max_probability"]),
                "mean_probability": float(complexity_stats["candidate_voxel_mean_probability"]),
                "core_count": int(estimator_stats.get("candidate_voxel_core_count", 0.0)),
                "tail_count": int(estimator_stats.get("candidate_voxel_tail_count", 0.0)),
                "core_mass": float(estimator_stats.get("candidate_voxel_core_mass", 0.0)),
                "tail_mass": float(estimator_stats.get("candidate_voxel_tail_mass", 0.0)),
                "active_tiles_without_sample_points": int(empty_support_tiles),
                "aggregation_mode": aggregation,
                "source_tile_count": int(source_summary.get("source_tile_count", 0.0)),
                "source_median_candidate_count": float(source_summary.get("source_median_candidate_count", 0.0)),
                "source_median_candidate_mass": float(source_summary.get("source_median_candidate_mass", 0.0)),
                "propagated_tile_count": int(complexity_stats.get("propagated_tiles_with_score", 0.0)),
            },
            "tiles": {
                tile_id: {
                    "score": float(final_mass.get(tile_id, 0.0)),
                    "gate": float(gate_by_tile.get(tile_id, 0.0)),
                    "stats": debug_stats_by_tile.get(tile_id, {}),
                    "contributors": sorted(
                        debug_contributors.get(tile_id, []),
                        key=lambda item: float(item.get("contribution", 0.0)),
                        reverse=True,
                    )[:debug_max_contributors or 0],
                }
                for tile_id in top_tile_ids
            },
        }

    if not any(value > 1e-12 for value in final_mass.values()):
        return uniform_tile_distribution(tiles), complexity_stats

    return {tile_id: float(np.clip(val, 0.0, 1.0)) for tile_id, val in final_mass.items()}, complexity_stats
