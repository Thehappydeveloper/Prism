from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np

from .path_planning import dijkstra_26_connected, path_direction
from .pose_intention import IntentionEstimate, estimate_intention, extract_position
from .raycast_visibility import (
    aggregate_tile_importance,
    sample_points_per_tile,
    uniform_tile_distribution,
)
from .voxel_grid import (
    VoxelGrid,
    VoxelIndex,
    build_occupancy_set,
    compute_expanded_aabb,
    enumerate_voxels_in_sphere,
    find_nearest_free_voxel,
)


@dataclass
class ReachabilityParams:
    voxel_size_l: float = 0.5
    r_nav: float = 0.5
    v_max: float = 1.5
    segment_duration_s: float = 1.0
    use_obstacles: bool = True
    k_path_dir: int = 3
    history_frames: int = 15
    s0: float = 0.2
    kappa_s: float = 4.0
    kappa_a: float = 0.5
    gamma: float = 1.0
    lambda_max: float = 0.9
    softmax_temp: float = 1.0
    samples_per_tile: int = 24
    max_voxels_eval: int = 128
    min_voxel_prob: float = 1e-4
    r_vis: float = 6.0
    sigma_g: float | None = None
    rng_seed: int | None = 0
    assumed_fps: float = 30.0
    smooth_window: int = 3
    max_rays_per_tile: int | None = None


@dataclass
class VoxelContext:
    voxel_grid: VoxelGrid
    occupied_voxels: set[VoxelIndex]


@dataclass
class NavigationState:
    voxel_grid: VoxelGrid
    occupied_voxels: set[VoxelIndex]
    candidate_voxels: set[VoxelIndex]
    free_voxels: set[VoxelIndex]
    start_voxel: VoxelIndex
    reach_radius: float


def build_voxel_context(points_xyz: np.ndarray, params: ReachabilityParams) -> VoxelContext:
    mins, maxs = compute_expanded_aabb(points_xyz, expansion=params.r_nav)
    grid = VoxelGrid(min_corner=mins, max_corner=maxs, voxel_size=float(params.voxel_size_l))
    occupied = build_occupancy_set(points_xyz, grid)
    return VoxelContext(voxel_grid=grid, occupied_voxels=occupied)


def build_navigation_space(
    voxel_context: VoxelContext,
    user_position: Sequence[float],
    params: ReachabilityParams,
) -> NavigationState | None:
    grid = voxel_context.voxel_grid
    occupied = voxel_context.occupied_voxels

    reach_radius = max(1e-6, float(params.v_max) * max(1e-6, float(params.segment_duration_s)))
    candidates = set(enumerate_voxels_in_sphere(grid, user_position, reach_radius))
    if not candidates:
        start_only = grid.world_to_voxel(user_position, clamp=True)
        candidates = {start_only}

    if params.use_obstacles:
        free = {v for v in candidates if v not in occupied}
    else:
        free = set(candidates)

    if not free:
        return None

    start = grid.world_to_voxel(user_position, clamp=True)
    if start not in free:
        nearest = find_nearest_free_voxel(start, free)
        if nearest is None:
            return None
        start = nearest

    return NavigationState(
        voxel_grid=grid,
        occupied_voxels=occupied,
        candidate_voxels=candidates,
        free_voxels=free,
        start_voxel=start,
        reach_radius=reach_radius,
    )


def _unit(vec: Sequence[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    n = float(np.linalg.norm(arr))
    if n <= 1e-12:
        return np.zeros(3, dtype=float)
    return arr / n


def _softmax_from_scores(scores: Dict[VoxelIndex, float], temperature: float = 1.0) -> Dict[VoxelIndex, float]:
    if not scores:
        return {}
    temp = max(float(temperature), 1e-6)
    keys = list(scores.keys())
    vals = np.asarray([scores[k] for k in keys], dtype=float) / temp
    vmax = float(np.max(vals))
    expv = np.exp(vals - vmax)
    denom = float(np.sum(expv))
    if denom <= 1e-12:
        w = 1.0 / float(len(keys))
        return {k: w for k in keys}
    probs = expv / denom
    return {k: float(p) for k, p in zip(keys, probs)}


def compute_voxel_probabilities(
    nav_state: NavigationState,
    intention: IntentionEstimate,
    user_position: Sequence[float],
    params: ReachabilityParams,
) -> Dict[VoxelIndex, float]:
    if not nav_state.free_voxels:
        return {}

    d_max = max(float(params.v_max) * max(float(params.segment_duration_s), 1e-6), 1e-6)
    lambda_t = float(np.clip(float(params.lambda_max) * intention.effective_confidence, 0.0, 1.0))

    move_dir = _unit(intention.move_dir)
    move_norm = float(np.linalg.norm(move_dir))

    scores: Dict[VoxelIndex, float] = {}

    if params.use_obstacles:
        dist_map, prev_map = dijkstra_26_connected(
            free_voxels=nav_state.free_voxels,
            start_voxel=nav_state.start_voxel,
            voxel_size=float(params.voxel_size_l),
        )

        for vox, dist in dist_map.items():
            d_norm = float(np.clip(dist / d_max, 0.0, 1.0))
            path_dir = path_direction(
                prev_map=prev_map,
                start=nav_state.start_voxel,
                target=vox,
                k_steps=max(1, int(params.k_path_dir)),
                voxel_to_center=nav_state.voxel_grid.voxel_to_world_center,
            )
            if move_norm <= 1e-9 or np.linalg.norm(path_dir) <= 1e-9:
                align = 0.5
            else:
                align = (1.0 + float(np.clip(np.dot(move_dir, path_dir), -1.0, 1.0))) * 0.5
            score = -((1.0 - lambda_t) * d_norm) + (lambda_t * align)
            scores[vox] = float(score)
    else:
        pos = np.asarray(user_position, dtype=float)
        for vox in nav_state.free_voxels:
            center = nav_state.voxel_grid.voxel_to_world_center(vox)
            vec = center - pos
            dist = float(np.linalg.norm(vec))
            d_norm = float(np.clip(dist / d_max, 0.0, 1.0))
            path_dir = _unit(vec)
            if move_norm <= 1e-9 or np.linalg.norm(path_dir) <= 1e-9:
                align = 0.5
            else:
                align = (1.0 + float(np.clip(np.dot(move_dir, path_dir), -1.0, 1.0))) * 0.5
            score = -((1.0 - lambda_t) * d_norm) + (lambda_t * align)
            scores[vox] = float(score)

    return _softmax_from_scores(scores, temperature=float(params.softmax_temp))


def prune_voxel_probabilities(
    voxel_probs: Mapping[VoxelIndex, float],
    max_voxels_eval: int,
    min_voxel_prob: float,
) -> Dict[VoxelIndex, float]:
    if not voxel_probs:
        return {}

    items = sorted(voxel_probs.items(), key=lambda kv: kv[1], reverse=True)

    threshold = max(0.0, float(min_voxel_prob))
    if threshold > 0:
        items = [kv for kv in items if kv[1] >= threshold]

    cap = max(1, int(max_voxels_eval)) if max_voxels_eval is not None else None
    if cap is not None:
        items = items[:cap]

    if not items:
        return {}

    z = float(sum(p for _, p in items))
    if z <= 1e-12:
        w = 1.0 / float(len(items))
        return {k: w for k, _ in items}

    return {k: float(p / z) for k, p in items}


def compute_tile_importance(
    tiles,
    points_xyz: np.ndarray,
    pose_now,
    pose_history,
    params: ReachabilityParams,
    voxel_context: VoxelContext | None = None,
    tile_samples: Mapping[str, np.ndarray] | None = None,
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Compute tile probabilities and return (tile_probs, debug_stats)."""
    if points_xyz.size == 0 or not tiles:
        return uniform_tile_distribution(tiles), {"reason": "empty_points_or_tiles"}

    try:
        user_position = extract_position(pose_now)
    except Exception:
        return uniform_tile_distribution(tiles), {"reason": "missing_pose"}

    local_context = voxel_context or build_voxel_context(points_xyz, params)
    nav_state = build_navigation_space(local_context, user_position=user_position, params=params)
    if nav_state is None:
        return uniform_tile_distribution(tiles), {"reason": "no_navigation_state"}

    intention = estimate_intention(
        pose_now=pose_now,
        pose_history=pose_history,
        s0=params.s0,
        kappa_s=params.kappa_s,
        kappa_a=params.kappa_a,
        gamma=params.gamma,
        assumed_fps=params.assumed_fps,
        smooth_window=params.smooth_window,
    )

    voxel_probs = compute_voxel_probabilities(
        nav_state=nav_state,
        intention=intention,
        user_position=user_position,
        params=params,
    )
    kept_voxel_probs = prune_voxel_probabilities(
        voxel_probs,
        max_voxels_eval=params.max_voxels_eval,
        min_voxel_prob=params.min_voxel_prob,
    )
    if not kept_voxel_probs:
        return uniform_tile_distribution(tiles), {"reason": "no_voxel_probs"}

    local_tile_samples = tile_samples or sample_points_per_tile(
        points_xyz=points_xyz,
        tiles=tiles,
        samples_per_tile=params.samples_per_tile,
        rng_seed=params.rng_seed,
    )

    sigma = float(params.sigma_g) if params.sigma_g is not None else float(params.r_vis) * 0.5
    sigma = max(sigma, 1e-6)

    tile_probs = aggregate_tile_importance(
        tiles=tiles,
        voxel_probabilities=kept_voxel_probs,
        tile_samples=local_tile_samples,
        occupied_set=nav_state.occupied_voxels,
        voxel_grid=nav_state.voxel_grid,
        user_position=user_position,
        r_vis=float(params.r_vis),
        sigma_g=sigma,
        max_rays_per_tile=params.max_rays_per_tile,
    )

    p_vals = np.asarray(list(kept_voxel_probs.values()), dtype=float)
    t_vals = np.asarray(list(tile_probs.values()), dtype=float) if tile_probs else np.asarray([], dtype=float)
    debug_stats = {
        "candidate_voxels": float(len(nav_state.candidate_voxels)),
        "free_voxels": float(len(nav_state.free_voxels)),
        "kept_voxels": float(len(kept_voxel_probs)),
        "voxel_prob_min": float(np.min(p_vals)) if p_vals.size else 0.0,
        "voxel_prob_mean": float(np.mean(p_vals)) if p_vals.size else 0.0,
        "voxel_prob_max": float(np.max(p_vals)) if p_vals.size else 0.0,
        "tile_nonzero_fraction": float(np.mean(t_vals > 0.0)) if t_vals.size else 0.0,
        "c_eff": float(intention.effective_confidence),
    }

    return tile_probs, debug_stats
