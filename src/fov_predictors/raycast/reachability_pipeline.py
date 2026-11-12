"""Core reachability pipeline used by the raycast predictor.

This module holds the stage-by-stage logic that converts a point cloud and pose
history into independent per-tile scores. `predictor.py` calls into this module once
it has loaded frame data; lower-level modules provide the voxel, path, and
visibility primitives used here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .path_planning import dijkstra_26_connected, dijkstra_connected, path_direction
from .pose_intention import IntentionEstimate, estimate_intention, extract_position
from .raycast_visibility import (
    allocate_samples_per_tile,
    aggregate_tile_importance,
    collect_points_per_tile,
    compute_tile_gates,
    sample_precollected_points_per_tile,
    sample_points_per_tile,
    summarize_tile_gates,
    uniform_tile_distribution,
)
from .voxel_grid import (
    VoxelGrid,
    VoxelIndex,
    build_occupancy_set,
    compute_expanded_aabb,
    enumerate_voxels_in_sphere,
    find_nearest_free_voxel,
    neighbor_offsets_26,
)


@dataclass
class ReachabilityParams:
    """Tunable parameters shared across voxel, intention, and visibility stages."""
    voxel_size_l: float = 0.5
    occupancy_voxel_size_l: float | None = None
    navigation_voxel_size_l: float | None = None
    visibility_voxel_size_l: float | None = None
    path_neighbor_connectivity: int = 6
    r_nav: float = 0.5
    v_max: float = 1.5
    segment_duration_s: float = 1.0
    prediction_horizon_s: float | None = None
    use_obstacles: bool = True
    k_path_dir: int = 3
    history_frames: int = 15
    history_horizons_s: tuple[float, ...] = (0.5, 1.5, 3.0)
    max_speed: float = 1.5
    max_acceleration: float = 3.0
    stability_weight: float = 0.7
    mismatch_penalty_strength: float = 1.0
    lambda_max: float = 0.9
    softmax_temp: float = 1.0
    samples_per_tile: int = 24
    max_voxels_eval: int = 128
    min_voxel_prob: float = 1e-4
    auto_samples_per_tile: bool = False
    auto_max_voxels_eval: bool = False
    auto_min_voxel_prob: bool = False
    r_vis: float = 6.0
    sigma_g: float | None = None
    visibility_floor: float = 0.0
    rng_seed: int | None = 0
    assumed_fps: float = 30.0
    smooth_window: int = 3
    max_rays_per_tile: int | None = None
    max_voxels_per_tile: int | None = None
    max_candidate_voxels: int | None = None
    voxel_selection_mode: str = "probability_diverse"
    voxel_selection_core_fraction: float = 0.7
    origin_distribution_mode: str = "motion_weighted"
    aggregation_mode: str = "source_propagation"
    source_presence_sigma_l: float | None = None
    source_score_tau: float = 1.0
    source_min_candidates: int = 1
    propagation_factor: float = 0.5
    propagation_visibility_threshold: float = 1.0 / 3.0
    raycast_processes: int = 1
    angle_convention: str = "unity_zxy"
    auto_visibility_time_budget_ms: float = 20.0
    auto_visibility_ms_per_ray: float = 0.002
    auto_voxel_mass_target: float = 0.95
    auto_tile_confidence: float = 0.95
    auto_tile_support_points: int = 32
    auto_min_samples_per_tile: int = 4
    auto_max_samples_per_tile: int = 64
    visual_debug_enabled: bool = False
    visual_debug_max_tiles: int = 24
    visual_debug_max_voxels: int = 32
    visual_debug_max_contributors_per_tile: int = 8
    visual_debug_max_rays_per_contributor: int = 4
    visual_debug_trace_voxels: bool = True


def _prediction_horizon_s(params: ReachabilityParams) -> float:
    """Return the horizon used for future reachability, falling back to segment duration."""
    value = params.prediction_horizon_s
    if value is None:
        value = params.segment_duration_s
    return max(float(value), 1e-6)


def _motion_horizons_s(params: ReachabilityParams) -> tuple[float, ...]:
    horizons = tuple(float(v) for v in params.history_horizons_s if float(v) > 0.0)
    if horizons:
        return horizons
    return (max(float(params.history_frames) / max(float(params.assumed_fps), 1e-6), 1e-3),)


def _path_k_for_horizon(horizon_s: float, params: ReachabilityParams) -> int:
    voxel_size = params.navigation_voxel_size_l or params.voxel_size_l
    voxel_size = max(float(voxel_size), 1e-6)
    return max(1, int(round(max(float(params.v_max), 1e-6) * max(float(horizon_s), 1e-6) / voxel_size)))


def _path_dir_steps(params: ReachabilityParams) -> tuple[int, ...]:
    return tuple(sorted({_path_k_for_horizon(horizon_s, params) for horizon_s in _motion_horizons_s(params)})) or (1,)


def _nearest_path_k(path_dir_maps: Mapping[int, Dict[VoxelIndex, np.ndarray]], requested_k: int) -> int | None:
    if not path_dir_maps:
        return None
    requested = int(requested_k)
    return min(path_dir_maps.keys(), key=lambda value: (abs(int(value) - requested), int(value)))


@dataclass
class VoxelContext:
    """Cached scene geometry: the voxel grid plus occupied voxels for a frame."""
    voxel_grid: VoxelGrid
    occupied_voxels: set[VoxelIndex]
    occupied_lookup: np.ndarray


@dataclass
class PathVoxelTopology:
    """Fine-subvoxel passability for one coarse path voxel."""
    free_mask: int
    component_by_cell: tuple[int, ...]
    component_faces: dict[int, int]


@dataclass
class PathContext(VoxelContext):
    """Coarse path grid whose edges are validated by a finer occupancy grid."""
    occupancy_context: VoxelContext | None = None
    topology_by_voxel: dict[VoxelIndex, PathVoxelTopology] | None = None
    passable_voxels: set[VoxelIndex] | None = None
    connectivity: int = 6


@dataclass
class NavigationState:
    """Reachable navigation subset derived from the current user position."""
    voxel_grid: VoxelGrid
    occupied_voxels: set[VoxelIndex]
    candidate_voxels: set[VoxelIndex]
    free_voxels: set[VoxelIndex]
    start_voxel: VoxelIndex
    reach_radius: float
    path_context: PathContext | None = None


@dataclass
class NavigationFeatures:
    """Reusable path features for scoring multiple intention hypotheses."""
    dist_map: Dict[VoxelIndex, float]
    prev_map: Dict[VoxelIndex, VoxelIndex | None]
    path_dir_maps: Dict[int, Dict[VoxelIndex, np.ndarray]]


@dataclass
class HorizonIntention:
    """Motion intention estimate paired with the matching path-direction scale."""
    horizon_s: float
    path_k: int
    estimate: IntentionEstimate


def build_voxel_context(points_xyz: np.ndarray, params: ReachabilityParams) -> VoxelContext:
    """Voxelize the frame point cloud and record which voxels are occupied."""
    mins, maxs = compute_expanded_aabb(points_xyz, expansion=params.r_nav)
    grid = VoxelGrid(min_corner=mins, max_corner=maxs, voxel_size=float(params.voxel_size_l))
    occupied = build_occupancy_set(points_xyz, grid)
    occupied_lookup = np.zeros(int(np.prod(grid.dims)), dtype=bool)
    if occupied:
        dims = np.asarray(grid.dims, dtype=np.int64)
        stride0 = int(dims[1]) * int(dims[2])
        stride1 = int(dims[2])
        occupied_idx = np.asarray(list(occupied), dtype=np.int64)
        flat_idx = (
            occupied_idx[:, 0] * stride0
            + occupied_idx[:, 1] * stride1
            + occupied_idx[:, 2]
        )
        occupied_lookup[flat_idx] = True
    return VoxelContext(voxel_grid=grid, occupied_voxels=occupied, occupied_lookup=occupied_lookup)


def build_unobstructed_navigation_context(points_xyz: np.ndarray, params: ReachabilityParams) -> VoxelContext:
    """Build a navigation grid that uses scene bounds but no occupancy information.

    This is the fast path for ``use_obstacles=False``.  Point-cloud geometry
    establishes the local coordinate extent only; no points are marked occupied
    and no free-space topology is constructed.  Visibility still uses its own
    occupied voxel context for ray casting.
    """
    mins, maxs = compute_expanded_aabb(points_xyz, expansion=params.r_nav)
    grid = VoxelGrid(min_corner=mins, max_corner=maxs, voxel_size=float(params.voxel_size_l))
    return VoxelContext(
        voxel_grid=grid,
        occupied_voxels=set(),
        occupied_lookup=np.zeros(int(np.prod(grid.dims)), dtype=bool),
    )


FACE_NEG_X = 1 << 0
FACE_POS_X = 1 << 1
FACE_NEG_Y = 1 << 2
FACE_POS_Y = 1 << 3
FACE_NEG_Z = 1 << 4
FACE_POS_Z = 1 << 5

_LOCAL_CELLS_2 = tuple((x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1))
_LOCAL_NEIGHBORS_6 = (
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
)


def _local_cell_index(cell: Sequence[int]) -> int:
    return int(cell[0]) * 4 + int(cell[1]) * 2 + int(cell[2])


def _local_cell_faces(cell: Sequence[int]) -> int:
    x, y, z = int(cell[0]), int(cell[1]), int(cell[2])
    faces = 0
    if x == 0:
        faces |= FACE_NEG_X
    if x == 1:
        faces |= FACE_POS_X
    if y == 0:
        faces |= FACE_NEG_Y
    if y == 1:
        faces |= FACE_POS_Y
    if z == 0:
        faces |= FACE_NEG_Z
    if z == 1:
        faces |= FACE_POS_Z
    return faces


def _topology_from_free_mask(free_mask: int) -> PathVoxelTopology:
    component_by_cell = [-1] * 8
    component_faces: dict[int, int] = {}
    component_id = 0
    for local in _LOCAL_CELLS_2:
        idx = _local_cell_index(local)
        if not (free_mask & (1 << idx)) or component_by_cell[idx] >= 0:
            continue
        stack = [local]
        component_by_cell[idx] = component_id
        faces = 0
        while stack:
            cell = stack.pop()
            faces |= _local_cell_faces(cell)
            for off in _LOCAL_NEIGHBORS_6:
                nxt = (cell[0] + off[0], cell[1] + off[1], cell[2] + off[2])
                if not (0 <= nxt[0] <= 1 and 0 <= nxt[1] <= 1 and 0 <= nxt[2] <= 1):
                    continue
                nxt_idx = _local_cell_index(nxt)
                if not (free_mask & (1 << nxt_idx)) or component_by_cell[nxt_idx] >= 0:
                    continue
                component_by_cell[nxt_idx] = component_id
                stack.append(nxt)
        component_faces[component_id] = faces
        component_id += 1

    return PathVoxelTopology(
        free_mask=int(free_mask),
        component_by_cell=tuple(component_by_cell),
        component_faces=component_faces,
    )


_TOPOLOGY_BY_FREE_MASK = tuple(_topology_from_free_mask(mask) for mask in range(256))


def _flat_voxel_index(grid: VoxelGrid, voxel: VoxelIndex) -> int | None:
    i, j, k = int(voxel[0]), int(voxel[1]), int(voxel[2])
    dims = grid.dims
    if i < 0 or j < 0 or k < 0 or i >= int(dims[0]) or j >= int(dims[1]) or k >= int(dims[2]):
        return None
    return i * int(dims[1]) * int(dims[2]) + j * int(dims[2]) + k


def _is_occupied(context: VoxelContext, voxel: VoxelIndex) -> bool:
    flat = _flat_voxel_index(context.voxel_grid, voxel)
    if flat is None:
        return True
    if context.occupied_lookup.size > flat:
        return bool(context.occupied_lookup[flat])
    return voxel in context.occupied_voxels


def build_path_context_from_occupancy(
    occupancy_context: VoxelContext,
    params: ReachabilityParams,
) -> PathContext:
    """Build a coarse path grid whose passability is checked against fine occupancy."""
    path_size = float(params.navigation_voxel_size_l or params.voxel_size_l)
    grid = VoxelGrid(
        min_corner=occupancy_context.voxel_grid.min_corner,
        max_corner=occupancy_context.voxel_grid.max_corner,
        voxel_size=max(path_size, 1e-6),
    )
    return PathContext(
        voxel_grid=grid,
        occupied_voxels=set(),
        occupied_lookup=np.zeros(int(np.prod(grid.dims)), dtype=bool),
        occupancy_context=occupancy_context,
        topology_by_voxel={},
        passable_voxels=set(),
        connectivity=26 if int(params.path_neighbor_connectivity) == 26 else 6,
    )


def _path_topology(context: PathContext, path_voxel: VoxelIndex) -> PathVoxelTopology:
    cache = context.topology_by_voxel
    if cache is None:
        cache = {}
        context.topology_by_voxel = cache
    cached = cache.get(path_voxel)
    if cached is not None:
        return cached

    occupancy_context = context.occupancy_context
    if occupancy_context is None:
        free_mask = 0 if path_voxel in context.occupied_voxels else 0xFF
    else:
        free_mask = 0
        base = (int(path_voxel[0]) * 2, int(path_voxel[1]) * 2, int(path_voxel[2]) * 2)
        dims = occupancy_context.voxel_grid.dims
        dx, dy, dz = int(dims[0]), int(dims[1]), int(dims[2])
        stride0 = dy * dz
        stride1 = dz
        lookup = occupancy_context.occupied_lookup
        for local_idx, local in enumerate(_LOCAL_CELLS_2):
            i = base[0] + local[0]
            j = base[1] + local[1]
            k = base[2] + local[2]
            if i < 0 or j < 0 or k < 0 or i >= dx or j >= dy or k >= dz:
                continue
            flat = i * stride0 + j * stride1 + k
            if lookup.size <= flat or not bool(lookup[flat]):
                free_mask |= 1 << local_idx

    topology = _TOPOLOGY_BY_FREE_MASK[int(free_mask)]
    cache[path_voxel] = topology
    return topology


def _offset_face_masks(offset: VoxelIndex) -> tuple[int, int]:
    src = 0
    dst = 0
    if offset[0] < 0:
        src |= FACE_NEG_X
        dst |= FACE_POS_X
    elif offset[0] > 0:
        src |= FACE_POS_X
        dst |= FACE_NEG_X
    if offset[1] < 0:
        src |= FACE_NEG_Y
        dst |= FACE_POS_Y
    elif offset[1] > 0:
        src |= FACE_POS_Y
        dst |= FACE_NEG_Y
    if offset[2] < 0:
        src |= FACE_NEG_Z
        dst |= FACE_POS_Z
    elif offset[2] > 0:
        src |= FACE_POS_Z
        dst |= FACE_NEG_Z
    return src, dst


def _has_component_touching(topology: PathVoxelTopology, faces: int) -> bool:
    if topology.free_mask == 0:
        return False
    for comp_faces in topology.component_faces.values():
        if (comp_faces & faces) == faces:
            return True
    return False


def _path_voxel_passable(context: PathContext, voxel: VoxelIndex) -> bool:
    topology = _path_topology(context, voxel)
    return topology.free_mask != 0 and bool(topology.component_faces)


def _path_can_traverse(context: PathContext, cur: VoxelIndex, nxt: VoxelIndex, offset: VoxelIndex) -> bool:
    src_faces, dst_faces = _offset_face_masks(offset)
    if src_faces == 0 or dst_faces == 0:
        return False
    return _has_component_touching(_path_topology(context, cur), src_faces) and _has_component_touching(
        _path_topology(context, nxt),
        dst_faces,
    )


def _path_neighbor_offsets(connectivity: int) -> tuple[VoxelIndex, ...]:
    if int(connectivity) == 6:
        return tuple(_LOCAL_NEIGHBORS_6)
    return tuple(neighbor_offsets_26())


def build_navigation_space(
    voxel_context: VoxelContext,
    user_position: Sequence[float],
    params: ReachabilityParams,
) -> NavigationState | None:
    """Restrict the full voxel grid to the locally reachable navigation region."""
    grid = voxel_context.voxel_grid
    occupied = voxel_context.occupied_voxels
    path_context = voxel_context if isinstance(voxel_context, PathContext) else None

    # Candidate navigation volume: one-segment reachable sphere around current user position.
    reach_radius = max(1e-6, float(params.v_max) * _prediction_horizon_s(params))
    candidates = set(enumerate_voxels_in_sphere(grid, user_position, reach_radius))
    if not candidates:
        start_only = grid.world_to_voxel(user_position, clamp=True)
        candidates = {start_only}

    if params.use_obstacles and path_context is not None:
        free = {v for v in candidates if _path_voxel_passable(path_context, v)}
    elif params.use_obstacles:
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
        path_context=path_context,
    )


def _unit(vec: Sequence[float]) -> np.ndarray:
    """Return a normalized 3D vector, or zeros for degenerate input."""
    arr = np.asarray(vec, dtype=float)
    n = float(np.linalg.norm(arr))
    if n <= 1e-12:
        return np.zeros(3, dtype=float)
    return arr / n


def _softmax_from_scores(scores: Dict[VoxelIndex, float], temperature: float = 1.0) -> Dict[VoxelIndex, float]:
    """Normalize voxel scores into a probability distribution."""
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


def compute_navigation_features(nav_state: NavigationState, params: ReachabilityParams) -> NavigationFeatures:
    """Compute obstacle-aware path distances/directions once for all intention hypotheses."""
    path_steps = _path_dir_steps(params)
    if params.use_obstacles:
        if nav_state.path_context is not None:
            path_context = nav_state.path_context
            dist_map, prev_map, path_dir_maps = dijkstra_connected(
                free_voxels=nav_state.free_voxels,
                start_voxel=nav_state.start_voxel,
                voxel_size=float(nav_state.voxel_grid.voxel_size),
                offsets=_path_neighbor_offsets(path_context.connectivity),
                can_traverse=lambda cur, nxt, off: _path_can_traverse(path_context, cur, nxt, off),
                k_steps=path_steps,
                voxel_to_center=nav_state.voxel_grid.voxel_to_world_center,
            )
        else:
            dist_map, prev_map, path_dir_maps = dijkstra_26_connected(
                free_voxels=nav_state.free_voxels,
                start_voxel=nav_state.start_voxel,
                voxel_size=float(nav_state.voxel_grid.voxel_size),
                k_steps=path_steps,
                voxel_to_center=nav_state.voxel_grid.voxel_to_world_center,
            )
        return NavigationFeatures(dist_map=dist_map, prev_map=prev_map, path_dir_maps=path_dir_maps)

    pos = nav_state.voxel_grid.voxel_to_world_center(nav_state.start_voxel)
    dist_map: Dict[VoxelIndex, float] = {}
    direct_path_dir_map: Dict[VoxelIndex, np.ndarray] = {}
    for vox in nav_state.free_voxels:
        center = nav_state.voxel_grid.voxel_to_world_center(vox)
        vec = center - pos
        dist_map[vox] = float(np.linalg.norm(vec))
        direct_path_dir_map[vox] = _unit(vec)
    return NavigationFeatures(
        dist_map=dist_map,
        prev_map={},
        path_dir_maps={path_k: direct_path_dir_map for path_k in path_steps},
    )


def _voxel_probabilities_from_features(
    nav_state: NavigationState,
    features: NavigationFeatures,
    intention: IntentionEstimate,
    user_position: Sequence[float],
    params: ReachabilityParams,
    path_k: int,
) -> Dict[VoxelIndex, float]:
    """Score reachable voxels for one intention estimate using shared path features."""
    if not features.dist_map:
        return {}

    d_max = max(float(params.v_max) * _prediction_horizon_s(params), 1e-6)
    lambda_t = float(np.clip(float(params.lambda_max) * intention.effective_confidence, 0.0, 1.0))

    move_dir = _unit(intention.move_dir)
    move_norm = float(np.linalg.norm(move_dir))

    scores: Dict[VoxelIndex, float] = {}
    resolved_path_k = _nearest_path_k(features.path_dir_maps, path_k)
    path_dir_map = features.path_dir_maps.get(resolved_path_k, {}) if resolved_path_k is not None else {}

    for vox, dist in features.dist_map.items():
        d_norm = float(np.clip(dist / d_max, 0.0, 1.0))
        path_dir = path_dir_map.get(vox)
        if path_dir is None and params.use_obstacles:
            path_dir = path_direction(
                prev_map=features.prev_map,
                start=nav_state.start_voxel,
                target=vox,
                k_steps=max(1, int(path_k)),
                voxel_to_center=nav_state.voxel_grid.voxel_to_world_center,
            )
        if path_dir is None or move_norm <= 1e-9 or np.linalg.norm(path_dir) <= 1e-9:
            align = 0.5
        else:
            align = (1.0 + float(np.clip(np.dot(move_dir, path_dir), -1.0, 1.0))) * 0.5
        score = -((1.0 - lambda_t) * d_norm) + (lambda_t * align)
        scores[vox] = float(score)

    return _softmax_from_scores(scores, temperature=float(params.softmax_temp))


def compute_voxel_probabilities(
    nav_state: NavigationState,
    intention: IntentionEstimate,
    user_position: Sequence[float],
    params: ReachabilityParams,
) -> Dict[VoxelIndex, float]:
    """Convert one intention estimate into probabilities over navigation voxels."""
    if not nav_state.free_voxels:
        return {}
    features = compute_navigation_features(nav_state, params)
    return _voxel_probabilities_from_features(
        nav_state=nav_state,
        features=features,
        intention=intention,
        user_position=user_position,
        params=params,
        path_k=_path_dir_steps(params)[0],
    )


def prune_voxel_probabilities(
    voxel_probs: Mapping[VoxelIndex, float],
    max_voxels_eval: int,
    min_voxel_prob: float,
) -> Dict[VoxelIndex, float]:
    """Trim the voxel distribution to the support worth evaluating with ray tests."""
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


def resolve_auto_pruning(
    voxel_probs: Mapping[VoxelIndex, float],
    active_tile_count: int,
    visibility_ray_budget: int,
    params: ReachabilityParams,
) -> tuple[int, float, Dict[str, float]]:
    """Resolve auto pruning settings from voxel mass and a ray-budget guardrail."""
    items = sorted(voxel_probs.items(), key=lambda kv: kv[1], reverse=True)
    if not items:
        return max(1, int(params.max_voxels_eval)), max(float(params.min_voxel_prob), 0.0), {}

    filtered = items
    if not params.auto_min_voxel_prob:
        threshold = max(0.0, float(params.min_voxel_prob))
        filtered = [kv for kv in filtered if kv[1] >= threshold]
        if not filtered:
            return max(1, int(params.max_voxels_eval)), threshold, {}

    mass_target = float(np.clip(float(params.auto_voxel_mass_target), 1e-6, 1.0))
    mass_count = len(filtered)
    cumulative = 0.0
    for idx, (_, prob) in enumerate(filtered, start=1):
        cumulative += float(prob)
        if cumulative >= mass_target:
            mass_count = idx
            break

    budget_cap = None
    if active_tile_count > 0:
        baseline_samples = max(1, int(params.auto_min_samples_per_tile))
        per_voxel_baseline_rays = int(active_tile_count) * baseline_samples
        if per_voxel_baseline_rays > 0:
            budget_cap = max(1, int(visibility_ray_budget) // per_voxel_baseline_rays)

    if params.auto_max_voxels_eval:
        resolved_cap = mass_count
    else:
        resolved_cap = max(1, int(params.max_voxels_eval))

    if budget_cap is not None:
        # The tile count can be very large for V-PCC content. A strict
        # tile_count * min_samples budget often collapses the reachable
        # distribution to one voxel, which makes the predictor behave like a
        # brittle point estimate. Keep a small probability frontier here and let
        # the tile sampler spend fewer samples per tile when the ray budget is
        # tight.
        min_frontier = min(len(filtered), max(8, int(np.ceil(np.sqrt(max(1, len(filtered)))))))
        resolved_cap = min(resolved_cap, max(int(budget_cap), min_frontier))
    resolved_cap = max(1, min(resolved_cap, len(filtered)))

    selected = filtered[:resolved_cap]
    resolved_min_prob = float(selected[-1][1]) if params.auto_min_voxel_prob and selected else max(float(params.min_voxel_prob), 0.0)
    selected_mass = float(sum(prob for _, prob in selected))
    stats = {
        "auto_prune_mass_target": mass_target,
        "auto_prune_mass_kept": selected_mass,
        "auto_prune_budget_cap": float(budget_cap) if budget_cap is not None else float(len(filtered)),
        "resolved_max_voxels_eval": float(resolved_cap),
        "resolved_min_voxel_prob": float(resolved_min_prob),
    }
    return resolved_cap, resolved_min_prob, stats


def _history_tail_by_seconds(pose_history, horizon_s: float, assumed_fps: float):
    if pose_history is None or len(pose_history) <= 1:
        return pose_history
    horizon_s = max(float(horizon_s), 1.0 / max(float(assumed_fps), 1e-6))
    time_columns = ("Time", "Time_s", "Timestamp", "time_s", "time")
    for col in time_columns:
        if hasattr(pose_history, "columns") and col in pose_history.columns:
            times = pose_history[col].to_numpy(dtype=float)
            if times.size and np.all(np.isfinite(times)):
                cutoff = float(times[-1]) - horizon_s
                return pose_history.loc[times >= cutoff]
    count = max(2, int(round(horizon_s * max(float(assumed_fps), 1e-6))))
    return pose_history.tail(count) if hasattr(pose_history, "tail") else pose_history


def _motion_consensus(intentions: Sequence[IntentionEstimate]) -> float:
    dirs = [_unit(intent.move_dir) for intent in intentions if np.linalg.norm(intent.move_dir) > 1e-9]
    if len(dirs) < 2:
        return 0.0
    vals = []
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            vals.append(max(0.0, float(np.clip(np.dot(dirs[i], dirs[j]), -1.0, 1.0))))
    return float(np.mean(vals)) if vals else 0.0


def estimate_multi_horizon_intentions(
    pose_now,
    pose_history,
    params: ReachabilityParams,
) -> tuple[list[HorizonIntention], Dict[str, float]]:
    """Estimate motion intent over several horizons and damp direction when they disagree."""
    horizons = _motion_horizons_s(params)

    intentions: list[IntentionEstimate] = []
    for horizon_s in horizons:
        window = _history_tail_by_seconds(pose_history, horizon_s=horizon_s, assumed_fps=params.assumed_fps)
        intentions.append(
            estimate_intention(
                pose_now=pose_now,
                pose_history=window,
                max_speed=params.max_speed,
                max_acceleration=params.max_acceleration,
                stability_weight=params.stability_weight,
                mismatch_penalty_strength=params.mismatch_penalty_strength,
                assumed_fps=params.assumed_fps,
                smooth_window=params.smooth_window,
                angle_convention=params.angle_convention,
            )
        )

    consensus = _motion_consensus(intentions)
    confidence_scale = 0.5 + 0.5 * consensus
    adjusted = [
        replace(intent, effective_confidence=float(np.clip(intent.effective_confidence * confidence_scale, 0.0, 1.0)))
        for intent in intentions
    ]
    path_ks = [_path_k_for_horizon(horizon_s, params) for horizon_s in horizons]
    horizon_intentions = [
        HorizonIntention(horizon_s=float(horizon_s), path_k=int(path_k), estimate=estimate)
        for horizon_s, path_k, estimate in zip(horizons, path_ks, adjusted)
    ]
    confidences = np.asarray([intent.confidence for intent in adjusted], dtype=float)
    effective = np.asarray([intent.effective_confidence for intent in adjusted], dtype=float)
    mismatches = np.asarray([intent.mismatch for intent in adjusted], dtype=float)
    speeds = np.asarray([intent.normalized_speed for intent in adjusted], dtype=float)
    accels = np.asarray([intent.normalized_acceleration for intent in adjusted], dtype=float)
    intensities = np.asarray([intent.motion_intensity for intent in adjusted], dtype=float)
    stabilities = np.asarray([intent.motion_stability for intent in adjusted], dtype=float)
    reliabilities = np.asarray([intent.direction_reliability for intent in adjusted], dtype=float)
    stats = {
        "intention_horizon_count": float(len(adjusted)),
        "path_dir_step_count": float(len(set(path_ks))),
        "intention_motion_consensus": float(consensus),
        "intention_confidence_scale": float(confidence_scale),
        "intention_confidence_mean": float(np.mean(confidences)) if confidences.size else 0.0,
        "intention_effective_confidence_mean": float(np.mean(effective)) if effective.size else 0.0,
        "intention_mismatch_mean": float(np.mean(mismatches)) if mismatches.size else 0.0,
        "normalized_speed": float(np.mean(speeds)) if speeds.size else 0.0,
        "normalized_acceleration": float(np.mean(accels)) if accels.size else 0.0,
        "motion_intensity": float(np.mean(intensities)) if intensities.size else 0.0,
        "motion_stability": float(np.mean(stabilities)) if stabilities.size else 0.0,
        "direction_reliability": float(np.mean(reliabilities)) if reliabilities.size else 0.0,
    }
    for idx, horizon_s in enumerate(horizons):
        if idx >= len(adjusted):
            continue
        stats[f"intention_horizon_{idx}_s"] = float(horizon_s)
        stats[f"intention_horizon_{idx}_path_k"] = float(path_ks[idx])
        stats[f"intention_horizon_{idx}_confidence"] = float(adjusted[idx].confidence)
        stats[f"intention_horizon_{idx}_effective_confidence"] = float(adjusted[idx].effective_confidence)
        stats[f"intention_horizon_{idx}_mismatch"] = float(adjusted[idx].mismatch)
    return horizon_intentions, stats


def _visual_horizon_debug(
    intentions: Sequence[HorizonIntention],
    user_position: Sequence[float],
    nav_state: NavigationState,
    params: ReachabilityParams,
) -> Dict[str, Any]:
    """Build compact per-horizon motion vectors for visual debugging."""
    origin = [float(v) for v in np.asarray(user_position, dtype=float)]
    sorted_free = sorted(nav_state.free_voxels)
    sample_cap = 1200
    step = max(1, int(np.ceil(len(sorted_free) / sample_cap))) if sorted_free else 1
    sampled_free = sorted_free[::step][:sample_cap]
    horizons = []
    for idx, horizon in enumerate(intentions):
        estimate = horizon.estimate
        direction = _unit(estimate.move_dir)
        distance_m = max(float(params.v_max), 1e-6) * max(float(horizon.horizon_s), 1e-6)
        endpoint = (np.asarray(origin, dtype=float) + direction * distance_m).tolist()
        horizons.append(
            {
                "idx": int(idx),
                "horizon_s": float(horizon.horizon_s),
                "path_k": int(horizon.path_k),
                "direction": [float(v) for v in direction],
                "latest_velocity_direction": [float(v) for v in _unit(estimate.latest_velocity_dir)],
                "origin": origin,
                "endpoint": [float(v) for v in endpoint],
                "distance_m": float(distance_m),
                "confidence": float(estimate.confidence),
                "effective_confidence": float(estimate.effective_confidence),
                "mismatch": float(estimate.mismatch),
                "normalized_speed": float(estimate.normalized_speed),
                "direction_reliability": float(estimate.direction_reliability),
            }
        )
    return {
        "origin": origin,
        "horizons": horizons,
        "reachable": {
            "center": origin,
            "radius_m": float(nav_state.reach_radius),
            "horizon_s": float(_prediction_horizon_s(params)),
            "max_speed_m_s": float(params.v_max),
            "candidate_voxels": int(len(nav_state.candidate_voxels)),
            "free_voxels": int(len(nav_state.free_voxels)),
            "start_voxel": [int(v) for v in nav_state.start_voxel],
            "sampled_free_voxel_count": int(len(sampled_free)),
            "sampled_free_voxels": [
                [float(v) for v in nav_state.voxel_grid.voxel_to_world_center(vox)]
                for vox in sampled_free
            ],
        },
    }


def mix_voxel_probabilities(
    nav_state: NavigationState,
    features: NavigationFeatures,
    intentions: Sequence[HorizonIntention],
    user_position: Sequence[float],
    params: ReachabilityParams,
) -> Dict[VoxelIndex, float]:
    """Mix one voxel distribution per intention horizon."""
    if not intentions:
        return {}
    raw_weights = np.asarray([max(0.05, float(intent.estimate.confidence)) for intent in intentions], dtype=float)
    denom = float(np.sum(raw_weights))
    if denom <= 1e-12:
        raw_weights = np.ones(len(intentions), dtype=float)
        denom = float(len(intentions))
    weights = raw_weights / denom
    mixed: Dict[VoxelIndex, float] = {}
    for weight, horizon_intention in zip(weights, intentions):
        part = _voxel_probabilities_from_features(
            nav_state=nav_state,
            features=features,
            intention=horizon_intention.estimate,
            user_position=user_position,
            params=params,
            path_k=horizon_intention.path_k,
        )
        for vox, prob in part.items():
            mixed[vox] = mixed.get(vox, 0.0) + float(weight) * float(prob)
    total = float(sum(mixed.values()))
    if total <= 1e-12:
        return {}
    return {vox: prob / total for vox, prob in mixed.items()}


def origin_distribution(
    nav_state: NavigationState,
    features: NavigationFeatures,
    intentions: Sequence[HorizonIntention],
    user_position: Sequence[float],
    params: ReachabilityParams,
) -> tuple[Dict[VoxelIndex, float], str]:
    """Return the ray-origin distribution selected for the configured ablation.

    ``current_position`` keeps the geometry-aware visibility test but evaluates
    it only from the latest pose. ``uniform_reachable`` retains physical
    reachability while removing motion, gaze, and path-probability weighting.
    The default uses the full multi-horizon motion-weighted distribution.
    """
    mode = str(params.origin_distribution_mode or "motion_weighted").strip().lower()
    if mode == "current_position":
        return {nav_state.start_voxel: 1.0}, mode
    if mode == "uniform_reachable":
        voxels = sorted(nav_state.free_voxels)
        if not voxels:
            return {}, mode
        weight = 1.0 / float(len(voxels))
        return {vox: weight for vox in voxels}, mode
    return mix_voxel_probabilities(
        nav_state=nav_state,
        features=features,
        intentions=intentions,
        user_position=user_position,
        params=params,
    ), "motion_weighted"


def map_navigation_probs_to_visibility_grid(
    nav_probs: Mapping[VoxelIndex, float],
    navigation_grid: VoxelGrid,
    visibility_grid: VoxelGrid,
) -> tuple[Dict[VoxelIndex, float], Dict[VoxelIndex, np.ndarray], Dict[VoxelIndex, VoxelIndex], Dict[str, float]]:
    """Map navigation-grid probability mass to visibility-grid start voxels."""
    mapped: Dict[VoxelIndex, float] = {}
    weighted_centers: Dict[VoxelIndex, np.ndarray] = {}
    center_masses: Dict[VoxelIndex, float] = {}
    representative_nav_voxels: Dict[VoxelIndex, VoxelIndex] = {}
    representative_nav_probs: Dict[VoxelIndex, float] = {}
    collisions = 0
    for nav_vox, prob in nav_probs.items():
        world = navigation_grid.voxel_to_world_center(nav_vox)
        vis_vox = visibility_grid.world_to_voxel(world, clamp=True)
        prob_f = float(prob)
        if vis_vox in mapped:
            collisions += 1
        mapped[vis_vox] = mapped.get(vis_vox, 0.0) + prob_f
        weighted_centers[vis_vox] = weighted_centers.get(vis_vox, np.zeros(3, dtype=float)) + (prob_f * world)
        center_masses[vis_vox] = center_masses.get(vis_vox, 0.0) + prob_f
        if prob_f > representative_nav_probs.get(vis_vox, -1.0):
            representative_nav_probs[vis_vox] = prob_f
            representative_nav_voxels[vis_vox] = nav_vox
    total = float(sum(mapped.values()))
    if total > 1e-12:
        mapped = {vox: prob / total for vox, prob in mapped.items()}
    centers: Dict[VoxelIndex, np.ndarray] = {}
    origin_offsets = []
    for vox, weighted_center in weighted_centers.items():
        mass = float(center_masses.get(vox, 0.0))
        if mass <= 1e-12:
            center = visibility_grid.voxel_to_world_center(vox)
        else:
            center = weighted_center / mass
        centers[vox] = np.asarray(center, dtype=float)
        origin_offsets.append(float(np.linalg.norm(centers[vox] - visibility_grid.voxel_to_world_center(vox))))
    return mapped, centers, representative_nav_voxels, {
        "navigation_to_visibility_voxels": float(len(mapped)),
        "navigation_to_visibility_collisions": float(collisions),
        "navigation_to_visibility_origin_offset_mean": float(np.mean(origin_offsets)) if origin_offsets else 0.0,
        "navigation_to_visibility_origin_offset_max": float(np.max(origin_offsets)) if origin_offsets else 0.0,
    }


def _path_points_for_voxel(
    target: VoxelIndex,
    start: VoxelIndex,
    prev_map: Mapping[VoxelIndex, VoxelIndex | None],
    voxel_grid: VoxelGrid,
    max_points: int = 256,
) -> list[list[float]]:
    """Return the Dijkstra predecessor path from start to target as world centers."""
    if target == start:
        center = voxel_grid.voxel_to_world_center(start)
        return [[float(v) for v in center]]
    if target not in prev_map:
        return []
    path = [target]
    seen = {target}
    current = target
    for _ in range(max(1, int(max_points))):
        previous = prev_map.get(current)
        if previous is None:
            break
        if previous in seen:
            break
        path.append(previous)
        seen.add(previous)
        current = previous
        if current == start:
            break
    if path[-1] != start:
        return []
    path.reverse()
    if len(path) > max_points:
        step = max(1, int(math.ceil(len(path) / float(max_points))))
        path = path[::step]
        if path[-1] != target:
            path.append(target)
    return [[float(v) for v in voxel_grid.voxel_to_world_center(vox)] for vox in path]


def compute_tile_importance(
    tiles,
    points_xyz: np.ndarray,
    pose_now,
    pose_history,
    params: ReachabilityParams,
    occupancy_context: VoxelContext | None = None,
    navigation_context: PathContext | VoxelContext | None = None,
    visibility_context: VoxelContext | None = None,
    tile_samples: Mapping[str, np.ndarray] | None = None,
    latency_session=None,
) -> tuple[Dict[str, float], Dict[str, Any]]:
    """Run the full pipeline and return independent tile scores plus debug metadata."""
    if points_xyz.size == 0 or not tiles:
        return uniform_tile_distribution(tiles), {"reason": "empty_points_or_tiles"}

    try:
        user_position = extract_position(pose_now)
    except Exception:
        return uniform_tile_distribution(tiles), {"reason": "missing_pose"}

    stage = latency_session.stage if latency_session is not None else None

    path_voxel_size = float(params.navigation_voxel_size_l or params.voxel_size_l)
    occupancy_voxel_size = float(params.occupancy_voxel_size_l or (path_voxel_size * 0.5))
    occ_params = replace(params, voxel_size_l=occupancy_voxel_size)
    nav_params = replace(params, voxel_size_l=path_voxel_size, navigation_voxel_size_l=path_voxel_size)
    vis_params = replace(params, voxel_size_l=float(params.visibility_voxel_size_l or params.voxel_size_l))
    if params.use_obstacles:
        local_occupancy_context = occupancy_context or build_voxel_context(points_xyz, occ_params)
        local_navigation_context = navigation_context or build_path_context_from_occupancy(local_occupancy_context, nav_params)
    else:
        local_navigation_context = navigation_context or build_unobstructed_navigation_context(points_xyz, nav_params)
        # Keep the existing statistics contract while making it explicit that
        # reachability used an empty (unobstructed) occupancy representation.
        local_occupancy_context = local_navigation_context
    local_visibility_context = visibility_context or build_voxel_context(points_xyz, vis_params)
    if stage is None:
        nav_state = build_navigation_space(local_navigation_context, user_position=user_position, params=nav_params)
    else:
        with stage("enumerate_navigation_space"):
            nav_state = build_navigation_space(local_navigation_context, user_position=user_position, params=nav_params)
    if nav_state is None:
        return uniform_tile_distribution(tiles), {"reason": "no_navigation_state"}

    sigma = float(params.sigma_g) if params.sigma_g is not None else float(params.r_vis) * 0.5
    sigma = max(sigma, 1e-6)
    auto_visibility = bool(
        params.auto_samples_per_tile or params.auto_max_voxels_eval or params.auto_min_voxel_prob
    )
    gate_by_tile = {}
    active_tile_point_sets = None
    gate_debug_stats: Dict[str, float] = {}
    gate_debug_details: list[dict] = []
    if auto_visibility:
        gate_by_tile, _distance_by_tile, gate_debug_stats, gate_debug_details = summarize_tile_gates(
            tiles=tiles,
            user_position=user_position,
            r_vis=float(params.r_vis),
            sigma_g=sigma,
        )
        active_tile_ids = {tile_id for tile_id, gate in gate_by_tile.items() if gate > 0.0}
        active_tile_count = len(active_tile_ids)
        if params.auto_samples_per_tile:
            active_tiles = [tile for tile in tiles if str(tile.unique_id) in active_tile_ids]
            if stage is None:
                active_tile_point_sets = collect_points_per_tile(points_xyz=points_xyz, tiles=active_tiles)
            else:
                with stage("collect_active_tile_points"):
                    active_tile_point_sets = collect_points_per_tile(points_xyz=points_xyz, tiles=active_tiles)
    else:
        active_tile_count = 0
    visibility_ms_per_ray = max(float(params.auto_visibility_ms_per_ray), 1e-6)
    visibility_ray_budget = max(1, int(float(params.auto_visibility_time_budget_ms) / visibility_ms_per_ray))

    # Stage A: infer motion/gaze intention confidence from multiple pose-history horizons.
    if stage is None:
        intentions, intention_stats = estimate_multi_horizon_intentions(
            pose_now=pose_now,
            pose_history=pose_history,
            params=params,
        )
    else:
        with stage("estimate_intention"):
            intentions, intention_stats = estimate_multi_horizon_intentions(
                pose_now=pose_now,
                pose_history=pose_history,
                params=params,
            )
    visual_intention_debug = (
        _visual_horizon_debug(intentions=intentions, user_position=user_position, nav_state=nav_state, params=nav_params)
        if bool(params.visual_debug_enabled)
        else None
    )

    # Stage B: choose ray origins. The two ablations deliberately remove either
    # the motion weighting or all future-position expansion, while retaining
    # the same navigation and visibility geometry.
    if stage is None:
        nav_features = compute_navigation_features(nav_state, nav_params)
        navigation_voxel_probs, resolved_origin_distribution_mode = origin_distribution(
            nav_state=nav_state,
            features=nav_features,
            intentions=intentions,
            user_position=user_position,
            params=nav_params,
        )
    else:
        with stage("compute_voxel_probabilities"):
            nav_features = compute_navigation_features(nav_state, nav_params)
            navigation_voxel_probs, resolved_origin_distribution_mode = origin_distribution(
                nav_state=nav_state,
                features=nav_features,
                intentions=intentions,
                user_position=user_position,
                params=nav_params,
            )
    voxel_probs, voxel_world_centers, representative_nav_voxels, mapping_stats = map_navigation_probs_to_visibility_grid(
        nav_probs=navigation_voxel_probs,
        navigation_grid=nav_state.voxel_grid,
        visibility_grid=local_visibility_context.voxel_grid,
    )
    prune_stats: Dict[str, float] = {}
    if params.auto_max_voxels_eval or params.auto_min_voxel_prob:
        resolved_max_voxels_eval, resolved_min_voxel_prob, prune_stats = resolve_auto_pruning(
            voxel_probs=voxel_probs,
            active_tile_count=active_tile_count,
            visibility_ray_budget=visibility_ray_budget,
            params=params,
        )
    else:
        resolved_max_voxels_eval = params.max_voxels_eval
        resolved_min_voxel_prob = params.min_voxel_prob

    if stage is None:
        kept_voxel_probs = prune_voxel_probabilities(
            voxel_probs,
            max_voxels_eval=resolved_max_voxels_eval,
            min_voxel_prob=resolved_min_voxel_prob,
        )
    else:
        with stage("prune_voxel_probabilities"):
            kept_voxel_probs = prune_voxel_probabilities(
                voxel_probs,
                max_voxels_eval=resolved_max_voxels_eval,
                min_voxel_prob=resolved_min_voxel_prob,
            )
    if not kept_voxel_probs:
        return uniform_tile_distribution(tiles), {"reason": "no_voxel_probs"}
    kept_voxel_world_centers = {
        vox: voxel_world_centers.get(vox, local_visibility_context.voxel_grid.voxel_to_world_center(vox))
        for vox in kept_voxel_probs.keys()
    }
    kept_voxel_paths = {
        vox: _path_points_for_voxel(
            target=representative_nav_voxels[vox],
            start=nav_state.start_voxel,
            prev_map=nav_features.prev_map,
            voxel_grid=nav_state.voxel_grid,
        )
        for vox in kept_voxel_probs.keys()
        if vox in representative_nav_voxels
    }

    # Stage C: sample representative tile points used for ray visibility checks.
    sample_stats: Dict[str, float] = {}
    if params.auto_samples_per_tile:
        local_tile_point_sets = active_tile_point_sets or {}
        voxel_cost_per_sample = len(kept_voxel_probs)
        candidate_cap = params.max_candidate_voxels if params.max_candidate_voxels is not None else params.max_voxels_per_tile
        if candidate_cap is not None:
            voxel_cost_per_sample = min(voxel_cost_per_sample, max(1, int(candidate_cap)))
        if stage is None:
            sample_counts, sample_stats = allocate_samples_per_tile(
                tile_point_sets=local_tile_point_sets,
                tile_weights=gate_by_tile,
                kept_voxel_count=voxel_cost_per_sample,
                ray_budget=visibility_ray_budget,
                target_confidence=float(params.auto_tile_confidence),
                support_points=max(1, int(params.auto_tile_support_points)),
                min_samples=max(1, int(params.auto_min_samples_per_tile)),
                max_samples=max(max(1, int(params.auto_min_samples_per_tile)), int(params.auto_max_samples_per_tile)),
            )
            local_tile_samples = sample_precollected_points_per_tile(
                tile_point_sets=active_tile_point_sets or {},
                tiles=tiles,
                samples_per_tile=sample_counts,
                rng_seed=params.rng_seed,
            )
        else:
            with stage("allocate_tile_samples"):
                sample_counts, sample_stats = allocate_samples_per_tile(
                    tile_point_sets=local_tile_point_sets,
                    tile_weights=gate_by_tile,
                    kept_voxel_count=voxel_cost_per_sample,
                    ray_budget=visibility_ray_budget,
                    target_confidence=float(params.auto_tile_confidence),
                    support_points=max(1, int(params.auto_tile_support_points)),
                    min_samples=max(1, int(params.auto_min_samples_per_tile)),
                    max_samples=max(max(1, int(params.auto_min_samples_per_tile)), int(params.auto_max_samples_per_tile)),
                )
            with stage("sample_tile_points"):
                local_tile_samples = sample_precollected_points_per_tile(
                    tile_point_sets=active_tile_point_sets or {},
                    tiles=tiles,
                    samples_per_tile=sample_counts,
                    rng_seed=params.rng_seed,
                )
    else:
        local_tile_samples = tile_samples or sample_points_per_tile(
            points_xyz=points_xyz,
            tiles=tiles,
            samples_per_tile=params.samples_per_tile,
            rng_seed=params.rng_seed,
        )

    # Stage D: aggregate voxel probabilities and visibility into per-tile importance.
    visibility_started = perf_counter()
    if stage is None:
        tile_scores, visibility_stats = aggregate_tile_importance(
            tiles=tiles,
            voxel_probabilities=kept_voxel_probs,
            tile_samples=local_tile_samples,
            occupied_set=local_visibility_context.occupied_voxels,
            occupied_lookup=local_visibility_context.occupied_lookup,
            voxel_grid=local_visibility_context.voxel_grid,
            user_position=user_position,
            r_vis=float(params.r_vis),
            sigma_g=sigma,
            max_rays_per_tile=params.max_rays_per_tile,
            visibility_floor=float(params.visibility_floor),
            max_voxels_per_tile=params.max_voxels_per_tile,
            max_candidate_voxels=params.max_candidate_voxels,
            voxel_selection_mode=params.voxel_selection_mode,
            voxel_selection_core_fraction=float(params.voxel_selection_core_fraction),
            aggregation_mode=str(params.aggregation_mode or "expected_visibility"),
            source_presence_sigma_l=params.source_presence_sigma_l,
            source_score_tau=float(params.source_score_tau),
            source_min_candidates=int(params.source_min_candidates),
            propagation_factor=float(params.propagation_factor),
            propagation_visibility_threshold=float(params.propagation_visibility_threshold),
            raycast_processes=int(params.raycast_processes),
            voxel_world_centers=kept_voxel_world_centers,
            voxel_navigation_paths=kept_voxel_paths,
            visual_debug={
                "enabled": bool(params.visual_debug_enabled),
                "max_tiles": int(params.visual_debug_max_tiles),
                "max_voxels": int(params.visual_debug_max_voxels),
                "max_contributors_per_tile": int(params.visual_debug_max_contributors_per_tile),
                "max_rays_per_contributor": int(params.visual_debug_max_rays_per_contributor),
                "trace_voxels": bool(params.visual_debug_trace_voxels),
            },
        )
    else:
        with stage("aggregate_tile_importance"):
            tile_scores, visibility_stats = aggregate_tile_importance(
                tiles=tiles,
                voxel_probabilities=kept_voxel_probs,
                tile_samples=local_tile_samples,
                occupied_set=local_visibility_context.occupied_voxels,
                occupied_lookup=local_visibility_context.occupied_lookup,
                voxel_grid=local_visibility_context.voxel_grid,
                user_position=user_position,
                r_vis=float(params.r_vis),
                sigma_g=sigma,
                max_rays_per_tile=params.max_rays_per_tile,
                visibility_floor=float(params.visibility_floor),
                max_voxels_per_tile=params.max_voxels_per_tile,
                max_candidate_voxels=params.max_candidate_voxels,
                voxel_selection_mode=params.voxel_selection_mode,
                voxel_selection_core_fraction=float(params.voxel_selection_core_fraction),
                aggregation_mode=str(params.aggregation_mode or "expected_visibility"),
                source_presence_sigma_l=params.source_presence_sigma_l,
                source_score_tau=float(params.source_score_tau),
                source_min_candidates=int(params.source_min_candidates),
                propagation_factor=float(params.propagation_factor),
                propagation_visibility_threshold=float(params.propagation_visibility_threshold),
                raycast_processes=int(params.raycast_processes),
                voxel_world_centers=kept_voxel_world_centers,
                voxel_navigation_paths=kept_voxel_paths,
                visual_debug={
                    "enabled": bool(params.visual_debug_enabled),
                    "max_tiles": int(params.visual_debug_max_tiles),
                    "max_voxels": int(params.visual_debug_max_voxels),
                    "max_contributors_per_tile": int(params.visual_debug_max_contributors_per_tile),
                    "max_rays_per_contributor": int(params.visual_debug_max_rays_per_contributor),
                    "trace_voxels": bool(params.visual_debug_trace_voxels),
                },
            )
    visibility_ms = (perf_counter() - visibility_started) * 1000.0

    p_vals = np.asarray(list(kept_voxel_probs.values()), dtype=float)
    t_vals = np.asarray(list(tile_scores.values()), dtype=float) if tile_scores else np.asarray([], dtype=float)
    debug_stats = {
        "occupied_voxels": float(len(local_visibility_context.occupied_voxels)),
        "candidate_voxels": float(len(nav_state.candidate_voxels)),
        "free_voxels": float(len(nav_state.free_voxels)),
        "kept_voxels": float(len(kept_voxel_probs)),
        "occupancy_voxel_size_l": float(local_occupancy_context.voxel_grid.voxel_size),
        "navigation_voxel_size_l": float(nav_state.voxel_grid.voxel_size),
        "visibility_voxel_size_l": float(local_visibility_context.voxel_grid.voxel_size),
        "path_neighbor_connectivity": float(getattr(local_navigation_context, "connectivity", params.path_neighbor_connectivity)),
        "path_topology_cache_size": float(len(getattr(local_navigation_context, "topology_by_voxel", {}) or {})),
        "occupancy_grid_dim_x": float(local_occupancy_context.voxel_grid.dims[0]),
        "occupancy_grid_dim_y": float(local_occupancy_context.voxel_grid.dims[1]),
        "occupancy_grid_dim_z": float(local_occupancy_context.voxel_grid.dims[2]),
        "occupancy_grid_total": float(np.prod(local_occupancy_context.voxel_grid.dims)),
        "navigation_grid_dim_x": float(nav_state.voxel_grid.dims[0]),
        "navigation_grid_dim_y": float(nav_state.voxel_grid.dims[1]),
        "navigation_grid_dim_z": float(nav_state.voxel_grid.dims[2]),
        "navigation_grid_total": float(np.prod(nav_state.voxel_grid.dims)),
        "visibility_grid_dim_x": float(local_visibility_context.voxel_grid.dims[0]),
        "visibility_grid_dim_y": float(local_visibility_context.voxel_grid.dims[1]),
        "visibility_grid_dim_z": float(local_visibility_context.voxel_grid.dims[2]),
        "visibility_grid_total": float(np.prod(local_visibility_context.voxel_grid.dims)),
        "voxel_prob_min": float(np.min(p_vals)) if p_vals.size else 0.0,
        "voxel_prob_mean": float(np.mean(p_vals)) if p_vals.size else 0.0,
        "voxel_prob_max": float(np.max(p_vals)) if p_vals.size else 0.0,
        "tile_nonzero_fraction": float(np.mean(t_vals > 0.0)) if t_vals.size else 0.0,
        "visibility_floor": float(params.visibility_floor),
        "motion_confidence": float(intention_stats.get("intention_confidence_mean", 0.0)),
        "c_eff": float(intention_stats.get("intention_effective_confidence_mean", 0.0)),
        "motion_mismatch": float(intention_stats.get("intention_mismatch_mean", 0.0)),
    }
    debug_stats.update(mapping_stats)
    debug_stats.update(intention_stats)
    debug_stats.update(prune_stats)
    debug_stats.update(sample_stats)
    debug_stats.update(gate_debug_stats)
    debug_stats.update(visibility_stats)
    debug_stats["origin_distribution_mode"] = resolved_origin_distribution_mode
    if visual_intention_debug is not None:
        debug_stats["_visual_intentions"] = visual_intention_debug
    debug_stats["visibility_ray_budget"] = float(visibility_ray_budget)
    debug_stats["aggregate_tile_importance_ms"] = float(visibility_ms)
    debug_stats["auto_visibility_time_budget_ms"] = float(params.auto_visibility_time_budget_ms)
    debug_stats["auto_visibility_ms_per_ray"] = float(params.auto_visibility_ms_per_ray)
    if gate_debug_details:
        debug_stats["tile_gate_details"] = gate_debug_details[: min(len(gate_debug_details), 12)]
        if gate_debug_stats.get("active_tiles", 0.0) <= 0.0:
            debug_stats["gating_reason"] = "all_tiles_beyond_r_vis"

    return tile_scores, debug_stats
