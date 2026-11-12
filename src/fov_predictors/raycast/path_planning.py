"""Shortest-path helpers over the local reachable voxel graph.

The reachability pipeline uses this module only when obstacle-aware navigation
is enabled. It computes voxel travel costs and extracts an initial path
direction so movement intention can be compared against feasible paths.
"""

from __future__ import annotations

import heapq
import math
from typing import Callable, Dict, Iterable, Sequence, Set, Tuple

import numpy as np

from .voxel_grid import VoxelIndex, neighbor_offsets_26


def _normalize_k_steps(k_steps: int | Iterable[int] | None) -> tuple[int, ...]:
    if k_steps is None:
        return (1,)
    if isinstance(k_steps, int):
        raw_values = [k_steps]
    else:
        raw_values = list(k_steps)
    values = sorted({max(1, int(value)) for value in raw_values})
    return tuple(values or [1])


def dijkstra_connected(
    free_voxels: Set[VoxelIndex],
    start_voxel: VoxelIndex,
    voxel_size: float,
    offsets: Iterable[VoxelIndex],
    can_traverse: Callable[[VoxelIndex, VoxelIndex, VoxelIndex], bool] | None = None,
    k_steps: int | Iterable[int] | None = None,
    voxel_to_center: Callable[[Sequence[int]], np.ndarray] | None = None,
) -> tuple[Dict[VoxelIndex, float], Dict[VoxelIndex, VoxelIndex], Dict[int, Dict[VoxelIndex, np.ndarray]]]:
    """Compute shortest travel distances over a caller-defined voxel graph."""
    if start_voxel not in free_voxels:
        return {}, {}, {}

    offsets = tuple(offsets)
    edge_cost = {
        off: float(voxel_size * math.sqrt(off[0] ** 2 + off[1] ** 2 + off[2] ** 2))
        for off in offsets
    }
    step_limits = _normalize_k_steps(k_steps)

    dist: Dict[VoxelIndex, float] = {start_voxel: 0.0}
    prev: Dict[VoxelIndex, VoxelIndex] = {}
    depth: Dict[VoxelIndex, int] = {start_voxel: 0}
    prefix_endpoint_by_k: Dict[int, Dict[VoxelIndex, VoxelIndex]] = {
        k: {start_voxel: start_voxel}
        for k in step_limits
    }
    heap: list[tuple[float, VoxelIndex]] = [(0.0, start_voxel)]

    while heap:
        d_cur, cur = heapq.heappop(heap)
        if d_cur > dist.get(cur, float("inf")):
            continue

        for off in offsets:
            nxt = (cur[0] + off[0], cur[1] + off[1], cur[2] + off[2])
            if nxt not in free_voxels:
                continue
            if can_traverse is not None and not can_traverse(cur, nxt, off):
                continue
            nd = d_cur + edge_cost[off]
            if nd < dist.get(nxt, float("inf")):
                dist[nxt] = nd
                prev[nxt] = cur
                cur_depth = depth[cur]
                nxt_depth = cur_depth + 1
                depth[nxt] = nxt_depth
                for step_limit in step_limits:
                    prefix_endpoint = prefix_endpoint_by_k[step_limit]
                    if nxt_depth <= step_limit:
                        prefix_endpoint[nxt] = nxt
                    else:
                        prefix_endpoint[nxt] = prefix_endpoint[cur]
                heapq.heappush(heap, (nd, nxt))

    if voxel_to_center is None:
        path_dir_maps = {
            step_limit: {start_voxel: np.zeros(3, dtype=float)}
            for step_limit in step_limits
        }
    else:
        start_center = voxel_to_center(start_voxel)
        path_dir_maps: Dict[int, Dict[VoxelIndex, np.ndarray]] = {}
        for step_limit, prefix_endpoint in prefix_endpoint_by_k.items():
            path_dirs: Dict[VoxelIndex, np.ndarray] = {start_voxel: np.zeros(3, dtype=float)}
            for vox, endpoint in prefix_endpoint.items():
                if vox == start_voxel:
                    continue
                path_dirs[vox] = _unit(voxel_to_center(endpoint) - start_center)
            path_dir_maps[step_limit] = path_dirs

    return dist, prev, path_dir_maps


def dijkstra_26_connected(
    free_voxels: Set[VoxelIndex],
    start_voxel: VoxelIndex,
    voxel_size: float,
    k_steps: int | Iterable[int] | None = None,
    voxel_to_center: Callable[[Sequence[int]], np.ndarray] | None = None,
) -> tuple[Dict[VoxelIndex, float], Dict[VoxelIndex, VoxelIndex], Dict[int, Dict[VoxelIndex, np.ndarray]]]:
    """Compute shortest travel distances and early path directions across the free voxel graph."""
    return dijkstra_connected(
        free_voxels=free_voxels,
        start_voxel=start_voxel,
        voxel_size=voxel_size,
        offsets=neighbor_offsets_26(),
        k_steps=k_steps,
        voxel_to_center=voxel_to_center,
    )


def _unit(vec: np.ndarray) -> np.ndarray:
    """Return a normalized vector, or zeros for degenerate input."""
    n = float(np.linalg.norm(vec))
    if n <= 1e-12:
        return np.zeros(3, dtype=float)
    return vec / n


def path_direction(
    prev_map: Dict[VoxelIndex, VoxelIndex],
    start: VoxelIndex,
    target: VoxelIndex,
    k_steps: int,
    voxel_to_center: Callable[[Sequence[int]], np.ndarray],
) -> np.ndarray:
    """Approximate the early movement direction along the path to a target voxel."""
    if target == start:
        return np.zeros(3, dtype=float)

    # Reconstruct path target -> start.
    back_path = [target]
    cur = target
    while cur != start:
        parent = prev_map.get(cur)
        if parent is None:
            # Fall back to direct geometric direction if predecessor chain is missing.
            return _unit(voxel_to_center(target) - voxel_to_center(start))
        back_path.append(parent)
        cur = parent

    path = list(reversed(back_path))  # start -> target
    if len(path) < 2:
        return np.zeros(3, dtype=float)

    num_steps = max(1, min(int(k_steps), len(path) - 1))
    disp = np.zeros(3, dtype=float)
    for i in range(num_steps):
        c0 = voxel_to_center(path[i])
        c1 = voxel_to_center(path[i + 1])
        disp += (c1 - c0)

    if float(np.linalg.norm(disp)) <= 1e-12:
        disp = voxel_to_center(target) - voxel_to_center(start)
    return _unit(disp)
