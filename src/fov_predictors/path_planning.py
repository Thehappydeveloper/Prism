from __future__ import annotations

import heapq
import math
from typing import Callable, Dict, Iterable, Sequence, Set, Tuple

import numpy as np

from .voxel_grid import VoxelIndex, neighbor_offsets_26


def dijkstra_26_connected(
    free_voxels: Set[VoxelIndex],
    start_voxel: VoxelIndex,
    voxel_size: float,
) -> tuple[Dict[VoxelIndex, float], Dict[VoxelIndex, VoxelIndex]]:
    """Run Dijkstra over a 26-connected voxel graph restricted to free voxels."""
    if start_voxel not in free_voxels:
        return {}, {}

    offsets = neighbor_offsets_26()
    edge_cost = {
        off: float(voxel_size * math.sqrt(off[0] ** 2 + off[1] ** 2 + off[2] ** 2))
        for off in offsets
    }

    dist: Dict[VoxelIndex, float] = {start_voxel: 0.0}
    prev: Dict[VoxelIndex, VoxelIndex] = {}
    heap: list[tuple[float, VoxelIndex]] = [(0.0, start_voxel)]

    while heap:
        d_cur, cur = heapq.heappop(heap)
        if d_cur > dist.get(cur, float("inf")):
            continue

        for off in offsets:
            nxt = (cur[0] + off[0], cur[1] + off[1], cur[2] + off[2])
            if nxt not in free_voxels:
                continue
            nd = d_cur + edge_cost[off]
            if nd < dist.get(nxt, float("inf")):
                dist[nxt] = nd
                prev[nxt] = cur
                heapq.heappush(heap, (nd, nxt))

    return dist, prev


def _unit(vec: np.ndarray) -> np.ndarray:
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
    """Direction from start to target estimated from first K path steps."""
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
