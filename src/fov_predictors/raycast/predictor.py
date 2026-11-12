"""Top-level predictor that connects FoV traces, point clouds, and ray casting.

`ReachabilityRaycastPredictor` is the package entry point used by the broader
predictor framework. It reads the current frame geometry and pose history,
builds cached voxel/tile context, and delegates the actual probability
computation to `reachability_pipeline.py`.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import numpy as np

from core.fov import Fov
from core.logger import get_run_logger
from fov_predictors.base import FovPredictor
from fov_predictors.raycast.reachability_pipeline import (
    ReachabilityParams,
    build_path_context_from_occupancy,
    build_unobstructed_navigation_context,
    build_voxel_context,
    compute_tile_importance,
)
from fov_predictors.raycast.raycast_visibility import sample_points_per_tile


class ReachabilityRaycastPredictor(FovPredictor):
    """
    Four-stage probabilistic FoV/tile-importance predictor:
    navigation voxel space -> intention estimation -> voxel probabilities -> tile exposure aggregation.
    """

    def __init__(
        self,
        navigation_voxel_size_l: float | str = "auto",
        visibility_voxel_size_l: float | str = "auto",
        r_nav: float | str = "auto",
        use_obstacles: bool = True,
        path_neighbor_connectivity: int = 6,
        k_path_dir: int = 3,
        history_horizons_s: str | float = "0.5,1.5,3.0",
        stability_weight: float = 0.7,
        mismatch_penalty_strength: float = 1.0,
        lambda_max: float = 0.9,
        softmax_temp: float = 1.0,
        samples_per_tile: int | str = "auto",
        max_voxels_eval: int | str = "auto",
        min_voxel_prob: float | str = "auto",
        r_vis: float = 6.0,
        sigma_g: float | None = None,
        visibility_floor: float = 0.0,
        rng_seed: int | None = 0,
        max_rays_per_tile: int | None = None,
        max_voxels_per_tile: int | None = None,
        max_candidate_voxels: int | None = None,
        voxel_selection_mode: str = "probability_diverse",
        voxel_selection_core_fraction: float = 0.7,
        origin_distribution_mode: str = "motion_weighted",
        aggregation_mode: str = "source_propagation",
        source_presence_sigma_l: float | None = None,
        source_score_tau: float = 1.0,
        source_min_candidates: int = 1,
        propagation_factor: float = 0.5,
        propagation_visibility_threshold: float = 1.0 / 3.0,
        raycast_processes: int = 1,
        auto_body_width: float = 0.6,
        auto_body_height: float = 1.8,
        auto_entry_width: float = 0.9,
        auto_entry_height: float = 2.0,
        auto_clearance_margin: float = 0.1,
        auto_r_nav_horizon_s: float = 3.0,
        auto_visibility_time_budget_ms: float = 20.0,
        auto_visibility_ms_per_ray_guess: float = 0.002,
        auto_voxel_mass_target: float = 0.95,
        auto_tile_confidence: float = 0.95,
        auto_tile_support_points: int = 32,
        auto_min_samples_per_tile: int = 4,
        auto_max_samples_per_tile: int = 64,
        visual_debug_enabled: bool = False,
        visual_debug_segments: str = "",
        visual_debug_max_tiles: int = 24,
        visual_debug_max_voxels: int = 32,
        visual_debug_max_contributors_per_tile: int = 8,
        visual_debug_max_rays_per_contributor: int = 4,
        visual_debug_trace_voxels: bool = True,
        cache_size: int = 8,
        debug: bool = False,
        **kwargs,
    ):
        """Configure the predictor and initialize small LRU-style caches."""
        super().__init__(**kwargs)
        self.navigation_voxel_size_l = navigation_voxel_size_l
        self.visibility_voxel_size_l = visibility_voxel_size_l
        self.r_nav = r_nav
        self.use_obstacles = use_obstacles
        self.path_neighbor_connectivity = path_neighbor_connectivity
        self.k_path_dir = k_path_dir
        self.history_horizons_s = history_horizons_s
        self.stability_weight = stability_weight
        self.mismatch_penalty_strength = mismatch_penalty_strength
        self.lambda_max = lambda_max
        self.softmax_temp = softmax_temp
        self.samples_per_tile = samples_per_tile
        self.max_voxels_eval = max_voxels_eval
        self.min_voxel_prob = min_voxel_prob
        self.r_vis = r_vis
        self.sigma_g = sigma_g
        self.visibility_floor = visibility_floor
        self.rng_seed = rng_seed
        self.max_rays_per_tile = max_rays_per_tile
        self.max_voxels_per_tile = max_voxels_per_tile
        self.max_candidate_voxels = max_candidate_voxels
        self.voxel_selection_mode = voxel_selection_mode
        self.voxel_selection_core_fraction = voxel_selection_core_fraction
        self.origin_distribution_mode = origin_distribution_mode
        self.aggregation_mode = aggregation_mode
        self.source_presence_sigma_l = source_presence_sigma_l
        self.source_score_tau = source_score_tau
        self.source_min_candidates = source_min_candidates
        self.propagation_factor = propagation_factor
        self.propagation_visibility_threshold = propagation_visibility_threshold
        self.raycast_processes = raycast_processes
        self.auto_body_width = auto_body_width
        self.auto_body_height = auto_body_height
        self.auto_entry_width = auto_entry_width
        self.auto_entry_height = auto_entry_height
        self.auto_clearance_margin = auto_clearance_margin
        self.auto_r_nav_horizon_s = auto_r_nav_horizon_s
        self.auto_visibility_time_budget_ms = auto_visibility_time_budget_ms
        self.auto_visibility_ms_per_ray_guess = auto_visibility_ms_per_ray_guess
        self.auto_voxel_mass_target = auto_voxel_mass_target
        self.auto_tile_confidence = auto_tile_confidence
        self.auto_tile_support_points = auto_tile_support_points
        self.auto_min_samples_per_tile = auto_min_samples_per_tile
        self.auto_max_samples_per_tile = auto_max_samples_per_tile
        self.visual_debug_enabled = visual_debug_enabled
        self.visual_debug_segments = visual_debug_segments
        self.visual_debug_max_tiles = visual_debug_max_tiles
        self.visual_debug_max_voxels = visual_debug_max_voxels
        self.visual_debug_max_contributors_per_tile = visual_debug_max_contributors_per_tile
        self.visual_debug_max_rays_per_contributor = visual_debug_max_rays_per_contributor
        self.visual_debug_trace_voxels = visual_debug_trace_voxels
        self.cache_size = max(1, int(cache_size))
        self.debug = debug
        self.last_visual_debug = None

        self.logger = get_run_logger(self.__class__.__name__)
        self._voxel_cache: OrderedDict[tuple, object] = OrderedDict()
        self._tile_sample_cache: OrderedDict[tuple, dict] = OrderedDict()
        self._visibility_ms_per_ray_ema: float | None = None

    PARAM_METADATA = {
        "navigation_voxel_size_l": {
            "label": "Navigation voxel size",
            "category": "Navigation",
            "help": "Voxel edge length used for reachable-space/path planning. Smaller values preserve narrow entries.",
        },
        "visibility_voxel_size_l": {
            "label": "Visibility voxel size",
            "category": "Visibility",
            "help": "Voxel edge length used for ray occlusion. Larger values keep raycasting affordable.",
        },
        "r_nav": {
            "label": "Navigation padding",
            "category": "Navigation",
            "advanced": True,
            "help": "Extra point-cloud bounding-box padding in meters. Auto uses the dataset max speed over the navigation horizon.",
        },
        "use_obstacles": {
            "label": "Obstacle-aware paths",
            "category": "Navigation",
            "help": "When enabled, candidate positions avoid occupied voxels and use shortest paths through free space.",
        },
        "path_neighbor_connectivity": {
            "label": "Path neighbor mode",
            "category": "Navigation",
            "help": "Coarse path-grid neighbor connectivity. Use 6 for 3D face-only movement or 26 for full diagonal movement.",
        },
        "k_path_dir": {
            "label": "Path direction steps",
            "category": "Navigation",
            "advanced": True,
            "help": "Legacy fallback for path direction steps. Normal scoring derives path-direction steps from the configured history horizons.",
        },
        "history_horizons_s": {
            "label": "History horizons",
            "category": "Motion confidence",
            "help": "Comma-separated motion-vector horizons in seconds, e.g. 0.5,1.5,3.0.",
        },
        "stability_weight": {
            "label": "Stability preference",
            "category": "Motion confidence",
            "help": "Blend weight for stable predictable motion versus any intense motion. Higher favors steady motion.",
        },
        "mismatch_penalty_strength": {
            "label": "Gaze mismatch penalty",
            "category": "Motion confidence",
            "help": "How strongly gaze-motion disagreement reduces effective movement confidence.",
        },
        "lambda_max": {
            "label": "Direction influence cap",
            "category": "Voxel scoring",
            "help": "Maximum weight given to movement-direction alignment when scoring reachable voxels.",
        },
        "softmax_temp": {
            "label": "Voxel softmax temperature",
            "category": "Voxel scoring",
            "advanced": True,
            "help": "Temperature for converting voxel scores to probabilities. Lower is sharper; higher is flatter.",
        },
        "samples_per_tile": {
            "label": "Tile samples",
            "category": "Visibility",
            "help": "Representative points sampled per tile for visibility tests. Auto allocates from the ray budget.",
        },
        "max_voxels_eval": {
            "label": "Voxel evaluation cap",
            "category": "Visibility",
            "help": "Maximum number of candidate voxels raycast during tile visibility aggregation. Auto keeps high-probability mass within budget.",
        },
        "min_voxel_prob": {
            "label": "Voxel probability floor",
            "category": "Visibility",
            "advanced": True,
            "help": "Minimum voxel probability retained before raycasting. Auto follows the adaptive pruning threshold.",
        },
        "r_vis": {
            "label": "Visibility radius",
            "category": "Visibility",
            "help": "Distance scale used to gate tiles before expensive visibility tests.",
        },
        "sigma_g": {
            "label": "Visibility gate sigma",
            "category": "Visibility",
            "advanced": True,
            "help": "Gaussian sigma for distance gating. Empty uses half of the visibility radius.",
        },
        "visibility_floor": {
            "label": "Visibility floor",
            "category": "Visibility",
            "advanced": True,
            "help": "Minimum per-voxel visibility contribution for active tiles. Use a small value to avoid hard zero scores from conservative occlusion.",
        },
        "rng_seed": {
            "label": "Sampling seed",
            "category": "Visibility",
            "advanced": True,
            "help": "Seed for deterministic tile point sampling. Empty disables the explicit seed.",
        },
        "max_rays_per_tile": {
            "label": "Ray cap per tile",
            "category": "Visibility",
            "advanced": True,
            "help": "Optional hard cap on rays per tile during visibility estimation.",
        },
        "max_voxels_per_tile": {
            "label": "Voxel cap per tile",
            "category": "Visibility",
            "advanced": True,
            "help": "Deprecated alias for max candidate voxels.",
        },
        "max_candidate_voxels": {
            "label": "Raycast origin budget",
            "category": "Visibility",
            "advanced": True,
            "help": "Optional cap on globally selected future-position voxels tested against each tile.",
        },
        "voxel_selection_mode": {
            "label": "Voxel selection mode",
            "category": "Visibility",
            "advanced": True,
            "help": "How raycast origins are selected from reachable voxel probabilities.",
        },
        "voxel_selection_core_fraction": {
            "label": "Probability focus",
            "category": "Visibility",
            "advanced": True,
            "help": "Fraction of the origin budget reserved for top-probability voxels; lower values add more spatial boundary coverage.",
        },
        "origin_distribution_mode": {
            "label": "Ray-origin distribution",
            "category": "Visibility",
            "advanced": True,
            "help": "Use motion-weighted future origins, uniform reachable origins, or only the current position.",
        },
        "aggregation_mode": {
            "label": "Tile aggregation",
            "category": "Visibility",
            "advanced": True,
            "help": "Tile scoring mode. source_propagation scores candidate-position source tiles first, then propagates one hop through visibility.",
        },
        "source_presence_sigma_l": {
            "label": "Source radius",
            "category": "Visibility",
            "advanced": True,
            "help": "Distance around a tile AABB where candidate positions count as source support. Empty uses 1.5 visibility voxels.",
        },
        "source_score_tau": {
            "label": "Source saturation",
            "category": "Visibility",
            "advanced": True,
            "help": "Relative median-normalized support where source tile score reaches about half strength.",
        },
        "source_min_candidates": {
            "label": "Min source candidates",
            "category": "Visibility",
            "advanced": True,
            "help": "Minimum assigned candidate positions required before a tile can become a source tile.",
        },
        "propagation_factor": {
            "label": "Propagation factor",
            "category": "Visibility",
            "advanced": True,
            "help": "Maximum fraction of a source tile score transmitted to a visible target tile.",
        },
        "propagation_visibility_threshold": {
            "label": "Propagation visibility",
            "category": "Visibility",
            "advanced": True,
            "help": "Minimum source-to-target visible ray fraction required for propagation. Default is one third.",
        },
        "raycast_processes": {
            "label": "Raycast processes",
            "category": "Visibility",
            "advanced": True,
            "help": "Number of worker processes used to split target-tile raycasting. 1 keeps the serial path.",
        },
        "auto_body_width": {"hidden": True},
        "auto_body_height": {"hidden": True},
        "auto_entry_width": {
            "label": "Auto entry width",
            "category": "Automatic sizing",
            "advanced": True,
            "help": "Smallest doorway/opening width that auto voxel sizing should preserve.",
        },
        "auto_entry_height": {
            "label": "Auto entry height",
            "category": "Automatic sizing",
            "advanced": True,
            "help": "Smallest doorway/opening height that auto voxel sizing should preserve.",
        },
        "auto_clearance_margin": {
            "label": "Auto clearance margin",
            "category": "Automatic sizing",
            "advanced": True,
            "help": "Relative safety margin applied when deriving automatic voxel size from entry dimensions.",
        },
        "auto_r_nav_horizon_s": {
            "label": "Navigation horizon",
            "category": "Automatic sizing",
            "advanced": True,
            "help": "Seconds of dataset max-speed motion used to derive automatic navigation padding.",
        },
        "auto_visibility_time_budget_ms": {
            "label": "Visibility budget",
            "category": "Runtime budget",
            "advanced": True,
            "help": "Target milliseconds for adaptive visibility work per prediction.",
        },
        "auto_visibility_ms_per_ray_guess": {
            "label": "Initial ray cost",
            "category": "Runtime budget",
            "advanced": True,
            "help": "Initial milliseconds-per-ray estimate before runtime measurements are available.",
        },
        "auto_voxel_mass_target": {
            "label": "Voxel mass target",
            "category": "Runtime budget",
            "advanced": True,
            "help": "Cumulative voxel probability mass retained by automatic voxel pruning.",
        },
        "auto_tile_confidence": {
            "label": "Tile sampling confidence",
            "category": "Runtime budget",
            "advanced": True,
            "help": "Detection-confidence target used by adaptive tile sampling.",
        },
        "auto_tile_support_points": {
            "label": "Tile support points",
            "category": "Runtime budget",
            "advanced": True,
            "help": "Minimum visible support size adaptive tile sampling tries not to miss.",
        },
        "auto_min_samples_per_tile": {
            "label": "Min tile samples",
            "category": "Runtime budget",
            "advanced": True,
            "help": "Lower bound for adaptive samples per tile.",
        },
        "auto_max_samples_per_tile": {
            "label": "Max tile samples",
            "category": "Runtime budget",
            "advanced": True,
            "help": "Upper bound for adaptive samples per tile.",
        },
        "cache_size": {
            "label": "Cache size",
            "category": "Runtime budget",
            "advanced": True,
            "help": "Number of recent voxel/tile sampling contexts retained in memory.",
        },
        "debug": {
            "label": "Debug logging",
            "category": "Runtime budget",
            "advanced": True,
            "help": "Emit top-tile and voxel statistics to the run logger.",
        },
        "visual_debug_enabled": {
            "label": "Visual predictor debug",
            "category": "Visualization",
            "advanced": True,
            "help": "Record compact voxel/ray explanations for the FoV comparison debugger.",
        },
        "visual_debug_segments": {
            "label": "Debug segments",
            "category": "Visualization",
            "advanced": True,
            "help": "Optional comma-separated target segment indices to record. Empty records every predicted segment when visual debug is enabled.",
        },
        "visual_debug_max_tiles": {
            "label": "Debug max tiles",
            "category": "Visualization",
            "advanced": True,
            "help": "Maximum top-scored tiles whose contributors are kept in the visual debug artifact.",
        },
        "visual_debug_max_voxels": {
            "label": "Debug max voxels",
            "category": "Visualization",
            "advanced": True,
            "help": "Maximum selected raycast origin voxels shown in the visual debug artifact.",
        },
        "visual_debug_max_contributors_per_tile": {
            "label": "Debug contributors per tile",
            "category": "Visualization",
            "advanced": True,
            "help": "Maximum voxel contributors retained for each debugged tile.",
        },
        "visual_debug_max_rays_per_contributor": {
            "label": "Debug rays per contributor",
            "category": "Visualization",
            "advanced": True,
            "help": "Maximum example rays retained for each tile/voxel contributor.",
        },
        "visual_debug_trace_voxels": {
            "label": "Debug ray voxel path",
            "category": "Visualization",
            "advanced": True,
            "help": "Include traversed voxel indices for stored example rays.",
        },
    }

    def _run_logger(self):
        """Resolve the current run-scoped logger after run logging has been configured."""
        return get_run_logger(self.__class__.__name__)

    def _cache_get(self, cache: OrderedDict, key):
        """Fetch a cached value and refresh its recency if present."""
        val = cache.get(key)
        if val is None:
            return None
        cache.move_to_end(key)
        return val

    def _cache_put(self, cache: OrderedDict, key, value):
        """Insert a cached value and evict the oldest entry when over capacity."""
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.cache_size:
            cache.popitem(last=False)

    def _frame_index_from_path(self, frame_path: Path) -> int:
        """Parse the numeric frame index embedded in a frame file name."""
        stem = Path(frame_path).stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        if not digits:
            raise ValueError(f"Could not extract frame index from {frame_path}")
        return int(digits)

    @staticmethod
    def _is_auto_param(value) -> bool:
        """Return whether a scalar hyperparameter is set to dynamic auto mode."""
        return isinstance(value, str) and value.strip().lower() == "auto"

    def _uniform_fallback(self, tiles):
        """Return a deterministic fallback when scene or pose inputs are unusable."""
        # Keep playback stable when trace/geometry inputs are incomplete.
        return {str(tile.unique_id): 1.0 for tile in tiles}

    def _visibility_ms_per_ray_estimate(self) -> float:
        """Return the current visibility-cost estimate used by the auto allocator."""
        estimate = self._visibility_ms_per_ray_ema
        if estimate is None or estimate <= 0.0:
            estimate = float(self.auto_visibility_ms_per_ray_guess)
        return max(float(estimate), 1e-6)

    def _auto_navigation_voxel_size_l(self) -> float:
        g = max(float(self.auto_clearance_margin), 0.0)
        entry_width = max(float(self.auto_entry_width), 1e-6)
        entry_height = max(float(self.auto_entry_height), 1e-6)
        # The path grid is 2x coarser than the fine occupancy grid, so this
        # preserves the old auto occupancy resolution while keeping Dijkstra
        # on the cheaper path grid.
        resolved = (min(entry_width, entry_height) * max(0.0, 1.0 - g)) / 2.0
        return max(float(resolved), 1e-3)

    def _resolve_navigation_voxel_size_l(self) -> float:
        """Resolve the navigation grid voxel size."""
        if not self._is_auto_param(self.navigation_voxel_size_l):
            return max(float(self.navigation_voxel_size_l), 1e-6)
        return self._auto_navigation_voxel_size_l()

    def _resolve_visibility_voxel_size_l(self) -> float:
        """Resolve the visibility grid voxel size."""
        if not self._is_auto_param(self.visibility_voxel_size_l):
            return max(float(self.visibility_voxel_size_l), 1e-6)
        return max(self._resolve_navigation_voxel_size_l(), 1e-3)

    def _history_horizons(self) -> tuple[float, ...]:
        raw = self.history_horizons_s
        if isinstance(raw, (int, float)):
            vals = [float(raw)]
        else:
            vals = []
            for part in str(raw or "").replace(";", ",").split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    vals.append(float(part))
                except ValueError:
                    continue
        vals = sorted({round(max(v, 1e-3), 6) for v in vals if v > 0.0})
        return tuple(vals or [0.5, 1.5, 3.0])

    def _max_history_frames(self, fov_data=None) -> int:
        horizons = self._history_horizons()
        frame_rate_hz = max(float(getattr(fov_data, "frame_rate_hz", 30.0)), 1e-6)
        return max(2, int(round(max(horizons) * frame_rate_hz)) + 1)

    def _resolve_r_nav(self, max_speed: float) -> float:
        """Resolve navigation padding from either a fixed value or a motion horizon."""
        if not self._is_auto_param(self.r_nav):
            return max(float(self.r_nav), 0.0)
        horizon_s = max(float(self.auto_r_nav_horizon_s), 0.0)
        return max(float(max_speed) * horizon_s, 0.0)

    def _update_visibility_ms_per_ray(self, debug_stats: dict) -> None:
        """Update the rolling ms-per-ray estimate from the latest visibility stage."""
        rays = float(debug_stats.get("total_ray_tests", 0.0))
        elapsed_ms = float(debug_stats.get("aggregate_tile_importance_ms", 0.0))
        if rays <= 0.0 or elapsed_ms <= 0.0:
            return
        sample = elapsed_ms / rays
        if not np.isfinite(sample) or sample <= 0.0:
            return
        if self._visibility_ms_per_ray_ema is None:
            self._visibility_ms_per_ray_ema = float(sample)
            return
        alpha = 0.2
        self._visibility_ms_per_ray_ema = ((1.0 - alpha) * self._visibility_ms_per_ray_ema) + (alpha * float(sample))

    def _visual_debug_enabled_for_request(self, request) -> bool:
        """Return whether this prediction should keep a compact visual trace."""
        if not bool(self.visual_debug_enabled):
            return False
        raw_segments = str(self.visual_debug_segments or "").strip()
        if not raw_segments:
            return True
        if request is None:
            return False
        try:
            segment_idx = int(request.target_segment_idx)
        except Exception:  # pylint: disable=broad-except
            return False
        selected = set()
        for part in raw_segments.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                selected.add(int(part))
            except ValueError:
                continue
        return segment_idx in selected

    def _resolve_segment_duration(self, tiles) -> float:
        """Infer the tile segment duration from MPD representation metadata."""
        for tile in tiles:
            for reps in tile.component_options.values():
                if reps:
                    return max(float(reps[0].duration_s), 1e-3)
        return 1.0

    def _dataset_motion_limits(self, fov_data: Fov) -> tuple[float, float]:
        """Read dataset-registered motion normalization limits from FoV context."""
        limits = getattr(fov_data, "motion_limits", None)
        if not isinstance(limits, dict):
            limits = {}
        max_speed = limits.get("max_speed", 1.5)
        max_acceleration = limits.get("max_acceleration", 3.0)
        return max(float(max_speed), 1e-6), max(float(max_acceleration), 1e-6)

    def _tile_signature(self, tiles) -> tuple:
        """Build a stable cache key for the current tile layout."""
        sig = []
        for tile in tiles:
            sig.append(
                (
                    str(tile.unique_id),
                    tuple(round(float(v), 5) for v in tile.min_corner),
                    tuple(round(float(v), 5) for v in tile.max_corner),
                )
            )
        return tuple(sig)

    def required_ply_frame_paths(self, frame_paths):
        """Reachability scoring reads only the latest observed point-cloud frame."""
        return [frame_paths[-1]] if frame_paths else []

    def _read_points_xyz(self, frame_path: Path) -> np.ndarray:
        """Load XYZ point-cloud coordinates for the requested frame."""
        ply_path = Path(frame_path)
        if ply_path.suffix.lower() != ".ply":
            ply_path = ply_path.with_suffix(ply_path.suffix + ".ply")
        shared_cache = getattr(self, "_active_shared_cache", None)
        cache_key = str(ply_path.resolve())
        if isinstance(shared_cache, dict):
            lock = shared_cache.get("lock")
            points_cache = shared_cache.setdefault("points_xyz", {})
            if lock is not None:
                with lock:
                    cached = points_cache.get(cache_key)
                    if cached is not None:
                        return cached
                    points = self.ply_reader.read(str(ply_path))
                    points_np = np.asarray(points)
                    if points_np.ndim != 2 or points_np.shape[1] < 3:
                        raise ValueError(f"PLY does not contain XYZ columns: {ply_path}")
                    points_xyz = np.asarray(points_np[:, :3], dtype=float)
                    points_cache[cache_key] = points_xyz
                    return points_xyz
        points = self.ply_reader.read(str(ply_path))
        points_np = np.asarray(points)
        if points_np.ndim != 2 or points_np.shape[1] < 3:
            raise ValueError(f"PLY does not contain XYZ columns: {ply_path}")
        return np.asarray(points_np[:, :3], dtype=float)

    def _get_pose_now_and_history(self, frame_idx: int, fov_data: Fov):
        """Resolve the current pose plus a bounded history window for intention estimation."""
        rel_idx = int(frame_idx - fov_data.start_frame_index)
        pose_now = fov_data.get_frame(rel_idx)
        if pose_now is None:
            return None, None

        if hasattr(fov_data, "get_history"):
            pose_history = fov_data.get_history(rel_idx, count_back=self._max_history_frames(fov_data), include_current=True)
            if pose_history is not None and len(pose_history) > 0:
                return pose_now, pose_history

        start = rel_idx - max(1, int(self._max_history_frames(fov_data))) + 1
        mask = (fov_data.df.index >= start) & (fov_data.df.index <= rel_idx)
        pose_history = fov_data.df.loc[mask]
        if pose_history is None or len(pose_history) == 0:
            pose_history = fov_data.df.loc[[rel_idx]] if rel_idx in fov_data.df.index else None
        return pose_now, pose_history

    def predict(self, tiles, frame_paths, fov_data):
        """Score tiles for the latest frame by chaining voxel, intention, and visibility stages."""
        self.last_visual_debug = None
        session = self._latency_session(fov_data)
        previous_shared_cache = getattr(self, "_active_shared_cache", None)
        self._active_shared_cache = getattr(fov_data, "shared_raycast_cache", None)
        request = getattr(fov_data, "_active_prediction_request", None)
        metadata = {
            "tile_count": len(tiles),
            "frame_count": len(frame_paths),
            "history_horizons_s": ",".join(str(v) for v in self._history_horizons()),
        }
        try:
            if not frame_paths:
                metadata["fallback_reason"] = "missing_frame_paths"
                return self._uniform_fallback(tiles)

            target_frame_path = Path(frame_paths[-1])
            metadata["frame_name"] = target_frame_path.name
            try:
                with session.stage("resolve_frame_and_pose_history"):
                    if request is not None:
                        frame_idx = request.observed_frame_idx
                        metadata["observed_frame_idx"] = None if frame_idx is None else int(frame_idx)
                        metadata["prediction_horizon_s"] = float(request.prediction_horizon_s)
                    else:
                        frame_idx = self._frame_index_from_path(target_frame_path)
                    if frame_idx is None:
                        pose_now, pose_history = None, None
                    else:
                        pose_now, pose_history = self._get_pose_now_and_history(int(frame_idx), fov_data)
            except Exception:
                metadata["fallback_reason"] = "frame_index_parse_failed"
                return self._uniform_fallback(tiles)
            if pose_now is None:
                metadata["fallback_reason"] = "missing_pose"
                return self._uniform_fallback(tiles)

            try:
                with session.stage("read_point_cloud"):
                    points_xyz = self._read_points_xyz(target_frame_path)
            except Exception:
                metadata["fallback_reason"] = "point_cloud_read_failed"
                return self._uniform_fallback(tiles)
            metadata["point_count"] = int(points_xyz.shape[0]) if getattr(points_xyz, "ndim", 0) >= 2 else 0

            # Stage 1: build and cache voxelized scene context for this frame geometry.
            segment_duration_s = self._resolve_segment_duration(tiles)
            max_speed, max_acceleration = self._dataset_motion_limits(fov_data)
            resolved_navigation_voxel_size_l = self._resolve_navigation_voxel_size_l()
            resolved_occupancy_voxel_size_l = max(resolved_navigation_voxel_size_l * 0.5, 1e-6)
            resolved_visibility_voxel_size_l = self._resolve_visibility_voxel_size_l()
            resolved_r_nav = self._resolve_r_nav(max_speed)
            auto_samples_per_tile = self._is_auto_param(self.samples_per_tile)
            auto_max_voxels_eval = self._is_auto_param(self.max_voxels_eval)
            auto_min_voxel_prob = self._is_auto_param(self.min_voxel_prob)
            max_candidate_voxels = self.max_candidate_voxels
            if max_candidate_voxels is None:
                max_candidate_voxels = self.max_voxels_per_tile
            visual_debug_enabled = self._visual_debug_enabled_for_request(request)

            params = ReachabilityParams(
                voxel_size_l=resolved_visibility_voxel_size_l,
                occupancy_voxel_size_l=resolved_occupancy_voxel_size_l,
                navigation_voxel_size_l=resolved_navigation_voxel_size_l,
                visibility_voxel_size_l=resolved_visibility_voxel_size_l,
                path_neighbor_connectivity=26 if int(self.path_neighbor_connectivity) == 26 else 6,
                r_nav=resolved_r_nav,
                v_max=float(max_speed),
                segment_duration_s=float(segment_duration_s),
                prediction_horizon_s=(
                    float(request.prediction_horizon_s)
                    if request is not None
                    else None
                ),
                use_obstacles=bool(self.use_obstacles),
                k_path_dir=max(1, int(self.k_path_dir)),
                history_frames=self._max_history_frames(fov_data),
                history_horizons_s=self._history_horizons(),
                assumed_fps=max(float(getattr(fov_data, "frame_rate_hz", 30.0)), 1e-6),
                max_speed=max_speed,
                max_acceleration=max_acceleration,
                stability_weight=float(np.clip(float(self.stability_weight), 0.0, 1.0)),
                mismatch_penalty_strength=float(np.clip(float(self.mismatch_penalty_strength), 0.0, 1.0)),
                lambda_max=float(self.lambda_max),
                softmax_temp=max(float(self.softmax_temp), 1e-6),
                samples_per_tile=max(1, int(self.auto_min_samples_per_tile if auto_samples_per_tile else self.samples_per_tile)),
                max_voxels_eval=max(1, int(1 if auto_max_voxels_eval else self.max_voxels_eval)),
                min_voxel_prob=max(float(0.0 if auto_min_voxel_prob else self.min_voxel_prob), 0.0),
                auto_samples_per_tile=auto_samples_per_tile,
                auto_max_voxels_eval=auto_max_voxels_eval,
                auto_min_voxel_prob=auto_min_voxel_prob,
                r_vis=max(float(self.r_vis), 1e-6),
                sigma_g=(None if self.sigma_g is None else max(float(self.sigma_g), 1e-6)),
                visibility_floor=float(np.clip(float(self.visibility_floor), 0.0, 1.0)),
                rng_seed=self.rng_seed,
                auto_visibility_time_budget_ms=max(float(self.auto_visibility_time_budget_ms), 1e-3),
                auto_visibility_ms_per_ray=self._visibility_ms_per_ray_estimate(),
                auto_voxel_mass_target=float(np.clip(float(self.auto_voxel_mass_target), 1e-6, 1.0)),
                auto_tile_confidence=float(np.clip(float(self.auto_tile_confidence), 1e-6, 1.0 - 1e-6)),
                auto_tile_support_points=max(1, int(self.auto_tile_support_points)),
                auto_min_samples_per_tile=max(1, int(self.auto_min_samples_per_tile)),
                auto_max_samples_per_tile=max(max(1, int(self.auto_min_samples_per_tile)), int(self.auto_max_samples_per_tile)),
                max_rays_per_tile=self.max_rays_per_tile,
                max_voxels_per_tile=(None if self.max_voxels_per_tile is None else max(1, int(self.max_voxels_per_tile))),
                max_candidate_voxels=(None if max_candidate_voxels is None else max(1, int(max_candidate_voxels))),
                voxel_selection_mode=str(self.voxel_selection_mode or "probability_diverse"),
                voxel_selection_core_fraction=float(np.clip(float(self.voxel_selection_core_fraction), 0.0, 1.0)),
                origin_distribution_mode=str(self.origin_distribution_mode or "motion_weighted"),
                aggregation_mode=str(self.aggregation_mode or "source_propagation"),
                source_presence_sigma_l=(
                    None
                    if self.source_presence_sigma_l is None
                    else max(float(self.source_presence_sigma_l), 1e-6)
                ),
                source_score_tau=max(float(self.source_score_tau), 1e-6),
                source_min_candidates=max(1, int(self.source_min_candidates)),
                propagation_factor=float(np.clip(float(self.propagation_factor), 0.0, 1.0)),
                propagation_visibility_threshold=float(np.clip(float(self.propagation_visibility_threshold), 0.0, 1.0)),
                raycast_processes=max(1, int(self.raycast_processes)),
                angle_convention=self.angle_convention,
                visual_debug_enabled=visual_debug_enabled,
                visual_debug_max_tiles=max(1, int(self.visual_debug_max_tiles)),
                visual_debug_max_voxels=max(1, int(self.visual_debug_max_voxels)),
                visual_debug_max_contributors_per_tile=max(1, int(self.visual_debug_max_contributors_per_tile)),
                visual_debug_max_rays_per_contributor=max(0, int(self.visual_debug_max_rays_per_contributor)),
                visual_debug_trace_voxels=bool(self.visual_debug_trace_voxels),
            )

            frame_key = str(target_frame_path.resolve())
            occupancy_cache_key = ("occupancy", frame_key, round(resolved_occupancy_voxel_size_l, 6), round(params.r_nav, 6))
            navigation_cache_key = (
                "path",
                frame_key,
                round(resolved_navigation_voxel_size_l, 6),
                round(resolved_occupancy_voxel_size_l, 6),
                int(params.path_neighbor_connectivity),
                round(params.r_nav, 6),
            )
            visibility_cache_key = ("visibility", frame_key, round(resolved_visibility_voxel_size_l, 6), round(params.r_nav, 6))

            def resolve_cached_voxel_context(cache_key, context_params, builder=build_voxel_context):
                shared_cache = getattr(self, "_active_shared_cache", None)
                voxel_context = None
                if isinstance(shared_cache, dict):
                    lock = shared_cache.get("lock")
                    voxel_cache = shared_cache.setdefault("voxel_context", {})
                    if lock is not None:
                        with lock:
                            voxel_context = voxel_cache.get(cache_key)
                            if voxel_context is not None:
                                self._cache_put(self._voxel_cache, cache_key, voxel_context)
                            else:
                                voxel_context = self._cache_get(self._voxel_cache, cache_key)
                                if voxel_context is None:
                                    voxel_context = builder(points_xyz, context_params)
                                    self._cache_put(self._voxel_cache, cache_key, voxel_context)
                                voxel_cache[cache_key] = voxel_context
                if voxel_context is None:
                    voxel_context = self._cache_get(self._voxel_cache, cache_key)
                    if voxel_context is None:
                        voxel_context = builder(points_xyz, context_params)
                        self._cache_put(self._voxel_cache, cache_key, voxel_context)
                return voxel_context

            if params.use_obstacles:
                with session.stage("build_occupancy_voxel_context"):
                    occupancy_context = resolve_cached_voxel_context(
                        occupancy_cache_key,
                        replace(params, voxel_size_l=resolved_occupancy_voxel_size_l),
                    )
                with session.stage("build_navigation_path_context"):
                    navigation_context = self._cache_get(self._voxel_cache, navigation_cache_key)
                    if navigation_context is None:
                        navigation_context = build_path_context_from_occupancy(occupancy_context, params)
                        self._cache_put(self._voxel_cache, navigation_cache_key, navigation_context)
            else:
                occupancy_context = None
                open_navigation_cache_key = (
                    "unobstructed_navigation",
                    frame_key,
                    round(resolved_navigation_voxel_size_l, 6),
                    round(params.r_nav, 6),
                )
                with session.stage("build_unobstructed_navigation_context"):
                    navigation_context = resolve_cached_voxel_context(
                        open_navigation_cache_key,
                        replace(params, voxel_size_l=resolved_navigation_voxel_size_l),
                        builder=build_unobstructed_navigation_context,
                    )
            with session.stage("build_visibility_voxel_context"):
                visibility_context = resolve_cached_voxel_context(
                    visibility_cache_key,
                    replace(params, voxel_size_l=resolved_visibility_voxel_size_l),
                )

            # Stage 2: cache tile point samples used by visibility ray tests.
            tile_samples = None
            if not params.auto_samples_per_tile:
                sample_cache_key = (
                    frame_key,
                    int(params.samples_per_tile),
                    int(params.rng_seed) if params.rng_seed is not None else None,
                    self._tile_signature(tiles),
                )
                with session.stage("sample_tile_points"):
                    shared_cache = getattr(self, "_active_shared_cache", None)
                    if isinstance(shared_cache, dict):
                        lock = shared_cache.get("lock")
                        sample_cache = shared_cache.setdefault("tile_samples", {})
                        if lock is not None:
                            with lock:
                                tile_samples = sample_cache.get(sample_cache_key)
                                if tile_samples is not None:
                                    self._cache_put(self._tile_sample_cache, sample_cache_key, tile_samples)
                                else:
                                    tile_samples = self._cache_get(self._tile_sample_cache, sample_cache_key)
                                    if tile_samples is None:
                                        tile_samples = sample_points_per_tile(
                                            points_xyz=points_xyz,
                                            tiles=tiles,
                                            samples_per_tile=params.samples_per_tile,
                                            rng_seed=params.rng_seed,
                                        )
                                        self._cache_put(self._tile_sample_cache, sample_cache_key, tile_samples)
                                    sample_cache[sample_cache_key] = tile_samples
                    if tile_samples is None:
                        tile_samples = self._cache_get(self._tile_sample_cache, sample_cache_key)
                        if tile_samples is None:
                            tile_samples = sample_points_per_tile(
                                points_xyz=points_xyz,
                                tiles=tiles,
                                samples_per_tile=params.samples_per_tile,
                                rng_seed=params.rng_seed,
                            )
                            self._cache_put(self._tile_sample_cache, sample_cache_key, tile_samples)

            # Stages 3/4: infer reachable-voxel probabilities and aggregate into independent tile scores.
            tile_scores, debug_stats = compute_tile_importance(
                tiles=tiles,
                points_xyz=points_xyz,
                pose_now=pose_now,
                pose_history=pose_history,
                params=params,
                occupancy_context=occupancy_context,
                navigation_context=navigation_context,
                visibility_context=visibility_context,
                tile_samples=tile_samples,
                latency_session=session,
            )
            visual_debug = debug_stats.pop("_visual_debug", None)
            visual_intentions = debug_stats.pop("_visual_intentions", None)
            if visual_debug is not None:
                if visual_intentions is not None:
                    visual_debug["intentions"] = visual_intentions
                visual_debug["segment_idx"] = int(request.target_segment_idx) if request is not None else None
                visual_debug["observed_frame_idx"] = (
                    int(request.observed_frame_idx)
                    if request is not None and request.observed_frame_idx is not None
                    else None
                )
                visual_debug["frame_name"] = target_frame_path.name
                visual_debug["prediction_horizon_s"] = (
                    float(request.prediction_horizon_s)
                    if request is not None and request.prediction_horizon_s is not None
                    else None
                )
                visual_debug["target_segment_duration_s"] = (
                    float(request.target_segment_duration_s)
                    if request is not None and request.target_segment_duration_s is not None
                    else None
                )
                visual_debug["scale_reference"] = {
                    "geometry_voxel_size_l": float(resolved_occupancy_voxel_size_l),
                    "navigation_voxel_size_l": float(resolved_navigation_voxel_size_l),
                    "visibility_voxel_size_l": float(resolved_visibility_voxel_size_l),
                }
                self.last_visual_debug = visual_debug
            self._update_visibility_ms_per_ray(debug_stats)
            metadata["free_voxels"] = int(debug_stats.get("free_voxels", 0.0))
            metadata["kept_voxels"] = int(debug_stats.get("kept_voxels", 0.0))
            metadata["occupancy_voxel_size_l"] = round(float(resolved_occupancy_voxel_size_l), 6)
            metadata["navigation_voxel_size_l"] = round(float(resolved_navigation_voxel_size_l), 6)
            metadata["visibility_voxel_size_l"] = round(float(resolved_visibility_voxel_size_l), 6)
            metadata["frame_rate_hz"] = round(float(params.assumed_fps), 6)
            metadata["path_neighbor_connectivity"] = int(params.path_neighbor_connectivity)
            metadata["obstacle_aware_reachability"] = bool(params.use_obstacles)
            metadata["navigation_uses_scene_occupancy"] = bool(params.use_obstacles)
            metadata["visibility_geometry_enabled"] = True
            metadata["resolved_r_nav"] = round(float(resolved_r_nav), 6)
            metadata["motion_max_speed"] = round(float(max_speed), 6)
            metadata["motion_max_acceleration"] = round(float(max_acceleration), 6)
            for key in (
                "occupied_voxels",
                "candidate_voxels",
                "occupancy_grid_dim_x",
                "occupancy_grid_dim_y",
                "occupancy_grid_dim_z",
                "occupancy_grid_total",
                "navigation_grid_dim_x",
                "navigation_grid_dim_y",
                "navigation_grid_dim_z",
                "navigation_grid_total",
                "visibility_grid_dim_x",
                "visibility_grid_dim_y",
                "visibility_grid_dim_z",
                "visibility_grid_total",
                "navigation_to_visibility_voxels",
                "navigation_to_visibility_collisions",
                "navigation_to_visibility_origin_offset_mean",
                "navigation_to_visibility_origin_offset_max",
                "path_topology_cache_size",
                "active_tiles",
                "gated_out_tiles",
                "active_sample_points",
                "ray_tests_per_visibility_eval",
                "voxel_probability_support",
                "effective_voxels_per_tile",
                "total_ray_tests",
                "candidate_voxel_core_fraction",
                "candidate_voxel_diversity_fraction",
                "candidate_voxel_count_raw",
                "candidate_voxel_count_selected",
                "candidate_voxel_selected_mass",
                "candidate_voxel_min_probability",
                "candidate_voxel_max_probability",
                "candidate_voxel_mean_probability",
                "legacy_distance_biased_enabled",
                "max_voxels_per_tile_deprecated",
                "selected_voxel_mass_mean",
                "resolved_max_voxels_eval",
                "resolved_min_voxel_prob",
                "auto_prune_mass_target",
                "auto_prune_mass_kept",
                "auto_prune_budget_cap",
                "auto_sample_allocated_tiles",
                "auto_sample_total",
                "auto_sample_confidence_mean",
                "auto_sample_min",
                "auto_sample_max",
                "visibility_ray_budget",
                "visibility_floor",
                "source_tile_count",
                "source_tiles_with_score",
                "source_median_candidate_count",
                "source_median_candidate_mass",
                "source_presence_radius_l",
                "source_score_tau",
                "source_min_candidates",
                "source_propagation_factor",
                "source_propagation_visibility_threshold",
                "source_propagation_direct_visibility_fallback",
                "source_target_pairs_tested",
                "source_target_pairs_passing_threshold",
                "propagated_tiles_with_score",
                "raycast_processes_requested",
                "raycast_processes_used",
                "raycast_parallel_fallback",
                "aggregate_tile_importance_ms",
                "auto_visibility_time_budget_ms",
                "auto_visibility_ms_per_ray",
                "nearest_tile_distance",
                "farthest_tile_distance",
                "max_tile_gate",
                "mean_tile_gate",
                "motion_confidence",
                "normalized_speed",
                "normalized_acceleration",
                "motion_intensity",
                "motion_stability",
                "direction_reliability",
                "motion_mismatch",
                "intention_horizon_count",
                "path_dir_step_count",
                "intention_motion_consensus",
                "intention_confidence_scale",
                "intention_confidence_mean",
                "intention_effective_confidence_mean",
                "intention_mismatch_mean",
            ):
                if key in debug_stats:
                    metadata[key] = int(debug_stats[key]) if float(debug_stats[key]).is_integer() else round(float(debug_stats[key]), 3)
            for idx in range(3):
                for suffix in ("s", "path_k", "confidence", "effective_confidence", "mismatch"):
                    key = f"intention_horizon_{idx}_{suffix}"
                    if key in debug_stats:
                        metadata[key] = round(float(debug_stats[key]), 3)
            for key in ("tile_gate_details", "gating_reason"):
                if key in debug_stats:
                    metadata[key] = debug_stats[key]
            if "candidate_voxel_selection_mode" in debug_stats:
                metadata["candidate_voxel_selection_mode"] = str(debug_stats["candidate_voxel_selection_mode"])
            if "tile_aggregation_mode" in debug_stats:
                metadata["tile_aggregation_mode"] = str(debug_stats["tile_aggregation_mode"])

            if self.debug:
                top_tiles = sorted(tile_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
                self._run_logger().info(
                    "ReachabilityRaycastPredictor frame=%s free_vox=%.0f kept_vox=%.0f c_eff=%.3f top_tiles=%s",
                    target_frame_path.name,
                    debug_stats.get("free_voxels", 0.0),
                    debug_stats.get("kept_voxels", 0.0),
                    debug_stats.get("c_eff", 0.0),
                    top_tiles,
                )

            if not tile_scores:
                metadata["fallback_reason"] = "empty_tile_scores"
            return tile_scores if tile_scores else self._uniform_fallback(tiles)
        finally:
            self._active_shared_cache = previous_shared_cache
            session.finish(metadata=metadata)

    def predict_for_request(self, tiles, request, fov_data):
        """Predict with geometry and observed FoV separated by the player."""
        previous_request = getattr(fov_data, "_active_prediction_request", None)
        fov_data._active_prediction_request = request
        try:
            return self.predict(tiles, request.geometry_frame_paths, fov_data)
        finally:
            if previous_request is None:
                try:
                    delattr(fov_data, "_active_prediction_request")
                except AttributeError:
                    pass
            else:
                fov_data._active_prediction_request = previous_request
