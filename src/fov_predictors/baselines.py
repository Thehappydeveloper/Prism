from fov_predictors.base import FovPredictor
from core.fov import Fov
from scipy.spatial.transform import Rotation as R
import numpy as np
import time
import logging
from collections import OrderedDict
from pathlib import Path

from fov_predictors.reachability_pipeline import (
    ReachabilityParams,
    build_voxel_context,
    compute_tile_importance,
)


class FrameIndexMixin:
    """Utility mixin to extract frame indices from filenames."""

    def _frame_index_from_path(self, frame_path: Path) -> int:
        stem = Path(frame_path).stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        if not digits:
            raise ValueError(f"Could not extract frame index from {frame_path}")
        return int(digits)


class LastObservedFovPredictor(FrameIndexMixin, FovPredictor):
    """User keeps looking at the last observed direction."""

    def predict(self, tiles, frame_paths, fov_data):
        frame = self._get_frame(frame_paths[-1], fov_data)
        forward = self._compute_forward(frame)
        points = self._read_ply(frame_paths[-1])
        mask = self._points_in_fov(points, frame[['HMDPX','HMDPY','HMDPZ']].to_numpy(), forward,
                                   euler_angles=None, fov_h=self.fov_h, fov_v=self.fov_v)
        return self.tile_fov_proportion(tiles, mask, points)

    def _get_frame(self, frame_path: Path, fov_data: Fov):
        frame_idx = self._frame_index_from_path(frame_path)
        return fov_data.get_frame(frame_idx - fov_data.start_frame_index)

    def _frame_index_from_path(self, frame_path: Path) -> int:
        stem = Path(frame_path).stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        if not digits:
            raise ValueError(f"Could not extract frame index from {frame_path}")
        return int(digits)

    def _compute_forward(self, frame):
        euler_angles = tuple(frame[['HMDRX','HMDRY','HMDRZ']])
        r = R.from_euler("ZXY", [euler_angles[2], euler_angles[0], euler_angles[1]], degrees=True)
        return r.apply([0, 0, 1])

    def _read_ply(self, frame_path: Path):
        t0 = time.time()
        ply_path = Path(frame_path)
        if ply_path.suffix.lower() != ".ply":
            ply_path = ply_path.with_suffix(ply_path.suffix + ".ply")
        points = self.ply_reader.read(str(ply_path))
        #print("PLY read time:", time.time() - t0)
        return points

class AverageMotionPredictor(FrameIndexMixin, FovPredictor):
    """Predict FoV as average motion over last `lookback` frames."""

    def __init__(self, lookback: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.lookback = lookback
        self.last_observed = LastObservedFovPredictor(ply_reader=self.ply_reader,
                                                      fov_h=self.fov_h, fov_v=self.fov_v)

    def predict(self, tiles, frame_paths, fov_data):
        frames = self._get_recent_frames(frame_paths[-1], fov_data)
        if not frames:
            return self.last_observed.predict(tiles, frame_paths, fov_data)

        position, forward = self._compute_average_motion(frames)
        points = self.last_observed._read_ply(frame_paths[-1])
        mask = self.last_observed._points_in_fov(points, position, forward,
                                                 euler_angles=None,
                                                 fov_h=self.fov_h, fov_v=self.fov_v)
        return self.last_observed.tile_fov_proportion(tiles, mask, points)

    def _get_recent_frames(self, frame_path: Path, fov_data: Fov):
        frame_idx = self._frame_index_from_path(frame_path)
        start = max(0, frame_idx - self.lookback)
        frames = [fov_data.get_frame(i - fov_data.start_frame_index) for i in range(start, frame_idx)]
        return [f for f in frames if f is not None]

    def _compute_average_motion(self, frames):
        positions = np.array([f[['HMDPX','HMDPY','HMDPZ']].to_numpy() for f in frames])
        avg_velocity = (positions[-1] - positions[0]) / max(1, len(positions) - 1)
        forward = avg_velocity / np.linalg.norm(avg_velocity) if np.linalg.norm(avg_velocity) > 0 else np.array([0, 0, 1])
        position = frames[-1][['HMDPX','HMDPY','HMDPZ']].to_numpy()
        return position, forward

class LinearExtrapolationPredictor(FrameIndexMixin, FovPredictor):
    """Predict FoV by linear extrapolation of direction over last `lookback` frames."""

    def __init__(self, lookback: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.lookback = lookback
        self.last_observed = LastObservedFovPredictor(ply_reader=self.ply_reader,
                                                      fov_h=self.fov_h, fov_v=self.fov_v)

    def predict(self, tiles, frame_paths, fov_data):
        frames = self._get_recent_frames(frame_paths[-1], fov_data)
        if len(frames) < 2:
            return self.last_observed.predict(tiles, frame_paths, fov_data)

        position, forward = self._extrapolate_forward(frames)
        points = self.last_observed._read_ply(frame_paths[-1])
        mask = self.last_observed._points_in_fov(points, position, forward,
                                                 euler_angles=None,
                                                 fov_h=self.fov_h, fov_v=self.fov_v)
        return self.last_observed.tile_fov_proportion(tiles, mask, points)

    def _get_recent_frames(self, frame_path: Path, fov_data: Fov):
        frame_idx = self._frame_index_from_path(frame_path)
        start = max(0, frame_idx - self.lookback)
        frames = [fov_data.get_frame(i - fov_data.start_frame_index) for i in range(start, frame_idx)]
        return [f for f in frames if f is not None]

    def _extrapolate_forward(self, frames):
        forwards = []
        for f in frames:
            euler_angles = tuple(f[['HMDRX','HMDRY','HMDRZ']])
            r = R.from_euler("ZXY", [euler_angles[2], euler_angles[0], euler_angles[1]], degrees=True)
            forwards.append(r.apply([0,0,1]))
        forwards = np.stack(forwards)
        delta = forwards[-1] - forwards[0]
        forward = forwards[-1] + delta
        forward /= np.linalg.norm(forward)
        position = frames[-1][['HMDPX','HMDPY','HMDPZ']].to_numpy()
        return position, forward


class ReachabilityRaycastPredictor(FrameIndexMixin, FovPredictor):
    """
    Four-stage probabilistic FoV/tile-importance predictor:
    navigation voxel space -> intention estimation -> voxel probabilities -> tile exposure aggregation.
    """

    def __init__(
        self,
        voxel_size_l: float = 0.5,
        r_nav: float = 0.5,
        v_max: float = 1.5,
        segment_duration_s: float | None = None,
        use_obstacles: bool = True,
        k_path_dir: int = 3,
        history_frames: int = 15,
        s0: float = 0.2,
        kappa_s: float = 4.0,
        kappa_a: float = 0.5,
        gamma: float = 1.0,
        lambda_max: float = 0.9,
        softmax_temp: float = 1.0,
        samples_per_tile: int = 24,
        max_voxels_eval: int = 128,
        min_voxel_prob: float = 1e-4,
        r_vis: float = 6.0,
        sigma_g: float | None = None,
        rng_seed: int | None = 0,
        max_rays_per_tile: int | None = None,
        cache_size: int = 8,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.voxel_size_l = voxel_size_l
        self.r_nav = r_nav
        self.v_max = v_max
        self.segment_duration_s_override = segment_duration_s
        self.use_obstacles = use_obstacles
        self.k_path_dir = k_path_dir
        self.history_frames = history_frames
        self.s0 = s0
        self.kappa_s = kappa_s
        self.kappa_a = kappa_a
        self.gamma = gamma
        self.lambda_max = lambda_max
        self.softmax_temp = softmax_temp
        self.samples_per_tile = samples_per_tile
        self.max_voxels_eval = max_voxels_eval
        self.min_voxel_prob = min_voxel_prob
        self.r_vis = r_vis
        self.sigma_g = sigma_g
        self.rng_seed = rng_seed
        self.max_rays_per_tile = max_rays_per_tile
        self.cache_size = max(1, int(cache_size))
        self.debug = debug

        self.logger = logging.getLogger(self.__class__.__name__)
        self._voxel_cache: OrderedDict[tuple, object] = OrderedDict()
        self._tile_sample_cache: OrderedDict[tuple, dict] = OrderedDict()

    def _cache_get(self, cache: OrderedDict, key):
        val = cache.get(key)
        if val is None:
            return None
        cache.move_to_end(key)
        return val

    def _cache_put(self, cache: OrderedDict, key, value):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > self.cache_size:
            cache.popitem(last=False)

    def _uniform_fallback(self, tiles):
        # Prism-safe fallback used when input data is missing.
        return {str(tile.unique_id): 1.0 for tile in tiles}

    def _resolve_segment_duration(self, tiles) -> float:
        if self.segment_duration_s_override is not None:
            return max(float(self.segment_duration_s_override), 1e-3)
        for tile in tiles:
            for reps in tile.component_options.values():
                if reps:
                    return max(float(reps[0].duration_s), 1e-3)
        return 1.0

    def _tile_signature(self, tiles) -> tuple:
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

    def _read_points_xyz(self, frame_path: Path) -> np.ndarray:
        ply_path = Path(frame_path)
        if ply_path.suffix.lower() != ".ply":
            ply_path = ply_path.with_suffix(ply_path.suffix + ".ply")
        points = self.ply_reader.read(str(ply_path))
        points_np = np.asarray(points)
        if points_np.ndim != 2 or points_np.shape[1] < 3:
            raise ValueError(f"PLY does not contain XYZ columns: {ply_path}")
        return np.asarray(points_np[:, :3], dtype=float)

    def _get_pose_now_and_history(self, frame_idx: int, fov_data: Fov):
        rel_idx = int(frame_idx - fov_data.start_frame_index)
        pose_now = fov_data.get_frame(rel_idx)
        if pose_now is None:
            return None, None

        if hasattr(fov_data, "get_history"):
            pose_history = fov_data.get_history(rel_idx, count_back=self.history_frames, include_current=True)
            if pose_history is not None and len(pose_history) > 0:
                return pose_now, pose_history

        # Backward-compatible fallback if helper is missing.
        start = rel_idx - max(1, int(self.history_frames)) + 1
        mask = (fov_data.df.index >= start) & (fov_data.df.index <= rel_idx)
        pose_history = fov_data.df.loc[mask]
        if pose_history is None or len(pose_history) == 0:
            pose_history = fov_data.df.loc[[rel_idx]] if rel_idx in fov_data.df.index else None
        return pose_now, pose_history

    def predict(self, tiles, frame_paths, fov_data):
        if not frame_paths:
            return self._uniform_fallback(tiles)

        target_frame_path = Path(frame_paths[-1])
        try:
            frame_idx = self._frame_index_from_path(target_frame_path)
        except Exception:
            return self._uniform_fallback(tiles)

        pose_now, pose_history = self._get_pose_now_and_history(frame_idx, fov_data)
        if pose_now is None:
            return self._uniform_fallback(tiles)

        try:
            points_xyz = self._read_points_xyz(target_frame_path)
        except Exception:
            return self._uniform_fallback(tiles)

        segment_duration_s = self._resolve_segment_duration(tiles)
        params = ReachabilityParams(
            voxel_size_l=float(self.voxel_size_l),
            r_nav=float(self.r_nav),
            v_max=float(self.v_max),
            segment_duration_s=float(segment_duration_s),
            use_obstacles=bool(self.use_obstacles),
            k_path_dir=max(1, int(self.k_path_dir)),
            history_frames=max(2, int(self.history_frames)),
            s0=float(self.s0),
            kappa_s=float(self.kappa_s),
            kappa_a=float(self.kappa_a),
            gamma=float(self.gamma),
            lambda_max=float(self.lambda_max),
            softmax_temp=max(float(self.softmax_temp), 1e-6),
            samples_per_tile=max(1, int(self.samples_per_tile)),
            max_voxels_eval=max(1, int(self.max_voxels_eval)),
            min_voxel_prob=max(float(self.min_voxel_prob), 0.0),
            r_vis=max(float(self.r_vis), 1e-6),
            sigma_g=(None if self.sigma_g is None else max(float(self.sigma_g), 1e-6)),
            rng_seed=self.rng_seed,
            max_rays_per_tile=self.max_rays_per_tile,
        )

        frame_key = str(target_frame_path.resolve())
        voxel_cache_key = (frame_key, round(params.voxel_size_l, 6), round(params.r_nav, 6))
        voxel_context = self._cache_get(self._voxel_cache, voxel_cache_key)
        if voxel_context is None:
            voxel_context = build_voxel_context(points_xyz, params)
            self._cache_put(self._voxel_cache, voxel_cache_key, voxel_context)

        sample_cache_key = (
            frame_key,
            int(params.samples_per_tile),
            int(params.rng_seed) if params.rng_seed is not None else None,
            self._tile_signature(tiles),
        )
        tile_samples = self._cache_get(self._tile_sample_cache, sample_cache_key)
        if tile_samples is None:
            from fov_predictors.raycast_visibility import sample_points_per_tile  # local import to keep module load light

            tile_samples = sample_points_per_tile(
                points_xyz=points_xyz,
                tiles=tiles,
                samples_per_tile=params.samples_per_tile,
                rng_seed=params.rng_seed,
            )
            self._cache_put(self._tile_sample_cache, sample_cache_key, tile_samples)

        tile_probs, debug_stats = compute_tile_importance(
            tiles=tiles,
            points_xyz=points_xyz,
            pose_now=pose_now,
            pose_history=pose_history,
            params=params,
            voxel_context=voxel_context,
            tile_samples=tile_samples,
        )

        if self.debug:
            top_tiles = sorted(tile_probs.items(), key=lambda kv: kv[1], reverse=True)[:5]
            self.logger.info(
                "ReachabilityRaycastPredictor frame=%s free_vox=%.0f kept_vox=%.0f c_eff=%.3f top_tiles=%s",
                target_frame_path.name,
                debug_stats.get("free_voxels", 0.0),
                debug_stats.get("kept_voxels", 0.0),
                debug_stats.get("c_eff", 0.0),
                top_tiles,
            )

        return tile_probs if tile_probs else self._uniform_fallback(tiles)
