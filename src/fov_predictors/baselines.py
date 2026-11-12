from fov_predictors.base import FovPredictor
from core.fov import Fov
from core.tile_fov import TileFovTrace, has_tile_fov_columns, reduce_tile_visibility_frames
import numpy as np
from pathlib import Path
from training.adapters import TrainedModelRuntime, fov_history_array, frame_index_from_path

from fov_predictors.raycast.predictor import ReachabilityRaycastPredictor as _ReachabilityRaycastPredictor


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

    def required_ply_frame_paths(self, frame_paths):
        return [frame_paths[-1]] if frame_paths else []

    def predict(self, tiles, frame_paths, fov_data):
        session = self._latency_session(fov_data)
        metadata = {"tile_count": len(tiles), "frame_count": len(frame_paths)}
        try:
            with session.stage("frame_lookup"):
                frame = self._get_frame(frame_paths[-1], fov_data)
            with session.stage("compute_forward"):
                forward, up = self._compute_orientation(frame)
            with session.stage("read_point_cloud"):
                points = self._read_ply(frame_paths[-1])
            with session.stage("points_in_fov"):
                mask = self._points_in_fov(
                    points,
                    frame[['HMDPX','HMDPY','HMDPZ']].to_numpy(),
                    forward,
                    euler_angles=None,
                    up=up,
                    fov_h=self.fov_h,
                    fov_v=self.fov_v,
                )
            with session.stage("aggregate_tiles"):
                return self.tile_fov_proportion(tiles, mask, points)
        finally:
            session.finish(metadata=metadata)

    def predict_for_request(self, tiles, request, fov_data):
        session = self._latency_session(fov_data)
        frame_paths = list(request.geometry_frame_paths or [])
        metadata = {
            "tile_count": len(tiles),
            "frame_count": len(frame_paths),
            "observed_frame_idx": request.observed_frame_idx,
            "prediction_horizon_s": float(request.prediction_horizon_s),
        }
        try:
            if not frame_paths or request.observed_frame_idx is None:
                metadata["fallback"] = "all_visible"
                return {tile.unique_id: 1.0 for tile in tiles}
            with session.stage("frame_lookup"):
                frame = self._get_observed_frame(request.observed_frame_idx, fov_data)
            if frame is None:
                metadata["fallback"] = "missing_observed_frame"
                return {tile.unique_id: 1.0 for tile in tiles}
            with session.stage("predict_from_observed_pose"):
                return self._predict_frame_impl(tiles, frame, Path(frame_paths[-1]))
        finally:
            session.finish(metadata=metadata)

    def _predict_frame_impl(self, tiles, frame, frame_path: Path):
        forward, up = self._compute_orientation(frame)
        points = self._read_ply(frame_path)
        mask = self._points_in_fov(
            points,
            frame[['HMDPX', 'HMDPY', 'HMDPZ']].to_numpy(),
            forward,
            euler_angles=None,
            up=up,
            fov_h=self.fov_h,
            fov_v=self.fov_v,
        )
        return self.tile_fov_proportion(tiles, mask, points)

    def _get_frame(self, frame_path: Path, fov_data: Fov):
        frame_idx = self._frame_index_from_path(frame_path)
        return fov_data.get_frame(frame_idx - fov_data.start_frame_index)

    def _get_observed_frame(self, frame_idx: int, fov_data: Fov):
        return fov_data.get_frame(int(frame_idx) - int(fov_data.start_frame_index))

    def _frame_index_from_path(self, frame_path: Path) -> int:
        stem = Path(frame_path).stem
        digits = "".join(ch for ch in stem if ch.isdigit())
        if not digits:
            raise ValueError(f"Could not extract frame index from {frame_path}")
        return int(digits)

    def _compute_forward(self, frame):
        return self._forward_from_frame(frame)

    def _compute_orientation(self, frame):
        quat = self._quaternion_from_frame(frame)
        return self.forward_from_quaternion_xyzw(quat), self.up_from_quaternion_xyzw(quat)

    def _read_ply(self, frame_path: Path):
        ply_path = Path(frame_path)
        if ply_path.suffix.lower() != ".ply":
            ply_path = ply_path.with_suffix(ply_path.suffix + ".ply")
        points = self.ply_reader.read(str(ply_path))
        return points


class GroundTruthFovPredictor(LastObservedFovPredictor):
    """
    Oracle predictor: compute actual tile visibility from the target segment's
    ground-truth FoV trace and original point-cloud frames.
    """

    uses_ground_truth_fov = True
    PARAM_METADATA = {
        "sample_stride": {
            "label": "Sample stride",
            "category": "Oracle",
            "help": "Use every Nth target frame when averaging oracle FoV visibility.",
        },
    }

    def __init__(self, sample_stride: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.sample_stride = max(1, int(sample_stride or 1))

    def required_ply_frame_paths(self, frame_paths):
        return list(frame_paths or [])[:: self.sample_stride]

    def predict(self, tiles, frame_paths, fov_data):
        session = self._latency_session(fov_data)
        sampled_paths = list(frame_paths or [])[:: self.sample_stride]
        metadata = {
            "tile_count": len(tiles),
            "frame_count": len(frame_paths or []),
            "sampled_frame_count": len(sampled_paths),
            "sample_stride": int(self.sample_stride),
            "oracle": True,
        }
        try:
            if not sampled_paths:
                metadata["fallback"] = "all_visible"
                return {tile.unique_id: 1.0 for tile in tiles}

            totals = {tile.unique_id: 0.0 for tile in tiles}
            used = 0
            for frame_path in sampled_paths:
                with session.stage("frame_lookup"):
                    frame = self._get_frame(frame_path, fov_data)
                if frame is None:
                    continue
                with session.stage("compute_forward"):
                    forward, up = self._compute_orientation(frame)
                with session.stage("read_point_cloud"):
                    points = self._read_ply(frame_path)
                with session.stage("points_in_fov"):
                    mask = self._points_in_fov(
                        points,
                        frame[["HMDPX", "HMDPY", "HMDPZ"]].to_numpy(),
                        forward,
                        euler_angles=None,
                        up=up,
                        fov_h=self.fov_h,
                        fov_v=self.fov_v,
                    )
                with session.stage("aggregate_tiles"):
                    proportions = self.tile_fov_proportion(tiles, mask, points)
                for tile_id, value in proportions.items():
                    totals[tile_id] = totals.get(tile_id, 0.0) + float(value or 0.0)
                used += 1

            if used <= 0:
                metadata["fallback"] = "all_visible"
                return {tile.unique_id: 1.0 for tile in tiles}
            metadata["used_frame_count"] = used
            return {tile_id: value / used for tile_id, value in totals.items()}
        finally:
            session.finish(metadata=metadata)


class GroundTruthTileFovPredictor(FrameIndexMixin, FovPredictor):
    """
    Strong oracle predictor: read future per-tile FoV scores embedded in the
    participant FoV trace and average them over the target segment frames.
    """

    uses_ground_truth_fov = True
    # Player only attaches future target frames to predictors with this marker.
    uses_future_target_frames = True
    PARAM_METADATA = {
        "sample_stride": {
            "label": "Sample stride",
            "category": "Oracle",
            "help": "Use every Nth target frame when averaging embedded tile FoV scores.",
        },
    }

    def __init__(self, sample_stride: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.sample_stride = max(1, int(sample_stride or 1))
        self._trace_path: Path | None = None
        self._tile_trace: TileFovTrace | None = None

    def _tile_fov_trace(self, fov_data: Fov) -> TileFovTrace | None:
        raw_path = getattr(fov_data, "csv_path", None)
        if not raw_path:
            self._trace_path = None
            self._tile_trace = None
            return None
        trace_path = Path(raw_path)
        if self._tile_trace is not None and self._trace_path == trace_path:
            return self._tile_trace
        if not has_tile_fov_columns(trace_path):
            self._trace_path = trace_path
            self._tile_trace = None
            return None
        self._trace_path = trace_path
        self._tile_trace = TileFovTrace(trace_path, require_tile_columns=False)
        return self._tile_trace

    def predict(self, tiles, frame_paths, fov_data):
        session = self._latency_session(fov_data)
        sampled_paths = list(frame_paths or [])[:: self.sample_stride]
        tile_ids = [str(getattr(tile, "unique_id", getattr(tile, "tile_id", ""))) for tile in tiles]
        metadata = {
            "tile_count": len(tiles),
            "frame_count": len(frame_paths or []),
            "sampled_frame_count": len(sampled_paths),
            "sample_stride": int(self.sample_stride),
            "oracle": True,
            "source": "embedded_tile_fov",
        }
        try:
            if not sampled_paths:
                metadata["fallback"] = "all_visible"
                return {tile_id: 1.0 for tile_id in tile_ids}

            with session.stage("load_tile_fov_trace"):
                tile_trace = self._tile_fov_trace(fov_data)
            if tile_trace is None:
                metadata["fallback"] = "missing_embedded_tile_fov"
                return {tile_id: 1.0 for tile_id in tile_ids}

            with session.stage("lookup_tile_scores"):
                return tile_trace.visibility_for_frame_paths(sampled_paths, tile_ids=tile_ids)
        finally:
            session.finish(metadata=metadata)

    def predict_for_request(self, tiles, request, fov_data):
        session = self._latency_session(fov_data)
        tile_ids = [str(getattr(tile, "unique_id", getattr(tile, "tile_id", ""))) for tile in tiles]
        metadata = {
            "tile_count": len(tiles),
            "frame_count": len(request.oracle_target_frame_paths or []),
            "sample_stride": int(self.sample_stride),
            "source": "embedded_tile_fov_target_segment",
            "observed_frame_idx": request.observed_frame_idx,
        }
        try:
            target_paths = list(request.oracle_target_frame_paths or [])
            if not target_paths:
                # This preserves the old safe fallback for callers that do not
                # opt into the oracle-only request fields.
                metadata["fallback"] = "missing_oracle_target_frames"
                return {tile_id: 1.0 for tile_id in tile_ids}
            with session.stage("load_tile_fov_trace"):
                tile_trace = self._tile_fov_trace(fov_data)
            if tile_trace is None:
                metadata["fallback"] = "missing_embedded_tile_fov"
                return {tile_id: 1.0 for tile_id in tile_ids}
            with session.stage("lookup_tile_scores"):
                frames = tile_trace.visibility_frames_for_frame_paths(target_paths, tile_ids=tile_ids)
                return reduce_tile_visibility_frames(
                    frames,
                    tile_ids=tile_ids,
                    mode=request.ground_truth_mode or "visible_fraction",
                    visible_threshold=0.01 if request.ground_truth_visible_threshold is None else request.ground_truth_visible_threshold,
                    peak_weight=0.3 if request.ground_truth_peak_weight is None else request.ground_truth_peak_weight,
                )
        finally:
            session.finish(metadata=metadata)


class UniformTileImportancePredictor(FovPredictor):
    """Assign the same independent visibility score to every tile."""

    PARAM_METADATA = {
        "score": {
            "label": "Tile score",
            "category": "Uniform baseline",
            "help": "Independent FoV score assigned to every tile. Values are clamped to [0, 1].",
        },
    }

    def __init__(self, score: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.score = float(np.clip(float(score), 0.0, 1.0))

    def required_ply_frame_paths(self, frame_paths):
        return []

    def predict(self, tiles, frame_paths, fov_data):
        session = self._latency_session(fov_data)
        metadata = {"tile_count": len(tiles), "frame_count": len(frame_paths or [])}
        try:
            if not tiles:
                return {}
            return {str(getattr(tile, "unique_id", getattr(tile, "tile_id", idx))): self.score for idx, tile in enumerate(tiles)}
        finally:
            session.finish(metadata=metadata)


class DistanceBasedTileImportancePredictor(FrameIndexMixin, FovPredictor):
    """Score nearby tiles by an exponential distance kernel around the current user position."""

    PARAM_METADATA = {
        "sigma": {
            "label": "Distance sigma",
            "category": "Distance baseline",
            "help": "Meters for the exponential distance falloff. Empty or non-positive values use 1 meter.",
        },
    }

    def __init__(self, sigma: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.sigma = max(float(sigma or 1.0), 1e-6)

    def required_ply_frame_paths(self, frame_paths):
        return []

    def predict(self, tiles, frame_paths, fov_data):
        session = self._latency_session(fov_data)
        metadata = {"tile_count": len(tiles), "frame_count": len(frame_paths or []), "sigma": float(self.sigma)}
        try:
            if not tiles:
                return {}
            with session.stage("user_position"):
                position = self._current_position(frame_paths, fov_data)
            with session.stage("distance_scores"):
                scores = {}
                for idx, tile in enumerate(tiles):
                    tile_id = str(getattr(tile, "unique_id", getattr(tile, "tile_id", idx)))
                    distance = self._distance_to_tile(position, tile)
                    scores[tile_id] = float(np.clip(np.exp(-distance / self.sigma), 0.0, 1.0))
            return scores
        finally:
            session.finish(metadata=metadata)

    def predict_for_request(self, tiles, request, fov_data):
        session = self._latency_session(fov_data)
        metadata = {
            "tile_count": len(tiles),
            "frame_count": len(request.geometry_frame_paths or []),
            "sigma": float(self.sigma),
            "observed_frame_idx": request.observed_frame_idx,
        }
        try:
            if not tiles:
                return {}
            with session.stage("user_position"):
                position = self._position_for_observed_frame(request.observed_frame_idx, fov_data)
            with session.stage("distance_scores"):
                scores = {}
                for idx, tile in enumerate(tiles):
                    tile_id = str(getattr(tile, "unique_id", getattr(tile, "tile_id", idx)))
                    distance = self._distance_to_tile(position, tile)
                    scores[tile_id] = float(np.clip(np.exp(-distance / self.sigma), 0.0, 1.0))
            return scores
        finally:
            session.finish(metadata=metadata)

    def _current_position(self, frame_paths, fov_data) -> np.ndarray:
        if frame_paths:
            try:
                frame_idx = self._frame_index_from_path(frame_paths[-1])
                frame = fov_data.get_frame(frame_idx - fov_data.start_frame_index)
                if frame is not None:
                    return frame[["HMDPX", "HMDPY", "HMDPZ"]].to_numpy(dtype=float)
            except Exception:  # pylint: disable=broad-except
                pass
        return np.zeros(3, dtype=float)

    def _position_for_observed_frame(self, observed_frame_idx, fov_data) -> np.ndarray:
        if observed_frame_idx is not None:
            try:
                frame = fov_data.get_frame(int(observed_frame_idx) - int(fov_data.start_frame_index))
                if frame is not None:
                    return frame[["HMDPX", "HMDPY", "HMDPZ"]].to_numpy(dtype=float)
            except Exception:  # pylint: disable=broad-except
                pass
        return np.zeros(3, dtype=float)

    @staticmethod
    def _distance_to_tile(position: np.ndarray, tile) -> float:
        mins = np.asarray(getattr(tile, "min_corner", (0.0, 0.0, 0.0)), dtype=float)
        maxs = np.asarray(getattr(tile, "max_corner", mins), dtype=float)
        clipped = np.minimum(np.maximum(position, mins), maxs)
        return float(np.linalg.norm(position - clipped))


class AverageMotionPredictor(FrameIndexMixin, FovPredictor):
    """Predict FoV as average motion over last `lookback` frames."""

    def __init__(self, lookback: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.lookback = lookback
        self.last_observed = LastObservedFovPredictor(ply_reader=self.ply_reader,
                                                      fov_h=self.fov_h, fov_v=self.fov_v)

    def required_ply_frame_paths(self, frame_paths):
        return [frame_paths[-1]] if frame_paths else []

    def predict(self, tiles, frame_paths, fov_data):
        session = self._latency_session(fov_data)
        metadata = {"tile_count": len(tiles), "frame_count": len(frame_paths), "lookback": int(self.lookback)}
        try:
            with session.stage("history_lookup"):
                frames = self._get_recent_frames(frame_paths[-1], fov_data)
            if not frames:
                metadata["fallback"] = "last_observed"
                with session.stage("fallback_last_observed"):
                    frame = self.last_observed._get_frame(frame_paths[-1], fov_data)
                    return self.last_observed._predict_frame_impl(tiles, frame, frame_paths[-1])

            with session.stage("compute_average_motion"):
                position, forward = self._compute_average_motion(frames)
            with session.stage("read_point_cloud"):
                points = self.last_observed._read_ply(frame_paths[-1])
            with session.stage("points_in_fov"):
                mask = self.last_observed._points_in_fov(
                    points,
                    position,
                    forward,
                    euler_angles=None,
                    fov_h=self.fov_h,
                    fov_v=self.fov_v,
                )
            with session.stage("aggregate_tiles"):
                return self.last_observed.tile_fov_proportion(tiles, mask, points)
        finally:
            session.finish(metadata=metadata)

    def predict_for_request(self, tiles, request, fov_data):
        session = self._latency_session(fov_data)
        frame_paths = list(request.geometry_frame_paths or [])
        metadata = {
            "tile_count": len(tiles),
            "frame_count": len(frame_paths),
            "lookback": int(self.lookback),
            "observed_frame_idx": request.observed_frame_idx,
        }
        try:
            if not frame_paths or request.observed_frame_idx is None:
                metadata["fallback"] = "all_visible"
                return {tile.unique_id: 1.0 for tile in tiles}
            with session.stage("history_lookup"):
                frames = self._get_recent_observed_frames(request.observed_frame_idx, fov_data)
            if not frames:
                metadata["fallback"] = "last_observed"
                with session.stage("fallback_last_observed"):
                    frame = self.last_observed._get_observed_frame(request.observed_frame_idx, fov_data)
                    return self.last_observed._predict_frame_impl(tiles, frame, Path(frame_paths[-1]))

            with session.stage("compute_average_motion"):
                position, forward = self._compute_average_motion(frames)
            with session.stage("read_point_cloud"):
                points = self.last_observed._read_ply(Path(frame_paths[-1]))
            with session.stage("points_in_fov"):
                mask = self.last_observed._points_in_fov(
                    points,
                    position,
                    forward,
                    euler_angles=None,
                    fov_h=self.fov_h,
                    fov_v=self.fov_v,
                )
            with session.stage("aggregate_tiles"):
                return self.last_observed.tile_fov_proportion(tiles, mask, points)
        finally:
            session.finish(metadata=metadata)

    def _get_recent_frames(self, frame_path: Path, fov_data: Fov):
        frame_idx = self._frame_index_from_path(frame_path)
        start = max(0, frame_idx - self.lookback)
        frames = [fov_data.get_frame(i - fov_data.start_frame_index) for i in range(start, frame_idx)]
        return [f for f in frames if f is not None]

    def _get_recent_observed_frames(self, observed_frame_idx: int, fov_data: Fov):
        frame_idx = int(observed_frame_idx)
        start = max(int(fov_data.start_frame_index), frame_idx - int(self.lookback))
        frames = [fov_data.get_frame(i - int(fov_data.start_frame_index)) for i in range(start, frame_idx + 1)]
        return [f for f in frames if f is not None]

    def _compute_average_motion(self, frames):
        positions = np.array([f[['HMDPX','HMDPY','HMDPZ']].to_numpy() for f in frames])
        avg_velocity = (positions[-1] - positions[0]) / max(1, len(positions) - 1)
        forward = avg_velocity / np.linalg.norm(avg_velocity) if np.linalg.norm(avg_velocity) > 0 else np.array([0, 0, 1])
        position = frames[-1][['HMDPX','HMDPY','HMDPZ']].to_numpy()
        return position, forward

class LinearExtrapolationPredictor(FrameIndexMixin, FovPredictor):
    """Predict FoV by extrapolating direction estimated from a recent time window."""

    DEFAULT_FRAME_RATE_HZ = 30.0
    PARAM_METADATA = {
        "lookback_s": {
            "label": "Lookback window (s)",
            "category": "History",
            "help": "Duration of observed pose history used to estimate the linear viewing-direction trend.",
        },
        "lookback": {
            "hidden": True,
        },
    }

    def __init__(self, lookback_s: float = 2.0, lookback: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.lookback_s = max(0.0, float(lookback_s))
        # Preserve historical saved runs exactly: their ``lookback`` value was
        # a frame count.  New presets must use lookback_s.
        self.legacy_lookback_frames = None if lookback is None else max(1, int(lookback))
        self.last_observed = LastObservedFovPredictor(ply_reader=self.ply_reader,
                                                      fov_h=self.fov_h, fov_v=self.fov_v)

    def _lookback_frame_count(self, fov_data=None) -> int:
        if self.legacy_lookback_frames is not None:
            return self.legacy_lookback_frames
        fps = max(float(getattr(fov_data, "frame_rate_hz", self.DEFAULT_FRAME_RATE_HZ)), 1e-6)
        return max(1, int(round(self.lookback_s * fps)))

    def _history_metadata(self, fov_data=None) -> dict:
        frame_count = self._lookback_frame_count(fov_data)
        return {
            "lookback_s": float(self.lookback_s),
            "lookback_frames": frame_count,
            "frame_rate_hz": max(float(getattr(fov_data, "frame_rate_hz", self.DEFAULT_FRAME_RATE_HZ)), 1e-6),
            "legacy_frame_lookback": self.legacy_lookback_frames is not None,
        }

    def required_ply_frame_paths(self, frame_paths):
        return [frame_paths[-1]] if frame_paths else []

    def predict(self, tiles, frame_paths, fov_data):
        session = self._latency_session(fov_data)
        metadata = {"tile_count": len(tiles), "frame_count": len(frame_paths), **self._history_metadata(fov_data)}
        try:
            with session.stage("history_lookup"):
                frames = self._get_recent_frames(frame_paths[-1], fov_data)
            if len(frames) < 2:
                metadata["fallback"] = "last_observed"
                with session.stage("fallback_last_observed"):
                    frame = self.last_observed._get_frame(frame_paths[-1], fov_data)
                    return self.last_observed._predict_frame_impl(tiles, frame, frame_paths[-1])

            with session.stage("extrapolate_forward"):
                position, forward = self._extrapolate_forward(frames)
            with session.stage("read_point_cloud"):
                points = self.last_observed._read_ply(frame_paths[-1])
            with session.stage("points_in_fov"):
                mask = self.last_observed._points_in_fov(
                    points,
                    position,
                    forward,
                    euler_angles=None,
                    fov_h=self.fov_h,
                    fov_v=self.fov_v,
                )
            with session.stage("aggregate_tiles"):
                return self.last_observed.tile_fov_proportion(tiles, mask, points)
        finally:
            session.finish(metadata=metadata)

    def predict_for_request(self, tiles, request, fov_data):
        session = self._latency_session(fov_data)
        frame_paths = list(request.geometry_frame_paths or [])
        metadata = {
            "tile_count": len(tiles),
            "frame_count": len(frame_paths),
            **self._history_metadata(fov_data),
            "observed_frame_idx": request.observed_frame_idx,
        }
        try:
            if not frame_paths or request.observed_frame_idx is None:
                metadata["fallback"] = "all_visible"
                return {tile.unique_id: 1.0 for tile in tiles}
            with session.stage("history_lookup"):
                frames = self._get_recent_observed_frames(request.observed_frame_idx, fov_data)
            if len(frames) < 2:
                metadata["fallback"] = "last_observed"
                with session.stage("fallback_last_observed"):
                    frame = self.last_observed._get_observed_frame(request.observed_frame_idx, fov_data)
                    return self.last_observed._predict_frame_impl(tiles, frame, Path(frame_paths[-1]))

            with session.stage("extrapolate_forward"):
                position, forward = self._extrapolate_forward(frames)
            with session.stage("read_point_cloud"):
                points = self.last_observed._read_ply(Path(frame_paths[-1]))
            with session.stage("points_in_fov"):
                mask = self.last_observed._points_in_fov(
                    points,
                    position,
                    forward,
                    euler_angles=None,
                    fov_h=self.fov_h,
                    fov_v=self.fov_v,
                )
            with session.stage("aggregate_tiles"):
                return self.last_observed.tile_fov_proportion(tiles, mask, points)
        finally:
            session.finish(metadata=metadata)

    def _get_recent_frames(self, frame_path: Path, fov_data: Fov):
        frame_idx = self._frame_index_from_path(frame_path)
        start = max(0, frame_idx - self._lookback_frame_count(fov_data))
        frames = [fov_data.get_frame(i - fov_data.start_frame_index) for i in range(start, frame_idx)]
        return [f for f in frames if f is not None]

    def _get_recent_observed_frames(self, observed_frame_idx: int, fov_data: Fov):
        frame_idx = int(observed_frame_idx)
        start = max(int(fov_data.start_frame_index), frame_idx - self._lookback_frame_count(fov_data))
        frames = [fov_data.get_frame(i - int(fov_data.start_frame_index)) for i in range(start, frame_idx + 1)]
        return [f for f in frames if f is not None]

    def _extrapolate_forward(self, frames):
        forwards = []
        for f in frames:
            forwards.append(self._forward_from_frame(f))
        forwards = np.stack(forwards)
        delta = forwards[-1] - forwards[0]
        forward = forwards[-1] + delta
        forward /= np.linalg.norm(forward)
        position = frames[-1][['HMDPX','HMDPY','HMDPZ']].to_numpy()
        return position, forward


class LinearPoseExtrapolationPredictor(LinearExtrapolationPredictor):
    """Linear baseline that extrapolates both viewing direction and position.

    It deliberately retains :class:`LinearExtrapolationPredictor`'s history
    window, direction calculation, and fallback behaviour.  The only change
    is that the displacement over the observed window is applied once more to
    the latest position, matching the existing forward-vector extrapolation.
    """

    def _extrapolate_forward(self, frames):
        position, forward = super()._extrapolate_forward(frames)
        first_position = frames[0][['HMDPX', 'HMDPY', 'HMDPZ']].to_numpy(dtype=float)
        latest_position = np.asarray(position, dtype=float)
        return latest_position + (latest_position - first_position), forward


class ReachabilityRaycastPredictor(_ReachabilityRaycastPredictor):
    """
    Argus: advanced reachability + visibility model.

    Defined here as a thin wrapper so predictor discovery in `server.py`
    continues to work without changing UI option loading.
    """

    DISPLAY_NAME = "Argus"


class TrainedFovPredictor(FovPredictor):
    PARAM_METADATA = {
        "model_id": {
            "label": "Exported model id",
            "category": "Trained model",
            "help": "Model id from the Training module export list.",
        },
    }

    def __init__(self, model_id: str = "", device: str = "auto", **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.runtime = TrainedModelRuntime(model_id=model_id, device=device)
        self._last_observed = LastObservedFovPredictor(
            ply_reader=self.ply_reader,
            fov_h=self.fov_h,
            fov_v=self.fov_v,
            angle_convention=self.angle_convention,
        )

    def required_ply_frame_paths(self, frame_paths):
        return [frame_paths[-1]] if frame_paths else []

    def _predict_from_anchor(self, tiles, geometry_frame_path: Path, anchor: int, fov_data, horizon_s: float | None = None):
        session = self._latency_session(fov_data)
        metadata = {
            "tile_count": len(tiles),
            "frame_count": 1,
            "model_id": self.model_id,
            "anchor_frame": int(anchor),
        }
        try:
            model_horizon_s = float((self.runtime.history or {}).get("prediction_horizon_s") or 1.0)
            effective_horizon_s = float(horizon_s if horizon_s is not None else model_horizon_s)
            frame_rate_hz = max(float(getattr(fov_data, "frame_rate_hz", 30.0)), 1e-6)
            horizon_frames = max(1, int(round(effective_horizon_s * frame_rate_hz)))
            metadata["target_frame"] = int(anchor + horizon_frames)
            metadata["horizon_frames"] = int(horizon_frames)
            metadata["horizon_s"] = float(effective_horizon_s)
            metadata["frame_rate_hz"] = frame_rate_hz
            with session.stage("history"):
                history = fov_history_array(self.runtime, fov_data, int(anchor))
            with session.stage("model"):
                pose = self.runtime.predict_array(history)
            if (self.runtime.stats or {}).get("target_kind") == "tile_visibility":
                tile_ids = [str(getattr(tile, "unique_id", getattr(tile, "tile_id", ""))) for tile in tiles]
                output_ids = [str(value) for value in ((self.runtime.stats or {}).get("output_tile_ids") or [])]
                values = np.clip(np.asarray(pose, dtype=float), 0.0, 1.0)
                by_id = {tile_id: float(values[idx]) for idx, tile_id in enumerate(output_ids) if idx < len(values)}
                return {tile_id: by_id.get(tile_id, 0.0) for tile_id in tile_ids}
            with session.stage("read_point_cloud"):
                points = self._last_observed._read_ply(geometry_frame_path)
            with session.stage("points_in_fov"):
                quaternion = pose[3:7]
                forward = self.forward_from_quaternion_xyzw(quaternion)
                up = self.up_from_quaternion_xyzw(quaternion)
                mask = self._points_in_fov(
                    points,
                    np.asarray(pose[:3], dtype=float),
                    forward,
                    euler_angles=None,
                    up=up,
                    fov_h=self.fov_h,
                    fov_v=self.fov_v,
                )
            with session.stage("aggregate_tiles"):
                return self.tile_fov_proportion(tiles, mask, points)
        except Exception as exc:  # pylint: disable=broad-except
            metadata["fallback"] = str(exc)
            frame = self._last_observed._get_observed_frame(int(anchor), fov_data)
            if frame is None:
                return {tile.unique_id: 1.0 for tile in tiles}
            return self._last_observed._predict_frame_impl(tiles, frame, geometry_frame_path)
        finally:
            session.finish(metadata=metadata)

    def predict(self, tiles, frame_paths, fov_data):
        if not frame_paths:
            return {tile.unique_id: 1.0 for tile in tiles}
        return self._predict_from_anchor(
            tiles=tiles,
            geometry_frame_path=Path(frame_paths[-1]),
            anchor=frame_index_from_path(frame_paths[-1]),
            fov_data=fov_data,
        )

    def predict_for_request(self, tiles, request, fov_data):
        frame_paths = list(request.geometry_frame_paths or [])
        if not frame_paths or request.observed_frame_idx is None:
            return {tile.unique_id: 1.0 for tile in tiles}
        return self._predict_from_anchor(
            tiles=tiles,
            geometry_frame_path=Path(frame_paths[-1]),
            anchor=int(request.observed_frame_idx),
            fov_data=fov_data,
            horizon_s=float(request.prediction_horizon_s),
        )
