from fov_predictors.base import FovPredictor
from core.fov import Fov
from scipy.spatial.transform import Rotation as R
import numpy as np
import time
from pathlib import Path


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
