import pandas as pd
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import numpy as np
from fov_predictors.latency import FovPredictorLatencyRecorder
from core.prediction_context import ObservedFovView, PredictionRequest
from core.rotations import ensure_fov_quaternions
from core.trace_columns import normalize_fov_dataframe
from core.world_transform_registry import apply_world_transform_to_trace

if TYPE_CHECKING:
    from fov_predictors.base import FovPredictor


class Fov:
    def __init__(
        self,
        csv_path: str,
        predictor: "FovPredictor",
        start_frame_index: int = 0,
        latency_recorder: FovPredictorLatencyRecorder | None = None,
        content_dir: str | Path | None = None,
        motion_limits: dict | None = None,
        frame_rate_hz: float = 30.0,
    ) -> None:
        """
        Initialize Fov object with a CSV file path.
        Loads the CSV into memory.
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        self.df: pd.DataFrame = normalize_fov_dataframe(pd.read_csv(self.csv_path), source=self.csv_path)
        if content_dir is not None:
            self.df = apply_world_transform_to_trace(self.csv_path, self.df, content_dir)
        angle_convention = getattr(predictor, "angle_convention", None)
        self.df = ensure_fov_quaternions(self.df, angle_convention=angle_convention)
        self.df.set_index("FrameNumber", inplace=True)
        self.df.index = self.df.index.astype(int) - int(start_frame_index)
        self.predictor = predictor
        self.start_frame_index = start_frame_index
        self.latency_recorder = latency_recorder
        self.motion_limits = dict(motion_limits or {})
        self.frame_rate_hz = max(float(frame_rate_hz), 1e-6)
        self._tile_prediction_cache_key = None
        self._tile_prediction_cache_value = None

    def get_frame(self, frame_number: int) -> Optional[pd.Series]:
        """
        Get a specific frame row by frame_number.
        Returns a pandas Series if found, else None.
        """
        try:
            return self.df.loc[frame_number]
        except KeyError:
            print(f"Frame {frame_number} not found in {self.csv_path.name}.")
            return None

    def get_total_frames(self) -> int:
        """
        Return total number of frames in this CSV.
        """
        return len(self.df)

    def get_frames_range(self, start_frame: int, end_frame: int) -> pd.DataFrame:
        """
        Return all rows whose FrameNumber index is in [start_frame, end_frame].
        Missing frames are skipped.
        """
        if end_frame < start_frame:
            return self.df.iloc[0:0].copy()
        mask = (self.df.index >= int(start_frame)) & (self.df.index <= int(end_frame))
        return self.df.loc[mask].copy()

    def get_history(self, frame_number: int, count_back: int, include_current: bool = True) -> pd.DataFrame:
        """
        Return a backward window ending at frame_number.
        If include_current=False, history ends at frame_number-1.
        """
        if count_back <= 0:
            return self.df.iloc[0:0].copy()
        end = int(frame_number) if include_current else int(frame_number) - 1
        start = end - int(count_back) + 1
        return self.get_frames_range(start, end)

    def get_all_frames(self) -> pd.DataFrame:
        """
        Return the full dataframe (all frames).
        """
        return self.df.copy()

    def _tile_prediction_key(self, tiles, frame_paths, request: PredictionRequest | None = None) -> tuple | None:
        if not frame_paths:
            return None
        if self.predictor is None:
            return None
        target_frame = str(Path(frame_paths[-1]).resolve())
        tile_ids = tuple(sorted(str(getattr(tile, "unique_id", getattr(tile, "tile_id", ""))) for tile in tiles))
        request_key = None
        if request is not None:
            request_key = (
                int(request.target_segment_idx),
                None if request.observed_frame_idx is None else int(request.observed_frame_idx),
                round(float(request.prediction_horizon_s), 6),
                round(float(request.target_segment_duration_s), 6),
                tuple(str(Path(path).resolve()) for path in request.geometry_frame_paths),
            )
        return (
            self.predictor.__class__.__name__,
            target_frame,
            len(frame_paths),
            tile_ids,
            request_key,
        )

    def predict_tiles(self, tiles, frame_paths, request: PredictionRequest | None = None):
        """
        Predict per-tile visibility/importance for a concrete tile list and frame window.
        Reuses the most recent identical request to avoid duplicate predictor work within one segment.
        """
        predictor = getattr(self, "predictor", None)
        has_oracle_target = bool(
            request
            and getattr(predictor, "uses_future_target_frames", False)
            and request.oracle_target_frame_paths
        )
        if predictor is None or (not frame_paths and not has_oracle_target):
            return None

        if request is None:
            request = getattr(self, "_active_prediction_request", None)
        cache_key = self._tile_prediction_key(tiles, frame_paths, request)
        if cache_key is not None and cache_key == self._tile_prediction_cache_key:
            cached = self._tile_prediction_cache_value
            return dict(cached) if isinstance(cached, dict) else cached

        if request is not None:
            observed_fov = ObservedFovView(self, request.observed_frame_idx)
            if hasattr(predictor, "predict_for_request"):
                result = predictor.predict_for_request(tiles, request, observed_fov)
            else:
                result = predictor.predict(tiles, request.geometry_frame_paths, observed_fov)
        else:
            result = predictor.predict(tiles, frame_paths, self)
        if cache_key is not None:
            self._tile_prediction_cache_key = cache_key
            self._tile_prediction_cache_value = dict(result) if isinstance(result, dict) else result
        return result
    
    def predict(self, frame_number: int):
        """
        Use the predictor to get position and forward vector for a given frame.
        """
        return self.predictor.predict(frame_number, self)
