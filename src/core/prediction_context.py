from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from core.fov import Fov


@dataclass(frozen=True)
class PredictionRequest:
    """Inputs available to a FoV predictor at one ABR decision point."""

    target_segment_idx: int
    geometry_frame_paths: list[Path]
    observed_frame_idx: int | None
    prediction_horizon_s: float
    target_segment_duration_s: float
    # Present only for explicit offline/oracle predictors. Normal predictors
    # must continue to use observed FoV plus geometry_frame_paths.
    oracle_target_frame_paths: list[Path] | None = None
    ground_truth_mode: str | None = None
    ground_truth_visible_threshold: float | None = None
    ground_truth_peak_weight: float | None = None


class ObservedFovView:
    """Read-only FoV view that refuses frames beyond playback observation."""

    def __init__(self, fov: "Fov", max_observed_frame_idx: int | None):
        self.fov = fov
        self.max_observed_frame_idx = max_observed_frame_idx
        self.csv_path = fov.csv_path
        self.predictor = fov.predictor
        self.start_frame_index = fov.start_frame_index
        self.latency_recorder = fov.latency_recorder
        self.motion_limits = dict(getattr(fov, "motion_limits", {}) or {})
        self.frame_rate_hz = max(float(getattr(fov, "frame_rate_hz", 30.0)), 1e-6)
        self.shared_raycast_cache = getattr(fov, "shared_raycast_cache", None)

    @property
    def df(self) -> pd.DataFrame:
        if self.max_observed_frame_idx is None:
            return self.fov.df.iloc[0:0]
        return self.fov.df.loc[self.fov.df.index <= self._relative_idx(self.max_observed_frame_idx)]

    def _relative_idx(self, absolute_frame_idx: int) -> int:
        return int(absolute_frame_idx) - int(self.start_frame_index)

    def _absolute_idx(self, frame_number: int) -> int:
        return int(frame_number) + int(self.start_frame_index)

    def get_frame(self, frame_number: int):
        if self.max_observed_frame_idx is None:
            return None
        if self._absolute_idx(frame_number) > int(self.max_observed_frame_idx):
            return None
        return self.fov.get_frame(frame_number)

    def get_history(self, frame_number: int, count_back: int, include_current: bool = True) -> pd.DataFrame:
        if count_back <= 0 or self.max_observed_frame_idx is None:
            return self.fov.df.iloc[0:0].copy()
        capped = min(int(frame_number), self._relative_idx(int(self.max_observed_frame_idx)))
        return self.fov.get_history(capped, count_back, include_current=include_current)

    def get_frames_range(self, start_frame: int, end_frame: int) -> pd.DataFrame:
        if self.max_observed_frame_idx is None:
            return self.fov.df.iloc[0:0].copy()
        capped_end = min(int(end_frame), self._relative_idx(int(self.max_observed_frame_idx)))
        return self.fov.get_frames_range(start_frame, capped_end)

    def get_total_frames(self) -> int:
        return len(self.df)
