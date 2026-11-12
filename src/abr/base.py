from abc import ABC, abstractmethod
from typing import List, Dict
from core.media_segment import TileComponentOptions as Tile
from core.network import Network
from core.fov import Fov
from core.buffer import Buffer
from pathlib import Path
from core.clock import Clock
class ABR(ABC):
    """Abstract Base Class for Adaptive Bitrate (ABR) algorithms for VPCC streaming."""
    abr_requires_predictor = False
    abr_supports_predictor = False

    @abstractmethod
    def decide(
        self,
        tiles: List[Tile],
        segment_frames_path: List[Path],
        network: Network,
        fov: Fov,
        buffer: Buffer,
        clock: Clock
    ) -> Dict[str, Dict[str, int]]:
        """
        Decide which tiles/bitrates to request for the next segment.

        Must be implemented by subclasses.
        """
        pass

    @staticmethod
    def group_tiles_by_id(tiles: List[Tile]) -> Dict[str, Tile]:
        """Utility: group tiles by their unique_id."""
        return {tile.unique_id: tile for tile in tiles}

    @staticmethod
    def group_tiles(tiles):
        return ABR.group_tiles_by_id(tiles)

    @staticmethod
    def component_reps(tile: Tile, component: str):
        return tile.component_options[component]

    @staticmethod
    def lowest_rep_idx(tile: Tile, component: str) -> int:
        return 0

    @staticmethod
    def highest_rep_idx(tile: Tile, component: str) -> int:
        return len(tile.component_options[component]) - 1

    def _prediction_request_key(self, fov, tile_groups, frame_paths):
        predictor = getattr(fov, "predictor", None)
        if predictor is None or not frame_paths:
            return None
        target_frame = str(Path(frame_paths[-1]).resolve())
        tile_ids = tuple(sorted(str(tile_id) for tile_id in tile_groups.keys()))
        request = getattr(fov, "_active_prediction_request", None)
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
            predictor.__class__.__name__,
            target_frame,
            len(frame_paths),
            tile_ids,
            request_key,
        )

    def _remember_predicted_fov(self, request_key, predicted_fov):
        self._last_predicted_fov_key = request_key
        self._last_predicted_fov = dict(predicted_fov) if isinstance(predicted_fov, dict) else predicted_fov

    def peek_predicted_fov(self, fov, tile_groups, frame_paths):
        request_key = self._prediction_request_key(fov, tile_groups, frame_paths)
        if request_key is None:
            return None
        if request_key != getattr(self, "_last_predicted_fov_key", None):
            return None
        cached = getattr(self, "_last_predicted_fov", None)
        return dict(cached) if isinstance(cached, dict) else cached

    def predict_fov(self, fov, tile_groups, frame_paths):
        """Predict FoV fraction for one representative per tile."""
        cached = self.peek_predicted_fov(fov, tile_groups, frame_paths)
        if cached is not None:
            return cached

        tiles_for_prediction = list(tile_groups.values())
        predictor = getattr(fov, "predictor", None)
        if frame_paths and predictor:
            request_key = self._prediction_request_key(fov, tile_groups, frame_paths)
            if hasattr(fov, "predict_tiles"):
                predicted = fov.predict_tiles(tiles_for_prediction, frame_paths)
            else:
                predicted = predictor.predict(tiles_for_prediction, frame_paths, fov)
            if request_key is not None and predicted is not None:
                self._remember_predicted_fov(request_key, predicted)
            return predicted
        return {tile_id: 1.0 for tile_id in tile_groups}  # default all visible
