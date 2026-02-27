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

    @staticmethod
    def predict_fov(fov, tile_groups, frame_paths):
        """Predict FoV fraction for one representative per tile."""
        tiles_for_prediction = list(tile_groups.values())
        predictor = getattr(fov, "predictor", None)
        if frame_paths and predictor:
            return predictor.predict(tiles_for_prediction, frame_paths, fov)
        return {tile_id: 1.0 for tile_id in tile_groups}  # default all visible
