from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from core.media_segment import (
    ComponentSegment,
    TileBounds,
    TileComponentOptions,
    TileComponentSet,
)
from core.mpd_parser import V3CMpd

logger = logging.getLogger(__name__)


class Video:
    """Lazy-access wrapper for V3C content described by an MPD."""

    def __init__(
        self,
        manifest_file: Path,
        original_content_dir: Path,
        media_root: Optional[Path] = None,
    ) -> None:
        if not manifest_file.is_file():
            raise FileNotFoundError(f"Manifest file not found: {manifest_file}")
        if original_content_dir and not original_content_dir.is_dir():
            raise FileNotFoundError(f"Content directory not found: {original_content_dir}")

        self.manifest_file = manifest_file
        self.original_content_dir = original_content_dir
        self.start_frame_idx: int
        self.total_frames: int

        self.tile_ids: List[str] = []
        self._component_quality_by_rep_idx: Dict[str, Dict[str, List[str]]] = {}
        self._component_rep_idx_by_quality: Dict[str, Dict[str, Dict[str, int]]] = {}
        self._segment_cache: Dict[tuple, TileComponentSet] = {}

        suffix = manifest_file.suffix.lower()
        if suffix != ".mpd":
            raise ValueError(f"Unsupported manifest format: {manifest_file.suffix}")
        self._init_from_mpd(manifest_file, media_root)
        self._init_frames_from_dir(original_content_dir)

    # --------- Internal helpers ---------

    def _init_from_mpd(self, manifest_file: Path, media_root: Optional[Path]) -> None:
        self.mpd = V3CMpd(manifest_file, media_root=media_root)
        self.total_playback_time = self.mpd.presentation_duration_s
        self.segment_duration_s = self.mpd.segment_duration_s
        self.segment_count = self.mpd.segment_count

        self.tile_ids = self.mpd.iter_tile_ids()
        self._init_component_orders()

    def _init_component_orders(self) -> None:
        for tile_id in self.tile_ids:
            self._component_quality_by_rep_idx[tile_id] = {}
            self._component_rep_idx_by_quality[tile_id] = {}
            components = self.mpd.tile_components.get(tile_id, {})
            for comp in components.keys():
                ordered = self.mpd.component_quality_order(tile_id, comp)
                if not ordered:
                    continue
                qualities = [qid for qid, _ in sorted(ordered, key=lambda x: x[1])]
                self._component_quality_by_rep_idx[tile_id][comp] = qualities
                self._component_rep_idx_by_quality[tile_id][comp] = {
                    qid: idx for idx, qid in enumerate(qualities)
                }

    def _init_frames_from_dir(self, frames_dir: Path) -> None:
        ply_files = sorted(frames_dir.glob("*.ply"))
        if not ply_files:
            raise FileNotFoundError(f"No .ply files found in {frames_dir}")

        frames: List[tuple[int, Path]] = []
        for f in ply_files:
            digits = "".join(ch for ch in f.stem if ch.isdigit())
            if not digits:
                continue
            frames.append((int(digits), f))

        if not frames:
            raise ValueError(f"Could not extract frame numbers from files in {frames_dir}")

        frames.sort(key=lambda x: x[0])
        self.start_frame_idx = frames[0][0]
        self.total_frames = len(frames)
        self._frame_numbers = [fn for fn, _ in frames]
        self._frame_paths = [p for _, p in frames]

    def _quality_id_from_rep_idx(self, tile_id: str, component: str, rep_idx: int) -> str:
        try:
            return self._component_quality_by_rep_idx[tile_id][component][rep_idx]
        except (KeyError, IndexError) as exc:
            raise KeyError(f"Unknown rep_idx {rep_idx} for tile {tile_id} component {component}") from exc

    def _estimate_size_bits(
        self, file_path: Path, bandwidth_bps: Optional[int], duration_s: float
    ) -> int:
        if file_path.is_file():
            return file_path.stat().st_size * 8
        if bandwidth_bps is not None:
            return int(bandwidth_bps * duration_s)
        logger.warning("Unable to estimate size for %s; defaulting to 0 bits", file_path)
        return 0

    def _build_component_segment(
        self,
        tile_id: str,
        component: str,
        segment_idx: int,
        rep_idx: int,
        quality_id: str,
        rep_info,
    ) -> ComponentSegment:
        segment_number = rep_info.template.start_number + segment_idx
        media_rel = rep_info.template.media.replace("$Number$", str(segment_number))
        file_path = (self.mpd.base_url / media_rel).resolve()
        duration_s = rep_info.template.duration / rep_info.template.timescale
        size_bits = self._estimate_size_bits(file_path, rep_info.bandwidth, duration_s)

        return ComponentSegment(
            tile_id=tile_id,
            component=component,
            segment_idx=segment_idx,
            quality_id=quality_id,
            rep_idx=rep_idx,
            file=file_path,
            duration_s=duration_s,
            size_bits=size_bits,
            bandwidth_bps=rep_info.bandwidth,
            metadata={"representation_id": rep_info.id},
        )

    def _get_bounds(self, tile_id: str, segment_idx: int) -> TileBounds:
        bounds = self.mpd.get_bounds(tile_id, segment_idx)
        if bounds is None:
            raise ValueError(f"No bounds found for tile {tile_id} segment {segment_idx}")
        return bounds

    def _build_tile_options(self, tile_id: str, segment_idx: int) -> TileComponentOptions:
        bounds = self._get_bounds(tile_id, segment_idx)
        component_options: Dict[str, list[ComponentSegment]] = {}
        components = self.mpd.tile_components.get(tile_id, {})
        for comp_name, reps in components.items():
            rep_list = []
            for rep_idx in range(len(self._component_quality_by_rep_idx[tile_id].get(comp_name, []))):
                quality_id = self._quality_id_from_rep_idx(tile_id, comp_name, rep_idx)
                rep_info = reps[quality_id]
                rep_list.append(
                    self._build_component_segment(
                        tile_id=tile_id,
                        component=comp_name,
                        segment_idx=segment_idx,
                        rep_idx=rep_idx,
                        quality_id=quality_id,
                        rep_info=rep_info,
                    )
                )
            if rep_list:
                component_options[comp_name] = rep_list
        return TileComponentOptions(
            tile_id=tile_id,
            segment_idx=segment_idx,
            bounds=bounds,
            component_options=component_options,
        )

    def build_tile_selection(
        self, tile_id: str, segment_idx: int, component_rep_map: Dict[str, int]
    ) -> TileComponentSet:
        cache_key = (tile_id, segment_idx, tuple(sorted(component_rep_map.items())))
        if cache_key in self._segment_cache:
            return self._segment_cache[cache_key]

        bounds = self._get_bounds(tile_id, segment_idx)
        components: Dict[str, ComponentSegment] = {}
        components_info = self.mpd.tile_components.get(tile_id, {})
        for comp_name, reps in components_info.items():
            rep_idx = component_rep_map.get(comp_name, 0)
            quality_id = self._quality_id_from_rep_idx(tile_id, comp_name, rep_idx)
            rep_info = reps[quality_id]
            components[comp_name] = self._build_component_segment(
                tile_id=tile_id,
                component=comp_name,
                segment_idx=segment_idx,
                rep_idx=rep_idx,
                quality_id=quality_id,
                rep_info=rep_info,
            )

        variant = TileComponentSet(
            tile_id=tile_id,
            segment_idx=segment_idx,
            bounds=bounds,
            components=components,
        )
        self._segment_cache[cache_key] = variant
        return variant

    # --------- Public API ---------

    def load_segment(self, tile_id: str, segment_idx: int, component_rep_map: Dict[str, int]) -> TileComponentSet:
        return self.build_tile_selection(tile_id, segment_idx, component_rep_map)

    def list_segment_tiles(self, segment_idx: int) -> List[TileComponentOptions]:
        return [self._build_tile_options(tile_id, segment_idx) for tile_id in self.tile_ids]

    def get_original_frames(self, segment_idx: int) -> Optional[List[Path]]:
        if segment_idx < 0:
            return None

        frames_per_segment = self.total_frames // self.segment_count
        remainder = self.total_frames % self.segment_count

        start_offset = segment_idx * frames_per_segment
        end_offset = start_offset + frames_per_segment
        if segment_idx == self.segment_count - 1:
            end_offset += remainder

        return self._frame_paths[start_offset:end_offset]

    # --------- Introspection ---------

    def num_tiles(self) -> int:
        return len(self.tile_ids)

    def num_segments(self) -> int:
        return self.segment_count

    def num_representations(self, tile_id: str) -> int:
        components = self._component_quality_by_rep_idx.get(tile_id, {})
        return max((len(v) for v in components.values()), default=0)

    def list_components(self, tile_id: str) -> List[str]:
        return list(self._component_quality_by_rep_idx.get(tile_id, {}).keys())
