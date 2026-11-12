from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class TileBounds:
    """Axis-aligned bounds for a tile."""

    min_corner: Tuple[float, float, float]
    max_corner: Tuple[float, float, float]

    @property
    def dimensions(self) -> Tuple[float, float, float]:
        return (
            self.max_corner[0] - self.min_corner[0],
            self.max_corner[1] - self.min_corner[1],
            self.max_corner[2] - self.min_corner[2],
        )

    @property
    def center(self) -> Tuple[float, float, float]:
        return (
            (self.min_corner[0] + self.max_corner[0]) / 2,
            (self.min_corner[1] + self.max_corner[1]) / 2,
            (self.min_corner[2] + self.max_corner[2]) / 2,
        )

    @property
    def volume(self) -> float:
        w, h, d = self.dimensions
        return w * h * d


@dataclass
class ComponentSegment:
    """Single component segment (geom/occp/attr/atlas) referenced by the MPD."""

    tile_id: str
    component: str
    segment_idx: int
    quality_id: str
    rep_idx: int
    file: Path
    duration_s: float
    size_bits: int
    bandwidth_bps: int | None = None
    metadata: Dict[str, object] | None = None

    def to_dict(self) -> dict:
        return {
            "tile_id": self.tile_id,
            "component": self.component,
            "segment_idx": self.segment_idx,
            "quality_id": self.quality_id,
            "rep_idx": self.rep_idx,
            "file": str(self.file),
            "duration_s": self.duration_s,
            "size_bits": self.size_bits,
            "bandwidth_bps": self.bandwidth_bps,
            "metadata": self.metadata,
        }


@dataclass
class TileComponentOptions:
    """Available component representations for a tile at a given segment."""

    tile_id: str
    segment_idx: int
    bounds: TileBounds
    component_options: Dict[str, List[ComponentSegment]]

    @property
    def components(self) -> List[str]:
        return list(self.component_options.keys())

    @property
    def unique_id(self) -> str:
        return str(self.tile_id)

    @property
    def min_corner(self) -> Tuple[float, float, float]:
        return self.bounds.min_corner

    @property
    def max_corner(self) -> Tuple[float, float, float]:
        return self.bounds.max_corner

    @property
    def center(self) -> Tuple[float, float, float]:
        return self.bounds.center

    @property
    def volume(self) -> float:
        return self.bounds.volume

    def lowest_rep_idx(self, component: str) -> int:
        return 0

    def highest_rep_idx(self, component: str) -> int:
        return len(self.component_options[component]) - 1


@dataclass
class TileComponentSet:
    """
    A concrete selection of component representations for a tile segment.
    """

    tile_id: str
    segment_idx: int
    bounds: TileBounds
    components: Dict[str, ComponentSegment]

    @property
    def size_bits(self) -> int:
        return sum(comp.size_bits for comp in self.components.values())

    @property
    def duration_s(self) -> float:
        # Components should be aligned; use the longest to be safe.
        return max(comp.duration_s for comp in self.components.values())

    @property
    def min_corner(self) -> Tuple[float, float, float]:
        return self.bounds.min_corner

    @property
    def max_corner(self) -> Tuple[float, float, float]:
        return self.bounds.max_corner

    @property
    def center(self) -> Tuple[float, float, float]:
        return self.bounds.center

    @property
    def volume(self) -> float:
        return self.bounds.volume

    @property
    def unique_id(self) -> str:
        # Keep grouping simple across segments.
        return str(self.tile_id)

    def to_dict(self) -> dict:
        return {
            "tile_id": self.tile_id,
            "segment_idx": self.segment_idx,
            "bounds": {
                "min_corner": self.min_corner,
                "max_corner": self.max_corner,
                "center": self.center,
                "volume": self.volume,
            },
            "size_bits": self.size_bits,
            "duration_s": self.duration_s,
            "components": {
                name: comp.to_dict() for name, comp in self.components.items()
            },
        }

    def __str__(self) -> str:
        return (
            f"TileComponentSet(tile={self.tile_id}, seg={self.segment_idx}, size_bits={self.size_bits})"
        )
