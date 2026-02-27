from typing import Dict
from core.media_segment import TileComponentSet

class Segment:
    """Represents a media segment composed of multiple tiles."""

    def __init__(
        self,
        segment_idx: int,
        duration_s: float,
        tiles: Dict[str, TileComponentSet],
    ) -> None:
        """
        Initialize a Segment.

        Args:
            segment_idx: Unique index of the segment.
            duration_s: Duration of the segment in seconds.
            size_bits: Total encoded size of the segment in bits.
            tiles: Dictionary of tiles keyed by unique tile ID.
        """
        self.segment_idx = segment_idx
        self.duration_s = duration_s
        self.tiles = tiles
        self.remaining_rendering_time_s = duration_s

    # --- Derived properties ---

    @property
    def unique_id(self) -> str:
        """Return a unique string ID based on the segment index."""
        return f"segment_{self.segment_idx}"

    @property
    def num_tiles(self) -> int:
        """Return the number of tiles in this segment."""
        return len(self.tiles)

    @property
    def size_bits(self) -> int:
        """Return the total size of all tiles in this segment (bits)."""
        return sum(tile.size_bits for tile in self.tiles.values())

    # --- Utilities ---

    def get_tile(self, tile_id: str):
        """Return a Tile by its unique ID, or None if not found."""
        return self.tiles.get(tile_id)
    
    def to_dict(self) -> Dict:
        """Convert the Segment to a dictionary representation."""
        return {
            "segment_idx": self.segment_idx,
            "duration_s": self.duration_s,
            "size_bits": self.size_bits,
            "num_tiles": self.num_tiles,
            "avg_tile_bitrate_kbps": (
                (self.size_bits / max(self.duration_s, 1e-9)) / 1e3 / max(self.num_tiles, 1)
            ),
            "tiles": {tile_id: tile.to_dict() for tile_id, tile in self.tiles.items()},
        }

    # --- Dunder methods ---

    def __repr__(self) -> str:
        return (
            f"Segment("
            f"segment_idx={self.segment_idx}, "
            f"duration_s={self.duration_s:.2f}, "
            f"size_bits={self.size_bits}, "
            f"num_tiles={self.num_tiles}, "
        )

    def __str__(self) -> str:
        tiles_str = "\n".join(f"  {tile}" for tile in self.tiles.values())
        return (
            f"Segment #{self.segment_idx} "
            f"({self.duration_s:.2f}s, {self.num_tiles} tiles, "
            f"size={self.size_bits} bits, tiles_total={self.size_bits} bits)\n"
            f"{tiles_str}"
        )
