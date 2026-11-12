from __future__ import annotations

import re
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


_TILE_COLUMN_RE = re.compile(r"^tile_(\d+)$")
_FRAME_NUMBER_RE = re.compile(r"(\d+)(?=\.[^.]+$)")
_FRAME_COLUMNS = ("frame", "framenumber", "frame_number", "frameid", "frame_id")
DEFAULT_GROUND_TRUTH_MODE = "visible_fraction"
DEFAULT_VISIBLE_THRESHOLD = 0.01
DEFAULT_PEAK_WEIGHT = 0.3


def clamp01(value) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def frame_number_from_path(path: str | Path) -> int | None:
    match = _FRAME_NUMBER_RE.search(Path(path).name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _tile_sort_key(tile_id: str) -> tuple[int, str]:
    try:
        return (int(tile_id), tile_id)
    except ValueError:
        return (10**9, tile_id)


def _frame_tile_visibility(entry: Mapping[str, Any]) -> Dict[str, float]:
    raw = entry.get("tile_visibility") or {}
    if not isinstance(raw, Mapping):
        return {}
    return {str(tile_id): clamp01(value) for tile_id, value in raw.items()}


def reduce_tile_visibility_frames(
    frames: Sequence[Mapping[str, Any]],
    *,
    tile_ids: Iterable[str] | None = None,
    mode: str = DEFAULT_GROUND_TRUTH_MODE,
    visible_threshold: float = DEFAULT_VISIBLE_THRESHOLD,
    peak_weight: float = DEFAULT_PEAK_WEIGHT,
) -> Dict[str, float]:
    """Reduce per-frame tile visibility into a segment-level target."""
    normalized_frames = [_frame_tile_visibility(frame) for frame in frames or []]
    if tile_ids is None:
        ids = sorted({tile_id for frame in normalized_frames for tile_id in frame.keys()}, key=_tile_sort_key)
    else:
        ids = sorted({str(tile_id) for tile_id in tile_ids}, key=_tile_sort_key)
    if not ids:
        return {}
    if not normalized_frames:
        return {tile_id: 0.0 for tile_id in ids}

    resolved_mode = str(mode or DEFAULT_GROUND_TRUTH_MODE).strip().lower()
    if resolved_mode in {"mean", "segment_mean", "mean_visibility", "duration_weighted_mean"}:
        return {
            tile_id: clamp01(sum(frame.get(tile_id, 0.0) for frame in normalized_frames) / len(normalized_frames))
            for tile_id in ids
        }
    if resolved_mode in {"max", "segment_max", "max_visibility"}:
        return {
            tile_id: clamp01(max(frame.get(tile_id, 0.0) for frame in normalized_frames))
            for tile_id in ids
        }
    if resolved_mode in {"visible_fraction", "segment_visible_fraction"}:
        threshold = clamp01(visible_threshold)
        return {
            tile_id: clamp01(
                sum(1.0 for frame in normalized_frames if frame.get(tile_id, 0.0) >= threshold) / len(normalized_frames)
            )
            for tile_id in ids
        }
    if resolved_mode in {"peak_weighted", "blend", "mean_max_blend"}:
        weight = clamp01(peak_weight)
        return {
            tile_id: clamp01(
                ((1.0 - weight) * (sum(frame.get(tile_id, 0.0) for frame in normalized_frames) / len(normalized_frames)))
                + (weight * max(frame.get(tile_id, 0.0) for frame in normalized_frames))
            )
            for tile_id in ids
        }

    return reduce_tile_visibility_frames(
        frames,
        tile_ids=ids,
        mode=DEFAULT_GROUND_TRUTH_MODE,
        visible_threshold=visible_threshold,
        peak_weight=peak_weight,
    )


def has_tile_fov_columns(csv_path: str | Path) -> bool:
    path = Path(csv_path)
    if not path.exists():
        return False
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            fieldnames = next(reader)
        except StopIteration:
            return False
    return any(_TILE_COLUMN_RE.match(str(column)) for column in fieldnames)


class TileFovTrace:
    """MazeLab per-frame tile FoV metadata used as ground truth."""

    def __init__(self, csv_path: str | Path, *, require_tile_columns: bool = True):
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Tile FoV CSV not found: {self.csv_path}")

        with self.csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            rows = list(reader)

        frame_col = self._resolve_frame_column(fieldnames)
        if frame_col is None:
            raise ValueError(f"Tile FoV CSV must contain a 'frame' column: {self.csv_path}")

        tile_columns: Dict[str, str] = {}
        for column in fieldnames:
            match = _TILE_COLUMN_RE.match(str(column))
            if match:
                tile_columns[str(match.group(1))] = column
        if not tile_columns and require_tile_columns:
            raise ValueError(f"Tile FoV CSV must contain at least one tile_N column: {self.csv_path}")
        if not tile_columns:
            self.values_by_frame = {}
            self.tile_ids = ()
            return

        values_by_frame: Dict[int, Dict[str, float]] = {}
        for row in rows:
            try:
                frame_number = int(float(row.get(frame_col, "")))
            except (TypeError, ValueError):
                continue
            values_by_frame[frame_number] = {
                tile_id: clamp01(row.get(column))
                for tile_id, column in tile_columns.items()
            }

        self.values_by_frame = dict(sorted(values_by_frame.items()))
        self.tile_ids = tuple(sorted(tile_columns.keys(), key=lambda value: int(value)))

    @staticmethod
    def _resolve_frame_column(columns: Iterable[str]) -> str | None:
        for column in columns:
            if str(column).strip().lower() in _FRAME_COLUMNS:
                return column
        return None

    def visibility_for_frame(self, frame_number: int, tile_ids: Iterable[str] | None = None) -> Dict[str, float]:
        ids = tuple(str(tile_id) for tile_id in (tile_ids or self.tile_ids))
        row = self.values_by_frame.get(int(frame_number))
        if row is None:
            return {tile_id: 0.0 for tile_id in ids}
        return {tile_id: clamp01(row.get(tile_id, 0.0)) for tile_id in ids}

    def visibility_for_frames(self, frame_numbers: Sequence[int], tile_ids: Iterable[str] | None = None) -> Dict[str, float]:
        ids = tuple(str(tile_id) for tile_id in (tile_ids or self.tile_ids))
        frames = [int(frame_number) for frame_number in frame_numbers if int(frame_number) in self.values_by_frame]
        if not frames:
            return {tile_id: 0.0 for tile_id in ids}
        return {
            tile_id: clamp01(
                sum(self.values_by_frame[frame].get(tile_id, 0.0) for frame in frames) / len(frames)
            )
            for tile_id in ids
        }

    def visibility_for_frame_paths(self, frame_paths: Sequence[str | Path], tile_ids: Iterable[str] | None = None) -> Dict[str, float]:
        frame_numbers = [
            frame_number
            for frame_number in (frame_number_from_path(path) for path in frame_paths)
            if frame_number is not None
        ]
        return self.visibility_for_frames(frame_numbers, tile_ids=tile_ids)

    def visibility_frames_for_frame_paths(
        self,
        frame_paths: Sequence[str | Path],
        tile_ids: Iterable[str] | None = None,
    ) -> list[Dict[str, Any]]:
        frame_numbers = [
            frame_number
            for frame_number in (frame_number_from_path(path) for path in frame_paths)
            if frame_number is not None
        ]
        return [
            {
                "frame": int(frame_number),
                "tile_visibility": self.visibility_for_frame(frame_number, tile_ids=tile_ids),
            }
            for frame_number in frame_numbers
        ]
