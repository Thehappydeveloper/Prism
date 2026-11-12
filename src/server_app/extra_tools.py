from __future__ import annotations

import csv
import math
import statistics
import struct
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import path_for_client, safe_path

MAX_WARNINGS = 200
NUMERIC_SAMPLE_LIMIT_PER_FILE = 500
SUPPORTED_FILTER_METHODS = {"iqr", "percentile"}
SUPPORTED_HISTOGRAM_SOURCES = {"raw", "filtered"}
NEAR_ZERO_EXTENT = 1e-9
SUPPORTED_DATASET_EXPORTS = {"scaled", "normalized", "voxelized", "both"}
PLY_SCALAR_TYPES: dict[str, tuple[str, int]] = {
    "char": ("b", 1),
    "int8": ("b", 1),
    "uchar": ("B", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "int16": ("h", 2),
    "ushort": ("H", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "int32": ("i", 4),
    "uint": ("I", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}

PLY_INT_TYPES = {
    "char",
    "int8",
    "uchar",
    "uint8",
    "short",
    "int16",
    "ushort",
    "uint16",
    "int",
    "int32",
    "uint",
    "uint32",
}
POINT_SEQUENCE_AGGREGATIONS = {"minimum", "maximum", "mean", "median", "standard_deviation"}
POINT_SEQUENCE_SCORE_RADII = (1, 2, 3, 4, 5)
MAX_RGB_DISTANCE = math.sqrt((255.0**2) * 3.0)


def _warning(warnings: list[dict[str, Any]], kind: str, message: str, path: Path | None = None) -> None:
    if len(warnings) >= MAX_WARNINGS:
        return
    entry: dict[str, Any] = {"kind": kind, "message": message}
    if path is not None:
        entry["path"] = path_for_client(path)
    warnings.append(entry)


def _csv_files(root: Path, warnings: list[dict[str, Any]]) -> list[Path]:
    if not root.exists():
        raise ValueError("Root folder does not exist.")
    if not root.is_dir():
        raise ValueError("Root path must be a folder.")

    files: list[Path] = []
    seen: set[tuple[int, int] | Path] = set()
    for path in root.rglob("*"):
        try:
            if not path.is_file() or path.suffix.lower() != ".csv":
                continue
            resolved = path.resolve()
            stat = resolved.stat()
            identity: tuple[int, int] | Path = (stat.st_dev, stat.st_ino)
        except OSError as exc:
            _warning(warnings, "skipped_file", f"Could not inspect file: {exc}", path)
            continue
        if identity in seen or resolved in seen:
            continue
        seen.add(identity)
        seen.add(resolved)
        files.append(resolved)
    files.sort(key=lambda item: str(item).lower())
    return files


def discover_speed_accel_columns(root_folder: str) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    root = safe_path(root_folder)
    files = _csv_files(root, warnings)
    columns: list[str] = []
    seen_columns: set[str] = set()
    numeric_columns: list[str] = []
    numeric_seen: set[str] = set()
    files_loaded = 0
    skipped_files = 0

    for path in files:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                header = reader.fieldnames or []
                if not header:
                    skipped_files += 1
                    _warning(warnings, "skipped_file", "CSV file is empty or missing a header row.", path)
                    continue
                files_loaded += 1
                sampled = 0
                for row in reader:
                    sampled += 1
                    for column in header:
                        clean_column = str(column or "").strip()
                        if clean_column and clean_column not in numeric_seen and _parse_numeric(row.get(column)) is not None:
                            numeric_seen.add(clean_column)
                            numeric_columns.append(clean_column)
                    if sampled >= NUMERIC_SAMPLE_LIMIT_PER_FILE:
                        break
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            skipped_files += 1
            _warning(warnings, "skipped_file", f"Could not read CSV header: {exc}", path)
            continue
        for raw_column in header:
            column = str(raw_column or "").strip()
            if column and column not in seen_columns:
                seen_columns.add(column)
                columns.append(column)

    return {
        "root_folder": path_for_client(root),
        "csv_files": [path_for_client(path) for path in files],
        "file_count": len(files),
        "files_loaded": files_loaded,
        "files_skipped": skipped_files,
        "columns": columns,
        "numeric_columns": numeric_columns,
        "warnings": warnings,
    }


def _parse_ply_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        first = handle.readline().strip()
        if first != b"ply":
            raise ValueError("Reference point cloud is not a valid PLY file.")
        fmt = ""
        vertex_count: int | None = None
        vertex_properties: list[tuple[str, str]] = []
        current_element = ""
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError("PLY header is missing end_header.")
            line = raw.decode("ascii", errors="replace").strip()
            if not line or line.startswith("comment"):
                continue
            parts = line.split()
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and current_element == "vertex":
                if len(parts) >= 5 and parts[1] == "list":
                    raise ValueError("PLY vertex list properties are not supported for scaling inspection.")
                if len(parts) >= 3:
                    vertex_properties.append((parts[2], parts[1]))
            elif parts[0] == "end_header":
                break
        if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
            raise ValueError(f"Unsupported PLY format: {fmt or 'unknown'}.")
        if vertex_count is None:
            raise ValueError("PLY file does not declare a vertex element.")
        if vertex_count <= 0:
            raise ValueError("PLY file has no vertices.")
        names = [name for name, _type_name in vertex_properties]
        for required in ("x", "y", "z"):
            if required not in names:
                raise ValueError("PLY vertex data must include x, y, and z properties.")
        return {
            "format": fmt,
            "vertex_count": vertex_count,
            "vertex_properties": vertex_properties,
            "data_offset": handle.tell(),
        }


def _ply_frame_number(path: Path) -> int | None:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return int(digits) if digits else None


def list_indexed_ply_frames(ply_folder: str | Path) -> list[tuple[int, Path]]:
    directory = safe_path(ply_folder)
    if not directory.is_dir():
        raise ValueError("PLY sequence path must be a folder.")
    frames: list[tuple[int, Path]] = []
    for path in sorted(directory.glob("*.ply")):
        frame_number = _ply_frame_number(path)
        if frame_number is None:
            continue
        frames.append((frame_number, path.resolve()))
    if not frames:
        raise ValueError("Could not extract frame numbers from PLY files in the sequence folder.")
    frames.sort(key=lambda item: item[0])
    return frames


def _point_sequence_files(root: Path, recursive: bool, warnings: list[dict[str, Any]]) -> list[tuple[int, Path]]:
    if not root.exists():
        raise ValueError("PLY sequence folder does not exist.")
    if not root.is_dir():
        raise ValueError("PLY sequence path must be a folder.")
    iterator = root.rglob("*.ply") if recursive else root.glob("*.ply")
    frames: list[tuple[int, Path]] = []
    seen: set[tuple[int, int] | Path] = set()
    for path in iterator:
        try:
            if not path.is_file():
                continue
            resolved = path.resolve()
            stat = resolved.stat()
            identity: tuple[int, int] | Path = (stat.st_dev, stat.st_ino)
        except OSError as exc:
            _warning(warnings, "skipped_file", f"Could not inspect PLY file: {exc}", path)
            continue
        if identity in seen or resolved in seen:
            continue
        seen.add(identity)
        seen.add(resolved)
        frame_index = _ply_frame_number(resolved)
        if frame_index is None:
            _warning(warnings, "skipped_file", "Could not extract a frame index from the PLY filename.", resolved)
            continue
        frames.append((frame_index, resolved))
    frames.sort(key=lambda item: (item[0], str(item[1]).lower()))
    return frames


def discover_point_cloud_sequence(root_folder: str, recursive: bool = True) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    root = safe_path(root_folder)
    frames = _point_sequence_files(root, recursive, warnings)
    frame_indices = [frame_index for frame_index, _path in frames]
    return {
        "root_folder": path_for_client(root),
        "recursive": bool(recursive),
        "ply_files": [path_for_client(path) for _frame_index, path in frames],
        "file_count": len(frames),
        "first_frame_index": min(frame_indices) if frame_indices else None,
        "last_frame_index": max(frame_indices) if frame_indices else None,
        "warnings": warnings,
    }


def _read_ply_points_for_sequence(
    path: Path,
    max_points_for_qp: int,
) -> dict[str, Any]:
    header = _parse_ply_header(path)
    props: list[tuple[str, str]] = header["vertex_properties"]
    names = [name for name, _type_name in props]
    x_idx = names.index("x")
    y_idx = names.index("y")
    z_idx = names.index("z")
    color_aliases = (("red", "r"), ("green", "g"), ("blue", "b"))
    color_indices: list[int] = []
    for aliases in color_aliases:
        index = next((names.index(alias) for alias in aliases if alias in names), None)
        if index is None:
            color_indices = []
            break
        color_indices.append(index)

    declared_count = int(header["vertex_count"])
    sample_limit = max(0, int(max_points_for_qp or 0))
    stride = max(1, math.ceil(declared_count / sample_limit)) if sample_limit and declared_count > sample_limit else 1
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    valid_points = 0
    sampled_points: list[tuple[float, float, float, tuple[float, float, float] | None]] = []

    def consume_row(row: list[Any], ordinal: int) -> None:
        nonlocal valid_points
        coords = [_parse_numeric(row[x_idx]), _parse_numeric(row[y_idx]), _parse_numeric(row[z_idx])]
        if any(value is None for value in coords):
            return
        x_value, y_value, z_value = (float(coords[0]), float(coords[1]), float(coords[2]))  # type: ignore[arg-type]
        mins[0] = min(mins[0], x_value)
        maxs[0] = max(maxs[0], x_value)
        mins[1] = min(mins[1], y_value)
        maxs[1] = max(maxs[1], y_value)
        mins[2] = min(mins[2], z_value)
        maxs[2] = max(maxs[2], z_value)
        valid_points += 1
        if ordinal % stride != 0:
            return
        color: tuple[float, float, float] | None = None
        if color_indices:
            raw_color = [_parse_numeric(row[index]) for index in color_indices]
            if not any(value is None for value in raw_color):
                color = tuple(max(0.0, min(255.0, float(value))) for value in raw_color)  # type: ignore[arg-type]
        sampled_points.append((x_value, y_value, z_value, color))

    if header["format"] == "ascii":
        with path.open("rb") as handle:
            handle.seek(header["data_offset"])
            for ordinal in range(declared_count):
                raw = handle.readline()
                if not raw:
                    break
                parts = raw.decode("utf-8", errors="replace").strip().split()
                if len(parts) < len(props):
                    continue
                consume_row(parts[: len(props)], ordinal)
    else:
        endian = "<" if header["format"] == "binary_little_endian" else ">"
        try:
            format_chars = [PLY_SCALAR_TYPES[type_name][0] for _name, type_name in props]
        except KeyError as exc:
            raise ValueError(f"Unsupported PLY vertex property type: {exc.args[0]}") from exc
        row_struct = struct.Struct(endian + "".join(format_chars))
        with path.open("rb") as handle:
            handle.seek(header["data_offset"])
            for ordinal in range(declared_count):
                raw = handle.read(row_struct.size)
                if len(raw) < row_struct.size:
                    break
                consume_row(list(row_struct.unpack(raw)), ordinal)

    if valid_points <= 0:
        raise ValueError("PLY file has no valid x/y/z vertex points.")
    bounds = {
        "min_x": mins[0],
        "max_x": maxs[0],
        "min_y": mins[1],
        "max_y": maxs[1],
        "min_z": mins[2],
        "max_z": maxs[2],
    }
    ranges = {
        "x": maxs[0] - mins[0],
        "y": maxs[1] - mins[1],
        "z": maxs[2] - mins[2],
    }
    return {
        "declared_vertex_count": declared_count,
        "point_count": valid_points,
        "sampled_point_count": len(sampled_points),
        "bounds": bounds,
        "ranges": ranges,
        "points": sampled_points,
        "has_color": bool(color_indices),
        "sampled": stride > 1,
        "sample_stride": stride,
    }


def _global_bounds_from_frames(frames: list[dict[str, Any]]) -> dict[str, float]:
    bounds = _empty_bounds()
    for frame in frames:
        bounds = _merge_bounds(bounds, frame["bounds"])
    finalized = _finalize_bounds(bounds)
    if finalized is None:
        raise ValueError("Could not compute sequence bounds from valid PLY frames.")
    return finalized


def _precompute_radius_shells() -> tuple[dict[int, list[tuple[int, int, int, float]]], dict[int, float]]:
    shells: dict[int, list[tuple[int, int, int, float]]] = {radius: [] for radius in POINT_SEQUENCE_SCORE_RADII}
    max_possible: dict[int, float] = {radius: 0.0 for radius in POINT_SEQUENCE_SCORE_RADII}
    max_radius = max(POINT_SEQUENCE_SCORE_RADII)
    for dx in range(-max_radius, max_radius + 1):
        for dy in range(-max_radius, max_radius + 1):
            for dz in range(-max_radius, max_radius + 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                distance = math.sqrt((dx * dx) + (dy * dy) + (dz * dz))
                if distance <= 0 or distance > max_radius:
                    continue
                radius = int(math.ceil(distance))
                if radius not in shells:
                    continue
                weight = 1.0 / (1.0 + distance)
                shells[radius].append((dx, dy, dz, weight))
                for score_radius in POINT_SEQUENCE_SCORE_RADII:
                    if radius <= score_radius:
                        max_possible[score_radius] += weight
    return shells, max_possible


def _voxelize_sequence_points(
    points: list[tuple[float, float, float, tuple[float, float, float] | None]],
    sequence_bounds: dict[str, float],
    voxel_bits: int,
    use_native_voxels: bool,
) -> tuple[set[tuple[int, int, int]], dict[tuple[int, int, int], tuple[float, float, float]]]:
    voxel_max = (2 ** int(voxel_bits)) - 1
    spans = {
        "x": max(sequence_bounds["max_x"] - sequence_bounds["min_x"], NEAR_ZERO_EXTENT),
        "y": max(sequence_bounds["max_y"] - sequence_bounds["min_y"], NEAR_ZERO_EXTENT),
        "z": max(sequence_bounds["max_z"] - sequence_bounds["min_z"], NEAR_ZERO_EXTENT),
    }
    color_sums: dict[tuple[int, int, int], list[float]] = {}
    color_counts: dict[tuple[int, int, int], int] = {}
    occupied: set[tuple[int, int, int]] = set()
    for x_value, y_value, z_value, color in points:
        if use_native_voxels:
            voxel = (int(round(x_value)), int(round(y_value)), int(round(z_value)))
        else:
            voxel = (
                int(max(0, min(voxel_max, round(((x_value - sequence_bounds["min_x"]) / spans["x"]) * voxel_max)))),
                int(max(0, min(voxel_max, round(((y_value - sequence_bounds["min_y"]) / spans["y"]) * voxel_max)))),
                int(max(0, min(voxel_max, round(((z_value - sequence_bounds["min_z"]) / spans["z"]) * voxel_max)))),
            )
        occupied.add(voxel)
        if color is not None:
            if voxel not in color_sums:
                color_sums[voxel] = [0.0, 0.0, 0.0]
                color_counts[voxel] = 0
            color_sums[voxel][0] += color[0]
            color_sums[voxel][1] += color[1]
            color_sums[voxel][2] += color[2]
            color_counts[voxel] += 1
    colors = {
        voxel: (sums[0] / color_counts[voxel], sums[1] / color_counts[voxel], sums[2] / color_counts[voxel])
        for voxel, sums in color_sums.items()
        if color_counts.get(voxel, 0) > 0
    }
    return occupied, colors


def _color_similarity(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    distance = math.sqrt(((first[0] - second[0]) ** 2) + ((first[1] - second[1]) ** 2) + ((first[2] - second[2]) ** 2))
    return max(0.0, min(1.0, 1.0 - (distance / MAX_RGB_DISTANCE)))


def _color_entropy_inverse(colors: dict[tuple[int, int, int], tuple[float, float, float]]) -> float | None:
    if not colors:
        return None
    counts: dict[int, int] = defaultdict(int)
    for red, green, blue in colors.values():
        r_bin = int(max(0, min(255, round(red)))) >> 3
        g_bin = int(max(0, min(255, round(green)))) >> 3
        b_bin = int(max(0, min(255, round(blue)))) >> 3
        counts[(r_bin << 10) | (g_bin << 5) | b_bin] += 1
    total = sum(counts.values())
    if total <= 0:
        return None
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return max(0.0, min(1.0, 1.0 - (entropy / 15.0)))


def _score_voxel_frame(
    occupied: set[tuple[int, int, int]],
    colors: dict[tuple[int, int, int], tuple[float, float, float]],
    previous_occupied: set[tuple[int, int, int]] | None,
    previous_colors: dict[tuple[int, int, int], tuple[float, float, float]] | None,
    shells: dict[int, list[tuple[int, int, int, float]]],
    max_possible: dict[int, float],
) -> dict[str, Any]:
    if not occupied:
        return {
            "geometry_redundancy": {f"g{radius}": 0.0 for radius in POINT_SEQUENCE_SCORE_RADII},
            "attribute_smoothness": {f"a{radius}": None for radius in POINT_SEQUENCE_SCORE_RADII},
            "geometry_stability": 1.0,
            "attribute_stability": None,
            "inverse_color_entropy": None,
            "geometry_qp_safety": 0.2,
            "attribute_qp_safety": None,
            "shared_color_voxels": 0,
        }
    geometry_totals = {radius: 0.0 for radius in POINT_SEQUENCE_SCORE_RADII}
    attribute_totals = {radius: 0.0 for radius in POINT_SEQUENCE_SCORE_RADII}
    attribute_counts = {radius: 0 for radius in POINT_SEQUENCE_SCORE_RADII}
    ordered = sorted(occupied)
    for voxel in ordered:
        progressive_support = 0.0
        progressive_color_sum = 0.0
        progressive_color_weight = 0.0
        source_color = colors.get(voxel)
        for radius in POINT_SEQUENCE_SCORE_RADII:
            for dx, dy, dz, weight in shells[radius]:
                neighbor = (voxel[0] + dx, voxel[1] + dy, voxel[2] + dz)
                if neighbor not in occupied:
                    continue
                progressive_support += weight
                if source_color is not None and neighbor in colors:
                    progressive_color_sum += weight * _color_similarity(source_color, colors[neighbor])
                    progressive_color_weight += weight
            geometry_totals[radius] += progressive_support / max_possible[radius] if max_possible[radius] > 0 else 0.0
            if source_color is not None:
                attribute_totals[radius] += progressive_color_sum / progressive_color_weight if progressive_color_weight > 0 else 0.0
                attribute_counts[radius] += 1

    geometry_scores = {
        f"g{radius}": max(0.0, min(1.0, geometry_totals[radius] / len(occupied)))
        for radius in POINT_SEQUENCE_SCORE_RADII
    }
    attribute_scores = {
        f"a{radius}": (
            max(0.0, min(1.0, attribute_totals[radius] / attribute_counts[radius]))
            if attribute_counts[radius] > 0
            else None
        )
        for radius in POINT_SEQUENCE_SCORE_RADII
    }
    if previous_occupied is None:
        geometry_stability = 1.0
    else:
        union_count = len(occupied | previous_occupied)
        geometry_stability = 1.0 - (len(occupied ^ previous_occupied) / union_count) if union_count else 1.0
    attribute_stability = None
    shared_color_voxels = 0
    if previous_colors:
        shared = set(colors).intersection(previous_colors)
        shared_color_voxels = len(shared)
        if shared:
            average_difference = statistics.fmean(
                1.0 - _color_similarity(colors[voxel], previous_colors[voxel])
                for voxel in shared
            )
            confidence = min(1.0, shared_color_voxels / max(1, min(len(colors), len(previous_colors))))
            attribute_stability = max(0.0, min(1.0, (1.0 - average_difference) * confidence))
    inverse_entropy = _color_entropy_inverse(colors)
    geometry_redundancy = (
        (0.40 * geometry_scores["g1"])
        + (0.25 * geometry_scores["g2"])
        + (0.20 * geometry_scores["g3"])
        + (0.10 * geometry_scores["g4"])
        + (0.05 * geometry_scores["g5"])
    )
    geometry_qp_safety = max(0.0, min(1.0, (0.80 * geometry_redundancy) + (0.20 * geometry_stability)))
    attribute_qp_safety = None
    if all(attribute_scores[f"a{radius}"] is not None for radius in POINT_SEQUENCE_SCORE_RADII) and inverse_entropy is not None:
        attribute_smoothness = (
            (0.40 * float(attribute_scores["a1"]))
            + (0.25 * float(attribute_scores["a2"]))
            + (0.20 * float(attribute_scores["a3"]))
            + (0.10 * float(attribute_scores["a4"]))
            + (0.05 * float(attribute_scores["a5"]))
        )
        stability = 1.0 if attribute_stability is None else attribute_stability
        attribute_qp_safety = max(0.0, min(1.0, (0.65 * attribute_smoothness) + (0.20 * stability) + (0.15 * inverse_entropy)))
    return {
        "geometry_redundancy": geometry_scores,
        "attribute_smoothness": attribute_scores,
        "geometry_stability": geometry_stability,
        "attribute_stability": attribute_stability,
        "inverse_color_entropy": inverse_entropy,
        "geometry_qp_safety": geometry_qp_safety,
        "attribute_qp_safety": attribute_qp_safety,
        "shared_color_voxels": shared_color_voxels,
    }


def _safe_density(point_count: int, ranges: dict[str, float]) -> float | None:
    volume = float(ranges["x"]) * float(ranges["y"]) * float(ranges["z"])
    if volume <= NEAR_ZERO_EXTENT or not math.isfinite(volume):
        return None
    return point_count / volume


def _aggregate_numeric(values: list[float], method: str) -> float | None:
    if not values:
        return None
    if method == "minimum":
        return min(values)
    if method == "maximum":
        return max(values)
    if method == "median":
        return statistics.median(values)
    if method == "standard_deviation":
        return statistics.pstdev(values) if len(values) > 1 else 0.0
    return statistics.fmean(values)


def _segment_score(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"score": None, "mean": None, "median": None, "min": None, "max": None, "stddev": None}
    median = statistics.median(values)
    minimum = min(values)
    return {
        "score": (0.60 * median) + (0.40 * minimum),
        "mean": statistics.fmean(values),
        "median": median,
        "min": minimum,
        "max": max(values),
        "stddev": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def _build_point_sequence_segments(frames: list[dict[str, Any]], segment_size: int, aggregation: str) -> list[dict[str, Any]]:
    if segment_size <= 0:
        segment_size = 30
    segments: list[dict[str, Any]] = []
    for offset in range(0, len(frames), segment_size):
        segment_frames = frames[offset : offset + segment_size]
        if not segment_frames:
            continue
        segment_id = (offset // segment_size) + 1
        def values(key: str) -> list[float]:
            return [float(frame[key]) for frame in segment_frames if frame.get(key) is not None]
        geometry_scores = [(float(frame["geometry_qp_safety"]), int(frame["frame_index"])) for frame in segment_frames if frame.get("geometry_qp_safety") is not None]
        attribute_scores = [(float(frame["attribute_qp_safety"]), int(frame["frame_index"])) for frame in segment_frames if frame.get("attribute_qp_safety") is not None]
        geometry_payload = _segment_score([score for score, _frame_index in geometry_scores])
        attribute_payload = _segment_score([score for score, _frame_index in attribute_scores])
        segment: dict[str, Any] = {
            "segment_id": segment_id,
            "start_frame_index": segment_frames[0]["frame_index"],
            "end_frame_index": segment_frames[-1]["frame_index"],
            "frame_count": len(segment_frames),
            "aggregation": aggregation,
            "min_x": min(float(frame["min_x"]) for frame in segment_frames),
            "max_x": max(float(frame["max_x"]) for frame in segment_frames),
            "min_y": min(float(frame["min_y"]) for frame in segment_frames),
            "max_y": max(float(frame["max_y"]) for frame in segment_frames),
            "min_z": min(float(frame["min_z"]) for frame in segment_frames),
            "max_z": max(float(frame["max_z"]) for frame in segment_frames),
            "range_x": _aggregate_numeric(values("range_x"), aggregation),
            "range_y": _aggregate_numeric(values("range_y"), aggregation),
            "range_z": _aggregate_numeric(values("range_z"), aggregation),
            "point_count": _aggregate_numeric(values("point_count"), aggregation),
            "density": _aggregate_numeric(values("density"), aggregation),
            "geometry_qp_safety": geometry_payload["score"],
            "attribute_qp_safety": attribute_payload["score"],
            "geometry_score_stats": geometry_payload,
            "attribute_score_stats": attribute_payload,
            "geometry_worst_frame_index": min(geometry_scores)[1] if geometry_scores else None,
            "attribute_worst_frame_index": min(attribute_scores)[1] if attribute_scores else None,
            "geometry_stability": _aggregate_numeric(values("geometry_stability"), aggregation),
            "attribute_stability": _aggregate_numeric(values("attribute_stability"), aggregation),
            "inverse_color_entropy": _aggregate_numeric(values("inverse_color_entropy"), aggregation),
        }
        for radius in POINT_SEQUENCE_SCORE_RADII:
            segment[f"g{radius}"] = _aggregate_numeric(values(f"g{radius}"), aggregation)
            segment[f"a{radius}"] = _aggregate_numeric(values(f"a{radius}"), aggregation)
        segments.append(segment)
    return segments


def analyze_point_cloud_sequence(
    root_folder: str,
    recursive: bool = True,
    segment_size: int = 30,
    aggregation: str = "mean",
    voxel_bits: int = 10,
    use_native_voxels: bool = False,
    max_points_for_qp: int = 120000,
) -> dict[str, Any]:
    warnings: list[dict[str, Any]] = []
    root = safe_path(root_folder)
    files = _point_sequence_files(root, recursive, warnings)
    if not files:
        raise ValueError("No indexed PLY files were found in the selected folder.")
    if aggregation not in POINT_SEQUENCE_AGGREGATIONS:
        raise ValueError("Unsupported segment aggregation method.")
    voxel_bits = int(voxel_bits)
    if voxel_bits < 1 or voxel_bits > 16:
        raise ValueError("Voxel bit depth must be between 1 and 16.")
    segment_size = max(1, int(segment_size or 30))
    max_points_for_qp = max(0, int(max_points_for_qp or 0))

    loaded: list[dict[str, Any]] = []
    for frame_index, path in files:
        try:
            payload = _read_ply_points_for_sequence(path, max_points_for_qp)
        except (OSError, UnicodeDecodeError, ValueError, struct.error) as exc:
            _warning(warnings, "skipped_frame", f"Could not read PLY frame: {exc}", path)
            continue
        loaded.append(
            {
                "frame_index": frame_index,
                "path": path,
                "file_name": path.name,
                **payload,
            }
        )
    if not loaded:
        raise ValueError("No readable PLY frames could be processed.")

    sequence_bounds = _global_bounds_from_frames(loaded)
    shells, max_possible = _precompute_radius_shells()
    frames: list[dict[str, Any]] = []
    previous_occupied: set[tuple[int, int, int]] | None = None
    previous_colors: dict[tuple[int, int, int], tuple[float, float, float]] | None = None
    color_frames = 0
    sampled_frames = 0
    for frame in loaded:
        occupied, colors = _voxelize_sequence_points(
            frame["points"],
            sequence_bounds,
            voxel_bits,
            bool(use_native_voxels),
        )
        if frame["has_color"] and colors:
            color_frames += 1
        if frame["sampled"]:
            sampled_frames += 1
        scores = _score_voxel_frame(occupied, colors, previous_occupied, previous_colors, shells, max_possible)
        ranges = frame["ranges"]
        bounds = frame["bounds"]
        row: dict[str, Any] = {
            "frame_index": frame["frame_index"],
            "file_name": frame["file_name"],
            "path": path_for_client(frame["path"]),
            "declared_vertex_count": frame["declared_vertex_count"],
            "point_count": frame["point_count"],
            "sampled_point_count": frame["sampled_point_count"],
            "sample_stride": frame["sample_stride"],
            "min_x": bounds["min_x"],
            "max_x": bounds["max_x"],
            "min_y": bounds["min_y"],
            "max_y": bounds["max_y"],
            "min_z": bounds["min_z"],
            "max_z": bounds["max_z"],
            "range_x": ranges["x"],
            "range_y": ranges["y"],
            "range_z": ranges["z"],
            "density": _safe_density(int(frame["point_count"]), ranges),
            "occupied_voxels": len(occupied),
            "has_color": bool(frame["has_color"] and colors),
            "geometry_stability": scores["geometry_stability"],
            "attribute_stability": scores["attribute_stability"],
            "inverse_color_entropy": scores["inverse_color_entropy"],
            "geometry_qp_safety": scores["geometry_qp_safety"],
            "attribute_qp_safety": scores["attribute_qp_safety"],
            "shared_color_voxels": scores["shared_color_voxels"],
        }
        row.update(scores["geometry_redundancy"])
        row.update(scores["attribute_smoothness"])
        frames.append(row)
        previous_occupied = occupied
        previous_colors = colors

    if color_frames < len(frames):
        _warning(warnings, "attribute_scoring", "Some frames do not contain usable RGB attributes; attribute QP scoring is unavailable for those frames.")
    if sampled_frames:
        _warning(warnings, "sampling", f"QP scoring used deterministic point sampling on {sampled_frames} frame(s). Increase the max-points option for full-resolution scoring.")

    segments = _build_point_sequence_segments(frames, segment_size, aggregation)
    indices = [frame_index for frame_index, _path in files]
    return {
        "root_folder": path_for_client(root),
        "recursive": bool(recursive),
        "summary": {
            "detected_ply_files": len(files),
            "first_frame_index": min(indices) if indices else None,
            "last_frame_index": max(indices) if indices else None,
            "processed_frames": len(frames),
            "skipped_frames": len(files) - len(frames),
            "color_frames": color_frames,
            "sampled_frames": sampled_frames,
        },
        "options": {
            "segment_size": segment_size,
            "aggregation": aggregation,
            "voxel_bits": voxel_bits,
            "use_native_voxels": bool(use_native_voxels),
            "max_points_for_qp": max_points_for_qp,
        },
        "sequence_bounds": sequence_bounds,
        "frames": frames,
        "segments": segments,
        "warnings": warnings,
    }


def resolve_reference_ply(reference_path: str) -> tuple[Path, dict[str, Any]]:
    path = safe_path(reference_path)
    if not path.exists():
        raise ValueError("Reference PLY path does not exist.")
    if path.is_file():
        if path.suffix.lower() != ".ply":
            raise ValueError("Reference PLY path must be a .ply file or a folder containing .ply frames.")
        return path, {
            "mode": "file",
            "reference_frame_number": None,
            "sequence_folder": None,
            "frame_count": 1,
        }
    if path.is_dir():
        frames = list_indexed_ply_frames(path)
        frame_number, first_frame = frames[0]
        return first_frame, {
            "mode": "sequence",
            "reference_frame_number": frame_number,
            "sequence_folder": path_for_client(path),
            "frame_count": len(frames),
        }
    raise ValueError("Reference PLY path must be a file or folder.")


def inspect_ply_point_cloud(ply_path: str) -> dict[str, Any]:
    path, reference_info = resolve_reference_ply(ply_path)
    if not path.exists():
        raise ValueError("Reference PLY file does not exist.")
    if not path.is_file():
        raise ValueError("Reference PLY path must be a file.")

    header = _parse_ply_header(path)
    props: list[tuple[str, str]] = header["vertex_properties"]
    names = [name for name, _type_name in props]
    x_idx = names.index("x")
    y_idx = names.index("y")
    z_idx = names.index("z")
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    valid_points = 0

    if header["format"] == "ascii":
        with path.open("rb") as handle:
            handle.seek(header["data_offset"])
            for _ in range(header["vertex_count"]):
                raw = handle.readline()
                if not raw:
                    break
                parts = raw.decode("utf-8", errors="replace").strip().split()
                if len(parts) < len(props):
                    continue
                values = [_parse_numeric(parts[x_idx]), _parse_numeric(parts[y_idx]), _parse_numeric(parts[z_idx])]
                if any(value is None for value in values):
                    continue
                for idx, value in enumerate(values):
                    mins[idx] = min(mins[idx], value)
                    maxs[idx] = max(maxs[idx], value)
                valid_points += 1
    else:
        endian = "<" if header["format"] == "binary_little_endian" else ">"
        try:
            format_chars = [PLY_SCALAR_TYPES[type_name][0] for _name, type_name in props]
        except KeyError as exc:
            raise ValueError(f"Unsupported PLY vertex property type: {exc.args[0]}") from exc
        row_struct = struct.Struct(endian + "".join(format_chars))
        with path.open("rb") as handle:
            handle.seek(header["data_offset"])
            for _ in range(header["vertex_count"]):
                row = handle.read(row_struct.size)
                if len(row) < row_struct.size:
                    break
                unpacked = row_struct.unpack(row)
                values = [_parse_numeric(unpacked[x_idx]), _parse_numeric(unpacked[y_idx]), _parse_numeric(unpacked[z_idx])]
                if any(value is None for value in values):
                    continue
                for idx, value in enumerate(values):
                    mins[idx] = min(mins[idx], value)
                    maxs[idx] = max(maxs[idx], value)
                valid_points += 1

    if valid_points <= 0:
        raise ValueError("PLY file has no valid x/y/z vertex points.")
    extents = {
        "x": maxs[0] - mins[0],
        "y": maxs[1] - mins[1],
        "z": maxs[2] - mins[2],
    }
    zero_axes = [axis.upper() for axis, extent in extents.items() if extent <= NEAR_ZERO_EXTENT or not math.isfinite(extent)]
    if zero_axes:
        raise ValueError(f"PLY bounding box has zero or near-zero extent on: {', '.join(zero_axes)}.")

    return {
        "path": path_for_client(path),
        "filename": path.name,
        "reference": reference_info,
        "point_count": valid_points,
        "declared_vertex_count": header["vertex_count"],
        "bounds": {
            "min_x": mins[0],
            "max_x": maxs[0],
            "min_y": mins[1],
            "max_y": maxs[1],
            "min_z": mins[2],
            "max_z": maxs[2],
        },
        "extents": extents,
    }


def _format_ply_value(value: Any, type_name: str) -> str:
    numeric = _parse_numeric(value)
    if numeric is None:
        return str(value)
    if type_name in PLY_INT_TYPES:
        return str(int(round(numeric)))
    return f"{numeric:.12g}"


def _read_ply_vertex_rows(path: Path, header: dict[str, Any]) -> list[list[Any]]:
    props: list[tuple[str, str]] = header["vertex_properties"]
    rows: list[list[Any]] = []
    if header["format"] == "ascii":
        with path.open("rb") as handle:
            handle.seek(header["data_offset"])
            for _ in range(header["vertex_count"]):
                raw = handle.readline()
                if not raw:
                    break
                parts = raw.decode("utf-8", errors="replace").strip().split()
                if len(parts) < len(props):
                    continue
                rows.append(parts[: len(props)])
        return rows

    endian = "<" if header["format"] == "binary_little_endian" else ">"
    try:
        format_chars = [PLY_SCALAR_TYPES[type_name][0] for _name, type_name in props]
    except KeyError as exc:
        raise ValueError(f"Unsupported PLY vertex property type: {exc.args[0]}") from exc
    row_struct = struct.Struct(endian + "".join(format_chars))
    with path.open("rb") as handle:
        handle.seek(header["data_offset"])
        for _ in range(header["vertex_count"]):
            row = handle.read(row_struct.size)
            if len(row) < row_struct.size:
                break
            rows.append(list(row_struct.unpack(row)))
    return rows


def _write_ascii_ply(path: Path, props: list[tuple[str, str]], rows: list[list[Any]], comments: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        for comment in comments or []:
            handle.write(f"comment {comment}\n")
        handle.write(f"element vertex {len(rows)}\n")
        for name, type_name in props:
            handle.write(f"property {type_name} {name}\n")
        handle.write("end_header\n")
        for row in rows:
            handle.write(" ".join(_format_ply_value(value, props[idx][1]) for idx, value in enumerate(row)))
            handle.write("\n")


def _summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"min": None, "max": None, "median": None, "mean": None, "stddev": None, "samples": 0}
    return {
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "stddev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "samples": len(values),
    }


def _quantile(sorted_values: list[float], percentile: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[int(rank)]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    return lower_value + ((upper_value - lower_value) * (rank - lower))


def _iqr_bounds(values: list[float]) -> dict[str, Any]:
    if len(values) < 4:
        return {"lower": None, "upper": None, "q1": None, "q3": None, "iqr": None}
    sorted_values = sorted(values)
    q1 = _quantile(sorted_values, 0.25)
    q3 = _quantile(sorted_values, 0.75)
    if q1 is None or q3 is None:
        return {"lower": None, "upper": None, "q1": q1, "q3": q3, "iqr": None}
    iqr = q3 - q1
    return {
        "lower": q1 - (1.5 * iqr),
        "upper": q3 + (1.5 * iqr),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
    }


def _percentile_bounds(values: list[float], keep_percent: float) -> dict[str, Any]:
    if not values:
        return {"lower": None, "upper": None, "q1": None, "q3": None, "iqr": None}
    clipped_keep = min(max(keep_percent, 0.0), 100.0)
    tail = (100.0 - clipped_keep) / 2.0
    sorted_values = sorted(values)
    lower = _quantile(sorted_values, tail / 100.0)
    upper = _quantile(sorted_values, 1.0 - (tail / 100.0))
    return {"lower": lower, "upper": upper, "q1": None, "q3": None, "iqr": None}


def _filter_values(values: list[float], method: str, percentile_keep: float) -> dict[str, Any]:
    if method == "percentile":
        bounds = _percentile_bounds(values, percentile_keep)
    else:
        bounds = _iqr_bounds(values)
    lower = bounds["lower"]
    upper = bounds["upper"]
    if lower is None or upper is None:
        filtered = list(values)
        removed: list[float] = []
        outliers = 0
    else:
        filtered = [value for value in values if lower <= value <= upper]
        removed = [value for value in values if value < lower or value > upper]
        outliers = len(values) - len(filtered)
    total = len(values)
    return {
        "values": filtered,
        "removed_values": removed,
        "bounds": bounds,
        "outliers": outliers,
        "total": total,
        "valid": len(filtered),
        "percentage_removed": (outliers / total) * 100 if total else 0,
    }


def _metric_payload(
    values: list[float],
    filter_outliers: bool,
    method: str,
    percentile_keep: float,
    histogram_source: str,
) -> dict[str, Any]:
    filtered_result = _filter_values(values, method, percentile_keep) if filter_outliers else {
        "values": list(values),
        "removed_values": [],
        "bounds": {"lower": None, "upper": None, "q1": None, "q3": None, "iqr": None},
        "outliers": 0,
        "total": len(values),
        "valid": len(values),
        "percentage_removed": 0,
    }
    histogram_values = values if histogram_source == "raw" else filtered_result["values"]
    return {
        "stats": _summarize(filtered_result["values"]),
        "histogram": _histogram(histogram_values),
        "raw_stats": _summarize(values),
        "filtered_stats": _summarize(filtered_result["values"]),
        "raw_histogram": _histogram(values),
        "filtered_histogram": _histogram(filtered_result["values"]),
        "outlier_summary": {
            "total_samples": filtered_result["total"],
            "valid_samples": filtered_result["valid"],
            "outlier_samples": filtered_result["outliers"],
            "percentage_removed": filtered_result["percentage_removed"],
            "largest_removed": max(filtered_result["removed_values"], key=abs) if filtered_result["removed_values"] else None,
            "largest_retained": max(filtered_result["values"], key=abs) if filtered_result["values"] else None,
            "bounds": filtered_result["bounds"],
        },
    }


def _histogram(values: list[float]) -> list[dict[str, Any]]:
    if not values:
        return []

    min_value = min(values)
    max_value = max(values)
    counts = [0 for _ in range(10)]
    if min_value == max_value:
        counts[-1] = len(values)
    else:
        span = max_value - min_value
        for value in values:
            index = int(((value - min_value) / span) * 10)
            counts[min(max(index, 0), 9)] += 1

    total = len(values)
    bins: list[dict[str, Any]] = []
    for index, count in enumerate(counts):
        lower_pct = index * 10
        upper_pct = (index + 1) * 10
        lower_value = min_value if min_value == max_value else min_value + ((lower_pct / 100) * (max_value - min_value))
        upper_value = max_value if index == 9 else min_value + ((upper_pct / 100) * (max_value - min_value))
        bins.append(
            {
                "label": f"{_format_bin_value(lower_value)} to {_format_bin_value(upper_value)}",
                "lower_percent": lower_pct,
                "upper_percent": upper_pct,
                "lower": lower_value,
                "upper": upper_value,
                "count": count,
                "percentage": (count / total) * 100 if total else 0,
            }
        )
    return bins


def _format_bin_value(value: float) -> str:
    if abs(value) >= 1000 or (0 < abs(value) < 0.01):
        return f"{value:.3e}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _parse_numeric(raw: Any) -> float | None:
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _has_column(fieldnames: list[str], selected: str) -> bool:
    return any(str(fieldname or "").strip() == selected for fieldname in fieldnames)


def _row_value(row: dict[str, Any], selected: str) -> Any:
    if selected in row:
        return row.get(selected)
    for key, value in row.items():
        if str(key or "").strip() == selected:
            return value
    return None


def _row_key(row: dict[str, Any], selected: str) -> str | None:
    if selected in row:
        return selected
    for key in row:
        if str(key or "").strip() == selected:
            return key
    return None


def _collision_free_path(output_dir: Path, filename: str, used_names: set[str]) -> tuple[Path, bool]:
    source_name = Path(filename).name
    stem = Path(source_name).stem or "trace"
    suffix = Path(source_name).suffix or ".csv"
    candidate = output_dir / f"{stem}{suffix}"
    index = 1
    renamed = False
    while candidate.name.lower() in used_names or candidate.exists():
        candidate = output_dir / f"{stem}_{index}{suffix}"
        index += 1
        renamed = True
    used_names.add(candidate.name.lower())
    return candidate, renamed


def _height_axis_output_values(
    x_value: float,
    y_value: float,
    z_value: float,
    source_height_axis: str,
    output_height_axis: str,
) -> tuple[float, float, float]:
    values = {
        "x": x_value,
        "y": y_value,
        "z": z_value,
    }
    if source_height_axis == output_height_axis:
        return x_value, y_value, z_value
    values[source_height_axis], values[output_height_axis] = values[output_height_axis], values[source_height_axis]
    return values["x"], values["y"], values["z"]


def _axis_bounds_after_swap(bounds: dict[str, Any], source_height_axis: str, output_height_axis: str) -> dict[str, tuple[float, float]]:
    mins = {
        "x": float(bounds["min_x"]),
        "y": float(bounds["min_y"]),
        "z": float(bounds["min_z"]),
    }
    maxs = {
        "x": float(bounds["max_x"]),
        "y": float(bounds["max_y"]),
        "z": float(bounds["max_z"]),
    }
    if source_height_axis != output_height_axis:
        mins[source_height_axis], mins[output_height_axis] = mins[output_height_axis], mins[source_height_axis]
        maxs[source_height_axis], maxs[output_height_axis] = maxs[output_height_axis], maxs[source_height_axis]
    return {axis: (min(mins[axis], maxs[axis]), max(mins[axis], maxs[axis])) for axis in ("x", "y", "z")}


def _bounds_extents(bounds: dict[str, tuple[float, float]]) -> dict[str, float]:
    return {axis: high - low for axis, (low, high) in bounds.items()}


def _tuple_bounds_to_dict(bounds: dict[str, tuple[float, float]]) -> dict[str, dict[str, float]]:
    return {axis: {"min": low, "max": high} for axis, (low, high) in bounds.items()}


def _operation_label(axis_swap_enabled: bool, positive_coordinates: bool, voxelized: bool = False) -> str:
    if voxelized:
        base = "voxelized transformed data" if axis_swap_enabled else "voxelized scaled data"
    else:
        base = "transformed data" if axis_swap_enabled else "scaled data"
    if positive_coordinates and not voxelized:
        return base.replace("data", "+ shifted data")
    if positive_coordinates and voxelized:
        return base.replace("data", "from shifted data")
    return base


def _empty_bounds() -> dict[str, float]:
    return {
        "min_x": math.inf,
        "max_x": -math.inf,
        "min_y": math.inf,
        "max_y": -math.inf,
        "min_z": math.inf,
        "max_z": -math.inf,
    }


def _expand_bounds(bounds: dict[str, float], x_value: float, y_value: float, z_value: float) -> None:
    bounds["min_x"] = min(bounds["min_x"], x_value)
    bounds["max_x"] = max(bounds["max_x"], x_value)
    bounds["min_y"] = min(bounds["min_y"], y_value)
    bounds["max_y"] = max(bounds["max_y"], y_value)
    bounds["min_z"] = min(bounds["min_z"], z_value)
    bounds["max_z"] = max(bounds["max_z"], z_value)


def _merge_bounds(first: dict[str, float], second: dict[str, float] | None) -> dict[str, float]:
    if not second:
        return first
    return {
        "min_x": min(first["min_x"], second["min_x"]),
        "max_x": max(first["max_x"], second["max_x"]),
        "min_y": min(first["min_y"], second["min_y"]),
        "max_y": max(first["max_y"], second["max_y"]),
        "min_z": min(first["min_z"], second["min_z"]),
        "max_z": max(first["max_z"], second["max_z"]),
    }


def _finalize_bounds(bounds: dict[str, float]) -> dict[str, float] | None:
    values = list(bounds.values())
    if not values or not all(math.isfinite(value) for value in values):
        return None
    return bounds


def _build_dataset_transform(
    ply_summary: dict[str, Any],
    height_axis: str,
    output_height_axis: str,
    real_height_m: float,
    height_axis_starts_at_zero: bool,
    positive_coordinates: bool,
    voxel_bits: int | None,
    bounds_override: dict[str, Any] | None = None,
    zero_shift_bounds_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    original_reference_bounds = _axis_bounds_after_swap(ply_summary["bounds"], height_axis, height_axis)
    swapped_reference_bounds = _axis_bounds_after_swap(ply_summary["bounds"], height_axis, output_height_axis)
    swapped_reference_extents = _bounds_extents(swapped_reference_bounds)
    measured_extent = float(swapped_reference_extents[output_height_axis])
    if measured_extent <= NEAR_ZERO_EXTENT or not math.isfinite(measured_extent):
        raise ValueError("Selected height extent is zero or near-zero; scaling cannot be computed.")
    scale_factor = real_height_m / measured_extent
    if scale_factor <= 0 or not math.isfinite(scale_factor):
        raise ValueError("Computed scale factor is invalid.")

    swapped_bounds = _axis_bounds_after_swap(bounds_override or ply_summary["bounds"], height_axis, output_height_axis)
    zero_shift_source_bounds = _axis_bounds_after_swap(zero_shift_bounds_override or ply_summary["bounds"], height_axis, output_height_axis)
    scaled_bounds = {
        axis: (low * scale_factor, high * scale_factor)
        for axis, (low, high) in swapped_bounds.items()
    }
    zero_shift_scaled_bounds = {
        axis: (low * scale_factor, high * scale_factor)
        for axis, (low, high) in zero_shift_source_bounds.items()
    }
    height_axis_min_before_zero_shift = float(zero_shift_scaled_bounds[output_height_axis][0])
    height_axis_zero_shift_offset_m = -height_axis_min_before_zero_shift if height_axis_starts_at_zero else 0.0
    zero_shifted_bounds = {
        axis: (
            low + (height_axis_zero_shift_offset_m if axis == output_height_axis else 0.0),
            high + (height_axis_zero_shift_offset_m if axis == output_height_axis else 0.0),
        )
        for axis, (low, high) in scaled_bounds.items()
    }
    height_axis_min_after_zero_shift = float(zero_shifted_bounds[output_height_axis][0])
    positive_translation = {
        axis: (-low if positive_coordinates and low < 0 else 0.0)
        for axis, (low, _high) in zero_shifted_bounds.items()
    }
    shifted_bounds = {
        axis: (low + positive_translation[axis], high + positive_translation[axis])
        for axis, (low, high) in zero_shifted_bounds.items()
    }
    voxel_scale = None
    voxel_max = None
    if voxel_bits is not None:
        voxel_max = (2 ** int(voxel_bits)) - 1
        largest_extent = max(max(high - low, 0.0) for low, high in shifted_bounds.values())
        if largest_extent <= NEAR_ZERO_EXTENT or not math.isfinite(largest_extent):
            raise ValueError("Reference bounds are too small for voxelization.")
        voxel_scale = float(voxel_max) / largest_extent

    return {
        "height_axis": height_axis,
        "output_height_axis": output_height_axis,
        "measured_extent_units": measured_extent,
        "scale_factor_m_per_unit": scale_factor,
        "height_axis_starts_at_zero": bool(height_axis_starts_at_zero),
        "height_axis_zero_shift_axis": output_height_axis,
        "height_axis_zero_shift_offset_m": height_axis_zero_shift_offset_m,
        "positive_coordinates": bool(positive_coordinates),
        "positive_translation_m": positive_translation,
        "scaled_reference_bounds_m": scaled_bounds,
        "height_axis_zero_shifted_bounds_m": zero_shifted_bounds,
        "shifted_reference_bounds_m": shifted_bounds,
        "final_exported_bounds_m": shifted_bounds,
        "voxel_bits": voxel_bits,
        "voxel_max": voxel_max,
        "voxel_scale": voxel_scale,
        "axis_swap": {
            "enabled": height_axis != output_height_axis,
            "source_height_axis": height_axis,
            "output_height_axis": output_height_axis,
        },
        "debug_details": {
            "original_reference_extents_units": ply_summary["extents"],
            "original_reference_bounds_units": _tuple_bounds_to_dict(original_reference_bounds),
            "swapped_reference_extents_units": swapped_reference_extents,
            "swapped_reference_bounds_units": _tuple_bounds_to_dict(swapped_reference_bounds),
            "combined_bounds_after_axis_swap_units": _tuple_bounds_to_dict(swapped_bounds),
            "height_axis_zero_shift_source_bounds_units": _tuple_bounds_to_dict(zero_shift_source_bounds),
            "original_height_axis": height_axis,
            "selected_height_axis_after_axis_swap": output_height_axis,
            "measured_height_extent_units": measured_extent,
            "desired_real_world_height_m": real_height_m,
            "known_real_world_height_m": real_height_m,
            "computed_scale_factor_m_per_unit": scale_factor,
            "height_axis_minimum_before_zero_shift_m": height_axis_min_before_zero_shift,
            "height_axis_zero_shift_offset_m": height_axis_zero_shift_offset_m,
            "height_axis_minimum_after_zero_shift_m": height_axis_min_after_zero_shift,
            "final_exported_extents_m": _bounds_extents(shifted_bounds),
        },
    }


def _apply_dataset_transform_values(
    x_value: float,
    y_value: float,
    z_value: float,
    transform: dict[str, Any],
    voxelized: bool = False,
) -> tuple[float, float, float] | tuple[int, int, int]:
    scale = float(transform["scale_factor_m_per_unit"])
    x_scaled, y_scaled, z_scaled = _height_axis_output_values(
        x_value * scale,
        y_value * scale,
        z_value * scale,
        str(transform["height_axis"]),
        str(transform["output_height_axis"]),
    )
    zero_shift_axis = str(transform.get("height_axis_zero_shift_axis") or transform["output_height_axis"])
    zero_shift_offset = float(transform.get("height_axis_zero_shift_offset_m") or 0.0)
    zero_shifted_values = {
        "x": x_scaled,
        "y": y_scaled,
        "z": z_scaled,
    }
    zero_shifted_values[zero_shift_axis] += zero_shift_offset
    translation = transform["positive_translation_m"]
    shifted = (
        zero_shifted_values["x"] + float(translation["x"]),
        zero_shifted_values["y"] + float(translation["y"]),
        zero_shifted_values["z"] + float(translation["z"]),
    )
    if not voxelized:
        return shifted
    voxel_scale = transform.get("voxel_scale")
    voxel_max = transform.get("voxel_max")
    if voxel_scale is None or voxel_max is None:
        raise ValueError("Voxelization requested but no voxel transform was computed.")
    return tuple(
        int(max(0, min(int(voxel_max), round(value * float(voxel_scale)))))
        for value in shifted
    )


def _frame_column_key(fieldnames: list[str]) -> str | None:
    for field in fieldnames:
        if str(field or "").strip().lower() in {"frame", "frame_index", "frame_id"}:
            return field
    return None


def _format_csv_number(value: float) -> str:
    if math.isfinite(value) and abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.12g}"


def _target_frame_numbers(frame_numbers: list[int] | None, target_count: int) -> list[int]:
    if frame_numbers and len(frame_numbers) == target_count:
        return [int(value) for value in frame_numbers]
    if frame_numbers and target_count > 1:
        start = int(frame_numbers[0])
        end = int(frame_numbers[-1])
        return [int(round(start + (idx * (end - start) / (target_count - 1)))) for idx in range(target_count)]
    if frame_numbers and target_count == 1:
        return [int(frame_numbers[0])]
    return list(range(1, target_count + 1))


def _resample_csv_rows(
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    target_count: int,
    frame_numbers: list[int] | None,
    coordinate_keys: set[str],
    voxelized: bool,
) -> list[dict[str, Any]]:
    if target_count <= 0 or not rows:
        return []
    if len(rows) == target_count:
        out = [dict(row) for row in rows]
    elif target_count == 1:
        out = [dict(rows[0])]
    elif len(rows) == 1:
        out = [dict(rows[0]) for _ in range(target_count)]
    else:
        out = []
        last_index = len(rows) - 1
        for target_idx in range(target_count):
            position = target_idx * last_index / (target_count - 1)
            lower_idx = int(math.floor(position))
            upper_idx = int(math.ceil(position))
            ratio = position - lower_idx
            lower = rows[lower_idx]
            upper = rows[upper_idx]
            nearest = lower if ratio < 0.5 else upper
            row: dict[str, Any] = {}
            for field in fieldnames:
                low_value = _parse_numeric(lower.get(field))
                high_value = _parse_numeric(upper.get(field))
                if low_value is not None and high_value is not None:
                    interpolated = float(low_value) + ((float(high_value) - float(low_value)) * ratio)
                    if voxelized and field in coordinate_keys:
                        row[field] = str(int(round(interpolated)))
                    else:
                        row[field] = _format_csv_number(interpolated)
                else:
                    row[field] = nearest.get(field, "")
            out.append(row)

    frame_key = _frame_column_key(fieldnames)
    if frame_key:
        for row, frame_number in zip(out, _target_frame_numbers(frame_numbers, target_count)):
            row[frame_key] = str(int(frame_number))
    return out


def _resolve_fov_resampling_plan(
    enabled: bool,
    requested_target: int | None,
    reference_ply_path: str,
    ply_input_path: str,
    transform_point_clouds: bool,
    warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False, "target_samples": None, "inferred": False, "source": None, "frame_numbers": None}

    source_raw = ply_input_path if transform_point_clouds and str(ply_input_path or "").strip() else reference_ply_path
    source = safe_path(source_raw) if str(source_raw or "").strip() else None
    frame_numbers: list[int] | None = None
    inferred_target: int | None = None
    source_label: str | None = None

    if source is not None and source.is_dir():
        try:
            frames = list_indexed_ply_frames(source)
            frame_numbers = [int(frame_number) for frame_number, _path in frames]
            inferred_target = len(frame_numbers)
            source_label = path_for_client(source)
        except ValueError as exc:
            _warning(warnings, "fov_resampling", f"Could not infer FoV CSV resampling target from PLY sequence: {exc}", source)

    target = int(requested_target) if requested_target is not None else inferred_target
    if target is None:
        return {"enabled": False, "target_samples": None, "inferred": False, "source": source_label, "frame_numbers": None}
    if target <= 0:
        raise ValueError("FoV CSV FPS / resampling target must be greater than zero.")

    return {
        "enabled": True,
        "target_samples": int(target),
        "inferred": requested_target is None and inferred_target is not None,
        "source": source_label,
        "source_frame_count": inferred_target,
        "frame_numbers": frame_numbers,
    }


def normalize_trace_height(
    root_folder: str,
    output_folder: str,
    x_column: str,
    y_column: str,
    z_column: str,
    ply_path: str,
    height_axis: str,
    real_height_m: float,
    output_height_axis: str | None = None,
) -> dict[str, Any]:
    if not x_column:
        raise ValueError("X position column is required.")
    if not y_column:
        raise ValueError("Y position column is required.")
    if not z_column:
        raise ValueError("Z position column is required.")
    if height_axis not in {"x", "y", "z"}:
        raise ValueError("Height axis must be x, y, or z.")
    output_height_axis = output_height_axis or height_axis
    if output_height_axis not in {"x", "y", "z"}:
        raise ValueError("Output height axis must be x, y, or z.")
    if real_height_m <= 0 or not math.isfinite(real_height_m):
        raise ValueError("Real-world height must be a positive number of meters.")

    warnings: list[dict[str, Any]] = []
    ply_summary = inspect_ply_point_cloud(ply_path)
    measured_extent = float(ply_summary["extents"][height_axis])
    if measured_extent <= NEAR_ZERO_EXTENT or not math.isfinite(measured_extent):
        raise ValueError("Selected height extent is zero or near-zero; scaling cannot be computed.")
    scale_factor = real_height_m / measured_extent
    if scale_factor <= 0 or not math.isfinite(scale_factor):
        raise ValueError("Computed scale factor is invalid.")
    if scale_factor < 0.001 or scale_factor > 1000:
        _warning(
            warnings,
            "scaling",
            "Scale factor is extreme; verify the selected height axis, PLY units, and real-world height.",
        )
    elif scale_factor < 0.01 or scale_factor > 100:
        _warning(
            warnings,
            "scaling",
            "Scale factor is suspiciously large or small; verify the PLY reference scale.",
        )

    root = safe_path(root_folder)
    output_dir = safe_path(output_folder)
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("Output path must be a directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    files = _csv_files(root, warnings)
    used_output_names = {path.name.lower() for path in output_dir.iterdir() if path.is_file()}
    files_processed = 0
    files_skipped = 0
    rows_seen = 0
    rows_transformed = 0
    rows_skipped = 0
    missing_coordinate_rows = 0
    non_numeric_coordinate_rows = 0
    output_files: list[dict[str, Any]] = []

    for path in files:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as read_handle:
                reader = csv.DictReader(read_handle)
                fieldnames = reader.fieldnames or []
                if not fieldnames:
                    files_skipped += 1
                    _warning(warnings, "skipped_file", "CSV file is empty or missing a header row.", path)
                    continue
                if not _has_column(fieldnames, x_column) or not _has_column(fieldnames, y_column) or not _has_column(fieldnames, z_column):
                    files_skipped += 1
                    _warning(warnings, "skipped_file", "CSV does not contain all selected position columns.", path)
                    continue
                column_keys = [_row_key(dict.fromkeys(fieldnames, ""), selected) for selected in (x_column, y_column, z_column)]
                if any(key is None for key in column_keys):
                    files_skipped += 1
                    _warning(warnings, "skipped_file", "CSV does not contain all selected position columns.", path)
                    continue

                output_path, renamed = _collision_free_path(output_dir, path.name, used_output_names)
                file_rows_seen = 0
                file_rows_written = 0
                file_rows_skipped = 0
                with output_path.open("w", encoding="utf-8", newline="") as write_handle:
                    writer = csv.DictWriter(write_handle, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    for row in reader:
                        rows_seen += 1
                        file_rows_seen += 1
                        raw_values = [_row_value(row, selected) for selected in (x_column, y_column, z_column)]
                        if any(value is None or str(value).strip() == "" for value in raw_values):
                            rows_skipped += 1
                            file_rows_skipped += 1
                            missing_coordinate_rows += 1
                            continue
                        numeric_values = [_parse_numeric(value) for value in raw_values]
                        if any(value is None for value in numeric_values):
                            rows_skipped += 1
                            file_rows_skipped += 1
                            non_numeric_coordinate_rows += 1
                            continue
                        scaled_values = [float(value) * scale_factor for value in numeric_values]
                        output_values = _height_axis_output_values(
                            scaled_values[0],
                            scaled_values[1],
                            scaled_values[2],
                            height_axis,
                            output_height_axis,
                        )
                        updated = dict(row)
                        for key, value in zip(column_keys, output_values):
                            updated[str(key)] = f"{value:.12g}"
                        writer.writerow(updated)
                        rows_transformed += 1
                        file_rows_written += 1

                files_processed += 1
                output_files.append(
                    {
                        "source": path_for_client(path),
                        "output": path_for_client(output_path),
                        "source_filename": path.name,
                        "output_filename": output_path.name,
                        "renamed": renamed,
                        "rows_seen": file_rows_seen,
                        "rows_transformed": file_rows_written,
                        "rows_skipped": file_rows_skipped,
                    }
                )
                if renamed:
                    _warning(
                        warnings,
                        "renamed_file",
                        f"{path.name} -> {output_path.name} (collision resolved)",
                        path,
                    )
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            files_skipped += 1
            _warning(warnings, "skipped_file", f"Could not process CSV: {exc}", path)

    if missing_coordinate_rows:
        _warning(warnings, "skipped_rows", f"Skipped {missing_coordinate_rows} rows with missing X, Y, or Z values.")
    if non_numeric_coordinate_rows:
        _warning(warnings, "skipped_rows", f"Skipped {non_numeric_coordinate_rows} rows with non-numeric X, Y, or Z values.")

    return {
        "root_folder": path_for_client(root),
        "output_folder": path_for_client(output_dir),
        "selected_columns": {
            "x": x_column,
            "y": y_column,
            "z": z_column,
        },
        "scaling": {
            "point_cloud": ply_summary,
            "height_axis": height_axis,
            "measured_extent_units": measured_extent,
            "real_height_m": real_height_m,
            "scale_factor_m_per_unit": scale_factor,
            "axis_swap": {
                "enabled": height_axis != output_height_axis,
                "source_height_axis": height_axis,
                "output_height_axis": output_height_axis,
            },
        },
        "summary": {
            "files_found": len(files),
            "files_processed": files_processed,
            "files_skipped": files_skipped,
            "rows_seen": rows_seen,
            "rows_transformed": rows_transformed,
            "rows_skipped": rows_skipped,
            "output_files": len(output_files),
        },
        "output_files": output_files,
        "warnings": warnings,
    }


def _transform_csv_folder(
    root: Path,
    output_dir: Path,
    x_column: str,
    y_column: str,
    z_column: str,
    transform: dict[str, Any],
    warnings: list[dict[str, Any]],
    voxelized: bool = False,
    resampling: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = _csv_files(root, warnings)
    used_output_names = {path.name.lower() for path in output_dir.iterdir() if path.is_file()}
    files_processed = 0
    files_skipped = 0
    rows_seen = 0
    rows_transformed = 0
    rows_skipped = 0
    missing_coordinate_rows = 0
    non_numeric_coordinate_rows = 0
    output_files: list[dict[str, Any]] = []

    for path in files:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as read_handle:
                reader = csv.DictReader(read_handle)
                fieldnames = reader.fieldnames or []
                if not fieldnames:
                    files_skipped += 1
                    _warning(warnings, "skipped_file", "CSV file is empty or missing a header row.", path)
                    continue
                if not _has_column(fieldnames, x_column) or not _has_column(fieldnames, y_column) or not _has_column(fieldnames, z_column):
                    files_skipped += 1
                    _warning(warnings, "skipped_file", "CSV does not contain all selected position columns.", path)
                    continue
                column_keys = [_row_key(dict.fromkeys(fieldnames, ""), selected) for selected in (x_column, y_column, z_column)]
                if any(key is None for key in column_keys):
                    files_skipped += 1
                    _warning(warnings, "skipped_file", "CSV does not contain all selected position columns.", path)
                    continue

                output_path, renamed = _collision_free_path(output_dir, path.name, used_output_names)
                file_rows_seen = 0
                file_rows_written = 0
                file_rows_skipped = 0
                transformed_rows: list[dict[str, Any]] = []
                with output_path.open("w", encoding="utf-8", newline="") as write_handle:
                    writer = csv.DictWriter(write_handle, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    for row in reader:
                        rows_seen += 1
                        file_rows_seen += 1
                        raw_values = [_row_value(row, selected) for selected in (x_column, y_column, z_column)]
                        if any(value is None or str(value).strip() == "" for value in raw_values):
                            rows_skipped += 1
                            file_rows_skipped += 1
                            missing_coordinate_rows += 1
                            continue
                        numeric_values = [_parse_numeric(value) for value in raw_values]
                        if any(value is None for value in numeric_values):
                            rows_skipped += 1
                            file_rows_skipped += 1
                            non_numeric_coordinate_rows += 1
                            continue
                        output_values = _apply_dataset_transform_values(
                            float(numeric_values[0]),
                            float(numeric_values[1]),
                            float(numeric_values[2]),
                            transform,
                            voxelized=voxelized,
                        )
                        updated = dict(row)
                        for key, value in zip(column_keys, output_values):
                            updated[str(key)] = str(value) if voxelized else f"{float(value):.12g}"
                        transformed_rows.append(updated)

                    if resampling and resampling.get("enabled"):
                        transformed_rows = _resample_csv_rows(
                            transformed_rows,
                            fieldnames,
                            int(resampling["target_samples"]),
                            resampling.get("frame_numbers"),
                            {str(key) for key in column_keys if key is not None},
                            voxelized=voxelized,
                        )

                    for updated in transformed_rows:
                        writer.writerow(updated)
                    rows_transformed += len(transformed_rows)
                    file_rows_written = len(transformed_rows)

                files_processed += 1
                output_files.append(
                    {
                        "source": path_for_client(path),
                        "output": path_for_client(output_path),
                        "source_filename": path.name,
                        "output_filename": output_path.name,
                        "renamed": renamed,
                        "rows_seen": file_rows_seen,
                        "rows_transformed": file_rows_written,
                        "rows_skipped": file_rows_skipped,
                        "resampled": bool(resampling and resampling.get("enabled")),
                        "resample_target_rows": int(resampling["target_samples"]) if resampling and resampling.get("enabled") else None,
                    }
                )
                if renamed:
                    _warning(warnings, "renamed_file", f"{path.name} -> {output_path.name} (collision resolved)", path)
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            files_skipped += 1
            _warning(warnings, "skipped_file", f"Could not process CSV: {exc}", path)

    if missing_coordinate_rows:
        _warning(warnings, "skipped_rows", f"Skipped {missing_coordinate_rows} rows with missing X, Y, or Z values.")
    if non_numeric_coordinate_rows:
        _warning(warnings, "skipped_rows", f"Skipped {non_numeric_coordinate_rows} rows with non-numeric X, Y, or Z values.")
    return {
        "root_folder": path_for_client(root),
        "output_folder": path_for_client(output_dir),
        "summary": {
            "files_found": len(files),
            "files_processed": files_processed,
            "files_skipped": files_skipped,
            "rows_seen": rows_seen,
            "rows_transformed": rows_transformed,
            "rows_skipped": rows_skipped,
            "output_files": len(output_files),
        },
        "output_files": output_files,
        "resampling": {
            key: value
            for key, value in (resampling or {"enabled": False}).items()
            if key != "frame_numbers"
        },
    }


def _csv_coordinate_bounds(
    root: Path,
    x_column: str,
    y_column: str,
    z_column: str,
    warnings: list[dict[str, Any]],
) -> dict[str, float] | None:
    bounds = _empty_bounds()
    for path in _csv_files(root, warnings):
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames or []
                if not _has_column(fieldnames, x_column) or not _has_column(fieldnames, y_column) or not _has_column(fieldnames, z_column):
                    continue
                for row in reader:
                    values = [_parse_numeric(_row_value(row, selected)) for selected in (x_column, y_column, z_column)]
                    if any(value is None for value in values):
                        continue
                    _expand_bounds(bounds, float(values[0]), float(values[1]), float(values[2]))
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            _warning(warnings, "skipped_file", f"Could not scan CSV bounds: {exc}", path)
    return _finalize_bounds(bounds)


def _ply_file_bounds(path: Path) -> dict[str, float] | None:
    header = _parse_ply_header(path)
    props: list[tuple[str, str]] = header["vertex_properties"]
    names = [name for name, _type_name in props]
    x_idx = names.index("x")
    y_idx = names.index("y")
    z_idx = names.index("z")
    bounds = _empty_bounds()
    for row in _read_ply_vertex_rows(path, header):
        values = [_parse_numeric(row[x_idx]), _parse_numeric(row[y_idx]), _parse_numeric(row[z_idx])]
        if any(value is None for value in values):
            continue
        _expand_bounds(bounds, float(values[0]), float(values[1]), float(values[2]))
    return _finalize_bounds(bounds)


def _ply_sequence_bounds(ply_input: str, worker_count: int, warnings: list[dict[str, Any]]) -> dict[str, float] | None:
    source = safe_path(ply_input)
    if source.is_file():
        frames = [(_ply_frame_number(source) or 0, source)]
    elif source.is_dir():
        frames = list_indexed_ply_frames(source)
    else:
        raise ValueError("PLY input path must be a file or folder.")
    paths = [path for _frame_number, path in frames]
    workers = max(1, int(worker_count or 1))
    try:
        if workers == 1 or len(paths) <= 1:
            bounds_list = [_ply_file_bounds(path) for path in paths]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                bounds_list = list(executor.map(_ply_file_bounds, paths))
    except ValueError:
        raise
    except OSError as exc:
        _warning(warnings, "skipped_file", f"Could not scan PLY bounds: {exc}")
        return None
    merged = _empty_bounds()
    for item in bounds_list:
        if item:
            merged = _merge_bounds(merged, item)
    return _finalize_bounds(merged)


def _transform_ply_frame(args: tuple[int, Path, Path, dict[str, Any], bool]) -> dict[str, Any]:
    frame_number, source_path, output_path, transform, voxelized = args
    header = _parse_ply_header(source_path)
    props: list[tuple[str, str]] = header["vertex_properties"]
    names = [name for name, _type_name in props]
    x_idx = names.index("x")
    y_idx = names.index("y")
    z_idx = names.index("z")
    rows = _read_ply_vertex_rows(source_path, header)
    transformed_rows: list[list[Any]] = []
    skipped = 0
    for row in rows:
        x_value = _parse_numeric(row[x_idx])
        y_value = _parse_numeric(row[y_idx])
        z_value = _parse_numeric(row[z_idx])
        if x_value is None or y_value is None or z_value is None:
            skipped += 1
            continue
        next_row = list(row)
        out_values = _apply_dataset_transform_values(x_value, y_value, z_value, transform, voxelized=voxelized)
        next_row[x_idx], next_row[y_idx], next_row[z_idx] = out_values
        transformed_rows.append(next_row)
    comments = [
        "transformed_by Prism Dataset Transformation Tool",
        f"coordinate_output {'voxelized' if voxelized else _operation_label(bool(transform['axis_swap']['enabled']), bool(transform['positive_coordinates']))}",
    ]
    _write_ascii_ply(output_path, props, transformed_rows, comments)
    return {
        "source": path_for_client(source_path),
        "output": path_for_client(output_path),
        "frame_number": frame_number,
        "source_filename": source_path.name,
        "output_filename": output_path.name,
        "points_written": len(transformed_rows),
        "points_skipped": skipped,
    }


def _transform_ply_sequence(
    ply_input: str,
    output_dir: Path,
    transform: dict[str, Any],
    warnings: list[dict[str, Any]],
    worker_count: int,
    voxelized: bool = False,
) -> dict[str, Any]:
    source = safe_path(ply_input)
    if source.is_file():
        frames = [(_ply_frame_number(source) or 0, source)]
    elif source.is_dir():
        frames = list_indexed_ply_frames(source)
    else:
        raise ValueError("PLY input path must be a file or folder.")
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(frame_number, path, output_dir / path.name, transform, voxelized) for frame_number, path in frames]
    workers = max(1, int(worker_count or 1))
    if workers == 1 or len(tasks) <= 1:
        output_files = [_transform_ply_frame(task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            output_files = list(executor.map(_transform_ply_frame, tasks))
    output_files.sort(key=lambda item: int(item["frame_number"]))
    points_written = sum(int(item["points_written"]) for item in output_files)
    points_skipped = sum(int(item["points_skipped"]) for item in output_files)
    if points_skipped:
        _warning(warnings, "skipped_points", f"Skipped {points_skipped} PLY vertices with non-numeric x/y/z values.")
    return {
        "input": path_for_client(source),
        "output_folder": path_for_client(output_dir),
        "summary": {
            "frames_found": len(frames),
            "frames_processed": len(output_files),
            "points_written": points_written,
            "points_skipped": points_skipped,
            "worker_count": workers,
        },
        "output_files": output_files,
    }


def transform_dataset(
    trace_root_folder: str = "",
    ply_input_path: str = "",
    output_folder: str = "",
    x_column: str = "",
    y_column: str = "",
    z_column: str = "",
    reference_ply_path: str = "",
    height_axis: str = "y",
    real_height_m: float = 1.8,
    output_height_axis: str | None = None,
    transform_traces: bool = True,
    transform_point_clouds: bool = False,
    height_axis_starts_at_zero: bool = True,
    positive_coordinates: bool = False,
    voxel_bits: int | None = None,
    export_mode: str = "scaled",
    worker_count: int = 1,
    fov_resample_enabled: bool = False,
    fov_resample_target: int | None = None,
) -> dict[str, Any]:
    if height_axis not in {"x", "y", "z"}:
        raise ValueError("Height axis must be x, y, or z.")
    output_height_axis = output_height_axis or height_axis
    if output_height_axis not in {"x", "y", "z"}:
        raise ValueError("Output height axis must be x, y, or z.")
    if real_height_m <= 0 or not math.isfinite(real_height_m):
        raise ValueError("Real-world height must be a positive number of meters.")
    if not transform_traces and not transform_point_clouds:
        raise ValueError("Select FoV traces, point clouds, or both for transformation.")
    if transform_traces and (not x_column or not y_column or not z_column):
        raise ValueError("X, Y, and Z trace columns are required when transforming traces.")
    export_mode = str(export_mode or "scaled").strip().lower()
    if export_mode not in SUPPORTED_DATASET_EXPORTS:
        raise ValueError("Export mode must be scaled, voxelized, or both.")
    if export_mode == "normalized":
        export_mode = "scaled"
    if export_mode in {"voxelized", "both"}:
        if voxel_bits is None:
            raise ValueError("Voxel bit depth is required for voxelized export.")
        voxel_bits = int(voxel_bits)
        if voxel_bits <= 0:
            raise ValueError("Voxel bit depth must be greater than zero.")
    else:
        voxel_bits = None

    reference_source = reference_ply_path or ply_input_path
    if not reference_source:
        raise ValueError("A reference PLY file or PLY sequence folder is required.")
    warnings: list[dict[str, Any]] = []
    ply_summary = inspect_ply_point_cloud(reference_source)
    resampling = _resolve_fov_resampling_plan(
        enabled=bool(fov_resample_enabled and transform_traces),
        requested_target=fov_resample_target,
        reference_ply_path=reference_source,
        ply_input_path=ply_input_path,
        transform_point_clouds=transform_point_clouds,
        warnings=warnings,
    )
    transform_bounds = dict(ply_summary["bounds"])
    zero_shift_bounds = dict(ply_summary["bounds"])
    if transform_point_clouds:
        ply_bounds = _ply_sequence_bounds(ply_input_path, worker_count, warnings)
        if ply_bounds:
            zero_shift_bounds = _merge_bounds(zero_shift_bounds, ply_bounds)
    if positive_coordinates or export_mode in {"voxelized", "both"}:
        if transform_traces:
            trace_bounds = _csv_coordinate_bounds(safe_path(trace_root_folder), x_column, y_column, z_column, warnings)
            if trace_bounds:
                transform_bounds = _merge_bounds(transform_bounds, trace_bounds)
        if transform_point_clouds:
            ply_bounds = zero_shift_bounds
            if ply_bounds:
                transform_bounds = _merge_bounds(transform_bounds, ply_bounds)
    transform = _build_dataset_transform(
        ply_summary,
        height_axis=height_axis,
        output_height_axis=output_height_axis,
        real_height_m=real_height_m,
        height_axis_starts_at_zero=height_axis_starts_at_zero,
        positive_coordinates=positive_coordinates,
        voxel_bits=voxel_bits,
        bounds_override=transform_bounds,
        zero_shift_bounds_override=zero_shift_bounds,
    )
    scale_factor = float(transform["scale_factor_m_per_unit"])
    if scale_factor < 0.001 or scale_factor > 1000:
        _warning(warnings, "scaling", "Scale factor is extreme; verify the selected height axis, PLY units, and real-world height.")
    elif scale_factor < 0.01 or scale_factor > 100:
        _warning(warnings, "scaling", "Scale factor is suspiciously large or small; verify the PLY reference scale.")

    output_dir = safe_path(output_folder)
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("Output path must be a directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    scaled_requested = export_mode in {"scaled", "both"}
    voxelized_requested = export_mode in {"voxelized", "both"}
    traces: dict[str, Any] | None = None
    traces_voxelized: dict[str, Any] | None = None
    point_clouds: dict[str, Any] | None = None
    point_clouds_voxelized: dict[str, Any] | None = None

    if transform_traces:
        root = safe_path(trace_root_folder)
        if scaled_requested:
            trace_output = output_dir if not transform_point_clouds and export_mode == "scaled" else output_dir / "traces_scaled"
            traces = _transform_csv_folder(root, trace_output, x_column, y_column, z_column, transform, warnings, voxelized=False, resampling=resampling)
        if voxelized_requested:
            traces_voxelized = _transform_csv_folder(root, output_dir / "traces_voxelized", x_column, y_column, z_column, transform, warnings, voxelized=True, resampling=resampling)
    if transform_point_clouds:
        if not ply_input_path:
            raise ValueError("PLY input path is required when transforming point clouds.")
        if scaled_requested:
            point_clouds = _transform_ply_sequence(ply_input_path, output_dir / "ply_scaled", transform, warnings, worker_count, voxelized=False)
        if voxelized_requested:
            point_clouds_voxelized = _transform_ply_sequence(ply_input_path, output_dir / "ply_voxelized", transform, warnings, worker_count, voxelized=True)

    return {
        "tool": "dataset_transformation",
        "output_folder": path_for_client(output_dir),
        "selected_columns": {"x": x_column, "y": y_column, "z": z_column},
        "scaling": {
            "point_cloud": ply_summary,
            "height_axis": height_axis,
            "measured_extent_units": float(transform["measured_extent_units"]),
            "real_height_m": real_height_m,
            "scale_factor_m_per_unit": scale_factor,
            "axis_swap": transform["axis_swap"],
            "height_axis_starts_at_zero": transform["height_axis_starts_at_zero"],
            "height_axis_zero_shift_axis": transform["height_axis_zero_shift_axis"],
            "height_axis_zero_shift_offset_m": transform["height_axis_zero_shift_offset_m"],
            "positive_coordinates": transform["positive_coordinates"],
            "positive_translation_m": transform["positive_translation_m"],
            "scaled_reference_bounds_m": transform["scaled_reference_bounds_m"],
            "height_axis_zero_shifted_bounds_m": transform["height_axis_zero_shifted_bounds_m"],
            "shifted_reference_bounds_m": transform["shifted_reference_bounds_m"],
            "final_exported_bounds_m": transform["final_exported_bounds_m"],
            "voxelization": {
                "enabled": voxel_bits is not None,
                "geometry_bit_depth": voxel_bits,
                "max_coordinate": transform["voxel_max"],
                "scale": transform["voxel_scale"],
            },
            "debug_details": transform["debug_details"],
        },
        "export_mode": export_mode,
        "trace_resampling": {key: value for key, value in resampling.items() if key != "frame_numbers"},
        "traces": traces,
        "traces_voxelized": traces_voxelized,
        "point_clouds": point_clouds,
        "point_clouds_voxelized": point_clouds_voxelized,
        "summary": {
            "trace_outputs": (traces or traces_voxelized or {}).get("summary", {}).get("output_files", 0),
            "point_cloud_outputs": (point_clouds or point_clouds_voxelized or {}).get("summary", {}).get("frames_processed", 0),
            "worker_count": max(1, int(worker_count or 1)),
        },
        "warnings": warnings,
    }


def _suspicious_jump_indexes(distances: list[float]) -> set[int]:
    if len(distances) < 4:
        return set()
    median_distance = statistics.median(distances)
    deviations = [abs(distance - median_distance) for distance in distances]
    mad = statistics.median(deviations)
    threshold = median_distance + (5 * mad)
    return {index for index, distance in enumerate(distances) if distance > threshold}


def analyze_speed_acceleration(
    root_folder: str,
    x_column: str,
    y_column: str,
    z_column: str,
    fps: float,
    ply_path: str,
    height_axis: str,
    real_height_m: float,
    filter_outliers: bool = True,
    filter_method: str = "iqr",
    percentile_keep: float = 99.0,
    histogram_source: str = "filtered",
    exclude_suspicious_jumps: bool = True,
) -> dict[str, Any]:
    if not x_column:
        raise ValueError("X position column is required.")
    if not y_column:
        raise ValueError("Y position column is required.")
    if not z_column:
        raise ValueError("Z position column is required.")
    if fps <= 0 or not math.isfinite(fps):
        raise ValueError("FPS must be a positive number.")
    if height_axis not in {"x", "y", "z"}:
        raise ValueError("Height axis must be x, y, or z.")
    if real_height_m <= 0 or not math.isfinite(real_height_m):
        raise ValueError("Real-world height must be a positive number of meters.")
    if filter_method not in SUPPORTED_FILTER_METHODS:
        raise ValueError("Filtering method must be 'iqr' or 'percentile'.")
    if histogram_source not in SUPPORTED_HISTOGRAM_SOURCES:
        raise ValueError("Histogram source must be 'raw' or 'filtered'.")

    warnings: list[dict[str, Any]] = []
    ply_summary = inspect_ply_point_cloud(ply_path)
    measured_extent = float(ply_summary["extents"][height_axis])
    if measured_extent <= 0 or not math.isfinite(measured_extent):
        raise ValueError("Selected height extent is zero or invalid; scaling cannot be computed.")
    if measured_extent < 1e-6:
        _warning(warnings, "scaling", "Selected height extent is unusually small; verify the point cloud units and height axis.")
    scale_factor = real_height_m / measured_extent
    root = safe_path(root_folder)
    files = _csv_files(root, warnings)
    rows_seen = 0
    valid_position_rows = 0
    invalid_position_rows = 0
    missing_position_rows = 0
    non_numeric_position_rows = 0
    files_read = 0
    skipped_files = 0
    dt = 1 / fps
    previous_position: tuple[float, float, float] | None = None
    motion_steps: list[dict[str, Any]] = []
    example_original: tuple[float, float, float] | None = None
    example_scaled: tuple[float, float, float] | None = None

    for path in files:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = reader.fieldnames or []
                if not _has_column(fieldnames, x_column) or not _has_column(fieldnames, y_column) or not _has_column(fieldnames, z_column):
                    skipped_files += 1
                    _warning(warnings, "skipped_file", "CSV does not contain all selected position columns.", path)
                    continue
                files_read += 1
                for row in reader:
                    rows_seen += 1
                    raw_values = [_row_value(row, x_column), _row_value(row, y_column), _row_value(row, z_column)]
                    if any(value is None or str(value).strip() == "" for value in raw_values):
                        invalid_position_rows += 1
                        missing_position_rows += 1
                        continue
                    position_values = [_parse_numeric(value) for value in raw_values]
                    if any(value is None for value in position_values):
                        invalid_position_rows += 1
                        non_numeric_position_rows += 1
                        continue
                    raw_position = (position_values[0], position_values[1], position_values[2])
                    position = (
                        raw_position[0] * scale_factor,
                        raw_position[1] * scale_factor,
                        raw_position[2] * scale_factor,
                    )
                    if example_original is None:
                        example_original = raw_position
                        example_scaled = position
                    valid_position_rows += 1

                    if previous_position is None:
                        previous_position = position
                        continue

                    dx = position[0] - previous_position[0]
                    dy = position[1] - previous_position[1]
                    dz = position[2] - previous_position[2]
                    distance = math.sqrt((dx * dx) + (dy * dy) + (dz * dz))
                    speed = distance / dt
                    velocity = (dx / dt, dy / dt, dz / dt)
                    motion_steps.append(
                        {
                            "distance": distance,
                            "speed": speed,
                            "velocity": velocity,
                        }
                    )

                    previous_position = position
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            skipped_files += 1
            _warning(warnings, "skipped_file", f"Could not process CSV: {exc}", path)

    suspicious_indexes = _suspicious_jump_indexes([step["distance"] for step in motion_steps])
    raw_speed_values = [step["speed"] for step in motion_steps]
    analysis_steps = [
        step
        for index, step in enumerate(motion_steps)
        if not (exclude_suspicious_jumps and index in suspicious_indexes)
    ]
    speed_values = [step["speed"] for step in analysis_steps]
    signed_acceleration_values: list[float] = []
    vector_acceleration_samples = 0
    for index in range(1, len(analysis_steps)):
        current = analysis_steps[index]
        previous = analysis_steps[index - 1]
        velocity = current["velocity"]
        previous_velocity = previous["velocity"]
        _acceleration = (
            (velocity[0] - previous_velocity[0]) / dt,
            (velocity[1] - previous_velocity[1]) / dt,
            (velocity[2] - previous_velocity[2]) / dt,
        )
        signed_acceleration_values.append((current["speed"] - previous["speed"]) / dt)
        vector_acceleration_samples += 1

    raw_signed_acceleration_values: list[float] = []
    for index in range(1, len(motion_steps)):
        raw_signed_acceleration_values.append((motion_steps[index]["speed"] - motion_steps[index - 1]["speed"]) / dt)

    if missing_position_rows:
        _warning(warnings, "skipped_rows", f"Ignored {missing_position_rows} rows with missing X, Y, or Z values.")
    if non_numeric_position_rows:
        _warning(
            warnings,
            "skipped_rows",
            f"Ignored {non_numeric_position_rows} rows with non-numeric X, Y, or Z values.",
        )
    if valid_position_rows < 2:
        _warning(warnings, "insufficient_data", "At least two valid position rows are required to calculate speed.")
    if valid_position_rows < 3:
        _warning(warnings, "insufficient_data", "At least three valid position rows are required to calculate acceleration.")
    if suspicious_indexes:
        _warning(
            warnings,
            "suspicious_jumps",
            f"Detected {len(suspicious_indexes)} suspicious position jumps using median distance + 5 x MAD.",
        )

    speed_metric = _metric_payload(speed_values, filter_outliers, filter_method, percentile_keep, histogram_source)
    acceleration_metric = _metric_payload(
        signed_acceleration_values,
        filter_outliers,
        filter_method,
        percentile_keep,
        histogram_source,
    )
    filtered_speed_values = speed_metric["filtered_histogram"] and _filter_values(speed_values, filter_method, percentile_keep)["values"] if filter_outliers else speed_values
    filtered_acceleration_values = (
        _filter_values(signed_acceleration_values, filter_method, percentile_keep)["values"] if filter_outliers else signed_acceleration_values
    )
    stationary_threshold = max(1e-9, (speed_metric["filtered_stats"]["median"] or 0) * 0.01)
    stationary_samples = sum(1 for value in filtered_speed_values if abs(value) <= stationary_threshold)
    accelerating_samples = sum(1 for value in filtered_acceleration_values if value > 1e-9)
    decelerating_samples = sum(1 for value in filtered_acceleration_values if value < -1e-9)
    recording_duration_s = max(valid_position_rows - 1, 0) * dt
    total_distance_m = sum(step["distance"] for step in analysis_steps)
    total_outliers_removed = (
        speed_metric["outlier_summary"]["outlier_samples"]
        + acceleration_metric["outlier_summary"]["outlier_samples"]
    )

    return {
        "root_folder": path_for_client(root),
        "selected_columns": {
            "x": x_column,
            "y": y_column,
            "z": z_column,
        },
        "timing": {
            "fps": fps,
            "sample_interval_s": dt,
        },
        "scaling": {
            "point_cloud": ply_summary,
            "height_axis": height_axis,
            "measured_extent_units": measured_extent,
            "real_height_m": real_height_m,
            "scale_factor_m_per_unit": scale_factor,
            "assumptions": [
                "Trace coordinates and reference point cloud share the same coordinate system.",
                "The scale factor derived from the selected height dimension is applied uniformly to X, Y, and Z.",
                "All motion calculations use scaled coordinates before distance, speed, and acceleration are derived.",
            ],
            "coordinate_example": {
                "original": list(example_original) if example_original is not None else None,
                "scaled_m": list(example_scaled) if example_scaled is not None else None,
            },
        },
        "filtering": {
            "filter_outliers": filter_outliers,
            "method": filter_method,
            "percentile_keep": percentile_keep,
            "histogram_source": histogram_source,
            "exclude_suspicious_jumps": exclude_suspicious_jumps,
        },
        "speed": speed_metric,
        "signed_acceleration": acceleration_metric,
        "quality": {
            "total_samples": len(raw_speed_values),
            "valid_samples": len(speed_values),
            "invalid_rows_skipped": invalid_position_rows,
            "suspicious_position_jumps": len(suspicious_indexes),
            "suspicious_position_jumps_excluded": len(suspicious_indexes) if exclude_suspicious_jumps else 0,
            "speed_outliers_detected": speed_metric["outlier_summary"]["outlier_samples"],
            "acceleration_outliers_detected": acceleration_metric["outlier_summary"]["outlier_samples"],
        },
        "motion_overview": {
            "recording_duration_s": recording_duration_s,
            "total_distance_m": total_distance_m,
            "average_speed_mps": speed_metric["filtered_stats"]["mean"],
            "median_speed_mps": speed_metric["filtered_stats"]["median"],
            "peak_speed_mps": speed_metric["filtered_stats"]["max"],
            "median_acceleration_mps2": acceleration_metric["filtered_stats"]["median"],
            "peak_acceleration_mps2": acceleration_metric["filtered_stats"]["max"],
            "time_stationary_s": stationary_samples * dt,
            "time_accelerating_s": accelerating_samples * dt,
            "time_decelerating_s": decelerating_samples * dt,
            "outliers_removed": total_outliers_removed,
        },
        "outlier_overview": {
            "speed_outliers_removed": speed_metric["outlier_summary"]["outlier_samples"],
            "acceleration_outliers_removed": acceleration_metric["outlier_summary"]["outlier_samples"],
            "largest_removed_speed_mps": speed_metric["outlier_summary"]["largest_removed"],
            "largest_removed_acceleration_mps2": acceleration_metric["outlier_summary"]["largest_removed"],
            "largest_retained_speed_mps": speed_metric["outlier_summary"]["largest_retained"],
            "largest_retained_acceleration_mps2": acceleration_metric["outlier_summary"]["largest_retained"],
        },
        "summary": {
            "files_discovered": len(files),
            "files_read": files_read,
            "files_skipped": skipped_files,
            "rows_seen": rows_seen,
            "valid_position_rows": valid_position_rows,
            "invalid_position_rows": invalid_position_rows,
            "missing_position_rows": missing_position_rows,
            "non_numeric_position_rows": non_numeric_position_rows,
            "speed_samples": len(speed_values),
            "raw_speed_samples": len(raw_speed_values),
            "velocity_samples": len(speed_values),
            "signed_acceleration_samples": len(signed_acceleration_values),
            "raw_signed_acceleration_samples": len(raw_signed_acceleration_values),
            "acceleration_samples": vector_acceleration_samples,
        },
        "warnings": warnings,
    }
