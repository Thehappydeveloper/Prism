#!/usr/bin/env python3
"""Render reproducible FoV score maps for the five-run visual comparison."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_selection import selected_run_ids  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from server_app.fov_comparison import (  # noqa: E402
    _draw_text, _fill_rect, _panel_bounds, _project, _project_point, _set_pixel, _stroke_rect,
    build_fov_comparison_dataset,
)

GROUP_NAME = "RankBudget_118Mbps_VisualComparison"
SEGMENTS = (12, 14)
CONDITIONS = (
    ("LastObservedFovPredictor", "__default__", "Last observed", "last_observed"),
    ("LinearExtrapolationPredictor", "linear_2s", "Linear extrapolation", "linear"),
    ("TrainedFovPredictor", "lstm_10", "Trained LSTM", "lstm"),
    ("TrainedFovPredictor", "mlp_10", "Trained MLP", "mlp"),
    ("ReachabilityRaycastPredictor", "raycast_fixed_origins_500_samples_6_6n_0.6_processes_8_paper_vis", "Argus", "argus"),
)
CORE = {
    "abr": "RankBudgetABR", "abr_preset_name": "perfect_bandwidth",
    "bandwidth_predictor": "GroundTruthBandwidthPredictor", "network_profile": "paper_wifi_118",
}


def load_runs() -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    allowed_ids = selected_run_ids()
    for path in (PROJECT_ROOT / "results").glob("run_*/config.json"):
        try:
            cfg = json.loads(path.read_text(encoding="utf-8"))
            status = json.loads((path.parent / "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if cfg.get("run_group_name") != GROUP_NAME or status.get("status") != "finished":
            continue
        if allowed_ids is not None and path.parent.name not in allowed_ids:
            continue
        if any(cfg.get(key) != expected for key, expected in CORE.items()):
            raise ValueError(f"{path.parent.name}: does not match the fixed visual-comparison protocol")
        runs[path.parent.name] = cfg
    if len(runs) != len(CONDITIONS):
        raise ValueError(f"Expected {len(CONDITIONS)} completed runs in {GROUP_NAME}; found {len(runs)}")
    return runs


def condition_runs(runs: dict[str, dict[str, Any]]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for predictor, preset, _label, slug in CONDITIONS:
        ids = [run_id for run_id, cfg in runs.items() if cfg.get("predictor") == predictor and (cfg.get("predictor_preset_name") or "__default__") == preset]
        if len(ids) != 1:
            raise ValueError(f"{slug}: expected one completed run, found {len(ids)}")
        selected[slug] = ids[0]
    return selected


SCORE_COLORS = (
    (219, 234, 254),  # light blue: no or very little predicted visibility
    (147, 197, 253),
    (59, 130, 246),
    (29, 78, 216),
    (11, 47, 107),    # deep navy: high predicted visibility
)


def score_color(value: Any) -> tuple[int, int, int]:
    """Use a restrained, colour-blind-friendly sequential paper palette."""
    score = max(0.0, min(1.0, float(value or 0.0)))
    scaled = score * (len(SCORE_COLORS) - 1)
    lower = min(int(scaled), len(SCORE_COLORS) - 2)
    fraction = scaled - lower
    start, end = SCORE_COLORS[lower], SCORE_COLORS[lower + 1]
    return tuple(round(start[channel] + (end[channel] - start[channel]) * fraction) for channel in range(3))


COMPACT_LABELS = {
    "GROUND TRUTH": "GT",
    "LAST OBSERVED": "LO",
    "LINEAR EXTRAPOLATION": "LE",
    "TRAINED LSTM": "LSTM",
    "TRAINED MLP": "MLP",
    "ARGUS": "ARGUS",
}


def _blend_pixel(buf: bytearray, width: int, height: int, x: int, y: int, color: tuple[int, int, int], alpha: float) -> None:
    if x < 0 or y < 0 or x >= width or y >= height:
        return
    index = (y * width + x) * 3
    for channel, target in enumerate(color):
        buf[index + channel] = round(buf[index + channel] * (1.0 - alpha) + target * alpha)


def _draw_density_cloud(buf: bytearray, width: int, height: int, cloud: dict[str, Any] | None, world: tuple[float, float, float, float], panel: tuple[int, int, int, int]) -> None:
    """Render geometry as a thin density layer instead of enlarged raw points."""
    # Projecting every point into a small screen bin both suppresses isolated
    # samples and preserves continuous walls.  The marker centre is the mean
    # projected location, so no artificial checkerboard is introduced.
    bins: dict[tuple[int, int], list[float]] = {}
    cell_size = 3
    for point in (cloud or {}).get("points") or []:
        projected = _project_point(point, world, panel)
        if projected is None:
            continue
        key = (projected[0] // cell_size, projected[1] // cell_size)
        values = bins.setdefault(key, [0.0, 0.0, 0.0])
        values[0] += 1.0
        values[1] += projected[0]
        values[2] += projected[1]
    for count, total_x, total_y in bins.values():
        x, y = round(total_x / count), round(total_y / count)
        # Log scaling makes dense surfaces coherent without allowing a few
        # very dense bins to become opaque.
        alpha = min(0.92, 0.40 + 0.32 * math.log2(count + 1.0))
        _blend_pixel(buf, width, height, x, y, (51, 65, 85), alpha)
        edge_alpha = alpha * 0.38
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            _blend_pixel(buf, width, height, x + dx, y + dy, (51, 65, 85), edge_alpha)


def _pose_projection(sample: dict[str, Any], world: tuple[float, float, float, float], panel: tuple[int, int, int, int]) -> tuple[int, int] | None:
    position = sample.get("position") if isinstance(sample, dict) else None
    if not isinstance(position, (list, tuple)) or len(position) < 3:
        return None
    return _project_point((position[0], position[2]), world, panel)


def _draw_line(buf: bytearray, width: int, height: int, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int], thickness: int = 2) -> None:
    dx, dy = end[0] - start[0], end[1] - start[1]
    steps = max(abs(dx), abs(dy), 1)
    radius = max(0, thickness - 1)
    for index in range(steps + 1):
        x = round(start[0] + dx * index / steps)
        y = round(start[1] + dy * index / steps)
        _fill_rect(buf, width, height, (x - radius, y - radius, x + radius + 1, y + radius + 1), color)


def _draw_motion_trace(buf: bytearray, width: int, height: int, frame: dict[str, Any], world: tuple[float, float, float, float], panel: tuple[int, int, int, int]) -> None:
    """Mirror the single-run debugger: observed/current in violet, target path in amber."""
    trace = frame.get("user_trace") or {}
    observed = [point for sample in trace.get("observed") or [] if (point := _pose_projection(sample, world, panel)) is not None]
    actual = [point for sample in trace.get("actual") or [] if (point := _pose_projection(sample, world, panel)) is not None]
    for path, color in ((observed, (126, 34, 206)), (actual, (245, 158, 11))):
        for start, end in zip(path, path[1:]):
            _draw_line(buf, width, height, start, end, color, thickness=3)
    if observed:
        _fill_rect(buf, width, height, (observed[0][0] - 2, observed[0][1] - 2, observed[0][0] + 3, observed[0][1] + 3), (126, 34, 206))
    for point in actual:
        _fill_rect(buf, width, height, (point[0] - 2, point[1] - 2, point[0] + 3, point[1] + 3), (245, 158, 11))
    current = _pose_projection(trace.get("current") or {}, world, panel)
    if current is not None:
        # A diamond distinguishes the prediction-time pose from the observed
        # path and from the amber target-segment samples.
        for dy in range(-6, 7):
            radius = 6 - abs(dy)
            _fill_rect(buf, width, height, (current[0] - radius, current[1] + dy, current[0] + radius + 1, current[1] + dy + 1), (255, 255, 255))
        for dy in range(-4, 5):
            radius = 4 - abs(dy)
            _fill_rect(buf, width, height, (current[0] - radius, current[1] + dy, current[0] + radius + 1, current[1] + dy + 1), (126, 34, 206))
        _fill_rect(buf, width, height, (current[0], current[1], current[0] + 1, current[1] + 1), (30, 41, 59))


def draw_score_panel(buf: bytearray, width: int, height: int, frame: dict[str, Any], scores: dict[str, Any], panel: tuple[int, int, int, int], cloud: dict[str, Any] | None, trace_overlay: bool) -> None:
    world = _panel_bounds(frame)
    _fill_rect(buf, width, height, panel, (250, 250, 250))
    boxes = []
    for tile_id in frame["tile_ids"]:
        box = _project(frame["tiles"][tile_id]["bounds_xz"], world, panel)
        _fill_rect(buf, width, height, box, score_color(scores.get(tile_id, 0.0)))
        boxes.append(box)
    _draw_density_cloud(buf, width, height, cloud, world, panel)
    # Tile boundaries belong above the geometry layer so the score grid stays
    # crisp even in dense parts of the point cloud.
    for box in boxes:
        _stroke_rect(buf, width, height, box, (255, 255, 255))
    if trace_overlay:
        _draw_motion_trace(buf, width, height, frame, world, panel)
    _stroke_rect(buf, width, height, panel, (71, 85, 105))
    _stroke_rect(buf, width, height, (panel[0] + 1, panel[1] + 1, panel[2] - 1, panel[3] - 1), (71, 85, 105))


def write_ppm(path: Path, panels: list[tuple[str, dict[str, Any], dict[str, Any]]], cloud: dict[str, Any] | None, trace_overlay: bool, *, grid: bool, compact: bool = False, segment_label: str = "", show_score_scale: bool = False, font_scale: float = 1.0) -> None:
    columns, rows = (3, 2) if grid else (2, 1)
    group_header = 0
    if grid and compact:
        # The all-segment figure is read as a vertical sequence.  Keep the
        # panels full-size, but remove redundant outer whitespace between
        # segment blocks and identify each block in its header strip.
        panel_w, panel_h, margin, heading, gap = (560, 420, 18, 50, 10)
        # The segment heading grows with the requested final paper font. Keep
        # a genuine header band so the score key never collides with it.
        group_header = max(96, round(54 + 40 * font_scale))
    else:
        panel_w, panel_h, margin, heading, gap = (560, 420, 38, 36, 24) if grid else (760, 570, 48, 42, 32)
    width = margin * 2 + columns * panel_w + (columns - 1) * gap
    height = margin * 2 + group_header + rows * (heading + panel_h) + (rows - 1) * gap
    buf = bytearray([255, 255, 255]) * (width * height)
    if compact:
        _fill_rect(buf, width, height, (4, 4, width - 4, height - 4), (248, 250, 252))
        _stroke_rect(buf, width, height, (4, 4, width - 4, height - 4), (51, 65, 85))
        _stroke_rect(buf, width, height, (5, 5, width - 5, height - 5), (51, 65, 85))
        if show_score_scale:
            scale_x, scale_y, cell_w = width - 430, 18, 46
            for index, color in enumerate(SCORE_COLORS):
                box = (scale_x + 120 + index * cell_w, scale_y, scale_x + 120 + (index + 1) * cell_w, scale_y + 26)
                _fill_rect(buf, width, height, box, color)
                _stroke_rect(buf, width, height, box, (51, 65, 85))
                _stroke_rect(buf, width, height, (box[0] + 1, box[1] + 1, box[2] - 1, box[3] - 1), (51, 65, 85))
    for index, (label, frame, scores) in enumerate(panels):
        col, row = index % columns, index // columns
        x = margin + col * (panel_w + gap)
        y = margin + group_header + row * (heading + panel_h + gap)
        if not compact:
            _draw_text(buf, width, height, x, y + 6, label, color=(30, 41, 59), scale=2)
        draw_score_panel(buf, width, height, frame, scores, (x, y + heading, x + panel_w, y + heading + panel_h), cloud, trace_overlay)
    with path.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii")); handle.write(buf)


def to_png(source: Path, output: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to render the FoV comparison PNG files")
    completed = subprocess.run([ffmpeg, "-y", "-v", "error", "-i", str(source), str(output)], text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "ffmpeg could not encode the FoV comparison PNG")


def add_compact_typography(output: Path, panels: list[tuple[str, dict[str, Any], dict[str, Any]]], segment_label: str, show_score_scale: bool, font_scale: float) -> None:
    """Use a normal sans-serif face for the paper-scale overview labels."""
    magick = shutil.which("magick") or shutil.which("convert")
    if not magick:
        raise RuntimeError("ImageMagick is required to render the compact figure typography")
    panel_w, panel_h, margin, heading, gap = 560, 420, 18, 50, 10
    group_header = max(96, round(54 + 40 * font_scale))
    width = margin * 2 + 3 * panel_w + 2 * gap
    command = [
        magick, str(output),
        "-font", "TeXGyreTermes-Bold", "-fill", "#1e293b", "-pointsize", str(round(40 * font_scale)),
        "-gravity", "North", "-annotate", "+0+34", f"Segment {segment_label}",
        "-gravity", "NorthWest", "-pointsize", str(round(28 * font_scale)),
    ]
    for index, (label, _frame, _scores) in enumerate(panels):
        col, row = index % 3, index // 3
        x = margin + col * (panel_w + gap)
        y = margin + group_header + row * (heading + panel_h + gap) + 13
        command.extend(["-annotate", f"+{x}+{y}", COMPACT_LABELS.get(label, label)])
    if show_score_scale:
        scale_x, scale_y, cell_w = width - 430, 18, 46
        # This is supporting annotation, not a panel heading. It stays
        # deliberately smaller than the requested panel-label font.
        score_font = round(max(18, 24 * font_scale))
        score_value_font = round(max(16, 20 * font_scale))
        command.extend([
            "-font", "TeXGyreTermes-Regular", "-fill", "#334155", "-pointsize", str(score_font),
            "-annotate", f"+{scale_x}+{scale_y + 3}", "Score",
            "-pointsize", str(score_value_font),
            "-annotate", f"+{scale_x + 120}+{scale_y + 38}", "0",
            "-annotate", f"+{scale_x + 120 + 5 * cell_w - 12}+{scale_y + 38}", "1",
        ])
    styled = output.with_name(f".{output.stem}.styled.png")
    completed = subprocess.run([*command, str(styled)], text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "ImageMagick could not render compact figure typography")
    styled.replace(output)


def render_png(output: Path, panels: list[tuple[str, dict[str, Any], dict[str, Any]]], cloud: dict[str, Any] | None, trace_overlay: bool, *, grid: bool, compact: bool = False, segment_label: str = "", show_score_scale: bool = False, font_scale: float = 1.0) -> None:
    with tempfile.TemporaryDirectory(prefix="prism-fov-report-") as directory:
        ppm = Path(directory) / "image.ppm"; write_ppm(ppm, panels, cloud, trace_overlay, grid=grid, compact=compact, segment_label=segment_label, show_score_scale=show_score_scale, font_scale=font_scale); to_png(ppm, output)
    if compact:
        add_compact_typography(output, panels, segment_label, show_score_scale, font_scale)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = condition_runs(load_runs())
    datasets = {slug: build_fov_comparison_dataset(run_id) for slug, run_id in runs.items()}
    overlay_point_cloud = os.environ.get("PRISM_REPORT_POINT_CLOUD_OVERLAY", "1").strip().lower() not in {"0", "false", "no"}
    overlay_motion_trace = os.environ.get("PRISM_REPORT_MOTION_TRACE_OVERLAY", "1").strip().lower() not in {"0", "false", "no"}
    font_scale = max(0.3, float(os.environ.get("PRISM_REPORT_FIGURE_FONT_SCALE", "1")))
    cloud = next((data.get("point_cloud_slice") for data in datasets.values() if data.get("point_cloud_slice")), None) if overlay_point_cloud else None
    frames = {
        slug: {int(frame["segment_idx"]): frame for frame in data["frames"]}
        for slug, data in datasets.items()
    }
    labelled = [(label, slug) for _predictor, _preset, label, slug in CONDITIONS]
    stack_grids: list[tuple[int, list[tuple[str, dict[str, Any], dict[str, Any]]]]] = []
    for segment in SEGMENTS:
        missing = [label for label, slug in labelled if segment not in frames[slug]]
        if missing:
            raise ValueError(f"Segment {segment} is absent for: {', '.join(missing)}")
        reference = frames[labelled[0][1]][segment]
        truth = ("GROUND TRUTH", reference, reference["actual"])
        grid_panels = [truth] + [(label.upper(), frames[slug][segment], frames[slug][segment]["predicted"]) for label, slug in labelled]
        stack_grids.append((segment, grid_panels))
    # One side-by-side figure keeps the S12 and S14 six-panel grids at the
    # same double-column scale as the previous overview.
    with tempfile.TemporaryDirectory(prefix="prism-fov-report-stack-") as directory:
        # Compact renderings remove the duplicated lower/upper margins of
        # adjacent grids while retaining the full-resolution panel content.
        output = args.output_dir / "segments_12_14_all_predictors.png"
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required to stack the FoV comparison PNG files")
        compact_grids = []
        for segment, panels in stack_grids:
            image = Path(directory) / f"segment_{segment:02d}.png"
            render_png(image, panels, cloud, overlay_motion_trace, grid=True, compact=True, segment_label=f"S{segment:02d}", show_score_scale=segment == SEGMENTS[0], font_scale=font_scale)
            compact_grids.append(image)
        inputs = [part for image in compact_grids for part in ("-i", str(image))]
        graph = "[0:v][1:v]hstack=inputs=2[out]"
        completed = subprocess.run([ffmpeg, "-y", "-v", "error", *inputs, "-filter_complex", graph, "-map", "[out]", str(output)], text=True, capture_output=True, check=False)
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or "ffmpeg could not arrange the FoV comparison figures")
    print(f"Rendered the S12/S14 1×2 six-panel grid ({'with' if overlay_point_cloud else 'without'} point cloud; {'with' if overlay_motion_trace else 'without'} motion traces).")


if __name__ == "__main__":
    main()
