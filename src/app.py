from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

from abr import baselines as abr_baselines
from bandwith_predictor import baselines as bandwidth_baselines
from core.event_logger import EventLogger
from core.run_lifecycle import (
    atomic_write_json,
    mark_run_complete,
    prepare_incomplete_dir,
    promote_incomplete_dir,
)
from core.run_index import upsert_run_directory
from core.run_manifest import write_run_manifest
from core.fov import Fov
from core.network import Network
from core.network_profiles import get_network_profile
from core.shared_content import SegmentContentLoader
from core.system_resources import auto_segment_worker_count
from core.tile_fov import (
    DEFAULT_GROUND_TRUTH_MODE,
    DEFAULT_PEAK_WEIGHT,
    DEFAULT_VISIBLE_THRESHOLD,
    TileFovTrace,
    has_tile_fov_columns,
)
from core.timing import TimingRecorder, timing_stage
from core.video import Video
from fov_predictors import baselines as fov_baselines
from fov_predictors.latency import FovPredictorLatencyRecorder
from player.player import Player


class PlaybackGroupCanceled(Exception):
    """Raised when a grouped playback run is cooperatively canceled."""


@dataclass
class PlaybackConfig:
    mpd: Path
    content_dir: Path
    fov_trace: Path
    network_trace: Optional[Path] = None
    network_profile: Optional[str] = None
    dataset_id: Optional[str] = None
    abr: str = "BandwidthAwareABR"
    predictor: Optional[str] = None
    bandwidth_predictor: Optional[str] = None
    abr_params: Dict[str, Any] | None = None
    predictor_params: Dict[str, Any] | None = None
    bandwidth_predictor_params: Dict[str, Any] | None = None
    abr_preset_name: Optional[str] = None
    predictor_preset_name: Optional[str] = None
    bandwidth_predictor_preset_name: Optional[str] = None
    headset_preset: Optional[str] = None
    fov_h_deg: Optional[float] = None
    fov_v_deg: Optional[float] = None
    angle_convention: Optional[str] = None
    rtt: float = 0.06
    block_size_bits: int = 250_000  # small default block size
    enable_fov_upgrades: bool = False
    enable_mid_download_cancel: bool = False
    buffer_window_segments: int = 2
    buffer_unbounded: bool = False
    # Zero processes the full video; a positive value stops after that many
    # segments, beginning with segment 0.
    segment_limit: int = 0
    show_progress: bool = True
    output_dir: Path = Path("results")
    run_id: Optional[str] = None
    record_fov_stage_latency: bool = False
    ground_truth_mode: str = DEFAULT_GROUND_TRUTH_MODE
    ground_truth_visible_threshold: float = DEFAULT_VISIBLE_THRESHOLD
    ground_truth_peak_weight: float = DEFAULT_PEAK_WEIGHT
    motion_limits: Dict[str, float] | None = None
    oracle_mode: str = "none"
    oracle_group_id: Optional[str] = None
    baseline_run_id: Optional[str] = None
    run_group_id: Optional[str] = None
    run_group_name: Optional[str] = None
    run_group_source: Optional[str] = None
    experiment_id: Optional[str] = None
    experiment_name: Optional[str] = None
    experiment_purpose: Optional[str] = None
    experiment_kind: Optional[str] = None
    experiment_primary_metrics: list[str] | None = None
    # Analysis provenance. QoE is computed from completed artifacts rather
    # than during playback, but a batch records the intended paper metric.
    analysis_qoe_function: Optional[str] = None
    analysis_qoe_scoring_version: Optional[str] = None
    # Evaluation-only prefix excluded when a predictor needs history before
    # its output is meaningful. Playback itself is unaffected.
    analysis_qoe_warmup_segments: int = 0
    # ``None`` uses the server setting.  Zero explicitly requests automatic
    # selection; a positive value fixes the worker count for this experiment.
    segment_workers_override: Optional[int] = None


def generate_run_id() -> str:
    now = datetime.datetime.now()
    suffix = uuid4().hex[:4]
    return f"run_{now.strftime('%y%m%d_%H%M%S')}_{suffix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CLI runner for the VPCC player.")
    parser.add_argument("--mpd", type=Path, required=True, help="Path to the V3C MPD manifest.")
    parser.add_argument("--content-dir", type=Path, required=True, help="Directory with original PLY frames.")
    network_source = parser.add_mutually_exclusive_group(required=True)
    network_source.add_argument("--network-trace", type=Path, help="CSV network trace path.")
    network_source.add_argument("--network-profile", help="Standardized network profile id.")
    parser.add_argument("--fov-trace", type=Path, required=True, help="CSV FoV trace path.")
    parser.add_argument("--abr", default="BandwidthAwareABR", help="ABR class name from abr.baselines.")
    parser.add_argument("--predictor", default=None, help="FoV predictor class name from fov_predictors.baselines.")
    parser.add_argument("--bandwidth-predictor", default=None, help="Bandwidth predictor class name from bandwith_predictor.baselines.")
    parser.add_argument("--rtt", type=float, default=0.06, help="Network RTT in seconds.")
    parser.add_argument("--block-size-bits", type=int, default=1_000_000, help="Network block size in bits.")
    parser.add_argument("--enable-fov-upgrades", action="store_true", help="Enable FoV-centric upgrade planning.")
    parser.add_argument(
        "--enable-mid-download-cancel",
        action="store_true",
        help="Enable mid-download cancellation/downswitch when bandwidth drops.",
    )
    parser.add_argument(
        "--buffer-window-segments",
        type=int,
        default=2,
        help="Max segments to buffer ahead before waiting for playback to drain (0 = fetch to end).",
    )
    parser.add_argument(
        "--buffer-unbounded",
        action="store_true",
        help="Disable the buffer limit and fetch until the end of the video.",
    )
    parser.add_argument(
        "--segment-limit",
        type=int,
        default=0,
        help="Process only the first N segments (0 = the complete video).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Base directory to store run artifacts (events, logs).")
    parser.add_argument("--run-id", default=None, help="Optional run id; defaults to timestamp.")
    parser.add_argument("--ground-truth-mode", default=DEFAULT_GROUND_TRUTH_MODE, help="Segment target reducer for per-frame tile FoV ground truth.")
    parser.add_argument("--ground-truth-visible-threshold", type=float, default=DEFAULT_VISIBLE_THRESHOLD, help="Visibility threshold for visible-fraction ground truth.")
    parser.add_argument("--ground-truth-peak-weight", type=float, default=DEFAULT_PEAK_WEIGHT, help="Max component weight for blend/peak-weighted ground truth.")
    return parser.parse_args()


def instantiate_abr(name: str, predictor_name: Optional[str] = None, abr_params: Dict[str, Any] | None = None):
    abr_cls = getattr(abr_baselines, name, None)
    if not abr_cls:
        raise ValueError(f"Unknown ABR class '{name}'.")
    has_predictor = bool(predictor_name and predictor_name != "None")
    if getattr(abr_cls, "abr_requires_predictor", False) and not has_predictor:
        raise ValueError(f"ABR '{name}' requires a FoV predictor; please select one.")
    if has_predictor and not getattr(abr_cls, "abr_supports_predictor", False):
        raise ValueError(f"ABR '{name}' does not support FoV predictors; select 'None'.")
    kwargs = dict(abr_params or {})
    try:
        return abr_cls(**kwargs)
    except TypeError as exc:
        raise ValueError(f"Invalid ABR parameters for '{name}': {exc}") from exc


def instantiate_predictor(
    name: str | None,
    predictor_params: Dict[str, Any] | None = None,
    fov_h_deg: float | None = None,
    fov_v_deg: float | None = None,
    angle_convention: str | None = None,
):
    if not name or name == "None":
        return None
    predictor_cls = getattr(fov_baselines, name, None)
    if not predictor_cls:
        raise ValueError(f"Unknown FoV predictor class '{name}'.")
    kwargs = dict(predictor_params or {})
    if fov_h_deg is not None and "fov_h" not in kwargs:
        kwargs["fov_h"] = math.radians(float(fov_h_deg))
    if fov_v_deg is not None and "fov_v" not in kwargs:
        kwargs["fov_v"] = math.radians(float(fov_v_deg))
    if angle_convention and "angle_convention" not in kwargs:
        kwargs["angle_convention"] = angle_convention
    try:
        return predictor_cls(**kwargs)
    except TypeError as exc:
        raise ValueError(f"Invalid FoV predictor parameters for '{name}': {exc}") from exc


def instantiate_bandwidth_predictor(
    name: str | None,
    predictor_params: Dict[str, Any] | None = None,
    preset_name: str | None = None,
):
    if not name or name == "None":
        return None
    predictor_cls = getattr(bandwidth_baselines, name, None)
    if not predictor_cls:
        raise ValueError(f"Unknown bandwidth predictor class '{name}'.")
    kwargs = dict(predictor_params or {})
    try:
        predictor = predictor_cls(**kwargs)
    except TypeError as exc:
        raise ValueError(f"Invalid bandwidth predictor parameters for '{name}': {exc}") from exc
    predictor.profile_name = preset_name or None
    return predictor


def _network_seed(config: PlaybackConfig) -> int:
    raw = json.dumps(
        {
            "run_id": config.run_id,
            "network_profile": config.network_profile,
            "network_trace": str(config.network_trace or ""),
        },
        sort_keys=True,
    )
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], 16)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def save_config(run_dir: Path, config: PlaybackConfig):
    """Write the effective playback configuration to run_dir/config.json."""
    atomic_write_json(run_dir / "config.json", asdict(config))


def _predictor_needs_ply(config: PlaybackConfig) -> bool:
    name = str(config.predictor or "None")
    return bool(name and name not in {"None", "GroundTruthTileFovPredictor"})


def _build_player_runtime(
    config: PlaybackConfig,
    run_dir: Path,
    timing: TimingRecorder,
    video: Video | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> tuple[Player, EventLogger, FovPredictorLatencyRecorder]:
    with timing_stage("instantiate_bandwidth_predictor", run_id=config.run_id):
        bandwidth_predictor = instantiate_bandwidth_predictor(
            config.bandwidth_predictor,
            config.bandwidth_predictor_params,
            config.bandwidth_predictor_preset_name,
        )
    profile = get_network_profile(config.network_profile)
    if config.network_profile and profile is None:
        raise ValueError(f"Unknown network profile '{config.network_profile}'.")
    with timing_stage("network.init", run_id=config.run_id, path=config.network_trace or config.network_profile):
        network = Network(
            config.network_trace,
            rtt=config.rtt,
            block_size_bits=config.block_size_bits,
            predictor=bandwidth_predictor,
            profile=profile,
            seed=_network_seed(config),
        )
    with timing_stage("instantiate_fov_predictor", run_id=config.run_id):
        predictor = instantiate_predictor(
            config.predictor,
            config.predictor_params,
            fov_h_deg=config.fov_h_deg,
            fov_v_deg=config.fov_v_deg,
            angle_convention=config.angle_convention,
        )
    if video is None:
        with timing_stage("video.init", run_id=config.run_id, mpd=config.mpd, content_dir=config.content_dir):
            video = Video(
                manifest_file=config.mpd,
                original_content_dir=config.content_dir,
                media_root=None,
            )
    if config.buffer_unbounded or config.buffer_window_segments == 0:
        buffer_window = video.num_segments()
    else:
        buffer_window = int(config.buffer_window_segments or 0)
        if buffer_window < 1:
            raise ValueError("Buffer window must be 0 (unbounded) or at least 1 segment.")
        if buffer_window > video.num_segments():
            buffer_window = video.num_segments()
    fov_latency_recorder = FovPredictorLatencyRecorder(
        enabled=bool(config.record_fov_stage_latency and predictor is not None),
        predictor_name=str(config.predictor or ""),
    )
    with timing_stage("fov.init", run_id=config.run_id, path=config.fov_trace):
        fov = Fov(
            csv_path=config.fov_trace,
            predictor=predictor,
            start_frame_index=video.start_frame_idx,
            latency_recorder=fov_latency_recorder,
            content_dir=config.content_dir,
            motion_limits=config.motion_limits,
            frame_rate_hz=video.frame_rate_hz,
        )
    with timing_stage("tile_fov.init", run_id=config.run_id, path=config.fov_trace):
        tile_fov = TileFovTrace(config.fov_trace, require_tile_columns=False) if has_tile_fov_columns(config.fov_trace) else None
    with timing_stage("abr.init", run_id=config.run_id):
        abr = instantiate_abr(config.abr, config.predictor, config.abr_params)

    event_logger = EventLogger()
    event_logger.ground_truth_mode = str(config.ground_truth_mode or DEFAULT_GROUND_TRUTH_MODE)
    event_logger.ground_truth_params = {
        "visible_threshold": float(config.ground_truth_visible_threshold),
        "peak_weight": float(config.ground_truth_peak_weight),
    }
    player = Player(
        abr=abr,
        network=network,
        fov=fov,
        video=video,
        tile_fov=tile_fov,
        event_logger=event_logger,
        show_progress=config.show_progress,
        run_dir=run_dir,
        run_id=config.run_id,
        enable_fov_upgrades=config.enable_fov_upgrades,
        enable_mid_download_cancel=config.enable_mid_download_cancel,
        max_buffer_segments=buffer_window,
        segment_limit=config.segment_limit,
        timing_recorder=timing,
        progress_callback=(
            (lambda current, total: progress_callback(str(config.run_id), current, total))
            if progress_callback is not None
            else None
        ),
        ground_truth_mode=config.ground_truth_mode,
        ground_truth_visible_threshold=config.ground_truth_visible_threshold,
        ground_truth_peak_weight=config.ground_truth_peak_weight,
    )
    return player, event_logger, fov_latency_recorder


def _save_runtime_outputs(run_dir: Path, event_logger: EventLogger, fov_latency_recorder: FovPredictorLatencyRecorder) -> None:
    event_logger.save(run_dir / "events.json.gz")
    if fov_latency_recorder.enabled and fov_latency_recorder.has_data():
        (run_dir / "predictor_diagnostics.json").write_text(
            json.dumps(fov_latency_recorder.summary(), indent=2),
            encoding="utf-8",
        )


def run_playback(config: PlaybackConfig, progress_callback: Callable[[str, int, int], None] | None = None) -> Path:
    run_id = config.run_id or generate_run_id()
    run_dir = config.output_dir.resolve() / run_id
    config.run_id = run_id
    staging_dir = prepare_incomplete_dir(run_dir)
    timing = TimingRecorder(run_id)
    try:
        with timing.activate():
            with timing_stage("config.save", run_id=run_id):
                save_config(run_dir, config)
            player, event_logger, fov_latency_recorder = _build_player_runtime(
                config,
                staging_dir,
                timing,
                progress_callback=progress_callback,
            )
            with timing_stage("player.play", run_id=run_id):
                player.play()
            with timing_stage("outputs.save", run_id=run_id):
                _save_runtime_outputs(staging_dir, event_logger, fov_latency_recorder)
    finally:
        timing.save(staging_dir / "timing.json")

    # Runtime artifacts remain invisible until all of them have been saved.
    promote_incomplete_dir(run_dir)
    write_run_manifest(run_id, config, player.video, run_dir)
    mark_run_complete(run_dir, run_id)
    upsert_run_directory(run_dir.parent, run_dir)

    return run_dir


def run_playback_group(
    configs: list[PlaybackConfig],
    ply_io_workers: int = 1,
    segment_workers: int = 1,
    segment_workers_auto_cap: int = 4,
    prefetch_segments: int = 0,
    progress_callback: Callable[[str, int, int], None] | None = None,
    progress_detail_callback: Callable[[str, Dict[str, Any]], None] | None = None,
    control_callback: Callable[[], None] | None = None,
) -> list[Path]:
    """Run compatible configs in segment lockstep with shared current-window PLY IO."""
    if not configs:
        return []
    first = configs[0]
    for config in configs:
        if Path(config.mpd).resolve() != Path(first.mpd).resolve() or Path(config.content_dir).resolve() != Path(first.content_dir).resolve():
            raise ValueError("run_playback_group requires configs with the same MPD and content directory.")

    batch_name = f"batch_{first.run_id or generate_run_id()}"
    batch_timing = TimingRecorder(batch_name)
    run_dirs: list[Path] = []
    runtimes = []
    try:
        with batch_timing.activate():
            with timing_stage("video.init.shared", mpd=first.mpd, content_dir=first.content_dir):
                shared_video = Video(
                    manifest_file=first.mpd,
                    original_content_dir=first.content_dir,
                    media_root=None,
            )
            for config in configs:
                run_id = config.run_id or generate_run_id()
                config.run_id = run_id
                run_dir = config.output_dir.resolve() / run_id
                run_dirs.append(run_dir)
                staging_dir = prepare_incomplete_dir(run_dir)
                if progress_detail_callback is not None:
                    progress_detail_callback(
                        str(run_id),
                        {
                            "phase": "preparing_runtime",
                            "dataset_id": str(config.dataset_id or ""),
                            "predictor": str(config.predictor or "None"),
                            "network": str(config.network_profile or config.network_trace or ""),
                        },
                    )
                timing = TimingRecorder(run_id)
                with timing.activate():
                    with timing_stage("config.save", run_id=run_id):
                        save_config(run_dir, config)
                    player, event_logger, fov_latency_recorder = _build_player_runtime(
                        config,
                        staging_dir,
                        timing,
                        video=shared_video,
                        progress_callback=progress_callback,
                    )
                    player.prepare()
                if progress_detail_callback is not None:
                    progress_detail_callback(
                        str(run_id),
                        {
                            "phase": "runtime_ready",
                            "dataset_id": str(config.dataset_id or ""),
                            "predictor": str(config.predictor or "None"),
                        },
                    )
                runtimes.append(
                    {
                        "config": config,
                        "run_dir": run_dir,
                        "staging_dir": staging_dir,
                        "timing": timing,
                        "player": player,
                        "event_logger": event_logger,
                        "fov_latency_recorder": fov_latency_recorder,
                        "timing_saved": False,
                    }
                )

            tile_fov_cache: dict[tuple[str, int, tuple[str, ...]], dict[str, float] | None] = {}
            tile_prediction_cache: dict[tuple[str, str, int, int, tuple[str, ...]], dict[str, float] | None] = {}
            raycast_prediction_cache: dict[tuple, dict[str, float] | None] = {}
            prediction_cache_lock = Lock()

            def segment_decision_frame_window(player: Player, config: PlaybackConfig, segment_idx: int) -> list[Path]:
                del config
                tile_options = player.video.list_segment_tiles(segment_idx)
                request = player._prediction_request_for_segment(segment_idx, tile_options)  # pylint: disable=protected-access
                return list(request.geometry_frame_paths or [])

            def segment_ply_frame_window(player: Player, config: PlaybackConfig, segment_idx: int) -> list[Path]:
                predictor = getattr(player.fov, "predictor", None)
                frame_paths = segment_decision_frame_window(player, config, segment_idx)
                if predictor is not None and hasattr(predictor, "required_ply_frame_paths"):
                    try:
                        return list(predictor.required_ply_frame_paths(frame_paths))
                    except Exception:  # pylint: disable=broad-except
                        return frame_paths
                return frame_paths

            def segment_content_plan(items: list[dict], segment_idx: int) -> tuple[list[Path], bool]:
                frame_paths: list[Path] = []
                needs_ply = False
                for item in items:
                    player = item["player"]
                    if segment_idx >= player.total_segments():
                        continue
                    item_needs_ply = _predictor_needs_ply(item["config"])
                    if item_needs_ply:
                        window = segment_ply_frame_window(player, item["config"], segment_idx)
                        frame_paths.extend(window)
                        needs_ply = True
                return frame_paths, needs_ply

            def segment_plan(segment_idx: int) -> tuple[list[dict], list[Path], bool]:
                items = [
                    item for item in runtimes
                    if not item["player"].is_done() and int(item["player"].next_segment) == int(segment_idx)
                ]
                frame_paths, needs_ply = segment_content_plan(items, segment_idx)
                return items, frame_paths, needs_ply

            def predictor_cache_key(config: PlaybackConfig, player: Player, predictor, tiles, segment_frames_path) -> tuple | None:
                if config.record_fov_stage_latency:
                    return None
                predictor_name = predictor.__class__.__name__
                if predictor_name not in {"ReachabilityRaycastPredictor"}:
                    return None
                for attr in ("samples_per_tile", "max_voxels_eval", "min_voxel_prob"):
                    if str(getattr(predictor, attr, "")).strip().lower() == "auto":
                        return None
                tile_ids = tuple(sorted(str(getattr(tile, "unique_id", getattr(tile, "tile_id", ""))) for tile in tiles))
                frame_key = tuple(str(Path(path).resolve()) for path in segment_frames_path)
                request = getattr(player.fov, "_active_prediction_request", None)
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
                    predictor_name,
                    _stable_json(config.predictor_params or {}),
                    str(config.content_dir.resolve()),
                    str(player.fov.csv_path.resolve()),
                    int(player.next_segment),
                    frame_key,
                    request_key,
                    tile_ids,
                    float(config.fov_h_deg or 0.0),
                    float(config.fov_v_deg or 0.0),
                    str(config.angle_convention or ""),
                    _stable_json(config.motion_limits or {}),
                )

            def run_segment_item(item: dict, content, shared_raycast_cache: dict) -> None:
                # A group can contain hundreds of run items.  Check control at
                # this finer boundary as well as between segments so a cancel
                # does not wait for every item in the current segment.
                if control_callback is not None:
                    control_callback()
                player = item["player"]
                config = item["config"]
                run_id = str(config.run_id)
                if progress_detail_callback is not None:
                    progress_detail_callback(
                        run_id,
                        {
                            "phase": "segment_step_running",
                            "segment_idx": int(player.next_segment),
                            "predictor": str(config.predictor or "None"),
                            "abr": str(config.abr or ""),
                        },
                    )
                original_ground_truth = player._ground_truth_fov_visibility  # pylint: disable=protected-access

                def cached_ground_truth(tile_options, segment_frames_path, *, _player=player, _original=original_ground_truth):
                    tile_ids = tuple(str(tile.tile_id) for tile in tile_options)
                    key = (str(_player.fov.csv_path), int(_player.next_segment), tile_ids)
                    with prediction_cache_lock:
                        cached = tile_fov_cache.get(key)
                        has_cached = key in tile_fov_cache
                    if not has_cached:
                        computed = _original(tile_options, segment_frames_path)
                        with prediction_cache_lock:
                            cached = tile_fov_cache.setdefault(key, computed)
                        batch_timing.increment("tile_fov.cache_misses")
                    else:
                        batch_timing.increment("tile_fov.cache_hits")
                    return dict(cached) if isinstance(cached, dict) else cached

                original_predict_tiles = player.fov.predict_tiles
                predictor = getattr(player.fov, "predictor", None)

                def cached_predict_tiles(tiles, segment_frames_path, *, _player=player, _original=original_predict_tiles, _predictor=predictor, _config=config):
                    if _predictor.__class__.__name__ != "GroundTruthTileFovPredictor":
                        key = predictor_cache_key(_config, _player, _predictor, tiles, segment_frames_path)
                        if key is None:
                            return _original(tiles, segment_frames_path)
                        with prediction_cache_lock:
                            cached_raycast = raycast_prediction_cache.get(key)
                            has_cached = key in raycast_prediction_cache
                        if not has_cached:
                            computed = _original(tiles, segment_frames_path)
                            with prediction_cache_lock:
                                cached_raycast = raycast_prediction_cache.setdefault(key, computed)
                            batch_timing.increment("raycast_prediction.cache_misses")
                        else:
                            batch_timing.increment("raycast_prediction.cache_hits")
                        return dict(cached_raycast) if isinstance(cached_raycast, dict) else cached_raycast
                    tile_ids = tuple(sorted(str(getattr(tile, "unique_id", getattr(tile, "tile_id", ""))) for tile in tiles))
                    key = (
                        _predictor.__class__.__name__,
                        str(_player.fov.csv_path),
                        int(_player.next_segment),
                        int(getattr(_predictor, "sample_stride", 1)),
                        tile_ids,
                    )
                    with prediction_cache_lock:
                        cached = tile_prediction_cache.get(key)
                        has_cached = key in tile_prediction_cache
                    if not has_cached:
                        computed = _original(tiles, segment_frames_path)
                        with prediction_cache_lock:
                            cached = tile_prediction_cache.setdefault(key, computed)
                        batch_timing.increment("tile_fov_prediction.cache_misses")
                    else:
                        batch_timing.increment("tile_fov_prediction.cache_hits")
                    return dict(cached) if isinstance(cached, dict) else cached

                previous_raycast_cache = getattr(player.fov, "shared_raycast_cache", None)
                player._ground_truth_fov_visibility = cached_ground_truth  # pylint: disable=protected-access,method-assign
                player.fov.predict_tiles = cached_predict_tiles  # type: ignore[method-assign]
                player.fov.shared_raycast_cache = shared_raycast_cache
                try:
                    with item["timing"].activate():
                        player.step_segment(shared_content=content)
                    if progress_detail_callback is not None:
                        progress_detail_callback(
                            run_id,
                            {
                                "phase": "segment_step_done",
                                "segment_idx": int(player.next_segment) - 1,
                                "next_segment": int(player.next_segment),
                                "predictor": str(config.predictor or "None"),
                                "abr": str(config.abr or ""),
                            },
                        )
                finally:
                    player._ground_truth_fov_visibility = original_ground_truth  # pylint: disable=protected-access,method-assign
                    player.fov.predict_tiles = original_predict_tiles  # type: ignore[method-assign]
                    if previous_raycast_cache is None:
                        try:
                            delattr(player.fov, "shared_raycast_cache")
                        except AttributeError:
                            pass
                    else:
                        player.fov.shared_raycast_cache = previous_raycast_cache

            with SegmentContentLoader(worker_count=ply_io_workers, timing=batch_timing) as loader:
                prefetch_enabled = max(0, int(prefetch_segments or 0)) > 0
                prefetch_executor = ThreadPoolExecutor(max_workers=1) if prefetch_enabled else None
                prefetched: tuple[int, Future] | None = None
                try:
                    while any(not item["player"].is_done() for item in runtimes):
                        if control_callback is not None:
                            control_callback()
                        active_items = [item for item in runtimes if not item["player"].is_done()]
                        next_segment = min(item["player"].next_segment for item in active_items)
                        segment_items, frame_paths, needs_ply = segment_plan(int(next_segment))
                        with timing_stage("shared_video.list_segment_tiles.prewarm", segment_idx=int(next_segment)):
                            shared_video.list_segment_tiles(int(next_segment))
                        if prefetched is not None and prefetched[0] == int(next_segment):
                            if progress_detail_callback is not None:
                                for item in segment_items:
                                    progress_detail_callback(
                                        str(item["config"].run_id),
                                        {
                                            "phase": "shared_ply_prefetch_wait",
                                            "segment_idx": int(next_segment),
                                            "segment_runs": len(segment_items),
                                        },
                                    )
                            content = prefetched[1].result()
                            batch_timing.increment("shared_content.prefetch_hits")
                            prefetched = None
                        else:
                            if prefetched is not None:
                                prefetched[1].cancel()
                                batch_timing.increment("shared_content.prefetch_stale")
                                prefetched = None
                            if progress_detail_callback is not None:
                                for item in segment_items:
                                    progress_detail_callback(
                                        str(item["config"].run_id),
                                        {
                                            "phase": "shared_ply_load",
                                            "segment_idx": int(next_segment),
                                            "segment_runs": len(segment_items),
                                            "frame_count": len(set(str(path) for path in frame_paths)),
                                            "needs_ply": bool(needs_ply),
                                        },
                                    )
                            content = loader.load_window(next_segment, frame_paths, needs_ply=needs_ply)

                        future_segment = int(next_segment) + 1
                        if prefetch_executor is not None and any(
                            future_segment < item["player"].total_segments() for item in active_items
                        ):
                            future_paths, future_needs_ply = segment_content_plan(active_items, future_segment)
                            if future_needs_ply and future_paths:
                                prefetched = (
                                    future_segment,
                                    prefetch_executor.submit(loader.load_window, future_segment, future_paths, needs_ply=future_needs_ply),
                                )
                                batch_timing.increment("shared_content.prefetch_submitted")
                        shared_raycast_cache = {"lock": Lock(), "points_xyz": {}, "voxel_context": {}, "tile_samples": {}}
                        requested_segment_workers = int(segment_workers or 0)
                        if requested_segment_workers <= 0:
                            segment_worker_count, memory = auto_segment_worker_count(
                                len(segment_items),
                                max(1, int(segment_workers_auto_cap or 1)),
                            )
                            worker_mode = "auto"
                        else:
                            segment_worker_count = min(len(segment_items), requested_segment_workers)
                            memory = None
                            worker_mode = "fixed"
                        if progress_detail_callback is not None:
                            for item in segment_items:
                                progress_detail_callback(
                                    str(item["config"].run_id),
                                    {
                                        "phase": "segment_step",
                                        "segment_idx": int(next_segment),
                                        "segment_runs": len(segment_items),
                                        "segment_workers": segment_worker_count,
                                        "segment_worker_mode": worker_mode,
                                        "segment_worker_cap": int(segment_workers_auto_cap) if worker_mode == "auto" else requested_segment_workers,
                                        "available_ram_mib": (memory.available_bytes // (1024 * 1024)) if memory and memory.available_bytes is not None else None,
                                        "process_rss_mib": (memory.process_rss_bytes // (1024 * 1024)) if memory and memory.process_rss_bytes is not None else None,
                                        "predictor": str(item["config"].predictor or "None"),
                                        "abr": str(item["config"].abr or ""),
                                    },
                                )
                        if segment_worker_count <= 1:
                            for item in segment_items:
                                run_segment_item(item, content, shared_raycast_cache)
                        else:
                            with ThreadPoolExecutor(max_workers=segment_worker_count) as segment_executor:
                                futures = [
                                    segment_executor.submit(run_segment_item, item, content, shared_raycast_cache)
                                    for item in segment_items
                                ]
                                for future in as_completed(futures):
                                    future.result()
                        del content
                finally:
                    if prefetched is not None:
                        prefetched[1].cancel()
                    if prefetch_executor is not None:
                        prefetch_executor.shutdown(wait=True, cancel_futures=True)

            for item in runtimes:
                with item["timing"].activate():
                    if progress_detail_callback is not None:
                        progress_detail_callback(
                            str(item["config"].run_id),
                            {
                                "phase": "finishing_outputs",
                                "segment_idx": int(item["player"].next_segment),
                                "segment_total": int(item["player"].total_segments()),
                            },
                        )
                    item["player"].finish()
                    with timing_stage("outputs.save", run_id=item["config"].run_id):
                        _save_runtime_outputs(item["staging_dir"], item["event_logger"], item["fov_latency_recorder"])
                    item["timing"].save(item["staging_dir"] / "timing.json")
                    item["timing_saved"] = True
                    promote_incomplete_dir(item["run_dir"])
                    write_run_manifest(
                        str(item["config"].run_id),
                        item["config"],
                        item["player"].video,
                        item["run_dir"],
                    )
                    mark_run_complete(item["run_dir"], str(item["config"].run_id))
                    upsert_run_directory(item["run_dir"].parent, item["run_dir"])
                batch_timing.merge(item["timing"], prefix=str(item["config"].run_id))
    finally:
        for item in runtimes:
            if not item.get("timing_saved"):
                item["timing"].save(item["staging_dir"] / "timing.json")
        batch_dir = first.output_dir.resolve() / "_batch_timing"
        batch_timing.save(batch_dir / f"{batch_name}.json")
    return run_dirs


def main():
    args = parse_args()
    config = PlaybackConfig(
        mpd=args.mpd,
        content_dir=args.content_dir,
        network_trace=args.network_trace,
        network_profile=args.network_profile,
        fov_trace=args.fov_trace,
        abr=args.abr,
        predictor=args.predictor,
        bandwidth_predictor=args.bandwidth_predictor,
        abr_params=None,
        predictor_params=None,
        bandwidth_predictor_params=None,
        abr_preset_name=None,
        predictor_preset_name=None,
        bandwidth_predictor_preset_name=None,
        headset_preset=None,
        fov_h_deg=None,
        fov_v_deg=None,
        angle_convention=None,
        rtt=args.rtt,
        block_size_bits=args.block_size_bits,
        enable_fov_upgrades=args.enable_fov_upgrades,
        enable_mid_download_cancel=args.enable_mid_download_cancel,
        buffer_window_segments=args.buffer_window_segments,
        buffer_unbounded=args.buffer_unbounded,
        segment_limit=args.segment_limit,
        output_dir=args.output_dir,
        run_id=args.run_id,
        record_fov_stage_latency=False,
        ground_truth_mode=args.ground_truth_mode,
        ground_truth_visible_threshold=args.ground_truth_visible_threshold,
        ground_truth_peak_weight=args.ground_truth_peak_weight,
    )
    run_dir = run_playback(config)
    print(f"Run completed. Artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
