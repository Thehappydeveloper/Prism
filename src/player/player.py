import logging
import math
from pathlib import Path
from typing import Callable, Dict, List

from tqdm import tqdm

from abr.base import ABR
from core.buffer import Buffer
from core.clock import Clock
from core.event_logger import EventLogger
from core.fov import Fov
from core.logger import configure_logging, get_run_logger
from core.media_segment import TileComponentOptions, TileComponentSet as Tile
from core.network import Network
from core.prediction_context import PredictionRequest
from core.segment import Segment
from core.shared_content import SegmentContentWindow
from core.tile_fov import (
    DEFAULT_GROUND_TRUTH_MODE,
    DEFAULT_PEAK_WEIGHT,
    DEFAULT_VISIBLE_THRESHOLD,
    TileFovTrace,
    reduce_tile_visibility_frames,
)
from core.timing import TimingRecorder, timing_stage
from core.video import Video


class Player:
    """Adaptive VPCC streaming player."""

    def __init__(
        self,
        abr: ABR,
        network: Network,
        fov: Fov,
        video: Video,
        tile_fov: TileFovTrace | None = None,
        event_logger: EventLogger = None,
        show_progress: bool = True,
        run_dir: Path | None = None,
        run_id: str | None = None,
        enable_fov_upgrades: bool = False,
        enable_mid_download_cancel: bool = False,
        max_buffer_segments: int = 10,
        segment_limit: int = 0,
        timing_recorder: TimingRecorder | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        ground_truth_mode: str = DEFAULT_GROUND_TRUTH_MODE,
        ground_truth_visible_threshold: float = DEFAULT_VISIBLE_THRESHOLD,
        ground_truth_peak_weight: float = DEFAULT_PEAK_WEIGHT,
    ) -> None:
        self.clock = Clock(0.0)
        self.buffer = Buffer(max_segments=max_buffer_segments, event_logger=event_logger, clock=self.clock)
        self.next_segment = 0
        self.abr = abr
        self.network = network
        self.fov = fov
        self.tile_fov = tile_fov
        self.metrics = []
        self.video = video
        self.event_logger = event_logger
        self.show_progress = show_progress
        self.run_id = run_id or "default_run"
        self.enable_fov_upgrades = enable_fov_upgrades
        self.enable_mid_download_cancel = enable_mid_download_cancel
        self.segment_limit = max(0, int(segment_limit or 0))
        self.last_segment_download_summary: Dict[str, float | int | str] = {}
        self.last_fetched_segment_idx: int | None = None
        self.last_bandwidth_prediction: Dict | None = None
        self.timing_recorder = timing_recorder
        self.progress_callback = progress_callback
        self.ground_truth_mode = ground_truth_mode or DEFAULT_GROUND_TRUTH_MODE
        self.ground_truth_visible_threshold = float(ground_truth_visible_threshold)
        self.ground_truth_peak_weight = float(ground_truth_peak_weight)
        self._pbar = None

        if run_dir:
            configure_logging(run_dir, self.run_id)
        else:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            )

        # Create a logger specific to this class
        self.logger = get_run_logger(f"Player[{self.run_id}]")
        self.logger.info("Player initialized with video containing %d tiles.", self.video.num_tiles())

    def prepare(self) -> None:
        """Reset playback state before segment stepping."""
        self.logger.info("Starting playback...")
        self.clock.set_time(0.0)
        self.buffer.clear()
        self.next_segment = 0
        self.last_fetched_segment_idx = None

        num_segments = self.total_segments()
        self.logger.debug("Total number of segments to download: %d", num_segments)

        self._pbar = tqdm(total=num_segments, desc="Downloading segments") if self.show_progress else None
        self._emit_progress()

    def is_done(self) -> bool:
        return self.next_segment >= self.total_segments()

    def total_segments(self) -> int:
        """Execution length, capped by the source video's segment count."""
        video_segments = self.video.num_segments()
        return video_segments if self.segment_limit == 0 else min(video_segments, self.segment_limit)

    def step_segment(self, shared_content: SegmentContentWindow | None = None) -> bool:
        """Process one segment. Returns False when playback is already complete."""
        if self.is_done():
            return False

        def _run_step() -> None:
            self.logger.info("Processing segment %d...", self.next_segment)
            with timing_stage("player.wait_for_buffer_space", run_id=self.run_id, segment_idx=self.next_segment):
                wait_time = self.wait_for_buffer_space()
                self.buffer.consume(wait_time)

            with timing_stage("player.list_segment_tiles", run_id=self.run_id, segment_idx=self.next_segment):
                tile_options = self.video.list_segment_tiles(self.next_segment)
            segment_frames_path = self._fov_frame_window_for_decision()
            with timing_stage("player.abr_decision", run_id=self.run_id, segment_idx=self.next_segment):
                decision, predicted_fov = self.abr_decision(tile_options, segment_frames_path)
            bandwidth_prediction = dict(self.last_bandwidth_prediction or {}) if self.last_bandwidth_prediction else None
            prediction_debug = getattr(getattr(self.fov, "predictor", None), "last_visual_debug", None)
            self.logger.debug("ABR decision for segment %d: %s", self.next_segment, decision)

            decision_time_s = self.clock.get_time()
            with timing_stage("player.ground_truth_fov", run_id=self.run_id, segment_idx=self.next_segment):
                ground_truth_frame_paths = self.video.get_original_frames(self.next_segment) or []
                ground_truth_fov = self._ground_truth_fov_visibility(
                    tile_options,
                    ground_truth_frame_paths,
                )
                ground_truth_fov_frames = self._ground_truth_fov_frames(tile_options, ground_truth_frame_paths)
                ground_truth_target = self._reduce_ground_truth_frames(ground_truth_fov_frames, tile_options)
            with timing_stage("player.perform_download", run_id=self.run_id, segment_idx=self.next_segment):
                download_time, applied_decision = self.perform_download(decision, tile_options)
            if self.event_logger:
                self.event_logger.log_decision(
                    segment_idx=self.next_segment,
                    time_s=decision_time_s,
                    requested_decision=decision,
                    applied_decision=applied_decision,
                    predicted_fov=predicted_fov,
                    bandwidth_prediction=bandwidth_prediction,
                )
                if prediction_debug is not None:
                    self.event_logger.log_fov_prediction_debug(
                        segment_idx=self.next_segment,
                        time_s=decision_time_s,
                        debug_payload=prediction_debug,
                    )
                if ground_truth_fov is not None:
                    self.event_logger.log_fov_ground_truth(
                        segment_idx=self.next_segment,
                        time_s=decision_time_s,
                        tile_visibility=ground_truth_fov,
                        source="embedded_fov_trace",
                        mode="mean_visibility",
                    )
                if ground_truth_fov_frames:
                    self.event_logger.log_fov_ground_truth_frames(
                        segment_idx=self.next_segment,
                        time_s=decision_time_s,
                        frames=ground_truth_fov_frames,
                        source="embedded_fov_trace",
                        target_mode=self.ground_truth_mode,
                        target_visibility=ground_truth_target,
                        params={
                            "visible_threshold": self.ground_truth_visible_threshold,
                            "peak_weight": self.ground_truth_peak_weight,
                        },
                    )
            summary = self.last_segment_download_summary
            self.logger.info(
                "Segment %d summary: size %.2f Mb, download %.2f s, avg bandwidth %.2f Mbps, predicted %.2f Mbps via %s, waited %.2f s, reps %s",
                self.next_segment,
                float(summary.get("size_mbit", 0.0)),
                download_time,
                float(summary.get("avg_bandwidth_mbps", 0.0)),
                float((bandwidth_prediction or {}).get("predicted_bandwidth_bps", 0.0)) / 1e6,
                str((bandwidth_prediction or {}).get("predictor", "None")),
                wait_time,
                str(summary.get("selection_summary", "-")),
            )

            self.next_segment += 1
            if self._pbar:
                self._pbar.update(1)
            self._emit_progress()

        if shared_content is None:
            _run_step()
        else:
            with shared_content.activate():
                _run_step()
        return True

    def finish(self) -> None:
        self._emit_progress()
        if self._pbar:
            self._pbar.close()
            self._pbar = None
        self.logger.info("Playback finished.")

    def _emit_progress(self) -> None:
        if self.progress_callback is None:
            return
        try:
            total = self.total_segments()
            self.progress_callback(min(self.next_segment, total), total)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.debug("Progress callback failed: %s", exc)

    def play(self) -> None:
        """Start playback until the last segment is downloaded."""
        self.prepare()
        while self.step_segment():
            pass
        self.finish()


    def wait_for_buffer_space(self) -> float:
        """Handle the case when the playback buffer is full."""
        if self.buffer.has_space_for_one_more:
            self.logger.debug("Buffer has space for another segment.")
            return 0.0
        else:
            remaining_time = self.buffer.segments[0].remaining_rendering_time_s
            self.logger.warning("Buffer full, waiting %.2f s", remaining_time)
            self.clock.add(remaining_time)
            return remaining_time

    # --- ABR augmentation helpers ---

    def _sanitize_decisions(self, decisions: Dict[str, Dict[str, int]], option_map: Dict[str, TileComponentOptions]) -> Dict[str, Dict[str, int]]:
        """Clamp ABR output to available reps and fill missing components with lowest quality."""
        sanitized: Dict[str, Dict[str, int]] = {}
        for tile_id, option in option_map.items():
            tile_decision = decisions.get(tile_id, {})
            fixed = {}
            for comp_name, reps in option.component_options.items():
                rep_idx = tile_decision.get(comp_name, 0)
                rep_idx = max(0, min(rep_idx, len(reps) - 1))
                fixed[comp_name] = rep_idx
            sanitized[tile_id] = fixed
        return sanitized

    def _decision_size_bits(self, decisions: Dict[str, Dict[str, int]], option_map: Dict[str, TileComponentOptions]) -> int:
        """Estimate total segment size (bits) for a decision map."""
        total = 0
        for tile_id, comp_map in decisions.items():
            option = option_map.get(tile_id)
            if option is None:
                continue
            for comp_name, reps in option.component_options.items():
                rep_idx = comp_map.get(comp_name, 0)
                rep_idx = max(0, min(rep_idx, len(reps) - 1))
                total += reps[rep_idx].size_bits
        return total

    def _estimate_view_position(self, frame_number: int | None):
        """Best-effort viewer position for distance ordering; None if unavailable."""
        if frame_number is None:
            return None
        row = self.fov.get_frame(int(frame_number) - int(self.fov.start_frame_index))
        if row is None:
            return None
        candidates = [
            ("PositionX", "PositionY", "PositionZ"),
            ("pos_x", "pos_y", "pos_z"),
            ("x", "y", "z"),
            ("X", "Y", "Z"),
        ]
        for keys in candidates:
            try:
                vals = tuple(float(row[k]) for k in keys)
            except Exception:  # pylint: disable=broad-except
                continue
            if all(math.isfinite(v) for v in vals):
                return vals
        return None

    def _tile_distance(self, tile_option: TileComponentOptions, view_pos) -> float:
        if not view_pos:
            return 0.0
        cx, cy, cz = tile_option.center
        vx, vy, vz = view_pos
        return math.sqrt((cx - vx) ** 2 + (cy - vy) ** 2 + (cz - vz) ** 2)

    def _plan_fov_centric_upgrades(
        self, decisions: Dict[str, Dict[str, int]], option_map: Dict[str, TileComponentOptions]
    ) -> Dict[str, Dict[str, int]]:
        """Greedily upgrade tiles starting in FoV then outward when time budget allows."""
        sanitized = self._sanitize_decisions(decisions, option_map)
        buffer_budget_s = self.buffer.buffer_duration_s()
        if buffer_budget_s <= 0:
            return sanitized

        segment_duration = 1.0
        for option in option_map.values():
            for reps in option.component_options.values():
                if reps:
                    segment_duration = reps[0].duration_s
                    break
            break
        if hasattr(self.network, "predict_bandwidth_for_decision"):
            bw_bps = self.network.predict_bandwidth_for_decision(self.clock.get_time(), segment_duration_s=segment_duration)
        else:
            bw_bps = self.network.get_instantaneous_bandwidth(self.clock.get_time())
        bw_bps = max(bw_bps, 1e-6)
        rtt_s = self.network.get_instantaneous_rtt(self.clock.get_time())
        base_bits = self._decision_size_bits(sanitized, option_map)
        est_base_time = base_bits / bw_bps + rtt_s
        if est_base_time >= buffer_budget_s:
            return sanitized

        frame_idx = self.current_observed_frame_idx()
        view_pos = self._estimate_view_position(frame_idx)
        request = self._prediction_request_for_segment(self.next_segment, list(option_map.values()))
        segment_frames_path: List[Path] = request.geometry_frame_paths
        previous_request = getattr(self.fov, "_active_prediction_request", None)
        self.fov._active_prediction_request = request
        try:
            proportions = self.abr.predict_fov(self.fov, option_map, segment_frames_path)
        finally:
            if previous_request is None:
                try:
                    delattr(self.fov, "_active_prediction_request")
                except AttributeError:
                    pass
            else:
                self.fov._active_prediction_request = previous_request

        candidates = []
        for tile_id, comp_map in sanitized.items():
            option = option_map.get(tile_id)
            if option is None:
                continue
            tile_in_fov = (proportions.get(tile_id, 0.0) or 0.0) > 0.0
            distance = self._tile_distance(option, view_pos)
            for comp_name, reps in option.component_options.items():
                current_idx = comp_map.get(comp_name, 0)
                for next_idx in range(current_idx + 1, len(reps)):
                    extra_bits = reps[next_idx].size_bits - reps[current_idx].size_bits
                    if extra_bits <= 0:
                        continue
                    candidates.append(
                        {
                            "tile_id": tile_id,
                            "component": comp_name,
                            "from_idx": current_idx,
                            "to_idx": next_idx,
                            "extra_bits": extra_bits,
                            "in_fov": tile_in_fov,
                            "distance": distance,
                        }
                    )

        candidates.sort(key=lambda c: (0 if c["in_fov"] else 1, c["distance"], c["extra_bits"]))
        extra_budget_bits = max(0.0, (buffer_budget_s - est_base_time) * bw_bps * 0.9)

        for cand in candidates:
            if cand["extra_bits"] <= extra_budget_bits:
                sanitized[cand["tile_id"]][cand["component"]] = cand["to_idx"]
                extra_budget_bits -= cand["extra_bits"]
        return sanitized

    def _best_downswitch_rep(
        self,
        reps,
        current_idx: int,
        remaining_bits: float,
        current_bw: float,
        time_to_deadline: float,
        rtt_s: float,
        reference_bw: float | None = None,
    ) -> int | None:
        """
        Pick the highest lower rep that can finish sooner than the current one and before the deadline.
        """
        if time_to_deadline <= 0 or current_idx <= 0:
            return None
        ref = reference_bw or current_bw
        if current_bw >= ref * 0.8:  # no meaningful drop
            return None

        time_finish_current = remaining_bits / max(current_bw, 1e-6)
        if time_finish_current <= time_to_deadline:
            return None  # We can finish current in time.

        best_idx = None
        for idx in range(current_idx - 1, -1, -1):
            alt_seg = reps[idx]
            restart_time = rtt_s + alt_seg.size_bits / max(current_bw, 1e-6)
            if restart_time < time_finish_current and restart_time < time_to_deadline:
                best_idx = idx  # higher idx in loop is closest to current; keep first found
                break
        return best_idx

    def _download_component(
        self,
        tile_id: str,
        component_name: str,
        target_rep_idx: int,
        option: TileComponentOptions,
        buffer_budget_s: float,
        segment_elapsed_s: float,
        rtt_s: float,
        time_step_ms: int,
    ) -> tuple[int, float, float]:
        reps = option.component_options.get(component_name, [])
        if not reps:
            return 0, 0.0, 0.0

        target_rep_idx = max(0, min(target_rep_idx, len(reps) - 1))
        current_seg = reps[target_rep_idx]
        lower_idx = target_rep_idx - 1
        lower_seg = reps[lower_idx] if lower_idx >= 0 else None

        remaining_bits = current_seg.size_bits
        comp_time = 0.0
        total_bandwidth = 0.0
        bandwidth_samples = 0
        reference_bw = None

        while remaining_bits > 0:
            bw_bits_per_s = max(self.network.get_instantaneous_bandwidth(self.clock.get_time()), 1e-6)
            reference_bw = reference_bw or bw_bits_per_s
            time_to_deadline = max(0.0, buffer_budget_s - (segment_elapsed_s + comp_time))

            if self.enable_mid_download_cancel and lower_seg:
                best_idx = self._best_downswitch_rep(
                    reps=reps,
                    current_idx=current_seg.rep_idx,
                    remaining_bits=remaining_bits,
                    current_bw=bw_bits_per_s,
                    time_to_deadline=time_to_deadline,
                    rtt_s=rtt_s,
                    reference_bw=reference_bw,
                )
            else:
                best_idx = None

            if best_idx is not None:
                next_seg = reps[best_idx]
                self.logger.info(
                    "Adaptive abort: tile %s component %s from rep %d to %d (deadline %.2fs)",
                    tile_id,
                    component_name,
                    current_seg.rep_idx,
                    next_seg.rep_idx,
                    time_to_deadline,
                )
                current_seg = next_seg
                remaining_bits = current_seg.size_bits
                lower_idx = current_seg.rep_idx - 1
                lower_seg = reps[lower_idx] if lower_idx >= 0 else None
                cancel_cost = rtt_s
                self.clock.add(cancel_cost)
                comp_time += cancel_cost
                total_bandwidth += bw_bits_per_s
                bandwidth_samples += 1
                continue

            downloaded_bits = min(remaining_bits, bw_bits_per_s * time_step_ms / 1000)
            actual_time_s = downloaded_bits / bw_bits_per_s
            self.clock.add(actual_time_s)
            comp_time += actual_time_s
            total_bandwidth += bw_bits_per_s
            bandwidth_samples += 1
            remaining_bits -= downloaded_bits

        avg_bw_mbps = (total_bandwidth / max(1, bandwidth_samples)) / 1e6
        self.logger.debug(
            "Tile %s component %s (rep %d) avg bandwidth during download: %.2f Mbps",
            tile_id,
            component_name,
            current_seg.rep_idx,
            avg_bw_mbps,
        )
        return current_seg.rep_idx, comp_time, avg_bw_mbps

    def _format_selection_summary(self, decisions: Dict[str, Dict[str, int]], limit: int = 8) -> str:
        chunks = []
        component_order = ("atlas", "geom", "occp", "attr")
        tile_ids = sorted(decisions.keys(), key=lambda value: (int(value) if str(value).isdigit() else 10**9, str(value)))
        for tile_id in tile_ids[:limit]:
            comp_map = decisions.get(tile_id, {})
            parts = [f"{component}{comp_map[component]}" for component in component_order if component in comp_map]
            parts.extend(f"{component}{comp_map[component]}" for component in sorted(comp_map.keys()) if component not in component_order)
            chunks.append(f"T{tile_id}:{'/'.join(parts) if parts else '-'}")
        if len(tile_ids) > limit:
            chunks.append(f"+{len(tile_ids) - limit} tiles")
        return "; ".join(chunks) if chunks else "-"

    def perform_download(
        self,
        decision: Dict[str, Dict[str, int]],
        tile_options: List[TileComponentOptions],
        time_step_ms: int = 10,
    ) -> tuple[float, Dict[str, Dict[str, int]]]:
        seg_idx = self.next_segment
        tile_option_map = {t.tile_id: t for t in tile_options}
        decisions_complete = self._sanitize_decisions(decision, tile_option_map)
        if self.enable_fov_upgrades:
            decisions_complete = self._plan_fov_centric_upgrades(decisions_complete, tile_option_map)

        download_time_s = 0.0
        buffer_budget_s = self.buffer.buffer_duration_s()
        tiles_in_segment: Dict[str, Tile] = {}
        applied_decision: Dict[str, Dict[str, int]] = {}
        bandwidth_samples: List[float] = []

        rtt_s = self.network.get_instantaneous_rtt(self.clock.get_time())
        self.logger.debug("RTT for segment %d: %.2f s", seg_idx, rtt_s)
        download_time_s += rtt_s
        self.clock.add(rtt_s)

        for tile_id, tile_decision in decisions_complete.items():
            option = tile_option_map.get(tile_id)
            if option is None:
                self.logger.warning("ABR returned unknown tile %s; skipping.", tile_id)
                continue

            final_selection: Dict[str, int] = {}
            for comp_name in option.component_options.keys():
                target_idx = tile_decision.get(comp_name, 0)
                rep_idx, comp_time, avg_bw_mbps = self._download_component(
                    tile_id=tile_id,
                    component_name=comp_name,
                    target_rep_idx=target_idx,
                    option=option,
                    buffer_budget_s=buffer_budget_s,
                    segment_elapsed_s=download_time_s,
                    rtt_s=rtt_s,
                    time_step_ms=time_step_ms,
                )
                download_time_s += comp_time
                final_selection[comp_name] = rep_idx
                if avg_bw_mbps > 0:
                    bandwidth_samples.append(avg_bw_mbps)

            if not final_selection:
                self.logger.warning("Tile %s had no components to download; skipping.", tile_id)
                continue

            tile = self.video.load_segment(tile_id, seg_idx, final_selection)
            tiles_in_segment[tile_id] = tile
            applied_decision[tile_id] = dict(final_selection)
            self.logger.debug(
                "Downloaded tile %s with reps %s (total size %.2f Mb).",
                tile_id,
                final_selection,
                tile.size_bits / 1e6,
            )

        if not tiles_in_segment:
            raise RuntimeError("No tiles were downloaded for this segment.")

        any_tile = next(iter(tiles_in_segment.values()))
        segment = Segment(segment_idx=seg_idx, duration_s=any_tile.duration_s, tiles=tiles_in_segment)

        self.buffer.consume(download_time_s)
        if self.event_logger:
            self.event_logger.log_consume(segment, download_time_s, self.clock.get_time())
        self.buffer.enqueue(segment)
        self.last_fetched_segment_idx = int(seg_idx)
        self.last_segment_download_summary = {
            "segment_idx": seg_idx,
            "size_mbit": segment.size_bits / 1e6,
            "avg_bandwidth_mbps": sum(bandwidth_samples) / max(1, len(bandwidth_samples)),
            "selection_summary": self._format_selection_summary(applied_decision),
        }
        if self.event_logger:
            self.event_logger.log_enqueue(segment, self.clock.get_time())

        return download_time_s, applied_decision


    def _predict_fov_visibility(
        self,
        tile_options: List[TileComponentOptions],
        segment_frames_path: List[Path],
    ) -> Dict[str, float]:
        tile_groups = self.abr.group_tiles(tile_options)
        try:
            proportions = self.abr.peek_predicted_fov(self.fov, tile_groups, segment_frames_path)
            if proportions is None:
                proportions = self.abr.predict_fov(self.fov, tile_groups, segment_frames_path)
            proportions = proportions or {}
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.debug("FoV prediction logging skipped due to error: %s", exc)
            proportions = {}

        out: Dict[str, float] = {}
        for tile_id in tile_groups.keys():
            raw_val = proportions.get(tile_id, 0.0)
            try:
                val = float(raw_val)
            except (TypeError, ValueError):
                val = 0.0
            if not math.isfinite(val):
                val = 0.0
            out[str(tile_id)] = max(0.0, val)
        return out

    def _ground_truth_fov_visibility(
        self,
        tile_options: List[TileComponentOptions],
        segment_frames_path: List[Path],
    ) -> Dict[str, float] | None:
        if self.tile_fov is None or not segment_frames_path:
            return None
        tile_ids = [str(tile.tile_id) for tile in tile_options]
        try:
            return self.tile_fov.visibility_for_frame_paths(segment_frames_path, tile_ids=tile_ids)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.debug("Tile FoV ground truth logging skipped due to error: %s", exc)
            return None

    def _ground_truth_fov_frames(
        self,
        tile_options: List[TileComponentOptions],
        segment_frames_path: List[Path],
    ) -> list[Dict]:
        if self.tile_fov is None or not segment_frames_path:
            return []
        tile_ids = [str(tile.tile_id) for tile in tile_options]
        try:
            return self.tile_fov.visibility_frames_for_frame_paths(segment_frames_path, tile_ids=tile_ids)
        except Exception as exc:  # pylint: disable=broad-except
            self.logger.debug("Tile FoV frame ground truth logging skipped due to error: %s", exc)
            return []

    def _reduce_ground_truth_frames(
        self,
        frames: list[Dict],
        tile_options: List[TileComponentOptions],
    ) -> Dict[str, float]:
        tile_ids = [str(tile.tile_id) for tile in tile_options]
        return reduce_tile_visibility_frames(
            frames,
            tile_ids=tile_ids,
            mode=self.ground_truth_mode,
            visible_threshold=self.ground_truth_visible_threshold,
            peak_weight=self.ground_truth_peak_weight,
        )

    def _fov_frame_window_for_decision(self) -> List[Path]:
        # The initial fetch window is deliberately conservative: those
        # segments are selected before any playback frame is observable. This
        # gate also applies to offline/oracle predictors so their first usable
        # prediction cannot arrive earlier than any normal predictor's.
        if self.next_segment < self.buffer.max_segments:
            return []
        return self.available_geometry_frames()

    def _fps(self) -> float:
        resolved_fps = getattr(self.video, "frame_rate_hz", None)
        if resolved_fps is not None:
            return max(float(resolved_fps), 1e-6)
        duration = float(getattr(self.video, "total_playback_time", 0.0) or 0.0)
        if duration > 0.0:
            return max(float(self.video.total_frames) / duration, 1e-6)
        segment_duration = float(getattr(self.video, "segment_duration_s", 0.0) or 0.0)
        if segment_duration > 0.0 and self.video.num_segments() > 0:
            return max(float(self.video.total_frames) / (segment_duration * self.video.num_segments()), 1e-6)
        return 30.0

    def current_observed_frame_idx(self) -> int | None:
        if self.buffer.total_playback_time_s <= 0.0:
            return None
        played_frames = int(math.floor(float(self.buffer.total_playback_time_s) * self._fps()))
        if played_frames <= 0:
            return None
        frame_idx = int(self.video.start_frame_idx + played_frames - 1)
        max_frame_idx = int(self.video.start_frame_idx + max(0, self.video.total_frames - 1))
        return min(frame_idx, max_frame_idx)

    def _segment_duration_for_tiles(self, tile_options: List[TileComponentOptions]) -> float:
        for option in tile_options:
            for reps in option.component_options.values():
                if reps:
                    return max(float(reps[0].duration_s), 1e-6)
        return max(float(getattr(self.video, "segment_duration_s", 1.0) or 1.0), 1e-6)

    def prediction_horizon_for_segment(
        self,
        segment_idx: int,
        target_segment_duration_s: float | None = None,
    ) -> float:
        duration = max(float(target_segment_duration_s or getattr(self.video, "segment_duration_s", 1.0) or 1.0), 1e-6)
        target_end_s = (int(segment_idx) * duration) + duration
        return max(0.0, target_end_s - float(self.buffer.total_playback_time_s))

    def available_geometry_frames(self) -> List[Path]:
        segment_idx = None
        if self.buffer.segments:
            segment_idx = int(self.buffer.segments[-1].segment_idx)
        elif self.last_fetched_segment_idx is not None:
            segment_idx = int(self.last_fetched_segment_idx)
        if segment_idx is None:
            return []
        frames = self.video.get_original_frames(segment_idx) or []
        return [frames[-1]] if frames else []

    def _prediction_request_for_segment(
        self,
        segment_idx: int,
        tile_options: List[TileComponentOptions],
    ) -> PredictionRequest:
        segment_duration = self._segment_duration_for_tiles(tile_options)
        geometry_frames = self.available_geometry_frames()
        predictor = getattr(self.fov, "predictor", None)
        use_oracle_target = bool(getattr(predictor, "uses_future_target_frames", False))
        oracle_target_frames = self.video.get_original_frames(segment_idx) if use_oracle_target else None
        return PredictionRequest(
            target_segment_idx=int(segment_idx),
            geometry_frame_paths=geometry_frames,
            observed_frame_idx=self.current_observed_frame_idx(),
            prediction_horizon_s=self.prediction_horizon_for_segment(segment_idx, segment_duration),
            target_segment_duration_s=segment_duration,
            oracle_target_frame_paths=list(oracle_target_frames or []) if use_oracle_target else None,
            ground_truth_mode=self.ground_truth_mode if use_oracle_target else None,
            ground_truth_visible_threshold=self.ground_truth_visible_threshold if use_oracle_target else None,
            ground_truth_peak_weight=self.ground_truth_peak_weight if use_oracle_target else None,
        )

    def abr_decision(
        self,
        tile_options: List[TileComponentOptions],
        segment_frames_path: List[Path] | None = None,
    ) -> tuple[Dict[str, Dict[str, int]], Dict[str, float]]:
        request = self._prediction_request_for_segment(self.next_segment, tile_options)
        segment_frames_path = segment_frames_path if segment_frames_path is not None else request.geometry_frame_paths
        if segment_frames_path is not request.geometry_frame_paths:
            request = PredictionRequest(
                target_segment_idx=request.target_segment_idx,
                geometry_frame_paths=list(segment_frames_path or []),
                observed_frame_idx=request.observed_frame_idx,
                prediction_horizon_s=request.prediction_horizon_s,
                target_segment_duration_s=request.target_segment_duration_s,
                oracle_target_frame_paths=request.oracle_target_frame_paths,
                ground_truth_mode=request.ground_truth_mode,
                ground_truth_visible_threshold=request.ground_truth_visible_threshold,
                ground_truth_peak_weight=request.ground_truth_peak_weight,
            )
        self.network.last_prediction = None
        previous_request = getattr(self.fov, "_active_prediction_request", None)
        self.fov._active_prediction_request = request
        try:
            decision = self.abr.decide(tile_options, segment_frames_path, self.network, self.fov, self.buffer, self.clock)
            self.last_bandwidth_prediction = dict(self.network.last_prediction or {}) if self.network.last_prediction else None
            predicted_fov = self._predict_fov_visibility(tile_options, segment_frames_path)
        finally:
            if previous_request is None:
                try:
                    delattr(self.fov, "_active_prediction_request")
                except AttributeError:
                    pass
            else:
                self.fov._active_prediction_request = previous_request
        return decision, predicted_fov
    
