import logging
import math
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from abr.base import ABR
from core.buffer import Buffer
from core.clock import Clock
from core.event_logger import EventLogger
from core.fov import Fov
from core.logger import configure_logging, get_run_logger
from core.media_segment import TileComponentOptions, TileComponentSet as Tile
from core.network import Network
from core.segment import Segment
from core.video import Video


class Player:
    """Adaptive VPCC streaming player."""

    def __init__(
        self,
        abr: ABR,
        network: Network,
        fov: Fov,
        video: Video,
        event_logger: EventLogger = None,
        show_progress: bool = True,
        run_dir: Path | None = None,
        run_id: str | None = None,
        enable_fov_upgrades: bool = False,
        enable_mid_download_cancel: bool = False,
        max_buffer_segments: int = 10,
    ) -> None:
        self.clock = Clock(0.0)
        self.buffer = Buffer(max_segments=max_buffer_segments, event_logger=event_logger, clock=self.clock)
        self.next_segment = 0
        self.abr = abr
        self.network = network
        self.fov = fov
        self.metrics = []
        self.video = video
        self.event_logger = event_logger
        self.show_progress = show_progress
        self.run_id = run_id or "default_run"
        self.enable_fov_upgrades = enable_fov_upgrades
        self.enable_mid_download_cancel = enable_mid_download_cancel

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

    def play(self) -> None:
        """Start playback until the last segment is downloaded."""
        self.logger.info("Starting playback...")
        self.clock.set_time(0.0)
        self.buffer.clear()
        self.next_segment = 0

        num_segments = self.video.num_segments()
        self.logger.debug("Total number of segments to download: %d", num_segments)

        # Only show tqdm if enabled
        pbar = tqdm(total=num_segments, desc="Downloading segments") if self.show_progress else None

        while self.next_segment < num_segments:
            self.logger.info("Processing segment %d...", self.next_segment)
            wait_time = self.wait_for_buffer_space()
            self.buffer.consume(wait_time)

            tile_options = self.video.list_segment_tiles(self.next_segment)
            decision = self.abr_decision(tile_options)
            self.logger.debug("ABR decision for segment %d: %s", self.next_segment, decision)

            download_time = self.perform_download(decision, tile_options)
            self.logger.info("Segment %d downloaded in %.2f s", self.next_segment, download_time)

            self.next_segment += 1
            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()
        self.logger.info("Playback finished.")


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

    def _estimate_view_position(self, frame_number: int):
        """Best-effort viewer position for distance ordering; None if unavailable."""
        row = self.fov.get_frame(frame_number)
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

        bw_bps = max(self.network.get_instantaneous_bandwidth(self.clock.get_time()), 1e-6)
        rtt_s = self.network.get_instantaneous_rtt(self.clock.get_time())
        base_bits = self._decision_size_bits(sanitized, option_map)
        est_base_time = base_bits / bw_bps + rtt_s
        if est_base_time >= buffer_budget_s:
            return sanitized

        frame_idx = self.video.start_frame_idx + max(self.next_segment, 0)
        view_pos = self._estimate_view_position(frame_idx)
        segment_frames_path: List[Path] = self.video.get_original_frames(self.next_segment - 1)
        proportions = self.abr.predict_fov(self.fov, option_map, segment_frames_path)

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
    ) -> tuple[int, float]:
        reps = option.component_options.get(component_name, [])
        if not reps:
            return 0, 0.0

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
        self.logger.info(
            "Tile %s component %s (rep %d) avg bandwidth during download: %.2f Mbps",
            tile_id,
            component_name,
            current_seg.rep_idx,
            avg_bw_mbps,
        )
        return current_seg.rep_idx, comp_time

    def perform_download(self, decision: Dict[str, Dict[str, int]], tile_options: List[TileComponentOptions], time_step_ms: int = 10) -> float:
        seg_idx = self.next_segment
        tile_option_map = {t.tile_id: t for t in tile_options}
        decisions_complete = self._sanitize_decisions(decision, tile_option_map)
        if self.enable_fov_upgrades:
            decisions_complete = self._plan_fov_centric_upgrades(decisions_complete, tile_option_map)

        download_time_s = 0.0
        buffer_budget_s = self.buffer.buffer_duration_s()
        tiles_in_segment: Dict[str, Tile] = {}

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
                rep_idx, comp_time = self._download_component(
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

            if not final_selection:
                self.logger.warning("Tile %s had no components to download; skipping.", tile_id)
                continue

            tile = self.video.load_segment(tile_id, seg_idx, final_selection)
            tiles_in_segment[tile_id] = tile
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
        self.logger.info("Segment %d total size: %.2f Mb", self.next_segment, segment.size_bits / 1e6)
        if self.event_logger:
            self.event_logger.log_enqueue(segment, self.clock.get_time())

        return download_time_s


    def abr_decision(self, tile_options: List[TileComponentOptions]) -> Dict[str, Dict[str, int]]:
        segment_frames_path: List[Path] = self.video.get_original_frames(self.next_segment-1)
        decision = self.abr.decide(tile_options, segment_frames_path, self.network, self.fov, self.buffer, self.clock)
        return decision
    
