import gzip
import json
import os
from pathlib import Path
from typing import List, Dict, Any

from core.segment import Segment


class EventLogger:
    """Logs events during VPCC playback to evaluate ABR performance."""

    def __init__(self):
        self.time_s: List[float] = []                  # Simulation time in seconds
        self.buffer_occupancy_s: List[float] = []      # Buffer duration in seconds
        self.enqueued_segments: List[Dict] = []        # Metadata for enqueued segments
        self.consumed_segments: List[Dict] = []        # Metadata for consumed segments
        self.rebuffering_events: List[Dict] = []       # Rebuffering events
        self.abr_decisions: List[Dict] = []            # ABR decision traces (requested/applied)
        self.fov_predictions: List[Dict] = []          # FoV predictor output snapshots
        self.fov_prediction_debug: List[Dict] = []     # Optional visual FoV predictor debug snapshots
        self.fov_ground_truth: List[Dict] = []         # Ground-truth FoV visibility snapshots
        self.fov_ground_truth_frames: List[Dict] = []  # Raw per-frame ground-truth FoV visibility snapshots
        self.bandwidth_predictions: List[Dict] = []    # Bandwidth predictor output snapshots
        self.ground_truth_mode: str = "visible_fraction"
        self.ground_truth_params: Dict[str, Any] = {"visible_threshold": 0.01, "peak_weight": 0.3}
        self.total_playback_time: float = 0.0
        self.total_rebuffering_time: float = 0.0
        self.current_buffer_s: float = 0.0             # Instantaneous buffer occupancy

    def log_enqueue(self, segment: Segment, time_s: float):
        """Log when a segment is added to the buffer."""
        self.enqueued_segments.append(segment.to_dict() | {"time_s": time_s})
        self._update_buffer(time_s, segment.duration_s)

    def log_consume(self, segment: Segment, consumed_s: float, time_s: float):
        """Log consumption of a segment; buffer goes down naturally."""
        self.consumed_segments.append({
            "time_s": time_s,
            "consumed_s": consumed_s,}
            | segment.to_dict()
            )
        self.total_playback_time += consumed_s
        self._update_buffer(time_s, -consumed_s)

    def log_rebuffering(self, duration_s: float, time_s: float):
        """Log rebuffering events (buffer drops to zero)."""
        self.rebuffering_events.append({
            "time_s": time_s,
            "duration_s": duration_s
        })
        self.total_rebuffering_time += duration_s
        self._update_buffer(time_s, -self.current_buffer_s)

    def log_decision(
        self,
        segment_idx: int,
        time_s: float,
        requested_decision: Dict[str, Dict[str, int]],
        applied_decision: Dict[str, Dict[str, int]],
        predicted_fov: Dict[str, float] | None = None,
        bandwidth_prediction: Dict[str, Any] | None = None,
    ) -> None:
        """Log per-segment ABR + predictor decisions for downstream visualizations."""
        self.abr_decisions.append(
            {
                "segment_idx": int(segment_idx),
                "time_s": float(time_s),
                "requested": requested_decision,
                "applied": applied_decision,
            }
        )
        if predicted_fov is not None:
            self.fov_predictions.append(
                {
                    "segment_idx": int(segment_idx),
                    "time_s": float(time_s),
                    "tile_visibility": predicted_fov,
                }
            )
        if bandwidth_prediction is not None:
            payload = dict(bandwidth_prediction)
            payload["segment_idx"] = int(segment_idx)
            payload["time_s"] = float(time_s)
            self.bandwidth_predictions.append(payload)

    def log_fov_prediction_debug(
        self,
        segment_idx: int,
        time_s: float,
        debug_payload: Dict[str, Any],
    ) -> None:
        """Log compact visual-debug data explaining the FoV prediction."""
        payload = dict(debug_payload)
        payload["segment_idx"] = int(segment_idx)
        payload["time_s"] = float(time_s)
        self.fov_prediction_debug.append(payload)

    def log_fov_ground_truth(
        self,
        segment_idx: int,
        time_s: float,
        tile_visibility: Dict[str, float],
        source: str,
        mode: str = "mean_visibility",
    ) -> None:
        """Log per-segment ground-truth tile visibility for evaluation."""
        self.fov_ground_truth.append(
            {
                "segment_idx": int(segment_idx),
                "time_s": float(time_s),
                "source": str(source),
                "mode": str(mode),
                "tile_visibility": tile_visibility,
            }
        )

    def log_fov_ground_truth_frames(
        self,
        segment_idx: int,
        time_s: float,
        frames: List[Dict[str, Any]],
        source: str,
        target_mode: str,
        target_visibility: Dict[str, float],
        params: Dict[str, Any] | None = None,
    ) -> None:
        """Log per-frame ground-truth tile visibility and its selected segment reduction."""
        self.ground_truth_mode = str(target_mode)
        if params is not None:
            self.ground_truth_params = dict(params)
        self.fov_ground_truth_frames.append(
            {
                "segment_idx": int(segment_idx),
                "time_s": float(time_s),
                "source": str(source),
                "target_mode": str(target_mode),
                "params": dict(params or {}),
                "target_visibility": target_visibility,
                "frames": frames,
            }
        )

    def _update_buffer(self, time_s: float, delta_s: float):
        """Update buffer occupancy tracking and ensure it never goes below zero."""
        self.current_buffer_s = max(self.current_buffer_s + delta_s, 0.0)
        self.time_s.append(time_s)
        self.buffer_occupancy_s.append(self.current_buffer_s)

    def average_buffered_bitrate_kbps(self) -> float:
        """Compute average bitrate of all enqueued segments."""
        if not self.enqueued_segments:
            return 0.0
        total_bits = sum(seg["size_bits"] for seg in self.enqueued_segments)
        total_duration_s = sum(seg["duration_s"] for seg in self.enqueued_segments)
        return (total_bits / total_duration_s) / 1e3 if total_duration_s > 0 else 0.0
    

    # ---------- Export & Import ----------

    def to_dict(self) -> Dict[str, Any]:
        """Convert the EventLogger state to a dictionary."""
        return {
            "time_s": self.time_s,
            "buffer_occupancy_s": self.buffer_occupancy_s,
            "enqueued_segments": self.enqueued_segments,
            "consumed_segments": self.consumed_segments,
            "rebuffering_events": self.rebuffering_events,
            "abr_decisions": self.abr_decisions,
            "fov_predictions": self.fov_predictions,
            "fov_prediction_debug": self.fov_prediction_debug,
            "fov_ground_truth": self.fov_ground_truth,
            "fov_ground_truth_frames": self.fov_ground_truth_frames,
            "ground_truth_mode": self.ground_truth_mode,
            "ground_truth_params": self.ground_truth_params,
            "bandwidth_predictions": self.bandwidth_predictions,
            "total_playback_time": self.total_playback_time,
            "total_rebuffering_time": self.total_rebuffering_time,
        }

    def save(self, path: Path) -> None:
        """Save experiment logs as JSON, optionally compressed by a ``.gz`` suffix."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        opener = gzip.open if path.name.endswith(".gz") else open
        with opener(temporary, "wt", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        with temporary.open("rb") as f:
            os.fsync(f.fileno())
        temporary.replace(path)

    def summary(self):
        print(f"Total playback time: {self.total_playback_time:.2f}s")
        print(f"Total rebuffering time: {self.total_rebuffering_time:.2f}s")
        print(f"Average buffered bitrate: {self.average_buffered_bitrate_kbps():.2f} kbps")
        print(f"Total segments enqueued: {len(self.enqueued_segments)}")
        print(f"Total segments consumed: {len(self.consumed_segments)}")
        print(f"Rebuffering events: {len(self.rebuffering_events)}")
