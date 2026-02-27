import json
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
            "total_playback_time": self.total_playback_time,
            "total_rebuffering_time": self.total_rebuffering_time,
        }

    def save(self, path: Path) -> None:
        """Save experiment logs to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def summary(self):
        print(f"Total playback time: {self.total_playback_time:.2f}s")
        print(f"Total rebuffering time: {self.total_rebuffering_time:.2f}s")
        print(f"Average buffered bitrate: {self.average_buffered_bitrate_kbps():.2f} kbps")
        print(f"Total segments enqueued: {len(self.enqueued_segments)}")
        print(f"Total segments consumed: {len(self.consumed_segments)}")
        print(f"Rebuffering events: {len(self.rebuffering_events)}")
