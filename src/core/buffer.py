from typing import List
from core.segment import Segment
from core.clock import Clock
from core.event_logger import EventLogger

class Buffer:
    """Playback buffer for downloaded Segment objects."""

    def __init__(self, max_segments: int = 10, event_logger: EventLogger=None, clock: Clock=None):
        self.segments: List[Segment] = []  # List of Segment objects
        self.max_segments = max_segments
        self.total_rebuffering_time_s = 0.0
        self.total_playback_time_s = 0.0
        self.event_logger = event_logger
        self.clock = clock

    # --- Core buffer operations ---

    def clear(self):
        """Empty the buffer and reset playback state."""
        self.segments.clear()

    def enqueue(self, segment: Segment):
        """Add a segment to the buffer."""
        if self.is_full:
            raise RuntimeError("Buffer full: cannot enqueue more segments")
        self.segments.append(segment)

    def consume(self, seconds: float):
        """
        Consume a given amount of seconds from the buffer.

        Decreases remaining playback time of the segments from oldest to newest.
        Fully consumed segments are removed from the buffer.
        If buffer is insufficient, consume everything and add remaining time to total rebuffering time.
        """
        remaining_s = seconds
        while remaining_s > 0 and self.segments:
            seg = self.segments[0]
            if seg.remaining_rendering_time_s > remaining_s:
                seg.remaining_rendering_time_s -= remaining_s
                self.total_playback_time_s += remaining_s
                remaining_s = 0
            else:
                remaining_s -= seg.remaining_rendering_time_s
                self.total_playback_time_s += seg.remaining_rendering_time_s
                self.segments.pop(0)  # fully consumed segment

        if remaining_s > 0:
            # Buffer ran out, treat remaining time as rebuffering
            if self.event_logger:
                self.event_logger.log_rebuffering(remaining_s, self.clock.get_time()-remaining_s)
            self.total_rebuffering_time_s += remaining_s

    # --- Checks ---

    @property
    def is_full(self) -> bool:
        """Check if buffer has reached its maximum capacity."""
        return len(self.segments) >= self.max_segments

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.segments) == 0

    @property
    def has_space_for_one_more(self) -> bool:
        """Check if there is space for at least one more segment."""
        return len(self.segments) < self.max_segments

    # --- Information utilities ---

    def __len__(self) -> int:
        """Return the number of segments in the buffer."""
        return len(self.segments)

    def buffer_duration_s(self) -> float:
        """Return the total remaining duration of buffered segments in seconds."""
        return sum(seg.remaining_rendering_time_s for seg in self.segments)

    # --- Segment access and replacement ---

    def replace_segment(self, position: int, new_segment: Segment):
        """Replace the segment at a given position with a new Segment object."""
        if position < 0 or position >= len(self.segments):
            raise IndexError("Invalid buffer position")
        self.segments[position] = new_segment

    # --- String representation ---

    def __str__(self) -> str:
        return (f"Buffer: {len(self.segments)} segments, "
                f"total remaining duration {self.buffer_duration_s():.2f}s, "
                f"space for one more: {self.has_space_for_one_more()}")
