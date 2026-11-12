from typing import Callable, List, Union

class Clock:
    """
    A simple clock that tracks time in seconds.
    Supports subscriptions for time updates.
    """

    def __init__(self, time_s: Union[int, float] = 0):
        """time_ms: Initial time in milliseconds."""
        self.time: float = float(time_s)
        self._subscribers: List[Callable[[float], None]] = []

    def __str__(self) -> str:
        return f"{self.time:.0f} s"

    def subscribe(self, callback: Callable[[float], None]) -> None:
        """Subscribe a function that receives the updated time in s."""
        self._subscribers.append(callback)

    def _notify(self) -> None:
        """Notify subscribers of the current time in s."""
        for callback in self._subscribers:
            callback(self.time)

    def add(self, amount_s: Union[int, float]) -> "Clock":
        """Add seconds to the current time."""
        self.time += float(amount_s)
        self._notify()
        return self

    def subtract(self, amount_s: Union[int, float]) -> "Clock":
        """Subtract seconds from the current time."""
        self.time -= float(amount_s)
        self._notify()
        return self
    
    def set_time(self, new_time_s: Union[int, float]) -> "Clock":
        """Set the clock to a specific time in ms."""
        self.time = float(new_time_s)
        self._notify()
        return self
    
    def get_time(self) -> float:
        """Get the current time in s."""
        return self.time

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Clock):
            return NotImplemented
        return abs(self.time - other.time) < 1e-3
