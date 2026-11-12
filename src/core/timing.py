from __future__ import annotations

import json
import time
from threading import Lock
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Iterator


_ACTIVE_TIMING: ContextVar["TimingRecorder | None"] = ContextVar("active_timing", default=None)


def active_timing() -> "TimingRecorder | None":
    return _ACTIVE_TIMING.get()


class TimingRecorder:
    """Small wall-clock timing accumulator for run and batch diagnostics."""

    def __init__(self, name: str | None = None) -> None:
        self.name = name or "timing"
        self.started_at = time.time()
        self.stages: Dict[str, Dict[str, Any]] = {}
        self.counters: Dict[str, float] = {}
        self._lock = Lock()

    @contextmanager
    def activate(self) -> Iterator["TimingRecorder"]:
        token = _ACTIVE_TIMING.set(self)
        try:
            yield self
        finally:
            _ACTIVE_TIMING.reset(token)

    @contextmanager
    def stage(self, name: str, **metadata: Any) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            with self._lock:
                bucket = self.stages.setdefault(
                    name,
                    {
                        "count": 0,
                        "total_ms": 0.0,
                        "min_ms": None,
                        "max_ms": 0.0,
                        "last_ms": 0.0,
                        "last_metadata": {},
                    },
                )
                bucket["count"] += 1
                bucket["total_ms"] += elapsed_ms
                bucket["min_ms"] = elapsed_ms if bucket["min_ms"] is None else min(bucket["min_ms"], elapsed_ms)
                bucket["max_ms"] = max(bucket["max_ms"], elapsed_ms)
                bucket["last_ms"] = elapsed_ms
                if metadata:
                    bucket["last_metadata"] = {key: str(value) for key, value in metadata.items()}

    def increment(self, name: str, amount: float = 1.0) -> None:
        with self._lock:
            self.counters[name] = float(self.counters.get(name, 0.0) + amount)

    def merge(self, other: "TimingRecorder", prefix: str | None = None) -> None:
        label = f"{prefix}." if prefix else ""
        for name, bucket in other.stages.items():
            target = self.stages.setdefault(
                f"{label}{name}",
                {
                    "count": 0,
                    "total_ms": 0.0,
                    "min_ms": None,
                    "max_ms": 0.0,
                    "last_ms": 0.0,
                    "last_metadata": {},
                },
            )
            target["count"] += int(bucket.get("count", 0))
            target["total_ms"] += float(bucket.get("total_ms", 0.0))
            source_min = bucket.get("min_ms")
            if source_min is not None:
                target["min_ms"] = source_min if target["min_ms"] is None else min(target["min_ms"], source_min)
            target["max_ms"] = max(float(target.get("max_ms", 0.0)), float(bucket.get("max_ms", 0.0)))
            target["last_ms"] = float(bucket.get("last_ms", 0.0))
            target["last_metadata"] = dict(bucket.get("last_metadata") or {})
        for name, value in other.counters.items():
            self.increment(f"{label}{name}", value)

    def to_dict(self) -> Dict[str, Any]:
        finished_at = time.time()
        stages = {}
        for name, bucket in self.stages.items():
            count = int(bucket.get("count", 0))
            total_ms = float(bucket.get("total_ms", 0.0))
            stages[name] = {
                **bucket,
                "avg_ms": total_ms / count if count else 0.0,
            }
        return {
            "name": self.name,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "wall_ms": (finished_at - self.started_at) * 1000.0,
            "stages": stages,
            "counters": dict(self.counters),
        }

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


@contextmanager
def timing_stage(name: str, **metadata: Any) -> Iterator[None]:
    recorder = active_timing()
    if recorder is None:
        yield
        return
    with recorder.stage(name, **metadata):
        yield
