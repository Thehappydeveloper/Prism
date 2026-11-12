"""Small, dependency-free system resource probes for runtime scheduling."""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path


_GIB = 1024**3


@dataclass(frozen=True)
class MemorySnapshot:
    total_bytes: int | None
    available_bytes: int | None
    process_rss_bytes: int | None


def _linux_memory_snapshot() -> MemorySnapshot:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            fields = value.split()
            if fields and fields[0].isdigit():
                values[key] = int(fields[0]) * 1024
    except OSError:
        return MemorySnapshot(None, None, _process_rss_bytes())
    return MemorySnapshot(values.get("MemTotal"), values.get("MemAvailable"), _process_rss_bytes())


def _windows_memory_snapshot() -> MemorySnapshot:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return MemorySnapshot(None, None, _process_rss_bytes())
    return MemorySnapshot(int(status.ullTotalPhys), int(status.ullAvailPhys), _process_rss_bytes())


def _process_rss_bytes() -> int | None:
    """Current resident set size when the host exposes it without extra deps."""
    try:
        statm = Path("/proc/self/statm").read_text(encoding="utf-8").split()
        if len(statm) >= 2:
            return int(statm[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError):
        pass
    return None


def memory_snapshot() -> MemorySnapshot:
    """Return physical-memory availability; unavailable fields are ``None``."""
    if sys.platform.startswith("linux"):
        return _linux_memory_snapshot()
    if sys.platform == "win32":
        return _windows_memory_snapshot()
    # Many POSIX systems expose these counters, including current macOS builds.
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total = int(os.sysconf("SC_PHYS_PAGES")) * page_size
        available = int(os.sysconf("SC_AVPHYS_PAGES")) * page_size
        return MemorySnapshot(total, available, _process_rss_bytes())
    except (AttributeError, OSError, ValueError):
        return MemorySnapshot(None, None, _process_rss_bytes())


def auto_segment_worker_count(active_runs: int, worker_cap: int) -> tuple[int, MemorySnapshot]:
    """Choose a conservative per-segment parallelism level from RAM and CPUs."""
    snapshot = memory_snapshot()
    cpu_cap = max(1, (os.cpu_count() or 1) - 1)
    cap = max(1, min(int(active_runs), int(worker_cap), cpu_cap))
    if snapshot.available_bytes is None:
        return cap, snapshot

    # Retain a 2-GiB system reserve and budget 768 MiB for each concurrent
    # raycast/ABR step.  This is intentionally conservative: shared geometry
    # is already resident before a segment begins, while per-run prediction
    # and point-cloud work can spike during the segment.
    reserve = 2 * _GIB
    per_worker = 768 * 1024**2
    ram_cap = max(1, (max(0, snapshot.available_bytes - reserve) // per_worker))
    return max(1, min(cap, int(ram_cap))), snapshot
