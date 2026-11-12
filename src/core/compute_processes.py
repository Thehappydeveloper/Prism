"""Emergency control for process pools owned by active batch computation.

``ProcessPoolExecutor`` deliberately has no public API for terminating a task
that is already running.  Prism nevertheless needs an operator escape hatch:
a hung prediction must not make the web server impossible to stop.  Pools
registered here are limited to batch PLY/raycast work and are terminated only
by an explicit force-cancel request.
"""

from __future__ import annotations

import threading
from concurrent.futures import ProcessPoolExecutor
from typing import Any


_LOCK = threading.RLock()
_ACTIVE_POOLS: dict[int, ProcessPoolExecutor] = {}
_FORCE_STOP_REQUESTED = threading.Event()


def begin_compute_session() -> None:
    """Clear a previous force-stop before a new batch group begins."""
    _FORCE_STOP_REQUESTED.clear()


def register_compute_pool(pool: ProcessPoolExecutor) -> None:
    with _LOCK:
        _ACTIVE_POOLS[id(pool)] = pool


def unregister_compute_pool(pool: ProcessPoolExecutor) -> None:
    with _LOCK:
        _ACTIVE_POOLS.pop(id(pool), None)


def force_stop_requested() -> bool:
    return _FORCE_STOP_REQUESTED.is_set()


def force_stop_compute_pools() -> dict[str, int]:
    """Terminate active batch-compute children without stopping the web server.

    Access to ``_processes`` is intentionally contained here: the standard
    executor interface cannot terminate already-running work.  Each process is
    joined by executor shutdown when its owning computation unwinds.
    """
    _FORCE_STOP_REQUESTED.set()
    with _LOCK:
        pools = list(_ACTIVE_POOLS.values())

    terminated = 0
    for pool in pools:
        processes: Any = getattr(pool, "_processes", {})  # pylint: disable=protected-access
        for process in list((processes or {}).values()):
            if process.is_alive():
                process.terminate()
                terminated += 1
        # Do not wait in the HTTP handler. The batch thread receives the
        # broken-pool error, unwinds its context managers, and reaps children.
        pool.shutdown(wait=False, cancel_futures=True)
    return {"pools": len(pools), "terminated_processes": terminated}
