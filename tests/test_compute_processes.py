from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.compute_processes import (  # noqa: E402
    begin_compute_session,
    force_stop_compute_pools,
    force_stop_requested,
    register_compute_pool,
    unregister_compute_pool,
)


class _FakeProcess:
    def __init__(self, alive: bool = True) -> None:
        self.alive = alive
        self.terminated = False

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False


class _FakePool:
    def __init__(self) -> None:
        self._processes = {1: _FakeProcess(True), 2: _FakeProcess(False)}
        self.shutdown_calls: list[tuple[bool, bool]] = []

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        self.shutdown_calls.append((wait, cancel_futures))


class ComputeProcessControlTests(unittest.TestCase):
    def test_force_stop_terminates_registered_active_workers(self) -> None:
        begin_compute_session()
        pool = _FakePool()
        register_compute_pool(pool)  # type: ignore[arg-type]
        try:
            result = force_stop_compute_pools()
        finally:
            unregister_compute_pool(pool)  # type: ignore[arg-type]
            begin_compute_session()

        self.assertTrue(force_stop_requested() is False)
        self.assertEqual(result, {"pools": 1, "terminated_processes": 1})
        self.assertTrue(pool._processes[1].terminated)
        self.assertFalse(pool._processes[2].terminated)
        self.assertEqual(pool.shutdown_calls, [(False, True)])

