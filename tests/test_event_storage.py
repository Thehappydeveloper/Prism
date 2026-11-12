from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.event_logger import EventLogger
from core.event_storage import event_log_path, read_event_log_text
from core.run_lifecycle import (
    atomic_write_json,
    completion_path,
    has_completion_marker,
    mark_run_complete,
    prepare_incomplete_dir,
    promote_incomplete_dir,
)


class EventStorageTests(unittest.TestCase):
    def test_event_logger_writes_and_reads_gzip_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            path = run_dir / "events.json.gz"
            logger = EventLogger()
            logger.total_playback_time = 12.5

            logger.save(path)

            self.assertTrue(path.exists())
            self.assertFalse((run_dir / ".events.json.gz.tmp").exists())
            self.assertEqual(event_log_path(run_dir), path)
            payload = json.loads(read_event_log_text(path))
            self.assertEqual(payload["total_playback_time"], 12.5)

    def test_plain_event_logs_remain_readable_and_gzip_is_preferred(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            plain = run_dir / "events.json"
            compressed = run_dir / "events.json.gz"
            plain.write_text('{"format":"plain"}', encoding="utf-8")

            self.assertEqual(event_log_path(run_dir), plain)
            self.assertEqual(json.loads(read_event_log_text(plain))["format"], "plain")

            EventLogger().save(compressed)
            self.assertEqual(event_log_path(run_dir), compressed)

    def test_completion_marker_is_atomic_and_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            status_path = run_dir / "status.json"

            atomic_write_json(status_path, {"status": "running"})
            self.assertEqual(json.loads(status_path.read_text(encoding="utf-8"))["status"], "running")
            self.assertFalse(has_completion_marker(run_dir))

            mark_run_complete(run_dir, "run_test")

            self.assertTrue(has_completion_marker(run_dir))
            self.assertEqual(json.loads(completion_path(run_dir).read_text(encoding="utf-8"))["run_id"], "run_test")
            self.assertFalse((run_dir / ".completion.json.tmp").exists())

    def test_atomic_write_json_supports_concurrent_status_updates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            status_path = Path(tmp) / "status.json"
            barrier = threading.Barrier(8)
            errors: list[Exception] = []

            def write_status(index: int) -> None:
                try:
                    barrier.wait()
                    atomic_write_json(status_path, {"writer": index})
                except Exception as exc:  # pragma: no cover - asserted below
                    errors.append(exc)

            threads = [threading.Thread(target=write_status, args=(index,)) for index in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(errors, [])
            self.assertIn(json.loads(status_path.read_text(encoding="utf-8"))["writer"], range(8))
            self.assertFalse(list(status_path.parent.glob(".status.json.*.tmp")))

    def test_runtime_artifacts_are_hidden_until_staging_is_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            staging = prepare_incomplete_dir(run_dir)
            (staging / "events.json.gz").write_bytes(b"compressed-events")
            (staging / "timing.json").write_text("{}", encoding="utf-8")

            self.assertFalse((run_dir / "events.json.gz").exists())
            promote_incomplete_dir(run_dir)

            self.assertTrue((run_dir / "events.json.gz").exists())
            self.assertTrue((run_dir / "timing.json").exists())
            self.assertFalse(staging.exists())


if __name__ == "__main__":
    unittest.main()
