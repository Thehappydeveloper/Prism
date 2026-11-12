"""Fast, dependency-light validation of Prism's run-artifact foundation.

This does not simulate media playback.  It verifies the common contract that
every simulator run relies on: a resolved manifest can be recorded, indexed,
and retrieved as a paginated browser row.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from core.run_index import query_run_index, rebuild_run_index
from core.run_lifecycle import mark_run_complete
from core.run_manifest import validate_run_manifest, write_run_manifest


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="prism-artifact-smoke-"))
    try:
        run_dir = root / "run_smoke"
        run_dir.mkdir()
        mpd = root / "content.mpd"
        trace = root / "trace.csv"
        mpd.write_text("<MPD />", encoding="utf-8")
        trace.write_text("FrameNumber\n0\n", encoding="utf-8")
        config = {
            "mpd": str(mpd), "content_dir": str(root), "fov_trace": str(trace),
            "abr": "ThroughputAwareABR", "predictor": "Argus", "dataset_id": "smoke",
        }
        (run_dir / "config.json").write_text(
            '{"mpd":"content.mpd","abr":"ThroughputAwareABR","predictor":"Argus","dataset_id":"smoke"}',
            encoding="utf-8",
        )
        (run_dir / "status.json").write_text(
            '{"run_id":"run_smoke","status":"finished","finished":"2026-01-01T00:00:00Z"}',
            encoding="utf-8",
        )
        video = SimpleNamespace(total_frames=30, start_frame_idx=0, total_playback_time=1.0, segment_count=1, frame_rate_hz=30.0)
        manifest_path = write_run_manifest("run_smoke", config, video, run_dir)
        import json
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert validate_run_manifest(manifest) == []
        mark_run_complete(run_dir, "run_smoke")
        assert rebuild_run_index(root) == 1
        rows, total = query_run_index(root, limit=10)
        assert total == 1 and rows[0]["run_id"] == "run_smoke"
        print("Artifact/index smoke test passed.")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
