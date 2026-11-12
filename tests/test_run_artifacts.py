from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.run_index import indexed_run_count, query_run_index, rebuild_run_index, remove_indexed_run, upsert_run_directory
from core.run_lifecycle import mark_run_complete
from core.run_manifest import RUN_MANIFEST_SCHEMA, validate_run_manifest, write_run_manifest


class RunArtifactTests(unittest.TestCase):
    def test_index_can_list_only_ungrouped_runs(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "results"
            for run_id, group_id in (("run_single", None), ("run_grouped", "batch_a")):
                run_dir = root / run_id
                run_dir.mkdir(parents=True)
                (run_dir / "config.json").write_text(json.dumps({"run_group_id": group_id} if group_id else {}), encoding="utf-8")
                (run_dir / "status.json").write_text(json.dumps({"status": "finished"}), encoding="utf-8")

            self.assertEqual(rebuild_run_index(root), 2)
            items, total = query_run_index(root, ungrouped=True)
            self.assertEqual(total, 1)
            self.assertEqual([item["run_id"] for item in items], ["run_single"])

    def test_indexed_run_can_be_removed_after_artifact_deletion(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "results"
            run_dir = root / "run_delete_me"
            run_dir.mkdir(parents=True)
            (run_dir / "config.json").write_text("{}", encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"status": "finished"}), encoding="utf-8")
            self.assertTrue(upsert_run_directory(root, run_dir))

            self.assertTrue(remove_indexed_run(root, "run_delete_me"))
            self.assertEqual(indexed_run_count(root), 0)
            self.assertFalse(remove_indexed_run(root, "run_delete_me"))

    def test_manifest_and_index_support_new_and_legacy_runs(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "results"
            run_dir = root / "run_new"
            run_dir.mkdir(parents=True)
            mpd = Path(temporary) / "video.mpd"
            trace = Path(temporary) / "fov.csv"
            mpd.write_text("<MPD />", encoding="utf-8")
            trace.write_text("FrameNumber\n0\n", encoding="utf-8")
            config = {
                "mpd": str(mpd), "content_dir": str(Path(temporary)), "fov_trace": str(trace),
                "dataset_id": "s15", "abr": "RankBudgetABR", "predictor": "LinearExtrapolationPredictor",
            }
            (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
            (run_dir / "events.json.gz").write_bytes(b"placeholder")
            video = SimpleNamespace(total_frames=300, start_frame_idx=0, total_playback_time=10.0, segment_count=2, frame_rate_hz=30.0)
            manifest_path = write_run_manifest("run_new", config, video, run_dir)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema"], RUN_MANIFEST_SCHEMA)
            self.assertEqual(validate_run_manifest(manifest), [])
            mark_run_complete(run_dir, "run_new")

            legacy = root / "run_legacy"
            legacy.mkdir()
            (legacy / "config.json").write_text(json.dumps({"abr": "BandwidthAwareABR"}), encoding="utf-8")
            (legacy / "status.json").write_text(json.dumps({"status": "finished"}), encoding="utf-8")

            self.assertTrue(upsert_run_directory(root, run_dir))
            self.assertEqual(indexed_run_count(root), 1)
            self.assertEqual(rebuild_run_index(root), 2)
            self.assertEqual(indexed_run_count(root), 2)


if __name__ == "__main__":
    unittest.main()
