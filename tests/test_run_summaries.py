from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from server_app.run_summaries import build_log_story, build_predictor_story, build_readable_summary


class RunSummaryTests(unittest.TestCase):
    def test_readable_summary_builds_segment_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            run_id = "run_test"
            (run_dir / "config.json").write_text(json.dumps({"abr": "AbrA", "predictor": "PredictorA"}), encoding="utf-8")
            (run_dir / "events.json").write_text(
                json.dumps(
                    {
                        "time_s": [0.0, 1.2],
                        "buffer_occupancy_s": [0.0, 2.0],
                        "enqueued_segments": [
                            {"segment_idx": 0, "time_s": 1.2, "duration_s": 2.0, "size_bits": 4_000_000}
                        ],
                        "consumed_segments": [{"segment_idx": 0, "time_s": 1.2, "consumed_s": 1.2}],
                        "rebuffering_events": [],
                        "abr_decisions": [
                            {
                                "segment_idx": 0,
                                "time_s": 0.0,
                                "requested": {"0": {"geom": 1}},
                                "applied": {"0": {"atlas": 0, "geom": 1, "occp": 0, "attr": 2}},
                            }
                        ],
                        "fov_predictions": [
                            {"segment_idx": 0, "time_s": 0.0, "tile_visibility": {"0": 0.8, "1": 0.2}}
                        ],
                        "total_playback_time": 2.0,
                        "total_rebuffering_time": 0.0,
                    }
                ),
                encoding="utf-8",
            )

            summary = build_readable_summary(run_dir, run_id)

            self.assertEqual(summary["overview"]["abr"], "AbrA")
            self.assertEqual(summary["overview"]["segment_count"], 1)
            self.assertEqual(summary["timeline"][0]["download_time_s"], 1.2)
            self.assertIn("T0", summary["timeline"][0]["selection_summary"])
            self.assertEqual(summary["timeline"][0]["top_fov_tiles"][0]["tile_id"], "0")

    def test_log_story_collapses_component_bandwidth(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            run_id = "run_test"
            (run_dir / f"{run_id}.log").write_text(
                "\n".join(
                    [
                        "2026-01-01 00:00:00 [INFO] run.Player: Processing segment 2...",
                        "2026-01-01 00:00:00 [WARNING] run.Player: Buffer full, waiting 1.50 s",
                        "2026-01-01 00:00:00 [INFO] run.Player: Tile 0 component geom (rep 3) avg bandwidth during download: 10.00 Mbps",
                        "2026-01-01 00:00:00 [INFO] run.Player: Tile 1 component attr (rep 2) avg bandwidth during download: 20.00 Mbps",
                        "2026-01-01 00:00:00 [INFO] run.Player: Segment 2 total size: 12.50 Mb",
                        "2026-01-01 00:00:00 [INFO] run.Player: Segment 2 downloaded in 0.75 s",
                    ]
                ),
                encoding="utf-8",
            )

            story = build_log_story(run_dir, run_id)

            self.assertTrue(story["available"])
            self.assertEqual(story["segments"][0]["component_count"], 2)
            self.assertEqual(story["segments"][0]["avg_component_bandwidth_mbps"], 15.0)
            self.assertIn("waited 1.50s", story["events"][0]["message"])

    def test_predictor_story_hides_raw_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "predictor_diagnostics.json").write_text(
                json.dumps(
                    {
                        "predictor_name": "Raycast",
                        "prediction_count": 2,
                        "total": {"median_ms": 10, "mean_ms": 12, "p95_ms": 20, "max_ms": 30},
                        "stages": [
                            {"name": "fast", "count": 2, "median_ms": 1, "mean_ms": 1, "p95_ms": 1, "max_ms": 1},
                            {"name": "slow", "count": 2, "median_ms": 5, "mean_ms": 8, "p95_ms": 15, "max_ms": 18},
                        ],
                        "metadata_summary": {
                            "candidate_voxels": {"count": 2, "min": 100, "median": 120, "max": 140},
                            "debug_noise": {"count": 2, "min": 1, "median": 1, "max": 1},
                        },
                        "metadata_samples": [{"large": "raw"}],
                    }
                ),
                encoding="utf-8",
            )

            story = build_predictor_story(run_dir)

            self.assertTrue(story["available"])
            self.assertEqual(story["slowest_stages"][0]["name"], "slow")
            self.assertEqual(story["metadata"][0]["name"], "candidate_voxels")
            self.assertNotIn("metadata_samples", story)

    def test_missing_predictor_diagnostics_is_readable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            story = build_predictor_story(Path(tmp))

            self.assertFalse(story["available"])
            self.assertIn("not found", story["summary"])


if __name__ == "__main__":
    unittest.main()
