import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from server_app import fov_comparison


def _write_run(root: Path, run_id: str = "run_test") -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"abr": "FoVRoundRobinABR"}), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"run_id": run_id, "status": "finished"}), encoding="utf-8")
    (run_dir / "events.json").write_text(
        json.dumps(
            {
                "enqueued_segments": [
                    {
                        "segment_idx": 0,
                        "time_s": 0.0,
                        "duration_s": 2.0,
                        "tiles": {
                            "0": {
                                "bounds": {
                                    "min_corner": [0.0, 0.0, 0.0],
                                    "max_corner": [1.0, 1.0, 1.0],
                                }
                            },
                            "1": {
                                "bounds": {
                                    "min_corner": [1.0, 0.0, 0.0],
                                    "max_corner": [2.0, 1.0, 1.0],
                                }
                            },
                        },
                    }
                ],
                "abr_decisions": [{"segment_idx": 0, "time_s": 0.25}],
                "fov_predictions": [{"segment_idx": 0, "tile_visibility": {"0": 1.2, "1": 0.25}}],
                "fov_ground_truth": [{"segment_idx": 0, "tile_visibility": {"0": 0.5, "1": 0.0}}],
            }
        ),
        encoding="utf-8",
    )
    return run_dir


class FovComparisonTests(unittest.TestCase):
    def test_builds_dataset_from_existing_run_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_run(root)
            with patch.object(fov_comparison, "RESULTS_ROOT", root):
                dataset = fov_comparison.build_fov_comparison_dataset("run_test")

        self.assertEqual(dataset["schema"], "prism.fov_comparison.v1")
        self.assertEqual(dataset["display"]["color_scale"], [0.0, 1.0])
        self.assertEqual(len(dataset["frames"]), 1)
        frame = dataset["frames"][0]
        self.assertEqual(frame["segment_idx"], 0)
        self.assertEqual(frame["time_s"], 0.25)
        self.assertEqual(frame["predicted"], {"0": 1.2, "1": 0.25})
        self.assertEqual(frame["actual"], {"0": 0.5, "1": 0.0})
        self.assertIn("soft_iou", frame["metrics"])
        self.assertIn("mae", frame["metrics"])

    def test_generate_persists_structured_error_when_ffmpeg_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_run(root)
            with patch.object(fov_comparison, "RESULTS_ROOT", root), patch("server_app.fov_comparison.shutil.which", return_value=None):
                result = fov_comparison.generate_fov_comparison_video("run_test")

            self.assertFalse(result["ok"])
            self.assertIn("ffmpeg is required", result["error"])
            self.assertTrue((root / "run_test" / "artifacts" / "fov_prediction_comparison.json").exists())
            status = json.loads((root / "run_test" / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["fov_comparison"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
