import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from server_app.diagnostics import _compute_run_metrics, tile_quality


class StaticQoe:
    def compute_time_series(self, _events, run_config=None):
        return [(0.0, 1.0)]


class FovPredictionDiagnosticsTests(unittest.TestCase):
    def test_base_only_component_has_zero_chart_quality(self):
        """A component never upgraded by the ABR must not divide by zero."""
        self.assertEqual(
            tile_quality(
                {"components": {"atlas": {"quality_id": "quality_0", "rep_idx": 0}}},
                {"atlas": 0},
            ),
            0.0,
        )

    def test_emits_compact_fov_statistics_for_plugins(self):
        run = {
            "run_id": "run_test",
            "config": {
                "abr": "FoVRoundRobinABR",
                "predictor": "UniformTileImportancePredictor",
                "bandwidth_predictor": "None",
            },
            "events": {
                "enqueued_segments": [
                    {
                        "segment_idx": 0,
                        "duration_s": 2.0,
                        "size_bits": 600.0,
                        "tiles": {
                            "0": {
                                "size_bits": 100.0,
                                "components": {"geom": {"quality_id": "quality_1", "rep_idx": 1, "size_bits": 100.0}},
                            },
                            "1": {
                                "size_bits": 200.0,
                                "components": {"geom": {"quality_id": "quality_0", "rep_idx": 0, "size_bits": 200.0}},
                            },
                        },
                    }
                ],
                "fov_predictions": [
                    {"segment_idx": 0, "tile_visibility": {"0": 0.8, "1": 0.2}},
                ],
                "fov_ground_truth": [
                    {"segment_idx": 0, "tile_visibility": {"0": 0.5, "1": 0.0}},
                ],
                "rebuffering_events": [],
                "buffer_occupancy_s": [],
            },
        }

        metrics = _compute_run_metrics(run, StaticQoe())

        stats = metrics["fov_prediction"]["run_test"]
        self.assertEqual(metrics["fov_segments"]["run_test"], [])
        self.assertEqual(len(stats["mae"]), 1)
        self.assertAlmostEqual(stats["mae"][0], 0.25)
        self.assertEqual(len(stats["weightedIou"]), 1)
        self.assertEqual(len(stats["wastedUnseenBits"]), 1)

    def test_prefers_per_frame_ground_truth_target_when_available(self):
        run = {
            "run_id": "run_test",
            "config": {
                "abr": "FoVRoundRobinABR",
                "predictor": "UniformTileImportancePredictor",
                "bandwidth_predictor": "None",
                "ground_truth_mode": "visible_fraction",
                "ground_truth_visible_threshold": 0.5,
            },
            "events": {
                "enqueued_segments": [
                    {
                        "segment_idx": 0,
                        "duration_s": 2.0,
                        "size_bits": 300.0,
                        "tiles": {
                            "0": {
                                "size_bits": 100.0,
                                "components": {"geom": {"quality_id": "quality_1", "rep_idx": 1, "size_bits": 100.0}},
                            },
                            "1": {
                                "size_bits": 200.0,
                                "components": {"geom": {"quality_id": "quality_1", "rep_idx": 1, "size_bits": 200.0}},
                            },
                        },
                    }
                ],
                "ground_truth_mode": "visible_fraction",
                "ground_truth_params": {"visible_threshold": 0.5, "peak_weight": 0.3},
                "fov_predictions": [
                    {"segment_idx": 0, "tile_visibility": {"0": 0.8, "1": 0.2}},
                ],
                "fov_ground_truth": [
                    {"segment_idx": 0, "tile_visibility": {"0": 0.4, "1": 0.3}},
                ],
                "fov_ground_truth_frames": [
                    {
                        "segment_idx": 0,
                        "target_mode": "visible_fraction",
                        "frames": [
                            {"frame": 10, "tile_visibility": {"0": 1.0, "1": 0.0}},
                            {"frame": 11, "tile_visibility": {"0": 0.0, "1": 1.0}},
                        ],
                    }
                ],
                "rebuffering_events": [],
                "buffer_occupancy_s": [],
            },
        }

        metrics = _compute_run_metrics(run, StaticQoe())

        stats = metrics["fov_prediction"]["run_test"]
        self.assertEqual(metrics["fov_segments"]["run_test"], [])
        self.assertEqual(len(stats["playbackMae"]), 1)
        self.assertEqual(len(stats["visibleFrameRecall"]), 1)
        self.assertEqual(len(stats["peakMissRate"]), 1)


if __name__ == "__main__":
    unittest.main()
