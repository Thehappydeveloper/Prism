from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

trace_columns_stub = types.ModuleType("core.trace_columns")
trace_columns_stub.save_trace_column_settings = lambda *args, **kwargs: None
trace_columns_stub.trace_column_config_payload = lambda *args, **kwargs: {}
sys.modules.setdefault("core.trace_columns", trace_columns_stub)

mpd_parser_stub = types.ModuleType("core.mpd_parser")
mpd_parser_stub.V3CMpd = object
sys.modules.setdefault("core.mpd_parser", mpd_parser_stub)

world_transform_stub = types.ModuleType("core.world_transform_registry")
world_transform_stub.WORLD_TRANSFORMS_FILE = Path("/tmp/world_transforms.json")
world_transform_stub.detect_header_transform = lambda *args, **kwargs: None
world_transform_stub.get_world_transform = lambda *args, **kwargs: None
world_transform_stub.list_world_transforms = lambda *args, **kwargs: []
world_transform_stub.normalize_content_dir = lambda *args, **kwargs: None
world_transform_stub.set_world_transform = lambda *args, **kwargs: None
sys.modules.setdefault("core.world_transform_registry", world_transform_stub)

runs_stub = types.ModuleType("server_app.runs")
runs_stub.RUNS = {}
runs_stub.RUNS_LOCK = Lock()
runs_stub.load_run = lambda *args, **kwargs: None
sys.modules.setdefault("server_app.runs", runs_stub)

from server_app.navis_export import build_navis_payload


class NavisExportTests(unittest.TestCase):
    def test_payload_exports_prediction_and_ground_truth_fov_separately(self) -> None:
        payload = build_navis_payload(
            run_id="run_test",
            status={"status": "finished", "started": "start", "finished": "finish"},
            run={
                "config": {
                    "abr": "AbrA",
                    "predictor": "PredictorA",
                    "content_dir": "/tmp/ply",
                    "mpd": "/tmp/content.mpd",
                    "fov_trace": "/tmp/fov.csv",
                    "network_trace": "/tmp/network.csv",
                    "buffer_window_segments": 4,
                    "rtt": 0.05,
                },
                "events": {
                    "enqueued_segments": [{"segment_idx": 0, "duration_s": 2.0, "tiles": {}}],
                    "abr_decisions": [
                        {
                            "segment_idx": 0,
                            "time_s": 0.0,
                            "requested": {"0": {"geom": 1}},
                            "applied": {"0": {"geom": 1}},
                        }
                    ],
                    "fov_predictions": [
                        {"segment_idx": 0, "time_s": 0.0, "tile_visibility": {"0": 0.25, 1: 0.5}}
                    ],
                    "fov_ground_truth": [
                        {"segment_idx": 0, "time_s": 0.0, "source": "embedded_fov_trace", "tile_visibility": {"0": 1.0}}
                    ],
                },
            },
        )

        segment = payload["segments"][0]
        self.assertEqual(segment["fov_prediction"], {"0": 0.25, "1": 0.5})
        self.assertEqual(segment["fov_ground_truth"], {"0": 1.0})
        self.assertEqual(segment["fov_visibility"], segment["fov_prediction"])
        self.assertNotIn("tile_fov_trace_path", payload["traces"])


if __name__ == "__main__":
    unittest.main()
