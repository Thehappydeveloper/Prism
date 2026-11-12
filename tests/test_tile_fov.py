from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.event_logger import EventLogger
from core.prediction_context import PredictionRequest
from core.tile_fov import TileFovTrace, reduce_tile_visibility_frames
from fov_predictors.baselines import GroundTruthTileFovPredictor


class TileFovTraceTests(unittest.TestCase):
    def test_loads_mazelab_tile_columns_as_mpd_tile_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "participant_tile_fov.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "frame,tile_0,tile_1,tile_2",
                        "1,0.0,0.5,1.2",
                        "2,1.0,-1.0,0.25",
                    ]
                ),
                encoding="utf-8",
            )

            trace = TileFovTrace(csv_path)

            self.assertEqual(trace.tile_ids, ("0", "1", "2"))
            self.assertEqual(trace.visibility_for_frame(1), {"0": 0.0, "1": 0.5, "2": 1.0})
            self.assertEqual(trace.visibility_for_frame(99, tile_ids=["0", "1"]), {"0": 0.0, "1": 0.0})

    def test_loads_embedded_tile_columns_from_fov_trace_frame_number(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "participant_fov.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "FrameNumber,HMDPX,HMDPY,HMDPZ,HMDRX,HMDRY,HMDRZ,tile_0,tile_1",
                        "1,0,0,0,0,0,0,0.25,1.0",
                        "2,0,0,0,0,0,0,0.5,0.0",
                    ]
                ),
                encoding="utf-8",
            )

            trace = TileFovTrace(csv_path, require_tile_columns=False)

            self.assertEqual(trace.tile_ids, ("0", "1"))
            self.assertEqual(trace.visibility_for_frame(1), {"0": 0.25, "1": 1.0})

    def test_missing_embedded_tile_columns_is_allowed_when_optional(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "participant_fov.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "FrameNumber,HMDPX,HMDPY,HMDPZ,HMDRX,HMDRY,HMDRZ",
                        "1,0,0,0,0,0,0",
                    ]
                ),
                encoding="utf-8",
            )

            trace = TileFovTrace(csv_path, require_tile_columns=False)

            self.assertEqual(trace.tile_ids, ())
            self.assertEqual(trace.visibility_for_frame(1), {})

    def test_averages_visibility_for_segment_frame_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "participant_tile_fov.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "frame,tile_0,tile_1",
                        "1,0.0,1.0",
                        "2,1.0,0.0",
                        "3,0.5,0.5",
                    ]
                ),
                encoding="utf-8",
            )

            trace = TileFovTrace(csv_path)
            visibility = trace.visibility_for_frame_paths(
                [Path("frame_1.ply"), Path("frame_2.ply"), Path("frame_3.ply")],
                tile_ids=["0", "1", "2"],
            )

            self.assertEqual(visibility, {"0": 0.5, "1": 0.5, "2": 0.0})

    def test_logs_per_frame_visibility_for_segment_frame_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "participant_tile_fov.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "frame,tile_0,tile_1",
                        "1,0.0,1.0",
                        "2,1.0,0.0",
                    ]
                ),
                encoding="utf-8",
            )

            trace = TileFovTrace(csv_path)
            frames = trace.visibility_frames_for_frame_paths(
                [Path("frame_1.ply"), Path("frame_2.ply")],
                tile_ids=["0", "1"],
            )

            self.assertEqual(
                frames,
                [
                    {"frame": 1, "tile_visibility": {"0": 0.0, "1": 1.0}},
                    {"frame": 2, "tile_visibility": {"0": 1.0, "1": 0.0}},
                ],
            )

    def test_reduces_per_frame_visibility_with_multiple_modes(self) -> None:
        frames = [
            {"frame": 1, "tile_visibility": {"0": 0.0, "1": 0.9}},
            {"frame": 2, "tile_visibility": {"0": 0.8, "1": 0.0}},
            {"frame": 3, "tile_visibility": {"0": 0.4, "1": 0.0}},
        ]

        self.assertEqual(
            reduce_tile_visibility_frames(frames, tile_ids=["0", "1"], mode="mean_visibility"),
            {"0": 0.4, "1": 0.3},
        )
        self.assertEqual(
            reduce_tile_visibility_frames(frames, tile_ids=["0", "1"], mode="max_visibility"),
            {"0": 0.8, "1": 0.9},
        )
        self.assertEqual(
            reduce_tile_visibility_frames(frames, tile_ids=["0", "1"], mode="visible_fraction", visible_threshold=0.5),
            {"0": 1 / 3, "1": 1 / 3},
        )


class TileFovEventTests(unittest.TestCase):
    def test_ground_truth_visibility_is_logged_separately_from_predictions(self) -> None:
        logger = EventLogger()
        logger.log_decision(
            segment_idx=0,
            time_s=0.0,
            requested_decision={"0": {"geom": 1}},
            applied_decision={"0": {"geom": 1}},
            predicted_fov={"0": 0.1, "1": 0.2},
        )
        logger.log_fov_ground_truth(
            segment_idx=0,
            time_s=0.0,
            tile_visibility={"0": 0.9, "1": 0.8},
            source="embedded_fov_trace",
        )
        logger.log_fov_ground_truth_frames(
            segment_idx=0,
            time_s=0.0,
            frames=[{"frame": 1, "tile_visibility": {"0": 1.0}}],
            source="embedded_fov_trace",
            target_mode="visible_fraction",
            target_visibility={"0": 1.0},
            params={"visible_threshold": 0.01},
        )
        events = logger.to_dict()

        self.assertEqual(events["fov_predictions"][0]["tile_visibility"], {"0": 0.1, "1": 0.2})
        self.assertEqual(events["fov_ground_truth"][0]["tile_visibility"], {"0": 0.9, "1": 0.8})
        self.assertEqual(events["fov_ground_truth"][0]["source"], "embedded_fov_trace")
        self.assertEqual(events["fov_ground_truth"][0]["mode"], "mean_visibility")
        self.assertEqual(events["ground_truth_mode"], "visible_fraction")
        self.assertEqual(events["fov_ground_truth_frames"][0]["target_visibility"], {"0": 1.0})
        self.assertEqual(events["fov_ground_truth_frames"][0]["frames"][0]["frame"], 1)


class GroundTruthTileFovPredictorTests(unittest.TestCase):
    def test_request_oracle_uses_target_frames_and_same_visible_fraction_reducer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "participant_fov.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "FrameNumber,tile_0,tile_1",
                        "10,0.2,0.8",
                        "11,0.8,0.2",
                    ]
                ),
                encoding="utf-8",
            )

            class TileStub:
                def __init__(self, unique_id: str):
                    self.unique_id = unique_id

            request = PredictionRequest(
                target_segment_idx=0,
                geometry_frame_paths=[],
                observed_frame_idx=None,
                prediction_horizon_s=5.0,
                target_segment_duration_s=5.0,
                oracle_target_frame_paths=[Path("frame_10.ply"), Path("frame_11.ply")],
                ground_truth_mode="visible_fraction",
                ground_truth_visible_threshold=0.5,
            )
            fov_stub = type("FovStub", (), {"csv_path": csv_path, "latency_recorder": None})()
            visibility = GroundTruthTileFovPredictor().predict_for_request(
                [TileStub("0"), TileStub("1")], request, fov_stub,
            )
            self.assertEqual(visibility, {"0": 0.5, "1": 0.5})

    def test_predictor_averages_future_embedded_tile_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "participant_fov.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "FrameNumber,HMDPX,HMDPY,HMDPZ,HMDRX,HMDRY,HMDRZ,tile_0,tile_1",
                        "1,0,0,0,0,0,0,0.0,1.0",
                        "2,0,0,0,0,0,0,0.5,0.5",
                        "3,0,0,0,0,0,0,1.0,0.0",
                    ]
                ),
                encoding="utf-8",
            )

            class TileStub:
                def __init__(self, unique_id: str):
                    self.unique_id = unique_id

            fov_stub = type("FovStub", (), {"csv_path": csv_path, "latency_recorder": None})()
            predictor = GroundTruthTileFovPredictor()
            visibility = predictor.predict(
                [TileStub("0"), TileStub("1"), TileStub("2")],
                [Path("frame_1.ply"), Path("frame_2.ply"), Path("frame_3.ply")],
                fov_stub,
            )

            self.assertEqual(visibility, {"0": 0.5, "1": 0.5, "2": 0.0})

    def test_predictor_falls_back_when_no_embedded_tile_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "participant_fov.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "FrameNumber,HMDPX,HMDPY,HMDPZ,HMDRX,HMDRY,HMDRZ",
                        "1,0,0,0,0,0,0",
                    ]
                ),
                encoding="utf-8",
            )

            class TileStub:
                def __init__(self, unique_id: str):
                    self.unique_id = unique_id

            fov_stub = type("FovStub", (), {"csv_path": csv_path, "latency_recorder": None})()
            predictor = GroundTruthTileFovPredictor()
            visibility = predictor.predict([TileStub("0")], [Path("frame_1.ply")], fov_stub)

            self.assertEqual(visibility, {"0": 1.0})


if __name__ == "__main__":
    unittest.main()
