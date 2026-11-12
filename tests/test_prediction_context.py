from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.fov import Fov
from core.prediction_context import ObservedFovView
from core.video import Video
from player.player import Player


class FovStub:
    csv_path = Path("trace.csv")
    predictor = None
    start_frame_index = 100
    latency_recorder = None
    motion_limits = {}
    frame_rate_hz = 24.0

    def __init__(self):
        self.df = pd.DataFrame(
            {"HMDPX": [0.0, 1.0, 2.0], "HMDPY": [0.0, 0.0, 0.0], "HMDPZ": [0.0, 0.0, 0.0]},
            index=[0, 1, 2],
        )

    def get_frame(self, frame_number: int):
        try:
            return self.df.loc[int(frame_number)]
        except KeyError:
            return None

    def get_history(self, frame_number: int, count_back: int, include_current: bool = True):
        end = int(frame_number) if include_current else int(frame_number) - 1
        start = end - int(count_back) + 1
        return self.df.loc[(self.df.index >= start) & (self.df.index <= end)].copy()

    def get_frames_range(self, start_frame: int, end_frame: int):
        return self.df.loc[(self.df.index >= int(start_frame)) & (self.df.index <= int(end_frame))].copy()


class PredictionContextTests(unittest.TestCase):
    def test_observed_fov_view_blocks_future_frames(self) -> None:
        view = ObservedFovView(FovStub(), max_observed_frame_idx=101)

        self.assertIsNotNone(view.get_frame(1))
        self.assertIsNone(view.get_frame(2))
        self.assertEqual(list(view.get_history(2, count_back=3).index), [0, 1])
        self.assertEqual(view.frame_rate_hz, 24.0)

    def test_current_observed_frame_idx_returns_latest_played_frame(self) -> None:
        player = Player.__new__(Player)
        player.buffer = SimpleNamespace(total_playback_time_s=2.0)
        player.video = SimpleNamespace(
            start_frame_idx=100,
            total_frames=300,
            total_playback_time=10.0,
            segment_duration_s=2.0,
            num_segments=lambda: 5,
        )

        self.assertEqual(player.current_observed_frame_idx(), 159)

    def test_player_prefers_video_resolved_frame_rate(self) -> None:
        player = Player.__new__(Player)
        player.video = SimpleNamespace(frame_rate_hz=24.0)

        self.assertEqual(player._fps(), 24.0)

    def test_video_frame_rate_uses_ply_count_and_mpd_duration(self) -> None:
        video = Video.__new__(Video)
        video.total_frames = 240
        video.total_playback_time = 10.0
        video.segment_duration_s = 5.0
        video.segment_count = 2

        self.assertEqual(video._resolve_frame_rate_hz(), 24.0)

    def test_fov_normalizes_absolute_frame_index_to_relative_lookup(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.csv"
            pd.DataFrame(
                {
                    "Frame": [100, 101],
                    "Position_X": [1.0, 2.0],
                    "Position_Y": [0.0, 0.0],
                    "Position_Z": [0.0, 0.0],
                    "Euler_X": [0.0, 0.0],
                    "Euler_Y": [0.0, 0.0],
                    "Euler_Z": [0.0, 0.0],
                }
            ).to_csv(path, index=False)

            fov = Fov(str(path), predictor=SimpleNamespace(angle_convention="unity_zxy"), start_frame_index=100)

            self.assertEqual(list(fov.df.index), [0, 1])
            self.assertEqual(float(fov.get_frame(0)["HMDPX"]), 1.0)
            self.assertEqual(float(fov.get_frame(1)["HMDPX"]), 2.0)


if __name__ == "__main__":
    unittest.main()
