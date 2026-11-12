import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from fov_predictors.baselines import LinearExtrapolationPredictor, LinearPoseExtrapolationPredictor


class ObservedFovStub:
    start_frame_index = 0

    def __init__(self, frame_rate_hz: float = 30.0):
        self.frame_rate_hz = frame_rate_hz

    def get_frame(self, frame_number):
        return int(frame_number)


class LinearExtrapolationPredictorTests(unittest.TestCase):
    def test_default_window_is_two_seconds_at_trace_rate(self):
        predictor = LinearExtrapolationPredictor()

        self.assertEqual(predictor._lookback_frame_count(), 60)
        self.assertEqual(
            predictor._get_recent_observed_frames(100, ObservedFovStub()),
            list(range(40, 101)),
        )

    def test_legacy_saved_frame_lookback_is_not_reinterpreted_as_seconds(self):
        predictor = LinearExtrapolationPredictor(lookback=3)

        self.assertEqual(predictor._lookback_frame_count(), 3)
        self.assertTrue(predictor._history_metadata()["legacy_frame_lookback"])

    def test_seconds_window_uses_content_frame_rate(self):
        predictor = LinearExtrapolationPredictor(lookback_s=2.0)
        observed_fov = ObservedFovStub(frame_rate_hz=24.0)

        self.assertEqual(predictor._lookback_frame_count(observed_fov), 48)
        self.assertEqual(
            predictor._get_recent_observed_frames(100, observed_fov),
            list(range(52, 101)),
        )

    def test_pose_variant_uses_same_two_second_window_and_extrapolates_translation(self):
        class PositionSlice:
            def __init__(self, values): self.values = values
            def to_numpy(self, dtype=None): return np.asarray(self.values, dtype=dtype)

        class Frame:
            def __init__(self, position): self.position = position
            def __getitem__(self, key):
                if key == ['HMDPX', 'HMDPY', 'HMDPZ']:
                    return PositionSlice(self.position)
                raise KeyError(key)

        predictor = LinearPoseExtrapolationPredictor()
        predictor._forward_from_frame = lambda _frame: np.asarray([0.0, 0.0, 1.0])
        position, forward = predictor._extrapolate_forward([Frame([1, 2, 3]), Frame([4, 6, 8])])

        self.assertEqual(predictor._lookback_frame_count(), 60)
        np.testing.assert_allclose(position, [7, 10, 13])
        np.testing.assert_allclose(forward, [0, 0, 1])


if __name__ == "__main__":
    unittest.main()
