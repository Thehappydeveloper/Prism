import unittest

from bandwith_predictor.base import BandwidthPredictor


class FakeNetwork:
    def get_instantaneous_bandwidth(self, time_s: float) -> float:
        return float(time_s) * 100.0


class SequenceBandwidthPredictor(BandwidthPredictor):
    def predict(self, time_s: float, window_s: float | None = None, horizon_s: float | None = None) -> float:
        del window_s, horizon_s
        return float(time_s) * 10.0


class ExtraDebugBandwidthPredictor(BandwidthPredictor):
    def predict(self, time_s: float, window_s: float | None = None, horizon_s: float | None = None) -> float:
        del window_s, horizon_s
        self.last_debug = {
            "sample_count": int(time_s),
            "predicted_bandwidth_bps": 1.0,
        }
        return float(time_s) * 10.0


class BandwidthPredictorDebugTests(unittest.TestCase):
    def test_debug_metadata_is_fresh_for_each_prediction(self) -> None:
        predictor = SequenceBandwidthPredictor(network=FakeNetwork(), window_s=3.0, horizon_s=1.0)

        _, first = predictor.predict_with_debug(1.0, window_s=None, horizon_s=None)
        _, second = predictor.predict_with_debug(2.0, window_s=None, horizon_s=4.0)

        self.assertEqual(first["time_s"], 1.0)
        self.assertEqual(first["measured_bandwidth_bps"], 100.0)
        self.assertEqual(first["predicted_bandwidth_bps"], 10.0)
        self.assertEqual(first["prediction_time_s"], 2.0)

        self.assertEqual(second["time_s"], 2.0)
        self.assertEqual(second["measured_bandwidth_bps"], 200.0)
        self.assertEqual(second["predicted_bandwidth_bps"], 20.0)
        self.assertEqual(second["prediction_time_s"], 6.0)

    def test_predictor_extra_debug_cannot_overwrite_core_fields(self) -> None:
        predictor = ExtraDebugBandwidthPredictor(network=FakeNetwork())

        _, debug = predictor.predict_with_debug(3.0, window_s=2.0, horizon_s=5.0)

        self.assertEqual(debug["sample_count"], 3)
        self.assertEqual(debug["predicted_bandwidth_bps"], 30.0)
        self.assertEqual(debug["prediction_time_s"], 8.0)


if __name__ == "__main__":
    unittest.main()
