from __future__ import annotations

import sys
import unittest
from importlib.util import find_spec
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.network_profiles import get_network_profile, network_profile_payload


class NetworkProfileTests(unittest.TestCase):
    def test_profile_payload_exposes_standard_profiles(self) -> None:
        payload = {profile["id"]: profile for profile in network_profile_payload()}
        self.assertIn("transatlantic", payload)
        self.assertEqual(payload["transatlantic"]["delay_ms"], 80)
        self.assertEqual(payload["transatlantic"]["jitter_ms"], 10)
        self.assertEqual(payload["transatlantic"]["loss_pct"], 0.1)
        self.assertEqual(payload["transatlantic"]["rate_mbit"], 100)

    @unittest.skipIf(find_spec("numpy") is None or find_spec("pandas") is None, "network runtime dependencies are unavailable")
    def test_network_profile_does_not_require_trace_file(self) -> None:
        from core.network import Network

        profile = get_network_profile("good_wifi")
        network = Network(None, profile=profile, seed=7)

        self.assertAlmostEqual(network.get_instantaneous_bandwidth(0.0), 50_000_000 * 0.999)
        self.assertGreaterEqual(network.get_instantaneous_rtt(0.0), 0.02)
        self.assertLessEqual(network.get_instantaneous_rtt(0.0), 0.04)

    def test_great_wifi_profile_matches_good_wifi_impairments_at_higher_rate(self) -> None:
        good_wifi = get_network_profile("good_wifi")
        great_wifi = get_network_profile("great_wifi")

        self.assertIsNotNone(good_wifi)
        self.assertIsNotNone(great_wifi)
        self.assertEqual(great_wifi.delay_ms, good_wifi.delay_ms)
        self.assertEqual(great_wifi.jitter_ms, good_wifi.jitter_ms)
        self.assertEqual(great_wifi.loss_pct, good_wifi.loss_pct)
        self.assertEqual(great_wifi.rate_mbit, 175)

    @unittest.skipIf(find_spec("numpy") is None or find_spec("pandas") is None, "network runtime dependencies are unavailable")
    def test_classical_trace_mode_still_requires_existing_file(self) -> None:
        from core.network import Network

        with self.assertRaises(FileNotFoundError):
            Network("/tmp/prism_missing_network_trace.csv")


if __name__ == "__main__":
    unittest.main()
