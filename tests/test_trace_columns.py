import unittest

import pandas as pd

from core.trace_columns import normalize_fov_dataframe, normalize_network_dataframe


class TraceColumnMappingTests(unittest.TestCase):
    def test_normalizes_external_fov_headers_to_prism_columns(self) -> None:
        df = pd.DataFrame(
            {
                "frame": [1],
                "Position_X": [0.1],
                "Position_Y": [0.2],
                "Position_Z": [0.3],
                "Euler_X": [1.0],
                "Euler_Y": [2.0],
                "Euler_Z": [3.0],
            }
        )

        normalized = normalize_fov_dataframe(df, source="trace.csv")

        self.assertIn("FrameNumber", normalized.columns)
        self.assertIn("HMDPX", normalized.columns)
        self.assertIn("HMDRZ", normalized.columns)

    def test_normalizes_network_headers_case_insensitively(self) -> None:
        df = pd.DataFrame({"time_s": [1.0], "bandwidth_mbps": [42.0]})

        normalized = normalize_network_dataframe(df, source="network.csv")

        self.assertEqual(["Time", "Bandwidth"], list(normalized.columns))

    def test_missing_field_error_reports_headers_and_attempted_aliases(self) -> None:
        df = pd.DataFrame({"Frame": [1], "Position_X": [0.1]})

        with self.assertRaisesRegex(ValueError, "Missing required field.*Position Y.*Detected headers: Frame, Position_X"):
            normalize_fov_dataframe(df, source="bad.csv")


if __name__ == "__main__":
    unittest.main()
