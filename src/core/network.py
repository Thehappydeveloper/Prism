import pandas as pd
import numpy as np
from pathlib import Path
from bandwith_predictor.base import BandwidthPredictor

class Network:
    """
    Simulate network conditions for VPCC streaming using Lumos5G traces.
    Trace format: tab-separated with two columns:
        Time [s], Bandwidth [Mbps]
    Internally, bandwidth is converted to bps.
    """

    def __init__(self, file_path: str, rtt: float = 50.0, block_size_bits: int = 1_000_000, predictor: BandwidthPredictor = None):
        """
        Initialize the Network object and load the CSV file.
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")

        self.rtt = rtt
        self.block_size_bits = block_size_bits
        self.predictor = predictor

        # Load the CSV immediately
        self.df: pd.DataFrame = pd.read_csv(
            self.file_path,
            sep=",",               # CSVs are comma-separated now
            header=0,              # First row is header
            names=["Time", "Bandwidth"],
        )

        # Convert to float explicitly
        self.times: np.ndarray = self.df["Time"].astype(float).values -1 # s
        self.bandwidths: np.ndarray = self.df["Bandwidth"].astype(float).values * 1e6 # Mbps → bps

    def get_instantaneous_bandwidth(self, time_s: float) -> float:
        """
        Interpolate and return the bandwidth (bps) at a given time (s).
        """
        time_s = float(time_s)
        return float(np.interp(time_s, self.times, self.bandwidths))

    def get_instantaneous_rtt(self, time_s: float) -> float:
        """
        Return constant RTT since Lumos traces don’t provide it.
        """
        return self.rtt

    def get_total_entries(self) -> int:
        return len(self.df)

    def get_all_entries(self) -> pd.DataFrame:
        return self.df.copy()
    
    def predict_bandwidth(self, time_s: float, window_s: float | None, horizon_s: float | None) -> float:
        """
        Predict the bandwidth (bps) at time `time_s + horizon_s`
        using the configured predictor.
        """
        return self.predictor.predict(time_s, window_s, horizon_s)
