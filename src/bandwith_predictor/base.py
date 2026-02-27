from abc import ABC, abstractmethod
import numpy as np


class BandwidthPredictor(ABC):
    """
    Abstract base class for bandwidth prediction models.
    """

    def __init__(self, network, window_s: float = 5.0, horizon_s: float = 0.0):
        """
        Each predictor operates on a Network instance.

        Args:
            network: The Network object providing bandwidth/time data.
            window_s (float): Window duration (seconds) for past samples.
            horizon_s (float): How far ahead (seconds) to predict.
        """
        self.network = network
        self.window_s = window_s
        self.horizon_s = horizon_s

    @abstractmethod
    def predict(self, time_s: float, window_s: float | None = None, horizon_s: float | None = None) -> float:
        """
        Predict bandwidth (bps) at `time_s + horizon_s` using the last `window_s` seconds.
        """
        pass

    def _get_window_data(self, time_s: float, window_s: float):
        """Helper: extract times and bandwidths within the past window."""
        start_time = max(self.network.times[0], time_s - window_s)
        mask = (self.network.times >= start_time) & (self.network.times <= time_s)
        if not np.any(mask):
            bw = self.network.get_instantaneous_bandwidth(time_s)
            return np.array([time_s]), np.array([bw])
        return self.network.times[mask], self.network.bandwidths[mask]
