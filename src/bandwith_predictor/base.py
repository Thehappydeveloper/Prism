from abc import ABC, abstractmethod
import math
import time
import numpy as np


class BandwidthPredictor(ABC):
    """
    Abstract base class for bandwidth prediction models.
    """

    PARAM_METADATA = {
        "network": {
            "hidden": True,
        },
        "window_s": {
            "label": "History window (s)",
            "category": "History",
            "help": "Seconds of past bandwidth samples used for each ABR decision.",
        },
        "horizon_s": {
            "hidden": True,
        },
    }

    def __init__(self, network=None, window_s: float = 5.0, horizon_s: float = 0.0):
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
        self.name = self.__class__.__name__
        self.profile_name = None
        self.last_debug = {}

    def bind_network(self, network) -> "BandwidthPredictor":
        self.network = network
        return self

    @abstractmethod
    def predict(self, time_s: float, window_s: float | None = None, horizon_s: float | None = None) -> float:
        """
        Predict bandwidth (bps) at `time_s + horizon_s` using the last `window_s` seconds.
        """
        pass

    def _get_window_data(self, time_s: float, window_s: float):
        """Helper: extract times and bandwidths within the past window."""
        if self.network is None:
            raise RuntimeError(f"{self.name} is not bound to a Network instance.")
        start_time = max(self.network.times[0], time_s - window_s)
        mask = (self.network.times >= start_time) & (self.network.times <= time_s)
        if not np.any(mask):
            bw = self.network.get_instantaneous_bandwidth(time_s)
            return np.array([time_s]), np.array([bw])
        return self.network.times[mask], self.network.bandwidths[mask]

    def predict_with_debug(self, time_s: float, window_s: float | None = None, horizon_s: float | None = None) -> tuple[float, dict]:
        """Predict and capture timing/debug metadata for logs and charts."""
        started = time.perf_counter()
        measured = self.network.get_instantaneous_bandwidth(time_s) if self.network is not None else 0.0
        self.last_debug = {}
        prediction = self.predict(time_s, window_s, horizon_s)
        if not math.isfinite(prediction) or prediction <= 0:
            prediction = measured
        resolved_window_s = float(window_s if window_s is not None else self.window_s)
        resolved_horizon_s = float(horizon_s if horizon_s is not None else self.horizon_s)
        prediction_time_s = float(time_s + resolved_horizon_s)
        target_measured = self.network.get_instantaneous_bandwidth(prediction_time_s) if self.network is not None else measured
        elapsed_ms = (time.perf_counter() - started) * 1000
        debug = {
            "predictor": self.name,
            "profile": self.profile_name,
            "time_s": float(time_s),
            "prediction_time_s": prediction_time_s,
            "window_s": resolved_window_s,
            "horizon_s": resolved_horizon_s,
            "measured_bandwidth_bps": float(measured),
            "target_measured_bandwidth_bps": float(target_measured),
            "predicted_bandwidth_bps": float(prediction),
            "execution_time_ms": float(elapsed_ms),
        }
        core_keys = set(debug)
        extra_debug = {
            key: value
            for key, value in (getattr(self, "last_debug", {}) or {}).items()
            if key not in core_keys
        }
        debug.update(extra_debug)
        self.last_debug = debug
        return float(prediction), debug
