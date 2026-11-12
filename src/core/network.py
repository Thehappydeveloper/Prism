import numpy as np
import pandas as pd
from pathlib import Path
from bandwith_predictor.base import BandwidthPredictor
from core.network_profiles import NetworkProfile, UNLIMITED_RATE_BPS
from core.trace_columns import normalize_network_dataframe

class Network:
    """
    Simulate network conditions for VPCC streaming using Lumos5G traces.
    Trace format: tab-separated with two columns:
        Time [s], Bandwidth [Mbps]
    Internally, bandwidth is converted to bps.
    """

    def __init__(
        self,
        file_path: str | Path | None,
        rtt: float = 50.0,
        block_size_bits: int = 1_000_000,
        predictor: BandwidthPredictor = None,
        profile: NetworkProfile | None = None,
        seed: int | None = None,
    ):
        """
        Initialize the Network object and load the CSV file.
        """
        self.profile = profile
        self.file_path = Path(file_path) if file_path else None
        if self.profile is None and (self.file_path is None or not self.file_path.exists()):
            raise FileNotFoundError(f"CSV file not found: {self.file_path}")
        self.rtt = rtt
        self.block_size_bits = block_size_bits
        self.predictor = predictor
        self.last_prediction: dict | None = None
        self._rng = np.random.default_rng(seed if seed is not None else 0)

        if self.profile is not None:
            rate_bps = self._profile_rate_bps()
            self.df = pd.DataFrame({"Time": [0.0, 3600.0], "Bandwidth": [rate_bps / 1e6, rate_bps / 1e6]})
            self.times = self.df["Time"].astype(float).values
            self.bandwidths = self.df["Bandwidth"].astype(float).values * 1e6
        else:
            # Load the CSV immediately
            self.df: pd.DataFrame = normalize_network_dataframe(pd.read_csv(self.file_path), source=self.file_path)

            # Convert to float explicitly
            self.times: np.ndarray = self.df["Time"].astype(float).values -1 # s
            self.bandwidths: np.ndarray = self.df["Bandwidth"].astype(float).values * 1e6 # Mbps → bps
        if self.predictor is not None:
            self.predictor.bind_network(self)

    def _profile_rate_bps(self) -> float:
        if self.profile is None:
            return UNLIMITED_RATE_BPS
        raw = UNLIMITED_RATE_BPS if self.profile.rate_mbit is None else float(self.profile.rate_mbit) * 1e6
        loss_factor = max(0.0, 1.0 - (float(self.profile.loss_pct) / 100.0))
        return max(1e-6, raw * loss_factor)

    def get_instantaneous_bandwidth(self, time_s: float) -> float:
        """
        Interpolate and return the bandwidth (bps) at a given time (s).
        """
        time_s = float(time_s)
        return float(np.interp(time_s, self.times, self.bandwidths))

    def get_average_bandwidth(self, start_time_s: float, end_time_s: float) -> float:
        """Return the time-weighted average interpolated bandwidth over an interval."""
        start = float(start_time_s)
        end = float(end_time_s)
        if end <= start:
            return self.get_instantaneous_bandwidth(start)
        interior = self.times[(self.times > start) & (self.times < end)]
        sample_times = np.concatenate(([start], interior, [end])).astype(float)
        sample_bandwidths = np.interp(sample_times, self.times, self.bandwidths)
        duration = max(end - start, 1e-9)
        return float(np.trapz(sample_bandwidths, sample_times) / duration)

    def get_instantaneous_rtt(self, time_s: float) -> float:
        """
        Return constant RTT since Lumos traces don’t provide it.
        """
        if self.profile is not None:
            base_s = max(0.0, float(self.profile.delay_ms) / 1000.0)
            jitter_s = max(0.0, float(self.profile.jitter_ms) / 1000.0)
            if jitter_s <= 0:
                return base_s
            if str(self.profile.delay_distribution).lower() == "normal":
                delta = float(self._rng.normal(0.0, jitter_s / 2.0))
            else:
                delta = float(self._rng.uniform(-jitter_s, jitter_s))
            return max(0.0, base_s + delta)
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
        if self.predictor is None:
            measured = self.get_instantaneous_bandwidth(time_s)
            self.last_prediction = {
                "predictor": "None",
                "profile": None,
                "time_s": float(time_s),
                "window_s": float(window_s or 0.0),
                "horizon_s": float(horizon_s or 0.0),
                "measured_bandwidth_bps": float(measured),
                "predicted_bandwidth_bps": float(measured),
                "execution_time_ms": 0.0,
            }
            return measured
        prediction, debug = self.predictor.predict_with_debug(time_s, window_s, horizon_s)
        self.last_prediction = debug
        return prediction

    def predict_bandwidth_for_decision(
        self,
        time_s: float,
        segment_duration_s: float | None = None,
        decision_interval_s: float | None = None,
        download_duration_s: float | None = None,
        window_s: float | None = None,
    ) -> float:
        """
        Estimate bandwidth for the next ABR decision.

        The horizon is derived from the playback pipeline rather than exposed as
        an arbitrary user parameter. Segment duration is the natural default,
        with decision interval or download duration used when available.
        """
        candidates = [download_duration_s, decision_interval_s, segment_duration_s]
        horizon = next((float(v) for v in candidates if v is not None and float(v) > 0), 0.0)
        history = window_s
        if history is None and self.predictor is not None:
            history = getattr(self.predictor, "window_s", None)
        return self.predict_bandwidth(time_s, history, horizon)
