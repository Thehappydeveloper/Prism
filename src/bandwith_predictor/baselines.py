import numpy as np
from bandwidth_predictor.base import BandwidthPredictor

# ======================================================
# 1️⃣ Averaging Predictor
# ======================================================
class AveragingPredictor(BandwidthPredictor):
    """
    Predicts bandwidth by averaging the past window and
    assuming the same average continues for the next horizon.
    """

    def predict(self, time_s: float, window_s: float | None = None, horizon_s: float | None = None) -> float:
        window_s = window_s or self.window_s
        horizon_s = horizon_s or self.horizon_s

        # Use window data up to time_s
        times, bws = self._get_window_data(time_s, window_s)
        avg_bw = np.mean(bws)

        # Simulate continuation over the horizon
        future_times = np.linspace(time_s, time_s + horizon_s, num=5)
        future_bws = np.full_like(future_times, avg_bw)
        return float(np.mean(future_bws))  # equivalent to avg_bw, but clearer intent


# ======================================================
# 2️⃣ Exponential Moving Average Predictor
# ======================================================
class ExponentialMovingAveragePredictor(BandwidthPredictor):
    """
    Exponential moving average (EMA) predictor that assumes the
    smoothed trend continues over the prediction horizon.
    """

    def __init__(self, network, alpha: float = 0.3, **kwargs):
        super().__init__(network, **kwargs)
        self.alpha = alpha
        self.last_prediction = None

    def predict(self, time_s: float, window_s: float | None = None, horizon_s: float | None = None) -> float:
        window_s = window_s or self.window_s
        horizon_s = horizon_s or self.horizon_s

        times, bws = self._get_window_data(time_s, window_s)
        ema = bws[0]
        for bw in bws[1:]:
            ema = self.alpha * bw + (1 - self.alpha) * ema
        self.last_prediction = ema

        # Assume EMA value persists into the horizon
        return float(ema)


# ======================================================
# 3️⃣ Harmonic Mean Predictor
# ======================================================
class HarmonicMeanPredictor(BandwidthPredictor):
    """
    Harmonic mean predictor — assumes the harmonic mean continues
    over the horizon (useful for smoothing high spikes).
    """

    def predict(self, time_s: float, window_s: float | None = None, horizon_s: float | None = None) -> float:
        window_s = window_s or self.window_s
        horizon_s = horizon_s or self.horizon_s

        times, bws = self._get_window_data(time_s, window_s)
        bws = bws[bws > 0]
        if len(bws) == 0:
            return self.network.get_instantaneous_bandwidth(time_s)

        hmean = len(bws) / np.sum(1.0 / bws)
        return float(hmean)  # assumed constant over horizon


# ======================================================
# 4️⃣ Linear Regression Predictor
# ======================================================
class LinearRegressionPredictor(BandwidthPredictor):
    """
    Linear regression predictor that fits a trend over the past window
    and extrapolates it over the given horizon.
    """

    def predict(self, time_s: float, window_s: float | None = None, horizon_s: float | None = None) -> float:
        window_s = window_s or self.window_s
        horizon_s = horizon_s or self.horizon_s

        times, bws = self._get_window_data(time_s, window_s)
        if len(times) < 2:
            # fallback to last observed bandwidth
            return self.network.get_instantaneous_bandwidth(time_s)

        # Fit linear regression: bws ≈ slope * t + intercept
        A = np.vstack([times - times[0], np.ones(len(times))]).T
        slope, intercept = np.linalg.lstsq(A, bws, rcond=None)[0]

        # Predict at time_s + horizon_s
        predicted = slope * (times[-1] + horizon_s - times[0]) + intercept
        return float(max(predicted, 0.0))
