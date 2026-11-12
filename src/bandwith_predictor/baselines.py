from __future__ import annotations

import numpy as np

from bandwith_predictor.base import BandwidthPredictor
from training.adapters import TrainedModelRuntime, bandwidth_history_array


def _clamp_factor(value: float, low: float = 0.0, high: float = 2.0) -> float:
    return max(low, min(high, float(value)))


class LastObservedBandwidthPredictor(BandwidthPredictor):
    """Use the current trace sample as the next ABR bandwidth estimate."""

    def predict(self, time_s: float, window_s: float | None = None, horizon_s: float | None = None) -> float:
        del window_s, horizon_s
        return float(self.network.get_instantaneous_bandwidth(time_s))


class GroundTruthBandwidthPredictor(BandwidthPredictor):
    """Oracle predictor: average the known trace over the next ABR decision interval."""

    def predict(self, time_s: float, window_s: float | None = None, horizon_s: float | None = None) -> float:
        del window_s
        ahead_s = float(horizon_s if horizon_s is not None else self.horizon_s)
        if ahead_s <= 0:
            return float(self.network.get_instantaneous_bandwidth(time_s))

        start = float(time_s)
        end = start + ahead_s
        if hasattr(self.network, "get_average_bandwidth"):
            prediction = float(self.network.get_average_bandwidth(start, end))
        else:
            mask = (self.network.times >= start) & (self.network.times <= end)
            samples = list(self.network.bandwidths[mask])
            samples.append(self.network.get_instantaneous_bandwidth(start))
            samples.append(self.network.get_instantaneous_bandwidth(end))
            prediction = float(np.mean(samples))
        self.last_debug = {
            "actual_average_bandwidth_bps": prediction,
            "target_measured_bandwidth_bps": float(self.network.get_instantaneous_bandwidth(end)),
        }
        return prediction


class MovingAverageBandwidthPredictor(BandwidthPredictor):
    """Arithmetic mean over recent measured bandwidth samples."""

    def predict(self, time_s: float, window_s: float | None = None, horizon_s: float | None = None) -> float:
        del horizon_s
        _, bws = self._get_window_data(time_s, window_s or self.window_s)
        return float(np.mean(bws))


class WeightedMovingAverageBandwidthPredictor(BandwidthPredictor):
    """Linearly weight recent samples more strongly than older samples."""

    def predict(self, time_s: float, window_s: float | None = None, horizon_s: float | None = None) -> float:
        del horizon_s
        _, bws = self._get_window_data(time_s, window_s or self.window_s)
        if len(bws) <= 1:
            return float(bws[-1])
        weights = np.linspace(1.0, float(len(bws)), num=len(bws))
        return float(np.average(bws, weights=weights))


class ExponentialMovingAverageBandwidthPredictor(BandwidthPredictor):
    PARAM_METADATA = {
        **BandwidthPredictor.PARAM_METADATA,
        "alpha": {
            "label": "EMA alpha",
            "category": "Smoothing",
            "help": "Higher values react faster to recent bandwidth changes.",
        },
    }

    def __init__(self, network=None, window_s: float = 5.0, horizon_s: float = 0.0, alpha: float = 0.35):
        super().__init__(network, window_s=window_s, horizon_s=horizon_s)
        self.alpha = _clamp_factor(alpha, 0.01, 1.0)

    def predict(self, time_s: float, window_s: float | None = None, horizon_s: float | None = None) -> float:
        del horizon_s
        _, bws = self._get_window_data(time_s, window_s or self.window_s)
        ema = float(bws[0])
        for bw in bws[1:]:
            ema = self.alpha * float(bw) + (1.0 - self.alpha) * ema
        return float(ema)


class RollingMedianBandwidthPredictor(BandwidthPredictor):
    """Median over a recent window to reject short spikes."""

    def predict(self, time_s: float, window_s: float | None = None, horizon_s: float | None = None) -> float:
        del horizon_s
        _, bws = self._get_window_data(time_s, window_s or self.window_s)
        return float(np.median(bws))


class LinearTrendExtrapolationBandwidthPredictor(BandwidthPredictor):
    PARAM_METADATA = {
        **BandwidthPredictor.PARAM_METADATA,
        "trend_window_s": {
            "label": "Trend window (s)",
            "category": "Trend",
            "help": "Recent seconds used to estimate a linear bandwidth trend.",
        },
        "safety_margin": {
            "label": "Safety margin",
            "category": "Decision bias",
            "help": "Multiplier applied after trend extrapolation.",
        },
    }

    def __init__(
        self,
        network=None,
        window_s: float = 5.0,
        horizon_s: float = 0.0,
        trend_window_s: float = 5.0,
        safety_margin: float = 1.0,
    ):
        super().__init__(network, window_s=window_s, horizon_s=horizon_s)
        self.trend_window_s = float(trend_window_s)
        self.safety_margin = _clamp_factor(safety_margin, 0.05, 2.0)

    def predict(self, time_s: float, window_s: float | None = None, horizon_s: float | None = None) -> float:
        history_s = window_s or self.trend_window_s or self.window_s
        ahead_s = horizon_s if horizon_s is not None else self.horizon_s
        times, bws = self._get_window_data(time_s, history_s)
        if len(times) < 2:
            return float(self.network.get_instantaneous_bandwidth(time_s)) * self.safety_margin
        x = times - times[0]
        slope, intercept = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, bws, rcond=None)[0]
        target_x = (time_s + ahead_s) - times[0]
        return float(max(0.0, slope * target_x + intercept) * self.safety_margin)


class HybridEmaTrendBandwidthPredictor(BandwidthPredictor):
    PARAM_METADATA = {
        **BandwidthPredictor.PARAM_METADATA,
        "alpha": ExponentialMovingAverageBandwidthPredictor.PARAM_METADATA["alpha"],
        "trend_window_s": LinearTrendExtrapolationBandwidthPredictor.PARAM_METADATA["trend_window_s"],
        "smoothing_strength": {
            "label": "Trend blend",
            "category": "Smoothing",
            "help": "0 uses EMA only, 1 uses trend only.",
        },
        "safety_margin": LinearTrendExtrapolationBandwidthPredictor.PARAM_METADATA["safety_margin"],
    }

    def __init__(
        self,
        network=None,
        window_s: float = 5.0,
        horizon_s: float = 0.0,
        alpha: float = 0.35,
        trend_window_s: float = 5.0,
        smoothing_strength: float = 0.35,
        safety_margin: float = 1.0,
    ):
        super().__init__(network, window_s=window_s, horizon_s=horizon_s)
        self.alpha = _clamp_factor(alpha, 0.01, 1.0)
        self.trend_window_s = float(trend_window_s)
        self.smoothing_strength = _clamp_factor(smoothing_strength, 0.0, 1.0)
        self.safety_margin = _clamp_factor(safety_margin, 0.05, 2.0)

    def predict(self, time_s: float, window_s: float | None = None, horizon_s: float | None = None) -> float:
        ema = ExponentialMovingAverageBandwidthPredictor(
            self.network,
            window_s=window_s or self.window_s,
            horizon_s=horizon_s if horizon_s is not None else self.horizon_s,
            alpha=self.alpha,
        ).predict(time_s)
        trend = LinearTrendExtrapolationBandwidthPredictor(
            self.network,
            window_s=window_s or self.window_s,
            horizon_s=horizon_s if horizon_s is not None else self.horizon_s,
            trend_window_s=self.trend_window_s,
            safety_margin=1.0,
        ).predict(time_s)
        blend = (1.0 - self.smoothing_strength) * ema + self.smoothing_strength * trend
        return float(max(0.0, blend) * self.safety_margin)


class ConservativeBandwidthPredictor(HybridEmaTrendBandwidthPredictor):
    PARAM_METADATA = {
        **HybridEmaTrendBandwidthPredictor.PARAM_METADATA,
        "safety_margin": {
            "label": "Safety margin",
            "category": "Decision bias",
            "help": "Lower than 1.0 intentionally underestimates bandwidth to reduce stalls.",
        },
    }

    def __init__(
        self,
        network=None,
        window_s: float = 5.0,
        horizon_s: float = 0.0,
        trend_window_s: float = 5.0,
        safety_margin: float = 0.8,
        alpha: float = 0.3,
        smoothing_strength: float = 0.2,
    ):
        super().__init__(
            network,
            window_s=window_s,
            horizon_s=horizon_s,
            trend_window_s=trend_window_s,
            safety_margin=safety_margin,
            alpha=alpha,
            smoothing_strength=smoothing_strength,
        )


class AggressiveBandwidthPredictor(HybridEmaTrendBandwidthPredictor):
    PARAM_METADATA = {
        **HybridEmaTrendBandwidthPredictor.PARAM_METADATA,
        "aggressiveness_factor": {
            "label": "Aggressiveness",
            "category": "Decision bias",
            "help": "Multiplier above the smoothed estimate to improve quality when risk is acceptable.",
        },
    }

    def __init__(
        self,
        network=None,
        window_s: float = 5.0,
        horizon_s: float = 0.0,
        trend_window_s: float = 5.0,
        aggressiveness_factor: float = 1.15,
        alpha: float = 0.45,
        smoothing_strength: float = 0.45,
    ):
        super().__init__(
            network,
            window_s=window_s,
            horizon_s=horizon_s,
            trend_window_s=trend_window_s,
            alpha=alpha,
            smoothing_strength=smoothing_strength,
            safety_margin=1.0,
        )
        self.aggressiveness_factor = _clamp_factor(aggressiveness_factor, 0.05, 2.0)

    def predict(self, time_s: float, window_s: float | None = None, horizon_s: float | None = None) -> float:
        return float(super().predict(time_s, window_s, horizon_s) * self.aggressiveness_factor)


# Backward-compatible names from the earlier prototype.
AveragingPredictor = MovingAverageBandwidthPredictor
ExponentialMovingAveragePredictor = ExponentialMovingAverageBandwidthPredictor
LinearRegressionPredictor = LinearTrendExtrapolationBandwidthPredictor


class HarmonicMeanPredictor(BandwidthPredictor):
    """Retained as a conservative legacy option."""

    def predict(self, time_s: float, window_s: float | None = None, horizon_s: float | None = None) -> float:
        del horizon_s
        times, bws = self._get_window_data(time_s, window_s or self.window_s)
        positive = bws[bws > 0]
        self.last_debug = {
            "sample_count": int(len(positive)),
            "window_start_s": float(times[0]) if len(times) else float(time_s),
            "window_end_s": float(times[-1]) if len(times) else float(time_s),
            "window_min_bandwidth_bps": float(np.min(positive)) if len(positive) else 0.0,
            "window_max_bandwidth_bps": float(np.max(positive)) if len(positive) else 0.0,
        }
        bws = positive
        if len(bws) == 0:
            return float(self.network.get_instantaneous_bandwidth(time_s))
        return float(len(bws) / np.sum(1.0 / bws))


class TrainedBandwidthPredictor(BandwidthPredictor):
    PARAM_METADATA = {
        **BandwidthPredictor.PARAM_METADATA,
        "model_id": {
            "label": "Exported model id",
            "category": "Trained model",
            "help": "Model id from the Training module export list.",
        },
    }

    def __init__(self, model_id: str = "", device: str = "auto", **kwargs):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.runtime = TrainedModelRuntime(model_id=model_id, device=device)

    def predict(self, time_s: float, window_s: float | None = None, horizon_s: float | None = None) -> float:
        del horizon_s
        history = bandwidth_history_array(self.runtime, self.network, time_s, window_s=window_s)
        mbps = float(self.runtime.predict_array(history)[0])
        self.last_debug = {"model_id": self.model_id}
        return max(mbps * 1e6, 1e-6)
