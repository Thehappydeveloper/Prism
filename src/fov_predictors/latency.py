from __future__ import annotations

from contextlib import contextmanager
from statistics import median
from time import perf_counter
from typing import Any, Dict, Iterator, List
import math


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = max(0.0, min(float(q), 1.0)) * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _summarize_numeric_metadata(samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    numeric_fields: Dict[str, List[float]] = {}
    for sample in samples:
        for key, value in sample.items():
            if isinstance(value, bool):
                continue
            if not isinstance(value, (int, float)):
                continue
            value_f = float(value)
            if not math.isfinite(value_f):
                continue
            numeric_fields.setdefault(str(key), []).append(value_f)

    summary: Dict[str, Dict[str, float]] = {}
    for key in sorted(numeric_fields.keys()):
        values = numeric_fields[key]
        summary[key] = {
            "count": len(values),
            "min": round(float(min(values)), 3),
            "median": round(float(median(values)), 3),
            "mean": round(float(sum(values) / len(values)), 3),
            "p95": round(float(_percentile(values, 0.95)), 3),
            "max": round(float(max(values)), 3),
        }
    return summary


class NullLatencySession:
    enabled = False

    @contextmanager
    def stage(self, _name: str) -> Iterator[None]:
        yield

    def finish(self, metadata: Dict[str, Any] | None = None, **_kwargs) -> None:
        return None


class PredictorLatencySession:
    enabled = True

    def __init__(self, recorder: "FovPredictorLatencyRecorder"):
        self._recorder = recorder
        self._started_at = perf_counter()
        self._stages_ms: Dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (perf_counter() - started) * 1000.0
            self._stages_ms[name] = self._stages_ms.get(name, 0.0) + elapsed_ms

    def finish(self, metadata: Dict[str, Any] | None = None) -> None:
        total_ms = (perf_counter() - self._started_at) * 1000.0
        self._recorder.record_prediction(total_ms=total_ms, stages_ms=self._stages_ms, metadata=metadata)


class FovPredictorLatencyRecorder:
    def __init__(self, enabled: bool = False, predictor_name: str = ""):
        self.enabled = bool(enabled)
        self.predictor_name = str(predictor_name or "").strip()
        self._total_ms: List[float] = []
        self._stage_samples: Dict[str, List[float]] = {}
        self._metadata_samples: List[Dict[str, Any]] = []

    def start_prediction(self, predictor_name: str | None = None) -> PredictorLatencySession | NullLatencySession:
        if predictor_name and not self.predictor_name:
            self.predictor_name = str(predictor_name)
        if not self.enabled:
            return NullLatencySession()
        return PredictorLatencySession(self)

    def record_prediction(
        self,
        total_ms: float,
        stages_ms: Dict[str, float],
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        self._total_ms.append(float(total_ms))
        for stage_name, value in stages_ms.items():
            self._stage_samples.setdefault(str(stage_name), []).append(float(value))
        if metadata:
            self._metadata_samples.append(dict(metadata))

    def has_data(self) -> bool:
        return bool(self._total_ms)

    def summary(self) -> Dict[str, Any]:
        total_count = len(self._total_ms)
        stage_summaries = []
        for stage_name in sorted(self._stage_samples.keys()):
            samples = self._stage_samples[stage_name]
            stage_summaries.append(
                {
                    "name": stage_name,
                    "count": len(samples),
                    "median_ms": round(float(median(samples)), 3),
                    "mean_ms": round(float(sum(samples) / len(samples)), 3),
                    "p95_ms": round(float(_percentile(samples, 0.95)), 3),
                    "max_ms": round(float(max(samples)), 3),
                }
            )

        total_summary = {
            "count": total_count,
            "median_ms": round(float(median(self._total_ms)), 3) if total_count else 0.0,
            "mean_ms": round(float(sum(self._total_ms) / total_count), 3) if total_count else 0.0,
            "p95_ms": round(float(_percentile(self._total_ms, 0.95)), 3) if total_count else 0.0,
            "max_ms": round(float(max(self._total_ms)), 3) if total_count else 0.0,
        }

        return {
            "enabled": self.enabled,
            "predictor_name": self.predictor_name or None,
            "prediction_count": total_count,
            "total": total_summary,
            "stages": stage_summaries,
            "metadata_summary": _summarize_numeric_metadata(self._metadata_samples),
            "metadata_samples": self._metadata_samples[:32],
        }
