from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .base import QoEFunction, UiField


@dataclass
class QoEWeights:
    """Weights controlling the QoE heuristic."""

    component_weights: Dict[str, float] = field(
        default_factory=lambda: {"attr": 0.35, "geom": 0.35, "occp": 0.2, "atlas": 0.0}
    )
    tile_saliency: Dict[str, float] = field(default_factory=dict)
    fov_saliency: float = 1.25
    temporal_penalty: float = 0.2
    stall_per_second: float = 0.15
    stall_fixed: float = 0.05
    default_component: float = 0.1

    def as_params(self) -> Dict[str, float]:
        params = dict(self.component_weights)
        params.update(
            {
                "default_component": self.default_component,
                "fov_saliency": self.fov_saliency,
                "temporal_penalty": self.temporal_penalty,
                "stall_per_second": self.stall_per_second,
                "stall_fixed": self.stall_fixed,
            }
        )
        return params


def _parse_quality_id(quality_id: Optional[str]) -> Optional[int]:
    if not quality_id:
        return None
    q = quality_id.lower().replace("quality_", "")
    if q.startswith("q"):
        q = q[1:]
    try:
        return int(q)
    except ValueError:
        return None


def _max_component_levels(events: Mapping) -> Dict[str, int]:
    max_levels: Dict[str, int] = {}
    for seg in events.get("enqueued_segments", []) or []:
        for tile in (seg.get("tiles") or {}).values():
            for comp_name, comp in (tile.get("components") or {}).items():
                if comp_name == "atlas":
                    continue
                lvl = _parse_quality_id(comp.get("quality_id"))
                if lvl is None:
                    lvl = comp.get("rep_idx", 0)
                max_levels[comp_name] = max(max_levels.get(comp_name, 0), int(lvl))
    return max_levels


def _stall_overlap_and_count(
    seg_start: float,
    seg_end: float,
    stalls: Sequence[Mapping],
) -> Tuple[float, int]:
    """Return (total_overlap_seconds, number_of_overlapping_stalls)."""
    overlap = 0.0
    count = 0
    for stall in stalls:
        st = float(stall.get("time_s", 0))
        et = st + float(stall.get("duration_s", 0))
        ov = max(0.0, min(seg_end, et) - max(seg_start, st))
        if ov > 0.0:
            overlap += ov
            count += 1
    return overlap, count


class WeightedQoEHeuristic(QoEFunction):
    """Composite QoE model for VPCC playback with harmonic component aggregation."""

    id = "WeightedQoEHeuristic"
    label = "Weighted harmonic QoE"
    equation = "QoE = H(component utility) * saliency - smoothness_penalty - stall_penalty"
    description = (
        "Concave utility per component, harmonic aggregation across components, FoV saliency, "
        "asymmetric smoothness, and saturating stall penalties."
    )
    implementation = "weighted"

    def __init__(self, weights: Optional[QoEWeights] = None, overrides: Mapping[str, Any] | None = None):
        self.weights = weights or QoEWeights()

        # Tunables for the improved heuristic (kept internal to avoid API breakage)
        self._alpha_per_component: Dict[str, float] = {"attr": 3.0, "geom": 4.0, "occp": 2.5}
        self._alpha_default: float = 3.0
        self._temporal_smoothing_rho: float = 0.5  # between 0 (no smoothing) and 1
        self._lambda_down_factor: float = 1.0      # scales temporal_penalty for downswings
        self._lambda_up_factor: float = 0.3        # upswings are less annoying
        self._stall_saturation_k: float = 1.5      # higher = faster saturation
        self._stall_count_penalty: float = 0.03    # per-stall additive penalty
        self._utility_epsilon: float = 1e-3        # floor for harmonic mean

        self.apply_overrides(overrides)

    @classmethod
    def ui_fields(cls) -> Sequence[UiField]:
        return [
            UiField("attr", "Attr weight", 0.35, min=0.0, step=0.01, help="Relative importance of attribute quality"),
            UiField("geom", "Geom weight", 0.35, min=0.0, step=0.01, help="Relative importance of geometry quality"),
            UiField("occp", "Occp weight", 0.2, min=0.0, step=0.01, help="Relative importance of occupancy quality"),
            UiField("atlas", "Atlas weight", 0.0, min=0.0, step=0.01, help="Atlas is ignored in QoE; keep at 0"),
            UiField("default_component", "Default component weight", 0.1, min=0.0, step=0.01, help="Fallback for unknown components"),
            UiField("fov_saliency", "FoV saliency", 1.25, min=0.0, step=0.05, help="Extra weight for tiles inside FoV"),
            UiField("temporal_penalty", "Temporal penalty", 0.2, min=0.0, step=0.01, help="Smoothness penalty scale"),
            UiField("stall_per_second", "Stall /s penalty", 0.15, min=0.0, step=0.01, help="Per-second stall penalty (saturating)"),
            UiField("stall_fixed", "Stall fixed penalty", 0.05, min=0.0, step=0.01, help="Penalty added when a stall overlaps a segment"),
        ]

    def apply_overrides(self, overrides: Mapping[str, Any] | None = None) -> None:
        if not overrides:
            return
        comp = self.weights.component_weights
        for key in ["attr", "geom", "occp", "atlas"]:
            if key in overrides:
                try:
                    comp[key] = float(overrides[key])
                except (TypeError, ValueError):
                    continue
        if "default_component" in overrides or "default" in overrides:
            try:
                self.weights.default_component = float(overrides.get("default_component", overrides.get("default")))
            except (TypeError, ValueError):
                pass
        for attr in ["fov_saliency", "temporal_penalty", "stall_per_second", "stall_fixed"]:
            if attr in overrides:
                try:
                    setattr(self.weights, attr, float(overrides[attr]))
                except (TypeError, ValueError):
                    continue

    def _component_weight(self, name: str) -> float:
        return self.weights.component_weights.get(name, self.weights.default_component)

    def _tile_weight(self, tile_id: str, viewport_tiles: Optional[Iterable[str]]) -> float:
        base = self.weights.tile_saliency.get(tile_id, 1.0)
        if viewport_tiles and tile_id in viewport_tiles:
            base *= self.weights.fov_saliency
        return base

    def _component_utility(self, lvl: int, max_lvl: int, comp_name: str) -> float:
        """Concave per-component utility in [0,1] using a log-based mapping."""
        if max_lvl <= 0:
            return 0.0
        x = max(0.0, min(1.0, lvl / float(max_lvl)))
        if x <= 0.0:
            return 0.0

        alpha = self._alpha_per_component.get(comp_name, self._alpha_default)
        num = math.log(1.0 + alpha * x)
        den = math.log(1.0 + alpha)
        return num / den if den > 0.0 else x

    def _segment_quality(
        self,
        segment: Mapping,
        max_levels: Mapping[str, int],
        viewport_tiles: Optional[Iterable[str]] = None,
    ) -> float:
        """Compute segment base quality with improved per-component and per-tile aggregation."""
        weighted_sum = 0.0
        weight_total = 0.0

        for tile_id, tile in (segment.get("tiles") or {}).items():
            tile_w = self._tile_weight(tile_id, viewport_tiles)
            sum_w = 0.0
            denom = 0.0

            for comp_name, comp in (tile.get("components") or {}).items():
                if comp_name == "atlas":
                    continue

                lvl = _parse_quality_id(comp.get("quality_id"))
                if lvl is None:
                    lvl = comp.get("rep_idx", 0) or 0

                max_lvl = max_levels.get(comp_name) or max(lvl, 1)
                u = self._component_utility(lvl, max_lvl, comp_name)
                cw = self._component_weight(comp_name)

                if cw <= 0.0:
                    continue

                sum_w += cw
                denom += cw / max(u, self._utility_epsilon)

            if sum_w > 0.0 and denom > 0.0:
                tile_quality = sum_w / denom  # weighted harmonic mean
                tile_quality = max(0.0, min(1.0, tile_quality))
            else:
                tile_quality = 0.0

            weighted_sum += tile_quality * tile_w
            weight_total += tile_w

        return weighted_sum / weight_total if weight_total > 0.0 else 0.0

    def compute_time_series(
        self,
        events: Mapping,
        viewport_tiles: Optional[Iterable[str]] = None,
    ) -> List[Tuple[float, float]]:
        """Return QoE time series as ``[(time_s, score), ...]`` with improved heuristic."""
        max_levels = _max_component_levels(events)
        stalls = events.get("rebuffering_events") or []
        segments = sorted(events.get("enqueued_segments") or [], key=lambda s: s.get("time_s", 0))

        results: List[Tuple[float, float]] = []
        prev_base_quality: Optional[float] = None
        prev_eff_quality: Optional[float] = None

        for seg in segments:
            seg_start = float(seg.get("time_s", 0))
            seg_end = seg_start + float(seg.get("duration_s", 0))

            base_quality = self._segment_quality(seg, max_levels, viewport_tiles)

            # --- Temporal smoothness (asymmetric, nonlinear) ---
            penalty = 0.0

            if prev_eff_quality is None:
                eff_quality = base_quality
            else:
                rho = self._temporal_smoothing_rho
                eff_quality = (1.0 - rho) * base_quality + rho * prev_eff_quality

            if prev_base_quality is not None:
                delta = base_quality - prev_base_quality
                down = max(0.0, -delta)
                up = max(0.0, delta)

                lambda_down = self.weights.temporal_penalty * self._lambda_down_factor
                lambda_up = self.weights.temporal_penalty * self._lambda_up_factor

                penalty += lambda_down * (down ** 2) + lambda_up * (up ** 2)

            # --- Stall penalty: saturating with small count penalty ---
            stall_overlap, stall_count = _stall_overlap_and_count(seg_start, seg_end, stalls)

            if stall_overlap > 0.0:
                dur_pen = self.weights.stall_per_second * (1.0 - math.exp(-self._stall_saturation_k * stall_overlap))
                count_pen = self._stall_count_penalty * stall_count
                penalty += self.weights.stall_fixed + dur_pen + count_pen

            score = max(0.0, min(1.0, base_quality - penalty))
            results.append((seg_start, score))

            prev_base_quality = base_quality
            prev_eff_quality = eff_quality

        return results


class LinearQoE(QoEFunction):
    """Simpler QoE: arithmetic quality with linear smoothness + stall penalties."""

    id = "LinearQoE"
    label = "Linear QoE"
    equation = "QoE = quality_avg - temporal_penalty*|delta_q| - stall_fixed - stall_per_second*stall_overlap"
    description = "Lightweight alternative useful for quick tuning; penalties are linear."
    implementation = "linear"

    def __init__(self, weights: Optional[QoEWeights] = None, overrides: Mapping[str, Any] | None = None):
        self.weights = weights or QoEWeights()
        self.apply_overrides(overrides)

    @classmethod
    def ui_fields(cls) -> Sequence[UiField]:
        return [
            UiField("fov_saliency", "FoV saliency", 1.0, min=0.0, step=0.05, help="Emphasis on viewport tiles"),
            UiField("temporal_penalty", "Temporal penalty", 0.12, min=0.0, step=0.01, help="Smoothness penalty scale"),
            UiField("stall_per_second", "Stall /s penalty", 0.12, min=0.0, step=0.01, help="Linear stall penalty per second"),
            UiField("stall_fixed", "Stall fixed penalty", 0.03, min=0.0, step=0.01, help="Penalty once any stall occurs"),
        ]

    @classmethod
    def defaults(cls) -> Dict[str, float]:
        return {
            "attr": 0.33,
            "geom": 0.33,
            "occp": 0.2,
            "atlas": 0.0,
            "default_component": 0.1,
            "fov_saliency": 1.0,
            "temporal_penalty": 0.12,
            "stall_per_second": 0.12,
            "stall_fixed": 0.03,
        }

    def _tile_weight(self, tile_id: str, viewport_tiles: Optional[Iterable[str]]) -> float:
        base = self.weights.tile_saliency.get(tile_id, 1.0)
        if viewport_tiles and tile_id in viewport_tiles:
            base *= self.weights.fov_saliency
        return base

    def _segment_quality(
        self,
        segment: Mapping,
        max_levels: Mapping[str, int],
        viewport_tiles: Optional[Iterable[str]] = None,
    ) -> float:
        weighted_sum = 0.0
        weight_total = 0.0
        for tile_id, tile in (segment.get("tiles") or {}).items():
            tile_w = self._tile_weight(tile_id, viewport_tiles)
            comp_sum = 0.0
            comp_weights = 0.0
            for comp_name, comp in (tile.get("components") or {}).items():
                lvl = _parse_quality_id(comp.get("quality_id"))
                if lvl is None:
                    lvl = comp.get("rep_idx", 0) or 0
                max_lvl = max_levels.get(comp_name) or max(lvl, 1)
                norm = max(0.0, min(1.0, lvl / float(max_lvl)))
                cw = self.weights.component_weights.get(comp_name, self.weights.default_component)
                comp_sum += norm * cw
                comp_weights += cw
            tile_quality = (comp_sum / comp_weights) if comp_weights > 0 else 0.0
            weighted_sum += tile_quality * tile_w
            weight_total += tile_w
        return weighted_sum / weight_total if weight_total > 0.0 else 0.0

    def compute_time_series(
        self,
        events: Mapping,
        viewport_tiles: Optional[Iterable[str]] = None,
    ) -> List[Tuple[float, float]]:
        max_levels = _max_component_levels(events)
        stalls = events.get("rebuffering_events") or []
        segments = sorted(events.get("enqueued_segments") or [], key=lambda s: s.get("time_s", 0))

        results: List[Tuple[float, float]] = []
        prev_quality: Optional[float] = None

        for seg in segments:
            seg_start = float(seg.get("time_s", 0))
            seg_end = seg_start + float(seg.get("duration_s", 0))
            base_quality = self._segment_quality(seg, max_levels, viewport_tiles)

            penalty = 0.0
            if prev_quality is not None:
                penalty += self.weights.temporal_penalty * abs(base_quality - prev_quality)

            stall_overlap, _ = _stall_overlap_and_count(seg_start, seg_end, stalls)
            if stall_overlap > 0.0:
                penalty += self.weights.stall_fixed + self.weights.stall_per_second * stall_overlap

            score = max(0.0, min(1.0, base_quality - penalty))
            results.append((seg_start, score))
            results.append((seg_end, score))

            prev_quality = base_quality

        return results


class StallAwareQoE(QoEFunction):
    """QoE that prioritizes avoiding rebuffering with an exponential stall penalty."""

    id = "StallAwareQoE"
    label = "Stall-aware QoE"
    equation = "QoE = quality * exp(-(stall_fixed + stall_per_second*t)) - temporal_penalty*|delta_q|"
    description = "Highlights stall hurt more than modest quality loss; good for stress-testing ABRs."
    implementation = "stall_first"

    def __init__(self, weights: Optional[QoEWeights] = None, overrides: Mapping[str, Any] | None = None):
        # Bias defaults slightly toward stronger stall punishment.
        tuned = weights or QoEWeights(
            component_weights={"attr": 0.35, "geom": 0.35, "occp": 0.2, "atlas": 0.0},
            fov_saliency=1.1,
            temporal_penalty=0.15,
            stall_per_second=0.25,
            stall_fixed=0.08,
            default_component=0.1,
        )
        self.weights = tuned
        self.apply_overrides(overrides)

    @classmethod
    def ui_fields(cls) -> Sequence[UiField]:
        return [
            UiField("fov_saliency", "FoV saliency", 1.1, min=0.0, step=0.05, help="Emphasis on viewport tiles"),
            UiField("temporal_penalty", "Temporal penalty", 0.15, min=0.0, step=0.01, help="Smoothness penalty scale"),
            UiField("stall_per_second", "Stall /s penalty", 0.25, min=0.0, step=0.01, help="Aggressive stall penalty per second"),
            UiField("stall_fixed", "Stall fixed penalty", 0.08, min=0.0, step=0.01, help="Penalty once any stall occurs"),
        ]

    @classmethod
    def defaults(cls) -> Dict[str, float]:
        return {
            "attr": 0.35,
            "geom": 0.35,
            "occp": 0.2,
            "atlas": 0.0,
            "default_component": 0.1,
            "fov_saliency": 1.1,
            "temporal_penalty": 0.15,
            "stall_per_second": 0.25,
            "stall_fixed": 0.08,
        }

    def _segment_quality(
        self,
        segment: Mapping,
        max_levels: Mapping[str, int],
        viewport_tiles: Optional[Iterable[str]] = None,
    ) -> float:
        # Reuse harmonic aggregation for robustness.
        harmonic = WeightedQoEHeuristic(self.weights)
        return harmonic._segment_quality(segment, max_levels, viewport_tiles)  # pylint: disable=protected-access

    def compute_time_series(
        self,
        events: Mapping,
        viewport_tiles: Optional[Iterable[str]] = None,
    ) -> List[Tuple[float, float]]:
        max_levels = _max_component_levels(events)
        stalls = events.get("rebuffering_events") or []
        segments = sorted(events.get("enqueued_segments") or [], key=lambda s: s.get("time_s", 0))

        results: List[Tuple[float, float]] = []
        prev_quality: Optional[float] = None

        for seg in segments:
            seg_start = float(seg.get("time_s", 0))
            seg_end = seg_start + float(seg.get("duration_s", 0))
            base_quality = self._segment_quality(seg, max_levels, viewport_tiles)

            smooth_penalty = 0.0
            if prev_quality is not None:
                smooth_penalty = self.weights.temporal_penalty * abs(base_quality - prev_quality)

            stall_overlap, _ = _stall_overlap_and_count(seg_start, seg_end, stalls)
            stall_penalty = 0.0
            if stall_overlap > 0.0:
                stall_penalty = math.exp(-(self.weights.stall_fixed + self.weights.stall_per_second * stall_overlap))

            score = max(0.0, min(1.0, base_quality * (stall_penalty or 1.0) - smooth_penalty))
            results.append((seg_start, score))
            results.append((seg_end, score))
            prev_quality = base_quality

        return results
