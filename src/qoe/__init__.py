"""
QoE heuristics for V3C/VPCC playback.

The :mod:`qoe.heuristics` module contains a tunable, multi-factor QoE model
that accounts for per-component quality, inter-tile aggregation, temporal
smoothness, FoV saliency, and stall penalties.
"""

from .base import QoEFunction, UiField
from .heuristics import FovAwarePlaybackUtility, LinearQoE, QoEWeights, StallAwareQoE, VisiblePlaybackUtility, WeightedQoEHeuristic

__all__ = ["QoEFunction", "UiField", "QoEWeights", "WeightedQoEHeuristic", "LinearQoE", "StallAwareQoE", "VisiblePlaybackUtility", "FovAwarePlaybackUtility"]
