"""Reachability-based FoV prediction built on voxelized ray casting.

This package exposes the public predictor and its parameter object. The
predictor entry point lives in `predictor.py`; the rest of the package is the
supporting pipeline for voxelization, intention inference, path scoring, and
tile visibility aggregation.
"""

from .predictor import ReachabilityRaycastPredictor
from .reachability_pipeline import ReachabilityParams

__all__ = [
    "ReachabilityParams",
    "ReachabilityRaycastPredictor",
]
