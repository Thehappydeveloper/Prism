"""Stable, user-facing names for FoV predictor implementation IDs."""

from __future__ import annotations


FOV_PREDICTOR_LABELS = {
    # Keep the implementation ID in configurations and artifacts.  It is part
    # of the backwards-compatible loading contract; only the presentation name
    # changes as the method receives its paper name.
    "ReachabilityRaycastPredictor": "Argus",
    "LinearPoseExtrapolationPredictor": "Linear pose extrapolation",
}


def predictor_label(identifier: str | None) -> str:
    value = str(identifier or "")
    return FOV_PREDICTOR_LABELS.get(value, value)
