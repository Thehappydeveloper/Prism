#!/usr/bin/env python3
"""Render the matched CoverageAwareUtilityABR bandwidth-QoE experiment table."""
from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from report_baselines_rank_budget import by_dataset, estimate, final_qoe  # noqa: E402
from report_selection import selected_run_ids  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from qoe.heuristics import FovAwarePlaybackUtility  # noqa: E402
from server_app.diagnostics import gather_diagnostics_for_run_ids  # noqa: E402

GROUP_NAMES = {"Paper_Baselines_CoverageAware_BandwidthQoE_SixRates", "Paper_Baselines_CoverageAware_BandwidthQoE_LinearPose_Catchup"}
CORE = {
    "abr": "CoverageAwareUtilityABR",
    "abr_preset_name": "cost_aware_coverage",
    "bandwidth_predictor": "GroundTruthBandwidthPredictor",
}
NETWORKS = (
    ("paper_wifi_55", 55),
    ("paper_wifi_74", 74),
    ("paper_wifi_112", 112),
    ("paper_wifi_150", 150),
    ("paper_wifi_188", 188),
    ("paper_wifi_225", 225),
)
PREDICTORS = (
    ("LastObservedFovPredictor", "__default__", "Last observed"),
    ("LinearExtrapolationPredictor", "linear_2s", "Linear extrapolation"),
    ("LinearPoseExtrapolationPredictor", "linear_2s", "Linear pose extrapolation"),
    ("DistanceBasedTileImportancePredictor", "sigma_8m", "Distance-based"),
    ("TrainedFovPredictor", "lstm_10", "Trained LSTM"),
    ("TrainedFovPredictor", "mlp_10", "Trained MLP"),
    ("GroundTruthTileFovPredictor", "__default__", "Ground-truth FoV oracle"),
    ("ReachabilityRaycastPredictor", "raycast_fixed_origins_500_samples_6_6n_0.6_processes_8", "Argus (8 raycasting processes)"),
)


def load_runs() -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    allowed_ids = selected_run_ids()
    for config_path in (PROJECT_ROOT / "results").glob("run_*/config.json"):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            status = json.loads((config_path.parent / "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if config.get("run_group_name") not in GROUP_NAMES or status.get("status") != "finished":
            continue
        if allowed_ids is not None and config_path.parent.name not in allowed_ids:
            continue
        mismatches = [key for key, expected in CORE.items() if config.get(key) != expected]
        if mismatches:
            raise ValueError(f"{config_path.parent.name}: unexpected {', '.join(mismatches)}")
        if config.get("network_profile") not in {profile for profile, _ in NETWORKS}:
            raise ValueError(f"{config_path.parent.name}: unexpected network profile")
        runs[config_path.parent.name] = config
    expected_run_count = len(PREDICTORS) * len(NETWORKS) * 27
    if len(runs) != expected_run_count:
        raise ValueError(
            f"Expected {expected_run_count} completed runs across the baseline and catch-up groups; found {len(runs)}"
        )
    return runs


def condition_runs(
    runs: dict[str, dict[str, Any]], predictor: str, preset: str, profile: str, label: str
) -> dict[str, dict[str, Any]]:
    entries = {
        run_id: config for run_id, config in runs.items()
        if config.get("predictor") == predictor
        and (config.get("predictor_preset_name") or "__default__") == preset
        and config.get("network_profile") == profile
    }
    counts: dict[str, int] = defaultdict(int)
    for config in entries.values():
        counts[str(config.get("dataset_id"))] += 1
    if len(entries) != 27 or dict(counts) != {"s7": 14, "s15": 13}:
        raise ValueError(f"{label}, {profile}: expected 14 s7 + 13 s15 runs; found {dict(counts)}")
    return entries


def playback_metric(metrics: dict[str, Any], run_id: str, section: str, key: str) -> float:
    value = (metrics.get(section, {}).get(run_id) or {}).get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def format_estimate(values: dict[str, list[float]], digits: int = 3) -> str:
    point, low, high = estimate(values)
    return f"{point:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def main() -> None:
    runs = load_runs()
    metrics = gather_diagnostics_for_run_ids(
        sorted(runs), datasets={"qoe_series", "stall", "buffer"}, qoe_model=FovAwarePlaybackUtility()
    )
    conditions = {
        (profile, label): condition_runs(runs, predictor, preset, profile, label)
        for predictor, preset, label in PREDICTORS for profile, _ in NETWORKS
    }
    print("# CoverageAwareUtilityABR bandwidth QoE by FoV predictor\n")
    print(f"All {len(runs):,} matched runs completed: {len(PREDICTORS)} predictor variants × 6 fixed-rate WiFi profiles × 27 traces. Each profile has 30 ms delay, 10 ms jitter, 0.1% loss, and uniform delay; only the rate changes. CoverageAwareUtilityABR uses the oracle bandwidth estimate with safety factor 1.0. FoV-aware playback utility v2 excludes segments 0--1 as predictor warm-up, then uses ground-truth visibility rather than predictor scores: missing visible quality receives a 0.78 weight, high-quality non-FoV allocation a 0.12 weight, and persistent-FoV quality changes a 0.10 weight. Playback gaps after warm-up score zero.\n")
    print("Each estimate equally weights the s7 and s15 trace means. Brackets are deterministic stratified 95% bootstrap intervals across traces (2,000 draws; seed 42).\n")
    print("## FoV-aware playback utility\n")
    print("| FoV predictor | 55 Mb/s | 74 Mb/s | 112 Mb/s | 150 Mb/s | 188 Mb/s | 225 Mb/s |\n|---|---:|---:|---:|---:|---:|---:|")
    for predictor, preset, label in PREDICTORS:
        row = [format_estimate(by_dataset(conditions[(profile, label)], lambda run_id: final_qoe(metrics, run_id))) for profile, _ in NETWORKS]
        print(f"| {label} | " + " | ".join(row) + " |")
    print("\n## Delivery outcomes\n")
    print("| Rate | FoV predictor | FoV-aware utility | Stall time (s) | Rebuffer events | Mean buffer (s) |\n|---:|---|---:|---:|---:|---:|")
    outcome_metrics: tuple[tuple[str, Callable[[str], float], int], ...] = (
        ("fov-aware", lambda run_id: final_qoe(metrics, run_id), 3),
        ("stall", lambda run_id: playback_metric(metrics, run_id, "stall", "duration"), 3),
        ("events", lambda run_id: playback_metric(metrics, run_id, "stall", "count"), 2),
        ("buffer", lambda run_id: playback_metric(metrics, run_id, "buffer", "mean"), 3),
    )
    for profile, rate in NETWORKS:
        for predictor, preset, label in PREDICTORS:
            entries = conditions[(profile, label)]
            cells = [format_estimate(by_dataset(entries, getter), digits) for _, getter, digits in outcome_metrics]
            print(f"| {rate} | {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
