#!/usr/bin/env python3
"""Print the paper table for the completed baseline predictor comparison."""
from __future__ import annotations

import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from report_raycast_cost_quality import cached_diagnostics, catastrophic_topk_miss_rate, predictor_latency_ms, run_metric  # noqa: E402
from report_selection import selected_run_ids  # noqa: E402

try:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from server_app.diagnostics import gather_diagnostics_for_run_ids  # type: ignore # noqa: E402
    from qoe.heuristics import VisiblePlaybackUtility  # type: ignore # noqa: E402
except ModuleNotFoundError:
    gather_diagnostics_for_run_ids = None

GROUP_NAMES = {"Paper_Baselines_RankBudget_great_wifi", "Paper_Baselines_RankBudget_great_wifi_LinearPose_Catchup"}
CORE = {"abr": "RankBudgetABR", "abr_preset_name": "paper_conservative", "bandwidth_predictor": "GroundTruthBandwidthPredictor", "network_profile": "great_wifi"}
PREDICTORS = (
    ("LastObservedFovPredictor", "__default__", "Last observed"),
    # The corrected matched rerun uses a two-second history window.  Keep this
    # exact preset ID here so regenerated paper artifacts cannot silently pick
    # the earlier frame-count configuration.
    ("LinearExtrapolationPredictor", "linear_2s", "Linear extrapolation"),
    ("LinearPoseExtrapolationPredictor", "linear_2s", "Linear pose extrapolation"),
    ("DistanceBasedTileImportancePredictor", "sigma_8m", "Distance-based"),
    ("TrainedFovPredictor", "lstm_10", "Trained LSTM"),
    ("TrainedFovPredictor", "mlp_10", "Trained MLP"),
    ("ReachabilityRaycastPredictor", "raycast_fixed_origins_500_samples_6_6n_0.6", "Argus (500 origins, 6 samples)"),
)


def load_runs() -> dict[str, dict[str, Any]]:
    out = {}
    allowed_ids = selected_run_ids()
    for config_path in (PROJECT_ROOT / "results").glob("run_*/config.json"):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8")); status = json.loads((config_path.parent / "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if status.get("status") != "finished" or config.get("run_group_name") not in GROUP_NAMES: continue
        if allowed_ids is not None and config_path.parent.name not in allowed_ids: continue
        for key, expected in CORE.items():
            if config.get(key) != expected: raise ValueError(f"{config_path.parent.name}: {key} differs")
        out[config_path.parent.name] = config
    # The matched group also contains the ground-truth tile oracle. It is a
    # delivery-study control and intentionally omitted from this master table.
    expected_run_count = (len(PREDICTORS) + 1) * 27
    if len(out) != expected_run_count: raise ValueError(f"Expected {expected_run_count} completed runs; found {len(out)}")
    return out


def groups(runs: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, dict[str, Any]]]]:
    result = []
    for predictor, preset, label in PREDICTORS:
        entries = {run_id: config for run_id, config in runs.items() if config.get("predictor") == predictor and (config.get("predictor_preset_name") or "__default__") == preset}
        counts = defaultdict(int)
        for config in entries.values(): counts[config["dataset_id"]] += 1
        if len(entries) != 27 or dict(counts) != {"s7": 14, "s15": 13}: raise ValueError(f"{label}: expected 14 s7 + 13 s15, found {dict(counts)}")
        result.append((label, entries))
    return result


def mean(values: Iterable[float]) -> float: return statistics.fmean(list(values))


def by_dataset(entries: dict[str, dict[str, Any]], getter: Callable[[str], float]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for run_id, config in entries.items():
        value = getter(run_id)
        if value == value: values[config["dataset_id"]].append(value)
    return dict(values)


def estimate(values: dict[str, list[float]], draws: int = 2000) -> tuple[float, float, float]:
    if set(values) != {"s7", "s15"}: raise ValueError(f"Expected both datasets; got {sorted(values)}")
    point = mean(mean(v) for v in values.values()); rng = random.Random(42); samples = []
    for _ in range(draws): samples.append(mean(mean(rng.choices(v, k=len(v))) for v in values.values()))
    samples.sort(); return point, samples[int(.025 * draws)], samples[int(.975 * draws) - 1]


def final_qoe(metrics: dict[str, Any], run_id: str) -> float:
    points = metrics.get("qoe_series", {}).get(run_id) or []
    if not points: return float("nan")
    area = sum(max(0.0, float(right["time_s"]) - float(left["time_s"])) * (float(left["score"]) + float(right["score"])) / 2.0 for left, right in zip(points, points[1:]))
    duration = sum(max(0.0, float(right["time_s"]) - float(left["time_s"])) for left, right in zip(points, points[1:]))
    return area / duration if duration > 1e-12 else float(points[-1]["score"])


def diagnostics(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ids = sorted(runs)
    return gather_diagnostics_for_run_ids(ids, datasets={"qoe_series", "fov_prediction", "latency"}, qoe_model=VisiblePlaybackUtility()) if gather_diagnostics_for_run_ids is not None else cached_diagnostics(ids)


def main() -> None:
    runs = load_runs(); metrics = diagnostics(runs)
    print(f"# Baseline predictor results\n\nSource execution group: `Paper_Baselines_RankBudget_great_wifi`. All {len(runs)} runs finished.\n\nEach estimate is the equally weighted mean of the s7 and s15 trace means; brackets are deterministic stratified 95% bootstrap intervals across traces (2,000 draws; seed 42).")
    print("\n| Predictor | Traces (s7/s15) | Stable visible utility | Weighted IoU | JS divergence | NDCG | Catastrophic miss rate |\n|---|---:|---:|---:|---:|---:|---:|")
    for label, entries in groups(runs):
        point, low, high = estimate(by_dataset(entries, lambda run_id: final_qoe(metrics, run_id)))
        weighted_iou = estimate(by_dataset(entries, lambda run_id: run_metric(metrics, run_id, "weightedIou")))
        js_divergence = estimate(by_dataset(entries, lambda run_id: run_metric(metrics, run_id, "jsDivergence")))
        ndcg = estimate(by_dataset(entries, lambda run_id: run_metric(metrics, run_id, "ndcg")))
        catastrophic = estimate(by_dataset(entries, catastrophic_topk_miss_rate))
        formatted = lambda triple: f"{triple[0]:.3f} [{triple[1]:.3f}, {triple[2]:.3f}]"
        print(f"| {label} | 14 / 13 | {point:.3f} [{low:.3f}, {high:.3f}] | {formatted(weighted_iou)} | {formatted(js_divergence)} | {formatted(ndcg)} | {formatted(catastrophic)} |")
