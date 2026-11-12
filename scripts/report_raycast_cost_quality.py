#!/usr/bin/env python3
"""Shared paper-metric helpers retained for baseline and scaling reports."""

from __future__ import annotations

import json
import gzip
import random
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from report_selection import selected_run_ids


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:  # The normal project environment uses the canonical diagnostics loader.
    from server_app.diagnostics import gather_diagnostics_for_run_ids  # type: ignore # noqa: E402
    from qoe.heuristics import VisiblePlaybackUtility  # type: ignore # noqa: E402
except ModuleNotFoundError:  # Permit report regeneration from an existing UI cache too.
    gather_diagnostics_for_run_ids = None


GROUP_NAME = "Paper_Raycast_CostQuality_RankBudget_great_wifi"
REFERENCE = {"origins": 500, "time_ms": 3000, "rays": 6, "processes": 8}
FIXED_CONFIG = {
    "abr": "RankBudgetABR",
    "abr_preset_name": "paper_conservative",
    "predictor": "ReachabilityRaycastPredictor",
    "bandwidth_predictor": "GroundTruthBandwidthPredictor",
    "network_profile": "great_wifi",
    "segment_workers": 1,
}


@dataclass(frozen=True)
class Cohort:
    title: str
    variable_label: str
    values: tuple[int, ...]
    value_for: Callable[[dict[str, Any]], int]
    includes: Callable[[dict[str, Any]], bool]
    fixed_label: str


def params(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("predictor_params") or {}


def origins(config: dict[str, Any]) -> int:
    return int(params(config)["max_candidate_voxels"])


def time_ms(config: dict[str, Any]) -> int:
    return int(params(config)["auto_visibility_time_budget_ms"])


def rays(config: dict[str, Any]) -> int:
    return int(params(config)["max_rays_per_tile"])


COHORTS = (
    Cohort(
        "Candidate-origin budget",
        "Candidate-origin budget (voxels)",
        (64, 128, 250, 500, 750, 1000),
        origins,
        lambda cfg: time_ms(cfg) == REFERENCE["time_ms"] and rays(cfg) == REFERENCE["rays"],
        "Visibility budget = 3000 ms; rays/tile = 6; raycast processes = 8",
    ),
    Cohort(
        "Visibility-time budget",
        "Visibility computation budget (ms)",
        (500, 1000, 2000, 3000),
        time_ms,
        lambda cfg: origins(cfg) == REFERENCE["origins"] and rays(cfg) == REFERENCE["rays"],
        "Candidate-origin budget = 500; rays/tile = 6; raycast processes = 8",
    ),
    Cohort(
        "Rays-per-tile budget",
        "Rays per tile",
        (1, 3, 6, 12),
        rays,
        lambda cfg: origins(cfg) == REFERENCE["origins"] and time_ms(cfg) == REFERENCE["time_ms"],
        "Candidate-origin budget = 500; visibility budget = 3000 ms; raycast processes = 8",
    ),
)


def load_completed_group_runs() -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    allowed_ids = selected_run_ids()
    for config_path in (PROJECT_ROOT / "results").glob("run_*/config.json"):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            status = json.loads((config_path.parent / "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if config.get("run_group_name") != GROUP_NAME or status.get("status") != "finished":
            continue
        if allowed_ids is not None and config_path.parent.name not in allowed_ids:
            continue
        run_id = config_path.parent.name
        validate_fixed_config(run_id, config)
        runs[run_id] = config
    if len(runs) != 324:
        raise ValueError(f"Expected 324 completed runs in {GROUP_NAME}; found {len(runs)}.")
    return runs


def validate_fixed_config(run_id: str, config: dict[str, Any]) -> None:
    for key, expected in FIXED_CONFIG.items():
        if key == "segment_workers":
            actual = int(config.get("segment_workers") or 1)
        else:
            actual = config.get(key)
        if actual != expected:
            raise ValueError(f"{run_id}: expected {key}={expected!r}, found {actual!r}")
    if int(params(config).get("raycast_processes") or 0) != REFERENCE["processes"]:
        raise ValueError(f"{run_id}: expected raycast_processes={REFERENCE['processes']}")


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else float("nan")


def run_metric(metrics: dict[str, Any], run_id: str, key: str) -> float:
    values = metrics.get("fov_prediction", {}).get(run_id, {}).get(key) or []
    return mean(float(value) for value in values)


def final_qoe(metrics: dict[str, Any], run_id: str) -> float:
    points = metrics.get("qoe_series", {}).get(run_id) or []
    if not points:
        return float("nan")
    if len(points) == 1:
        return float(points[0]["score"])
    area = 0.0
    duration = 0.0
    for left, right in zip(points, points[1:]):
        dt = max(0.0, float(right["time_s"]) - float(left["time_s"]))
        area += dt * (float(left["score"]) + float(right["score"])) / 2.0
        duration += dt
    return area / duration if duration > 1e-12 else float(points[-1]["score"])


def predictor_latency_ms(metrics: dict[str, Any], run_id: str) -> float:
    cached = (metrics.get("latency", {}).get(run_id) or {}).get("mean_ms")
    if cached is not None:
        return float(cached)
    try:
        diagnostics = json.loads(
            (PROJECT_ROOT / "results" / run_id / "predictor_diagnostics.json").read_text(encoding="utf-8")
        )
        return float((diagnostics.get("total") or {}).get("mean_ms") or float("nan"))
    except (OSError, ValueError, TypeError):
        return float("nan")


def catastrophic_topk_miss_rate(run_id: str) -> float:
    """Rate of top-K predictions capturing under 10% of soft FoV mass."""
    path = PROJECT_ROOT / "results" / run_id / "events.json.gz"
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            events = json.load(handle)
    except (OSError, ValueError, TypeError):
        return float("nan")
    predicted = {int(item.get("segment_idx", index)): item.get("tile_visibility") or {} for index, item in enumerate(events.get("fov_predictions") or [])}
    actual = {int(item.get("segment_idx", index)): item.get("tile_visibility") or {} for index, item in enumerate(events.get("fov_ground_truth") or [])}
    misses: list[float] = []
    for segment_idx, scores in predicted.items():
        target = actual.get(segment_idx)
        if not target:
            continue
        tiles = set(map(str, scores)) | set(map(str, target))
        visible = [tile_id for tile_id in tiles if float(target.get(tile_id, 0.0) or 0.0) >= 0.01]
        if not visible:
            continue
        ranked = sorted(tiles, key=lambda tile_id: (-float(scores.get(tile_id, 0.0) or 0.0), tile_id))
        top_k = ranked[:max(1, len(visible))]
        total_mass = sum(max(0.0, float(target.get(tile_id, 0.0) or 0.0)) for tile_id in tiles)
        captured_mass = sum(max(0.0, float(target.get(tile_id, 0.0) or 0.0)) for tile_id in top_k)
        misses.append(1.0 if total_mass > 1e-12 and captured_mass / total_mass < 0.10 else 0.0)
    return mean(misses) if misses else float("nan")


def balanced_bootstrap_ci(by_dataset: dict[str, list[float]], seed: int = 42, draws: int = 2000) -> tuple[float, float, float]:
    """Equal-weight the dataset means, then bootstrap traces within each dataset."""
    usable = {name: values for name, values in by_dataset.items() if values}
    if set(usable) != {"s7", "s15"}:
        raise ValueError(f"Expected both s7 and s15 values; found {sorted(usable)}")
    point = mean(mean(values) for values in usable.values())
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        samples.append(mean(mean(rng.choices(values, k=len(values))) for values in usable.values()))
    samples.sort()
    return point, samples[int(0.025 * draws)], samples[int(0.975 * draws) - 1]


def format_estimate(by_dataset: dict[str, list[float]], digits: int) -> str:
    point, lower, upper = balanced_bootstrap_ci(by_dataset)
    return f"{point:.{digits}f} [{lower:.{digits}f}, {upper:.{digits}f}]"


def cohort_runs(cohort: Cohort, runs: dict[str, dict[str, Any]]) -> dict[int, dict[str, dict[str, Any]]]:
    selected: dict[int, dict[str, dict[str, Any]]] = {value: {} for value in cohort.values}
    for run_id, config in runs.items():
        if not cohort.includes(config):
            continue
        value = cohort.value_for(config)
        if value in selected:
            selected[value][run_id] = config
    for value, entries in selected.items():
        by_dataset = defaultdict(int)
        for config in entries.values():
            by_dataset[str(config.get("dataset_id"))] += 1
        if len(entries) != 27 or dict(by_dataset) != {"s7": 14, "s15": 13}:
            raise ValueError(f"{cohort.title}, {value}: expected 14 s7 + 13 s15 matched runs; found {dict(by_dataset)}")
    return selected


def values_by_dataset(entries: dict[str, dict[str, Any]], metric: Callable[[str], float]) -> dict[str, list[float]]:
    result: dict[str, list[float]] = defaultdict(list)
    for run_id, config in entries.items():
        value = metric(run_id)
        if value == value:  # omit NaN without making a pandas dependency mandatory
            result[str(config["dataset_id"])].append(value)
    return dict(result)


def cached_diagnostics(run_ids: Iterable[str]) -> dict[str, Any]:
    """Read only the current paper-QoE entries from the comparison cache.

    The cache can legitimately hold several generations of metrics for the
    same run.  Paper reports must never combine those generations: a missing
    current value is an error that requires recomputation, not a reason to
    fall back to a legacy QoE series.
    """
    wanted = set(run_ids)
    out: dict[str, Any] = {name: {} for name in ("qoe_series", "fov_prediction", "latency")}
    cache_root = PROJECT_ROOT / ".prism_cache" / "graphs" / "run_metrics"
    for path in cache_root.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            key = json.loads(payload.get("key") or "{}")
            qoe_signature = json.loads(key.get("qoe") or "{}")
            cached = payload.get("metrics") or {}
        except (OSError, ValueError, TypeError):
            continue
        if (
            qoe_signature.get("class") != "VisiblePlaybackUtility"
            or qoe_signature.get("scoring_version") != "mpd_stable_visible_utility_v1"
        ):
            continue
        available = wanted & set((cached.get("meta") or {}).keys())
        for run_id in available:
            for dataset in out:
                value = (cached.get(dataset) or {}).get(run_id)
                if value is not None:
                    out[dataset][run_id] = value
        if all(len(out[dataset]) == len(wanted) for dataset in out):
            break
    missing = sorted(wanted - set(out["qoe_series"]))
    if missing:
        raise RuntimeError(
            "The project dependencies are unavailable and the UI metric cache is incomplete "
            f"({len(missing)} runs missing). Run this script in the Prism environment after opening "
            "the cohort in Advanced Compare."
        )
    return out
