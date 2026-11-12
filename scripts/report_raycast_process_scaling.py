#!/usr/bin/env python3
"""Print the paper table for the completed Raycast_Process_Scaling group."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from report_baselines_rank_budget import by_dataset, estimate, final_qoe  # noqa: E402
from report_raycast_cost_quality import cached_diagnostics, catastrophic_topk_miss_rate, predictor_latency_ms, run_metric  # noqa: E402
from report_selection import selected_run_ids  # noqa: E402

try:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from server_app.diagnostics import gather_diagnostics_for_run_ids  # type: ignore # noqa: E402
    from qoe.heuristics import VisiblePlaybackUtility  # type: ignore # noqa: E402
except ModuleNotFoundError:
    gather_diagnostics_for_run_ids = None

GROUP_NAME = "Raycast_Process_Scaling"
PROCESSES = (1, 2, 4, 8, 16)
CORE = {"abr": "RankBudgetABR", "abr_preset_name": "paper_conservative", "predictor": "ReachabilityRaycastPredictor", "bandwidth_predictor": "GroundTruthBandwidthPredictor", "network_profile": "great_wifi", "segment_workers_override": 1}


def load_runs() -> dict[str, dict[str, Any]]:
    found = {}
    allowed_ids = selected_run_ids()
    for path in (PROJECT_ROOT / "results").glob("run_*/config.json"):
        try:
            config = json.loads(path.read_text(encoding="utf-8")); status = json.loads((path.parent / "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if config.get("run_group_name") != GROUP_NAME or status.get("status") != "finished":
            continue
        if allowed_ids is not None and path.parent.name not in allowed_ids:
            continue
        for key, expected in CORE.items():
            if config.get(key) != expected: raise ValueError(f"{path.parent.name}: {key} differs")
        params = config.get("predictor_params") or {}
        if int(params.get("max_candidate_voxels") or 0) != 500 or int(params.get("max_rays_per_tile") or 0) != 6 or int(params.get("auto_visibility_time_budget_ms") or 0) != 3000:
            raise ValueError(f"{path.parent.name}: raycast reference setting differs")
        found[path.parent.name] = config
    if len(found) != 135: raise ValueError(f"Expected 135 completed runs; found {len(found)}")
    return found


def groups(runs: dict[str, dict[str, Any]]) -> list[tuple[int, dict[str, dict[str, Any]]]]:
    result = []
    for process_count in PROCESSES:
        entries = {run_id: config for run_id, config in runs.items() if int((config.get("predictor_params") or {}).get("raycast_processes") or 0) == process_count}
        counts = defaultdict(int)
        for config in entries.values(): counts[config["dataset_id"]] += 1
        if len(entries) != 27 or dict(counts) != {"s7": 14, "s15": 13}: raise ValueError(f"{process_count} processes: expected 14 s7 + 13 s15; found {dict(counts)}")
        result.append((process_count, entries))
    return result


def diagnostics(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ids = sorted(runs)
    return gather_diagnostics_for_run_ids(ids, datasets={"qoe_series", "fov_prediction", "latency"}, qoe_model=VisiblePlaybackUtility()) if gather_diagnostics_for_run_ids is not None else cached_diagnostics(ids)


def main() -> None:
    runs = load_runs(); metrics = diagnostics(runs)
    latency_by_process = {count: estimate(by_dataset(entries, lambda run_id: predictor_latency_ms(metrics, run_id)))[0] for count, entries in groups(runs)}
    baseline_latency = latency_by_process[1]
    print("# Argus process-scaling results\n\nSource execution group: `Raycast_Process_Scaling`. All 135 runs finished.\n\nEach estimate is the equally weighted mean of the s7 and s15 trace means; brackets are deterministic stratified 95% bootstrap intervals across traces (2,000 draws; seed 42). Speedup is the pooled one-process predictor latency divided by the pooled latency at the listed process count.")
    print("\n| Argus raycasting processes | Traces (s7/s15) | Predictor latency (ms) | Speedup | Stable visible utility | Weighted IoU | JS divergence | NDCG | Catastrophic miss rate |\n|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for count, entries in groups(runs):
        latency = estimate(by_dataset(entries, lambda run_id: predictor_latency_ms(metrics, run_id)))
        values = {"qoe": estimate(by_dataset(entries, lambda run_id: final_qoe(metrics, run_id))), "weighted": estimate(by_dataset(entries, lambda run_id: run_metric(metrics, run_id, "weightedIou"))), "js": estimate(by_dataset(entries, lambda run_id: run_metric(metrics, run_id, "jsDivergence"))), "ndcg": estimate(by_dataset(entries, lambda run_id: run_metric(metrics, run_id, "ndcg"))), "catastrophic": estimate(by_dataset(entries, catastrophic_topk_miss_rate))}
        fmt = lambda item: f"{item[0]:.3f} [{item[1]:.3f}, {item[2]:.3f}]"
        print(f"| {count} | 14 / 13 | {fmt(latency)} | {baseline_latency / latency[0]:.2f}× | {fmt(values['qoe'])} | {fmt(values['weighted'])} | {fmt(values['js'])} | {fmt(values['ndcg'])} | {fmt(values['catastrophic'])} |")


if __name__ == "__main__": main()
