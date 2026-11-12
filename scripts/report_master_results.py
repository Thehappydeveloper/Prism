#!/usr/bin/env python3
"""Render the consolidated, detailed paper-result master table.

The individual design-ablation and cost/quality reports were intentionally
retired. Their completed, matched run groups remain inputs to this table so
the paper still has one reviewable record of every retained configuration.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import report_baselines_rank_budget as baselines  # noqa: E402
import report_raycast_cost_quality as cost_quality  # noqa: E402
from report_selection import selected_run_ids  # noqa: E402


Metric = Callable[[dict[str, Any], str], float]
METRICS: tuple[tuple[str, Metric], ...] = (
    ("Weighted IoU ↑", lambda diagnostics, run_id: baselines.run_metric(diagnostics, run_id, "weightedIou")),
    ("JS divergence ↓", lambda diagnostics, run_id: baselines.run_metric(diagnostics, run_id, "jsDivergence")),
    ("NDCG ↑", lambda diagnostics, run_id: baselines.run_metric(diagnostics, run_id, "ndcg")),
    ("Top-k mass miss ↓", lambda _diagnostics, run_id: baselines.catastrophic_topk_miss_rate(run_id)),
    ("Latency (ms) ↓", lambda diagnostics, run_id: baselines.predictor_latency_ms(diagnostics, run_id)),
)

ABLATION_GROUP = "Paper_Raycast_Ablations_RankBudget_great_wifi"
ABLATION_PRESETS = (
    ("raycast_fixed_origins_500_samples_6_6n_0.6_processes_8", "Argus (design reference)"),
    ("raycast_fixed_origins_500_samples_6_6n_0.6_expected_visibility_processes_8", "Direct visibility"),
    ("raycast_fixed_origins_500_samples_6_26n_0.6_processes_8", "26-neighbor connectivity"),
    ("raycast_fixed_origins_500_samples_6_6n_0.6_single_horizon_processes_8", "Single history horizon"),
    ("raycast_fixed_origins_500_samples_6_6n_0.6_stability_0.8_processes_8", "Higher stability weight"),
    ("raycast_fixed_origins_500_samples_6_6n_0.6_topk_processes_8", "Top-k voxel selection"),
    ("raycast_fixed_origins_500_samples_6_6n_0.6_current_position_processes_8", "Current-position origins"),
    ("raycast_fixed_origins_500_samples_6_6n_0.6_uniform_reachable_processes_8", "Uniform reachable origins"),
    ("raycast_fixed_origins_500_samples_6_6n_0.6_no_obstacles_processes_8", "No navigation obstacles"),
)


def _load_ablation_runs() -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    allowed_ids = selected_run_ids()
    for config_path in (PROJECT_ROOT / "results").glob("run_*/config.json"):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            status = json.loads((config_path.parent / "status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if config.get("run_group_name") == ABLATION_GROUP and status.get("status") == "finished":
            if allowed_ids is not None and config_path.parent.name not in allowed_ids:
                continue
            runs[config_path.parent.name] = config
    if len(runs) != len(ABLATION_PRESETS) * 27:
        raise ValueError(f"Expected 216 completed design-ablation runs; found {len(runs)}")
    return runs


def _preset_groups(runs: dict[str, dict[str, Any]], presets: tuple[tuple[str, str], ...]) -> list[tuple[str, dict[str, dict[str, Any]]]]:
    groups = []
    for preset, label in presets:
        entries = {run_id: config for run_id, config in runs.items() if config.get("predictor_preset_name") == preset}
        counts: dict[str, int] = defaultdict(int)
        for config in entries.values():
            counts[str(config.get("dataset_id"))] += 1
        if len(entries) != 27 or dict(counts) != {"s7": 14, "s15": 13}:
            raise ValueError(f"{label}: expected 14 s7 + 13 s15 runs; found {dict(counts)}")
        groups.append((label, entries))
    return groups


def _values(entries: dict[str, dict[str, Any]], metric: Metric, diagnostics: dict[str, Any]) -> dict[str, list[float]]:
    by_dataset: dict[str, list[float]] = defaultdict(list)
    for run_id, config in entries.items():
        value = metric(diagnostics, run_id)
        if value == value:
            by_dataset[str(config["dataset_id"])].append(value)
    return dict(by_dataset)


def _cell(entries: dict[str, dict[str, Any]], metric: Metric, diagnostics: dict[str, Any]) -> str:
    try:
        return f"{baselines.estimate(_values(entries, metric, diagnostics))[0]:.3f}"
    except ValueError:
        # The baseline Argus run predates latency instrumentation. Preserve
        # that provenance instead of silently substituting another experiment.
        return "—"


def _print_section(title: str, rows: list[tuple[str, dict[str, dict[str, Any]]]], diagnostics: dict[str, Any]) -> None:
    print(f"| **{title}** |" + " |" * len(METRICS))
    for label, entries in rows:
        print(f"| {label} | " + " | ".join(_cell(entries, metric, diagnostics) for _, metric in METRICS) + " |")


def _cost_rows(runs: dict[str, dict[str, Any]], cohort_index: int, suffix: str = "") -> list[tuple[str, dict[str, dict[str, Any]]]]:
    cohort = cost_quality.COHORTS[cohort_index]
    return [(f"{value}{suffix}", entries) for value, entries in cost_quality.cohort_runs(cohort, runs).items()]


def main() -> None:
    baseline_runs = baselines.load_runs()
    ablation_runs = _load_ablation_runs()
    cost_runs = cost_quality.load_completed_group_runs()

    print("# Master results table — detailed")
    print()
    print("Matched s7/s15 traces, equally weighted by dataset mean. This consolidated table retains the rows from the retired design-ablation and cost/quality reports; those are separate experiments from the matched predictor comparison. Values are point estimates; the dedicated bandwidth and process-scaling reports retain confidence intervals.")
    print()
    print("| Configuration | " + " | ".join(label for label, _ in METRICS) + " |")
    print("|---|" + "---:|" * len(METRICS))
    _print_section("Predictors", baselines.groups(baseline_runs), baselines.diagnostics(baseline_runs))
    _print_section("Design ablations", _preset_groups(ablation_runs, ABLATION_PRESETS), baselines.diagnostics(ablation_runs))
    cost_diagnostics = baselines.diagnostics(cost_runs)
    _print_section("Origin cap", _cost_rows(cost_runs, 0, " origins"), cost_diagnostics)
    _print_section("Visibility budget", _cost_rows(cost_runs, 1, " ms"), cost_diagnostics)
    _print_section("Sample cap (per origin–tile pair)", _cost_rows(cost_runs, 2), cost_diagnostics)


if __name__ == "__main__":
    main()
