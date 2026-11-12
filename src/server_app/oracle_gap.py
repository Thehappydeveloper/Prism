from __future__ import annotations

from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping

from qoe.base import QoEFunction

from .diagnostics import gather_diagnostics


ORACLE_MODE_LABELS = {
    "none": "Actual",
    "bandwidth": "Perfect bandwidth",
    "fov_trace": "Perfect FoV trace",
    "fov_tile_scores": "Perfect FoV tile scores",
    "bandwidth_and_fov_trace": "Perfect bandwidth + FoV trace",
    "bandwidth_and_fov_tile_scores": "Perfect bandwidth + FoV tile scores",
    "offline_optimal": "Offline greedy optimal",
}

COMPARABILITY_FIELDS = (
    "dataset_id",
    "mpd",
    "content_dir",
    "network_trace",
    "network_profile",
    "fov_trace",
    "rtt",
    "block_size_bits",
    "buffer_window_segments",
    "buffer_unbounded",
    "segment_limit",
    "fov_h_deg",
    "fov_v_deg",
    "angle_convention",
)


def _final_qoe(metrics: Mapping[str, Any], run_id: str) -> float | None:
    series = metrics.get("qoe_series", {}).get(run_id) or []
    if series:
        try:
            return float(series[-1].get("score", 0.0))
        except (TypeError, ValueError):
            return None
    values = metrics.get("qoe", {}).get(run_id) or []
    if not values:
        return None
    try:
        return float(values[-1])
    except (TypeError, ValueError):
        return None


def _avg(values: Iterable[Any]) -> float:
    parsed = []
    for value in values:
        try:
            parsed.append(float(value))
        except (TypeError, ValueError):
            continue
    return mean(parsed) if parsed else 0.0


def _mode_for_run(run: Mapping[str, Any]) -> str:
    config = run.get("config") or {}
    return str(config.get("oracle_mode") or "none")


def _group_id_for_run(run: Mapping[str, Any]) -> str:
    config = run.get("config") or {}
    return str(config.get("oracle_group_id") or config.get("baseline_run_id") or run.get("run_id"))


def _baseline_id_for_group(group_runs: List[Mapping[str, Any]]) -> str | None:
    for run in group_runs:
        config = run.get("config") or {}
        if str(config.get("oracle_mode") or "none") == "none":
            return str(run.get("run_id"))
    for run in group_runs:
        config = run.get("config") or {}
        if config.get("baseline_run_id"):
            return str(config.get("baseline_run_id"))
    return str(group_runs[0].get("run_id")) if group_runs else None


def _validation(group_runs: List[Mapping[str, Any]], baseline_id: str | None) -> Dict[str, Any]:
    baseline = next((run for run in group_runs if run.get("run_id") == baseline_id), None)
    if baseline is None:
        return {"valid": False, "reason": "Baseline run is missing from this oracle group."}
    base_config = baseline.get("config") or {}
    mismatches = []
    for run in group_runs:
        config = run.get("config") or {}
        for field in COMPARABILITY_FIELDS:
            if str(config.get(field)) != str(base_config.get(field)):
                mismatches.append({"run_id": run.get("run_id"), "field": field})
    if mismatches:
        return {"valid": False, "reason": "Comparable-run fields differ.", "mismatches": mismatches[:20]}
    return {"valid": True, "reason": ""}


def _run_breakdown(metrics: Mapping[str, Any], run_id: str) -> Dict[str, Any]:
    stall = metrics.get("stall", {}).get(run_id) or {}
    component_bits = metrics.get("component_bits", {}).get(run_id) or {}
    fov_quality = metrics.get("fov_quality", {}).get(run_id) or {}
    non_quality = fov_quality.get("non") or []
    return {
        "stall_duration_s": float(stall.get("duration", 0.0) or 0.0),
        "stall_count": int(stall.get("count", 0) or 0),
        "avg_visible_quality": _avg(fov_quality.get("fov") or []),
        "avg_non_visible_quality": _avg(non_quality),
        "total_bits": sum(float(value or 0.0) for value in component_bits.values()),
        "component_bits": component_bits,
    }


def build_oracle_gap(runs: List[Dict[str, Any]], qoe_model: QoEFunction | None = None) -> Dict[str, Any]:
    metrics = gather_diagnostics(runs, qoe_model)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for run in runs:
        groups.setdefault(_group_id_for_run(run), []).append(run)

    out_groups = []
    for group_id, group_runs in sorted(groups.items()):
        baseline_id = _baseline_id_for_group(group_runs)
        validation = _validation(group_runs, baseline_id)
        baseline_qoe = _final_qoe(metrics, baseline_id) if baseline_id else None
        run_entries = []
        mode_to_run: Dict[str, Dict[str, Any]] = {}
        for run in group_runs:
            run_id = str(run.get("run_id"))
            mode = _mode_for_run(run)
            qoe = _final_qoe(metrics, run_id)
            entry = {
                "run_id": run_id,
                "mode": mode,
                "label": ORACLE_MODE_LABELS.get(mode, mode),
                "qoe": qoe,
                "gap_from_actual": (qoe - baseline_qoe) if qoe is not None and baseline_qoe is not None else None,
                "breakdown": _run_breakdown(metrics, run_id),
                "config": {
                    "abr": (run.get("config") or {}).get("abr"),
                    "predictor": (run.get("config") or {}).get("predictor") or "None",
                    "bandwidth_predictor": (run.get("config") or {}).get("bandwidth_predictor") or "None",
                },
            }
            run_entries.append(entry)
            mode_to_run[mode] = entry

        offline_qoe = (mode_to_run.get("offline_optimal") or {}).get("qoe")
        combined = mode_to_run.get("bandwidth_and_fov_tile_scores") or mode_to_run.get("bandwidth_and_fov_trace")
        combined_qoe = combined.get("qoe") if combined else None
        total_gap = (offline_qoe - baseline_qoe) if offline_qoe is not None and baseline_qoe is not None else None
        bandwidth_qoe = (mode_to_run.get("bandwidth") or {}).get("qoe")
        fov_qoe = (mode_to_run.get("fov_tile_scores") or mode_to_run.get("fov_trace") or {}).get("qoe")

        derived = {
            "bandwidth_prediction_gap": (bandwidth_qoe - baseline_qoe) if bandwidth_qoe is not None and baseline_qoe is not None else None,
            "fov_prediction_gap": (fov_qoe - baseline_qoe) if fov_qoe is not None and baseline_qoe is not None else None,
            "combined_prediction_gap": (combined_qoe - baseline_qoe) if combined_qoe is not None and baseline_qoe is not None else None,
            "abr_decision_gap": (offline_qoe - combined_qoe) if offline_qoe is not None and combined_qoe is not None else None,
            "total_oracle_gap": total_gap,
        }
        if total_gap and abs(total_gap) > 1e-12:
            derived["bandwidth_gap_pct"] = derived["bandwidth_prediction_gap"] / total_gap if derived["bandwidth_prediction_gap"] is not None else None
            derived["fov_gap_pct"] = derived["fov_prediction_gap"] / total_gap if derived["fov_prediction_gap"] is not None else None
            derived["abr_gap_pct"] = derived["abr_decision_gap"] / total_gap if derived["abr_decision_gap"] is not None else None

        run_entries.sort(key=lambda entry: list(ORACLE_MODE_LABELS).index(entry["mode"]) if entry["mode"] in ORACLE_MODE_LABELS else 999)
        out_groups.append(
            {
                "oracle_group_id": group_id,
                "baseline_run_id": baseline_id,
                "validation": validation,
                "runs": run_entries,
                "derived_gaps": derived,
            }
        )

    return {"groups": out_groups, "metrics": metrics}
