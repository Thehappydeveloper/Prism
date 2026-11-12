from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import re
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from core.event_storage import event_log_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARGUS_MMSYS2027_ROOT = PROJECT_ROOT / "docs" / "argus_mmsys2027"
REPORTS_ROOT = ARGUS_MMSYS2027_ROOT / "reports"
GENERATED_ROOT = ARGUS_MMSYS2027_ROOT / "generated"
BATCH_PLANS_ROOT = ARGUS_MMSYS2027_ROOT / "batch-plans"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _spec_path(report_id: str) -> Path:
    if not report_id or Path(report_id).name != report_id:
        raise ValueError("Invalid report id")
    return REPORTS_ROOT / f"{report_id}.prism-report.json"


def _load_spec(report_id: str) -> dict[str, Any]:
    spec = _read_json(_spec_path(report_id))
    if not spec or spec.get("id") != report_id:
        raise ValueError(f"Unknown report: {report_id}")
    return spec


def list_reports() -> list[dict[str, Any]]:
    reports = []
    for path in sorted(REPORTS_ROOT.glob("*.prism-report.json")):
        spec = _read_json(path)
        if spec.get("id"):
            outputs = list(spec.get("outputs") or [])
            if (spec.get("legacy_renderers") or {}).get("table_script"):
                outputs.append("paper-table.md")
            if (spec.get("legacy_renderers") or {}).get("figure_script"):
                outputs.append("figures/*")
            reports.append({
                "id": spec["id"], "title": spec.get("title") or spec["id"],
                "description": spec.get("description") or "", "outputs": outputs,
                "render_options": spec.get("render_options") or {},
                "parent_report": spec.get("parent_report"),
                "included_reports": (spec.get("legacy_renderers") or {}).get("included_reports") or [],
                "batch_plans": spec.get("batch_plans") or [],
            })
    return reports


def list_report_builds(report_id: str) -> list[dict[str, Any]]:
    _load_spec(report_id)
    root = GENERATED_ROOT / report_id
    builds = []
    for path in sorted(root.iterdir(), reverse=True) if root.exists() else ():
        manifest = _read_json(path / "report-manifest.json")
        if manifest:
            builds.append({"build_id": path.name, "built_at": manifest.get("built_at"), "runs": (manifest.get("validation") or {}).get("runs_found"), "outputs": manifest.get("outputs") or []})
    return builds


def load_report_batch_plan(report_id: str, plan_id: str) -> dict[str, Any]:
    """Return only a plan explicitly associated with the requested report."""
    spec = _load_spec(report_id)
    plan = next((item for item in spec.get("batch_plans") or [] if isinstance(item, dict) and item.get("id") == plan_id), None)
    if not plan:
        raise ValueError("Unknown report batch plan")
    relative_path = Path(str(plan.get("path") or ""))
    path = (PROJECT_ROOT / relative_path).resolve()
    try:
        path.relative_to(BATCH_PLANS_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("Invalid report batch plan path") from exc
    payload = _read_json(path)
    if not payload:
        raise ValueError("Report batch plan could not be read")
    return {"report_id": report_id, "plan_id": plan_id, "label": plan.get("label") or plan_id, "path": str(relative_path), "plan": payload}


def update_latest_figure_font_size(report_id: str, font_size_pt: float) -> dict[str, Any]:
    """Adjust just the current figure to a requested final printed font size."""
    spec = _load_spec(report_id)
    option = (spec.get("render_options") or {}).get("figure_font_size_pt") or {}
    if option.get("type") != "number":
        raise ValueError("This report has no adjustable final figure font size.")
    minimum, maximum = float(option.get("min", 8)), float(option.get("max", 48))
    if not minimum <= font_size_pt <= maximum:
        raise ValueError(f"Final figure font size must be between {minimum:g} and {maximum:g} pt.")
    publication = spec.get("publication_layout") or {}
    width_inches = float(publication.get("width_inches") or 0)
    width_px = float(publication.get("figure_width_px") or 0)
    reference_font_px = float(publication.get("reference_font_px") or 0)
    if not width_inches or not width_px or not reference_font_px:
        raise ValueError("This report lacks its publication-size specification.")
    font_size_px = font_size_pt * width_px / (72.0 * width_inches)
    builds = list_report_builds(report_id)
    if not builds:
        raise ValueError("Generate the report once before adjusting its figure font size.")
    build_id = str(builds[0]["build_id"])
    build_root = GENERATED_ROOT / report_id / build_id
    svg_paths = sorted(build_root.rglob("*.svg"))
    for path in svg_paths:
        text = path.read_text(encoding="utf-8")
        first_font = re.search(r"font:(\d+(?:\.\d+)?)px", text)
        if not first_font:
            continue
        current_base = float(first_font.group(1))
        text = re.sub(r"font:(\d+(?:\.\d+)?)px", lambda match: f"font:{float(match.group(1)) * font_size_px / current_base:g}px", text)
        path.write_text(text, encoding="utf-8")
    if svg_paths:
        outputs = [str(path.relative_to(build_root)) for path in svg_paths]
    else:
        figure_script = (spec.get("legacy_renderers") or {}).get("figure_script")
        if not figure_script:
            raise ValueError("The latest build has no adjustable figure.")
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
        environment["PRISM_REPORT_FIGURE_FONT_SCALE"] = str(font_size_px / reference_font_px)
        with tempfile.TemporaryDirectory(prefix="prism-figure-font-") as directory:
            rendered = Path(directory)
            result = subprocess.run([sys.executable, str(PROJECT_ROOT / str(figure_script)), "--output-dir", str(rendered)], cwd=PROJECT_ROOT, env=environment, text=True, capture_output=True, check=False)
            if result.returncode:
                raise RuntimeError(f"Figure-only renderer failed: {result.stderr.strip() or result.stdout.strip()}")
            sources = [path for path in rendered.iterdir() if path.is_file()]
            if not sources:
                raise RuntimeError("Figure-only renderer produced no figure files.")
            figure_dir = build_root / "figures"
            figure_dir.mkdir(exist_ok=True)
            for source in sources:
                shutil.copy2(source, figure_dir / source.name)
        outputs = [str(Path("figures") / source.name) for source in sources]
    return {"report_id": report_id, "build_id": build_id, "outputs": outputs, "font_size_pt": font_size_pt}


def _retain_latest_build(report_id: str, latest: Path) -> None:
    """Remove superseded generated artifacts after a new build succeeds."""
    root = (GENERATED_ROOT / report_id).resolve()
    latest = latest.resolve()
    if latest.parent != root:
        raise ValueError("Report build path is outside its generated-artifact directory")
    for candidate in root.iterdir():
        if candidate == latest or not candidate.is_dir() or candidate.name.startswith("."):
            continue
        shutil.rmtree(candidate)


def _declared_group_runs(results_root: Path, group: dict[str, Any]) -> list[dict[str, Any]]:
    """Load group members from the portable manifest, without an SQLite index."""
    rows: list[dict[str, Any]] = []
    for run_id in group.get("run_ids") or []:
        run_dir = results_root / str(run_id)
        config = _read_json(run_dir / "config.json")
        status = _read_json(run_dir / "status.json")
        if not config:
            continue
        rows.append(
            {
                "run_id": str(run_id),
                "status": status.get("status") or "unknown",
                "dataset_id": config.get("dataset_id") or "",
                # A precomputed cohort may combine historic base and catch-up
                # runs.  Report-facing snapshots use the selected manifest's
                # canonical name; the original config keeps that provenance.
                "run_group_name": group.get("name") or config.get("run_group_name") or "",
            }
        )
    return rows


def _resolve_runs(spec: dict[str, Any], results_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    groups_file = _read_json(results_root / "_run_groups.json")
    raw_groups = groups_file.get("groups") or {}
    groups = list(raw_groups.values()) if isinstance(raw_groups, dict) else raw_groups
    inputs = spec.get("inputs") or {}
    wanted = {str(value) for value in inputs.get("group_names", [])}
    group_ids = {str(value) for value in (spec.get("inputs") or {}).get("group_ids", [])}
    expected_counts = {str(name): int(count) for name, count in (inputs.get("group_run_counts") or {}).items()}
    selected = [group for group in groups if str(group.get("id")) in group_ids]
    missing: list[str] = []
    for name in sorted(wanted):
        candidates = [group for group in groups if str(group.get("name")) == name]
        expected = expected_counts.get(name)
        if expected is not None:
            candidates = [group for group in candidates if len(group.get("run_ids") or []) == expected]
        if not candidates:
            missing.append(name)
            continue
        # A complete rerun is a new group with the same canonical name.  Its
        # timestamp makes it supersede the bundled precomputed cohort.
        selected.append(max(candidates, key=lambda group: str(group.get("updated_at") or group.get("created_at") or "")))
    runs: dict[str, dict[str, Any]] = {}
    for group in selected:
        for run in _declared_group_runs(results_root, group):
            runs[str(run["run_id"])] = run
    return list(runs.values()), missing


def _matching_variant_count(requirement: dict[str, Any], runs: list[dict[str, Any]], results_root: Path) -> int:
    """Count selected finished runs whose persisted configuration matches a variant."""
    config_values = requirement.get("config") or {}
    param_values = requirement.get("predictor_params") or {}
    count = 0
    for run in runs:
        if run.get("status") != "finished":
            continue
        config = _read_json(results_root / str(run.get("run_id") or "") / "config.json")
        if any(config.get(key) != value for key, value in config_values.items()):
            continue
        params = config.get("predictor_params") or {}
        if any(params.get(key) != value for key, value in param_values.items()):
            continue
        count += 1
    return count


def _fingerprint(results_root: Path, run_id: str) -> str:
    digest = hashlib.sha256()
    run_dir = results_root / run_id
    for path in (run_dir / "config.json", event_log_path(run_dir), run_dir / "predictor_diagnostics.json"):
        if path is None or not path.exists():
            continue
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def _mean(values: Any) -> float | None:
    numbers = [float(value) for value in (values or []) if isinstance(value, (int, float))]
    return statistics.fmean(numbers) if numbers else None


def _time_weighted_mean(points: Any) -> float | None:
    if not isinstance(points, list) or not points:
        return None
    if len(points) == 1:
        return float(points[0].get("score", 0.0))
    area = duration = 0.0
    for left, right in zip(points, points[1:]):
        step = max(0.0, float(right.get("time_s", 0.0)) - float(left.get("time_s", 0.0)))
        area += step * (float(left.get("score", 0.0)) + float(right.get("score", 0.0))) / 2.0
        duration += step
    return area / duration if duration else float(points[-1].get("score", 0.0))


def _derived_metric_rows(spec: dict[str, Any], runs: list[dict[str, Any]], results_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requested = {str(metric.get("id")) for metric in spec.get("metrics") or [] if isinstance(metric, dict)}
    if not requested:
        return [], {}
    # Keep the QoE implementation in one canonical place. These values are
    # derived from event logs and never written back into run artifacts.
    from qoe.heuristics import FovAwarePlaybackUtility, VisiblePlaybackUtility
    from server_app.diagnostics import gather_diagnostics_for_run_ids

    ids = [str(run["run_id"]) for run in runs]
    qoe_model = FovAwarePlaybackUtility() if "fov_aware_playback_utility" in requested else VisiblePlaybackUtility()
    diagnostics = gather_diagnostics_for_run_ids(ids, datasets={"qoe_series", "fov_prediction", "latency"}, qoe_model=qoe_model)
    rows = []
    for run in runs:
        run_id = str(run["run_id"])
        prediction = (diagnostics.get("fov_prediction") or {}).get(run_id) or {}
        latency = (diagnostics.get("latency") or {}).get(run_id) or {}
        total_latency = latency.get("total") or {}
        row: dict[str, Any] = {"run_id": run_id, "dataset_id": run.get("dataset_id") or "", "group": run.get("run_group_name") or run.get("run_group_id") or ""}
        if "stable_visible_utility" in requested:
            row["stable_visible_utility"] = _time_weighted_mean((diagnostics.get("qoe_series") or {}).get(run_id))
        if "fov_aware_playback_utility" in requested:
            row["fov_aware_playback_utility"] = _time_weighted_mean((diagnostics.get("qoe_series") or {}).get(run_id))
        if "weighted_iou" in requested:
            row["weighted_iou"] = _mean(prediction.get("weightedIou"))
        if "predictor_latency_ms" in requested:
            row["predictor_latency_ms"] = total_latency.get("mean_ms")
        rows.append(row)
    signature = {"qoe_model": qoe_model.__class__.__name__, "qoe_scoring_version": getattr(qoe_model, "_scoring_version", None), "metrics": sorted(requested)}
    return rows, signature


def _run_legacy_renderers(
    spec: dict[str, Any],
    staging: Path,
    results_root: Path,
    selected_runs: list[dict[str, Any]],
    render_options: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[str]:
    """Render the established paper outputs into a report build unchanged."""
    legacy = spec.get("legacy_renderers") or {}
    if not isinstance(legacy, dict):
        return []
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    selection_path = staging / ".selected-run-ids.json"
    selection_path.write_text(json.dumps(sorted(str(run["run_id"]) for run in selected_runs)), encoding="utf-8")
    environment["PRISM_REPORT_RUN_IDS_FILE"] = str(selection_path)
    for key, value in (render_options or {}).items():
        environment[f"PRISM_REPORT_{str(key).upper()}"] = "1" if value else "0"
    outputs: list[str] = []
    # A paper-level build can collect the established renderers for its
    # component analyses.  Keeping their files in named subdirectories makes
    # the resulting snapshot self-contained while preserving provenance.
    for child_id in legacy.get("included_reports") or []:
        child_spec = _load_spec(str(child_id))
        child_staging = staging / str(child_id)
        child_staging.mkdir(exist_ok=True)
        child_runs, child_missing = _resolve_runs(child_spec, results_root)
        if child_missing:
            raise RuntimeError(f"Child report {child_id} is missing source groups: {', '.join(child_missing)}")
        outputs.extend(
            str(Path(str(child_id)) / path)
            for path in _run_legacy_renderers(child_spec, child_staging, results_root, child_runs, render_options, progress)
        )
    table_script = legacy.get("table_script")
    if table_script:
        if progress:
            progress(f"Rendering {spec.get('title') or 'report'} table")
        result = subprocess.run([sys.executable, str(PROJECT_ROOT / str(table_script))], cwd=PROJECT_ROOT, env=environment, text=True, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError(f"Table renderer failed: {result.stderr.strip() or result.stdout.strip()}")
        (staging / "paper-table.md").write_text(result.stdout, encoding="utf-8")
        outputs.append("paper-table.md")
    figure_script = legacy.get("figure_script")
    if figure_script:
        if progress:
            progress(f"Rendering {spec.get('title') or 'report'} figure")
        figures = staging / "figures"
        figures.mkdir(exist_ok=True)
        result = subprocess.run([sys.executable, str(PROJECT_ROOT / str(figure_script)), "--output-dir", str(figures)], cwd=PROJECT_ROOT, env=environment, text=True, capture_output=True, check=False)
        if result.returncode:
            raise RuntimeError(f"Figure renderer failed: {result.stderr.strip() or result.stdout.strip()}")
        outputs.extend(str(path.relative_to(staging)) for path in sorted(path for path in figures.iterdir() if path.is_file()))
    return outputs


def validate_report(report_id: str, results_root: Path) -> dict[str, Any]:
    spec = _load_spec(report_id)
    runs, missing_groups = _resolve_runs(spec, results_root)
    finished = [run for run in runs if run.get("status") == "finished"]
    expected = int((spec.get("validation") or {}).get("expected_run_count") or 0)
    issues = []
    if missing_groups:
        issues.append({"severity": "error", "message": f"Missing source groups: {', '.join(missing_groups)}"})
    if not runs:
        issues.append({"severity": "error", "message": "No source runs were found."})
    if len(finished) != len(runs):
        issues.append({"severity": "error", "message": f"{len(runs) - len(finished)} selected runs are not finished."})
    if expected and len(runs) != expected:
        issues.append({"severity": "warning", "message": f"Expected {expected} runs; found {len(runs)}."})
    for requirement in (spec.get("validation") or {}).get("required_variants") or []:
        if not isinstance(requirement, dict):
            continue
        expected_variant_count = int(requirement.get("count") or 0)
        actual_variant_count = _matching_variant_count(requirement, runs, results_root)
        if expected_variant_count and actual_variant_count != expected_variant_count:
            label = str(requirement.get("label") or "required variant")
            issues.append({
                "severity": "error",
                "message": f"{label}: expected {expected_variant_count} verified finished runs; found {actual_variant_count}.",
            })
    return {"report": {"id": report_id, "title": spec.get("title")}, "status": "ready" if not issues else ("error" if any(i["severity"] == "error" for i in issues) else "warning"), "runs_found": len(runs), "runs_finished": len(finished), "issues": issues, "groups": sorted({str(run.get("run_group_name") or run.get("run_group_id") or "") for run in runs})}


def build_report(
    report_id: str,
    results_root: Path,
    render_options: dict[str, Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if progress:
        progress("Validating the complete report cohort")
    validation = validate_report(report_id, results_root)
    if validation["status"] == "error":
        return {**validation, "built": False}
    spec = _load_spec(report_id)
    allowed_options = spec.get("render_options") or {}
    requested_options = render_options or {}
    resolved_options = {
        key: bool(requested_options.get(key, setting.get("default", False)))
        for key, setting in allowed_options.items()
        if isinstance(setting, dict) and setting.get("type") == "boolean"
    }
    runs, _ = _resolve_runs(spec, results_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = GENERATED_ROOT / report_id / stamp
    staging = target.with_name(f".{stamp}.staging")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        if progress:
            progress("Freezing selected run inputs")
        snapshots = [{"run_id": run["run_id"], "artifact_fingerprint": _fingerprint(results_root, run["run_id"]), "status": run.get("status"), "group": run.get("run_group_name") or run.get("run_group_id")} for run in runs]
        counts = Counter(str(item.get("group") or "Ungrouped") for item in snapshots)
        (staging / "inputs.json").write_text(json.dumps(snapshots, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with (staging / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["run_id", "status", "group", "artifact_fingerprint"])
            writer.writeheader(); writer.writerows(snapshots)
        metric_rows, metric_signature = _derived_metric_rows(spec, runs, results_root)
        if metric_rows:
            fields = list(metric_rows[0])
            with (staging / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader(); writer.writerows(metric_rows)
        table = [f"# {spec.get('title') or report_id}", "", "## Input snapshot", "", "| Group | Runs |", "|---|---:|"]
        table.extend(f"| {group} | {count} |" for group, count in sorted(counts.items()))
        table.extend(["", f"Frozen input runs: {len(snapshots)}"])
        if metric_rows:
            table.extend(["", "## Derived metrics", "", "| Metric | Mean across selected runs |", "|---|---:|"])
            for field in [name for name in metric_rows[0] if name not in {"run_id", "dataset_id", "group"}]:
                value = _mean([row.get(field) for row in metric_rows])
                table.append(f"| {field} | {value:.4f} |" if value is not None else f"| {field} | — |")
        table.extend(["", "Derived metric and figure renderers can consume `runs.csv`, `metrics.csv`, and `inputs.json` from this immutable build."])
        (staging / "tables.md").write_text("\n".join(table) + "\n", encoding="utf-8")
        outputs = ["tables.md", "inputs.json", "runs.csv"] + (["metrics.csv"] if metric_rows else [])
        manifest = {"schema": "prism.report-build.v1", "report": spec, "render_options": resolved_options, "built_at": datetime.now(timezone.utc).isoformat(), "validation": validation, "inputs_file": "inputs.json", "data_file": "runs.csv", "metric_file": "metrics.csv" if metric_rows else None, "metric_signature": metric_signature, "outputs": outputs}
        (staging / "report-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        legacy_outputs = _run_legacy_renderers(spec, staging, results_root, runs, resolved_options, progress)
        if legacy_outputs:
            outputs.extend(legacy_outputs)
            manifest["outputs"] = outputs
            (staging / "report-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        target.parent.mkdir(parents=True, exist_ok=True)
        if progress:
            progress("Finalizing immutable report build")
        shutil.move(str(staging), str(target))
        _retain_latest_build(report_id, target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    try:
        output_dir = str(target.relative_to(PROJECT_ROOT))
    except ValueError:
        output_dir = str(target)
    output_names = outputs + ["report-manifest.json"]
    return {**validation, "built": True, "build_id": stamp, "output_dir": output_dir, "outputs": output_names, "output_urls": {name: f"/reports/{report_id}/{stamp}/{name}" for name in output_names}}
