from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from core.rotations import QUAT_COLUMNS, ensure_fov_quaternions
from core.trace_columns import normalize_fov_dataframe, normalize_network_dataframe, read_trace_csv_subset
from server_app.config import safe_path

from .config import SAMPLE_INDEX_CACHE_ROOT, digest_payload, list_csv_files, tree_fingerprint, write_json
from .registry import get_training_dataset


def _tile_columns(df: pd.DataFrame) -> list[str]:
    return sorted(
        [str(col) for col in df.columns if str(col).startswith("tile_")],
        key=lambda value: int(value.split("_", 1)[1]) if value.split("_", 1)[1].isdigit() else value,
    )


def _history_offsets(history: dict[str, Any], fps: float) -> list[int]:
    fixed = int(history.get("fixed_history_samples") or 0)
    step_s = max(float(history.get("history_step_s") or 0.5), 1e-6)
    duration_s = max(float(history.get("history_duration_s") or 0.0), step_s)
    count = fixed if fixed > 0 else int(np.floor(duration_s / step_s)) + 1
    include_current = bool(history.get("include_current", True))
    step = max(1, int(round(step_s * max(float(fps), 1e-6))))
    if include_current:
        return [-(count - idx - 1) * step for idx in range(count)]
    return [-(count - idx) * step for idx in range(count)]


def _sample_stride(history: dict[str, Any], fps: float) -> int:
    stride_s = float(history.get("sample_stride_s") or 0.0)
    if stride_s <= 0:
        return 1
    return max(1, int(round(stride_s * max(float(fps), 1e-6))))


def _read_trace(path: Path, family: str, angle_convention: str | None = None, include_tile_columns: bool = False) -> pd.DataFrame:
    df = read_trace_csv_subset(path, family, include_tile_columns=include_tile_columns)
    if family == "fov":
        df = normalize_fov_dataframe(df, source=path)
        df = ensure_fov_quaternions(df, angle_convention=angle_convention)
        columns = ["FrameNumber", "HMDPX", "HMDPY", "HMDPZ", *QUAT_COLUMNS]
        if include_tile_columns:
            columns.extend(_tile_columns(df))
        df = df[columns].copy()
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        df = df.sort_values("FrameNumber").reset_index(drop=True)
    else:
        df = normalize_network_dataframe(df, source=path)
        df = df[["Time", "Bandwidth"]].copy()
        df = df.apply(pd.to_numeric, errors="coerce").dropna()
        df = df.sort_values("Time").reset_index(drop=True)
    return df


def _valid_rows_for_file(
    path: Path,
    family: str,
    history: dict[str, Any],
    fps: float,
    angle_convention: str | None = None,
    target: str = "pose",
) -> list[dict[str, Any]]:
    tile_target = family == "fov" and target == "tile_visibility"
    df = _read_trace(path, family, angle_convention=angle_convention, include_tile_columns=tile_target)
    if len(df) < 2:
        return []
    horizon_s = max(float(history.get("prediction_horizon_s") or 0.0), 0.0)
    offsets = _history_offsets(history, fps)
    rows: list[dict[str, Any]] = []
    if family == "fov":
        tile_columns = _tile_columns(df) if tile_target else []
        if tile_target and not tile_columns:
            return []
        frames = df["FrameNumber"].astype(int).to_numpy()
        frame_to_pos = {int(frame): idx for idx, frame in enumerate(frames)}
        horizon = max(1, int(round(horizon_s * max(float(fps), 1e-6))))
        sample_stride = _sample_stride(history, fps)
        for pos, frame in enumerate(frames):
            if pos % sample_stride != 0:
                continue
            history_frames = [int(frame + offset) for offset in offsets]
            target_frame = int(frame + horizon)
            if target_frame not in frame_to_pos:
                continue
            if any(hist not in frame_to_pos for hist in history_frames):
                continue
            rows.append(
                {
                    "trace": str(path),
                    "family": family,
                    "anchor": int(frame),
                    "history": history_frames,
                    "target": target_frame,
                    "angle_convention": angle_convention,
                    "target_kind": "tile_visibility" if tile_target else "pose",
                    "tile_columns": tile_columns,
                }
            )
    else:
        times = df["Time"].astype(float).to_numpy()
        if len(times) < 2:
            return []
        step_s = max(float(history.get("history_step_s") or 0.5), 1e-6)
        duration_s = max(float(history.get("history_duration_s") or 0.0), step_s)
        fixed = int(history.get("fixed_history_samples") or 0)
        count = fixed if fixed > 0 else int(np.floor(duration_s / step_s)) + 1
        include_current = bool(history.get("include_current", True))
        sample_stride = _sample_stride(history, 1.0 / step_s)
        for idx, anchor in enumerate(times):
            if idx % sample_stride != 0:
                continue
            start = anchor if include_current else anchor - step_s
            hist_times = [start - (count - pos - 1) * step_s for pos in range(count)]
            target_time = anchor + horizon_s
            if hist_times[0] < times[0] or target_time > times[-1]:
                continue
            rows.append(
                {
                    "trace": str(path),
                    "family": family,
                    "anchor": float(anchor),
                    "history": [float(v) for v in hist_times],
                    "target": float(target_time),
                    "row": int(idx),
                    "target_kind": "bandwidth",
                }
            )
    return rows


def _group_key(row: dict[str, Any], mode: str) -> str:
    trace = Path(str(row["trace"]))
    if mode == "by_dataset":
        return str(row.get("dataset_id") or trace.parent)
    if mode == "by_participant":
        for part in reversed(trace.parts):
            lower = part.lower()
            if lower.startswith("p") and any(ch.isdigit() for ch in lower):
                return part
        return trace.parent.name
    if mode in {"by_scenario", "by_content"}:
        return trace.parts[-3] if len(trace.parts) >= 3 else trace.parent.name
    return str(trace)


def _split_grouped_rows(
    rows: list[dict[str, Any]],
    seed: int,
    validation_fraction: float,
    test_fraction: float,
    mode: str,
) -> dict[str, list[dict[str, Any]]]:
    by_trace: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_trace.setdefault(_group_key(row, mode), []).append(row)
    traces = sorted(by_trace)
    rng = random.Random(seed)
    rng.shuffle(traces)
    n = len(traces)
    test_n = max(1 if n > 2 and test_fraction > 0 else 0, int(round(n * test_fraction)))
    val_n = max(1 if n - test_n > 1 and validation_fraction > 0 else 0, int(round(n * validation_fraction)))
    test_traces = set(traces[:test_n])
    val_traces = set(traces[test_n : test_n + val_n])
    splits = {"train": [], "validation": [], "test": []}
    for trace in traces:
        key = "test" if trace in test_traces else "validation" if trace in val_traces else "train"
        splits[key].extend(by_trace[trace])
    if not splits["train"] and (splits["validation"] or splits["test"]):
        source = "validation" if splits["validation"] else "test"
        splits["train"].append(splits[source].pop())
    return splits


def _split_random_samples(rows: list[dict[str, Any]], seed: int, validation_fraction: float, test_fraction: float) -> dict[str, list[dict[str, Any]]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    test_n = int(round(n * test_fraction))
    val_n = int(round(n * validation_fraction))
    return {
        "test": shuffled[:test_n],
        "validation": shuffled[test_n : test_n + val_n],
        "train": shuffled[test_n + val_n :],
    }


def _split_manual(rows: list[dict[str, Any]], assignments: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    splits = {"train": [], "validation": [], "test": []}
    exact_assignments = {str(key): value for key, value in assignments.items()}
    lower_assignments = {str(key).strip().lower(): value for key, value in assignments.items()}
    default = str(exact_assignments.get("*") or "train").lower()
    for row in rows:
        dataset_id = str(row.get("dataset_id") or "")
        dataset_name = str(row.get("dataset_name") or "")
        trace_name = Path(str(row.get("trace") or "")).name
        role_value = default
        for key in (dataset_id, dataset_name, trace_name):
            if not key:
                continue
            if key in exact_assignments:
                role_value = exact_assignments[key]
                break
            lower_key = key.strip().lower()
            if lower_key in lower_assignments:
                role_value = lower_assignments[lower_key]
                break
        role = str(role_value).lower()
        if role not in splits:
            role = "train"
        splits[role].append(row)
    return splits


def build_sample_index(plan: dict[str, Any], force: bool = False, progress_callback: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    datasets = []
    fingerprints = []
    dataset_snapshots = []
    for dataset_id in plan.get("dataset_ids") or []:
        dataset = get_training_dataset(str(dataset_id))
        if not dataset:
            raise ValueError(f"Training dataset '{dataset_id}' was not found.")
        datasets.append(dataset)
        fingerprints.append(tree_fingerprint(dataset["trace_path"]))
        dataset_snapshots.append(
            {
                "id": dataset.get("id"),
                "name": dataset.get("name"),
                "family": dataset.get("family"),
                "trace_path": dataset.get("trace_path"),
                "fps": dataset.get("fps"),
                "units": dataset.get("units"),
                "angle_convention": dataset.get("angle_convention"),
                "content_dir": dataset.get("content_dir"),
                "reference_ply": dataset.get("reference_ply"),
                "prism_dataset_id": dataset.get("prism_dataset_id"),
                "prism_dataset_snapshot": dataset.get("prism_dataset_snapshot"),
                "transform": dataset.get("transform"),
                "motion_limits": dataset.get("motion_limits"),
                "fingerprint": dataset.get("fingerprint"),
                "updated_at": dataset.get("updated_at"),
            }
        )
    key = digest_payload(
        {
            "version": 5,
            "plan_id": plan.get("id"),
            "family": plan.get("family"),
            "dataset_ids": plan.get("dataset_ids"),
            "target": plan.get("target"),
            "model": {
                "model_type": (plan.get("model") or {}).get("model_type"),
            },
            "history": plan.get("history"),
            "split": {
                "strategy": plan.get("split_strategy"),
                "validation_fraction": plan.get("validation_fraction"),
                "test_fraction": plan.get("test_fraction"),
                "seed": plan.get("seed"),
                "assignments": plan.get("split_assignments") or {},
            },
            "mixing": {
                "strategy": plan.get("mixing_strategy"),
                "max_samples_per_dataset": plan.get("max_samples_per_dataset"),
                "manual_dataset_weights": plan.get("manual_dataset_weights") or {},
            },
            "fingerprints": fingerprints,
            "dataset_snapshots": dataset_snapshots,
        }
    )
    index_dir = SAMPLE_INDEX_CACHE_ROOT / key
    manifest_path = index_dir / "manifest.json"
    if manifest_path.exists() and not force:
        if progress_callback:
            progress_callback({"phase": "indexing", "current": 1, "total": 1, "message": "using cached sample index"})
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    all_rows: list[dict[str, Any]] = []
    per_dataset = []
    dataset_files = [(dataset, list_csv_files(dataset["trace_path"])) for dataset in datasets]
    total_files = sum(len(files) for _, files in dataset_files)
    processed_files = 0
    if progress_callback:
        progress_callback({"phase": "indexing", "current": 0, "total": total_files, "message": f"discovered {total_files} trace files"})
    for dataset, files in dataset_files:
        dataset_rows: list[dict[str, Any]] = []
        for path in files:
            file_rows = _valid_rows_for_file(
                path,
                dataset["family"],
                plan.get("history") or {},
                float(dataset.get("fps") or 1.0),
                angle_convention=str(dataset.get("angle_convention") or "unity_zxy"),
                target=str(plan.get("target") or "pose"),
            )
            dataset_rows.extend(file_rows)
            processed_files += 1
            if progress_callback:
                progress_callback(
                    {
                        "phase": "indexing",
                        "current": processed_files,
                        "total": total_files,
                        "dataset_id": dataset.get("id"),
                        "dataset_name": dataset.get("name") or dataset.get("id"),
                        "trace": str(path),
                        "file_samples": len(file_rows),
                        "dataset_samples": len(dataset_rows),
                        "message": f"{dataset.get('name') or dataset.get('id')}: {path.name} -> {len(file_rows)} samples",
                    }
                )
        cap = int(plan.get("max_samples_per_dataset") or 0)
        if cap > 0 and len(dataset_rows) > cap:
            rng = random.Random(int(plan.get("seed") or 0))
            rng.shuffle(dataset_rows)
            dataset_rows = dataset_rows[:cap]
        for row in dataset_rows:
            row["dataset_id"] = dataset["id"]
            row["dataset_name"] = dataset.get("name") or dataset["id"]
        per_dataset.append({"dataset_id": dataset["id"], "dataset_name": dataset.get("name") or dataset["id"], "valid_samples": len(dataset_rows), "trace_count": len(files)})
        all_rows.extend(dataset_rows)
    if not all_rows:
        raise ValueError("No valid samples were found for this plan.")

    split_strategy = str(plan.get("split_strategy") or "by_trace").lower()
    if split_strategy == "random_sample":
        splits = _split_random_samples(
            all_rows,
            int(plan.get("seed") or 0),
            float(plan.get("validation_fraction") or 0.0),
            float(plan.get("test_fraction") or 0.0),
        )
    elif split_strategy == "manual":
        splits = _split_manual(all_rows, dict(plan.get("split_assignments") or {}))
    else:
        splits = _split_grouped_rows(
            all_rows,
            int(plan.get("seed") or 0),
            float(plan.get("validation_fraction") or 0.0),
            float(plan.get("test_fraction") or 0.0),
            split_strategy,
        )
    index_dir.mkdir(parents=True, exist_ok=True)
    split_files = {}
    for split, rows in splits.items():
        path = index_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        split_files[split] = str(path)
    manifest = {
        "cache_key": key,
        "index_dir": str(index_dir),
        "split_files": split_files,
        "counts": {split: len(rows) for split, rows in splits.items()},
        "per_dataset": per_dataset,
        "datasets": dataset_snapshots,
        "split_strategy": split_strategy,
        "mixing_strategy": str(plan.get("mixing_strategy") or "proportional"),
    }
    write_json(manifest_path, manifest)
    return manifest


def read_index_rows(index_file: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(index_file).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
