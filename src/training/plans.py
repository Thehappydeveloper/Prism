from __future__ import annotations

from typing import Any

from .config import TRAINING_PLANS_FILE, new_id, now_iso, read_json, write_json
from .models import is_trace_only_model, requires_point_cloud_features
from .registry import get_training_dataset


DEFAULT_MODEL_CONFIG = {
    "model_type": "mlp",
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": 0.0,
}

DEFAULT_TRAINING_CONFIG = {
    "epochs": 5,
    "batch_size": 64,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "loss": "mse",
    "early_stopping_patience": 0,
    "checkpoint_frequency": 1,
    "device": "auto",
    "num_workers": 0,
    "trace_cache_entries": 256,
    "mixed_precision": False,
}

DEFAULT_HISTORY_CONFIG = {
    "history_duration_s": 2.0,
    "history_step_s": 0.5,
    "sample_stride_s": 0.5,
    "prediction_horizon_s": 1.0,
    "include_current": True,
    "fixed_history_samples": 0,
}


def _registry() -> dict[str, Any]:
    raw = read_json(TRAINING_PLANS_FILE, {"version": 1, "entries": {}})
    entries = raw.get("entries") if isinstance(raw, dict) else {}
    return {"version": 1, "entries": entries if isinstance(entries, dict) else {}}


def list_training_plans() -> list[dict[str, Any]]:
    out = []
    for plan_id, entry in sorted(_registry()["entries"].items()):
        if isinstance(entry, dict):
            item = dict(entry)
            item["id"] = str(item.get("id") or plan_id)
            out.append(item)
    return out


def get_training_plan(plan_id: str | None) -> dict[str, Any] | None:
    if not plan_id:
        return None
    entry = _registry()["entries"].get(str(plan_id))
    return dict(entry) if isinstance(entry, dict) else None


def save_training_plan(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Training plan payload must be an object/dict.")
    family = str(payload.get("family") or "").strip().lower()
    if family not in {"fov", "bandwidth"}:
        raise ValueError("Plan family must be 'fov' or 'bandwidth'.")
    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError("Plan name is required.")
    dataset_ids = [str(v) for v in (payload.get("dataset_ids") or []) if str(v).strip()]
    if not dataset_ids:
        raise ValueError("Select at least one training dataset.")
    for dataset_id in dataset_ids:
        dataset = get_training_dataset(dataset_id)
        if not dataset:
            raise ValueError(f"Training dataset '{dataset_id}' was not found.")
        if dataset.get("family") != family:
            raise ValueError(f"Dataset '{dataset_id}' does not match plan family '{family}'.")
    model = {**DEFAULT_MODEL_CONFIG, **dict(payload.get("model") or {})}
    target = str(payload.get("target") or ("pose" if family == "fov" else "future_bandwidth"))
    if family == "fov" and target == "tile_visibility" and is_trace_only_model(model):
        raise ValueError("Trace-only FoV models (MLP/GRU/LSTM) must predict XYZ + quaternion pose, not tile visibility.")
    if family == "fov" and requires_point_cloud_features(model):
        missing_point_cloud = []
        for dataset_id in dataset_ids:
            dataset = get_training_dataset(dataset_id) or {}
            if not dataset.get("content_dir") or not dataset.get("reference_ply"):
                missing_point_cloud.append(dataset_id)
        if missing_point_cloud:
            raise ValueError(
                "Point-cloud feature models require FoV datasets with content_dir and reference_ply: "
                + ", ".join(missing_point_cloud)
            )

    plan_id = str(payload.get("id") or "").strip() or new_id("tpl")
    entry = {
        "id": plan_id,
        "family": family,
        "name": name,
        "dataset_ids": dataset_ids,
        "split_strategy": str(payload.get("split_strategy") or "by_trace"),
        "validation_fraction": float(payload.get("validation_fraction") or 0.15),
        "test_fraction": float(payload.get("test_fraction") or 0.15),
        "mixing_strategy": str(payload.get("mixing_strategy") or "proportional"),
        "manual_dataset_weights": dict(payload.get("manual_dataset_weights") or {}),
        "split_assignments": dict(payload.get("split_assignments") or {}),
        "max_samples_per_dataset": int(payload.get("max_samples_per_dataset") or 0),
        "seed": int(payload.get("seed") or 0),
        "history": {**DEFAULT_HISTORY_CONFIG, **dict(payload.get("history") or {})},
        "model": model,
        "training": {**DEFAULT_TRAINING_CONFIG, **dict(payload.get("training") or {})},
        "target": target,
        "notes": str(payload.get("notes") or ""),
        "updated_at": now_iso(),
    }
    registry = _registry()
    entries = dict(registry["entries"])
    entries[plan_id] = entry
    write_json(TRAINING_PLANS_FILE, {"version": 1, "entries": entries})
    return {"plans": list_training_plans(), "settings_path": str(TRAINING_PLANS_FILE)}


def delete_training_plan(plan_id: str) -> dict[str, Any]:
    registry = _registry()
    entries = dict(registry["entries"])
    entries.pop(str(plan_id), None)
    write_json(TRAINING_PLANS_FILE, {"version": 1, "entries": entries})
    return {"plans": list_training_plans(), "settings_path": str(TRAINING_PLANS_FILE)}


def training_plans_payload() -> dict[str, Any]:
    return {"plans": list_training_plans(), "settings_path": str(TRAINING_PLANS_FILE)}
