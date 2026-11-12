from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from .config import TRAINING_EXPORTS_ROOT, TRAINING_MODELS_ROOT, TRAINING_RUNS_ROOT, now_iso, read_json, write_json

_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def _normalize_model_name(value: str, field_name: str = "model name") -> str:
    name = str(value or "").strip()
    if not name:
        return ""
    if not _MODEL_NAME_RE.fullmatch(name) or name in {".", ".."}:
        raise ValueError(f"{field_name} may only contain letters, numbers, dots, dashes, and underscores.")
    return name


def _metadata_from_run_model_file(run_dir: Path) -> dict[str, Any] | None:
    metadata = read_json(run_dir / "model.json", None)
    if not isinstance(metadata, dict):
        return None
    model_id = str(metadata.get("id") or "").strip()
    if not model_id:
        model_id = f"{run_dir.name}_model"
        metadata["id"] = model_id
    metadata.setdefault("run_id", run_dir.name)
    metadata.setdefault("checkpoint", str(run_dir / "checkpoints" / "best.pt"))
    return metadata


def _with_model_display_name(metadata: dict[str, Any]) -> dict[str, Any]:
    """Add a stable plan-name fallback for older model metadata."""
    item = dict(metadata)
    if item.get("name"):
        return item
    run_id = str(item.get("run_id") or "").strip()
    if not run_id:
        return item
    plan = read_json(TRAINING_RUNS_ROOT / run_id / "plan.json", None)
    if isinstance(plan, dict) and plan.get("name"):
        item["name"] = str(plan["name"])
    return item


def list_training_models() -> list[dict[str, Any]]:
    models_by_id: dict[str, dict[str, Any]] = {}
    if TRAINING_MODELS_ROOT.exists():
        for model_dir in sorted(TRAINING_MODELS_ROOT.iterdir(), reverse=True):
            if not model_dir.is_dir():
                continue
            metadata = read_json(model_dir / "metadata.json", None)
            if isinstance(metadata, dict):
                item = _with_model_display_name(metadata)
                models_by_id[str(item.get("id") or model_dir.name)] = item
    if TRAINING_RUNS_ROOT.exists():
        for run_dir in sorted(TRAINING_RUNS_ROOT.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            metadata = _metadata_from_run_model_file(run_dir)
            if metadata:
                item = _with_model_display_name(metadata)
                models_by_id.setdefault(str(item.get("id")), item)
    return sorted(
        models_by_id.values(),
        key=lambda item: str(item.get("created_at") or item.get("id") or ""),
        reverse=True,
    )


def get_model_metadata(model_id: str) -> dict[str, Any]:
    metadata = read_json(TRAINING_MODELS_ROOT / model_id / "metadata.json", None)
    if not isinstance(metadata, dict):
        raise ValueError("Trained model was not found.")
    return metadata


def export_model(model_id: str, export_name: str | None = None) -> dict[str, Any]:
    metadata = get_model_metadata(model_id)
    export_id = _normalize_model_name(export_name or "", "Export name") or model_id
    family = str(metadata.get("family") or "")
    if family not in {"fov", "bandwidth"}:
        raise ValueError("Model family is not exportable.")
    checkpoint = Path(str(metadata.get("checkpoint") or ""))
    if not checkpoint.exists():
        raise ValueError("Model checkpoint is missing; cannot export.")
    stats = metadata.get("stats") or {}
    if not stats.get("input_shape") or not stats.get("output_size"):
        raise ValueError("Model preprocessing metadata is incomplete; cannot export.")
    if family == "fov" and stats.get("target_kind") == "tile_visibility" and not stats.get("output_tile_ids"):
        raise ValueError("Tile-visibility FoV exports require output tile ids.")
    if family == "fov" and stats.get("target_kind") != "tile_visibility" and int(stats.get("output_size") or 0) != 7:
        raise ValueError("FoV exports require a 7-value position + quaternion output.")
    if family == "bandwidth" and int(stats.get("output_size") or 0) != 1:
        raise ValueError("Bandwidth exports require a single throughput output.")
    export_dir = TRAINING_EXPORTS_ROOT / export_id
    export_dir.mkdir(parents=True, exist_ok=True)
    export_metadata = dict(metadata)
    export_metadata.update(
        {
            "id": export_id,
            "source_model_id": model_id,
            "name": export_id,
            "exported": True,
            "exported_at": now_iso(),
            "predictor_class": "TrainedFovPredictor" if family == "fov" else "TrainedBandwidthPredictor",
            "predictor_params": {"model_id": export_id},
        }
    )
    write_json(export_dir / "metadata.json", export_metadata)
    source_checkpoint = checkpoint
    if source_checkpoint.exists():
        import shutil

        shutil.copy2(source_checkpoint, export_dir / "model.pt")
        export_metadata["checkpoint"] = str(export_dir / "model.pt")
        write_json(export_dir / "metadata.json", export_metadata)
    model_dir = TRAINING_MODELS_ROOT / model_id
    metadata["exported"] = True
    metadata["exported_at"] = export_metadata["exported_at"]
    metadata["exported_as"] = export_id
    if export_name:
        metadata["name"] = export_id
    write_json(model_dir / "metadata.json", metadata)
    return export_metadata


def delete_model(model_id: str) -> dict[str, Any]:
    model_dir = TRAINING_MODELS_ROOT / model_id
    metadata = read_json(model_dir / "metadata.json", None)
    export_ids = {model_id}
    if isinstance(metadata, dict):
        exported_as = _normalize_model_name(str(metadata.get("exported_as") or ""), "Export name")
        if exported_as:
            export_ids.add(exported_as)
    deleted = False
    paths = [model_dir, *(TRAINING_EXPORTS_ROOT / export_id for export_id in sorted(export_ids))]
    for path in paths:
        if path.exists() and path.is_dir():
            import shutil

            shutil.rmtree(path)
            deleted = True
    return {"model_id": model_id, "deleted": deleted}


def load_exported_model_metadata(model_id: str) -> dict[str, Any]:
    metadata = read_json(TRAINING_EXPORTS_ROOT / model_id / "metadata.json", None)
    if not isinstance(metadata, dict):
        metadata = get_model_metadata(model_id)
    return metadata
