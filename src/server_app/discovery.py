from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Dict, Iterable, List, Type

from abr import baselines as abr_baselines
from abr.base import ABR
from bandwith_predictor import baselines as bandwidth_baselines
from bandwith_predictor.base import BandwidthPredictor
from core.network_profiles import network_profile_payload
from fov_predictors import baselines as fov_baselines
from fov_predictors.base import FovPredictor
from fov_predictors.display import FOV_PREDICTOR_LABELS
from qoe import heuristics as qoe_functions
from qoe.base import QoEFunction
from qoe.heuristics import WeightedQoEHeuristic
from training.config import TRAINING_EXPORTS_ROOT, read_json

from .config import (
    ANGLE_CONVENTIONS,
    PROJECT_ROOT,
    headset_settings,
    list_dataset_profiles,
    navis_status,
    path_for_client,
    safe_path,
    trace_settings,
)


def list_files(base: Path, patterns: Iterable[str]) -> List[str]:
    files: List[str] = []
    if not base.exists():
        return files
    for pattern in patterns:
        files.extend([path_for_client(p.resolve()) for p in base.rglob(pattern) if p.is_file()])
    return sorted(files)


def collect_files(bases: Iterable[Path], patterns: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for base in bases:
        for path in list_files(base, patterns):
            if path in seen:
                continue
            seen.add(path)
            out.append(path)
    return sorted(out)


def list_subclasses(module, base_type: Type) -> List[str]:
    names = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        if not issubclass(obj, base_type) or obj is base_type:
            continue
        if obj.__name__.startswith("_"):
            continue
        names.append(obj.__name__)
    return sorted(set(names))


def abr_requirements() -> Dict[str, bool]:
    reqs: Dict[str, bool] = {}
    for _, obj in inspect.getmembers(abr_baselines, inspect.isclass):
        if obj.__module__ != abr_baselines.__name__:
            continue
        if not issubclass(obj, ABR) or obj is ABR:
            continue
        reqs[obj.__name__] = bool(getattr(obj, "abr_requires_predictor", False))
    return reqs


def abr_predictor_support() -> Dict[str, bool]:
    support: Dict[str, bool] = {}
    for _, obj in inspect.getmembers(abr_baselines, inspect.isclass):
        if obj.__module__ != abr_baselines.__name__:
            continue
        if not issubclass(obj, ABR) or obj is ABR:
            continue
        support[obj.__name__] = bool(getattr(obj, "abr_supports_predictor", False))
    return support


def annotation_name(annotation: Any) -> str:
    if annotation is inspect._empty:
        return ""
    text = str(annotation)
    if text.startswith("typing."):
        text = text.replace("typing.", "")
    return text


def param_kind(default: Any, annotation: Any) -> str | None:
    if isinstance(default, bool):
        return "boolean"
    if isinstance(default, int) and not isinstance(default, bool):
        return "integer"
    if isinstance(default, float):
        return "number"
    if isinstance(default, str):
        return "string"
    if default is None:
        annot = annotation_name(annotation).lower()
        if "bool" in annot:
            return "boolean"
        if "int" in annot:
            return "integer"
        if "float" in annot:
            return "number"
        if "str" in annot:
            return "string"
        return "number"
    return None


def class_param_spec(cls: Type) -> List[Dict[str, Any]]:
    init = getattr(cls, "__init__", None)
    if init is None:
        return []
    try:
        sig = inspect.signature(init)
    except (TypeError, ValueError):
        return []

    metadata_by_name = getattr(cls, "PARAM_METADATA", {}) or {}
    fields: List[Dict[str, Any]] = []
    for param in sig.parameters.values():
        if param.name == "self":
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if param.default is inspect._empty:
            continue
        meta = dict(metadata_by_name.get(param.name) or {})
        if meta.get("hidden"):
            continue
        kind = param_kind(param.default, param.annotation)
        if kind is None:
            continue
        fields.append(
            {
                "name": param.name,
                "label": meta.get("label") or param.name.replace("_", " "),
                "kind": kind,
                "default": param.default,
                "annotation": annotation_name(param.annotation),
                "category": meta.get("category") or "General",
                "help": meta.get("help") or "",
                "advanced": bool(meta.get("advanced")),
            }
        )
    return fields


def algorithm_param_specs(module, base_type: Type) -> Dict[str, List[Dict[str, Any]]]:
    specs: Dict[str, List[Dict[str, Any]]] = {}
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ != module.__name__:
            continue
        if not issubclass(obj, base_type) or obj is base_type:
            continue
        if obj.__name__.startswith("_"):
            continue
        specs[obj.__name__] = class_param_spec(obj)
    return specs


def exported_model_choices(family: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not TRAINING_EXPORTS_ROOT.exists():
        return []
    for export_dir in sorted(TRAINING_EXPORTS_ROOT.iterdir()):
        if not export_dir.is_dir():
            continue
        metadata = read_json(export_dir / "metadata.json", None)
        if not isinstance(metadata, dict):
            continue
        if str(metadata.get("family") or "") != family:
            continue
        model_id = str(metadata.get("id") or export_dir.name).strip()
        if not model_id:
            continue
        source = str(metadata.get("source_model_id") or "").strip()
        label = str(metadata.get("name") or model_id).strip()
        model_type = str((metadata.get("model") or {}).get("model_type") or "").strip()
        details = " · ".join(part for part in (model_type, source) if part)
        rows.append(
            {
                "value": model_id,
                "label": f"{label} ({details})" if details else label,
                "source": source or model_id,
                "alias": "1" if source and model_id != source else "0",
            }
        )
    rows.sort(key=lambda item: (item["source"], item["alias"] != "1", item["value"]))
    choices: List[Dict[str, str]] = []
    seen_sources: set[str] = set()
    for row in rows:
        source = row["source"]
        if source in seen_sources:
            continue
        seen_sources.add(source)
        choices.append({"value": row["value"], "label": row["label"]})
    return choices


def attach_exported_model_choices(specs: Dict[str, List[Dict[str, Any]]], predictor_name: str, choices: List[Dict[str, str]]) -> None:
    if predictor_name not in specs:
        return
    for field in specs[predictor_name]:
        if field.get("name") == "model_id":
            field["choices"] = choices
            if choices:
                field["default"] = choices[0]["value"]
            return


def has_csv_under(paths: Iterable[Path]) -> bool:
    for base in paths:
        if not base.exists():
            continue
        if any(base.rglob("*.csv")):
            return True
    return False


def ensure_traces_ready() -> Dict[str, bool]:
    network_root = PROJECT_ROOT / "data" / "network_traces"
    fov_root = PROJECT_ROOT / "data" / "fov_traces"
    fov_dataset = fov_root / "6DoF-Video-FoV-Dataset"
    network_root.mkdir(parents=True, exist_ok=True)
    fov_root.mkdir(parents=True, exist_ok=True)

    status = {
        "network": has_csv_under([network_root]),
        "fov": has_csv_under([fov_root, fov_dataset]),
    }
    return status


def qoe_options() -> List[Dict[str, Any]]:
    models: List[Dict[str, Any]] = []
    for _, obj in inspect.getmembers(qoe_functions, inspect.isclass):
        if obj.__module__ != qoe_functions.__name__:
            continue
        if not issubclass(obj, QoEFunction) or obj is QoEFunction:
            continue
        if inspect.isabstract(obj) or obj.__name__.startswith("_"):
            continue
        try:
            meta = obj.metadata()
        except Exception as exc:  # pylint: disable=broad-except
            meta = {"id": obj.__name__, "label": obj.__name__, "error": str(exc)}
        meta.setdefault("id", obj.__name__)
        meta.setdefault("label", obj.__name__)
        models.append(meta)
    return sorted(models, key=lambda m: m.get("label") or m.get("id"))


def resolve_qoe_model(model_id: str | None, params: Dict | None) -> QoEFunction:
    fallback = WeightedQoEHeuristic
    target_cls = None
    if model_id:
        for _, obj in inspect.getmembers(qoe_functions, inspect.isclass):
            if obj.__module__ != qoe_functions.__name__:
                continue
            if not issubclass(obj, QoEFunction) or obj is QoEFunction:
                continue
            if inspect.isabstract(obj) or obj.__name__.startswith("_"):
                continue
            if getattr(obj, "id", None) == model_id or obj.__name__ == model_id:
                target_cls = obj
                break
    cls = target_cls or fallback
    try:
        return cls(overrides=params or {})
    except TypeError:
        inst = cls()
        try:
            inst.apply_overrides(params or {})
        except Exception:  # pylint: disable=broad-except
            pass
        return inst
    except Exception:  # pylint: disable=broad-except
        return fallback(overrides=params or {})


def discover_options() -> Dict[str, List[str] | Dict]:
    settings = trace_settings()
    network_root = safe_path(settings["network_root"])
    fov_root = safe_path(settings["fov_root"])
    network_root.mkdir(parents=True, exist_ok=True)
    fov_root.mkdir(parents=True, exist_ok=True)

    fov_traces = collect_files([fov_root], ["*.csv"])
    network_traces = collect_files([network_root], ["*.csv"])
    dataset_profiles = list_dataset_profiles()
    fov_root_abs = fov_root.resolve()
    fov_set = set(fov_traces)
    filtered_network = []
    for p in network_traces:
        abs_path = (PROJECT_ROOT / p).resolve()
        if p in fov_set or fov_root_abs in abs_path.parents:
            continue
        filtered_network.append(p)
    network_traces = filtered_network or network_traces

    headset = headset_settings()
    fov_param_specs = algorithm_param_specs(fov_baselines, FovPredictor)
    bandwidth_param_specs = algorithm_param_specs(bandwidth_baselines, BandwidthPredictor)
    attach_exported_model_choices(fov_param_specs, "TrainedFovPredictor", exported_model_choices("fov"))
    attach_exported_model_choices(bandwidth_param_specs, "TrainedBandwidthPredictor", exported_model_choices("bandwidth"))

    return {
        "mpds": list_files(PROJECT_ROOT, ["*.mpd"]),
        "network_traces": network_traces,
        "network_profiles": network_profile_payload(),
        "fov_traces": fov_traces,
        "network_root": path_for_client(network_root),
        "fov_root": path_for_client(fov_root),
        "abr_classes": list_subclasses(abr_baselines, ABR),
        "fov_predictors": list_subclasses(fov_baselines, FovPredictor),
        "fov_predictor_labels": FOV_PREDICTOR_LABELS,
        "bandwidth_predictors": list_subclasses(bandwidth_baselines, BandwidthPredictor),
        "abr_requires_predictor": abr_requirements(),
        "abr_supports_predictor": abr_predictor_support(),
        "abr_param_specs": algorithm_param_specs(abr_baselines, ABR),
        "fov_param_specs": fov_param_specs,
        "bandwidth_param_specs": bandwidth_param_specs,
        "headset_presets": headset.get("presets", []),
        "default_headset_preset": headset.get("default_preset"),
        "angle_convention": headset.get("angle_convention"),
        "angle_conventions": ANGLE_CONVENTIONS,
        "qoe_functions": qoe_options(),
        "navis": navis_status(),
        "datasets": dataset_profiles,
    }
