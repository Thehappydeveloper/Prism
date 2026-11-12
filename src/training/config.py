from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from server_app.config import PROJECT_ROOT, RESULTS_ROOT, SETTINGS_ROOT, path_for_client, safe_path

TRAINING_DATASETS_FILE = SETTINGS_ROOT / "training_datasets.json"
TRAINING_PLANS_FILE = SETTINGS_ROOT / "training_plans.json"
TRAINING_ROOT = RESULTS_ROOT / "training"
TRAINING_RUNS_ROOT = TRAINING_ROOT / "runs"
TRAINING_MODELS_ROOT = TRAINING_ROOT / "models"
TRAINING_EXPORTS_ROOT = TRAINING_ROOT / "exports"
TRAINING_CACHE_ROOT = PROJECT_ROOT / ".prism_cache" / "training"
SAMPLE_INDEX_CACHE_ROOT = TRAINING_CACHE_ROOT / "sample_indexes"


def now_iso() -> str:
    return datetime.now().isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%y%m%d_%H%M%S')}_{uuid4().hex[:4]}"


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp_path.replace(path)


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def digest_payload(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def file_fingerprint(path_value: str | Path) -> dict[str, Any]:
    path = safe_path(path_value)
    if not path.exists():
        return {"path": path_for_client(path), "missing": True}
    stat = path.stat()
    return {
        "path": path_for_client(path),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }


def tree_fingerprint(path_value: str | Path, patterns: tuple[str, ...] = ("*.csv",)) -> dict[str, Any]:
    root = safe_path(path_value)
    if root.is_file():
        return {"root": path_for_client(root), "files": [file_fingerprint(root)]}
    files = []
    if root.exists():
        for pattern in patterns:
            for path in sorted(root.rglob(pattern)):
                if path.is_file():
                    files.append(file_fingerprint(path))
    return {"root": path_for_client(root), "files": files}


def list_csv_files(path_value: str | Path) -> list[Path]:
    path = safe_path(path_value)
    if path.is_file():
        return [path] if path.suffix.lower() == ".csv" else []
    if not path.exists():
        return []
    return sorted(p for p in path.rglob("*.csv") if p.is_file())

