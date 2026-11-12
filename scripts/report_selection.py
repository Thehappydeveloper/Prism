"""Select exactly the run IDs frozen by the report service, when provided."""

from __future__ import annotations

import json
import os
from pathlib import Path


def selected_run_ids() -> set[str] | None:
    raw_path = os.environ.get("PRISM_REPORT_RUN_IDS_FILE")
    if not raw_path:
        return None
    try:
        values = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"Could not read report run selection: {raw_path}") from exc
    if not isinstance(values, list):
        raise RuntimeError("Report run selection must be a JSON list.")
    return {str(value) for value in values}
