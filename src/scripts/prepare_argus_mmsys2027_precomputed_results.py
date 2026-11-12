#!/usr/bin/env python3
"""Create canonical complete groups for the bundled Argus MMSys 2027 runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMBINED_GROUPS = {
    "Paper_Baselines_RankBudget_great_wifi": (
        "Paper_Baselines_RankBudget_great_wifi",
        "Paper_Baselines_RankBudget_great_wifi_LinearPose_Catchup",
    ),
    "Paper_Baselines_CoverageAware_BandwidthQoE_SixRates": (
        "Paper_Baselines_CoverageAware_BandwidthQoE_SixRates",
        "Paper_Baselines_CoverageAware_BandwidthQoE_LinearPose_Catchup",
    ),
}
EXPECTED_COUNTS = {
    "Paper_Baselines_RankBudget_great_wifi": 216,
    "Paper_Baselines_CoverageAware_BandwidthQoE_SixRates": 1296,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=PROJECT_ROOT / "results")
    args = parser.parse_args()
    path = args.results / "_run_groups.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    groups = raw.get("groups")
    if not isinstance(groups, dict):
        raise RuntimeError("Expected a dictionary of run groups.")
    timestamp = datetime.now(timezone.utc).isoformat()
    for canonical_name, source_names in COMBINED_GROUPS.items():
        run_ids: set[str] = set()
        for group in groups.values():
            if isinstance(group, dict) and group.get("name") in source_names:
                run_ids.update(str(run_id) for run_id in (group.get("run_ids") or []))
        expected = EXPECTED_COUNTS[canonical_name]
        if len(run_ids) != expected:
            raise RuntimeError(f"{canonical_name}: expected {expected} runs, found {len(run_ids)}.")
        group_id = f"argus_mmsys2027_precomputed_{canonical_name.lower().replace(' ', '_')}"
        groups[group_id] = {
            "id": group_id,
            "name": canonical_name,
            "description": "Bundled complete Argus MMSys 2027 precomputed cohort.",
            "created_at": timestamp,
            "updated_at": timestamp,
            "run_ids": sorted(run_ids),
        }
        print(f"Prepared {canonical_name}: {len(run_ids)} runs")
    path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
