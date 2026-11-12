from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.run_index import rebuild_run_index
from reporting import service


class ReportingTests(unittest.TestCase):
    def test_validate_and_build_freeze_completed_group_runs(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "results"
            report_root = root / "reports"
            generated = root / "generated"
            report_root.mkdir()
            (results / "_run_groups.json").parent.mkdir(parents=True, exist_ok=True)
            (results / "_run_groups.json").write_text(json.dumps({"groups": {"g": {"id": "g", "name": "Experiment", "run_ids": ["run_a"]}}}), encoding="utf-8")
            run_dir = results / "run_a"
            run_dir.mkdir()
            (run_dir / "config.json").write_text(json.dumps({"run_group_id": "g", "run_group_name": "Experiment"}), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"status": "finished"}), encoding="utf-8")
            (report_root / "demo.prism-report.json").write_text(json.dumps({"id": "demo", "title": "Demo", "inputs": {"group_names": ["Experiment"]}, "validation": {"expected_run_count": 1}}), encoding="utf-8")
            rebuild_run_index(results)
            original_roots = service.REPORTS_ROOT, service.GENERATED_ROOT
            service.REPORTS_ROOT, service.GENERATED_ROOT = report_root, generated
            try:
                validation = service.validate_report("demo", results)
                self.assertEqual(validation["status"], "ready")
                built = service.build_report("demo", results)
                self.assertTrue(built["built"])
                output = root / built["output_dir"]
                self.assertTrue((output / "inputs.json").exists())
                self.assertTrue((output / "runs.csv").exists())
                self.assertTrue((output / "report-manifest.json").exists())
            finally:
                service.REPORTS_ROOT, service.GENERATED_ROOT = original_roots


if __name__ == "__main__":
    unittest.main()
