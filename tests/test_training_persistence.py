from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

numpy_stub = types.ModuleType("numpy")
numpy_stub.pi = 3.141592653589793
numpy_stub.mean = lambda values: sum(values) / len(values)
numpy_stub.random = types.SimpleNamespace(seed=lambda *args, **kwargs: None, get_state=lambda: None)
sys.modules.setdefault("numpy", numpy_stub)

trace_columns_stub = types.ModuleType("core.trace_columns")
trace_columns_stub.save_trace_column_settings = lambda *args, **kwargs: None
trace_columns_stub.trace_column_config_payload = lambda *args, **kwargs: {}
sys.modules.setdefault("core.trace_columns", trace_columns_stub)

mpd_parser_stub = types.ModuleType("core.mpd_parser")
mpd_parser_stub.V3CMpd = object
sys.modules.setdefault("core.mpd_parser", mpd_parser_stub)

world_transform_stub = types.ModuleType("core.world_transform_registry")
world_transform_stub.WORLD_TRANSFORMS_FILE = Path("/tmp/world_transforms.json")
world_transform_stub.detect_header_transform = lambda *args, **kwargs: None
world_transform_stub.get_world_transform = lambda *args, **kwargs: None
world_transform_stub.list_world_transforms = lambda *args, **kwargs: []
world_transform_stub.normalize_content_dir = lambda *args, **kwargs: None
world_transform_stub.set_world_transform = lambda *args, **kwargs: None
sys.modules.setdefault("core.world_transform_registry", world_transform_stub)

datasets_stub = types.ModuleType("training.datasets")
datasets_stub.IndexedTraceDataset = object
datasets_stub.TraceFrameCache = object
datasets_stub.compute_stats = lambda *args, **kwargs: {}
sys.modules.setdefault("training.datasets", datasets_stub)

indexing_stub = types.ModuleType("training.indexing")
indexing_stub.build_sample_index = lambda *args, **kwargs: {}
indexing_stub.read_index_rows = lambda *args, **kwargs: []
sys.modules.setdefault("training.indexing", indexing_stub)

plans_stub = types.ModuleType("training.plans")
plans_stub.get_training_plan = lambda *args, **kwargs: None
sys.modules.setdefault("training.plans", plans_stub)

from training import export, trainer


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class TrainingPersistenceTests(unittest.TestCase):
    def test_list_training_runs_recovers_run_without_status_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "train_260101_000000_abcd"
            run_dir.mkdir(parents=True)
            _write_json(run_dir / "plan.json", {"id": "plan_a", "family": "fov"})
            _write_json(
                run_dir / "sample_index.json",
                {"counts": {"train": 10, "validation": 2, "test": 1}, "per_dataset": [{"dataset_id": "ds_a", "train": 10}]},
            )
            _write_json(
                run_dir / "metrics.json",
                {
                    "epochs": [{"epoch": 1, "train_loss": 0.4, "validation_mae": 0.2, "elapsed_s": 3.5}],
                    "test": {"mae": 0.18},
                },
            )
            (run_dir / "training.log").write_text("queued\nstarted\nfinished\n", encoding="utf-8")

            with patch.object(trainer, "TRAINING_RUNS_ROOT", root / "runs"), patch.object(trainer, "TRAINING_JOBS", {}):
                runs = trainer.list_training_runs()

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["run_id"], "train_260101_000000_abcd")
        self.assertEqual(runs[0]["plan_id"], "plan_a")
        self.assertEqual(runs[0]["status"], "finished")
        self.assertEqual(runs[0]["split_counts"]["train"], 10)
        self.assertEqual(runs[0]["progress"]["latest_validation_mae"], 0.2)
        self.assertEqual(runs[0]["log"][-1], "finished")

    def test_list_training_models_falls_back_to_run_model_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "train_260101_000000_abcd"
            _write_json(
                run_dir / "model.json",
                {
                    "id": "model_from_run",
                    "run_id": "train_260101_000000_abcd",
                    "plan_id": "plan_a",
                    "family": "bandwidth",
                    "created_at": "2026-01-01T00:00:00",
                },
            )

            with (
                patch.object(export, "TRAINING_MODELS_ROOT", root / "models"),
                patch.object(export, "TRAINING_RUNS_ROOT", root / "runs"),
            ):
                models = export.list_training_models()

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["id"], "model_from_run")
        self.assertEqual(models[0]["run_id"], "train_260101_000000_abcd")

    def test_export_model_can_use_export_name_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "models" / "model_a"
            checkpoint = model_dir / "model.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            _write_json(
                model_dir / "metadata.json",
                {
                    "id": "model_a",
                    "family": "fov",
                    "stats": {"input_shape": [15, 7], "output_size": 7, "target_kind": "pose"},
                    "checkpoint": str(checkpoint),
                    "created_at": "2026-01-01T00:00:00",
                    "exported": False,
                },
            )

            with (
                patch.object(export, "TRAINING_MODELS_ROOT", root / "models"),
                patch.object(export, "TRAINING_EXPORTS_ROOT", root / "exports"),
            ):
                metadata = export.export_model("model_a", "lstm")
                source_metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
                exported_metadata = json.loads((root / "exports" / "lstm" / "metadata.json").read_text(encoding="utf-8"))

        self.assertEqual(metadata["id"], "lstm")
        self.assertEqual(metadata["source_model_id"], "model_a")
        self.assertEqual(metadata["predictor_params"]["model_id"], "lstm")
        self.assertEqual(source_metadata["exported_as"], "lstm")
        self.assertEqual(source_metadata["name"], "lstm")
        self.assertEqual(exported_metadata["checkpoint"], str(root / "exports" / "lstm" / "model.pt"))
