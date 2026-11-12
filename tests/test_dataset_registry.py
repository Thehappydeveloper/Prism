from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("pandas", types.ModuleType("pandas"))

from server_app import config


class DatasetRegistryTests(unittest.TestCase):
    def test_save_new_dataset_with_existing_id_preserves_existing_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = root / ".prism_settings"
            datasets_file = settings / "datasets.json"
            world_transforms_file = settings / "world_transforms.json"
            mpd_dir = root / "v3c" / "s4"
            content_dir = root / "ply" / "s4"
            mpd_dir.mkdir(parents=True)
            content_dir.mkdir(parents=True)
            (mpd_dir / "s4.mpd").write_text("<MPD />", encoding="utf-8")
            datasets_file.parent.mkdir(parents=True)
            datasets_file.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "entries": {
                            "longdress": {
                                "id": "longdress",
                                "name": "Longdress",
                                "mpd_folder": "/mnt/storage/v3c_content/longdress",
                                "content_dir": "/mnt/storage/datasets/8i/longdress/longdress/Ply",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with (
                patch.object(config, "DATASET_PROFILES_FILE", datasets_file),
                patch.object(config, "PROJECT_ROOT", root),
                patch("core.world_transform_registry.WORLD_TRANSFORMS_FILE", world_transforms_file),
            ):
                response = config.save_dataset_profile(
                    {
                        "id": "longdress",
                        "name": "S4",
                        "mpd_folder": str(mpd_dir),
                        "content_dir": str(content_dir),
                        "scale_mode": "already_normalized",
                        "already_normalized": True,
                    }
                )

            dataset_ids = {dataset["id"] for dataset in response["datasets"]}
            self.assertIn("longdress", dataset_ids)
            self.assertIn("s4", dataset_ids)

            registry = json.loads(datasets_file.read_text(encoding="utf-8"))
            self.assertEqual(registry["entries"]["longdress"]["name"], "Longdress")
            self.assertEqual(registry["entries"]["s4"]["name"], "S4")

    def test_update_existing_dataset_replaces_entry_when_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = root / ".prism_settings"
            datasets_file = settings / "datasets.json"
            world_transforms_file = settings / "world_transforms.json"
            mpd_dir = root / "v3c" / "longdress"
            content_dir = root / "ply" / "longdress"
            mpd_dir.mkdir(parents=True)
            content_dir.mkdir(parents=True)
            (mpd_dir / "longdress.mpd").write_text("<MPD />", encoding="utf-8")
            datasets_file.parent.mkdir(parents=True)
            datasets_file.write_text(
                json.dumps({"version": 1, "entries": {"longdress": {"id": "longdress", "name": "Old"}}}),
                encoding="utf-8",
            )

            with (
                patch.object(config, "DATASET_PROFILES_FILE", datasets_file),
                patch.object(config, "PROJECT_ROOT", root),
                patch("core.world_transform_registry.WORLD_TRANSFORMS_FILE", world_transforms_file),
            ):
                response = config.save_dataset_profile(
                    {
                        "id": "longdress",
                        "name": "Longdress Updated",
                        "mpd_folder": str(mpd_dir),
                        "content_dir": str(content_dir),
                        "scale_mode": "already_normalized",
                        "already_normalized": True,
                        "update_existing": True,
                    }
                )

            self.assertEqual([dataset["id"] for dataset in response["datasets"]], ["longdress"])
            registry = json.loads(datasets_file.read_text(encoding="utf-8"))
            self.assertEqual(registry["entries"]["longdress"]["name"], "Longdress Updated")


if __name__ == "__main__":
    unittest.main()
