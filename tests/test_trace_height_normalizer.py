from __future__ import annotations

import csv
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.modules.setdefault("numpy", types.SimpleNamespace())
sys.modules.setdefault("pandas", types.SimpleNamespace())

from server_app.extra_tools import normalize_trace_height, transform_dataset


class TraceHeightNormalizerTests(unittest.TestCase):
    def write_csv(self, path: Path, rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["frame", "x", "y", "z", "label"])
            writer.writeheader()
            writer.writerows(rows)

    def write_ply(self, path: Path) -> None:
        path.write_text(
            "\n".join(
                [
                    "ply",
                    "format ascii 1.0",
                    "element vertex 2",
                    "property float x",
                    "property float y",
                    "property float z",
                    "end_header",
                    "0 0 0",
                    "1 2 1",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    def test_normalizes_csvs_to_flat_output_with_collision_suffixes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            output = Path(tmp) / "output"
            ply = Path(tmp) / "reference.ply"
            self.write_ply(ply)
            self.write_csv(
                root / "session1" / "trace.csv",
                [
                    {"frame": "0", "x": "1", "y": "2", "z": "3", "label": "a"},
                    {"frame": "1", "x": "bad", "y": "2", "z": "3", "label": "skip"},
                ],
            )
            self.write_csv(
                root / "session2" / "trace.csv",
                [{"frame": "0", "x": "4", "y": "5", "z": "6", "label": "b"}],
            )

            result = normalize_trace_height(
                root_folder=str(root),
                output_folder=str(output),
                x_column="x",
                y_column="y",
                z_column="z",
                ply_path=str(ply),
                height_axis="y",
                real_height_m=4.0,
            )

            self.assertEqual(result["summary"]["files_found"], 2)
            self.assertEqual(result["summary"]["files_processed"], 2)
            self.assertEqual(result["summary"]["rows_transformed"], 2)
            self.assertEqual(result["summary"]["rows_skipped"], 1)
            self.assertEqual(result["scaling"]["scale_factor_m_per_unit"], 2.0)
            self.assertTrue((output / "trace.csv").exists())
            self.assertTrue((output / "trace_1.csv").exists())
            self.assertFalse((output / "session1").exists())
            self.assertTrue(any(warning["kind"] == "renamed_file" for warning in result["warnings"]))

            with (output / "trace.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["x"], "2")
            self.assertEqual(rows[0]["y"], "6")
            self.assertEqual(rows[0]["z"], "4")
            self.assertEqual(rows[0]["label"], "a")
            self.assertEqual(
                result["scaling"]["axis_swap"],
                {
                    "enabled": True,
                    "source_height_axis": "z",
                    "output_height_axis": "y",
                },
            )

    def test_z_height_axis_keeps_coordinate_columns_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            output = Path(tmp) / "output"
            ply = Path(tmp) / "reference.ply"
            self.write_ply(ply)
            self.write_csv(
                root / "trace.csv",
                [{"frame": "0", "x": "1", "y": "2", "z": "3", "label": "a"}],
            )

            result = normalize_trace_height(
                root_folder=str(root),
                output_folder=str(output),
                x_column="x",
                y_column="y",
                z_column="z",
                ply_path=str(ply),
                height_axis="z",
                real_height_m=4.0,
            )

            with (output / "trace.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["x"], "4")
            self.assertEqual(rows[0]["y"], "8")
            self.assertEqual(rows[0]["z"], "12")
            self.assertEqual(
                result["scaling"]["axis_swap"],
                {
                    "enabled": False,
                    "source_height_axis": "z",
                    "output_height_axis": "z",
                },
            )

    def test_transform_dataset_uses_first_indexed_ply_frame_and_exports_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            ply_dir = Path(tmp) / "ply"
            output = Path(tmp) / "output"
            ply_dir.mkdir()
            self.write_ply(ply_dir / "frame_0002.ply")
            (ply_dir / "frame_0010.ply").write_text(
                "\n".join(
                    [
                        "ply",
                        "format ascii 1.0",
                        "element vertex 2",
                        "property float x",
                        "property float y",
                        "property float z",
                        "end_header",
                        "10 10 10",
                        "11 12 11",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            self.write_csv(
                root / "trace.csv",
                [{"frame": "0", "x": "1", "y": "2", "z": "3", "label": "a"}],
            )

            result = transform_dataset(
                trace_root_folder=str(root),
                ply_input_path=str(ply_dir),
                output_folder=str(output),
                x_column="x",
                y_column="y",
                z_column="z",
                reference_ply_path=str(ply_dir),
                height_axis="y",
                output_height_axis="y",
                real_height_m=4.0,
                transform_traces=True,
                transform_point_clouds=True,
                positive_coordinates=True,
                voxel_bits=9,
                export_mode="both",
                worker_count=2,
            )

            self.assertEqual(result["scaling"]["point_cloud"]["reference"]["reference_frame_number"], 2)
            self.assertEqual(result["point_clouds"]["summary"]["frames_processed"], 2)
            self.assertEqual(result["point_clouds_voxelized"]["summary"]["frames_processed"], 2)
            self.assertEqual([item["frame_number"] for item in result["point_clouds"]["output_files"]], [2, 10])
            self.assertTrue((output / "ply_normalized" / "frame_0002.ply").exists())
            self.assertTrue((output / "ply_voxelized" / "frame_0010.ply").exists())
            with (output / "traces_normalized" / "trace.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["x"], "2")
            self.assertEqual(rows[0]["y"], "6")
            self.assertEqual(rows[0]["z"], "4")

    def test_transform_dataset_resamples_traces_to_reference_ply_frame_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            ply_dir = Path(tmp) / "ply"
            output = Path(tmp) / "output"
            ply_dir.mkdir()
            self.write_ply(ply_dir / "frame_0002.ply")
            self.write_ply(ply_dir / "frame_0010.ply")
            self.write_csv(
                root / "trace.csv",
                [
                    {"frame": "1", "x": "0", "y": "0", "z": "0", "label": "a"},
                    {"frame": "2", "x": "10", "y": "10", "z": "10", "label": "b"},
                    {"frame": "3", "x": "20", "y": "20", "z": "20", "label": "c"},
                    {"frame": "4", "x": "30", "y": "30", "z": "30", "label": "d"},
                ],
            )

            result = transform_dataset(
                trace_root_folder=str(root),
                output_folder=str(output),
                x_column="x",
                y_column="y",
                z_column="z",
                reference_ply_path=str(ply_dir),
                height_axis="y",
                output_height_axis="y",
                real_height_m=4.0,
                transform_traces=True,
                transform_point_clouds=False,
                fov_resample_enabled=True,
            )

            self.assertTrue(result["trace_resampling"]["enabled"])
            self.assertTrue(result["trace_resampling"]["inferred"])
            self.assertEqual(result["trace_resampling"]["target_samples"], 2)
            with (output / "trace.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual([row["frame"] for row in rows], ["2", "10"])
            self.assertEqual(rows[0]["x"], "0")
            self.assertEqual(rows[1]["x"], "60")
            self.assertTrue(result["traces"]["output_files"][0]["resampled"])


if __name__ == "__main__":
    unittest.main()
