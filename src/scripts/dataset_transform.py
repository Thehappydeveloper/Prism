from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset Transformation Tool for Prism FoV traces and PLY sequences.")
    parser.add_argument("--trace-root", default="", help="Folder containing FoV trace CSV files.")
    parser.add_argument("--ply-input", default="", help="Reference PLY file or folder containing a PLY sequence.")
    parser.add_argument("--reference-ply", default="", help="Optional reference PLY file or sequence folder. Defaults to --ply-input.")
    parser.add_argument("--output", required=True, help="Output dataset folder.")
    parser.add_argument("--x-column", default="", help="FoV trace X position column.")
    parser.add_argument("--y-column", default="", help="FoV trace Y position column.")
    parser.add_argument("--z-column", default="", help="FoV trace Z position column.")
    parser.add_argument("--height-axis", choices=["x", "y", "z"], default="y", help="Reference PLY axis used as the real-world height extent.")
    parser.add_argument("--output-height-axis", choices=["x", "y", "z"], default=None, help="Output axis that should contain height.")
    parser.add_argument("--real-height-m", type=float, required=True, help="Real-world reference height in meters.")
    parser.add_argument("--traces", action="store_true", help="Transform FoV trace CSV files.")
    parser.add_argument("--point-clouds", action="store_true", help="Transform PLY point-cloud frames.")
    parser.add_argument("--no-height-axis-zero", action="store_true", help="Do not shift the selected output height axis so its minimum starts at 0.")
    parser.add_argument("--positive", action="store_true", help="Shift transformed coordinates into positive coordinate space.")
    parser.add_argument("--export-mode", choices=["scaled", "voxelized", "both", "normalized"], default="scaled", help="Geometry export mode. 'normalized' is accepted as a legacy alias for scaled.")
    parser.add_argument("--voxel-bits", type=int, default=None, help="MPEG PCC geometry bit depth, e.g. 9, 10, 11, 12.")
    parser.add_argument("--workers", type=int, default=1, help="Worker count for PLY sequence processing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from server_app.extra_tools import transform_dataset

    result = transform_dataset(
        trace_root_folder=args.trace_root,
        ply_input_path=args.ply_input,
        output_folder=args.output,
        x_column=args.x_column,
        y_column=args.y_column,
        z_column=args.z_column,
        reference_ply_path=args.reference_ply or args.ply_input,
        height_axis=args.height_axis,
        output_height_axis=args.output_height_axis,
        real_height_m=args.real_height_m,
        transform_traces=args.traces,
        transform_point_clouds=args.point_clouds,
        height_axis_starts_at_zero=not args.no_height_axis_zero,
        positive_coordinates=args.positive,
        voxel_bits=args.voxel_bits,
        export_mode=args.export_mode,
        worker_count=args.workers,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
