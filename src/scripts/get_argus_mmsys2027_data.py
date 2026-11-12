#!/usr/bin/env python3
"""Install the exact external inputs used by the Argus MMSys 2027 plans.

The default is the immutable published Zenodo version used for the Argus paper.
Use ``--record-id`` only to deliberately install a different version.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZENODO_RECORD_ID = "24118210"
ZENODO_API = "https://zenodo.org/api/records/{record_id}"
ARGUS_MMSYS2027_DATA_DIR = "argus_mmsys2027"

# These are checksums of the files prepared for the Zenodo upload.  Do not
# change them when publishing the record; a mismatch is a reproducibility
# failure, not a warning.
FILES = {
    "s7.zip": "881490a27446c2166f827755ffe9ac015fef2d878799507a4a057106e219c853",
    "S15.zip": "23a7320b084a3e7e43e422749eada3c33fc2d5951c103d438caed7e467ed2c9b",
    "s7.mpd": "a94be2f201e448324f336cba3145c4e31e5cf2f3371a4204f60fb33deb52dc46",
    "s15.mpd": "8592b362843329359ccd8549b87a7418f98b085e56efd284418df5324223f161",
    "mazelab_fov_traces.zip": "ef854cfceff502893c89024de1fad69939aef2aacb7a5240ae8310165360e201",
}
ZENODO_MD5 = {
    "s7.zip": "1304b0de22921a66d9c55a952179e100",
    "S15.zip": "cc6ff78e32d07373c0a549aa06b9620f",
    "s7.mpd": "3c590fe672266f3ca793db52279e3c0b",
    "s15.mpd": "5b0f730459e82d7f851ad0a6f34c1fb3",
    "mazelab_fov_traces.zip": "960499e929ed312ea1127a5af27cff3c",
}

# The source archive uses this original directory name.  Paper plans retained
# the historical short name, so install it under that name without changing
# the trace bytes.
TRACE_RENAMES = {
    "Experiment_7/DefaultParticipant_202479164949697": "Experiment_7/_202479164949697",
}

PAPER_TILE_EXTENT_STATS = {
    "source": "mpd_tile_bounds",
    "tile_bounds_count": 2232,
    "axis": {
        "x": {"min": -45.24037170410156, "max": -9.268698692321777, "extent": 35.971673011779785},
        "y": {"min": -1.0191881656646729, "max": 2.0, "extent": 3.019188165664673},
        "z": {"min": -54.793853759765625, "max": -9.835256576538086, "extent": 44.95859718322754},
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_size(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{value:.0f} B"
        value /= 1024
    return f"{value:.1f} TiB"


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return "<1s"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


@dataclass
class DownloadProgress:
    total_bytes: int
    file_count: int
    completed_bytes: int = 0
    completed_files: int = 0
    started_at: float = 0.0
    last_update_at: float = 0.0

    def __post_init__(self) -> None:
        self.started_at = time.monotonic()

    def report(self, name: str, file_bytes: int, file_total: int, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_update_at < 0.4:
            return
        self.last_update_at = now
        overall_bytes = self.completed_bytes + file_bytes
        elapsed = max(now - self.started_at, 0.001)
        rate = overall_bytes / elapsed
        remaining = max(self.total_bytes - overall_bytes, 0)
        eta = remaining / rate if rate > 0 else float("inf")
        file_percent = 100 * file_bytes / file_total if file_total else 0.0
        overall_percent = 100 * overall_bytes / self.total_bytes if self.total_bytes else 0.0
        eta_text = format_duration(eta) if eta != float("inf") else "estimating…"
        line = (
            f"[{self.completed_files + 1}/{self.file_count}] {name}: {file_percent:5.1f}% "
            f"({format_size(file_bytes)}/{format_size(file_total)}) | overall {overall_percent:5.1f}% "
            f"| {format_size(rate)}/s | ETA {eta_text}"
        )
        print(line, end="\r" if sys.stdout.isatty() else "\n", flush=True)

    def finish(self, name: str, file_total: int) -> None:
        self.report(name, file_total, file_total, force=True)
        self.completed_bytes += file_total
        self.completed_files += 1
        if sys.stdout.isatty():
            print()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def required_traces(data_dir: Path) -> list[Path]:
    """Read the trace paths from the committed paper plans, not a duplicate list."""
    plans_dir = PROJECT_ROOT / "docs" / "argus_mmsys2027" / "batch-plans"
    traces: set[Path] = set()
    for plan_path in plans_dir.glob("*.prism-batch-plan.json"):
        plan = read_json(plan_path)
        pairs = ((plan.get("config") or {}).get("mediaPairs") or [])
        for pair in pairs:
            for raw_path in pair.get("fov_traces") or []:
                path = Path(str(raw_path))
                if path.parts[:2] == ("data", "fov_traces") and "mazelab_fov_traces_verified" in path.parts:
                    traces.add(data_dir / path.relative_to("data"))
    return sorted(traces)


def is_complete(data_dir: Path) -> bool:
    scenes = ("s7", "s15")
    media_ready = all(
        (data_dir / ARGUS_MMSYS2027_DATA_DIR / scene / f"{scene}.mpd").is_file()
        and len(list((data_dir / ARGUS_MMSYS2027_DATA_DIR / scene / "GlobalCompletePCDs").glob("*.ply"))) == 2700
        for scene in scenes
    )
    return media_ready and all(trace.is_file() for trace in required_traces(data_dir))


def zenodo_files(record_id: str) -> dict[str, tuple[str, int]]:
    request = urllib.request.Request(
        ZENODO_API.format(record_id=record_id), headers={"Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(request) as response:
            record = json.load(response)
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not read Zenodo record {record_id}: {exc}") from exc
    found: dict[str, tuple[str, int]] = {}
    for entry in record.get("files") or []:
        name = entry.get("key") or entry.get("filename")
        url = (entry.get("links") or {}).get("content") or (entry.get("links") or {}).get("self")
        size = entry.get("size")
        checksum = str(entry.get("checksum") or "").removeprefix("md5:")
        if isinstance(name, str) and isinstance(url, str) and isinstance(size, int):
            if name in ZENODO_MD5 and checksum != ZENODO_MD5[name]:
                raise RuntimeError(
                    f"Zenodo metadata checksum mismatch for {name}: expected md5:{ZENODO_MD5[name]}, "
                    f"got {entry.get('checksum') or 'none'}."
                )
            found[name] = (url, size)
    missing = sorted(set(FILES) - set(found))
    if missing:
        raise RuntimeError(
            f"Zenodo record {record_id} is missing required file(s): {', '.join(missing)}. "
            "Publish the files with their documented names."
        )
    return found


def download(
    url: str,
    destination: Path,
    expected_sha256: str,
    expected_size: int,
    progress: DownloadProgress,
) -> None:
    print(f"Downloading {destination.name} ({format_size(expected_size)}) …")
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "Prism-paper-data-fetcher"})
    bytes_written = 0
    with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
            digest.update(chunk)
            bytes_written += len(chunk)
            progress.report(destination.name, bytes_written, expected_size)
    if bytes_written != expected_size:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download size mismatch for {destination.name}: expected {format_size(expected_size)}, "
            f"received {format_size(bytes_written)}."
        )
    actual = digest.hexdigest()
    if actual != expected_sha256:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum mismatch for {destination.name}: expected {expected_sha256}, got {actual}."
        )
    progress.finish(destination.name, expected_size)


def extract_archive(archive: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    target_root = target.resolve()
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        for member in members:
            destination = (target / member.filename).resolve()
            if destination != target_root and target_root not in destination.parents:
                raise RuntimeError(f"Unsafe archive entry in {archive.name}: {member.filename}")
        files = [member for member in members if not member.is_dir()]
        total_bytes = sum(member.file_size for member in files)
        completed = 0
        started_at = time.monotonic()
        last_update_at = 0.0
        print(f"Extracting {archive.name} ({len(files):,} files, {format_size(total_bytes)}) …")
        for index, member in enumerate(files, start=1):
            bundle.extract(member, target)
            completed += member.file_size
            now = time.monotonic()
            if index != len(files) and now - last_update_at < 0.4:
                continue
            last_update_at = now
            elapsed = max(now - started_at, 0.001)
            rate = completed / elapsed
            remaining = max(total_bytes - completed, 0)
            eta = remaining / rate if rate else float("inf")
            percent = 100 * completed / total_bytes if total_bytes else 100.0
            eta_text = format_duration(eta) if eta != float("inf") else "estimating…"
            line = (
                f"Extracting {archive.name}: {percent:5.1f}% "
                f"({format_size(completed)}/{format_size(total_bytes)}, {index:,}/{len(files):,} files) "
                f"| {format_size(rate)}/s | ETA {eta_text}"
            )
            print(line, end="\r" if sys.stdout.isatty() and index != len(files) else "\n", flush=True)


def install_payload(staging: Path, data_dir: Path) -> None:
    media_root = data_dir / ARGUS_MMSYS2027_DATA_DIR
    traces_root = data_dir / "fov_traces" / "mazelab_fov_traces_verified"
    extracted = staging / "extracted"
    extract_archive(staging / "s7.zip", extracted / "s7")
    extract_archive(staging / "S15.zip", extracted / "s15")
    extract_archive(staging / "mazelab_fov_traces.zip", extracted / "traces")

    source_trace_root = extracted / "traces" / "mazelab_fov_traces"
    if not source_trace_root.is_dir():
        raise RuntimeError("Trace archive does not contain mazelab_fov_traces/.")
    for old, new in TRACE_RENAMES.items():
        old_path = source_trace_root / old
        new_path = source_trace_root / new
        if old_path.is_dir() and not new_path.exists():
            old_path.rename(new_path)

    destinations = {
        extracted / "s7" / "GlobalCompletePCDs": media_root / "s7" / "GlobalCompletePCDs",
        extracted / "s15" / "GlobalCompletePCDs": media_root / "s15" / "GlobalCompletePCDs",
        source_trace_root: traces_root,
    }
    for source, destination in destinations.items():
        if not source.exists():
            raise RuntimeError(f"Archive is missing expected payload: {source.relative_to(extracted)}")
        if destination.exists():
            raise RuntimeError(
                f"Refusing to merge into existing {destination}. Remove the incomplete destination and rerun."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    for scene in ("s7", "s15"):
        shutil.copy2(staging / f"{scene}.mpd", media_root / scene / f"{scene}.mpd")


def register_profiles(data_dir: Path) -> None:
    """Register the two local paper datasets and their identity transforms."""
    settings = PROJECT_ROOT / ".prism_settings"
    datasets_path = settings / "datasets.json"
    registry = read_json(datasets_path)
    entries = registry.get("entries") if isinstance(registry.get("entries"), dict) else {}
    transforms_path = settings / "world_transforms.json"
    transforms = read_json(transforms_path)
    transform_entries = transforms.get("entries") if isinstance(transforms.get("entries"), dict) else {}
    for scene in ("s7", "s15"):
        content_dir = (data_dir / ARGUS_MMSYS2027_DATA_DIR / scene / "GlobalCompletePCDs").resolve()
        mpd_dir = (data_dir / ARGUS_MMSYS2027_DATA_DIR / scene).resolve()
        profile = {
            "id": scene,
            "name": scene.upper(),
            "mpd_folder": str(mpd_dir),
            "mpd": str(mpd_dir / f"{scene}.mpd"),
            "content_dir": str(content_dir),
            "scale_mode": "already_normalized",
            "angle_convention": "unity_zxy",
            "calibration": {"mode": "already_normalized"},
            "motion_limits": {"max_speed": 3.6391, "max_acceleration": 6.268},
            # Informational UI summary; simulation reads tile bounds from MPD.
            "extent_stats": PAPER_TILE_EXTENT_STATS,
            "transform": {
                "scale": 1.0,
                "translation": [0.0, 0.0, 0.0],
                "units": "meters",
                "source": "paper_data_fetcher",
                "notes": "MMSys paper reproduction dataset",
                "apply_to_fov_trace": False,
            },
            "apply_to_fov_trace": False,
            "already_normalized": True,
            "notes": "MMSys paper reproduction dataset",
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        entries[scene] = profile
        transform_entries[str(content_dir)] = {
            **profile["transform"],
            "metadata": {
                "dataset_id": scene,
                "dataset_name": scene.upper(),
                "scale_mode": "already_normalized",
                "angle_convention": "unity_zxy",
                "calibration": profile["calibration"],
                "motion_limits": profile["motion_limits"],
            },
        }
    write_json(datasets_path, {"version": 1, "entries": entries})
    write_json(transforms_path, {"version": 1, "entries": transform_entries})


def configure_paper_runtime_settings() -> None:
    """Enable the only server setting required by a paper report.

    Worker counts remain local: plans explicitly define their semantic
    segment-worker configuration, whereas PLY I/O parallelism is hardware
    tuning and should not be copied from the author's workstation.
    """
    settings_path = PROJECT_ROOT / ".prism_settings" / "prism.json"
    settings = read_json(settings_path)
    settings["record_fov_stage_latency"] = True
    write_json(settings_path, settings)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-id", help="Zenodo record ID (overrides PRISM_ARGUS_ZENODO_RECORD_ID and the paper default).")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data", help="Prism data directory.")
    parser.add_argument("--keep-archives", action="store_true", help="Keep verified ZIP downloads in a temporary directory.")
    parser.add_argument("--no-register", action="store_true", help="Do not create/update the s7 and s15 local dataset profiles.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    if is_complete(data_dir):
        if not args.no_register:
            register_profiles(data_dir)
            configure_paper_runtime_settings()
        print(f"Argus MMSys 2027 data already present in {data_dir}; nothing downloaded.")
        return 0
    record_id = (
        args.record_id
        or os.environ.get("PRISM_ARGUS_ZENODO_RECORD_ID")
        or os.environ.get("PRISM_MMSYS2027_ZENODO_RECORD_ID")  # legacy override
        or DEFAULT_ZENODO_RECORD_ID
    )
    if not record_id:
        raise SystemExit(
            "No Zenodo record ID configured. Run with --record-id <record-id> or set "
            "PRISM_ARGUS_ZENODO_RECORD_ID."
        )
    try:
        remote_files = zenodo_files(record_id)
        progress = DownloadProgress(
            total_bytes=sum(size for _, size in remote_files.values()), file_count=len(FILES)
        )
        # Stage beside data/ so verified extraction can be moved atomically
        # without temporarily duplicating 250 GB of PLY files on another FS.
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="argus-mmsys2027-", dir=data_dir.parent) as temporary:
            staging = Path(temporary)
            for name, checksum in FILES.items():
                url, size = remote_files[name]
                download(url, staging / name, checksum, size, progress)
            print("All downloads verified. Extracting and installing archives …")
            install_payload(staging, data_dir)
            if args.keep_archives:
                archive_dir = data_dir / ARGUS_MMSYS2027_DATA_DIR / "archives"
                archive_dir.mkdir(parents=True, exist_ok=True)
                for name in FILES:
                    shutil.copy2(staging / name, archive_dir / name)
        if not is_complete(data_dir):
            raise RuntimeError("Installation finished but the Argus MMSys 2027 completeness check failed.")
        if not args.no_register:
            print("Registering portable s7/s15 dataset profiles …")
            register_profiles(data_dir)
            configure_paper_runtime_settings()
    except (OSError, RuntimeError, urllib.error.URLError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Installed and verified Argus MMSys 2027 inputs in {data_dir / ARGUS_MMSYS2027_DATA_DIR}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
