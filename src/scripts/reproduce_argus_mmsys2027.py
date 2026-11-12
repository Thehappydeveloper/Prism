#!/usr/bin/env python3
"""Reproduce the Argus MMSys 2027 report from bundled runs or prepare a rerun."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import threading
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from reporting import build_report, validate_report  # noqa: E402


REPORT_ID = "argus_mmsys2027"
MODEL_DIRECTORIES = (
    "model_260723_171401_5ca2",
    "model_260723_171707_8df3",
)
BASE_MODULES = ("numpy", "pandas", "scipy", "tqdm", "openpyxl")


def print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


class BuildProgress:
    """Small terminal progress display for the fixed Argus parent-report build."""

    def __init__(self, total_phases: int = 10) -> None:
        self.total_phases = total_phases
        self.phase = 0
        self.label = "Starting"
        self.started_at = time.monotonic()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def update(self, label: str) -> None:
        with self._lock:
            self.phase = min(self.phase + 1, self.total_phases)
            self.label = label
            self._render()

    def finish(self) -> None:
        with self._lock:
            self.phase = self.total_phases
            self.label = "Complete"
            self._render(final=True)
        self._stop.set()
        self._thread.join(timeout=1)

    def fail(self) -> None:
        with self._lock:
            self.label = "Failed"
            self._render(final=True)
        self._stop.set()
        self._thread.join(timeout=1)

    def _heartbeat(self) -> None:
        while not self._stop.wait(10):
            with self._lock:
                self._render()

    def _render(self, *, final: bool = False) -> None:
        width = 24
        filled = round(width * self.phase / self.total_phases)
        elapsed = int(time.monotonic() - self.started_at)
        minutes, seconds = divmod(elapsed, 60)
        line = f"[{('=' * filled).ljust(width, '-')}] {self.phase}/{self.total_phases} {self.label} ({minutes}:{seconds:02d})"
        if sys.stdout.isatty() and not final:
            print(line, end="\r", flush=True)
        else:
            print(line, flush=True)


def check_runtime(*, require_torch: bool = False) -> bool:
    failures: list[str] = []
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        failures.append(f"Python {sys.version.split()[0]} is unsupported; use Python 3.13.")
    modules = BASE_MODULES + (("torch",) if require_torch else ())
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures.append(f"{name} cannot be imported ({exc}); install the applicable locked requirements file.")
    if failures:
        print("Environment is not ready:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return False
    return True


def precomputed(results_root: Path) -> int:
    """Validate and build the self-contained reviewer-facing report."""
    if not check_runtime():
        return 1
    missing_tools = []
    if shutil.which("ffmpeg") is None:
        missing_tools.append("ffmpeg")
    if not (shutil.which("magick") or shutil.which("convert")):
        missing_tools.append("ImageMagick (magick or convert)")
    if missing_tools:
        print(f"ERROR: install {', '.join(missing_tools)} for the tile-score visual comparison.", file=sys.stderr)
        return 1
    validation = validate_report(REPORT_ID, results_root)
    print_json(validation)
    if validation.get("status") != "ready":
        print("\nBundled results are incomplete. Restore results/_run_groups.json and all 2,219 run directories.", file=sys.stderr)
        return 1
    progress = BuildProgress()
    print("\nBuilding the parent report. The current phase and elapsed time will update below.")
    progress.start()
    try:
        result = build_report(REPORT_ID, results_root, progress=progress.update)
    except Exception:
        progress.fail()
        raise
    if result.get("status") == "error":
        progress.fail()
    else:
        progress.finish()
    print_json(result)
    if result.get("status") == "error":
        return 1
    print("\nReport build complete. Review the immutable manifest, input snapshot, tables, and child-report figures in:")
    print(result.get("output_dir", "docs/argus_mmsys2027/generated/argus_mmsys2027/<build-id>"))
    return 0


def rerun_readiness() -> int:
    """Check prerequisites before the reviewer imports the six plans in the UI."""
    failures: list[str] = []
    print("[1/4] Checking Python and required packages …", flush=True)
    if not check_runtime(require_torch=True):
        return 1
    print("[2/4] Checking released MLP and LSTM models …", flush=True)
    for name in MODEL_DIRECTORIES:
        if not (PROJECT_ROOT / "results" / "training" / "models" / name / "model.pt").is_file():
            failures.append(f"released model missing: results/training/models/{name}/model.pt")
    print("[3/4] Checking the installed Argus data …", flush=True)
    from get_argus_mmsys2027_data import is_complete

    if not is_complete(PROJECT_ROOT / "data"):
        failures.append("Argus data is not installed; run python src/scripts/get_argus_mmsys2027_data.py.")
    print("[4/4] Locating the six reviewed batch plans …", flush=True)
    plans = sorted((PROJECT_ROOT / "docs" / "argus_mmsys2027" / "batch-plans").glob("*.prism-batch-plan.json"))
    if failures:
        print("Rerun prerequisites are not ready:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Rerun prerequisites are ready. Start Prism, then use the six Generate buttons in Argus MMSys 2027 reproduction.")
    print("The UI deliberately loads each JSON plan for review before launching it.")
    for plan in plans:
        print(f"- {plan.relative_to(PROJECT_ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--precomputed", action="store_true", help="Validate bundled artifacts and build the parent report.")
    mode.add_argument("--rerun", action="store_true", help="Check full re-simulation prerequisites and list its six plans.")
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results", help="Run-artifact directory for --precomputed (default: results).")
    args = parser.parse_args()
    return precomputed(args.results_root) if args.precomputed else rerun_readiness()


if __name__ == "__main__":
    raise SystemExit(main())
