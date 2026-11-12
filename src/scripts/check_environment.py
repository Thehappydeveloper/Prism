"""Check the dependencies required by a selected Prism workflow."""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_MODULES = ("numpy", "pandas", "scipy", "tqdm", "openpyxl")
ARGUS_REPORT_ID = "argus_mmsys2027"


def check(label: str, ok: bool, detail: str) -> bool:
    print(f"{'✓' if ok else '✕'} {label}: {detail}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the local Prism environment.")
    parser.add_argument("--training", action="store_true", help="Also require PyTorch for training workflows.")
    parser.add_argument("--ui", action="store_true", help="Also require Node.js and a built browser frontend.")
    parser.add_argument(
        "--precomputed",
        action="store_true",
        help="Also validate the complete bundled Argus MMSys 2027 result cohort and ffmpeg.",
    )
    args = parser.parse_args()
    success = True
    supported_python = (3, 11) <= sys.version_info[:2] <= (3, 13)
    success &= check("Python", supported_python, sys.version.split()[0] + " (supported: 3.11–3.13; 3.13 recommended for Argus)")
    modules = REQUIRED_MODULES + (("torch",) if args.training else ())
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    success &= check("Python packages", not missing, "all required packages installed" if not missing else "missing " + ", ".join(missing))
    if args.ui:
        node = shutil.which("node")
        if node:
            version = subprocess.run([node, "--version"], check=False, capture_output=True, text=True).stdout.strip()
            try:
                major = int(version.removeprefix("v").split(".", 1)[0])
            except ValueError:
                major = 0
            success &= check("Node", major >= 18, version + " (supported: Node 18+; Node 22 used for development)")
        else:
            success &= check("Node", False, "not found")
        dist = PROJECT_ROOT / "src" / "web" / "dist" / "index.html"
        success &= check("Frontend build", dist.exists(), "present" if dist.exists() else "missing; run: cd src/web && npm ci && npm run build")
    results = PROJECT_ROOT / "results"
    try:
        results.mkdir(exist_ok=True)
        probe = results / ".prism_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        writable = True
    except OSError:
        writable = False
    success &= check("Results directory", writable, str(results))
    if args.precomputed:
        ffmpeg = shutil.which("ffmpeg")
        image_magick = shutil.which("magick") or shutil.which("convert")
        success &= check(
            "ffmpeg",
            ffmpeg is not None,
            ffmpeg or "not found; required for the tile-score visual comparison",
        )
        success &= check(
            "ImageMagick",
            image_magick is not None,
            image_magick or "not found; required for tile-score figure labels",
        )
        src_root = str(PROJECT_ROOT / "src")
        if src_root not in sys.path:
            sys.path.insert(0, src_root)
        try:
            from reporting import validate_report

            validation = validate_report(ARGUS_REPORT_ID, results)
            ready = validation.get("status") == "ready"
            detail = (
                f"{validation.get('runs_finished', 0)}/{validation.get('runs_found', 0)} finished runs "
                f"(expected 2,219)"
                if ready
                else "; ".join(str(issue.get("message")) for issue in validation.get("issues") or [])
            )
            success &= check("Argus bundled results", ready, detail or "validation failed")
        except Exception as exc:  # Make setup failures actionable rather than traceback-only.
            success &= check("Argus bundled results", False, f"could not validate: {exc}")
    if not success:
        print("\nFix the failed checks, then run this command again.")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
