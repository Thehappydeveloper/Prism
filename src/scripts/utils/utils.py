import os
import shutil
from pathlib import Path
import stat
import time


def handle_remove_readonly(func, path, exc_info):
    """Clear read-only and retry delete."""
    exc_type, exc_value, _ = exc_info
    if func in (os.rmdir, os.remove, os.unlink):
        os.chmod(path, stat.S_IWRITE)
        try:
            func(path)
        except Exception:
            pass
    else:
        raise exc_value


def safe_remove_dir(path: Path, retries=3):
    """Remove a directory safely even if files are locked (Windows-safe)."""
    if path.exists():
        for attempt in range(retries):
            try:
                print(f"Removing existing directory: {path}")
                shutil.rmtree(path, onerror=handle_remove_readonly)
                return
            except PermissionError:
                print(f"⚠️ PermissionError on attempt {attempt+1}, retrying...")
                time.sleep(1)
        print(f"❌ Failed to remove {path} after {retries} retries.")
