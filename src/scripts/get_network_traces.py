import shutil
import subprocess
from pathlib import Path
import pandas as pd

try:
    # Imported as part of the `scripts` package (e.g., from src/server.py)
    from .utils.utils import safe_remove_dir
except ImportError:
    # Executed directly: `python src/scripts/get_network_traces.py`
    from utils.utils import safe_remove_dir


def convert_traces_to_csv(target_dir: Path):
    for file_path in target_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() == ".csv":
            # Already converted
            continue

        try:
            df = pd.read_csv(
                file_path,
                sep=r"\s+",
                header=None,
                names=["Time_s", "Bandwidth_Mbps"],
                engine="python",
            )

            # Determine subfolder
            parts_lower = [p.lower() for p in file_path.parts]
            if "4g" in parts_lower:
                csv_dir = target_dir / "4G"
            elif "5g" in parts_lower:
                csv_dir = target_dir / "5G"
            else:
                csv_dir = target_dir / "unknown"
            csv_dir.mkdir(parents=True, exist_ok=True)

            csv_path = csv_dir / f"{file_path.stem}.csv"
            df.to_csv(csv_path, index=False)

            file_path.unlink()
            print(f"Converted {file_path.name} → {csv_path.relative_to(target_dir)}")

        except Exception as e:
            print(f"⚠️ Failed to convert {file_path}: {e}")


def main():
    project_root = Path(__file__).resolve().parents[2]
    target_dir = project_root / "data" / "network_traces" / "Lumous5G"
    target_dir.mkdir(parents=True, exist_ok=True)

    repo_url = "https://github.com/Thehappydeveloper/artifact.git"
    subfolder = Path("Video-Streaming/Network-Traces/Lumous5G")
    tmp_dir = project_root / "artifact_tmp"

    # Skip if already exists
    if any(target_dir.iterdir()):
        print(f"Dataset already exists in '{target_dir}'. Skipping clone.")
    else:
        print("Cloning dataset repository...")
        if tmp_dir.exists():
            print(f"Temporary folder {tmp_dir} already exists, removing it...")
            safe_remove_dir(tmp_dir)
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(tmp_dir)], check=True
        )
        src_path = tmp_dir / subfolder

        if src_path.exists():
            shutil.move(str(src_path), str(target_dir))
            print(f"Moved {src_path} → {target_dir}")
        else:
            print(f"⚠️ Expected folder '{src_path}' not found in repo.")

        safe_remove_dir(tmp_dir)

    convert_traces_to_csv(target_dir)
    print(f"✅ All traces converted to CSV inside '{target_dir}'.")


if __name__ == "__main__":
    main()
