import os
import subprocess
from pathlib import Path
import pandas as pd

try:
    from .utils.utils import safe_remove_dir
except ImportError:
    from utils.utils import safe_remove_dir

# -------------------------
# Download and rename dataset
# -------------------------

# -------------------------
# Configuration
# -------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
FOV_ROOT = DATA_DIR / "fov_traces"
TARGET_DIR = FOV_ROOT / "6DoF-Video-FoV-Dataset"
REPO_TMP = PROJECT_ROOT / "repo_tmp"
REPO_URL = "https://github.com/Thehappydeveloper/6DoF_Video_FoV_Dataset.git"

RENAME_MAP = {
    "longdress.xlsx": "Longdress.csv",
    "loot.xlsx": "Loot.csv",
    "redandblack.xlsx": "RedAndBlack.csv",
    "soldier.xlsx": "Soldier.csv",
}


# -------------------------
# Helper Functions
# -------------------------


def clone_and_convert_dataset():
    """Clone the dataset repo and convert Excel files to CSV."""
    FOV_ROOT.mkdir(parents=True, exist_ok=True)
    expected_csvs = [TARGET_DIR / new for new in RENAME_MAP.values()]

    # Skip donwload if all CSVs exist
    if all(f.exists() for f in expected_csvs):
        print("✅ All CSV files already exist. Skipping download.")
        return

    print(f"📁 Preparing dataset directory at: {TARGET_DIR}")
    os.makedirs(TARGET_DIR, exist_ok=True)

    # Remove old temporary repo if it exists
    safe_remove_dir(REPO_TMP)

    print(f"⬇️ Cloning dataset from {REPO_URL}")
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(REPO_TMP)], check=True
    )

    # Convert Excel -> CSV
    for old_name, new_name in RENAME_MAP.items():
        old_path = REPO_TMP / old_name
        new_path = TARGET_DIR / new_name

        if not old_path.exists():
            print(f"⚠️ Warning: {old_name} not found in the repo.")
            continue
        try:
            df = pd.read_excel(old_path)
            df.to_csv(new_path, index=False)
            print(f"✅ Converted {old_name} → {new_name}")
        except Exception as e:
            print(f"❌ Failed to convert {old_name}: {e}")

    # Clean up temporary repo
    safe_remove_dir(REPO_TMP)
    print("🎉 Dataset downloaded and converted successfully.")


# -------------------------
# Split CSVs by participant and loops
# -------------------------
def split_csv_by_participants_and_loops(
    csv_path: Path, output_base_dir: Path, loop_size=300
):
    print(f"\n📂 Processing: {csv_path.name}")
    df = pd.read_csv(csv_path)
    dataset_name = csv_path.stem

    if "Participant" not in df.columns:
        raise ValueError(f"{csv_path} does not contain a 'Participant' column")

    for participant, part_df in df.groupby("Participant"):
        out_dir = output_base_dir / dataset_name / str(participant)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Skip already processed
        existing_loops = list(out_dir.glob("loop_*.csv"))
        if existing_loops:
            print(
                f"⏩ Skipping participant {participant}: already processed ({len(existing_loops)} loops found)"
            )
            continue

        num_frames = len(part_df)
        num_loops = num_frames // loop_size

        if num_loops == 0:
            print(
                f"⚠️ Skipping participant {participant}: not enough frames ({num_frames})"
            )
            continue

        if "FrameNumber" in part_df.columns:
            part_df = part_df.sort_values("FrameNumber")

        for loop_idx in range(num_loops):
            start = loop_idx * loop_size
            end = (loop_idx + 1) * loop_size
            loop_df = part_df.iloc[start:end].copy()

            if "FrameNumber" in loop_df.columns:
                loop_df["FrameNumber"] = range(loop_size)

            out_file = out_dir / f"loop_{loop_idx+1}.csv"
            loop_df.to_csv(out_file, index=False)

        print(f"✅ Participant {participant}: {num_loops} loops saved.")

    # ---------------------------
    # 🔥 DELETE the original CSV
    # ---------------------------
    try:
        csv_path.unlink()
        print(f"🗑️ Deleted global CSV: {csv_path.name}")
    except Exception as e:
        print(f"⚠️ Failed to delete {csv_path.name}: {e}")



def process_all_csvs():
    """Split all CSVs in the dataset folder by participant and loop."""
    for csv_file in TARGET_DIR.glob("*.csv"):
        split_csv_by_participants_and_loops(csv_file, TARGET_DIR)

    print("\n🎯 All CSVs split and global CSVs removed successfully.")


# -------------------------
# Main Entry Point
# -------------------------


def main():
    clone_and_convert_dataset()
    process_all_csvs()


if __name__ == "__main__":
    main()
