"""Download TuSimple test_set, extract only the 20-frame clips we want, delete zip.

Downloads the full ~21.6 GB zip, but immediately:
  1. Extracts only the annotation file + a curated 5 clips' worth of frames
  2. Deletes the zip file
This keeps peak disk usage under ~25 GB.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path

DATA_ROOT = Path("data/tusimple")
EXTRACT_TO = DATA_ROOT / "test_subset"
ZIP_PATH = DATA_ROOT / "tusimple.zip"

# Clips that have all 20 frames (we already identified these earlier)
TARGET_CLIPS = [
    "1492626397007603377_0",
    "1492626617873533069_0",
    "1492626760788443246_0",
    "1492627171538356342_0",
    "1492627288467128445_0",
]


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    # Step 1: Download annotation file (small, 3.6 MB)
    print("Step 1: Downloading test_label.json...")
    subprocess.run(
        ["kaggle", "datasets", "download", "manideep1108/tusimple",
         "-f", "TUSimple/test_label.json", "-p", str(DATA_ROOT),
         "--unzip", "--force"],
        check=True,
    )
    print(f"  {DATA_ROOT / 'test_label.json'} present: {(DATA_ROOT / 'test_label.json').exists()}")

    # Step 2: Download the full zip (~21.6 GB)
    print("\nStep 2: Downloading full TuSimple zip (~21.6 GB)...")
    print("        This takes 3-5 minutes on Codespaces' fast network.")
    subprocess.run(
        ["kaggle", "datasets", "download", "manideep1108/tusimple",
         "-p", str(DATA_ROOT), "--force", "--quiet"],
        check=True,
    )
    print(f"  Zip size: {ZIP_PATH.stat().st_size / 1e9:.1f} GB")

    # Step 3: Extract ONLY the frames we need (5 clips × 20 frames = 100 files)
    print(
        f"\nStep 3: Extracting frames for {len(TARGET_CLIPS)} target clips...")
    EXTRACT_TO.mkdir(parents=True, exist_ok=True)

    n_extracted = 0
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        for member in zf.namelist():
            if not member.endswith(".jpg"):
                continue
            # Check if this file belongs to one of our target clips
            for clip_id in TARGET_CLIPS:
                if f"/{clip_id}/" in member:
                    # Strip the "TUSimple/test_set/" prefix
                    relative = member.replace("TUSimple/test_set/", "")
                    dest = EXTRACT_TO / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, open(dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    n_extracted += 1
                    break

    print(
        f"  Extracted {n_extracted} files (expected {len(TARGET_CLIPS) * 20} = {len(TARGET_CLIPS)*20})")

    # Step 4: Delete the zip to free disk
    print("\nStep 4: Deleting zip to free disk space...")
    ZIP_PATH.unlink()

    # Step 5: Make a tiny annotation file containing only our target clips
    print("\nStep 5: Building subset annotation file...")
    annotations = []
    with open(DATA_ROOT / "test_label.json") as f:
        for line in f:
            line = line.strip()
            if line:
                annotations.append(json.loads(line))

    # Filter to target clips
    subset = []
    for entry in annotations:
        for clip_id in TARGET_CLIPS:
            if f"/{clip_id}/" in entry["raw_file"]:
                subset.append(entry)
                break

    subset_path = EXTRACT_TO / "test_label_subset.json"
    with open(subset_path, "w") as f:
        for entry in subset:
            f.write(json.dumps(entry) + "\n")
    print(f"  Saved {len(subset)} annotations to {subset_path}")

    # Final disk usage report
    print("\nDone. Final state:")
    subprocess.run(["du", "-sh", str(EXTRACT_TO)], check=False)
    subprocess.run(["df", "-h", "/workspaces"], check=False)


if __name__ == "__main__":
    main()
