"""Download a small subset of TuSimple test images for development.

Strategy:
  1. Parse test_label.json to get list of all annotated clips.
  2. Take the first N clips.
  3. For the first FULL_CLIPS clips, download all 20 frames (for temporal demo).
  4. For the rest, download only frame 20.jpg (the annotated frame).
  5. Save a trimmed annotation file matching what we actually have.

This avoids downloading the full 21.6 GB zip — we only fetch ~150 small JPGs (~30 MB total).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

# === Config ===
ANNOTATION_FILE = Path("data/tusimple/test_label.json")
OUTPUT_DIR = Path("data/tusimple/test_subset")
DATASET = "manideep1108/tusimple"

# How many annotated frames total
NUM_CLIPS = 100

# How many of those clips also need their preceding 19 frames
# (for the temporal smoothing demo — needs consecutive frames)
FULL_CLIPS = 5


def download_one_file(kaggle_path: str, local_dir: Path) -> bool:
    """Download a single file from the Kaggle dataset using the CLI.

    Args:
        kaggle_path: Path inside the Kaggle dataset (e.g. "TUSimple/test_set/clips/.../20.jpg")
        local_dir:   Local directory to save into.

    Returns:
        True on success, False on failure.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle", "datasets", "download", DATASET,
        "-f", kaggle_path,
        "-p", str(local_dir),
        "--unzip",
        "--force",  # overwrite if exists; kaggle CLI gets confused otherwise
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def main() -> None:
    if not ANNOTATION_FILE.exists():
        print(f"ERROR: {ANNOTATION_FILE} not found.")
        print("Download it first with:")
        print("  kaggle datasets download manideep1108/tusimple "
              "-f TUSimple/test_label.json -p data/tusimple --unzip")
        sys.exit(1)

    # Load annotations
    print(f"Reading {ANNOTATION_FILE}...")
    annotations = []
    with open(ANNOTATION_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                annotations.append(json.loads(line))
    print(f"Total annotated frames: {len(annotations)}")

    subset = annotations[:NUM_CLIPS]
    print(f"\nWill download:")
    print(f"  - First {FULL_CLIPS} clips: all 20 frames each "
          f"({FULL_CLIPS * 20} files)")
    print(f"  - Next {NUM_CLIPS - FULL_CLIPS} clips: only frame 20.jpg "
          f"({NUM_CLIPS - FULL_CLIPS} files)")
    total_files = FULL_CLIPS * 20 + (NUM_CLIPS - FULL_CLIPS)
    print(f"  Total: {total_files} files\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save trimmed annotation file matching our subset
    subset_anno_path = OUTPUT_DIR / "test_label_subset.json"
    with open(subset_anno_path, "w") as f:
        for entry in subset:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved subset annotation file: {subset_anno_path}\n")

    # Download images
    success = 0
    failed = 0
    skipped = 0
    start = time.time()

    for i, entry in enumerate(subset):
        raw_file = entry["raw_file"]  # "clips/0530/<clip_id>/20.jpg"
        clip_dir_relative = "/".join(raw_file.split("/")[:-1])

        # How many frames for this clip?
        frames = list(range(1, 21)) if i < FULL_CLIPS else [20]

        for frame_num in frames:
            kaggle_path = f"TUSimple/test_set/{clip_dir_relative}/{frame_num}.jpg"
            local_path = OUTPUT_DIR / clip_dir_relative / f"{frame_num}.jpg"

            if local_path.exists() and local_path.stat().st_size > 0:
                skipped += 1
                continue

            ok = download_one_file(kaggle_path, local_path.parent)
            if ok and local_path.exists():
                success += 1
            else:
                failed += 1
                if failed <= 3:  # Only show first 3 failures
                    print(f"  FAIL: {kaggle_path}")

        # Progress every 10 clips
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start
            rate = (success + skipped) / max(elapsed, 0.1)
            print(f"  {i+1}/{NUM_CLIPS} clips | "
                  f"ok={success} skip={skipped} fail={failed} | "
                  f"{rate:.1f} files/sec")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Success: {success}")
    print(f"  Skipped (already present): {skipped}")
    print(f"  Failed: {failed}")

    # Disk usage report
    result = subprocess.run(
        ["du", "-sh", str(OUTPUT_DIR)],
        capture_output=True, text=True
    )
    print(f"\nFinal subset size: {result.stdout.strip()}")


if __name__ == "__main__":
    main()
