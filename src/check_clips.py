"""Quick utility: list which clips have all 20 frames downloaded.

Run from the project root:
    python src/check_clips.py
"""

from pathlib import Path


CLIPS_ROOT = Path("data/tusimple/test_subset/clips/0530")


def main() -> None:
    if not CLIPS_ROOT.exists():
        print(f"ERROR: {CLIPS_ROOT} not found.")
        print("Did you run download_subset.py?")
        return

    full_clips = []
    partial_clips = []

    for clip_dir in sorted(CLIPS_ROOT.iterdir()):
        if not clip_dir.is_dir():
            continue
        n_frames = len(list(clip_dir.glob("*.jpg")))
        if n_frames == 20:
            full_clips.append((clip_dir.name, n_frames))
        else:
            partial_clips.append((clip_dir.name, n_frames))

    print(f"Total clip folders: {len(full_clips) + len(partial_clips)}")
    print(f"Clips with all 20 frames: {len(full_clips)}")
    print(f"Clips with fewer frames:  {len(partial_clips)}")
    print()
    print("Full clips (usable for temporal smoothing):")
    for name, n in full_clips[:10]:
        print(f"  {name}  ({n} frames)")

    if partial_clips and len(partial_clips) < 10:
        print("\nPartial clips:")
        for name, n in partial_clips:
            print(f"  {name}  ({n} frames)")


if __name__ == "__main__":
    main()