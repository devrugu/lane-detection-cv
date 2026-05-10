"""TuSimple Lane Detection Dataset Loader.

This module loads the TuSimple test annotations and provides clean access
to image paths and lane coordinates. We use it throughout the project to
avoid scattering JSON parsing logic everywhere.

The TuSimple annotation format (one JSON per line):
    {
      "raw_file": "clips/0530/<clip_id>/20.jpg",
      "h_samples": [160, 170, 180, ..., 710],   # y-coordinates (rows)
      "lanes": [
        [-2, -2, 850, 845, ...],   # x-coordinates for each y in h_samples
        [-2, 632, 625, 617, ...],  # -2 means "lane not visible at this y"
        ...
      ]
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LaneAnnotation:
    """A single annotated frame with its lane data.

    Attributes:
        image_path: Absolute path to the .jpg image on disk.
        h_samples:  List of y-coordinates where lanes are sampled.
                    Always 56 values from 160 to 710 in steps of 10.
        lanes:      List of lanes. Each lane is a list of x-coordinates,
                    one per y in h_samples. Value -2 means "not visible".
        raw_file:   Original relative path from the annotation file.
                    Useful for matching predictions to ground truth.
    """

    image_path: Path
    h_samples: list[int]
    lanes: list[list[int]]
    raw_file: str

    def get_lane_points(self, lane_idx: int) -> list[tuple[int, int]]:
        """Return (x, y) tuples for one lane, skipping invisible points (-2).

        Example:
            lane.get_lane_points(0)  # returns [(850, 180), (845, 190), ...]
        """
        return [
            (x, y)
            for x, y in zip(self.lanes[lane_idx], self.h_samples)
            if x != -2
        ]

    @property
    def num_lanes(self) -> int:
        """Total number of lanes annotated (usually 4, sometimes 3 or 5)."""
        return len(self.lanes)

    @property
    def num_visible_lanes(self) -> int:
        """Number of lanes that have at least one visible point."""
        return sum(1 for lane in self.lanes if any(x != -2 for x in lane))


class TuSimpleDataset:
    """Loads TuSimple annotations and resolves them to image file paths.

    Usage:
        dataset = TuSimpleDataset(
            annotation_file="data/tusimple/test_label.json",
            image_root="data/tusimple/TUSimple/test_set",
        )
        print(f"Loaded {len(dataset)} annotated frames")
        sample = dataset[0]
        print(sample.image_path)
        print(sample.num_visible_lanes)
    """

    def __init__(self, annotation_file: str | Path, image_root: str | Path):
        self.annotation_file = Path(annotation_file)
        self.image_root = Path(image_root)

        if not self.annotation_file.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {self.annotation_file}"
            )

        # Read all annotations into memory. The file is small (~3.6 MB)
        # so this is fine — no need for lazy loading.
        self._annotations: list[LaneAnnotation] = []
        with open(self.annotation_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                self._annotations.append(
                    LaneAnnotation(
                        image_path=self.image_root / data["raw_file"],
                        h_samples=data["h_samples"],
                        lanes=data["lanes"],
                        raw_file=data["raw_file"],
                    )
                )

    def __len__(self) -> int:
        return len(self._annotations)

    def __getitem__(self, idx: int) -> LaneAnnotation:
        return self._annotations[idx]

    def __iter__(self):
        return iter(self._annotations)

    def existing_only(self) -> list[LaneAnnotation]:
        """Return only annotations whose image actually exists on disk.

        Useful when working with a subset (we may have annotations for
        thousands of frames but only downloaded images for some).
        """
        return [a for a in self._annotations if a.image_path.exists()]

    def stats(self) -> dict:
        """Quick summary statistics — useful for debugging and reports."""
        all_lane_counts = [a.num_visible_lanes for a in self._annotations]
        existing = self.existing_only()
        return {
            "total_annotations": len(self._annotations),
            "images_on_disk": len(existing),
            "min_visible_lanes": min(all_lane_counts) if all_lane_counts else 0,
            "max_visible_lanes": max(all_lane_counts) if all_lane_counts else 0,
            "h_samples_range": (
                min(self._annotations[0].h_samples),
                max(self._annotations[0].h_samples),
            ) if self._annotations else None,
        }


# === Quick self-test when run directly ===
# Usage:  python -m src.tusimple_loader
#         python src/tusimple_loader.py
if __name__ == "__main__":
    dataset = TuSimpleDataset(
        annotation_file="data/tusimple/test_subset/test_label_subset.json",
        image_root="data/tusimple/test_subset",
    )

    print("=" * 60)
    print("TuSimple Dataset Inspection")
    print("=" * 60)

    stats = dataset.stats()
    for key, value in stats.items():
        print(f"  {key:25s}: {value}")
    print()

    # Show a sample
    sample = dataset[0]
    print(f"Sample annotation [0]:")
    print(f"  raw_file:       {sample.raw_file}")
    print(f"  image_path:     {sample.image_path}")
    print(f"  image exists:   {sample.image_path.exists()}")
    print(f"  num lanes:      {sample.num_lanes}")
    print(f"  visible lanes:  {sample.num_visible_lanes}")
    print(f"  h_samples[:5]:  {sample.h_samples[:5]}")
    print()

    # Show first few points of each lane
    for i in range(sample.num_lanes):
        points = sample.get_lane_points(i)
        if points:
            print(f"  Lane {i}: {len(points)} visible points, "
                  f"first 3 = {points[:3]}")
        else:
            print(f"  Lane {i}: not visible")
