"""Pre-compute predictions for all 3 methods on all clips (memory-safe version).

Processes one clip at a time, writing intermediate results to disk so we
don't hold all 400 frames in RAM at once.

Output:
  results/predictions_cache.pkl  — predictions only (small, ~few MB)
  Frames are NOT cached; video renderer re-reads from disk per clip.
"""

from __future__ import annotations
from benchmark import UNet, unet_predict_polynomials
from perspective import PerspectiveWarper
from temporal import EMASmoother
from pipeline import ClassicalPipeline

import gc
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))


def load_clip_frames(clip_dir: Path) -> list[np.ndarray]:
    """Load all 20 JPGs of one clip into RAM as RGB ndarrays."""
    paths = sorted(clip_dir.glob("*.jpg"), key=lambda p: int(p.stem))
    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return frames


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_root = Path("data/tusimple/test_subset/clips/0530")
    clip_dirs = sorted([d for d in data_root.iterdir()
                        if d.is_dir() and len(list(d.glob("*.jpg"))) == 20])
    print(f"Found {len(clip_dirs)} clips with all 20 frames\n")

    # === Load models / pipelines once ===
    print("Initializing pipelines...")
    classical = ClassicalPipeline()
    warper = PerspectiveWarper()

    ckpt_path = Path("checkpoints/unet_best.pth")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    unet = UNet().to(device)
    unet.load_state_dict(checkpoint["model_state_dict"])
    unet.eval()
    print(f"  U-Net loaded (epoch {checkpoint['epoch']}, "
          f"val loss {checkpoint['val_loss']:.4f})")

    # === Per-clip processing ===
    # We collect predictions only (small) — NOT the raw frames.
    # The video renderer will re-read frames from disk per clip.
    all_predictions = []
    clip_metadata = []
    clip_paths = []   # store the directory so the video renderer can re-load

    overall_start = time.time()

    for clip_idx, clip_dir in enumerate(clip_dirs):
        clip_start = time.time()
        print(f"\nClip {clip_idx + 1}/{len(clip_dirs)} ({clip_dir.name})")

        # Load this clip's frames into RAM
        frames = load_clip_frames(clip_dir)

        # Fresh smoother per clip (different scenes, no temporal continuity)
        smoother = EMASmoother(alpha=0.25, max_consecutive_misses=5,
                               pipeline=classical)

        clip_preds = {
            "classical": [],
            "ema": [],
            "unet": [],
            "ema_fallback": [],
        }

        for frame_idx, frame in enumerate(frames):
            # Classical (independent per frame)
            raw_result = classical.detect(frame)
            clip_preds["classical"].append(
                (raw_result.left_coeffs, raw_result.right_coeffs)
            )

            # Classical + EMA (stateful within this clip)
            smoothed = smoother.process(frame)
            clip_preds["ema"].append(
                (smoothed.smoothed_left, smoothed.smoothed_right)
            )
            clip_preds["ema_fallback"].append(
                (smoothed.used_fallback_left, smoothed.used_fallback_right)
            )

            # U-Net
            t0 = time.perf_counter()
            un_left, un_right = unet_predict_polynomials(
                unet, frame, warper, device
            )
            unet_time = time.perf_counter() - t0
            clip_preds["unet"].append((un_left, un_right))

            if frame_idx % 5 == 0:
                print(f"  Frame {frame_idx + 1:2d}/{len(frames)} "
                      f"(U-Net: {unet_time:.2f}s)")

        all_predictions.append(clip_preds)
        clip_metadata.append({
            "clip_id": clip_dir.name,
            "n_frames": len(frames),
        })
        clip_paths.append(str(clip_dir))

        # === CRITICAL: free this clip's frames BEFORE loading next clip ===
        del frames
        del smoother
        gc.collect()

        clip_time = time.time() - clip_start
        print(f"  Clip done in {clip_time:.1f}s")

        # === Save incrementally — if anything dies, we keep partial progress ===
        cache = {
            "clip_paths": clip_paths,
            "clip_metadata": clip_metadata,
            "all_predictions": all_predictions,
            "warper_dst_size": warper.dst_size,
        }
        out_path = Path("results/predictions_cache.pkl")
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(cache, f)

    total_time = time.time() - overall_start
    print(f"\n{'=' * 60}")
    print(f"Done. {len(all_predictions)} clips processed in "
          f"{total_time / 60:.1f} min")
    print(
        f"Cache: {Path('results/predictions_cache.pkl').stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
