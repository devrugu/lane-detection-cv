"""Render 3-column comparison video using cached predictions.

Output: results/comparison_video.mp4
"""

from __future__ import annotations
from functools import cache
from polyfit import evaluate_polynomial
from perspective import PerspectiveWarper

import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


# Video parameters
COL_W = 640                # width of each column in the output video
COL_H = 360                # height of each column
HEADER_H = 80              # height of header strip on top
TRANSITION_FRAMES = 12     # frames showing "Clip X/N" between clips
PLAYBACK_FPS = 4           # output video FPS (slow = easier to see effects)


def draw_overlay_on_frame(image_rgb, left_coeffs, right_coeffs,
                          warper, draw_polygon=True):
    """Draw lane curves + drivable polygon on a camera-view image.

    Identical look across all three methods so visual differences are real
    detection differences, not styling.
    """
    H_bev = warper.dst_size
    y_eval = np.linspace(0, H_bev - 1, H_bev)
    out_size = (image_rgb.shape[1], image_rgb.shape[0])
    result = image_rgb.copy()

    # Drivable area polygon (only when both lanes present)
    if (draw_polygon
            and left_coeffs is not None and right_coeffs is not None):
        left_x = evaluate_polynomial(left_coeffs, y_eval)
        right_x = evaluate_polynomial(right_coeffs, y_eval)
        left_pts = np.array([np.transpose(np.vstack([left_x, y_eval]))])
        right_pts = np.array(
            [np.flipud(np.transpose(np.vstack([right_x, y_eval])))]
        )
        polygon_pts = np.hstack((left_pts, right_pts)).astype(np.int32)
        bev_polygon = np.zeros((H_bev, H_bev, 3), dtype=np.uint8)
        cv2.fillPoly(bev_polygon, polygon_pts, (0, 200, 0))
        polygon_camera = warper.unwarp_image(bev_polygon, out_size)
        result = cv2.addWeighted(result, 1.0, polygon_camera, 0.3, 0)

    # Lane curves
    bev_lines = np.zeros((H_bev, H_bev, 3), dtype=np.uint8)
    if left_coeffs is not None:
        left_x = evaluate_polynomial(left_coeffs, y_eval)
        pts = np.array(
            list(zip(left_x.astype(int), y_eval.astype(int))), dtype=np.int32
        )
        cv2.polylines(bev_lines, [pts], False, (0, 255, 255), thickness=12)
    if right_coeffs is not None:
        right_x = evaluate_polynomial(right_coeffs, y_eval)
        pts = np.array(
            list(zip(right_x.astype(int), y_eval.astype(int))), dtype=np.int32
        )
        cv2.polylines(bev_lines, [pts], False, (255, 200, 0), thickness=12)
    lines_camera = warper.unwarp_image(bev_lines, out_size)
    line_mask = lines_camera.sum(axis=2) > 0
    result[line_mask] = lines_camera[line_mask]

    return result


def make_column_panel(image_rgb_overlay, col_w, col_h):
    """Resize image to column dimensions while preserving aspect ratio."""
    return cv2.resize(image_rgb_overlay, (col_w, col_h),
                      interpolation=cv2.INTER_AREA)


def draw_header(canvas, fps_dict, font=cv2.FONT_HERSHEY_SIMPLEX):
    """Draw the column headers at the top of the canvas."""
    headers = [
        ("Classical CV", f"{fps_dict['classical']:.1f} fps",
         "single-frame baseline"),
        ("Classical + EMA", f"{fps_dict['ema']:.1f} fps",
         "OUR contribution"),
        ("U-Net", f"{fps_dict['unet']:.1f} fps",
         "deep learning"),
    ]
    canvas[:HEADER_H, :] = (30, 30, 30)  # dark grey strip

    for i, (title, fps_text, subtitle) in enumerate(headers):
        x_center = i * COL_W + COL_W // 2
        cv2.putText(canvas, title, (x_center - 100, 28),
                    font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, fps_text, (x_center - 60, 52),
                    font, 0.6, (0, 255, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, subtitle, (x_center - 100, 72),
                    font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)


def draw_clip_marker(canvas, clip_idx, n_clips, frame_idx, n_frames):
    """Draw a "Clip X/N — Frame Y/Z" label at the bottom."""
    H, W = canvas.shape[:2]
    text = f"Clip {clip_idx + 1}/{n_clips}   Frame {frame_idx + 1}/{n_frames}"
    cv2.putText(canvas, text, (20, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def make_transition_card(clip_idx, n_clips, total_w, total_h):
    """Build a 'Clip X/N' card to show between clips."""
    card = np.zeros((total_h, total_w, 3), dtype=np.uint8)
    text = f"Clip {clip_idx + 1} / {n_clips}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, 2.0, 4)
    cv2.putText(card, text, ((total_w - tw) // 2, (total_h + th) // 2),
                font, 2.0, (255, 255, 255), 4, cv2.LINE_AA)
    sub = "next clip..."
    (sw, sh), _ = cv2.getTextSize(sub, font, 0.8, 2)
    cv2.putText(card, sub, ((total_w - sw) // 2, (total_h + th) // 2 + 50),
                font, 0.8, (180, 180, 180), 2, cv2.LINE_AA)
    return card


def main() -> None:
    # === Load cache ===
    cache_path = Path("results/predictions_cache.pkl")
    if not cache_path.exists():
        print(
            f"ERROR: {cache_path} not found. Run precompute_predictions.py first.")
        return

    print(f"Loading prediction cache from {cache_path}...")
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)


        clip_paths = cache["clip_paths"]
        clip_metadata = cache["clip_metadata"]
        all_predictions = cache["all_predictions"]
        n_clips = len(clip_paths)
        n_frames_total = sum(m["n_frames"] for m in clip_metadata)
        print(f"  {n_clips} clips, {n_frames_total} frames total")

    # === Load FPS numbers ===
    fps_path = Path("results/fps_benchmark.json")
    if not fps_path.exists():
        print(f"ERROR: {fps_path} not found. Run benchmark.py first.")
        return

    with open(fps_path) as f:
        bench = json.load(f)

    fps_dict = {
        "classical": bench["results"][0]["avg_fps"],
        "ema":       bench["results"][1]["avg_fps"],
        "unet":      bench["results"][2]["avg_fps"],
    }
    print(f"  FPS: classical={fps_dict['classical']:.1f}, "
          f"ema={fps_dict['ema']:.1f}, unet={fps_dict['unet']:.1f}")

    # === Set up warper for drawing ===
    warper = PerspectiveWarper(dst_size=cache["warper_dst_size"])

    # === Video writer ===
    total_w = COL_W * 3
    total_h = HEADER_H + COL_H
    out_path = Path("results/comparison_video_noclip.mp4")

    # mp4v codec works in OpenCV without extra dependencies
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, PLAYBACK_FPS,
                             (total_w, total_h))
    if not writer.isOpened():
        print(f"ERROR: Failed to open video writer for {out_path}")
        return

    print(f"\nRendering video: {out_path}")
    print(f"  Resolution: {total_w}x{total_h}, FPS: {PLAYBACK_FPS}")

    n_total_frames = 0

    # === Per-clip rendering ===
    for clip_idx in range(n_clips):

        # Load this clip's frames from disk (saves RAM)
        clip_dir = Path(clip_paths[clip_idx])
        paths = sorted(clip_dir.glob("*.jpg"), key=lambda p: int(p.stem))
        frames = []
        for p in paths:
            img = cv2.imread(str(p))
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        preds = all_predictions[clip_idx]

        for frame_idx, frame in enumerate(frames):
            # Build the 3 overlays
            cl_left, cl_right = preds["classical"][frame_idx]
            ema_left, ema_right = preds["ema"][frame_idx]
            un_left, un_right = preds["unet"][frame_idx]

            classical_overlay = draw_overlay_on_frame(
                frame, cl_left, cl_right, warper
            )
            ema_overlay = draw_overlay_on_frame(
                frame, ema_left, ema_right, warper
            )
            unet_overlay = draw_overlay_on_frame(
                frame, un_left, un_right, warper
            )

            # Build canvas
            canvas = np.zeros((total_h, total_w, 3), dtype=np.uint8)

            # Headers
            draw_header(canvas, fps_dict)

            # Resize each overlay to a column
            cls_col = make_column_panel(classical_overlay, COL_W, COL_H)
            ema_col = make_column_panel(ema_overlay, COL_W, COL_H)
            unet_col = make_column_panel(unet_overlay, COL_W, COL_H)

            canvas[HEADER_H:HEADER_H + COL_H,           0:COL_W] = cls_col
            canvas[HEADER_H:HEADER_H + COL_H,       COL_W:2 * COL_W] = ema_col
            canvas[HEADER_H:HEADER_H + COL_H, 2 * COL_W:3 * COL_W] = unet_col

            # Vertical separators
            canvas[HEADER_H:, COL_W - 2:COL_W + 2] = (255, 255, 255)
            canvas[HEADER_H:, 2 * COL_W - 2:2 * COL_W + 2] = (255, 255, 255)

            # Bottom marker
            draw_clip_marker(canvas, clip_idx, n_clips,
                             frame_idx, len(frames))

            # Convert RGB -> BGR for OpenCV writer
            writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            n_total_frames += 1

        print(f"  Clip {clip_idx + 1}/{n_clips} rendered "
              f"({len(frames)} frames)")
        del frames  # free RAM before next clip
        import gc
        gc.collect()

    writer.release()
    print(f"\nDone! Wrote {n_total_frames} frames "
          f"({n_total_frames / PLAYBACK_FPS:.1f}s video) "
          f"to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
