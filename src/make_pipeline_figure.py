"""Build a 6-panel figure showing each stage of the classical CV pipeline.

Output: results/pipeline_stages.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from tusimple_loader import TuSimpleDataset
from perspective import PerspectiveWarper
from thresholding import threshold_pipeline
from sliding_window import sliding_window_search
from polyfit import fit_lane_polynomial, evaluate_polynomial
from pipeline import ClassicalPipeline


SAMPLE_IDX = 0


def draw_window(img, win, color, thickness=2):
    """Draw a single sliding window rectangle. Handles dict and tuple formats."""
    if isinstance(win, dict):
        # Try various common key names
        x_low  = win.get("x_low",  win.get("win_x_low",  win.get("x", None)))
        x_high = win.get("x_high", win.get("win_x_high", None))
        y_low  = win.get("y_low",  win.get("win_y_low",  win.get("y", None)))
        y_high = win.get("y_high", win.get("win_y_high", None))
        # Width/height alternative
        if x_high is None and "w" in win:
            x_high = x_low + win["w"]
        if y_high is None and "h" in win:
            y_high = y_low + win["h"]
    elif isinstance(win, (tuple, list)) and len(win) >= 4:
        # Assume (x_low, y_low, x_high, y_high) — adjust if your code uses (x,y,w,h)
        x_low, y_low, x_high, y_high = win[0], win[1], win[2], win[3]
    else:
        print(f"  WARN: unknown window format: {type(win)}: {win}")
        return

    if None in (x_low, y_low, x_high, y_high):
        return
    cv2.rectangle(
        img,
        (int(x_low), int(y_low)),
        (int(x_high), int(y_high)),
        color, thickness
    )


def main() -> None:
    project_root = Path(__file__).parent.parent

    dataset = TuSimpleDataset(
        annotation_file=project_root / "data/tusimple/test_subset/test_label_subset.json",
        image_root=project_root / "data/tusimple/test_subset",
    )
    available = dataset.existing_only()
    if not available:
        print("ERROR: no test frames found.")
        return

    sample = available[SAMPLE_IDX]
    print(f"Using sample: {sample.raw_file}")

    img_bgr = cv2.imread(str(sample.image_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    warper = PerspectiveWarper()
    pipeline = ClassicalPipeline(warper=warper)

    # Stage 2: BEV warp
    bev = warper.warp_image(img_rgb)

    # Stage 3: binary threshold (camera view) → warp to BEV
    binary_mask_camera = threshold_pipeline(img_rgb)
    binary_mask_bev = warper.warp_image(np.dstack([binary_mask_camera] * 3))[:, :, 0]
    binary_mask_bev = (binary_mask_bev > 127).astype(np.uint8) * 255

    # Stage 4: sliding window
    sw = sliding_window_search(binary_mask_bev)
    sw_vis = np.dstack([binary_mask_bev] * 3)
    if sw["left_detected"]:
        for x, y in zip(sw["left_x_idx"], sw["left_y_idx"]):
            cv2.circle(sw_vis, (int(x), int(y)), 1, (255, 0, 0), -1)
    if sw["right_detected"]:
        for x, y in zip(sw["right_x_idx"], sw["right_y_idx"]):
            cv2.circle(sw_vis, (int(x), int(y)), 1, (0, 100, 255), -1)
    # Draw window rectangles (green)
    for win in sw.get("windows_left", []):
        draw_window(sw_vis, win, (0, 255, 0))
    for win in sw.get("windows_right", []):
        draw_window(sw_vis, win, (0, 255, 0))

    # Stage 5: polynomial fit visualized in BEV
    left_coeffs = fit_lane_polynomial(sw["left_x_idx"], sw["left_y_idx"]) \
        if sw["left_detected"] else None
    right_coeffs = fit_lane_polynomial(sw["right_x_idx"], sw["right_y_idx"]) \
        if sw["right_detected"] else None

    poly_vis = np.dstack([binary_mask_bev] * 3)
    H_bev = binary_mask_bev.shape[0]
    y_eval = np.arange(0, H_bev)
    if left_coeffs is not None:
        x_eval = evaluate_polynomial(left_coeffs, y_eval)
        pts = np.array(list(zip(x_eval.astype(int), y_eval.astype(int))), dtype=np.int32)
        cv2.polylines(poly_vis, [pts], False, (255, 50, 50), thickness=4)
    if right_coeffs is not None:
        x_eval = evaluate_polynomial(right_coeffs, y_eval)
        pts = np.array(list(zip(x_eval.astype(int), y_eval.astype(int))), dtype=np.int32)
        cv2.polylines(poly_vis, [pts], False, (50, 100, 255), thickness=4)

    # Stage 6: final overlay
    result = pipeline.detect(img_rgb)
    final_overlay = pipeline.draw_overlay(img_rgb, result)

    # === Build figure ===
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    panels = [
        (img_rgb, "1. Input frame", "RGB camera image (1280 × 720)"),
        (bev, "2. BEV warp", "Perspective transform to bird's-eye view (720 × 720)"),
        (binary_mask_bev, "3. Binary threshold (in BEV)",
         "Sobel x-gradient ∪ HLS color → binary lane mask"),
        (sw_vis, "4. Sliding window",
         "Histogram peak → 9 windows track lane pixels (red = left, blue = right)"),
        (poly_vis, "5. Polynomial fit",
         "Least-squares 2nd-degree polynomial x = ay² + by + c"),
        (final_overlay, "6. Final overlay",
         "Curves unwarped back to camera view + drivable polygon"),
    ]

    for ax, (img, title, subtitle) in zip(axes.flat, panels):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight="bold", loc="left")
        ax.text(0.0, -0.05, subtitle, transform=ax.transAxes,
                fontsize=10, color="#666666", style="italic", va="top")
        ax.axis("off")

    plt.suptitle("Classical CV Pipeline — Stage-by-Stage Output",
                  fontsize=17, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = project_root / "results/pipeline_stages.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    out_pdf = project_root / "results/pipeline_stages.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_pdf}")


if __name__ == "__main__":
    main()