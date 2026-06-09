"""Processing core for the demo app.

Thin wrapper around the project's classical pipeline (src/) so the GUI never
imports OpenCV-heavy logic directly. Handles the one real-world gotcha: the
perspective transform is calibrated for a 1280x720 TuSimple camera, so every
input is fitted to that working size before detection and the overlay is
scaled back to the original resolution for display/export.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np

# --- Make the project's src/ importable whether running from source or a
#     PyInstaller bundle. In a frozen app, src/ is unpacked next to the exe
#     in sys._MEIPASS. From source we walk up to the repo root. ---


def _add_src_to_path() -> None:
    if getattr(sys, "frozen", False):
        base = sys._MEIPASS  # type: ignore[attr-defined]
        candidates = [os.path.join(base, "src"), base]
    else:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [os.path.join(os.path.dirname(here), "src")]
    for c in candidates:
        if c not in sys.path:
            sys.path.insert(0, c)


_add_src_to_path()

from pipeline import ClassicalPipeline, LaneDetectionResult  # noqa: E402
from temporal import EMASmoother  # noqa: E402

# The calibration these constants were tuned for.
WORK_W = 1280
WORK_H = 720


@dataclass
class FrameOutput:
    """Everything the GUI needs for one processed frame."""
    overlay_rgb: np.ndarray            # overlay at the ORIGINAL input size
    left_coeffs: np.ndarray | None     # smoothed (video) or raw (image)
    right_coeffs: np.ndarray | None
    left_detected: bool
    right_detected: bool
    used_fallback_left: bool
    used_fallback_right: bool


def _fit_to_work_size(image_rgb: np.ndarray) -> np.ndarray:
    """Resize an arbitrary input to the calibrated working resolution."""
    if image_rgb.shape[1] == WORK_W and image_rgb.shape[0] == WORK_H:
        return image_rgb
    return cv2.resize(image_rgb, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)


class ImageProcessor:
    """Single-frame detector for Image mode (no temporal smoothing)."""

    def __init__(self) -> None:
        self.pipeline = ClassicalPipeline()

    def process(self, image_rgb: np.ndarray) -> FrameOutput:
        orig_h, orig_w = image_rgb.shape[:2]
        work = _fit_to_work_size(image_rgb)
        result: LaneDetectionResult = self.pipeline.detect(work)
        overlay_work = self.pipeline.draw_overlay(work, result)
        overlay = cv2.resize(
            overlay_work, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
        )
        return FrameOutput(
            overlay_rgb=overlay,
            left_coeffs=result.left_coeffs,
            right_coeffs=result.right_coeffs,
            left_detected=result.left_detected,
            right_detected=result.right_detected,
            used_fallback_left=False,
            used_fallback_right=False,
        )


class VideoProcessor:
    """Stateful per-frame detector for Video mode (classical CV + EMA)."""

    def __init__(self, alpha: float = 0.25) -> None:
        self.smoother = EMASmoother(alpha=alpha)

    def reset(self) -> None:
        self.smoother.reset()

    def process(self, frame_rgb: np.ndarray) -> FrameOutput:
        orig_h, orig_w = frame_rgb.shape[:2]
        work = _fit_to_work_size(frame_rgb)
        sm = self.smoother.process(work)

        # Draw the SMOOTHED lanes (the project's proposed contribution) using
        # the pipeline's own overlay routine, by swapping the raw result's
        # coefficients for the smoothed ones.
        raw = sm.raw_result
        raw.left_coeffs = sm.smoothed_left
        raw.right_coeffs = sm.smoothed_right
        raw.left_detected = sm.smoothed_left is not None
        raw.right_detected = sm.smoothed_right is not None
        overlay_work = self.smoother.pipeline.draw_overlay(work, raw)
        overlay = cv2.resize(
            overlay_work, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
        )
        return FrameOutput(
            overlay_rgb=overlay,
            left_coeffs=sm.smoothed_left,
            right_coeffs=sm.smoothed_right,
            left_detected=sm.smoothed_left is not None,
            right_detected=sm.smoothed_right is not None,
            used_fallback_left=sm.used_fallback_left,
            used_fallback_right=sm.used_fallback_right,
        )
