"""Temporal smoothing for lane detection across video frames.

Provides EMASmoother — wraps the per-frame classical pipeline with an
Exponential Moving Average (EMA) over the polynomial coefficients.

Algorithm per frame:
    1. Run classical pipeline on the new frame.
    2. If the lane is detected, EMA-blend its coefficients with the previous
       smoothed coefficients:
          smoothed = alpha * current + (1 - alpha) * smoothed_prev
    3. If the lane is NOT detected, fall back entirely to the previous
       smoothed coefficients (i.e., "carry forward").
    4. After `max_consecutive_misses` consecutive failures, give up and report
       no detection (the smoother is no longer trustworthy on stale info).

This is the core proposed contribution of the project. It runs at the same
speed as the per-frame pipeline (smoothing adds ~one vector add per frame)
but is significantly more robust on video sequences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pipeline import ClassicalPipeline, LaneDetectionResult


# Tuned defaults
DEFAULT_ALPHA = 0.25                   # EMA learning rate
DEFAULT_MAX_CONSECUTIVE_MISSES = 5     # frames a smoothed estimate may carry forward


@dataclass
class SmoothedDetectionResult:
    """Output of the smoother for one frame.

    Attributes:
        raw_result:        The unsmoothed per-frame result.
        smoothed_left:     EMA-smoothed left lane coefficients (or None).
        smoothed_right:    EMA-smoothed right lane coefficients (or None).
        used_fallback_left:  True if this frame's left lane came from carry-forward.
        used_fallback_right: True if this frame's right lane came from carry-forward.
    """
    raw_result: LaneDetectionResult
    smoothed_left: np.ndarray | None
    smoothed_right: np.ndarray | None
    used_fallback_left: bool
    used_fallback_right: bool


class EMASmoother:
    """Stateful temporal smoother for lane detection.

    Wraps a ClassicalPipeline and maintains an exponential moving average
    over the polynomial coefficients of the left and right lanes.

    Usage:
        smoother = EMASmoother(alpha=0.25)
        for frame in video:
            result = smoother.process(frame)
            # result.smoothed_left, result.smoothed_right are the polished outputs
    """

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        max_consecutive_misses: int = DEFAULT_MAX_CONSECUTIVE_MISSES,
        pipeline: ClassicalPipeline | None = None,
    ):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self.max_consecutive_misses = max_consecutive_misses
        self.pipeline = pipeline if pipeline is not None else ClassicalPipeline()
        self.reset()

    def reset(self) -> None:
        """Clear all temporal state. Call between unrelated video sequences."""
        self._smoothed_left: np.ndarray | None = None
        self._smoothed_right: np.ndarray | None = None
        self._misses_left = 0
        self._misses_right = 0

    def process(self, image_rgb: np.ndarray) -> SmoothedDetectionResult:
        """Process one video frame and return the smoothed result."""
        raw = self.pipeline.detect(image_rgb)

        smoothed_left, used_fb_left = self._update(
            current=raw.left_coeffs,
            detected=raw.left_detected,
            side="left",
        )
        smoothed_right, used_fb_right = self._update(
            current=raw.right_coeffs,
            detected=raw.right_detected,
            side="right",
        )

        return SmoothedDetectionResult(
            raw_result=raw,
            smoothed_left=smoothed_left,
            smoothed_right=smoothed_right,
            used_fallback_left=used_fb_left,
            used_fallback_right=used_fb_right,
        )

    def _update(
        self,
        current: np.ndarray | None,
        detected: bool,
        side: str,
    ) -> tuple[np.ndarray | None, bool]:
        """Apply EMA update for one lane (left or right).

        Returns:
            (smoothed_coeffs, used_fallback)
        """
        if side == "left":
            smoothed_prev = self._smoothed_left
            misses_attr = "_misses_left"
        else:
            smoothed_prev = self._smoothed_right
            misses_attr = "_misses_right"

        # Case A: current detection is good
        if detected and current is not None:
            if smoothed_prev is None:
                # First good detection — initialize
                new_smoothed = current.copy()
            else:
                new_smoothed = self.alpha * current + (1 - self.alpha) * smoothed_prev
            setattr(self, misses_attr, 0)
            self._set_smoothed(side, new_smoothed)
            return new_smoothed, False

        # Case B: current detection failed — try to carry forward
        misses = getattr(self, misses_attr) + 1
        setattr(self, misses_attr, misses)

        if smoothed_prev is not None and misses <= self.max_consecutive_misses:
            # Carry forward the previous smoothed estimate
            return smoothed_prev, True

        # Case C: too many misses or no history — give up
        self._set_smoothed(side, None)
        return None, True

    def _set_smoothed(self, side: str, value: np.ndarray | None) -> None:
        if side == "left":
            self._smoothed_left = value
        else:
            self._smoothed_right = value


if __name__ == "__main__":
    # Self-test: simulate a fake sequence of detections
    print("Testing EMASmoother on a synthetic frame sequence...")
    
    smoother = EMASmoother(alpha=0.25, max_consecutive_misses=3)
    
    # Mock a small sequence: 3 good frames, 2 bad frames, 1 good
    fake_results = [
        # (left_coeffs, right_coeffs, left_det, right_det)
        (np.array([0.001, -0.2, 100.0]), np.array([0.001, 0.1, 600.0]), True, True),
        (np.array([0.001, -0.2, 102.0]), np.array([0.001, 0.1, 598.0]), True, True),
        (np.array([0.001, -0.2, 104.0]), np.array([0.001, 0.1, 596.0]), True, True),
        (None, None, False, False),  # detection lost
        (None, None, False, False),  # still lost
        (np.array([0.001, -0.2, 110.0]), np.array([0.001, 0.1, 590.0]), True, True),
    ]
    
    # Use the smoother's internal _update directly to avoid running pipeline
    for i, (lc, rc, ld, rd) in enumerate(fake_results):
        sl, fb_l = smoother._update(lc, ld, "left")
        sr, fb_r = smoother._update(rc, rd, "right")
        print(f"Frame {i}: detected=({ld},{rd})  "
              f"smoothed_left[c]={None if sl is None else sl[2]:.2f}  "
              f"fallback=({fb_l},{fb_r})")
    
    print()
    print("Module temporal.py loaded successfully.")