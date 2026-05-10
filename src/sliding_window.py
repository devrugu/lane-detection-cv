"""Sliding window lane pixel grouping.

Takes a binary lane mask (from thresholding) and partitions the white pixels into
left-lane and right-lane collections, by walking small windows up the image
starting from histogram peaks at the bottom.

A `base_min_count` threshold rejects lanes whose histogram peak is too small
to be a real lane (e.g. noise at the edge of the BEV). When a lane is rejected,
the corresponding `*_detected` field is False and no pixels are returned.
"""

from __future__ import annotations

import numpy as np


# === Tuned defaults for TuSimple BEV at 720x720 ===
DEFAULT_N_WINDOWS = 9
DEFAULT_MARGIN = 100         # half-width of each window
DEFAULT_MIN_PIXELS = 50      # min pixels in a window to recenter the next one
DEFAULT_BASE_MIN_COUNT = 3000  # min histogram peak height to call a lane "detected"


def sliding_window_search(
    mask: np.ndarray,
    n_windows: int = DEFAULT_N_WINDOWS,
    margin: int = DEFAULT_MARGIN,
    min_pixels: int = DEFAULT_MIN_PIXELS,
    base_min_count: int = DEFAULT_BASE_MIN_COUNT,
) -> dict:
    """Run sliding window search on a binary lane mask.

    Args:
        mask:           binary uint8 mask, lane pixels = 255
        n_windows:      number of vertical windows
        margin:         half-width of each window in pixels
        min_pixels:     min pixels in a window to recenter the next one
        base_min_count: min histogram peak height to declare a lane present

    Returns dict with keys:
        left_x_idx, left_y_idx:   per-pixel indices for the left lane
        right_x_idx, right_y_idx: per-pixel indices for the right lane
        windows_left, windows_right: window rectangles (x_low, y_low, x_high, y_high)
        left_base, right_base:    starting x positions (None if not detected)
        left_detected, right_detected: booleans
        left_peak_value, right_peak_value: histogram peak heights (for diagnostics)
    """
    H, W = mask.shape
    window_height = H // n_windows

    nonzero = mask.nonzero()
    nonzero_y = nonzero[0]
    nonzero_x = nonzero[1]

    # Histogram of bottom half
    bottom_half = mask[H // 2:, :]
    histogram = np.sum(bottom_half, axis=0)
    midpoint = W // 2

    left_peak_value = int(histogram[:midpoint].max())
    right_peak_value = int(histogram[midpoint:].max())

    left_detected = left_peak_value >= base_min_count
    right_detected = right_peak_value >= base_min_count

    left_base = int(np.argmax(histogram[:midpoint])) if left_detected else None
    right_base = int(np.argmax(histogram[midpoint:]) + midpoint) if right_detected else None

    left_x_current = left_base
    right_x_current = right_base

    left_lane_inds: list[np.ndarray] = []
    right_lane_inds: list[np.ndarray] = []

    windows_left: list[tuple[int, int, int, int]] = []
    windows_right: list[tuple[int, int, int, int]] = []

    for window_idx in range(n_windows):
        win_y_low = H - (window_idx + 1) * window_height
        win_y_high = H - window_idx * window_height

        if left_detected:
            xl_low = left_x_current - margin
            xl_high = left_x_current + margin
            windows_left.append((xl_low, win_y_low, xl_high, win_y_high))
            good_left = (
                (nonzero_y >= win_y_low) & (nonzero_y < win_y_high)
                & (nonzero_x >= xl_low) & (nonzero_x < xl_high)
            ).nonzero()[0]
            left_lane_inds.append(good_left)
            if len(good_left) > min_pixels:
                left_x_current = int(np.mean(nonzero_x[good_left]))

        if right_detected:
            xr_low = right_x_current - margin
            xr_high = right_x_current + margin
            windows_right.append((xr_low, win_y_low, xr_high, win_y_high))
            good_right = (
                (nonzero_y >= win_y_low) & (nonzero_y < win_y_high)
                & (nonzero_x >= xr_low) & (nonzero_x < xr_high)
            ).nonzero()[0]
            right_lane_inds.append(good_right)
            if len(good_right) > min_pixels:
                right_x_current = int(np.mean(nonzero_x[good_right]))

    left_concat = np.concatenate(left_lane_inds) if left_lane_inds else np.array([], dtype=int)
    right_concat = np.concatenate(right_lane_inds) if right_lane_inds else np.array([], dtype=int)

    return {
        "left_x_idx": nonzero_x[left_concat],
        "left_y_idx": nonzero_y[left_concat],
        "right_x_idx": nonzero_x[right_concat],
        "right_y_idx": nonzero_y[right_concat],
        "windows_left": windows_left,
        "windows_right": windows_right,
        "left_base": left_base,
        "right_base": right_base,
        "left_detected": left_detected,
        "right_detected": right_detected,
        "left_peak_value": left_peak_value,
        "right_peak_value": right_peak_value,
    }


if __name__ == "__main__":
    # Self-test on a synthetic mask with one fake "lane"
    test = np.zeros((720, 720), dtype=np.uint8)
    test[200:700, 100:115] = 255   # fake left lane
    test[200:700, 600:615] = 255   # fake right lane
    res = sliding_window_search(test)
    print(f"Self-test on synthetic mask:")
    print(f"  Left detected: {res['left_detected']} (peak={res['left_peak_value']})")
    print(f"  Right detected: {res['right_detected']} (peak={res['right_peak_value']})")
    print(f"  Left pixels: {len(res['left_x_idx'])}")
    print(f"  Right pixels: {len(res['right_x_idx'])}")
    print(f"  Module sliding_window.py loaded successfully.")