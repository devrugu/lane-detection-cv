"""Thresholding pipeline — extract lane pixels from a bird's-eye view image.

Combines two complementary techniques:
- Sobel x-gradient (with Gaussian pre-blur) for vertical edge detection
- HLS S+L color thresholding for yellow and white lane lines

The final output is a binary mask (uint8, 0 or 255) the same size as the input.
A region-of-interest mask removes warping artifacts at the BEV bottom edge.

Tuning was done in notebooks/03_thresholding.ipynb on the TuSimple test_set.
Different lighting (night driving, rain, different road materials) may need
re-tuning; this is a known limitation of classical thresholding.
"""

from __future__ import annotations

import cv2
import numpy as np


# === Tuned defaults for TuSimple test_set ===
DEFAULT_SOBEL_KSIZE = 5
DEFAULT_SOBEL_THRESH = (25, 200)
DEFAULT_BLUR_SIZE = 3

DEFAULT_S_THRESH = (130, 255)   # high saturation → yellow lanes
DEFAULT_L_THRESH = (220, 255)   # high lightness  → white lanes

DEFAULT_BOTTOM_MARGIN = 0.05    # discard bottom 5% (warping artifacts)


def sobel_x_threshold(
    image_rgb: np.ndarray,
    ksize: int = DEFAULT_SOBEL_KSIZE,
    thresh: tuple[int, int] = DEFAULT_SOBEL_THRESH,
    blur_size: int = DEFAULT_BLUR_SIZE,
) -> np.ndarray:
    """Binary mask of vertical edges via Sobel x-gradient."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    if blur_size > 0:
        gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    abs_s = np.abs(sobel_x)
    scaled = np.uint8(255 * abs_s / max(np.max(abs_s), 1))
    mask = np.zeros_like(scaled)
    mask[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 255
    return mask


def hls_color_threshold(
    image_rgb: np.ndarray,
    s_thresh: tuple[int, int] = DEFAULT_S_THRESH,
    l_thresh: tuple[int, int] = DEFAULT_L_THRESH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HLS-based binary mask. Returns (combined, s_mask, l_mask)."""
    hls = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HLS)
    l_ch = hls[:, :, 1]
    s_ch = hls[:, :, 2]
    s_mask = np.zeros_like(s_ch)
    s_mask[(s_ch >= s_thresh[0]) & (s_ch <= s_thresh[1])] = 255
    l_mask = np.zeros_like(l_ch)
    l_mask[(l_ch >= l_thresh[0]) & (l_ch <= l_thresh[1])] = 255
    combined = np.zeros_like(s_ch)
    combined[(s_mask == 255) | (l_mask == 255)] = 255
    return combined, s_mask, l_mask


def make_roi_mask(
    shape: tuple[int, int],
    top_margin: float = 0.0,
    bottom_margin: float = DEFAULT_BOTTOM_MARGIN,
) -> np.ndarray:
    """Region-of-interest mask. Masks out unreliable parts of the BEV."""
    H, W = shape
    roi = np.ones(shape, dtype=np.uint8) * 255
    top_cut = int(H * top_margin)
    bot_cut = int(H * bottom_margin)
    if top_cut > 0:
        roi[:top_cut, :] = 0
    if bot_cut > 0:
        roi[H - bot_cut:, :] = 0
    return roi


def threshold_pipeline(image_rgb: np.ndarray) -> np.ndarray:
    """Full thresholding pipeline. Returns binary mask of suspected lane pixels."""
    sobel = sobel_x_threshold(image_rgb)
    color, _, _ = hls_color_threshold(image_rgb)
    combined = np.zeros_like(sobel)
    combined[(sobel == 255) | (color == 255)] = 255
    roi = make_roi_mask(combined.shape)
    combined[roi == 0] = 0
    return combined


if __name__ == "__main__":
    # Quick self-test on a fake image
    test = np.zeros((720, 720, 3), dtype=np.uint8)
    test[300:400, 100:110, :] = 255  # white vertical bar (fake lane)
    mask = threshold_pipeline(test)
    print(f"Self-test: input shape {test.shape}, mask shape {mask.shape}")
    print(f"Lane pixels found: {np.sum(mask == 255)}")
    print(f"Module thresholding.py loaded successfully.")