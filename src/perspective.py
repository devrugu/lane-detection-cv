"""Perspective transform module — bird's-eye view conversion.

Provides a single class `PerspectiveWarper` that wraps the homography matrices
and gives clean methods for warping images, points, and the inverse direction.

Source points were tuned interactively in notebooks/02_perspective_transform.ipynb
on the TuSimple test_set. The TuSimple camera position is fixed across all clips
so a single calibration generalizes.
"""

from __future__ import annotations

import cv2
import numpy as np


# === Tuned constants for TuSimple test_set ===
# These were found by interactive tuning in notebook 02.
# The trapezoid covers the ego-lane region of the road.
DEFAULT_SRC_PTS: np.ndarray = np.float32([
    [470,  400],   # top-left
    [900,  400],   # top-right
    [1280, 700],   # bottom-right
    [110,  700],   # bottom-left
])

DEFAULT_DST_SIZE: int = 720


class PerspectiveWarper:
    """Wraps the perspective transform matrices and warping operations.

    Usage:
        warper = PerspectiveWarper()
        bev = warper.warp_image(camera_image)        # camera -> bird's-eye
        cam = warper.unwarp_image(bev)                # bird's-eye -> camera
        bev_points = warper.warp_points(camera_points)
        cam_points = warper.unwarp_points(bev_points)
    """

    def __init__(
        self,
        src_pts: np.ndarray | None = None,
        dst_size: int | None = None,
    ):
        self.src_pts = (
            src_pts.astype(np.float32) if src_pts is not None
            else DEFAULT_SRC_PTS.copy()
        )
        self.dst_size = dst_size if dst_size is not None else DEFAULT_DST_SIZE
        self.dst_pts = np.float32([
            [0,             0],
            [self.dst_size, 0],
            [self.dst_size, self.dst_size],
            [0,             self.dst_size],
        ])

        self.M = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        self.M_inv = cv2.getPerspectiveTransform(self.dst_pts, self.src_pts)

    # --- Image warping ---

    def warp_image(self, image: np.ndarray) -> np.ndarray:
        """Camera view -> bird's-eye view."""
        return cv2.warpPerspective(
            image, self.M, (self.dst_size, self.dst_size),
            flags=cv2.INTER_LINEAR,
        )

    def unwarp_image(
        self, bev_image: np.ndarray, output_size: tuple[int, int],
    ) -> np.ndarray:
        """Bird's-eye view -> camera view.

        Args:
            bev_image:   the warped image
            output_size: (width, height) of original camera frame, e.g. (1280, 720)
        """
        return cv2.warpPerspective(
            bev_image, self.M_inv, output_size, flags=cv2.INTER_LINEAR,
        )

    # --- Point warping ---

    def warp_points(self, points: np.ndarray) -> np.ndarray:
        """Apply forward transform to (N, 2) array of (x, y) points.

        Returns: (N, 2) array of warped points.
        """
        pts = np.array(points, dtype=np.float32).reshape(1, -1, 2)
        return cv2.perspectiveTransform(pts, self.M).reshape(-1, 2)

    def unwarp_points(self, points: np.ndarray) -> np.ndarray:
        """Apply inverse transform to (N, 2) array of (x, y) points."""
        pts = np.array(points, dtype=np.float32).reshape(1, -1, 2)
        return cv2.perspectiveTransform(pts, self.M_inv).reshape(-1, 2)


# Self-test when run directly
if __name__ == "__main__":
    warper = PerspectiveWarper()
    print("PerspectiveWarper initialized")
    print(f"  src_pts:\n{warper.src_pts}")
    print(f"  dst_size: {warper.dst_size}")
    print(f"  M:\n{warper.M}")
    print()

    # Round-trip test: warp a point and unwarp it — should return ~original
    test_point = np.array([[640, 600]])  # center-bottom
    warped = warper.warp_points(test_point)
    unwarped = warper.unwarp_points(warped)
    print(f"Round-trip test:")
    print(f"  original: {test_point[0]}")
    print(f"  warped:   {warped[0]}")
    print(f"  unwarped: {unwarped[0]}  (should match original)")
    error = np.linalg.norm(test_point[0] - unwarped[0])
    print(f"  error:    {error:.6f} pixels")
