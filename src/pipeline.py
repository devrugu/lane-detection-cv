"""End-to-end classical lane detection pipeline.

Combines:
- Perspective warping (perspective.py)
- Threshold-based lane pixel extraction (thresholding.py)
- Sliding window pixel grouping (sliding_window.py)
- 2nd-degree polynomial fitting

Provides a single class `ClassicalPipeline` that takes a camera-view image
and returns the detected left/right lane polynomials plus a visualization.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from perspective import PerspectiveWarper
from thresholding import threshold_pipeline
from sliding_window import sliding_window_search
from polyfit import fit_lane_polynomial, evaluate_polynomial

@dataclass
class LaneDetectionResult:
    """Output of one frame's lane detection."""
    left_coeffs: np.ndarray | None     # [a, b, c] for x = ay² + by + c, or None
    right_coeffs: np.ndarray | None    # same for right lane
    left_detected: bool
    right_detected: bool
    bev: np.ndarray                    # bird's-eye view of input image
    mask: np.ndarray                   # binary lane-pixel mask in BEV
    sliding_window_diagnostics: dict   # intermediate sliding-window output

class ClassicalPipeline:
    """End-to-end classical lane detector for one frame.

    Usage:
        pipe = ClassicalPipeline()
        result = pipe.detect(camera_image_rgb)
        overlay = pipe.draw_overlay(camera_image_rgb, result)
    """

    def __init__(self, warper: PerspectiveWarper | None = None):
        self.warper = warper if warper is not None else PerspectiveWarper()

    def detect(self, image_rgb: np.ndarray) -> LaneDetectionResult:
        """Detect lanes in one camera-view RGB image."""
        bev = self.warper.warp_image(image_rgb)
        mask = threshold_pipeline(bev)
        sw = sliding_window_search(mask)
        
        left_coeffs = None
        right_coeffs = None
        if sw["left_detected"]:
            left_coeffs = fit_lane_polynomial(sw["left_x_idx"], sw["left_y_idx"])
        if sw["right_detected"]:
            right_coeffs = fit_lane_polynomial(sw["right_x_idx"], sw["right_y_idx"])
        
        return LaneDetectionResult(
            left_coeffs=left_coeffs,
            right_coeffs=right_coeffs,
            left_detected=sw["left_detected"] and left_coeffs is not None,
            right_detected=sw["right_detected"] and right_coeffs is not None,
            bev=bev,
            mask=mask,
            sliding_window_diagnostics=sw,
        )

    def draw_overlay(self, image_rgb: np.ndarray,
                     result: LaneDetectionResult) -> np.ndarray:
        """Draw lane curves and drivable-area polygon on the original image."""
        H_bev = result.bev.shape[0]
        W_bev = result.bev.shape[1]
        y_eval = np.linspace(0, H_bev - 1, H_bev)
        
        out_size = (image_rgb.shape[1], image_rgb.shape[0])
        result_img = image_rgb.copy()
        
        # Drivable-area polygon (only when both lanes detected)
        if result.left_detected and result.right_detected:
            left_x = evaluate_polynomial(result.left_coeffs, y_eval)
            right_x = evaluate_polynomial(result.right_coeffs, y_eval)
            
            left_pts = np.array([np.transpose(np.vstack([left_x, y_eval]))])
            right_pts = np.array([np.flipud(np.transpose(np.vstack([right_x, y_eval])))])
            polygon_pts = np.hstack((left_pts, right_pts)).astype(np.int32)
            
            bev_polygon = np.zeros((H_bev, W_bev, 3), dtype=np.uint8)
            cv2.fillPoly(bev_polygon, polygon_pts, (0, 200, 0))
            polygon_camera = self.warper.unwarp_image(bev_polygon, out_size)
            result_img = cv2.addWeighted(result_img, 1.0, polygon_camera, 0.3, 0)
        
        # Polynomial curves
        bev_lines = np.zeros((H_bev, W_bev, 3), dtype=np.uint8)
        if result.left_detected:
            left_x = evaluate_polynomial(result.left_coeffs, y_eval)
            pts = np.array(list(zip(left_x.astype(int), y_eval.astype(int))), dtype=np.int32)
            cv2.polylines(bev_lines, [pts], False, (0, 255, 255), thickness=15)
        if result.right_detected:
            right_x = evaluate_polynomial(result.right_coeffs, y_eval)
            pts = np.array(list(zip(right_x.astype(int), y_eval.astype(int))), dtype=np.int32)
            cv2.polylines(bev_lines, [pts], False, (255, 200, 0), thickness=15)
        
        lines_camera = self.warper.unwarp_image(bev_lines, out_size)
        line_mask = lines_camera.sum(axis=2) > 0
        result_img[line_mask] = lines_camera[line_mask]
        
        return result_img


if __name__ == "__main__":
    print("ClassicalPipeline module loaded.")
    pipe = ClassicalPipeline()
    print(f"Pipeline initialized with PerspectiveWarper (dst_size={pipe.warper.dst_size})")