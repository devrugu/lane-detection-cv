"""Re-compute TuSimple metrics for classical CV and U-Net using the new
SOTA-style accuracy alongside our matched-lane precision metric.

We re-run on the 100-frame test subset (NOT the 20 clips × 20 frames in
the prediction cache — those are for the temporal smoothing video).

For each frame:
  - Run classical pipeline → polynomial coefficients → sample at h_samples → metric
  - Run U-Net → BEV mask → sliding window → polynomial → sample at h_samples → metric

Outputs:
  results/tusimple_metrics_classical.json
  results/tusimple_metrics_unet.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from tusimple_loader import TuSimpleDataset
from perspective import PerspectiveWarper
from pipeline import ClassicalPipeline
from polyfit import evaluate_polynomial
from evaluate import match_lanes_one_frame, evaluate_dataset
from benchmark import UNet, unet_predict_polynomials


def predict_lanes_camera_view(pipeline, image_rgb, h_samples):
    """Run classical pipeline and return predicted lane x at each h_sample y."""
    result = pipeline.detect(image_rgb)
    H_bev = result.bev.shape[0]
    y_bev_dense = np.linspace(0, H_bev - 1, num=H_bev * 2)

    pred_lanes = []
    for coeffs, detected in [(result.left_coeffs, result.left_detected),
                              (result.right_coeffs, result.right_detected)]:
        if not detected or coeffs is None:
            continue
        x_bev_dense = evaluate_polynomial(coeffs, y_bev_dense)
        bev_points = np.stack([x_bev_dense, y_bev_dense], axis=1)
        cam_points = pipeline.warper.unwarp_points(bev_points)
        cam_x, cam_y = cam_points[:, 0], cam_points[:, 1]
        order = np.argsort(cam_y)
        cam_y_sorted = cam_y[order]
        cam_x_sorted = cam_x[order]

        pred_x = np.full_like(h_samples, -2.0, dtype=np.float32)
        in_range = (h_samples >= cam_y_sorted.min() - 5) & \
                   (h_samples <= cam_y_sorted.max() + 5)
        if in_range.any():
            pred_x[in_range] = np.interp(
                h_samples[in_range], cam_y_sorted, cam_x_sorted
            )
        pred_lanes.append(pred_x)
    return pred_lanes


def predict_lanes_unet_camera_view(model, image_rgb, h_samples, warper, device):
    """Run U-Net and return predicted lane x at each h_sample y."""
    left_coeffs, right_coeffs = unet_predict_polynomials(
        model, image_rgb, warper, device
    )
    H_bev = warper.dst_size
    y_bev_dense = np.linspace(0, H_bev - 1, num=H_bev * 2)

    pred_lanes = []
    for coeffs in [left_coeffs, right_coeffs]:
        if coeffs is None:
            continue
        x_bev_dense = evaluate_polynomial(coeffs, y_bev_dense)
        bev_points = np.stack([x_bev_dense, y_bev_dense], axis=1)
        cam_points = warper.unwarp_points(bev_points)
        cam_x, cam_y = cam_points[:, 0], cam_points[:, 1]
        order = np.argsort(cam_y)
        cam_y_sorted = cam_y[order]
        cam_x_sorted = cam_x[order]

        pred_x = np.full_like(h_samples, -2.0, dtype=np.float32)
        in_range = (h_samples >= cam_y_sorted.min() - 5) & \
                   (h_samples <= cam_y_sorted.max() + 5)
        if in_range.any():
            pred_x[in_range] = np.interp(
                h_samples[in_range], cam_y_sorted, cam_x_sorted
            )
        pred_lanes.append(pred_x)
    return pred_lanes


def main() -> None:
    project_root = Path(__file__).parent.parent

    dataset = TuSimpleDataset(
        annotation_file=project_root / "data/tusimple/test_subset/test_label_subset.json",
        image_root=project_root / "data/tusimple/test_subset",
    )
    available = dataset.existing_only()[:100]
    print(f"Evaluating on {len(available)} frames")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # === Classical ===
    print("\n=== Classical CV ===")
    classical = ClassicalPipeline()
    cls_results = []
    for sample in tqdm(available):
        img = cv2.cvtColor(cv2.imread(str(sample.image_path)), cv2.COLOR_BGR2RGB)
        h_samples = np.array(sample.h_samples, dtype=np.float32)
        gt_lanes = [np.array(l, dtype=np.float32) for l in sample.lanes]
        pred = predict_lanes_camera_view(classical, img, h_samples)
        cls_results.append(match_lanes_one_frame(pred, gt_lanes))

    cls_metrics = evaluate_dataset(cls_results)
    print(cls_metrics)

    with open(project_root / "results/tusimple_metrics_classical.json", "w") as f:
        json.dump({
            "method": "classical pipeline (single-frame)",
            "accuracy_sota": float(cls_metrics.accuracy_sota),
            "accuracy_matched": float(cls_metrics.accuracy_matched),
            "fpr": float(cls_metrics.fpr),
            "fnr": float(cls_metrics.fnr),
            "n_frames": cls_metrics.n_frames,
            "total_pred_lanes": cls_metrics.total_pred_lanes,
            "total_gt_lanes": cls_metrics.total_gt_lanes,
            "pixel_thresh": 20,
            "pt_thresh": 0.85,
        }, f, indent=2)
    print("Saved: results/tusimple_metrics_classical.json")

    # === U-Net ===
    print("\n=== U-Net ===")
    ckpt = torch.load(project_root / "checkpoints/unet_best.pth",
                       map_location=device, weights_only=False)
    unet = UNet().to(device)
    unet.load_state_dict(ckpt["model_state_dict"])
    unet.eval()
    warper = PerspectiveWarper()

    unet_results = []
    for sample in tqdm(available):
        img = cv2.cvtColor(cv2.imread(str(sample.image_path)), cv2.COLOR_BGR2RGB)
        h_samples = np.array(sample.h_samples, dtype=np.float32)
        gt_lanes = [np.array(l, dtype=np.float32) for l in sample.lanes]
        pred = predict_lanes_unet_camera_view(unet, img, h_samples, warper, device)
        unet_results.append(match_lanes_one_frame(pred, gt_lanes))

    un_metrics = evaluate_dataset(unet_results)
    print(un_metrics)

    with open(project_root / "results/tusimple_metrics_unet.json", "w") as f:
        json.dump({
            "method": "U-Net (single-frame)",
            "accuracy_sota": float(un_metrics.accuracy_sota),
            "accuracy_matched": float(un_metrics.accuracy_matched),
            "fpr": float(un_metrics.fpr),
            "fnr": float(un_metrics.fnr),
            "n_frames": un_metrics.n_frames,
            "total_pred_lanes": un_metrics.total_pred_lanes,
            "total_gt_lanes": un_metrics.total_gt_lanes,
            "checkpoint_epoch": int(ckpt["epoch"]),
            "checkpoint_val_loss": float(ckpt["val_loss"]),
            "pixel_thresh": 20,
            "pt_thresh": 0.85,
        }, f, indent=2)
    print("Saved: results/tusimple_metrics_unet.json")


if __name__ == "__main__":
    main()