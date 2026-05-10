"""TuSimple lane detection benchmark evaluation.

Implements the official TuSimple metric: a predicted lane matches a ground-truth
lane if at least 85% of their x-values (sampled at the standard h_samples y-rows)
are within 20 pixels of each other.

Reference: https://github.com/TuSimple/tusimple-benchmark/blob/master/evaluate/lane.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# === Official TuSimple parameters ===
PIXEL_THRESH = 20      # |x_pred - x_gt| <= 20 px counts as a hit
PT_THRESH = 0.85       # at least 85% of points must be hits to call lanes matched


@dataclass
class TuSimpleMetrics:
    """Aggregate metrics for a set of frames."""
    accuracy: float        # fraction of correctly predicted points
    fpr: float             # false positive rate (extra predicted lanes)
    fnr: float             # false negative rate (missed GT lanes)
    n_frames: int          # number of frames evaluated
    total_pred_lanes: int
    total_gt_lanes: int

    def __str__(self) -> str:
        return (
            f"TuSimple metrics over {self.n_frames} frames:\n"
            f"  Accuracy: {100 * self.accuracy:6.2f}%\n"
            f"  FPR:      {100 * self.fpr:6.2f}%\n"
            f"  FNR:      {100 * self.fnr:6.2f}%\n"
            f"  Predicted lanes: {self.total_pred_lanes}\n"
            f"  GT lanes:        {self.total_gt_lanes}"
        )


def match_one_lane(
    pred_xs: np.ndarray,    # shape (N,) — predicted x at each h_sample y
    gt_xs: np.ndarray,      # shape (N,) — ground-truth x at each h_sample y
    pixel_thresh: float = PIXEL_THRESH,
) -> tuple[float, int]:
    """Compare one predicted lane against one ground-truth lane.

    Returns:
        (accuracy_for_this_lane, num_valid_points)
        where accuracy = fraction of *jointly valid* points within pixel_thresh.
        Jointly valid means both pred_xs != -2 AND gt_xs != -2.
        
        This is the standard interpretation: if our pipeline cannot see a
        portion of the lane (e.g., above our perspective trapezoid horizon),
        it correctly marks those rows as -2 and they should not count
        against accuracy.
    """
    valid_mask = (gt_xs != -2) & (pred_xs != -2)
    if not valid_mask.any():
        return 0.0, 0

    diffs = np.abs(pred_xs[valid_mask] - gt_xs[valid_mask])
    hits = (diffs <= pixel_thresh).sum()
    return hits / valid_mask.sum(), int(valid_mask.sum())


def match_lanes_one_frame(
    pred_lanes: list[np.ndarray],
    gt_lanes: list[np.ndarray],
    pixel_thresh: float = PIXEL_THRESH,
    pt_thresh: float = PT_THRESH,
) -> dict:
    """Greedy 1-to-1 matching of predicted lanes to GT lanes.

    A lane is "matched" if at least pt_thresh fraction of jointly-valid
    points are within pixel_thresh pixels.

    For accuracy aggregation, we count:
      - n_correct_points: hits over all matched (pred, gt) pairs
      - n_total_points: jointly-valid rows summed over MATCHED pairs ONLY
        (this avoids penalizing the predictor for not seeing horizon
        regions covered by GT but outside the BEV trapezoid)

    Returns dict with:
        n_correct_points
        n_total_points
        n_pred_matched, n_pred_unmatched
        n_gt_matched, n_gt_unmatched
    """
    if not pred_lanes:
        return {
            "n_correct_points": 0,
            "n_total_points": 0,
            "n_pred_matched": 0,
            "n_pred_unmatched": 0,
            "n_gt_matched": 0,
            "n_gt_unmatched": len(gt_lanes),
        }

    n_pred = len(pred_lanes)
    n_gt = len(gt_lanes)
    accuracy_matrix = np.zeros((n_pred, n_gt))
    valid_pts_matrix = np.zeros((n_pred, n_gt), dtype=int)

    for i, pred in enumerate(pred_lanes):
        for j, gt in enumerate(gt_lanes):
            acc, n_valid = match_one_lane(pred, gt, pixel_thresh)
            accuracy_matrix[i, j] = acc
            valid_pts_matrix[i, j] = n_valid

    pred_matched = [False] * n_pred
    gt_matched = [False] * n_gt
    n_correct_points = 0
    n_total_points = 0   # only sums over matched pairs

    while True:
        best_acc = -1
        best_i, best_j = -1, -1
        for i in range(n_pred):
            if pred_matched[i]:
                continue
            for j in range(n_gt):
                if gt_matched[j]:
                    continue
                if accuracy_matrix[i, j] > best_acc:
                    best_acc = accuracy_matrix[i, j]
                    best_i, best_j = i, j

        if best_acc < pt_thresh:
            break

        pred_matched[best_i] = True
        gt_matched[best_j] = True
        n_pts = valid_pts_matrix[best_i, best_j]
        n_correct_points += int(round(best_acc * n_pts))
        n_total_points += n_pts

    return {
        "n_correct_points": n_correct_points,
        "n_total_points": n_total_points,
        "n_pred_matched": sum(pred_matched),
        "n_pred_unmatched": sum(1 for m in pred_matched if not m),
        "n_gt_matched": sum(gt_matched),
        "n_gt_unmatched": sum(1 for m in gt_matched if not m),
    }


def evaluate_dataset(
    frame_results: list[dict],
) -> TuSimpleMetrics:
    """Aggregate metrics over many frames.

    Args:
        frame_results: list of dicts, each from match_lanes_one_frame()
    """
    total_correct = sum(r["n_correct_points"] for r in frame_results)
    total_points = sum(r["n_total_points"] for r in frame_results)
    total_pred_unmatched = sum(r["n_pred_unmatched"] for r in frame_results)
    total_gt_unmatched = sum(r["n_gt_unmatched"] for r in frame_results)
    total_pred = sum(r["n_pred_matched"] + r["n_pred_unmatched"] for r in frame_results)
    total_gt = sum(r["n_gt_matched"] + r["n_gt_unmatched"] for r in frame_results)
    
    accuracy = total_correct / max(total_points, 1)
    fpr = total_pred_unmatched / max(total_pred, 1)
    fnr = total_gt_unmatched / max(total_gt, 1)
    
    return TuSimpleMetrics(
        accuracy=accuracy,
        fpr=fpr,
        fnr=fnr,
        n_frames=len(frame_results),
        total_pred_lanes=total_pred,
        total_gt_lanes=total_gt,
    )


if __name__ == "__main__":
    # Self-test: build two synthetic lanes that should match
    h_samples = np.arange(160, 720, 10, dtype=np.float32)  # 56 values
    
    # GT lane: vertical line at x=400
    gt_xs = np.full_like(h_samples, 400.0)
    
    # Predicted lane: same line + 5px offset everywhere — should match (within 20px tolerance)
    pred_xs = gt_xs + 5
    
    acc, n = match_one_lane(pred_xs, gt_xs)
    print(f"Test 1 — perfect match within tolerance:")
    print(f"  accuracy = {100 * acc:.1f}%, n_valid = {n}")
    assert acc == 1.0
    
    # Predicted lane: same line + 30px offset everywhere — should NOT match
    pred_xs_bad = gt_xs + 30
    acc_bad, _ = match_one_lane(pred_xs_bad, gt_xs)
    print(f"Test 2 — exceeds tolerance:")
    print(f"  accuracy = {100 * acc_bad:.1f}% (should be 0%)")
    assert acc_bad == 0.0
    
    # Frame-level test
    frame_result = match_lanes_one_frame([pred_xs, pred_xs_bad], [gt_xs])
    print(f"\nTest 3 — one good prediction, one bad, one GT lane:")
    print(f"  {frame_result}")
    
    # Aggregate test
    metrics = evaluate_dataset([frame_result])
    print(f"\nAggregate metrics:")
    print(metrics)
    
    print("\nModule evaluate.py loaded successfully.")