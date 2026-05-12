"""TuSimple lane detection benchmark evaluation.

Implements TWO metrics on the same predictions:

  1. Per-frame point accuracy (SOTA convention used by UFLD, SCNN, etc.)
     - Counts hits over ALL GT-visible points across all frames
     - Penalizes both bad detections AND missed lanes
     - Use this for direct comparison against published numbers

  2. Matched-lane precision (our original metric)
     - Counts hits over only the rows where prediction was also valid AND matched
     - Measures geometric precision ON THE LANES WE DETECT
     - Use this to show our pipeline's geometric quality independent of lane coverage

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
    """Aggregate metrics for a set of frames.

    Both accuracy numbers are computed:
      accuracy_sota:    SOTA convention (per-frame averaged, denominator = all GT points)
      accuracy_matched: our matched-lane precision (only on matched pairs)
    """
    accuracy_sota: float        # comparable to UFLD/SCNN published numbers
    accuracy_matched: float     # geometric precision on matched lanes only
    fpr: float
    fnr: float
    n_frames: int
    total_pred_lanes: int
    total_gt_lanes: int

    def __str__(self) -> str:
        return (
            f"TuSimple metrics over {self.n_frames} frames:\n"
            f"  Accuracy (SOTA convention):    {100 * self.accuracy_sota:6.2f}%\n"
            f"  Accuracy (matched-lane only):  {100 * self.accuracy_matched:6.2f}%\n"
            f"  FPR:                            {100 * self.fpr:6.2f}%\n"
            f"  FNR:                            {100 * self.fnr:6.2f}%\n"
            f"  Predicted lanes: {self.total_pred_lanes}\n"
            f"  GT lanes:        {self.total_gt_lanes}"
        )


def match_one_lane(
    pred_xs: np.ndarray,
    gt_xs: np.ndarray,
    pixel_thresh: float = PIXEL_THRESH,
) -> tuple[float, int]:
    """Compare one predicted lane against one ground-truth lane.

    Returns (accuracy_for_this_lane, num_jointly_valid_points).
    Accuracy is computed over rows where BOTH pred and GT are valid (not -2).
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

    Returns dict with:
        n_correct_points:    hits over MATCHED pairs (numerator)
        n_total_points_matched: jointly-valid rows on MATCHED pairs (denominator A)
        n_total_gt_points:   GT-visible rows summed over ALL GT lanes (denominator B)
        n_pred_matched, n_pred_unmatched
        n_gt_matched, n_gt_unmatched

    Denominator A is used for "matched-lane precision".
    Denominator B is used for "SOTA per-frame accuracy".
    """
    n_total_gt_points = sum(int((g != -2).sum()) for g in gt_lanes)

    if not pred_lanes:
        return {
            "n_correct_points": 0,
            "n_total_points_matched": 0,
            "n_total_gt_points": n_total_gt_points,
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
    n_total_points_matched = 0

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
        n_total_points_matched += n_pts

    return {
        "n_correct_points": n_correct_points,
        "n_total_points_matched": n_total_points_matched,
        "n_total_gt_points": n_total_gt_points,
        "n_pred_matched": sum(pred_matched),
        "n_pred_unmatched": sum(1 for m in pred_matched if not m),
        "n_gt_matched": sum(gt_matched),
        "n_gt_unmatched": sum(1 for m in gt_matched if not m),
    }


def evaluate_dataset(frame_results: list[dict]) -> TuSimpleMetrics:
    """Aggregate metrics over many frames.

    accuracy_sota:    sum(correct points) / sum(ALL GT-visible points)
                      averages over frames implicitly by giving every GT point equal weight
    accuracy_matched: sum(correct points) / sum(jointly-valid points on matched pairs)
                      our original metric — precision conditioned on detection
    """
    total_correct = sum(r["n_correct_points"] for r in frame_results)
    total_gt_points = sum(r["n_total_gt_points"] for r in frame_results)
    total_matched_points = sum(r["n_total_points_matched"] for r in frame_results)

    total_pred_unmatched = sum(r["n_pred_unmatched"] for r in frame_results)
    total_gt_unmatched = sum(r["n_gt_unmatched"] for r in frame_results)
    total_pred = sum(r["n_pred_matched"] + r["n_pred_unmatched"] for r in frame_results)
    total_gt = sum(r["n_gt_matched"] + r["n_gt_unmatched"] for r in frame_results)

    accuracy_sota    = total_correct / max(total_gt_points, 1)
    accuracy_matched = total_correct / max(total_matched_points, 1)
    fpr = total_pred_unmatched / max(total_pred, 1)
    fnr = total_gt_unmatched / max(total_gt, 1)

    return TuSimpleMetrics(
        accuracy_sota=accuracy_sota,
        accuracy_matched=accuracy_matched,
        fpr=fpr,
        fnr=fnr,
        n_frames=len(frame_results),
        total_pred_lanes=total_pred,
        total_gt_lanes=total_gt,
    )


if __name__ == "__main__":
    # === Self-tests ===
    h_samples = np.arange(160, 720, 10, dtype=np.float32)

    # Test 1: perfect match
    gt_xs = np.full_like(h_samples, 400.0)
    pred_xs = gt_xs + 5
    acc, n = match_one_lane(pred_xs, gt_xs)
    print(f"Test 1 — perfect match within tolerance:")
    print(f"  accuracy = {100 * acc:.1f}%, n_valid = {n}")
    assert acc == 1.0

    # Test 2: prediction too far
    pred_bad = gt_xs + 30
    acc_bad, _ = match_one_lane(pred_bad, gt_xs)
    print(f"Test 2 — exceeds tolerance:")
    print(f"  accuracy = {100 * acc_bad:.1f}% (should be 0%)")
    assert acc_bad == 0.0

    # Test 3: prediction matches only half the GT rows
    pred_partial = np.full_like(h_samples, -2.0)
    pred_partial[:len(h_samples)//2] = 400.0  # only first half valid
    acc_p, n_p = match_one_lane(pred_partial, gt_xs)
    print(f"Test 3 — prediction covers only half of GT rows:")
    print(f"  accuracy = {100 * acc_p:.1f}% (matched-lane precision is 100% on the half it covers)")
    print(f"  n_jointly_valid = {n_p}")

    # Test 4: full frame with 2 predictions, 4 GT lanes — like our pipeline
    gt1 = np.full_like(h_samples, 200.0)
    gt2 = np.full_like(h_samples, 500.0)
    gt3 = np.full_like(h_samples, 800.0)
    gt4 = np.full_like(h_samples, 1100.0)
    pred1 = gt1 + 5
    pred2 = gt2 + 5
    frame = match_lanes_one_frame([pred1, pred2], [gt1, gt2, gt3, gt4])
    print(f"\nTest 4 — 2 predictions, 4 GT (typical for our pipeline):")
    print(f"  correct points:    {frame['n_correct_points']}")
    print(f"  matched-pair denom:{frame['n_total_points_matched']}")
    print(f"  all-GT denom:      {frame['n_total_gt_points']}")
    print(f"  pred matched/unmatched: {frame['n_pred_matched']}/{frame['n_pred_unmatched']}")
    print(f"  gt matched/unmatched:   {frame['n_gt_matched']}/{frame['n_gt_unmatched']}")

    metrics = evaluate_dataset([frame])
    print(f"\nAggregate from 1 frame:")
    print(metrics)
    print()
    print("Note: SOTA accuracy is HALF of matched-lane accuracy because we only")
    print("detected 2 of 4 GT lanes (with perfect geometric precision).")