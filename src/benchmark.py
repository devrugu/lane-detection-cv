"""Benchmark FPS for the three lane detection methods.

Measures pipeline-only inference time (NOT image loading/saving) to give
fair comparison numbers. Image is pre-loaded into RAM before timing starts.

Outputs:
  results/fps_benchmark.json
  results/fps_comparison.png   (bar chart for the report)
"""

from __future__ import annotations
import torch.nn.functional as F
import torch.nn as nn
from polyfit import fit_lane_polynomial, evaluate_polynomial
from sliding_window import sliding_window_search
from perspective import PerspectiveWarper
from temporal import EMASmoother
from pipeline import ClassicalPipeline

import json
import sys
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))


# === U-Net architecture (must match training code in Colab) ===

IMG_HEIGHT = 288
IMG_WIDTH = 512


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        c = base_channels
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.enc4 = DoubleConv(c * 4, c * 8)
        self.bottleneck = DoubleConv(c * 8, c * 16)

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)
        self.dec4 = DoubleConv(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)
        self.dec3 = DoubleConv(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.dec2 = DoubleConv(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.dec1 = DoubleConv(c * 2, c)
        self.out_conv = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bottleneck(F.max_pool2d(e4, 2))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)


# === U-Net inference helper ===
def unet_predict_polynomials(model, image_rgb, warper, device):
    """U-Net forward pass + BEV-based lane separation. Returns (left_coeffs, right_coeffs)."""
    orig_h, orig_w = image_rgb.shape[:2]
    img_resized = cv2.resize(
        image_rgb, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
    img_t = (torch.from_numpy(img_resized.astype(np.float32) / 255.0)
             .permute(2, 0, 1).unsqueeze(0).to(device))

    with torch.no_grad():
        logits = model(img_t)
        probs = torch.sigmoid(logits)

    mask_small = (probs[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
    mask_camera = cv2.resize(mask_small, (orig_w, orig_h),
                             interpolation=cv2.INTER_NEAREST)

    mask_3ch = np.dstack([mask_camera, mask_camera, mask_camera])
    mask_bev_3ch = warper.warp_image(mask_3ch)
    mask_bev = (mask_bev_3ch[:, :, 0] > 127).astype(np.uint8) * 255

    sw = sliding_window_search(mask_bev)
    left_coeffs = fit_lane_polynomial(
        sw["left_x_idx"], sw["left_y_idx"]) if sw["left_detected"] else None
    right_coeffs = fit_lane_polynomial(
        sw["right_x_idx"], sw["right_y_idx"]) if sw["right_detected"] else None
    return left_coeffs, right_coeffs


def benchmark_method(name: str, fn: Callable, frames: list[np.ndarray],
                     n_warmup: int = 5) -> dict:
    """Run fn on each frame in frames and time it (pipeline only).

    Args:
        name: display name
        fn: callable that takes one RGB image, returns whatever (we don't care)
        frames: list of pre-loaded RGB images
        n_warmup: warmup iterations (not counted) — first few calls are slower
                  due to caching, JIT, etc.

    Returns dict with avg_fps, std_fps, total_time, per_frame_times.
    """
    print(f"\nBenchmarking: {name}")
    print(f"  Frames: {len(frames)}, warmup: {n_warmup}")

    # Warmup (not timed)
    for i in range(min(n_warmup, len(frames))):
        _ = fn(frames[i])

    # Timed pass
    per_frame_times = []
    for frame in frames:
        t0 = time.perf_counter()
        _ = fn(frame)
        per_frame_times.append(time.perf_counter() - t0)

    per_frame = np.array(per_frame_times)
    total = per_frame.sum()
    avg_fps = len(frames) / total
    median_fps = 1.0 / np.median(per_frame)
    p95_time = np.percentile(per_frame, 95)
    p95_fps = 1.0 / p95_time

    print(f"  Average FPS:    {avg_fps:6.2f}")
    print(f"  Median FPS:     {median_fps:6.2f}")
    print(f"  95-percentile:  {p95_fps:6.2f}  (worst-case)")
    print(
        f"  Per frame:      {per_frame.mean()*1000:6.2f} +/- {per_frame.std()*1000:.2f} ms")
    print(f"  Total time:     {total:.2f}s")

    return {
        "name": name,
        "n_frames": len(frames),
        "avg_fps": float(avg_fps),
        "median_fps": float(median_fps),
        "p95_fps": float(p95_fps),
        "avg_time_ms": float(per_frame.mean() * 1000),
        "std_time_ms": float(per_frame.std() * 1000),
        "total_time_s": float(total),
        "per_frame_times_ms": [float(t * 1000) for t in per_frame],
    }


def main() -> None:
    # === Setup ===
    print("=" * 70)
    print("Lane Detection FPS Benchmark")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Inform user about CPU caveat
    if device.type == "cpu":
        print("\n*** Note: running on CPU. U-Net FPS will be MUCH lower than on GPU. ***")

    # === Load frames into RAM (NOT counted in benchmark) ===
    data_root = Path("data/tusimple/test_subset/clips/0530")
    clip_dirs = sorted([d for d in data_root.iterdir()
                        if d.is_dir() and len(list(d.glob("*.jpg"))) == 20])

    if not clip_dirs:
        print(f"\nERROR: No 20-frame clips found in {data_root}")
        print("Run src/download_data.py first.")
        return

    print(f"\nLoading frames from {len(clip_dirs)} clips...")
    frames = []
    for clip_dir in clip_dirs:
        paths = sorted(clip_dir.glob("*.jpg"), key=lambda p: int(p.stem))
        for p in paths:
            img = cv2.imread(str(p))
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    print(f"  Loaded {len(frames)} frames into RAM")

    # === Load models / pipelines ===
    print("\nInitializing pipelines...")
    classical = ClassicalPipeline()
    warper = PerspectiveWarper()

    # U-Net (load checkpoint)
    ckpt_path = Path("checkpoints/unet_best.pth")
    if not ckpt_path.exists():
        print(f"\nERROR: {ckpt_path} not found.")
        print("Download the trained U-Net checkpoint into checkpoints/")
        return

    print(f"  Loading U-Net checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    unet = UNet().to(device)
    unet.load_state_dict(checkpoint["model_state_dict"])
    unet.eval()
    print(
        f"    Epoch: {checkpoint['epoch']}, val loss: {checkpoint['val_loss']:.4f}")

    # === Define the per-frame callables ===
    # Classical (single-frame)
    def fn_classical(img):
        return classical.detect(img)

    # Classical + EMA — each call uses a fresh smoother per "clip" (20 frames),
    # but for benchmarking we just use a single rolling smoother. The cost is
    # essentially the same as classical + a vector add per frame.
    smoother_state = {"smoother": EMASmoother(alpha=0.25, max_consecutive_misses=5,
                                              pipeline=classical)}

    def fn_classical_ema(img):
        return smoother_state["smoother"].process(img)

    # U-Net
    def fn_unet(img):
        return unet_predict_polynomials(unet, img, warper, device)

    # === Run benchmarks ===
    results = []
    results.append(benchmark_method(
        "Classical CV (single-frame)", fn_classical, frames))
    results.append(benchmark_method(
        "Classical + EMA smoothing", fn_classical_ema, frames))
    results.append(benchmark_method(
        f"U-Net ({device.type.upper()})", fn_unet, frames))

    # === Save results ===
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / "fps_benchmark.json"
    with open(out_path, "w") as f:
        json.dump({
            "device": str(device),
            "n_frames": len(frames),
            "results": results,
        }, f, indent=2)
    print(f"\nSaved benchmark to: {out_path}")

    # === Bar chart for the report ===
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    names = [r["name"].split(" (")[0] for r in results]  # short names
    avg_fps = [r["avg_fps"] for r in results]
    colors = ["#4a90d9", "#2da77e", "#d97a4a"]
    bars = ax.bar(names, avg_fps, color=colors,
                  edgecolor="black", linewidth=0.5)

    # Label bars with their FPS values
    for bar, fps in zip(bars, avg_fps):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(avg_fps) * 0.02,
                f"{fps:.1f} fps",
                ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Average FPS", fontsize=12)
    ax.set_title(f"Lane detection throughput on {device.type.upper()} "
                 f"({len(frames)} frames)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(avg_fps) * 1.2)
    plt.tight_layout()

    chart_path = results_dir / "fps_comparison.png"
    plt.savefig(chart_path, dpi=120, bbox_inches="tight")
    print(f"Saved chart to: {chart_path}")

    # === Print final summary table ===
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Method':<35} {'Avg FPS':>10} {'Per-frame ms':>14}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<35} {r['avg_fps']:>10.2f} {r['avg_time_ms']:>14.2f}")


if __name__ == "__main__":
    main()
